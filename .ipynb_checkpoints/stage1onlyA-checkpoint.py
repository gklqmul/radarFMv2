import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


# 假设 dataset.py 在同级目录下，请确保它里面的 _process_single_sample 已经是修复后的版本（输出6通道）
from dataset import DataReader, RadarDiffusionDataset
from flowmodels import collate_fn_for_cross_modal

import torch.distributed as dist


EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 26), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (8, 10),
    (3, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (15, 17),
    (0, 18), (18, 19), (19, 20), (20, 21), (0, 22), (22, 23), (23, 24), (24, 25)
]

PARENT = {
    1: 0, 2: 1, 3: 2, 26: 3,
    4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 8,
    11: 3, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 15,
    18: 0, 19: 18, 20: 19, 21: 20,
    22: 0, 23: 22, 24: 23, 25: 24
}


class TimeAwareCompressedRadarEncoder(nn.Module):
    """
    Input:  pc [B, 128, 6]  (xyz, doppler, snr, time(?) ...)
    Output: z_radar [B, 64, 256]
    """
    def __init__(self, in_channels=6, embed_dim=256, num_latents=64):
        super().__init__()
        assert in_channels == 6, "Expect pc dim=6"

        # 这里按：前5维为物理特征（xyz,doppler,snr），最后1维为 time（或其他）
        self.phys_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, embed_dim)
        )

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 2,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.latents = nn.Parameter(torch.randn(1, num_latents, embed_dim))
        self.compress_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, pc):
        """
        pc: [B,128,6]
        """
        pc = pc.clone()

        # 如果第4维确实是 snr，保留你的归一化习惯；否则把这一行删掉
        # 这里假设 pc[...,4] 是 snr
        pc[..., 4] = pc[..., 4] / 1000.0

        phys_feat = pc[..., :5]      # [B,128,5]
        time_feat = pc[..., 5:]     # [B,128,1]

        h = self.phys_mlp(phys_feat) + self.time_mlp(time_feat)
        h = self.fusion(h)

        feat_128 = self.transformer(h)  # [B,128,embed]

        B = pc.shape[0]
        latents = self.latents.repeat(B, 1, 1)  # [B,64,embed]
        z_compressed, _ = self.compress_attn(query=latents, key=feat_128, value=feat_128)
        z_radar = self.norm_out(z_compressed)   # [B,64,embed]

        return z_radar


class TemporalAdapter(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, z_flat, B, T):
        """
        z_flat: [B*T, 64, embed]
        valid_mask: [B,T] bool, True=真实帧
        return: [B*T, 64, embed]
        """
        z = z_flat.view(B, T, 64, -1)  # [B,T,64,embed]
        z = z.permute(0, 2, 1, 3).reshape(B * 64, T, -1)  # [B*64,T,embed]

        z = self.transformer(z)     # [B*64,T,embed]
        z = z.view(B, 64, T, -1).permute(0, 2, 1, 3).contiguous()     # [B,T,64,embed]
        return z.reshape(B * T, 64, -1)


class CoarseSkeletonHead(nn.Module):
    """
    FK coarse head: returns joints [B,J,3]
    """
    def __init__(self, latent_dim, num_joints, parent):
        super().__init__()
        self.num_joints = num_joints
        self.parent = parent

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.SiLU(),
            nn.Linear(512, (num_joints - 1) * 4)
        )

    def forward(self, z):
        B = z.shape[0]
        J = self.num_joints
        raw = self.mlp(z).view(B, J - 1, 4)

        dir_raw = raw[..., :3]
        len_raw = raw[..., 3]

        direction = F.normalize(dir_raw, dim=-1, eps=1e-6)
        length = F.softplus(len_raw)
        offset_non_root = direction * length.unsqueeze(-1)

        offsets = torch.zeros(B, J, 3, device=z.device)
        offsets[:, 1:] = offset_non_root

        joints = torch.zeros_like(offsets)
        joints[:, 0] = 0.0

        for j in range(1, J):
            p = self.parent[j]
            joints[:, j] = joints[:, p] + offsets[:, j]

        return joints, offsets, length


class DirectJointHead(nn.Module):
    def __init__(self, latent_dim=256, num_joints=27):
        super().__init__()
        self.num_joints = num_joints
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, num_joints * 3)
        )

    def forward(self, z_global):
        B = z_global.shape[0]
        return self.net(z_global).view(B, self.num_joints, 3)


class RadarStage1Model(nn.Module):
    def __init__(
        self,
        in_channels=6,
        radar_embed_dim=256,
        num_latents=64,
        num_joints=27,
        parent_list=None,
        use_temporal=False,   # 模块 B 开关
        use_fk=False,         # 模块 C 开关 (True 则用 CoarseSkeletonHead, False 用 Direct)
    ):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_fk = use_fk
        self.num_joints = num_joints

        # 模块 A: Encoder
        self.encoder = TimeAwareCompressedRadarEncoder(
            in_channels=in_channels,
            embed_dim=radar_embed_dim,
            num_latents=num_latents
        )
        
        # 模块 B: Temporal Adapter
        if self.use_temporal:
            self.temporal_adapter = TemporalAdapter(embed_dim=radar_embed_dim)

        # 模块 C: FK Head
        if self.use_fk:
            self.coarse_head = CoarseSkeletonHead(
                latent_dim=radar_embed_dim,
                num_joints=num_joints,
                parent=parent_list
            )
        else:
            # Direct Head (作为 Baseline 或 消融对比)
            self.direct_head = DirectJointHead(
                latent_dim=radar_embed_dim,
                num_joints=num_joints
            )

    def forward(self, radar_seq):
        B, T, N, C = radar_seq.shape
        radar_flat = radar_seq.view(B * T, N, C)

        # 1. 运行模块 A
        z = self.encoder(radar_flat) # [BT, 64, embed]

        # 2. 运行模块 B (可选)
        if self.use_temporal:
            # 这里的 z 进出都是 [BT, 64, embed]，但在内部做了跨 T 的特征交换
            z = self.temporal_adapter(z, B, T)

        # 3. 聚合特征
        z_global = z.mean(dim=1)  # [BT, embed]

        # 4. 运行预测头 (模块 C 或 Direct)
        if self.use_fk:
            x0, _, _ = self.coarse_head(z_global)
        else:
            x0 = self.direct_head(z_global)

        return x0.view(B, T, self.num_joints, 3)
# ==========================================
# 3. 损失函数与评估函数
# ==========================================

def stage1_loss(pred, gt):
    """
    pred, gt: [B, T, J, 3] (meters, root-relative)
    valid_mask: [B, T] bool
    """
    valid_mask = (gt.abs().sum(dim=(2, 3)) > 1e-6)
    # [B,T,J]
    err = torch.norm(pred - gt, dim=-1)
    # [B,T]
    err = err.mean(dim=-1)
    err = err[valid_mask]
    if err.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    return err.mean()
    # return F.mse_loss(pred, gt)


@torch.no_grad()
def evaluate_sequence(
    dataloader, model, device,
    vis_dir=None, vis_edges=None, num_vis_samples=2,
    pck_thresholds=(50.0, 100.0),
    auc_max_threshold=80.0,
    auc_step=5.0,
):
    """
    Evaluation for sequence prediction (Flow Matching inference via steps).

    Conventions:
    - Inputs/GT are in meters; we convert to millimeters for reporting.
    - All pose metrics are computed on root-relative joints (joint 0 as root).
    - Valid frames are those where GT is non-zero (mask from skeleton_seq).

    Metrics:
    Frame-weighted (by #valid frames M):
        - MPJPE, PA-MPJPE, PCK@th, AUC(PCK), SSC, bone_mae
    Diff-weighted (by #valid diffs):
        - MPJVE (velocity error vs GT)
        - MJAE (acceleration error vs GT)  [if you have compute_mjae, use it]
        - mpjv_pred/mpja_pred/mpjj_pred (pred motion magnitude)
        - mpjve_dyn/mpjae_dyn/mpjje_dyn (dynamic errors; same as above but explicit)
    Sequence-stat (by batch count):
        - bone_var_pred/bone_var_gt (bone length temporal variance; mean over bones)
          (You can switch to diff-weighted if needed; here it's per-batch stable.)
    """
    model.eval()

    # -----------------------
    # Accumulators
    # -----------------------
    # frame-weighted accumulators (sum over valid frames)
    sum_mpjpe = 0.0
    sum_pampjpe = 0.0
    sum_auc = 0.0
    sum_pck = {th: 0.0 for th in pck_thresholds}
    sum_ssc = 0.0
    sum_bone_mae = 0.0
    n_frames = 0  # total valid frames M across dataset

    # diff-weighted accumulators (sum over valid diffs)
    sum_mpjve = 0.0
    n_vel = 0

    sum_mjae = 0.0
    n_acc = 0

    # prediction-only motion magnitude
    sum_mpjv_pred, n_mpjv = 0.0, 0
    sum_mpja_pred, n_mpja = 0.0, 0
    sum_mpjj_pred, n_mpjj = 0.0, 0

    # dynamic errors vs GT (explicit)
    sum_mpjve_dyn, n_mpjve_dyn = 0.0, 0
    sum_mpjae_dyn, n_mpjae_dyn = 0.0, 0
    sum_mpjje_dyn, n_mpjje_dyn = 0.0, 0

    # bone variance (per-batch mean over bones)
    sum_bone_var_pred = 0.0
    sum_bone_var_gt = 0.0
    n_bone_var = 0

    # -----------------------
    # Visualization setup
    # -----------------------
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
    has_visualized = 0
    pbar = tqdm(dataloader, desc=f"Eval (Steps=1)")
    # edges for bone metrics / visualization
    edge_i = edge_j = None
    if vis_edges:
        edge_i = torch.tensor([e[0] for e in vis_edges], device=device, dtype=torch.long)
        edge_j = torch.tensor([e[1] for e in vis_edges], device=device, dtype=torch.long)

    for batch_idx, (radar_seq, skeleton_seq) in enumerate(pbar):
        radar_seq = radar_seq.to(device).float()       # [B,T,...]
        skeleton_seq = skeleton_seq.to(device).float() # [B,T,J,3] in meters
        B, T = skeleton_seq.shape[0], skeleton_seq.shape[1]

        # -----------------------
        # 1) Valid mask from GT
        # -----------------------
        valid_mask = (skeleton_seq.abs().sum(dim=(2, 3)) > 1e-6)  # [B,T] bool
        if not valid_mask.any():
            continue

        pred = model(radar_seq)
        pred_mm = pred * 1000.0
        gt_mm = skeleton_seq * 1000.0
        pred_rel = pred_mm - pred_mm[:, :, 0:1, :]
        gt_rel = gt_mm - gt_mm[:, :, 0:1, :]

        # -----------------------
        # 4) Frame-level metrics on valid frames only
        # -----------------------
        v_pred = pred_rel[valid_mask]  # [M,J,3]
        v_gt = gt_rel[valid_mask]      # [M,J,3]
        M = v_gt.shape[0]
        if M == 0:
            continue

        # MPJPE / PA-MPJPE / PCK / AUC
        mpjpe_val = compute_mpjpe(v_pred, v_gt).item()
        pampjpe_val = compute_pampjpe(v_pred, v_gt).item()
        auc_val = compute_auc_pck(v_pred, v_gt, max_threshold=auc_max_threshold, step=auc_step).item()
        pck_vals = {th: compute_pck(v_pred, v_gt, th).item() for th in pck_thresholds}

        sum_mpjpe += mpjpe_val * M
        sum_pampjpe += pampjpe_val * M
        sum_auc += auc_val * M
        for th in pck_thresholds:
            sum_pck[th] += pck_vals[th] * M
        n_frames += M

        # SSC
        ssc_val = compute_spatial_structure_corr(v_pred, v_gt).item()
        sum_ssc += ssc_val * M

        # bone_mae (mean abs bone length error) on valid frames
        if edge_i is not None:
            d_pred = torch.norm(v_pred[:, edge_i] - v_pred[:, edge_j], dim=-1)  # [M,E]
            d_gt = torch.norm(v_gt[:, edge_i] - v_gt[:, edge_j], dim=-1)        # [M,E]
            bone_mae_val = (d_pred - d_gt).abs().mean().item()
            sum_bone_mae += bone_mae_val * M

        # -----------------------
        # 5) Diff-based metrics (only on consecutive valid frames)
        # -----------------------
        # velocity/accel/jerk masks
        m_v = valid_mask[:, 1:] & valid_mask[:, :-1]   # [B,T-1]
        m_a = m_v[:, 1:] & m_v[:, :-1]                 # [B,T-2]
        m_j = m_a[:, 1:] & m_a[:, :-1]                 # [B,T-3]

        # velocity error vs GT
        if m_v.any():
            v_err = _diff1(pred_rel) - _diff1(gt_rel)  # [B,T-1,J,3]
            s, c = _masked_dyn_sum(v_err, m_v)
            sum_mpjve += s.item()
            n_vel += int(c.item())

            # also record as "dyn" explicitly
            sum_mpjve_dyn += s.item()
            n_mpjve_dyn += int(c.item())

            # pred-only velocity magnitude
            v_mag = _diff1(pred_rel)
            s, c = _masked_dyn_sum(v_mag, m_v)
            sum_mpjv_pred += s.item()
            n_mpjv += int(c.item())

        # acceleration error vs GT (MJAE)
        if m_a.any():
            a_err = _diff2(pred_rel) - _diff2(gt_rel)  # [B,T-2,J,3]
            s, c = _masked_dyn_sum(a_err, m_a)
            sum_mjae += s.item()
            n_acc += int(c.item())

            sum_mpjae_dyn += s.item()
            n_mpjae_dyn += int(c.item())

            a_mag = _diff2(pred_rel)
            s, c = _masked_dyn_sum(a_mag, m_a)
            sum_mpja_pred += s.item()
            n_mpja += int(c.item())

        # jerk error vs GT (optional but consistent)
        if m_j.any():
            j_err = _diff3(pred_rel) - _diff3(gt_rel)  # [B,T-3,J,3]
            s, c = _masked_dyn_sum(j_err, m_j)
            sum_mpjje_dyn += s.item()
            n_mpjje_dyn += int(c.item())

            j_mag = _diff3(pred_rel)
            s, c = _masked_dyn_sum(j_mag, m_j)
            sum_mpjj_pred += s.item()
            n_mpjj += int(c.item())

        # -----------------------
        # 6) Bone length temporal variance (per-batch)
        # -----------------------
        # Definition: for each bone e, var over time of bone length d[t,e] on valid frames,
        # then mean over bones. This measures temporal stability of skeleton structure.
        if edge_i is not None and M > 1:
            d_pred = torch.norm(v_pred[:, edge_i] - v_pred[:, edge_j], dim=-1)  # [M,E]
            d_gt = torch.norm(v_gt[:, edge_i] - v_gt[:, edge_j], dim=-1)        # [M,E]
            bone_var_pred = d_pred.var(dim=0, unbiased=False).mean().item()
            bone_var_gt = d_gt.var(dim=0, unbiased=False).mean().item()
            sum_bone_var_pred += bone_var_pred
            sum_bone_var_gt += bone_var_gt
            n_bone_var += 1

        if vis_dir and has_visualized < num_vis_samples and vis_edges:
                # 找到当前 Batch 中有有效数据的样本索引
                valid_bs = [i for i in range(B) if valid_mask[i].any().item()]

                if valid_bs:
                    # 1) 随机锁定一个样本 b
                    b = random.choice(valid_bs)
                    # 获取该样本下所有有效的帧索引 (全长)
                    valid_indices = torch.where(valid_mask[b])[0]  # (T_valid,)

                    if valid_indices.numel() > 0:
                        # --- A) 生成全长 GIF ---
                        gt_seq_full = gt_rel[b, valid_indices].detach().cpu().numpy()
                        pred_seq_full = pred_rel[b, valid_indices].detach().cpu().numpy()
                        num_total_frames = gt_seq_full.shape[0]

                        fig = plt.figure(figsize=(10, 5))
                        ax_gt = fig.add_subplot(121, projection="3d")
                        ax_pr = fig.add_subplot(122, projection="3d")

                        def update(idx):
                            ax_gt.cla()
                            ax_pr.cla()
                            actual_frame_idx = int(valid_indices[idx].item())
                            draw_skeleton_3d(ax_gt, gt_seq_full[idx], vis_edges, "green", f"GT Frame {actual_frame_idx}")
                            draw_skeleton_3d(ax_pr, pred_seq_full[idx], vis_edges, "red", f"Pred Frame {actual_frame_idx}")
                            return []

                        ani = animation.FuncAnimation(fig, update, frames=num_total_frames, interval=100)
                        gif_name = f"Sample{has_visualized}_B{batch_idx}_S{b}_full_seq.gif"
                        ani.save(os.path.join(vis_dir, gif_name), writer="pillow", fps=10)
                        plt.close(fig)

                        # --- B) 生成相同样本的 10 帧 HTML ---
                        num_html = 10
                        if num_total_frames > num_html:
                            start_f = num_total_frames // 4
                            html_indices = valid_indices[start_f:start_f + num_html]
                        else:
                            html_indices = valid_indices

                        for t_idx in html_indices:
                            t = int(t_idx.item())
                            html_name = f"Sample{has_visualized}_B{batch_idx}_f{t:03d}.html"
                            plot_skeleton(
                                gt_joints=gt_rel[b, t].detach().cpu().numpy(),
                                pred_joints=pred_rel[b, t].detach().cpu().numpy(),
                                edges=vis_edges,
                                frame_id=f"Sample{has_visualized}-Frame{t}",
                                out_html=os.path.join(vis_dir, html_name)
                            )

                        has_visualized += 1

        pbar.set_postfix({"mpjpe": f"{sum_mpjpe/max(n_frames, 1):.2f}mm"})

    # -----------------------
    # 8) Final aggregation
    # -----------------------
    denom = max(n_frames, 1)
    final = {
        "mpjpe": sum_mpjpe / denom,
        "pa_mpjpe": sum_pampjpe / denom,
        "auc_pck": sum_auc / denom,
        "ssc": sum_ssc / denom,
    }
    for th in pck_thresholds:
        final[f"pck@{int(th)}"] = sum_pck[th] / denom

    if n_vel > 0:
        final["mpjve"] = sum_mpjve / n_vel  # mm/frame

    if n_acc > 0:
        final["mjae"] = sum_mjae / n_acc    # mm/frame^2

    if edge_i is not None:
        final["bone_mae"] = sum_bone_mae / denom
        if n_bone_var > 0:
            final["bone_var_pred"] = sum_bone_var_pred / n_bone_var
            final["bone_var_gt"] = sum_bone_var_gt / n_bone_var

    # prediction-only motion magnitude (weighted by valid diffs)
    if n_mpjv > 0: final["mpjv_pred"] = sum_mpjv_pred / n_mpjv
    if n_mpja > 0: final["mpja_pred"] = sum_mpja_pred / n_mpja
    if n_mpjj > 0: final["mpjj_pred"] = sum_mpjj_pred / n_mpjj

    # dynamic errors vs GT (explicit)
    if n_mpjve_dyn > 0: final["mpjve_dyn"] = sum_mpjve_dyn / n_mpjve_dyn
    if n_mpjae_dyn > 0: final["mpjae_dyn"] = sum_mpjae_dyn / n_mpjae_dyn
    if n_mpjje_dyn > 0: final["mpjje_dyn"] = sum_mpjje_dyn / n_mpjje_dyn

    return final

def calculate_mpjpe(pred, gt):
    # pred, gt: [B, 27, 3]
    return torch.norm(pred - gt, dim=-1).mean().item()


def _masked_dyn_sum(x, m):
    """
    x: [B, T', J, 3]  (or [B,T',J]) representing diff vectors or errors
    m: [B, T'] bool mask for valid diffs
    Return: (sum_scalar, count) where scalar is sum of per-(b,t) mean-joint L2.
    """
    # L2 over xyz -> [B,T',J]
    if x.dim() == 4:
        x = torch.norm(x, dim=-1)
    # mean over joints -> [B,T']
    if x.dim() == 3:
        x = x.mean(dim=-1)

    m_f = m.float()
    sum_val = (x * m_f).sum()
    cnt = m_f.sum()
    return sum_val, cnt

def _diff1(x):  # [B,T,J,3] -> [B,T-1,J,3]
    return x[:, 1:] - x[:, :-1]

def _diff2(x):  # [B,T,J,3] -> [B,T-2,J,3]
    return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]

def _diff3(x):  # [B,T,J,3] -> [B,T-3,J,3]
    return x[:, 3:] - 3 * x[:, 2:-1] + 3 * x[:, 1:-2] - x[:, :-3]
    
def masked_mean(x, mask, eps=1e-8):
    """
    x: [B,T,J,3] or [B,T,J] etc.
    mask: [B,T] bool
    returns scalar
    """
    # 先把 x 压成 [B,T] 标量（按你的定义）
    # 这里假设 x 是向量，先L2 -> [B,T,J]，再mean joint -> [B,T]
    if x.dim() == 4:            # [B,T,J,3]
        x = torch.norm(x, dim=-1)        # [B,T,J]
    if x.dim() == 3:            # [B,T,J]
        x = x.mean(dim=-1)               # [B,T]

    x = x * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return x.sum() / denom 
def compute_spatial_structure_corr(pred_rel, gt_rel):
    """
    pred_rel, gt_rel: (M, J, 3)
    return: scalar correlation averaged over frames
    """
    J = pred_rel.shape[1]

    def pairwise_dist(x):
        diff = x.unsqueeze(2) - x.unsqueeze(1)  # (M, J, J, 3)
        return torch.norm(diff, dim=-1)         # (M, J, J)

    D_pred = pairwise_dist(pred_rel)
    D_gt = pairwise_dist(gt_rel)

    # 去掉对角线
    mask = ~torch.eye(J, dtype=torch.bool, device=pred_rel.device)
    D_pred = D_pred[:, mask]
    D_gt = D_gt[:, mask]

    # 计算 Pearson correlation
    pred_mean = D_pred.mean(dim=1, keepdim=True)
    gt_mean = D_gt.mean(dim=1, keepdim=True)

    pred_centered = D_pred - pred_mean
    gt_centered = D_gt - gt_mean

    numerator = (pred_centered * gt_centered).sum(dim=1)
    denominator = torch.sqrt(
        (pred_centered ** 2).sum(dim=1) *
        (gt_centered ** 2).sum(dim=1)
    ) + 1e-8

    corr = numerator / denominator  # (M,)

    return corr.mean()


def compute_jitter_metrics(pred_rel, valid_mask):
    # pred_rel: [B, T, J, 3] (mm)
    # valid_mask: [B, T] bool
    
    # velocity/accel/jerk existence masks
    m_v = valid_mask[:, 1:] & valid_mask[:, :-1]           # [B, T-1]
    m_a = m_v[:, 1:] & m_v[:, :-1]                         # [B, T-2]
    m_j = m_a[:, 1:] & m_a[:, :-1]                         # [B, T-3]

    v = diff1(pred_rel)  # [B, T-1, J, 3]
    a = diff2(pred_rel)  # [B, T-2, J, 3]
    j = diff3(pred_rel)  # [B, T-3, J, 3]

    out = {}
    if m_v.any():
        out["mpjv"] = masked_mean(v, m_v).item()           # 预测自身速度幅值
    if m_a.any():
        out["mpja"] = masked_mean(a, m_a).item()           # 预测自身加速度幅值（抖动）
    if m_j.any():
        out["mpjj"] = masked_mean(j, m_j).item()           # 预测自身jerk幅值（抖动）
    return out

def compute_dynamic_errors(pred_rel, gt_rel, valid_mask):
    # 与GT的速度/加速度/jerk误差
    m_v = valid_mask[:, 1:] & valid_mask[:, :-1]
    m_a = m_v[:, 1:] & m_v[:, :-1]
    m_j = m_a[:, 1:] & m_a[:, :-1]

    ev = diff1(pred_rel) - diff1(gt_rel)
    ea = diff2(pred_rel) - diff2(gt_rel)
    ej = diff3(pred_rel) - diff3(gt_rel)

    out = {}
    if m_v.any():
        out["mpjve"] = masked_mean(ev, m_v).item()
    if m_a.any():
        out["mpjae"] = masked_mean(ea, m_a).item()
    if m_j.any():
        out["mpjje"] = masked_mean(ej, m_j).item()
    return out
    
def compute_mpjpe(pred, gt):
    """
    pred, gt: [M, J, 3]  (mm)
    return: float (mm)
    """
    return torch.norm(pred - gt, dim=-1).mean()
    
def batch_procrustes_align(pred, gt, eps=1e-8):
    """
    Procrustes alignment (scale+rotation+translation) for each sample.
    pred, gt: [M, J, 3]
    return: pred_aligned [M, J, 3]
    """
    # subtract mean
    muX = pred.mean(dim=1, keepdim=True)  # [M,1,3]
    muY = gt.mean(dim=1, keepdim=True)
    X0 = pred - muX
    Y0 = gt - muY

    # compute covariance
    # [M,3,3] = [M,3,J] @ [M,J,3]
    H = torch.matmul(X0.transpose(1, 2), Y0)

    # SVD
    U, S, Vt = torch.linalg.svd(H)  # U:[M,3,3], Vt:[M,3,3]
    V = Vt.transpose(1, 2)

    # handle reflection
    detR = torch.det(torch.matmul(V, U.transpose(1, 2)))  # [M]
    sign = torch.ones_like(detR)
    sign[detR < 0] = -1.0

    # build correction matrix
    Z = torch.eye(3, device=pred.device, dtype=pred.dtype).unsqueeze(0).repeat(pred.shape[0], 1, 1)
    Z[:, 2, 2] = sign

    R = torch.matmul(torch.matmul(V, Z), U.transpose(1, 2))  # [M,3,3]

    # scale
    varX = (X0 ** 2).sum(dim=(1, 2)) + eps  # [M]
    scale = (S * Z.diagonal(dim1=-2, dim2=-1)).sum(dim=1) / varX  # [M]

    # aligned
    X_aligned = scale.view(-1, 1, 1) * torch.matmul(X0, R) + muY
    return X_aligned

def compute_pampjpe(pred, gt):
    """
    PA-MPJPE (Procrustes aligned MPJPE)
    pred, gt: [M, J, 3]
    return: float (mm)
    """
    pred_aligned = batch_procrustes_align(pred, gt)
    return compute_mpjpe(pred_aligned, gt)

def compute_pck(pred, gt, threshold_mm=50.0):
    """
    PCK@threshold: percentage of joints with error < threshold
    pred, gt: [M, J, 3]
    return: float in [0,1]
    """
    err = torch.norm(pred - gt, dim=-1)  # [M,J]
    return (err < threshold_mm).float().mean()

def compute_auc_pck(pred, gt, max_threshold=150.0, step=5.0):
    """
    AUC for PCK curve from 0..max_threshold (mm) with given step.
    pred, gt: [M, J, 3]
    return: float in [0,1]
    """
    thresholds = torch.arange(0.0, max_threshold + 1e-6, step, device=pred.device, dtype=pred.dtype)
    err = torch.norm(pred - gt, dim=-1)  # [M,J]
    pcks = [(err < th).float().mean() for th in thresholds]
    pcks = torch.stack(pcks)  # [K]
    # normalized area (divide by max_threshold)
    auc = torch.trapz(pcks, thresholds) / max_threshold
    return auc

def compute_mpjve(pred_seq, gt_seq):
    """
    MPJVE: Mean Per-Joint Velocity Error (mm/frame)
    pred_seq, gt_seq: [B, T, J, 3] (mm), already root-relative & masked outside (we will mask later)
    return: per-frame velocity error tensor [B, T-1] or scalar depending usage.
    """
    v_pred = pred_seq[:, 1:] - pred_seq[:, :-1]  # [B, T-1, J, 3]
    v_gt = gt_seq[:, 1:] - gt_seq[:, :-1]
    ve = torch.norm(v_pred - v_gt, dim=-1).mean(dim=-1)  # [B, T-1] (mean over joints)
    return ve
    
def compute_bone_length_mae(pred_rel, gt_rel, edges):
    """
    pred_rel, gt_rel: (M, J, 3)
    edges: list[(i,j)]
    return: scalar MAE over all bones
    """
    device = pred_rel.device
    idx_i = torch.tensor([e[0] for e in edges], device=device)
    idx_j = torch.tensor([e[1] for e in edges], device=device)

    d_pred = torch.norm(pred_rel[:, idx_i] - pred_rel[:, idx_j], dim=-1)  # (M,E)
    d_gt   = torch.norm(gt_rel[:, idx_i]   - gt_rel[:, idx_j], dim=-1)
    return (d_pred - d_gt).abs().mean()

def compute_bone_length_var(x_rel, edges):
    device = x_rel.device
    idx_i = torch.tensor([e[0] for e in edges], device=device)
    idx_j = torch.tensor([e[1] for e in edges], device=device)
    d = torch.norm(x_rel[:, idx_i] - x_rel[:, idx_j], dim=-1)  # (M,E)
    return d.var(dim=0).mean()  # across frames, then average bones

def compute_mjae(seq_pred, seq_gt):
    """
    seq_pred, seq_gt: (B, T, J, 3) root-relative mm
    return: (B, T-2) per-frame acceleration error (mean over joints)
    """
    a_pred = seq_pred[:, 2:] - 2*seq_pred[:, 1:-1] + seq_pred[:, :-2]
    a_gt   = seq_gt[:, 2:]   - 2*seq_gt[:, 1:-1]   + seq_gt[:, :-2]
    err = torch.norm(a_pred - a_gt, dim=-1).mean(dim=-1)  # (B, T-2)
    return err

def compute_jerk_energy(seq):
    """
    seq: (B, T, J, 3) root-relative mm
    return: scalar jerk energy (lower = smoother)
    """
    # third difference
    j = seq[:, 3:] - 3*seq[:, 2:-1] + 3*seq[:, 1:-2] - seq[:, :-3]
    return (j**2).sum(dim=-1).mean()

def best_of_k_mpjpe(samples, gt_rel):
    """
    samples: (K, M, J, 3)  K samples for each frame
    gt_rel:  (M, J, 3)
    return: scalar best-of-K MPJPE
    """
    # mpjpe per sample
    errs = torch.norm(samples - gt_rel.unsqueeze(0), dim=-1).mean(dim=-1)  # (K,M)
    best = errs.min(dim=0).values  # (M,)
    return best.mean()

def diff1(x):  # velocity: x[t]-x[t-1]
    return x[:, 1:] - x[:, :-1]

def diff2(x):  # accel
    v = diff1(x)
    return v[:, 1:] - v[:, :-1]

def diff3(x):  # jerk
    a = diff2(x)
    return a[:, 1:] - a[:, :-1]

def masked_mean(val, mask):
    # val: [B, T', J] or [B, T', J, 3] -> reduce last dim outside
    # mask: [B, T'] boolean
    if val.ndim == 4:
        val = torch.norm(val, dim=-1)  # -> [B, T', J]
    m = mask.unsqueeze(-1).float()     # [B, T', 1]
    denom = m.sum() * val.shape[-1] + 1e-8
    return (val * m).sum() / denom
    
def compute_spatial_structure_corr(pred_rel, gt_rel):
    """
    pred_rel, gt_rel: (M, J, 3)
    return: scalar correlation averaged over frames
    """
    J = pred_rel.shape[1]

    def pairwise_dist(x):
        diff = x.unsqueeze(2) - x.unsqueeze(1)  # (M, J, J, 3)
        return torch.norm(diff, dim=-1)         # (M, J, J)

    D_pred = pairwise_dist(pred_rel)
    D_gt = pairwise_dist(gt_rel)

    # 去掉对角线
    mask = ~torch.eye(J, dtype=torch.bool, device=pred_rel.device)
    D_pred = D_pred[:, mask]
    D_gt = D_gt[:, mask]

    # 计算 Pearson correlation
    pred_mean = D_pred.mean(dim=1, keepdim=True)
    gt_mean = D_gt.mean(dim=1, keepdim=True)

    pred_centered = D_pred - pred_mean
    gt_centered = D_gt - gt_mean

    numerator = (pred_centered * gt_centered).sum(dim=1)
    denominator = torch.sqrt(
        (pred_centered ** 2).sum(dim=1) *
        (gt_centered ** 2).sum(dim=1)
    ) + 1e-8

    corr = numerator / denominator  # (M,)

    return corr.mean()


def compute_jitter_metrics(pred_rel, valid_mask):
    # pred_rel: [B, T, J, 3] (mm)
    # valid_mask: [B, T] bool
    
    # velocity/accel/jerk existence masks
    m_v = valid_mask[:, 1:] & valid_mask[:, :-1]           # [B, T-1]
    m_a = m_v[:, 1:] & m_v[:, :-1]                         # [B, T-2]
    m_j = m_a[:, 1:] & m_a[:, :-1]                         # [B, T-3]

    v = diff1(pred_rel)  # [B, T-1, J, 3]
    a = diff2(pred_rel)  # [B, T-2, J, 3]
    j = diff3(pred_rel)  # [B, T-3, J, 3]

    out = {}
    if m_v.any():
        out["mpjv"] = masked_mean(v, m_v).item()           # 预测自身速度幅值
    if m_a.any():
        out["mpja"] = masked_mean(a, m_a).item()           # 预测自身加速度幅值（抖动）
    if m_j.any():
        out["mpjj"] = masked_mean(j, m_j).item()           # 预测自身jerk幅值（抖动）
    return out

def compute_dynamic_errors(pred_rel, gt_rel, valid_mask):
    # 与GT的速度/加速度/jerk误差
    m_v = valid_mask[:, 1:] & valid_mask[:, :-1]
    m_a = m_v[:, 1:] & m_v[:, :-1]
    m_j = m_a[:, 1:] & m_a[:, :-1]

    ev = diff1(pred_rel) - diff1(gt_rel)
    ea = diff2(pred_rel) - diff2(gt_rel)
    ej = diff3(pred_rel) - diff3(gt_rel)

    out = {}
    if m_v.any():
        out["mpjve"] = masked_mean(ev, m_v).item()
    if m_a.any():
        out["mpjae"] = masked_mean(ea, m_a).item()
    if m_j.any():
        out["mpjje"] = masked_mean(ej, m_j).item()
    return out
    
def compute_mpjpe(pred, gt):
    """
    pred, gt: [M, J, 3]  (mm)
    return: float (mm)
    """
    return torch.norm(pred - gt, dim=-1).mean()
    
def batch_procrustes_align(pred, gt, eps=1e-8):
    """
    Procrustes alignment (scale+rotation+translation) for each sample.
    pred, gt: [M, J, 3]
    return: pred_aligned [M, J, 3]
    """
    # subtract mean
    muX = pred.mean(dim=1, keepdim=True)  # [M,1,3]
    muY = gt.mean(dim=1, keepdim=True)
    X0 = pred - muX
    Y0 = gt - muY

    # compute covariance
    # [M,3,3] = [M,3,J] @ [M,J,3]
    H = torch.matmul(X0.transpose(1, 2), Y0)

    # SVD
    U, S, Vt = torch.linalg.svd(H)  # U:[M,3,3], Vt:[M,3,3]
    V = Vt.transpose(1, 2)

    # handle reflection
    detR = torch.det(torch.matmul(V, U.transpose(1, 2)))  # [M]
    sign = torch.ones_like(detR)
    sign[detR < 0] = -1.0

    # build correction matrix
    Z = torch.eye(3, device=pred.device, dtype=pred.dtype).unsqueeze(0).repeat(pred.shape[0], 1, 1)
    Z[:, 2, 2] = sign

    R = torch.matmul(torch.matmul(V, Z), U.transpose(1, 2))  # [M,3,3]

    # scale
    varX = (X0 ** 2).sum(dim=(1, 2)) + eps  # [M]
    scale = (S * Z.diagonal(dim1=-2, dim2=-1)).sum(dim=1) / varX  # [M]

    # aligned
    X_aligned = scale.view(-1, 1, 1) * torch.matmul(X0, R) + muY
    return X_aligned

def compute_pampjpe(pred, gt):
    """
    PA-MPJPE (Procrustes aligned MPJPE)
    pred, gt: [M, J, 3]
    return: float (mm)
    """
    pred_aligned = batch_procrustes_align(pred, gt)
    return compute_mpjpe(pred_aligned, gt)

def compute_pck(pred, gt, threshold_mm=50.0):
    """
    PCK@threshold: percentage of joints with error < threshold
    pred, gt: [M, J, 3]
    return: float in [0,1]
    """
    err = torch.norm(pred - gt, dim=-1)  # [M,J]
    return (err < threshold_mm).float().mean()

def compute_auc_pck(pred, gt, max_threshold=150.0, step=5.0):
    """
    AUC for PCK curve from 0..max_threshold (mm) with given step.
    pred, gt: [M, J, 3]
    return: float in [0,1]
    """
    thresholds = torch.arange(0.0, max_threshold + 1e-6, step, device=pred.device, dtype=pred.dtype)
    err = torch.norm(pred - gt, dim=-1)  # [M,J]
    pcks = [(err < th).float().mean() for th in thresholds]
    pcks = torch.stack(pcks)  # [K]
    # normalized area (divide by max_threshold)
    auc = torch.trapz(pcks, thresholds) / max_threshold
    return auc

def compute_mpjve(pred_seq, gt_seq):
    """
    MPJVE: Mean Per-Joint Velocity Error (mm/frame)
    pred_seq, gt_seq: [B, T, J, 3] (mm), already root-relative & masked outside (we will mask later)
    return: per-frame velocity error tensor [B, T-1] or scalar depending usage.
    """
    v_pred = pred_seq[:, 1:] - pred_seq[:, :-1]  # [B, T-1, J, 3]
    v_gt = gt_seq[:, 1:] - gt_seq[:, :-1]
    ve = torch.norm(v_pred - v_gt, dim=-1).mean(dim=-1)  # [B, T-1] (mean over joints)
    return ve
    
def compute_bone_length_mae(pred_rel, gt_rel, edges):
    """
    pred_rel, gt_rel: (M, J, 3)
    edges: list[(i,j)]
    return: scalar MAE over all bones
    """
    device = pred_rel.device
    idx_i = torch.tensor([e[0] for e in edges], device=device)
    idx_j = torch.tensor([e[1] for e in edges], device=device)

    d_pred = torch.norm(pred_rel[:, idx_i] - pred_rel[:, idx_j], dim=-1)  # (M,E)
    d_gt   = torch.norm(gt_rel[:, idx_i]   - gt_rel[:, idx_j], dim=-1)
    return (d_pred - d_gt).abs().mean()

def compute_bone_length_var(x_rel, edges):
    device = x_rel.device
    idx_i = torch.tensor([e[0] for e in edges], device=device)
    idx_j = torch.tensor([e[1] for e in edges], device=device)
    d = torch.norm(x_rel[:, idx_i] - x_rel[:, idx_j], dim=-1)  # (M,E)
    return d.var(dim=0).mean()  # across frames, then average bones

def compute_mjae(seq_pred, seq_gt):
    """
    seq_pred, seq_gt: (B, T, J, 3) root-relative mm
    return: (B, T-2) per-frame acceleration error (mean over joints)
    """
    a_pred = seq_pred[:, 2:] - 2*seq_pred[:, 1:-1] + seq_pred[:, :-2]
    a_gt   = seq_gt[:, 2:]   - 2*seq_gt[:, 1:-1]   + seq_gt[:, :-2]
    err = torch.norm(a_pred - a_gt, dim=-1).mean(dim=-1)  # (B, T-2)
    return err

def compute_jerk_energy(seq):
    """
    seq: (B, T, J, 3) root-relative mm
    return: scalar jerk energy (lower = smoother)
    """
    # third difference
    j = seq[:, 3:] - 3*seq[:, 2:-1] + 3*seq[:, 1:-2] - seq[:, :-3]
    return (j**2).sum(dim=-1).mean()

def best_of_k_mpjpe(samples, gt_rel):
    """
    samples: (K, M, J, 3)  K samples for each frame
    gt_rel:  (M, J, 3)
    return: scalar best-of-K MPJPE
    """
    # mpjpe per sample
    errs = torch.norm(samples - gt_rel.unsqueeze(0), dim=-1).mean(dim=-1)  # (K,M)
    best = errs.min(dim=0).values  # (M,)
    return best.mean()

def draw_skeleton_3d(ax, joints, edges, color, title=None):
    ax.cla()

    for i, j in edges:
        ax.plot(
            [joints[i, 0], joints[j, 0]],  # x
            [joints[i, 1], joints[j, 1]],  # y
            [joints[i, 2], joints[j, 2]],  # z
            color=color,
            linewidth=2
        )

    ax.scatter(
        joints[:, 0],
        joints[:, 1],
        joints[:, 2],
        color=color,
        s=20
    )

    # ---- 坐标轴范围（你的定义）----
    ax.set_xlim(-500, 500)        # x
    ax.set_ylim(1000, -1000)      # y：⚠️ 反向（负在上，正在下）
    ax.set_zlim(-500, 500)        # z

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 保持比例一致，避免人体被拉伸
    ax.set_box_aspect([1000, 2000, 1000])

    # 固定视角，防止动画抖动
    ax.view_init(elev=15, azim=70)

    if title:
        ax.set_title(title)
        
def plot_skeleton(gt_joints, pred_joints, edges, frame_id=0, out_html=None):
    # 可视化函数，与你之前一致
    # 注意：输入应该是 numpy 数组 (27, 3)
    gt_joints   -= gt_joints[0] # Root relative for vis
    pred_joints -= pred_joints[0]
    
    gt_scatter = go.Scatter3d(
        x=gt_joints[:,0], y=gt_joints[:,1], z=gt_joints[:,2],
        mode='markers', marker=dict(size=4, color='blue'), name='GT'
    )
    pred_scatter = go.Scatter3d(
        x=pred_joints[:,0], y=pred_joints[:,1], z=pred_joints[:,2],
        mode='markers', marker=dict(size=4, color='red'), name='Pred'
    )
    
    def make_lines(joints, color):
        xs, ys, zs = [], [], []
        for (i,j) in edges:
            xs += [float(joints[i,0]), float(joints[j,0]), None]
            ys += [float(joints[i,1]), float(joints[j,1]), None]
            zs += [float(joints[i,2]), float(joints[j,2]), None]
        return go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', line=dict(color=color, width=3))

    fig = go.Figure(data=[gt_scatter, pred_scatter, make_lines(gt_joints, 'blue'), make_lines(pred_joints, 'red')])
    fig.update_layout(scene=dict(aspectmode='data'), title=f"Frame {frame_id}")
    if out_html: fig.write_html(out_html)
    
def main():
    # --- 1. 设备与目录准备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "/code/radarFMv2/checkpoints_baseline3"
    os.makedirs(save_dir, exist_ok=True)
    
    # --- 2. 数据准备 ---
    dataset = RadarDiffusionDataset(root_dir="/code/radarFMv2/dataset", num_joints=27)
    train_set = dataset.get_train_set()
    val_set = dataset.get_val_set()

    train_loader = DataLoader(
        train_set, batch_size=8, shuffle=True, 
        collate_fn=collate_fn_for_cross_modal, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=8, shuffle=False, 
        collate_fn=collate_fn_for_cross_modal, num_workers=4, pin_memory=True
    )

    # --- 3. 模型初始化 ---
    model = RadarStage1Model(
        in_channels=6, radar_embed_dim=256, num_latents=64,
        num_joints=27, parent_list=PARENT
    ).to(device)

    ckpt_path = "/code/radarFMv2/checkpoints_baseline3/best_model.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    all_metrics = {}
    

    current_vis_dir = os.path.join(save_dir, f"vis")
    metrics = evaluate_sequence(
        dataloader=val_loader,
        model=model,
        device=device,
        vis_dir=current_vis_dir,  # 只在第一档可视化，避免生成太多文件
        vis_edges=EDGES,
        num_vis_samples=5,
    )
    all_metrics[1] = metrics
    
    for k, v in metrics.items():
        if isinstance(v, float) and not (isinstance(v, float) and math.isnan(v)):
                # pck 这类本来是 0~1，这里也照样打印；你想打印百分号我也可以改
            print(f"{k}: {v:.4f}")
        elif isinstance(v, float) and math.isnan(v):
            print(f"{k}: nan")
        else:
            print(f"{k}: {v}")

    # # --- 4. 优化器、混合精度与调度器 ---
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    # scaler = torch.cuda.amp.GradScaler()
    # # 动态调整学习率：如果5次验证误差不降，LR减半
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # # --- 5. 断点续传逻辑 (Resume) ---
    # start_epoch = 1
    # best_mpjpe = float('inf')
    # resume_path = os.path.join(save_dir, "latest.pt")

    # # --- 6. 训练循环 ---
    # for epoch in range(start_epoch, 201):
    #     model.train()
    #     total_loss = 0
        
    #     for radar_seq, skeleton_seq in train_loader:
    #         radar_seq, skeleton_seq = radar_seq.to(device), skeleton_seq.to(device)
    #         # valid_mask = valid_mask.to(device).bool()

    #         optimizer.zero_grad(set_to_none=True)
            
    #         with torch.cuda.amp.autocast():
    #             pred = model(radar_seq)
    #             loss = stage1_loss(pred, skeleton_seq)

    #         scaler.scale(loss).backward()
    #         scaler.unscale_(optimizer)
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪防止炸掉
    #         scaler.step(optimizer)
    #         scaler.update()
            
    #         total_loss += loss.item()

    #     # --- 7. 每 5 个 Epoch 验证并保存 ---
    #     if epoch % 2 == 0 or epoch == 1:
    #         metrics = evaluate_sequence(val_loader, model, device)
    #         mpjpe = metrics["mpjpe"]
    #         scheduler.step(mpjpe) # 更新学习率调度器
            
    #         print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Val MPJPE: {mpjpe:.2f}mm")

    #         # 构建保存字典
    #         checkpoint_data = {
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'best_mpjpe': best_mpjpe,
    #             'current_mpjpe': mpjpe
    #         }

    #         # 如果误差创新低，保存最好的模型
    #         if mpjpe < best_mpjpe:
    #             best_mpjpe = mpjpe
    #             checkpoint_data['best_mpjpe'] = best_mpjpe
    #             torch.save(checkpoint_data, os.path.join(save_dir, "best_model.pt"))
    #             print(f"⭐ 发现更好模型，已保存至 best_model.pt")

    # print("🎉 训练完成！")


# if __name__ == "__main__":
#     main()