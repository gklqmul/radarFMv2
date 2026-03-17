import math
import os
import torch
import torch.nn as nn
import torch.distributed as dist
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

# 假设 dataset.py 在同级目录下，请确保它里面的 _process_single_sample 已经是修复后的版本（输出6通道）
from dataset import DataReader, RadarDiffusionDataset
from flowmodels import collate_fn_for_cross_modal
from tools import _masked_dyn_sum, compute_auc_pck, compute_mpjpe, compute_pampjpe, compute_pck, compute_spatial_structure_corr, diff1, diff2, diff3, draw_skeleton_3d, plot_skeleton


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
    def __init__(self, in_channels=4, embed_dim=256, num_latents=64):
        super().__init__()
        # 物理特征 MLP (仅处理 x, y, z)
        self.phys_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        # 时间特征 MLP (处理第 4 维 time_offset)
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
        # pc: [B*T, 128, 4]
        phys_feat = pc[..., :3]  # x, y, z
        time_feat = pc[..., 3:]  # time_offset

        h = self.phys_mlp(phys_feat) + self.time_mlp(time_feat)
        h = self.fusion(h)
        feat_128 = self.transformer(h)

        B_T = pc.shape[0]
        latents = self.latents.repeat(B_T, 1, 1)
        z_compressed, _ = self.compress_attn(query=latents, key=feat_128, value=feat_128)
        return self.norm_out(z_compressed)

class CoarseSkeletonHead(nn.Module):
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
        direction = F.normalize(raw[..., :3], dim=-1)
        length = F.softplus(raw[..., 3])
        
        offsets = torch.zeros(B, J, 3, device=z.device)
        offsets[:, 1:] = direction * length.unsqueeze(-1)
        
        joints = torch.zeros_like(offsets)
        for j in range(1, J):
            joints[:, j] = joints[:, self.parent[j]] + offsets[:, j]
        return joints, offsets, length

class TemporalAdapter(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, z_flat, B, T):
        # z_flat: [B*T, 64, D] -> [B, T, 64, D]
        z = z_flat.view(B, T, 64, -1).permute(0, 2, 1, 3).reshape(B * 64, T, -1)
        z = self.transformer(z)
        return z.view(B, 64, T, -1).permute(0, 2, 1, 3).contiguous().view(B * T, 64, -1)

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
    """
    radar -> encoder -> temporal_adapter -> coarse_head/direct_head
    输入只有 4 维 + valid_mask
    """
    def __init__(
        self,
        in_channels=4,
        radar_embed_dim=256,
        num_latents=64,
        num_joints=17,
        parent_list=None,
        use_direct_head=False,
    ):
        super().__init__()
        assert in_channels == 4, "Expect radar_cond dim=6"

        self.encoder = TimeAwareCompressedRadarEncoder(
            in_channels=in_channels,
            embed_dim=radar_embed_dim,
            num_latents=num_latents
        )
        self.temporal_adapter = TemporalAdapter(embed_dim=radar_embed_dim)

        self.coarse_head = CoarseSkeletonHead(
            latent_dim=radar_embed_dim,
            num_joints=num_joints,
            parent=parent_list
        )
        self.direct_head = DirectJointHead(
            latent_dim=radar_embed_dim,
            num_joints=num_joints
        )

        self.use_direct_head = use_direct_head
        self.num_joints = num_joints

    def forward(self, radar_seq):
        """
        radar_seq:  [B, T, 128, 6]
        valid_mask: [B, T] bool, True=真实帧（强烈建议传）
        return: x0 [B, T, J, 3]
        """
        B, T, N, C = radar_seq.shape
        assert C == 4, f"radar_seq last dim should be 4, got {C}"

        radar_flat = radar_seq.view(B * T, N, C)

        # encoder: [BT,64,embed]
        z = self.encoder(radar_flat)

        # temporal adapter: mask padding frames
        z = self.temporal_adapter(z, B, T)

        z_global = z.mean(dim=1)  # [BT,embed]

        if self.use_direct_head:
            x0 = self.direct_head(z_global)       # [BT,J,3]
        else:
            x0, _, _ = self.coarse_head(z_global) # [BT,J,3]

        return x0.view(B, T, self.num_joints, 3)

# ==========================================
# 3. 损失函数与评估函数
# ==========================================

# def stage1_loss(pred, gt, parent_dict):
#     valid_mask = (gt.abs().sum(dim=(2, 3)) > 1e-6)
#     if not valid_mask.any():
#         return torch.tensor(0.0, device=pred.device, requires_grad=True)

#     # 1. 基础位姿误差 (米)
#     dist = torch.norm(pred - gt, dim=-1)
#     loss_mpjpe = dist[valid_mask].mean()

#     # 2. 几何结构误差
#     bone_len_losses = []
#     bone_dir_losses = []
    
#     for j, p in parent_dict.items():
#         vec_pred = pred[:, :, j] - pred[:, :, p]
#         vec_gt = gt[:, :, j] - gt[:, :, p]
        
#         # 长度误差 (MSE)
#         len_pred = torch.norm(vec_pred, dim=-1)
#         len_gt = torch.norm(vec_gt, dim=-1)
#         bone_len_losses.append(F.mse_loss(len_pred[valid_mask], len_gt[valid_mask]))
        
#         # 方向误差 (1 - CosSim)
#         cos_sim = F.cosine_similarity(vec_pred, vec_gt, dim=-1)
#         bone_dir_losses.append(1.0 - cos_sim[valid_mask].mean())

#     # --- 关键改动：取平均值，不要直接 sum ---
#     loss_geom = torch.stack(bone_len_losses).mean() + torch.stack(bone_dir_losses).mean()

#     # 3. 时序速度误差
#     if pred.shape[1] > 1:
#         vel_pred = pred[:, 1:] - pred[:, :-1]
#         vel_gt = gt[:, 1:] - gt[:, :-1]
#         # 给末端关节的速度误差加点权重，减少手脚乱抖
#         loss_vel = F.mse_loss(vel_pred, vel_gt)
#     else:
#         loss_vel = 0.0

#     # 4. 最终加权：保持 MPJPE 是大头
#     # 这样总 Loss 应该在 0.1 ~ 0.3 左右，更容易观察
#     total_loss = loss_mpjpe * 1.0 + loss_geom * 0.5 + loss_vel * 0.1
    
#     return total_loss
def stage1_loss(pred, gt):
    """
    pred, gt: [B, T, 27, 3] (meters, root-relative)
    """
    valid_mask = (gt.abs().sum(dim=(2, 3)) > 1e-6)  # 过滤 Padding 帧

    err = torch.norm(pred - gt, dim=-1)             # L2 距离 [B, T, 27]
    err = err.mean(dim=-1)                          # MPJPE [B, T]

    err = err[valid_mask]
    if err.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    return err.mean()
    
@torch.no_grad()
def evaluate_sequence(
    dataloader, model, device,
    vis_dir=None, vis_edges=None, num_vis_samples=2,
    steps=10,
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
    # if vis_dir is not None:
    #     os.makedirs(vis_dir, exist_ok=True)
    has_visualized = 0

    # edges for bone metrics / visualization
    edge_i = edge_j = None
    if vis_edges:
        edge_i = torch.tensor([e[0] for e in vis_edges], device=device, dtype=torch.long)
        edge_j = torch.tensor([e[1] for e in vis_edges], device=device, dtype=torch.long)

    pbar = tqdm(dataloader, desc=f"Eval (Steps={steps})")

    for batch_idx, (radar_seq, skeleton_seq) in enumerate(pbar):
        radar_seq = radar_seq.to(device).float()       # [B,T,...]
        skeleton_seq = skeleton_seq.to(device).float() # [B,T,J,3] in meters
        B, T = skeleton_seq.shape[0], skeleton_seq.shape[1]
        indices = torch.tensor([0, 1, 2, 5]).to(device)
        radar_input = torch.index_select(radar_seq, -1, indices)

        # -----------------------
        # 1) Valid mask from GT
        # -----------------------
        valid_mask = (skeleton_seq.abs().sum(dim=(2, 3)) > 1e-6)  # [B,T] bool
        if not valid_mask.any():
            continue

        # -----------------------
        # 2) Inference
        # -----------------------
        pred = model(radar_input)  # [B,T,J,3] meters

        # -----------------------
        # 3) Convert to mm + root-relative
        # -----------------------
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
            v_err = diff1(pred_rel) - diff1(gt_rel)  # [B,T-1,J,3]
            s, c = _masked_dyn_sum(v_err, m_v)
            sum_mpjve += s.item()
            n_vel += int(c.item())

            # also record as "dyn" explicitly
            sum_mpjve_dyn += s.item()
            n_mpjve_dyn += int(c.item())

            # pred-only velocity magnitude
            v_mag = diff1(pred_rel)
            s, c = _masked_dyn_sum(v_mag, m_v)
            sum_mpjv_pred += s.item()
            n_mpjv += int(c.item())

        # acceleration error vs GT (MJAE)
        if m_a.any():
            a_err = diff2(pred_rel) - diff2(gt_rel)  # [B,T-2,J,3]
            s, c = _masked_dyn_sum(a_err, m_a)
            sum_mjae += s.item()
            n_acc += int(c.item())

            sum_mpjae_dyn += s.item()
            n_mpjae_dyn += int(c.item())

            a_mag = diff2(pred_rel)
            s, c = _masked_dyn_sum(a_mag, m_a)
            sum_mpja_pred += s.item()
            n_mpja += int(c.item())

        # jerk error vs GT (optional but consistent)
        if m_j.any():
            j_err = diff3(pred_rel) - diff3(gt_rel)  # [B,T-3,J,3]
            s, c = _masked_dyn_sum(j_err, m_j)
            sum_mpjje_dyn += s.item()
            n_mpjje_dyn += int(c.item())

            j_mag = diff3(pred_rel)
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

        # if vis_dir and has_visualized < num_vis_samples and vis_edges:
        #         # 找到当前 Batch 中有有效数据的样本索引
        #         valid_bs = [i for i in range(B) if valid_mask[i].any().item()]

        #         if valid_bs:
        #             # 1) 随机锁定一个样本 b
        #             b = random.choice(valid_bs)
        #             # 获取该样本下所有有效的帧索引 (全长)
        #             valid_indices = torch.where(valid_mask[b])[0]  # (T_valid,)

        #             if valid_indices.numel() > 0:
        #                 # --- A) 生成全长 GIF ---
        #                 gt_seq_full = gt_rel[b, valid_indices].detach().cpu().numpy()
        #                 pred_seq_full = pred_rel[b, valid_indices].detach().cpu().numpy()
        #                 num_total_frames = gt_seq_full.shape[0]

        #                 fig = plt.figure(figsize=(10, 5))
        #                 ax_gt = fig.add_subplot(121, projection="3d")
        #                 ax_pr = fig.add_subplot(122, projection="3d")

        #                 def update(idx):
        #                     ax_gt.cla()
        #                     ax_pr.cla()
        #                     actual_frame_idx = int(valid_indices[idx].item())
        #                     draw_skeleton_3d(ax_gt, gt_seq_full[idx], vis_edges, "green", f"GT Frame {actual_frame_idx}")
        #                     draw_skeleton_3d(ax_pr, pred_seq_full[idx], vis_edges, "red", f"Pred Frame {actual_frame_idx}")
        #                     return []

        #                 ani = animation.FuncAnimation(fig, update, frames=num_total_frames, interval=100)
        #                 gif_name = f"Sample{has_visualized}_B{batch_idx}_S{b}_full_seq.gif"
        #                 ani.save(os.path.join(vis_dir, gif_name), writer="pillow", fps=10)
        #                 plt.close(fig)

        #                 # --- B) 生成相同样本的 10 帧 HTML ---
        #                 num_html = 10
        #                 if num_total_frames > num_html:
        #                     start_f = num_total_frames // 4
        #                     html_indices = valid_indices[start_f:start_f + num_html]
        #                 else:
        #                     html_indices = valid_indices

        #                 for t_idx in html_indices:
        #                     t = int(t_idx.item())
        #                     html_name = f"Sample{has_visualized}_B{batch_idx}_f{t:03d}.html"
        #                     plot_skeleton(
        #                         gt_joints=gt_rel[b, t].detach().cpu().numpy(),
        #                         pred_joints=pred_rel[b, t].detach().cpu().numpy(),
        #                         edges=vis_edges,
        #                         frame_id=f"Sample{has_visualized}-Frame{t}",
        #                         out_html=os.path.join(vis_dir, html_name)
        #                     )

        #                 has_visualized += 1

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

def main():
    # --- 1. 设备与目录准备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "/code/radarFMv2/checkpoints_baseline10"
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
        in_channels=4, radar_embed_dim=256, num_latents=64,
        num_joints=27, parent_list=PARENT
    ).to(device)

    ckpt_path = "/code/radarFMv2/checkpoints_baseline10/best_model.pt"
    # ckpt_path = "./checkpoints_refine/best_refiner.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    steps_list = [1]

    all_metrics = {}
    
    for s in steps_list:
        current_vis_dir = os.path.join(save_dir, f"vis_s{s}") if s == 1 else None
        metrics = evaluate_sequence(
            dataloader=val_loader,
            model=model,
            device=device,
            vis_dir=current_vis_dir,  # 只在第一档可视化，避免生成太多文件
            vis_edges=EDGES,
            num_vis_samples=5,
            steps=s
        )
        all_metrics[s] = metrics
    # --- 核心：生成 Excel 专用多行块 ---
    print("\n" + "="*40)
    print("请从下方【表头】开始全选，粘贴至 Excel:")
    print("="*40)

    if steps_list:
        sample_metrics = all_metrics[steps_list[0]]
        # 表头：Steps + 所有的指标 Key
        header = ["Steps"] + list(sample_metrics.keys())
        print("\t".join(header))

        # 2. 遍历每个 step，打印对应的一行数据
        for s in steps_list:
            row = [str(s)]  # 第一列放 step 数值
            metrics = all_metrics[s]
            
            for k in sample_metrics.keys():
                v = metrics.get(k, "N/A")
                # 格式化浮点数
                if isinstance(v, float):
                    val_str = f"{v:.4f}" if not math.isnan(v) else "nan"
                else:
                    val_str = str(v)
                row.append(val_str)
            
            # 将这一行用制表符连接并打印
            print("\t".join(row))

    print("="*40)

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
    #         radar_seq = radar_seq.to(device)
    #         skeleton_seq = skeleton_seq.to(device)
    #         indices = torch.tensor([0, 1, 2, 5]).to(device)
    #         radar_input = torch.index_select(radar_seq, -1, indices)
    #         # valid_mask = valid_mask.to(device).bool()
        
    #         optimizer.zero_grad(set_to_none=True)
        
    #         with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    #             pred = model(radar_input)
    #             loss = stage1_loss(pred, skeleton_seq)
            
    #         scaler.scale(loss).backward()
    #         scaler.unscale_(optimizer)
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #         scaler.step(optimizer)
    #         scaler.update()
            
    #         total_loss += loss.item()

    #     # --- 7. 每 5 个 Epoch 验证并保存 ---
    #     if epoch % 2 == 0 or epoch == 1:
    #         mpjpe = eval_stage1(val_loader, model, device, epoch, EDGES_27)
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

    #         # 永远保存最新的，防止服务器宕机
    #         torch.save(checkpoint_data, os.path.join(save_dir, "latest.pt"))

    #         # 如果误差创新低，保存最好的模型
    #         if mpjpe < best_mpjpe:
    #             best_mpjpe = mpjpe
    #             checkpoint_data['best_mpjpe'] = best_mpjpe
    #             torch.save(checkpoint_data, os.path.join(save_dir, "best_model.pt"))
    #             print(f"⭐ 发现更好模型，已保存至 best_model.pt")

    # print("🎉 训练完成！")































    