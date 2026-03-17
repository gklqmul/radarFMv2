import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from tqdm.auto import tqdm
import random
import matplotlib.pyplot as plt
from matplotlib import animation
import math
import yaml

# 外部依赖
from mmfi_lib.mmfidataset import make_dataset, make_dataloader
from tools import _masked_dyn_sum, compute_auc_pck, compute_mpjpe, compute_pampjpe, compute_pck, compute_spatial_structure_corr, diff1, diff2, diff3, draw_skeleton_3d, plot_skeleton, batch_procrustes_align

EDGES = [
    # --- 躯干中轴 (Central Axis) ---
    (0, 7),   # hip -> spine
    (7, 8),   # spine -> thorax
    (8, 9),   # thorax -> neck
    (9, 10),  # neck -> head

    # --- 右腿 (Right Leg) ---
    (0, 1),   # hip -> rhip
    (1, 2),   # rhip -> rknee
    (2, 3),   # rknee -> rfoot

    # --- 左腿 (Left Leg) ---
    (0, 4),   # hip -> lhip
    (4, 5),   # lhip -> lknee
    (5, 6),   # lknee -> lfoot

    # --- 左臂 (Left Arm) - 连接到 8 (thorax) ---
    (8, 11),  # thorax -> lshoulder
    (11, 12), # lshoulder -> lelbow
    (12, 13), # lelbow -> lwrist

    # --- 右臂 (Right Arm) - 连接到 8 (thorax) ---
    (8, 14),  # thorax -> rshoulder
    (14, 15), # rshoulder -> relbow
    (15, 16)  # relbow -> rwrist
]

PARENT = {
    7: 0,
    8: 7,
    9: 8,
    10: 9,
    1: 0,
    2: 1,
    3: 2,
    4: 0,
    5: 4,
    6: 5,
    11: 8,
    12: 11,
    13: 12,
    14: 8,
    15: 14,
    16: 15
}
PARENT_LIST = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9,  8,  11, 12, 8,  14, 15]

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
    def __init__(self, latent_dim=256, num_joints=17):
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


def GFM_skeleton_mask(parent_list, num_joints=17):
    """构造骨架拓扑掩码：0.0 表示允许关注，-inf 表示禁止关注"""
    mask = torch.full((num_joints, num_joints), float('-inf'))
    for child, parent in enumerate(parent_list):
        mask[child, child] = 0.0
        if child != parent:
            mask[child, parent] = 0.0
            mask[parent, child] = 0.0
    return mask

class SingleFrameFlowTransformer(nn.Module):
    """
    输出 v(x_t, t | radar, x0) 作为 ODE 速度场 dx/dt
    """
    def __init__(self, num_joints=17, radar_in_channels=6, embed_dim=512,
                 local_radius=0.1, parent_list=PARENT_LIST, nhead=8, num_layers=6):
        super().__init__()
        self.num_joints = num_joints
        self.base_radius = local_radius
        self.nhead = nhead

        # 1) 特征投影
        self.pc_proj = nn.Linear(radar_in_channels, embed_dim)
        self.joint_embed = nn.Linear(3, embed_dim)
        self.coarse_embed = nn.Linear(3, embed_dim)
        self.diff_embed = nn.Linear(3, embed_dim)
        self.joint_id_embed = nn.Embedding(num_joints, embed_dim)

        # Doppler feature -> embed_dim
        self.doppler_head = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, embed_dim)
        )

        # 时间嵌入（t in [0,1]）
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 融合结构：当前关节 + parent 关节
        self.diffusion_proj = nn.Linear(embed_dim * 2, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=2048,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        if parent_list is not None:
            self.register_buffer("topo_mask", GFM_skeleton_mask(parent_list, num_joints))
            self.register_buffer("p_idx", torch.tensor(parent_list).long())
        else:
            self.topo_mask = None
            self.p_idx = None

        # 局部半径缩放：核心关节更小、末端更大
        scales = torch.ones(num_joints)
        core_joints = [0, 1, 4, 7, 8]
        scales[core_joints] = 0.6
        self.end_effectors = [3,6,13,16,10]
        scales[self.end_effectors] = 1.5
        self.register_buffer("radius_scales", scales.view(1, num_joints, 1))

        # 输出速度 v = dx/dt
        self.vel_head = nn.Linear(embed_dim, 3)

    def forward(self, x_t, t, x0, pc_raw):
        """
        x_t:   [BT, J, 3] 当前状态
        t:     [BT, 1]    当前时间(0~1)
        x0:    [BT, J, 3] coarse prior (起点条件)
        pc_raw:[BT, N, 6] radar point cloud
        return v: [BT, J, 3] 速度场 dx/dt
        """
        BT, J, _ = x_t.shape
        device = x_t.device

        pc_xyz = pc_raw[:, :, :3]          # [BT, N, 3]
        pc_doppler = pc_raw[:, :, 3:4]     # [BT, N, 1]

        # --- 1) memory ---
        raw_memory = self.pc_proj(pc_raw)  # [BT, N, D]

        # --- 2) density weighting based on current x_t ---
        dist = torch.cdist(x_t, pc_xyz, p=2)  # [BT, J, N]
        adaptive_radius = self.radius_scales * self.base_radius  # [1,J,1]
        adaptive_sigma = adaptive_radius / 2.0

        density_weight = torch.exp(-(dist ** 2) / (2 * adaptive_sigma ** 2))  # [BT,J,N]
        point_importance, _ = density_weight.max(dim=1, keepdim=True)         # [BT,1,N]
        memory = raw_memory * point_importance.transpose(1, 2)               # [BT,N,D]

        # --- 3) doppler global feature ---
        doppler_feat = self.doppler_head(pc_doppler)  # [BT,N,D]

        local_denom = density_weight.sum(dim=-1, keepdim=True) + 1e-6  # [BT,J,1]
        h_doppler = torch.einsum("bjn,bnd->bjd", density_weight, doppler_feat) / local_denom  # [BT,J,D]

        # --- 4) query tokens ---
        h_xt = self.joint_embed(x_t)                 # [BT,J,D]
        h_t = self.time_embed(t).unsqueeze(1)        # [BT,1,D]
        h0 = self.coarse_embed(x0)                   # [BT,J,D]
        h_diff = self.diff_embed(x_t - x0)           # [BT,J,D]
        h_id = self.joint_id_embed(torch.arange(J, device=device)).unsqueeze(0)  # [1,J,D]

        if self.p_idx is not None:
            h_parent = h_xt[:, self.p_idx, :]        # [BT,J,D]
            h_struct = self.diffusion_proj(torch.cat([h_xt, h_parent], dim=-1))
        else:
            h_struct = h_xt

        query = h_struct + h0 + h_diff + h_id + h_doppler + h_t  # broadcast h_t

        # --- 5) spatial mask ---
        spatial_mask = dist > adaptive_radius  # [BT,J,N]
        # rescue: ensure each joint attends at least one point
        min_idx = dist.argmin(dim=-1, keepdim=True)  # [BT,J,1]
        rescue = ~torch.zeros_like(spatial_mask).scatter_(-1, min_idx, 1).bool()
        spatial_mask = torch.where(spatial_mask.all(dim=-1, keepdim=True), rescue, spatial_mask)

        mem_mask = spatial_mask.repeat_interleave(self.nhead, dim=0)  # [BT*nhead,J,N]

        refined_feat = self.transformer(
            tgt=query,
            memory=memory,
            tgt_mask=self.topo_mask,
            memory_mask=mem_mask
        )

        v = self.vel_head(refined_feat)  # [BT,J,3]
        return v

class RadarPoseRefiner(nn.Module):
    """
    Stage1: radar -> coarse x0
    Stage2: learn ODE velocity field v(x,t|radar,x0), then integrate with multiple steps
    """
    def __init__(self, in_channels=6, radar_embed_dim=256, num_latents=64,
                 num_joints=17, parent_list=PARENT_LIST, refine_embed_dim=512):
        super().__init__()

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

        self.refiner = SingleFrameFlowTransformer(
            num_joints=num_joints,
            radar_in_channels=6,
            embed_dim=refine_embed_dim,
            parent_list=parent_list
        )
        # self.direct_head = DirectJointHead(
        #     latent_dim=radar_embed_dim,
        #     num_joints=num_joints
        # )

        self.parent_dict = PARENT
        self.parent_list = PARENT_LIST
        self.num_joints = num_joints

        # 末端加权（可保留）
        self.end_effectors = [3,6,13,16,10]

    def get_coarse_prior(self, radar_input):
        """
        radar_input: [B,T,N,6]
        return: x0 [BT,J,3], z_seq_flat [BT,64,256]
        """
        B, T, N, C = radar_input.shape
        radar_flat = radar_input.view(B * T, N, C)

        z_seq = self.encoder(radar_flat)                 # [BT,64,256]
        z_seq_flat = self.temporal_adapter(z_seq, B, T)  # [BT,64,256]

        z_global = z_seq_flat.mean(dim=1)                # [BT,256]
        x0, _, _ = self.coarse_head(z_global)            # [BT,J,3]
        return z_seq_flat, x0

    def forward(self, x_t, t, radar_input):
        """
        return v(x_t,t|radar,x0)
        """
        with torch.no_grad():
            _, x0 = self.get_coarse_prior(radar_input)
        radar_raw = radar_input.flatten(0, 1)  # [BT,N,6]
        v_pred = self.refiner(x_t, t, x0, radar_raw)
        return v_pred

    def compute_fm_loss(self, radar_input, x_gt, noise_level=0.01, eps=1e-4):
        """
        多步积分友好的 FM/rectified-flow 风格监督：
        v_target(t) = (x1 - x_t)/(1-t)
        使 v 真正成为 dx/dt 的速度场（越接近 t=1 速度越小）
        """
        B, T = radar_input.shape[:2]
        device = radar_input.device
        x1 = x_gt.reshape(B * T, self.num_joints, 3)  # x_gt_flat


        # 2) sample t and build bridge x_t
        # t = torch.rand(B * T, 1, device=device)  # [BT,1]
        t = torch.distributions.Beta(2,2).sample((B*T,1)).to(device)
        t_b = t.view(-1, 1, 1)

         # 1) coarse prior (detached)
        with torch.no_grad():
            _, x0 = self.get_coarse_prior(radar_input)
            x0 = x0.detach()
            if noise_level > 0:
                # x0 = x0 + torch.randn_like(x0) * noise_level
                sigma = noise_level * (1 - t_b)
                x0 = x0 + torch.randn_like(x0) * sigma

        # linear bridge
        x_t = (1 - t_b) * x0 + t_b * x1

        # ✅ t-dependent velocity target: remaining-to-go speed
        # v*(x_t,t) = (x1 - x_t) / (1 - t)
        # v_target = (x1 - x_t) / (1.0 - t_b + eps)
        v_target = x1 - x0

        radar_raw = radar_input.flatten(0, 1)
        v_pred = self.refiner(x_t, t, x0, radar_raw)

        # 3) velocity loss
        loss_unit = F.mse_loss(v_pred, v_target, reduction='none')  # [BT,J,3]
        weights = torch.ones(self.num_joints, device=device)
        weights[self.end_effectors] = 3.0

        # （可选）时间权重：越靠近 1 越强调精修（你原先那套也能用）
        time_weights = 1.0 + t.view(-1, 1, 1) * 2.0

        loss_velocity = (loss_unit * time_weights).mean(dim=-1)  # [BT,J]
        loss_velocity = (loss_velocity * weights).mean()

        # 4) geometry auxiliary loss（用“多步推理的终点”更合理）
        # 这里为了训练效率，用一个便宜的“近似终点”：从 x_t 走到 1 的单步欧拉：
        # x1_hat ≈ x_t + (1-t)*v_pred
        x1_hat = x_t + (1.0 - t_b) * v_pred
        # x1_hat = x0 + 1.0 * v_pred

        gt_lens, gt_dirs = self.calc_bone_lengths_and_dirs(x1, self.parent_dict)
        pred_lens, pred_dirs = self.calc_bone_lengths_and_dirs(x1_hat, self.parent_dict)

        loss_geom = F.mse_loss(pred_lens, gt_lens)
        loss_dir = F.mse_loss(pred_dirs, gt_dirs)

        # scale（按你原先量级习惯）
        total = loss_velocity * 100.0 + (loss_geom * 100.0 + 0.1 * loss_dir * 10.0)
        return total

    def calc_bone_lengths_and_dirs(self, x, parent_dict):
        children = sorted(parent_dict.keys())
        parents = [parent_dict[c] for c in children]

        child_pos = x[:, children, :]
        parent_pos = x[:, parents, :]
        diff = child_pos - parent_pos

        lengths = torch.norm(diff, dim=-1)
        directions = F.normalize(diff, dim=-1, eps=1e-6)
        return lengths, directions

    @torch.no_grad()
    def inference(self, radar_input, steps=3, method="heun"):
        """
        多步 ODE 推理：
        - method="euler": 便宜，但误差大
        - method="heun": 二阶（预测-校正），通常更稳，同样 steps 下更准
        """
        self.eval()
        B, T = radar_input.shape[:2]
        device = radar_input.device
        BT = B * T

        _, x0 = self.get_coarse_prior(radar_input)

        radar_raw = radar_input.flatten(0, 1)  # [BT,N,6]

        x = x0.clone()
        dt = 1.0 / steps

        for i in range(steps):
            t0 = torch.full((BT, 1), i / steps, device=device)

            if method.lower() == "euler":
                v = self.refiner(x, t0, x0, radar_raw)
                x = x + dt * v

            elif method.lower() == "heun":
                # predictor
                v0 = self.refiner(x, t0, x0, radar_raw)
                x_euler = x + dt * v0

                # corrector at next time
                t1 = torch.full((BT, 1), (i + 1) / steps, device=device)
                v1 = self.refiner(x_euler, t1, x0, radar_raw)

                x = x + dt * 0.5 * (v0 + v1)

            else:
                raise ValueError(f"Unknown method={method}, choose from ['euler','heun']")

        return x.view(B, T, self.num_joints, 3)

    def load_pretrained_stage1(self, ckpt_path, freeze=True):
        if not os.path.exists(ckpt_path):
            print(f"Warning: Stage 1 checkpoint not found at {ckpt_path}")
            return

        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

        self.load_state_dict(state_dict, strict=False)
        print(f"✅ Successfully loaded Stage 1 weights from {ckpt_path}")

        if freeze:
            for name, param in self.named_parameters():
                if "refiner" not in name:
                    param.requires_grad = False
            print("❄️ Stage 1 components are frozen.")



@torch.no_grad()
def evaluate_sequence(
    dataloader, model, device, 
    vis_dir=None, vis_edges=None, num_vis_samples=2,
    steps=10,
    pck_thresholds=(50.0, 100.0),
    auc_max_threshold=80.0,
    auc_step=5.0,
):
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

    # edges for bone metrics / visualization
    edge_i = edge_j = None
    if vis_edges:
        edge_i = torch.tensor([e[0] for e in vis_edges], device=device, dtype=torch.long)
        edge_j = torch.tensor([e[1] for e in vis_edges], device=device, dtype=torch.long)

    pbar = tqdm(dataloader, desc=f"Eval (Steps={steps})")

    for batch_idx, batch in enumerate(dataloader):
        radar_seq = batch['radar_cond']
        skeleton_seq = batch['pointcloud']
        radar_seq = radar_seq.to(device).float()     # [B, T, 128, 6]
        skeleton_seq = skeleton_seq.to(device).float() # [B, T, 17, 3]

        B, T = skeleton_seq.shape[0], skeleton_seq.shape[1]

        # -----------------------
        # 1) Valid mask from GT
        # -----------------------
        valid_mask = (skeleton_seq.abs().sum(dim=(2, 3)) > 1e-6)  # [B,T] bool
        if not valid_mask.any():
            continue

        # -----------------------
        # 2) Inference
        # -----------------------
        pred = model.inference(radar_seq, steps=steps)  # [B,T,J,3] meters

        # -----------------------
        # 3) Convert to mm + root-relative
        # -----------------------
        pred_mm = pred * 1000.0
        gt_mm = skeleton_seq * 1000.0
        pred_rel = pred_mm - pred_mm[:, :, 0:1, :]
        gt_rel = gt_mm - gt_mm[:, :, 0:1, :]
        aligned_pred_rel_full = torch.zeros_like(pred_rel)

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
        aligned_pred = batch_procrustes_align(v_pred,v_gt)
        aligned_pred_rel = aligned_pred - aligned_pred[:, 0:1, :]
        
        pampjpe_val = compute_mpjpe(aligned_pred, v_gt).item()
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
                        aligned_pred_rel_full[valid_mask] = aligned_pred_rel
                        aligned_seq_full = aligned_pred_rel_full[b, valid_indices].detach().cpu().numpy()
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
                                aligned= aligned_pred_rel_full[b, t].detach().cpu().numpy(),
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

# ==========================================
# 4. Main 训练循环
# ==========================================

def main():
    # --- 1. 设备与环境配置 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 性能优化：启用算力加速
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # 启用原生算子加速
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # 设置保存路径
    save_path = "/data/mmfi/checkpoints_basemodel_cross_act"
    os.makedirs(save_path, exist_ok=True)
    
   # --- 2. 数据准备 ---
    dataset_root = '/data/mmfi/combined'
    config_file = '/code/mmfi/config.yaml'
    with open(config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    train_dataset, val_dataset = make_dataset(dataset_root, config)

    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])

    # --- 3. 模型实例化与权重加载 ---
    model = RadarPoseRefiner(
        in_channels=6,
        radar_embed_dim=256,
        num_latents=64,
        num_joints=17,
        parent_list=PARENT_LIST,
        refine_embed_dim=512
    ).to(device)

    # # 加载你那个 98mm 的 Stage 1 权重
    # # freeze=True 意味着我们只训练 Refiner，保持 Stage 1 不动
    # ckpt_stage1 = "/data/mmfi/checkpoints_baseline1_cross_act/best_model.pt"
    # model.load_pretrained_stage1(ckpt_stage1, freeze=True)
   

    ckpt_path = "/data/mmfi/checkpoints_basemodel_cross_act/best_refiner.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()

    steps_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # steps_list = [1]

    all_metrics = {}
    
    for s in steps_list:
        current_vis_dir = os.path.join(save_path, f"vis_s{s}") if s == 1 else None
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

    # # --- 4. 优化器配置 ---
    # # 仅更新需要梯度的参数（即 Refiner 部分）
    # optimizer = torch.optim.AdamW(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=1e-4,
    #     weight_decay=1e-2
    # )
    
    # # 学习率调度：验证集 MPJPE 停滞时减半
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # # 启用 AMP 混合精度训练
    # scaler = torch.amp.GradScaler('cuda')

    # # --- 5. 训练循环 ---
    # best_mpjpe = float("inf")
    
    # for epoch in range(1, 150):
    #     model.train()
    #     total_epoch_loss = 0.0
        
    #     for batch_idx, batch in enumerate(train_loader):
    #         radar_seq = batch['radar_cond']
    #         skeleton_seq = batch['pointcloud']
    #         radar_seq = radar_seq.to(device).float()     # [B, T, 128, 6]
    #         skeleton_seq = skeleton_seq.to(device).float() # [B, T, 17, 3]

    #         optimizer.zero_grad(set_to_none=True)

    #         # 使用混合精度训练
    #         with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    #             # 调用模型内置的 FM Loss 计算函数
    #             loss = model.compute_fm_loss(radar_seq, skeleton_seq)

    #         scaler.scale(loss).backward()
    #         # 梯度裁剪防止 Refiner 训练早期震荡
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         scaler.step(optimizer)
    #         scaler.update()
            
    #         total_epoch_loss += loss.item()
         

    #     # --- 6. 验证与保存 ---
    #     if epoch % 2 == 0 or epoch == 1:
    #         # 调用你优化后的 evaluate_sequence 函数
    #         metrics = evaluate_sequence(
    #             dataloader=val_loader,
    #             model=model,
    #             device=device,
    #             vis_edges=EDGES,
    #             num_vis_samples=1,
    #             steps=3  # 验证时使用 10 步 ODE 积分
    #         )

    #         val_mpjpe = metrics["mpjpe"]
    #         print(f"\n[Epoch {epoch}] Train Loss: {total_epoch_loss/len(train_loader):.6f} | Val MPJPE: {val_mpjpe:.2f}mm")
            
    #         scheduler.step(val_mpjpe)

    #         if val_mpjpe < best_mpjpe:
    #             best_mpjpe = val_mpjpe
    #             save_file = f"{save_path}/best_refiner.pt"
    #             torch.save({
    #                 'epoch': epoch,
    #                 'model_state_dict': model.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'best_mpjpe': best_mpjpe,
    #             }, save_file)
    #             print(f"⭐ New Best Model Saved: {val_mpjpe:.2f}mm")

    # print("Training Complete!")

if __name__ == "__main__":
    main()