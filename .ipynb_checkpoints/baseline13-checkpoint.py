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

# 假设 dataset.py 在同级目录下，请确保它里面的 _process_single_sample 已经是修复后的版本（输出6通道）
from dataset import DataReader, RadarDiffusionDataset
from flowmodels import collate_fn_for_cross_modal
from tools import _masked_dyn_sum, compute_auc_pck, compute_mpjpe, compute_pampjpe, compute_pck, compute_spatial_structure_corr, diff1, diff2, diff3, draw_skeleton_3d, plot_skeleton

# ==========================================
# 1. 常量定义
# ==========================================
EDGES = [
    (0,1), (1,2), (2,3), (3,26), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (8,10),
    (3,11), (11,12), (12,13), (13,14), (14,15), (15,16), (15,17),
    (0,18), (18,19), (19,20), (20,21), (0,22), (22,23), (23,24), (24,25)
]
PARENT = {
    0:0, 1:0, 2:1, 3:2, 26:3,
    4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:8,
    11:3, 12:11, 13:12, 14:13, 15:14, 16:15, 17:15,
    18:0, 19:18, 20:19, 21:20,
    22:0, 23:22, 24:23, 25:24
}
PARENT_LIST = [
    0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  8, 
    3, 11, 12, 13, 14, 15, 15,  0, 18, 19, 20, 
    0, 22, 23, 24,  3
]
# ==========================================
# 2. 模型定义
# ==========================================

class CoarseSkeletonHead(nn.Module):
    """
    Stage 1: 从全局雷达特征生成初始骨架 x0
    Input: [B, radar_embed_dim]
    Output: [B, 27, 3]
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
        # z: [B, 256]
        B = z.shape[0]
        J = self.num_joints
        raw = self.mlp(z).view(B, J - 1, 4)

        dir_raw = raw[..., :3]              # [B, J-1, 3]
        len_raw = raw[..., 3]               # [B, J-1]

        direction = F.normalize(dir_raw, dim=-1, eps=1e-6)
        length = F.softplus(len_raw)        # [B, J-1]
        offset_non_root = direction * length.unsqueeze(-1)

        offsets = torch.zeros(B, J, 3, device=z.device)
        offsets[:, 1:] = offset_non_root

        joints = torch.zeros_like(offsets)
        joints[:, 0] = 0.0                  # root at origin

        # Forward Kinematics
        for j in range(1, J):
            p = self.parent[j]
            joints[:, j] = joints[:, p] + offsets[:, j]

        return joints, offsets, length # Returns x_coarse [B, 27, 3]

class TemporalAdapter(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        # 这是一个专门处理时间维度的 Transformer
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim*2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, z_flat, B, T):
        """
        Input:  z_flat [B*T, 64, 256] (来自 Encoder 的输出)
        Output: z_refined_flat [B*T, 64, 256] (增强了时序信息后再次压平)
        """
        # 1. 恢复时间维度: [B*T, 64, 256] -> [B, T, 64, 256]
        z = z_flat.view(B, T, 64, 256)
        
        # 2. 维度置换: 
        # 我们希望 Attention 发生在 T 维度上。
        # 对于 Transformer 来说，Input 应该是 (Batch_Size, Seq_Len, Dim)
        # 这里 Seq_Len = T。
        # 这里的 "Batch_Size" 应该是 "真实的 Batch * Latent点数 (64)"
        # 意思是：第 b 个样本的第 n 个 latent点，在 T 帧之间做交互。
        
        # [B, T, 64, 256] -> [B, 64, T, 256] -> [B*64, T, 256]
        z = z.permute(0, 2, 1, 3).reshape(B * 64, T, 256)
        
        # 3. 时序 Attention (混合前后帧信息)
        z = self.transformer(z) # [B*64, T, 256]
        
        # 4. 还原形状: [B*64, T, 256] -> [B, 64, T, 256] -> [B, T, 64, 256]
        z = z.view(B, 64, T, 256).permute(0, 2, 1, 3).contiguous()
        
        # 5. 再次压平，准备喂给 Coarse Head
        z_refined_flat = z.reshape(B * T, 64, 256)
        
        return z_refined_flat
        

class TimeAwareCompressedRadarEncoder(nn.Module):
    """
    Encoder: 处理原始点云，输出压缩后的特征序列
    Input: [B, 128, 6]
    Output: [B, 64, 256]
    """
    def __init__(self, in_channels=6, embed_dim=256, num_latents=64):
        super().__init__()
        
        # 物理特征编码 (x, y, z, doppler, snr) -> 前5维
        self.phys_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        # 时间特征编码 (time_idx) -> 第6维
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

        # 初始 Transformer (128点互交)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*2, 
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 压缩模块
        self.latents = nn.Parameter(torch.randn(1, num_latents, embed_dim))
        self.compress_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, batch_first=True
        )
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, pc):
        # pc: [B, 128, 6]
        
        pc[..., 4] = pc[..., 4] / 1000.0
        phys_feat = pc[..., :5] # [B, 128, 5]
        time_idx = pc[..., 5:]  # [B, 128, 1]
        

        # 独立编码并融合 (Broadcasting add)
        h = self.phys_mlp(phys_feat) + self.time_mlp(time_idx) # [B, 128, 256]
        h = self.fusion(h)
        
        # Self-Attention
        feat_128 = self.transformer(h) # [B, 128, 256]
        
        # Cross-Attention 压缩
        B = pc.shape[0]
        latents = self.latents.repeat(B, 1, 1) # [B, 64, 256]
        
        # Query=Latents, Key/Value=feat_128
        z_compressed, _ = self.compress_attn(query=latents, key=feat_128, value=feat_128)
        
        z_radar = self.norm_out(z_compressed) # [B, 64, 256]
        return z_radar

class DirectJointHead(nn.Module):
    def __init__(self, latent_dim=256, num_joints=27):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, num_joints * 3)
        )
        self.num_joints = num_joints

    def forward(self, z_global):
        B = z_global.shape[0]
        x = self.net(z_global).view(B, self.num_joints, 3)
        return x


        
class DecoupledRegressionRefiner(nn.Module):
    """
    保留原始架构思想：
    1. 显式的关节感应 Token
    2. 基于 Cross-Attention 的特征采样
    3. 物理隔绝的解耦回归头 (ModuleList)
    """
    def __init__(self, num_joints=27, embed_dim=256):
        super().__init__()
        self.num_joints = num_joints
        
        # 1. 关节感知 Token
        self.joint_tokens = nn.Parameter(torch.randn(1, num_joints, embed_dim))
        
        # 2. 空间位置嵌入 (用于将 coarse pose 的 (x,y,z) 映射到 embedding 空间)
        self.pos_embed = nn.Linear(3, embed_dim)
        
        # 3. Cross-Attention
        # 注意：embed_dim 必须与输入特征的最后一维一致
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        
        # 4. 解耦回归头：完全保留 ModuleList 结构，确保每个关节的权重独立
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 3)
            ) for _ in range(num_joints)
        ])

    def forward(self, x_coarse, z_radar_feat):
        """
        x_coarse: [BT, 27, 3]
        z_radar_feat: [BT, 64, 256]
        """
        B_T = x_coarse.shape[0]
        
        # --- 维度校准 ---
        # 你的 z_radar_feat 是 256 维，embed_dim 也是 256 维，这是对的。
        # 将 coarse pose 映射到 256 维
        x_feat = self.pos_embed(x_coarse)  # [BT, 27, 256]
        
        # --- 特征融合 (Memory) ---
        # 将 Coarse Pose 的位置特征和雷达特征拼接，作为 Cross-Attention 的“字典”
        # 维度：[BT, 27 + 64, 256] = [BT, 91, 256]
        memory = torch.cat([x_feat, z_radar_feat], dim=1) 
        
        # --- 采样特征 ---
        # Query: 预设的关节 Token [BT, 27, 256]
        # Key/Value: 融合后的 Memory [BT, 91, 256]
        tokens = self.joint_tokens.expand(B_T, -1, -1)
        refined_tokens, _ = self.cross_attn(
            query=tokens, 
            key=memory, 
            value=memory
        )
        
        # --- 解耦回归 (保留你的 ModuleList 循环逻辑) ---
        deltas = []
        for i in range(self.num_joints):
            # 提取第 i 个关节的特征进行独立回归
            joint_feat = refined_tokens[:, i, :] # [BT, 256]
            delta_i = self.reg_heads[i](joint_feat) # [BT, 3]
            deltas.append(delta_i)
            
        # 重新组装成 [BT, 27, 3]
        return torch.stack(deltas, dim=1)

class RadarPoseRefiner(nn.Module):
    def __init__(self, in_channels=6, radar_embed_dim=256, num_joints=27, parent_list=None):
        super().__init__()
        
        # --- Stage 1: Coarse (保持不变) ---
        # --- Stage 1 组件 ---
        self.encoder = TimeAwareCompressedRadarEncoder(
            in_channels=in_channels,
            embed_dim=radar_embed_dim,
            num_latents=64
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
        
        self.refiner = DecoupledRegressionRefiner(
            num_joints=num_joints, 
            embed_dim=256
        )
        self.parent_dict = PARENT
        self.end_effectors = [9, 10, 16, 17, 21, 24, 25, 26]

    def forward(self, radar_input):
        """
        完全的一步法回归推理
        """
        # 1. Stage 1
        B, T = radar_input.shape[:2]
        radar_flat = radar_input.view(B * T, 128, 6)
        z_seq = self.encoder(radar_flat)
        z_seq_flat = self.temporal_adapter(z_seq, B, T)
        z_global = z_seq_flat.mean(dim=1)
        # x_coarse = self.direct_head(z_global)
        x_coarse = torch.zeros(B*T, 27, 3, device=radar_input.device)# [BT,256]

        # 2. Stage 2: 纯回归
        # 将 Stage 1 的特征 z_seq_flat 作为 context 传给 refiner
        delta = self.refiner(x_coarse, z_seq_flat)
        
        x_final = x_coarse + delta
        return x_final.view(B, T, 27, 3)

    def compute_loss(self, radar_input, x_gt):
        """
        损失函数现在只关心一步回归的精度
        """
        B, T = radar_input.shape[:2]
        x_gt_flat = x_gt.reshape(B * T, 27, 3)

        # 得到输出
        x_pred_full = self.forward(radar_input)
        x_pred = x_pred_full.view(B * T, 27, 3)

        # 基础坐标回归 Loss
        loss_pose = F.mse_loss(x_pred, x_gt_flat) * 100.0
        
        # 骨架几何约束 (确保回归出来的骨架不畸形)
        gt_lens, _ = self.calc_bone_lengths_and_dirs(x_gt_flat)
        pred_lens, _ = self.calc_bone_lengths_and_dirs(x_pred)
        loss_geom = F.mse_loss(pred_lens, gt_lens) * 100.0

        return loss_pose + loss_geom

    @torch.no_grad()
    def inference(self, radar_input):
        self.eval()
        return self.forward(radar_input) # 没有 steps 概念，只有 forward

    def calc_bone_lengths_and_dirs(self, x):
        children = sorted(self.parent_dict.keys())
        parents = [self.parent_dict[c] for c in children]
        diff = x[:, children, :] - x[:, parents, :]
        return torch.norm(diff, dim=-1), F.normalize(diff, dim=-1)

    def load_pretrained_stage1(self, ckpt_path, freeze=True):
        """加载 Stage 1 权重并选择性冻结"""
        if not os.path.exists(ckpt_path):
            print(f"Warning: Stage 1 checkpoint not found at {ckpt_path}")
            return
            
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # 如果是之前保存的完整 state_dict，需要按需加载
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # 加载匹配的权重
        self.load_state_dict(state_dict, strict=False)
        print(f"✅ Successfully loaded Stage 1 weights from {ckpt_path}")

        if freeze:
            for name, param in self.named_parameters():
                if "refiner" not in name:
                    param.requires_grad = False
            print("❄️ Stage 1 components are frozen.")

# ==========================================
# 3. 工具函数
# ==========================================

def calculate_mpjpe(pred, gt):
    # pred, gt: [B, 27, 3]
    return torch.norm(pred - gt, dim=-1).mean().item()


import os
import random
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
from matplotlib import animation

# 依赖你已有的函数：
# compute_mpjpe, compute_pampjpe, compute_pck, compute_auc_pck, compute_mpjve
# draw_skeleton_3d, plot_skeleton
#
# 以及你新增的：
# compute_spatial_structure_corr
# compute_bone_length_mae, compute_bone_length_var
# compute_mjae, compute_jerk_energy
#
# 注意：这些函数都应支持 CUDA tensor（除了可视化函数）


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

        # -----------------------
        # 1) Valid mask from GT
        # -----------------------
        valid_mask = (skeleton_seq.abs().sum(dim=(2, 3)) > 1e-6)  # [B,T] bool
        if not valid_mask.any():
            continue

        # -----------------------
        # 2) Inference
        # -----------------------
        pred = model.inference(radar_seq)  # [B,T,J,3] meters

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
    save_path = "/code/radarFMv2/checkpoints_baseline13newdata"
    # save_path = "./checkpoints_refine"
    os.makedirs(save_path, exist_ok=True)
    
    dataset = RadarDiffusionDataset(
        root_dir='/code/radarFMv2/dataset', 
        # root_dir='./dataset',
        sample_level='sequence', 
        num_joints=27
    )
    
    # 单机运行，直接使用常规 DataLoader
    train_loader = DataLoader(
        dataset.get_train_set(),
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn_for_cross_modal,
        num_workers=0,        # ⚠️ 必须改为 0
        pin_memory=False,     # ⚠️ 报错时建议设为 False
        persistent_workers=False
    )
    
    # 修改 val_loader
    val_loader = DataLoader(
        dataset.get_val_set(),
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn_for_cross_modal,
        num_workers=0,        # ⚠️ 必须改为 0
        pin_memory=False,     # ⚠️ 必须改为 False
        persistent_workers=False
    )

    # --- 3. 模型实例化与权重加载 ---
    model = RadarPoseRefiner(
        in_channels=6,
        radar_embed_dim=256,
        num_joints=27,
        parent_list=PARENT
    ).to(device)

    # 加载你那个 98mm 的 Stage 1 权重

    # ckpt_stage1 = "/code/radarFMv2/checkpoints_baseline2/best_model.pt"
    # model.load_pretrained_stage1(ckpt_stage1, freeze=True)

    ckpt_path = "/code/radarFMv2/checkpoints_baseline13newdata/best_refiner.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    steps_list = [1]

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
    
        print(f"\n===== Eval steps = {s} =====")
        for k, v in metrics.items():
            if isinstance(v, float) and not (isinstance(v, float) and math.isnan(v)):
                # pck 这类本来是 0~1，这里也照样打印；你想打印百分号我也可以改
                print(f"{k}: {v:.4f}")
            elif isinstance(v, float) and math.isnan(v):
                print(f"{k}: nan")
            else:
                print(f"{k}: {v}")



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
    
    # for epoch in range(1, 251):
    #     model.train()
    #     total_epoch_loss = 0.0
        
    #     pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    #     for radar_seq, skeleton_seq in pbar:
    #         radar_seq = radar_seq.to(device)
    #         skeleton_seq = skeleton_seq.to(device)

    #         optimizer.zero_grad(set_to_none=True)

    #         # 使用混合精度训练
    #         with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    #             # 调用模型内置的 FM Loss 计算函数
         
    #             loss = model.compute_loss(radar_seq, skeleton_seq)

    #         scaler.scale(loss).backward()
    #         scaler.unscale_(optimizer)
    #         # 梯度裁剪防止 Refiner 训练早期震荡
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         scaler.step(optimizer)
    #         scaler.update()
            
    #         total_epoch_loss += loss.item()
    #         pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    #     # --- 6. 验证与保存 ---
    #     if epoch % 2 == 0 or epoch == 1:
    #         # 调用你优化后的 evaluate_sequence 函数
    #         metrics = evaluate_sequence(
    #             dataloader=val_loader,
    #             model=model,
    #             device=device,
    #             vis_edges=EDGES,
    #             num_vis_samples=1,
    #             steps=1  # 验证时使用 10 步 ODE 积分
    #          )
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

# def main():
#     # --- 1. 设备与环境配置 ---
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
    
#     # 性能优化：启用算力加速
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#     # 启用原生算子加速
#     torch.backends.cuda.enable_flash_sdp(True)
#     torch.backends.cuda.enable_mem_efficient_sdp(True)
    
#     # 设置保存路径
#     save_path = None
#     # save_path = "./checkpoints_refine"
#     # os.makedirs(save_path, exist_ok=True)
    
#     dataset = RadarDiffusionDataset(
#         root_dir='/code/radarFMv2/dataset', 
#         # root_dir='./dataset',
#         sample_level='sequence', 
#         num_joints=27
#     )
    
    # 单机运行，直接使用常规 DataLoader
    # train_loader = DataLoader(
    #     dataset.get_train_set(),
    #     batch_size=16,
    #     shuffle=True,
    #     collate_fn=collate_fn_for_cross_modal,
    #     num_workers=0,        # ⚠️ 必须改为 0
    #     pin_memory=False,     # ⚠️ 报错时建议设为 False
    #     persistent_workers=False
    # )
    
    # 修改 val_loader
    # val_loader = DataLoader(
    #     dataset.get_val_set(),
    #     batch_size=16,
    #     shuffle=False,
    #     collate_fn=collate_fn_for_cross_modal,
    #     num_workers=0,        # ⚠️ 必须改为 0
    #     pin_memory=False,     # ⚠️ 必须改为 False
    #     persistent_workers=False
    # )

    # --- 3. 模型实例化与权重加载 ---
    # model = RadarPoseRefiner(
    #     in_channels=6,
    #     radar_embed_dim=256,
    #     num_joints=27,
    #     parent_list=PARENT
    # ).to(device)

    # # # 加载你那个 98mm 的 Stage 1 权重
    # # # freeze=True 意味着我们只训练 Refiner，保持 Stage 1 不动
    # # ckpt_stage1 = "/code/radarFMv2/checkpoints_stage1olduseful/best_model.pt"
    # # model.load_pretrained_stage1(ckpt_stage1, freeze=True)

    # ckpt_path = "/code/radarFMv2/checkpoints_baseline6/best_refiner.pt"
    # # # ckpt_path = "./checkpoints_refine/best_refiner.pt"
    # checkpoint = torch.load(ckpt_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # test_box_sizes = [(2.0, 1.0, 2.0)]

    # for size in test_box_sizes:
    #     # 1. 动态生成验证集
    #     v_set = dataset.get_val_set(box_size=size, occl_prob=1.0, drop_ratio=0.8, box_center=(0.0,-0.4,0.0))
        
    #     # 2. 重新绑定 Loader
    #     val_loader = DataLoader(v_set, batch_size=16, shuffle=False, collate_fn=collate_fn_for_cross_modal)
        
    #     steps_list = [1]
    
    #     all_metrics = {}
    #     print(f"\n======box_size={size}")
    #     for s in steps_list:
    #         # current_vis_dir = os.path.join(save_path, f"vis_s{s}") if s == 1 else None
    #         current_vis_dir = None
    #         metrics = evaluate_sequence(
    #             dataloader=val_loader,
    #             model=model,
    #             device=device,
    #             vis_dir=current_vis_dir,  # 只在第一档可视化，避免生成太多文件
    #             vis_edges=EDGES,
    #             num_vis_samples=5,
    #             steps=s
    #         )
    #         all_metrics[s] = metrics
        
    #         print(f"\n===== Eval steps = {s} =====")
    #         for k, v in metrics.items():
    #             if isinstance(v, float) and not (isinstance(v, float) and math.isnan(v)):
    #                 # pck 这类本来是 0~1，这里也照样打印；你想打印百分号我也可以改
    #                 print(f"{k}: {v:.4f}")
    #             elif isinstance(v, float) and math.isnan(v):
    #                 print(f"{k}: nan")
    #             else:
    #                 print(f"{k}: {v}")
