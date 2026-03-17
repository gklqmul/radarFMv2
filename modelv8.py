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

def GFM_skeleton_mask(parent_list, num_joints=27):
    """构造骨架拓扑掩码：0.0 表示允许关注，-inf 表示禁止关注"""
    mask = torch.full((num_joints, num_joints), float('-inf'))
    for child, parent in enumerate(parent_list):
        mask[child, child] = 0.0
        if child != parent:
            mask[child, parent] = 0.0
            mask[parent, child] = 0.0
    return mask

class SingleFrameFlowTransformer(nn.Module):
    def __init__(self, num_joints=27, radar_in_channels=6, embed_dim=512, local_radius=0.1, parent_list=PARENT_LIST):
        super().__init__()
        self.num_joints = num_joints
        self.base_radius = local_radius 
        
        # 1. 特征投影
        # 输入 6 维 (XYZ, Doppler, Power, Range)
        self.pc_proj = nn.Linear(radar_in_channels, embed_dim) 
        self.joint_embed = nn.Linear(3, embed_dim)
        self.coarse_embed = nn.Linear(3, embed_dim)
        self.diff_embed = nn.Linear(3, embed_dim) 
        self.joint_id_embed = nn.Embedding(num_joints, embed_dim)
        
        # 【新增】：Doppler 径向对齐层。用来学习点云多普勒与关节运动趋势的匹配度
        self.doppler_head = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, embed_dim)
        )

        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        
        self.diffusion_proj = nn.Linear(embed_dim * 2, embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=2048, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        if parent_list is not None:
            # 确保 GFM_skeleton_mask 函数已定义
            self.register_buffer("topo_mask", GFM_skeleton_mask(parent_list, num_joints))
            self.register_buffer("p_idx", torch.tensor(parent_list).long())
        
        scales = torch.ones(num_joints)
        core_joints = [0, 1, 2, 3, 18, 22] 
        scales[core_joints] = 0.6
        self.end_effectors = [9, 10, 16, 17, 20, 21, 24, 25, 26]
        scales[self.end_effectors] = 1.5
        self.register_buffer("radius_scales", scales.view(1, num_joints, 1))

        self.vel_head = nn.Linear(embed_dim, 3)

    def forward(self, x_t, tau, x_coarse, pc_raw):
        B, J, _ = x_t.shape
        device = x_t.device

        # --- 1. Memory 处理：特征调制 ---
        # pc_raw: [B, 128, 6], memory: [B, 128, 512]
        pc_xyz = pc_raw[:, :, :3]
        pc_doppler = pc_raw[:, :, 3:4] # 提取 Doppler [B, 128, 1]
        
        raw_memory = self.pc_proj(pc_raw) 
        
        # --- 2. 局部概率场计算 (Density Weight) ---
        dist = torch.cdist(x_t, pc_xyz, p=2) # [B, J, 128]
        adaptive_radius = self.radius_scales * self.base_radius 
        adaptive_sigma = adaptive_radius / 2.0
        
        # 高斯概率权重 [B, J, 128]
        density_weight = torch.exp(-(dist**2) / (2 * adaptive_sigma**2))
        
        # 为了不引起内存爆炸，我们取每个点相对于所有关节的最大“被关注度”
        # 这样 memory 依然维持 [B, 128, 512]，但关键点的特征会被显著放大
        point_importance, _ = density_weight.max(dim=1, keepdim=True) # [B, 1, 128]
        memory = raw_memory * point_importance.transpose(1, 2) 

        # --- 3. Doppler 物理一致性注入 ---
        # 学习点云自带的 Doppler 速度特征，作为全局运动补充
        h_doppler = self.doppler_head(pc_doppler).mean(dim=1, keepdim=True) # [B, 1, 512]

        # --- 4. 准备 Query ---
        h_xt = self.joint_embed(x_t)
        h_tau = self.time_embed(tau).unsqueeze(1)
        h_coarse = self.coarse_embed(x_coarse)
        h_diff = self.diff_embed(x_t - x_coarse)
        h_id = self.joint_id_embed(torch.arange(J, device=device)).unsqueeze(0)

        h_parent = h_xt[:, self.p_idx, :]
        h_struct = self.diffusion_proj(torch.cat([h_xt, h_parent], dim=-1))

        # 将 Doppler 特征注入每一个 Query 节点
        query = h_struct + h_tau + h_coarse + h_diff + h_id + h_doppler

        # --- 5. 空间 Mask (硬约束) ---
        spatial_mask = dist > adaptive_radius
        min_idx = dist.argmin(dim=-1, keepdim=True)
        rescue = ~torch.zeros_like(spatial_mask).scatter_(-1, min_idx, 1).bool()
        spatial_mask = torch.where(spatial_mask.all(dim=-1, keepdim=True), rescue, spatial_mask)
        
        nhead = 8
        mem_mask = spatial_mask.repeat_interleave(nhead, dim=0)

        # --- 6. Transformer 交互 ---
        refined_feat = self.transformer(
            tgt=query, 
            memory=memory, 
            tgt_mask=self.topo_mask,
            memory_mask=mem_mask
        )
        
        return self.vel_head(refined_feat)

class RadarPoseRefiner(nn.Module):
    """
    完整的二阶段模型：
    Stage 1: 生成粗糙骨架 (Coarse Skeleton)
    Stage 2: 基于 Flow Matching 进行精细化修正 (Refinement)
    """
    def __init__(self, in_channels=6, radar_embed_dim=256, num_latents=64, 
                 num_joints=27, parent_list=None, refine_embed_dim=512):
        super().__init__()
        
        # --- Stage 1 组件 ---
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
        # --- Stage 2 组件 (Flow Matching Refiner) ---
        self.refiner = SingleFrameFlowTransformer(
            num_joints=num_joints,
            radar_in_channels=6,
            embed_dim=refine_embed_dim
        )
        self.parent_dict = PARENT
        self.parent_list = PARENT_LIST
        # 定义末端节点 (Hands, Feet, Head) 用于加权
        self.end_effectors = [9, 10, 16, 17, 21, 24, 25, 26]

    def get_coarse_prior(self, radar_input):
        """
        输入: radar_input [B, T, 128, 6]
        输出: 
            z_seq_flat: 逐帧的特征序列 [B*T, 64, 256] (用于 Refiner 的 Memory)
            x_coarse: 初始粗糙骨架 [B*T, 27, 3] (作为 Flow 的起点 x0)
        """
        B, T, N, C = radar_input.shape
        radar_flat = radar_input.view(B * T, N, C) 
        
        # 1. 提取雷达序列特征
        z_seq = self.encoder(radar_flat)          # [B*T, 64, 256]
        z_seq_flat = self.temporal_adapter(z_seq, B, T)
        
        # 2. 得到全局特征并预测粗糙骨架
        z_global = z_seq_flat.mean(dim=1)         # [B*T, 256]
        x_coarse, _, _ = self.coarse_head(z_global)
        
        return z_seq_flat, x_coarse

    def forward(self, x_t, t, radar_input):
        """
        Flow Matching 预测速度向量场 v_t
        用于计算 Training Loss 或 ODE 积分步
        """
        # 提取 Stage 1 的特征和起点作为条件
        z_seq, x_coarse = self.get_coarse_prior(radar_input)

        radar_raw = radar_input.flatten(0, 1)
        # 预测速度向量 v_t
        v_pred = self.refiner(x_t, t, x_coarse, radar_raw)
        return v_pred

    def compute_fm_loss(self, radar_input, x_gt):
        """
        核心训练逻辑：Conditional Flow Matching Loss
        """
        B, T = radar_input.shape[:2]
        device = radar_input.device
        x_gt_flat = x_gt.reshape(B * T, 27, 3)

        # 1. 获取 Stage 1 的输出作为先验 (通常在 Refine 训练时 detach)
        with torch.no_grad():
            _, x_coarse = self.get_coarse_prior(radar_input)
            x_coarse = x_coarse.detach()
            noise_level = 0.01 # 3cm 噪声
            x_coarse_perturbed = x_coarse + torch.randn_like(x_coarse) * noise_level

        # 2. 采样时间步 t 并在 [x_coarse, x_gt] 之间插值
        t = torch.rand(B * T, 1, device=device) # [B*T, 1]
        t_broad = t.view(-1, 1, 1)

        # x_t = (1-t)*x_0 + t*x_1
        x_t = (1 - t_broad) * x_coarse_perturbed + t_broad * x_gt_flat
        v_target = x_gt_flat - x_coarse_perturbed 

        # 3. 模型预测
        radar_raw = radar_input.flatten(0, 1)
        v_pred = self.refiner(x_t, t, x_coarse, radar_raw)

        # 4. 基础 Flow Loss
        loss_unit = F.mse_loss(v_pred, v_target, reduction='none') # [BT, 27, 3]
        weights = torch.ones(27, device=device)
        weights[self.end_effectors] = 3.0 # 末端点贡献 3 倍 Loss

        time_weights = 1.0 + t.view(-1, 1, 1) * 2.0
        
        loss_velocity = (loss_unit * time_weights).mean(dim=-1) # [BT, 27]
        loss_velocity = (loss_velocity * weights).mean() * 100.0

        # 5. 辅助几何 Loss (Refine 后的骨骼长度一致性)
        x_refined_final = x_coarse_perturbed + v_pred
        gt_lens, gt_dirs = self.calc_bone_lengths_and_dirs(x_gt_flat, self.parent_dict)
        pred_lens, pred_dirs = self.calc_bone_lengths_and_dirs(x_refined_final, self.parent_dict)
        
        loss_geom = F.mse_loss(pred_lens, gt_lens) * 100.0 # 保持量级一致
        loss_dir = F.mse_loss(pred_dirs, gt_dirs) * 10.0

        # 总 Loss
        return loss_velocity + (loss_geom + 0.1 * loss_dir)

    def calc_bone_lengths_and_dirs(self, x, parent_dict):
        """辅助计算骨骼特征用于几何约束"""
        # 提取子节点和对应的父节点
        children = sorted(parent_dict.keys())
        parents = [parent_dict[c] for c in children]
        
        child_pos = x[:, children, :]
        parent_pos = x[:, parents, :]
        diff = child_pos - parent_pos
        
        lengths = torch.norm(diff, dim=-1)
        directions = F.normalize(diff, dim=-1, eps=1e-6)
        return lengths, directions

    @torch.no_grad()
    def inference(self, radar_input, steps=10):
        """
        基于 ODE 积分的推理过程
        """
        self.eval()
        B, T = radar_input.shape[:2]
        device = radar_input.device
        
        # 1. 获取起点
        z_seq, x_coarse = self.get_coarse_prior(radar_input)
        
        # 2. 迭代修正
        x_curr = x_coarse.clone()
        dt = 1.0 / steps
        
        for i in range(steps):
            t_val = i / steps
            t_tensor = torch.full((B * T, 1), t_val, device=device).float()
            # 预测当前位置的速度
            radar_raw = radar_input.flatten(0, 1)
            # 预测速度向量 v_t
            v_pred = self.refiner(x_curr, t_tensor, x_coarse, radar_raw)
            
            # Euler 积分步
            x_curr = x_curr + v_pred * dt
        return x_curr.view(B, T, 27, 3)

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
    针对 Flow Matching 优化的序列评估函数 - 保留 GIF 生成功能
    + 新增指标：SSC / bone_mae / bone_var / MJAE / jerk
    """
    model.eval()

    # -----------------------
    # 累计变量（保留原有 + 新增）
    # -----------------------
    total_mpjpe, total_pampjpe, total_auc = 0.0, 0.0, 0.0
    total_pck = {th: 0.0 for th in pck_thresholds}
    total_mpjve, total_vel_frames, total_samples = 0.0, 0, 0

    total_ssc = 0.0
    total_bone_mae = 0.0

    total_bone_var_pred = 0.0
    total_bone_var_gt = 0.0
    total_bone_var_count = 0

    total_mjae = 0.0
    total_acc_frames = 0

    total_jerk_pred = 0.0
    total_jerk_gt = 0.0
    total_jerk_count = 0

    # -----------------------
    # 可视化目录（保留原有）
    # -----------------------
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
    has_visualized = 0

    # -----------------------
    # 预先把 edges 转成 GPU index（性能优化，不改变功能）
    # -----------------------
    edge_i = edge_j = None
    if vis_edges is not None and len(vis_edges) > 0:
        edge_i = torch.tensor([e[0] for e in vis_edges], device=device, dtype=torch.long)
        edge_j = torch.tensor([e[1] for e in vis_edges], device=device, dtype=torch.long)

    pbar = tqdm(dataloader, desc=f"Eval (Steps={steps})")

    for batch_idx, (radar_seq, skeleton_seq) in enumerate(pbar):
        radar_seq = radar_seq.to(device).float()
        skeleton_seq = skeleton_seq.to(device).float()
        B, T, N, C = radar_seq.shape

        # 1) 识别并提取有效帧掩码（GPU）
        is_valid_mask = (skeleton_seq.abs().sum(dim=(2, 3)) > 1e-6)  # (B,T) bool

        # 2) 模型推理（GPU）
        pred = model.inference(radar_seq)  # (B,T,J,3)

        # 3) 转换 mm 并 Root-Relative（GPU）
        pred_mm = pred * 1000.0
        gt_mm = skeleton_seq * 1000.0
        pred_rel = pred_mm - pred_mm[:, :, 0:1, :]
        gt_rel = gt_mm - gt_mm[:, :, 0:1, :]

        # 4) 指标计算（GPU）
        mask = is_valid_mask  # (B,T) bool on GPU

        if mask.any():
            # 展平成有效帧集合：(M,J,3)
            v_pred = pred_rel[mask]
            v_gt = gt_rel[mask]
            M = v_gt.shape[0]

            # ---- 原有指标：MPJPE / PA-MPJPE / PCK / AUC ----
            batch_mpjpe = compute_mpjpe(v_pred, v_gt).item()
            batch_pampjpe = compute_pampjpe(v_pred, v_gt).item()
            batch_pck = {th: compute_pck(v_pred, v_gt, th).item() for th in pck_thresholds}
            batch_auc = compute_auc_pck(v_pred, v_gt, max_threshold=auc_max_threshold, step=auc_step).item()

            total_mpjpe += batch_mpjpe * M
            total_pampjpe += batch_pampjpe * M
            total_auc += batch_auc * M
            for th in pck_thresholds:
                total_pck[th] += batch_pck[th] * M
            total_samples += M

            # ---- 原有指标：MPJVE（速度误差）----
            vel_mask = mask[:, 1:] & mask[:, :-1]  # (B,T-1)
            if vel_mask.any():
                ve = compute_mpjve(pred_rel, gt_rel)  # 期望输出 (B,T-1)
                v_ve = ve[vel_mask]
                total_mpjve += v_ve.mean().item() * v_ve.numel()
                total_vel_frames += v_ve.numel()

            # ---- 新增：SSC（空间结构相关性）----
            batch_ssc = compute_spatial_structure_corr(v_pred, v_gt).item()
            total_ssc += batch_ssc * M

            # ---- 新增：bone length MAE（需要 edges）----
            if edge_i is not None:
                # 直接用预计算 index（更快）
                d_pred = torch.norm(v_pred[:, edge_i] - v_pred[:, edge_j], dim=-1)  # (M,E)
                d_gt = torch.norm(v_gt[:, edge_i] - v_gt[:, edge_j], dim=-1)        # (M,E)
                batch_bone_mae = (d_pred - d_gt).abs().mean().item()
                total_bone_mae += batch_bone_mae * M

            # ---- 新增：MJAE（加速度误差）----
            # compute_mjae 输出 (B,T-2)
            acc_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
            if acc_mask.any():
                ae = compute_mjae(pred_rel, gt_rel)  # (B,T-2)
                v_ae = ae[acc_mask]
                total_mjae += v_ae.mean().item() * v_ae.numel()
                total_acc_frames += v_ae.numel()

            # ---- 新增：jerk energy（预测/GT 各一个 scalar）----
            jerk_mask = mask[:, 3:] & mask[:, 2:-1] & mask[:, 1:-2] & mask[:, :-3]
            if jerk_mask.any():
                total_jerk_pred += compute_jerk_energy(pred_rel).item()
                total_jerk_gt += compute_jerk_energy(gt_rel).item()
                total_jerk_count += 1

            # ---- 新增：bone length var（预测/GT 各一个 scalar）----
            if edge_i is not None:
                # 更严谨：只用有效帧来算 var，避免无效帧(全0)干扰
                x_pred_valid = pred_rel[mask]  # (M,J,3)
                x_gt_valid = gt_rel[mask]
                if x_pred_valid.shape[0] > 1:
                    bone_var_pred = compute_bone_length_var(x_pred_valid, vis_edges).item()
                    bone_var_gt = compute_bone_length_var(x_gt_valid, vis_edges).item()
                    total_bone_var_pred += bone_var_pred
                    total_bone_var_gt += bone_var_gt
                    total_bone_var_count += 1

            # --- 5) 恢复的可视化逻辑 (GIF & HTML) ---（必须 CPU/NumPy）
            if vis_dir and has_visualized < num_vis_samples and vis_edges:
                # 找到当前 Batch 中有有效数据的样本索引
                valid_bs = [i for i in range(B) if mask[i].any().item()]

                if valid_bs:
                    # 1) 随机锁定一个样本 b
                    b = random.choice(valid_bs)
                    # 获取该样本下所有有效的帧索引 (全长)
                    valid_indices = torch.where(mask[b])[0]  # (T_valid,)

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

        # 进度条（保留原有 mpjpe 显示；我没删）
        pbar.set_postfix({"mpjpe": f"{total_mpjpe/max(total_samples, 1):.2f}mm"})

    # -----------------------
    # 汇总（保留原有输出键 + 新增）
    # -----------------------
    denom = max(total_samples, 1)
    final = {
        "mpjpe": total_mpjpe / denom,
        "pa_mpjpe": total_pampjpe / denom,
        "auc_pck": total_auc / denom,
        "avg_ssc": total_ssc / denom,  # 保持你原来的 key 名
    }

    for th in pck_thresholds:
        final[f"pck@{int(th)}"] = total_pck[th] / denom

    if total_vel_frames > 0:
        final["mpjve"] = total_mpjve / total_vel_frames

    # 新增指标输出
    if total_acc_frames > 0:
        final["mjae"] = total_mjae / total_acc_frames

    if edge_i is not None:
        final["bone_mae"] = total_bone_mae / denom
        if total_bone_var_count > 0:
            final["bone_var_pred"] = total_bone_var_pred / total_bone_var_count
            final["bone_var_gt"] = total_bone_var_gt / total_bone_var_count

    if total_jerk_count > 0:
        final["jerk_pred"] = total_jerk_pred / total_jerk_count
        final["jerk_gt"] = total_jerk_gt / total_jerk_count

    return final


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
    save_path = "/code/radarFMv2/checkpoints_basemodel"
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
        num_latents=64,
        num_joints=27,
        parent_list=PARENT,
        refine_embed_dim=512
    ).to(device)

    # 加载你那个 98mm 的 Stage 1 权重
    # freeze=True 意味着我们只训练 Refiner，保持 Stage 1 不动
    ckpt_stage1 = "/code/radarFMv2/checkpoints_stage1v2/best_model.pt"
    model.load_pretrained_stage1(ckpt_stage1, freeze=True)

    # ckpt_path = "/code/radarFMv2/checkpoints_baseline6/best_refiner.pt"
    # # ckpt_path = "./checkpoints_refine/best_refiner.pt"
    # checkpoint = torch.load(ckpt_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # steps_list = [1, 2, 5, 10, 15, 20, 25]

    # all_metrics = {}
    
    # for s in steps_list:
    #     current_vis_dir = os.path.join(save_path, f"vis_s{s}") if s == 1 else None
    #     metrics = evaluate_sequence(
    #         dataloader=val_loader,
    #         model=model,
    #         device=device,
    #         vis_dir=current_vis_dir,  # 只在第一档可视化，避免生成太多文件
    #         vis_edges=EDGES,
    #         num_vis_samples=5,
    #         steps=s
    #     )
    #     all_metrics[s] = metrics
    
    #     print(f"\n===== Eval steps = {s} =====")
    #     for k, v in metrics.items():
    #         if isinstance(v, float) and not (isinstance(v, float) and math.isnan(v)):
    #             # pck 这类本来是 0~1，这里也照样打印；你想打印百分号我也可以改
    #             print(f"{k}: {v:.4f}")
    #         elif isinstance(v, float) and math.isnan(v):
    #             print(f"{k}: nan")
    #         else:
    #             print(f"{k}: {v}")

    # # 额外：打印“相对变化”（和基准 steps 做对比）
    # base_steps = steps_list[0]
    # base = all_metrics[base_steps]
    
    # print(f"\n===== Delta vs steps = {base_steps} =====")
    # for s in steps_list[1:]:
    #     cur = all_metrics[s]
    #     print(f"\n--- steps {s} - steps {base_steps} ---")
    #     for k in base.keys():
    #         if isinstance(base[k], float) and isinstance(cur.get(k, None), float):
    #             if math.isnan(base[k]) or math.isnan(cur[k]):
    #                 print(f"{k}: nan")
    #             else:
    #                 print(f"{k}: {cur[k] - base[k]:+.4f}")

    # # --- 4. 优化器配置 ---
    # # 仅更新需要梯度的参数（即 Refiner 部分）
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-2
    )
    
    # 学习率调度：验证集 MPJPE 停滞时减半
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # 启用 AMP 混合精度训练
    scaler = torch.amp.GradScaler('cuda')

    # --- 5. 训练循环 ---
    best_mpjpe = float("inf")
    
    for epoch in range(1, 251):
        model.train()
        total_epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for radar_seq, skeleton_seq in pbar:
            radar_seq = radar_seq.to(device)
            skeleton_seq = skeleton_seq.to(device)

            optimizer.zero_grad(set_to_none=True)

            # 使用混合精度训练
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # 调用模型内置的 FM Loss 计算函数
                loss = model.compute_fm_loss(radar_seq, skeleton_seq)
           
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # 梯度裁剪防止 Refiner 训练早期震荡
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- 6. 验证与保存 ---
        if epoch % 2 == 0 or epoch == 1:
            # 调用你优化后的 evaluate_sequence 函数
            metrics = evaluate_sequence(
                dataloader=val_loader,
                model=model,
                device=device,
                vis_edges=EDGES,
                num_vis_samples=1,
                steps=3  # 验证时使用 10 步 ODE 积分
             )
            val_mpjpe = metrics["mpjpe"]
            print(f"\n[Epoch {epoch}] Train Loss: {total_epoch_loss/len(train_loader):.6f} | Val MPJPE: {val_mpjpe:.2f}mm")
            
            scheduler.step(val_mpjpe)

            if val_mpjpe < best_mpjpe:
                best_mpjpe = val_mpjpe
                save_file = f"{save_path}/best_refiner.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mpjpe': best_mpjpe,
                }, save_file)
                print(f"⭐ New Best Model Saved: {val_mpjpe:.2f}mm")

    print("Training Complete!")

if __name__ == "__main__":
    main()