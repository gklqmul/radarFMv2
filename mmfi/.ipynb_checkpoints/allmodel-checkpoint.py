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

class CoarseSkeletonHead(nn.Module):
    """
    Stage 1: 从全局雷达特征生成初始骨架 x0
    Input: [B, radar_embed_dim]
    Output: [B, 17, 3]
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

        return joints, offsets, length # Returns x_coarse [B, 17, 3]

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
    def __init__(self, num_joints=17, radar_in_channels=6, embed_dim=512, local_radius=0.1, parent_list=PARENT_LIST):
        super().__init__()
        self.num_joints = num_joints
        self.base_radius = local_radius 
        self.parent_list = parent_list
        
        # 1. 特征投影
        self.pc_proj = nn.Linear(radar_in_channels, embed_dim) 
        self.joint_embed = nn.Linear(3, embed_dim)
        self.coarse_embed = nn.Linear(3, embed_dim)
        self.diff_embed = nn.Linear(3, embed_dim) 
        self.joint_id_embed = nn.Embedding(num_joints, embed_dim)
        
        self.doppler_head = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, embed_dim)
        )
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.diffusion_proj = nn.Linear(embed_dim * 2, embed_dim)
        
        # Transformer 层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=2048, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # 注册 Buffer
        if parent_list is not None:
            self.register_buffer("topo_mask", GFM_skeleton_mask(parent_list, num_joints))
            self.register_buffer("p_idx", torch.tensor(parent_list).long())
        
        # 自适应半径系数
        scales = torch.ones(num_joints)
        core_joints = [0, 1, 4, 7,8,9] 
        scales[core_joints] = 0.6
        self.end_effectors = [16,13,3,6]
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
        
        # 计算高斯权重 [B, J, 128]
        density_weight = torch.exp(-(dist**2) / (2 * adaptive_sigma**2))

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
                 num_joints=17, parent_list=None, refine_embed_dim=512):
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
        self.end_effectors = [3,6,13,16]

    def get_coarse_prior(self, radar_input):
        """
        输入: radar_input [B, T, 128, 6]
        输出: 
            z_seq_flat: 逐帧的特征序列 [B*T, 64, 256] (用于 Refiner 的 Memory)
            x_coarse: 初始粗糙骨架 [B*T, 17, 3] (作为 Flow 的起点 x0)
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
        x_gt_flat = x_gt.reshape(B * T, 17, 3)

        # 1. 获取 Stage 1 的输出作为先验 (通常在 Refine 训练时 detach)
        with torch.no_grad():
            _, x_coarse = self.get_coarse_prior(radar_input)
            x_coarse = x_coarse.detach()
            noise_level = 0.03 # 3cm 噪声
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
        loss_unit = F.mse_loss(v_pred, v_target, reduction='none') # [BT, 17, 3]
        weights = torch.ones(17, device=device)
        weights[self.end_effectors] = 3.0 # 末端点贡献 3 倍 Loss

        time_weights = 1.0 + t.view(-1, 1, 1) * 2.0
        
        loss_velocity = (loss_unit * time_weights).mean(dim=-1) # [BT, 17]
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
        return x_curr.view(B, T, 17, 3)

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