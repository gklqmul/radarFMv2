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
from dataset import RadarDiffusionDataset
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
        time_feat = pc[..., 5:6]     # [B,128,1]

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

    def forward(self, z_flat, B, T, valid_mask=None):
        """
        z_flat: [B*T, 64, embed]
        valid_mask: [B,T] bool, True=真实帧
        return: [B*T, 64, embed]
        """
        z = z_flat.view(B, T, 64, -1)  # [B,T,64,embed]
        z = z.permute(0, 2, 1, 3).reshape(B * 64, T, -1)  # [B*64,T,embed]

        # TransformerEncoder 的 src_key_padding_mask: True 表示忽略
        if valid_mask is not None:
            key_padding = (~valid_mask).repeat_interleave(64, dim=0)  # [B*64,T]
        else:
            key_padding = None

        z = self.transformer(z, src_key_padding_mask=key_padding)     # [B*64,T,embed]
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

        self.refiner = SingleFrameFlowTransformer(
            num_joints=num_joints,
            radar_in_channels=in_channels,
            embed_dim=refine_embed_dim
        )

        self.parent_dict = PARENT
        self.parent_list = PARENT_LIST
        self.end_effectors = [9, 10, 16, 17, 21, 24, 25, 26]
        self.num_joints = num_joints

    def get_coarse_prior(self, radar_input, valid_mask=None):
        """
        radar_input: [B,T,128,6]
        valid_mask:  [B,T] bool (可选但推荐)
        return:
            z_seq_flat: [B*T,64,256]
            x_coarse:   [B*T,J,3]
        """
        B, T, N, C = radar_input.shape
        radar_flat = radar_input.view(B * T, N, C)

        z_seq = self.encoder(radar_flat)  # [BT,64,256]
        z_seq = self.temporal_adapter(z_seq, B, T, valid_mask=valid_mask)  # [BT,64,256]

        z_global = z_seq.mean(dim=1)      # [BT,256]
        x_coarse = self.direct_head(z_global)  # [BT,J,3]（和你现在一致）

        return z_seq, x_coarse

    def forward(self, x_t, t, radar_input, valid_mask=None):
        z_seq, x_coarse = self.get_coarse_prior(radar_input, valid_mask=valid_mask)
        radar_raw = radar_input.flatten(0, 1)  # [BT,128,6]
        return self.refiner(x_t, t, x_coarse, radar_raw)
        
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
@torch.no_grad()
def evaluate_sequence(
    dataloader, model, device,
    vis_dir=None, vis_edges=None, num_vis_samples=2,
    steps=10,
    pck_thresholds=(50.0, 100.0),
    auc_max_threshold=80.0,
    auc_step=5.0,
    do_k_sampling=False,
    k_samples=5,
    sample_noise_level=0.01,
    do_traj_metrics=False,
    traj_batches=2,
):
    model.eval()

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

    total_bok_mpjpe = 0.0
    total_diversity = 0.0
    total_k_bone_mae_mean = 0.0
    total_k_bone_mae_std = 0.0
    total_k_count = 0

    total_traj_jerk = 0.0
    total_traj_path = 0.0
    total_traj_count = 0

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
    has_visualized = 0

    edge_i = edge_j = None
    if vis_edges is not None and len(vis_edges) > 0:
        edge_i = torch.tensor([e[0] for e in vis_edges], device=device, dtype=torch.long)
        edge_j = torch.tensor([e[1] for e in vis_edges], device=device, dtype=torch.long)

    def compute_sample_diversity(samples_bt, valid_mask_bt=None):
        K, B, T, J, _ = samples_bt.shape
        x = samples_bt.reshape(K, B * T, J, 3)
        if valid_mask_bt is not None:
            m = valid_mask_bt.reshape(B * T)
            x = x[:, m]
        if x.shape[1] == 0:
            return torch.tensor(0.0, device=samples_bt.device)
        dsum, cnt = 0.0, 0
        for i in range(K):
            for j in range(i + 1, K):
                d = torch.norm(x[i] - x[j], dim=-1).mean(dim=-1)  # [M]
                dsum = dsum + d.mean()
                cnt += 1
        return dsum / max(cnt, 1)

    def compute_k_bone_mae_mean_std(samples_bt, gt_bt, valid_mask_bt=None):
        if edge_i is None:
            return None
        K, B, T, J, _ = samples_bt.shape
        x = samples_bt.reshape(K, B * T, J, 3)
        gt = gt_bt.reshape(B * T, J, 3)
        if valid_mask_bt is not None:
            m = valid_mask_bt.reshape(B * T)
            x = x[:, m]
            gt = gt[m]
        if gt.shape[0] == 0:
            return (torch.tensor(0.0, device=samples_bt.device),
                    torch.tensor(0.0, device=samples_bt.device))
        maes = []
        for k in range(K):
            d_pred = torch.norm(x[k][:, edge_i] - x[k][:, edge_j], dim=-1)
            d_gt = torch.norm(gt[:, edge_i] - gt[:, edge_j], dim=-1)
            maes.append((d_pred - d_gt).abs().mean())
        maes = torch.stack(maes)
        return maes.mean(), maes.std(unbiased=False)

    def compute_traj_jerk_energy(traj_steps):
        if traj_steps.shape[0] < 4:
            return torch.tensor(0.0, device=traj_steps.device)
        j = traj_steps[3:] - 3 * traj_steps[2:-1] + 3 * traj_steps[1:-2] - traj_steps[:-3]
        return (j ** 2).sum(dim=-1).mean()

    def compute_traj_path_length(traj_steps):
        v = traj_steps[1:] - traj_steps[:-1]
        step_len = torch.norm(v, dim=-1).mean(dim=-1)  # [S-1,B,T]
        return step_len.mean()

    pbar = tqdm(dataloader, desc=f"Eval (Steps={steps})")

    for batch_idx, batch in enumerate(pbar):
        # ✅ 现在要求 dataloader 返回 valid_mask
        if len(batch) == 3:
            radar_seq, skeleton_seq, valid_mask = batch
        else:
            # 兼容：如果你暂时没改 dataloader，就直接报错，避免悄悄算错
            raise ValueError("Dataloader must return (radar_seq, skeleton_seq, valid_mask).")

        radar_seq = radar_seq.to(device).float()
        skeleton_seq = skeleton_seq.to(device).float()
        valid_mask = valid_mask.to(device).bool()  # [B,T]
        B, T, N, C = radar_seq.shape

        # 2) 推理（建议把 valid_mask 也传进去，让模型内部忽略 padding）
        try:
            pred = model.inference(radar_seq, steps=steps)
        except TypeError:
            pred = model.inference(radar_seq, steps=steps)

        # 3) mm + root-relative
        pred_mm = pred * 1000.0
        gt_mm = skeleton_seq * 1000.0
        pred_rel = pred_mm - pred_mm[:, :, 0:1, :]
        gt_rel = gt_mm - gt_mm[:, :, 0:1, :]

        mask = valid_mask  # ✅ 统一口径：所有指标都用这个

        if mask.any():
            v_pred = pred_rel[mask]  # [M,J,3]
            v_gt = gt_rel[mask]
            M = v_gt.shape[0]

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

            # MPJVE：只在相邻两帧都有效时统计
            vel_mask = mask[:, 1:] & mask[:, :-1]
            if vel_mask.any():
                ve = compute_mpjve(pred_rel, gt_rel)  # [B,T-1]
                v_ve = ve[vel_mask]
                total_mpjve += v_ve.mean().item() * v_ve.numel()
                total_vel_frames += v_ve.numel()

            batch_ssc = compute_spatial_structure_corr(v_pred, v_gt).item()
            total_ssc += batch_ssc * M

            if edge_i is not None:
                d_pred = torch.norm(v_pred[:, edge_i] - v_pred[:, edge_j], dim=-1)
                d_gt = torch.norm(v_gt[:, edge_i] - v_gt[:, edge_j], dim=-1)
                total_bone_mae += (d_pred - d_gt).abs().mean().item() * M

            # MJAE：三帧窗口都有效
            acc_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
            if acc_mask.any():
                ae = compute_mjae(pred_rel, gt_rel)  # [B,T-2]
                v_ae = ae[acc_mask]
                total_mjae += v_ae.mean().item() * v_ae.numel()
                total_acc_frames += v_ae.numel()

            # jerk：四帧窗口都有效
            jerk_mask = mask[:, 3:] & mask[:, 2:-1] & mask[:, 1:-2] & mask[:, :-3]
            if jerk_mask.any():
                total_jerk_pred += compute_jerk_energy_masked(pred_rel, mask).item()
                total_jerk_gt += compute_jerk_energy_masked(gt_rel, mask).item()
                total_jerk_count += 1

            if edge_i is not None:
                x_pred_valid = pred_rel[mask]
                x_gt_valid = gt_rel[mask]
                if x_pred_valid.shape[0] > 1:
                    total_bone_var_pred += compute_bone_length_var(x_pred_valid, vis_edges).item()
                    total_bone_var_gt += compute_bone_length_var(x_gt_valid, vis_edges).item()
                    total_bone_var_count += 1

            # K-sampling
            if do_k_sampling and k_samples >= 2:
                samples = [pred]  # ✅ sample0 用上面算过的 pred，保证一致
                for _ in range(k_samples - 1):
                    try:
                        samples.append(model.inference(radar_seq, valid_mask=valid_mask, steps=steps, noise_level=sample_noise_level))
                    except TypeError:
                        samples.append(model.inference(radar_seq, steps=steps, noise_level=sample_noise_level))
                samples = torch.stack(samples, dim=0)  # [K,B,T,J,3]

                samples_mm = samples * 1000.0
                samples_rel = samples_mm - samples_mm[:, :, :, 0:1, :]

                bok = best_of_k_mpjpe_bt(samples_rel, gt_rel, valid_mask_bt=mask)

                # ✅ 健康检查：bok 必须 <= sample0 的误差
                err0 = torch.norm(samples_rel[0] - gt_rel, dim=-1).mean(dim=-1)  # [B,T]
                err0 = err0[mask].mean()
                assert bok <= err0 + 1e-4, (bok.item(), err0.item())

                div = compute_sample_diversity(samples_rel, valid_mask_bt=mask)

                total_bok_mpjpe += bok.item() * M
                total_diversity += div.item() * M
                total_k_count += M

                if edge_i is not None:
                    k_bone_mean, k_bone_std = compute_k_bone_mae_mean_std(samples_rel, gt_rel, valid_mask_bt=mask)
                    total_k_bone_mae_mean += k_bone_mean.item() * M
                    total_k_bone_mae_std += k_bone_std.item() * M

            # traj metrics（可选）：这里也不需要额外 mask，最终统计也应只看 valid_mask，但你现在定义是沿 ODE steps 的平滑度
            if do_traj_metrics and (total_traj_count < traj_batches):
                try:
                    pred2, traj = model.inference(radar_seq, valid_mask=valid_mask, steps=steps,
                                                  noise_level=sample_noise_level, return_traj=True)
                except TypeError:
                    pred2, traj = model.inference(radar_seq, steps=steps,
                                                  noise_level=sample_noise_level, return_traj=True)
                traj_mm = traj * 1000.0
                traj_rel = traj_mm - traj_mm[..., 0:1, :]
                total_traj_jerk += compute_traj_jerk_energy(traj_rel).item()
                total_traj_path += compute_traj_path_length(traj_rel).item()
                total_traj_count += 1

        pbar.set_postfix({"mpjpe": f"{total_mpjpe/max(total_samples, 1):.2f}mm"})

    denom = max(total_samples, 1)
    final = {
        "mpjpe": total_mpjpe / denom,
        "pa_mpjpe": total_pampjpe / denom,
        "auc_pck": total_auc / denom,
        "avg_ssc": total_ssc / denom,
    }
    for th in pck_thresholds:
        final[f"pck@{int(th)}"] = total_pck[th] / denom

    if total_vel_frames > 0:
        final["mpjve"] = total_mpjve / total_vel_frames
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

    if total_k_count > 0:
        final[f"best_of_{k_samples}_mpjpe"] = total_bok_mpjpe / total_k_count
        final[f"diversity_{k_samples}"] = total_diversity / total_k_count
        if edge_i is not None:
            final[f"k_bone_mae_mean_{k_samples}"] = total_k_bone_mae_mean / total_k_count
            final[f"k_bone_mae_std_{k_samples}"] = total_k_bone_mae_std / total_k_count

    if total_traj_count > 0:
        final["traj_jerk"] = total_traj_jerk / total_traj_count
        final["traj_path"] = total_traj_path / total_traj_count

    return final


@torch.no_grad()
def sample_k_predictions(model, radar_input, K=5, steps=10, noise_level=0.02):
    """
    返回 samples: [K, B, T, J, 3] (GPU tensor)
    """
    outs = []
    for _ in range(K):
        pred = model.inference(radar_input, steps=steps, noise_level=noise_level)
        outs.append(pred)
    return torch.stack(outs, dim=0)

def best_of_k_mpjpe_bt(samples_bt, gt_bt, valid_mask_bt=None):
    """
    samples_bt: [K, B, T, J, 3]  (mm, root-relative 推荐)
    gt_bt:      [B, T, J, 3]
    valid_mask_bt: [B, T] bool, True 表示该帧有效（可选）

    返回 scalar
    """
    # [K,B,T,J]
    err = torch.norm(samples_bt - gt_bt.unsqueeze(0), dim=-1).mean(dim=-1)

    # [B,T] 取每帧最小误差
    best = err.min(dim=0).values

    if valid_mask_bt is not None:
        best = best[valid_mask_bt]
    return best.mean()

def compute_sample_diversity(samples_bt, valid_mask_bt=None):
    """
    samples_bt: [K, B, T, J, 3] (mm, root-relative 推荐)
    返回 scalar diversity（越大越多样）
    """
    K, B, T, J, _ = samples_bt.shape
    x = samples_bt.reshape(K, B*T, J, 3)  # [K, M, J, 3]
    if valid_mask_bt is not None:
        m = valid_mask_bt.reshape(B*T)
        x = x[:, m]

    # pairwise distances between samples: O(K^2) 但 K<=10 很小
    # dist(i,j) = mean_joints ||x_i - x_j||
    dsum = 0.0
    cnt = 0
    for i in range(K):
        for j in range(i+1, K):
            d = torch.norm(x[i] - x[j], dim=-1).mean(dim=-1)  # [M]
            dsum = dsum + d.mean()
            cnt += 1
    return dsum / max(cnt, 1)

def compute_bone_mae_k(samples_bt, gt_bt, edges, valid_mask_bt=None):
    """
    返回：
      mean_bone_mae: K 个样本 bone_mae 的均值
      std_bone_mae:  K 个样本 bone_mae 的标准差（反映结构稳定性）
    """
    K, B, T, J, _ = samples_bt.shape
    x = samples_bt.reshape(K, B*T, J, 3)
    gt = gt_bt.reshape(B*T, J, 3)
    if valid_mask_bt is not None:
        m = valid_mask_bt.reshape(B*T)
        x = x[:, m]
        gt = gt[m]

    maes = []
    for k in range(K):
        maes.append(compute_bone_length_mae(x[k], gt, edges))
    maes = torch.stack(maes)  # [K]
    return maes.mean(), maes.std(unbiased=False)

def compute_traj_jerk_energy(traj_steps):
    """
    traj_steps: [S, B, T, J, 3] (S=steps+1)
    返回 scalar：越小表示积分轨迹更平滑（沿采样步）
    """
    # third difference along S
    if traj_steps.shape[0] < 4:
        return torch.tensor(0.0, device=traj_steps.device)
    j = traj_steps[3:] - 3*traj_steps[2:-1] + 3*traj_steps[1:-2] - traj_steps[:-3]
    return (j**2).sum(dim=-1).mean()

def compute_traj_path_length(traj_steps):
    """
    traj_steps: [S, B, T, J, 3]
    返回 scalar：路径越短说明走得更直接、更稳定（一般）
    """
    v = traj_steps[1:] - traj_steps[:-1]          # [S-1,B,T,J,3]
    step_len = torch.norm(v, dim=-1).mean(dim=-1) # [S-1,B,T]
    return step_len.mean()

def compute_bone_energy_steps(traj_steps, edges):
    """
    traj_steps: [S, B, T, J, 3]
    返回 [S]：每一步的骨长方差（或能量）随步数变化
    """
    S, B, T, J, _ = traj_steps.shape
    x = traj_steps.reshape(S, B*T, J, 3)
    energies = []
    for s in range(S):
        energies.append(compute_bone_length_var(x[s], edges))
    return torch.stack(energies)  # [S]

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
    
def batch_procrustes_align(X, Y, eps=1e-8):
    # X, Y: [M, J, 3]
    muX = X.mean(dim=1, keepdim=True)
    muY = Y.mean(dim=1, keepdim=True)
    X0 = X - muX
    Y0 = Y - muY

    cov = X0.transpose(1, 2) @ Y0  # [M,3,3]
    U, S, Vh = torch.linalg.svd(cov, full_matrices=False)
    V = Vh.transpose(1, 2)

    R = V @ U.transpose(1, 2)      # [M,3,3]

    # reflection fix: enforce det(R)=+1 by adjusting V
    detR = torch.det(R)
    mask = detR < 0
    if mask.any():
        V[mask, :, -1] *= -1
        R = V @ U.transpose(1, 2)

    # scale: trace(R^T cov) / ||X0||^2
    varX = (X0 ** 2).sum(dim=(1, 2)) + eps
    trace = torch.einsum('bij,bij->b', R, cov)  # trace(R^T cov) == sum_{ij} R_ij * cov_ij
    scale = S.sum(dim=1) / varX

    X_aligned = scale.view(-1, 1, 1) * (X0 @ R) + muY
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
    
def compute_jerk_energy_masked(seq, valid_mask, eps=1e-8):
    """
    seq: [B,T,J,3]  (mm, root-relative)
    valid_mask: [B,T] bool
    return: scalar jerk energy over valid windows only
    """
    # [B,T-3]：每个 jerk 项对应的 4 帧都得有效
    m = valid_mask[:, 3:] & valid_mask[:, 2:-1] & valid_mask[:, 1:-2] & valid_mask[:, :-3]
    if not m.any():
        return torch.tensor(0.0, device=seq.device, dtype=seq.dtype)

    j = seq[:, 3:] - 3*seq[:, 2:-1] + 3*seq[:, 1:-2] - seq[:, :-3]  # [B,T-3,J,3]
    e = (j**2).sum(dim=-1)  # [B,T-3,J]
    # 只统计有效 jerk window
    m3 = m.unsqueeze(-1).float()  # [B,T-3,1]
    denom = m3.sum() * e.shape[-1] + eps  # 有效项数量 * J
    return (e * m3).sum() / denom

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

import torch
from tqdm import tqdm

@torch.no_grad()
def quick_validate(
    dataloader, model, device,
    steps=2,                 # ODE 积分步数，训练时建议 1~3
    max_batches=5,           # 只跑前几个 batch，加速
    pck_thr=50.0,            # mm
    compute_vel=True,        # 是否输出 mpjve
):
    model.eval()

    total_mpjpe = 0.0
    total_pck = 0.0
    total_mpjve = 0.0
    total_frames = 0
    total_vel_frames = 0

    it = 0
    for radar_seq, skeleton_seq, valid_mask in dataloader:
        radar_seq = radar_seq.to(device).float()
        skeleton_seq = skeleton_seq.to(device).float()
        valid_mask = valid_mask.to(device).bool()  # [B,T]

        B, T, N, C = radar_seq.shape

        # 推理（如果你的 inference 支持 valid_mask 就传，不支持也没关系）
      
        pred = model.inference(radar_seq, steps=steps)
      

        # mm + root-relative
        pred_mm = pred * 1000.0
        gt_mm = skeleton_seq * 1000.0
        pred_rel = pred_mm - pred_mm[:, :, 0:1, :]
        gt_rel = gt_mm - gt_mm[:, :, 0:1, :]

        mask = valid_mask
        if mask.any():
            v_pred = pred_rel[mask]  # [M,J,3]
            v_gt = gt_rel[mask]
            M = v_gt.shape[0]

            # MPJPE
            mpjpe = torch.norm(v_pred - v_gt, dim=-1).mean()  # scalar
            total_mpjpe += mpjpe.item() * M

            # PCK@thr
            err = torch.norm(v_pred - v_gt, dim=-1)  # [M,J]
            pck = (err < pck_thr).float().mean()
            total_pck += pck.item() * M

            total_frames += M

        # MPJVE（相邻两帧都有效才算）
        if compute_vel:
            vel_mask = mask[:, 1:] & mask[:, :-1]  # [B,T-1]
            if vel_mask.any():
                v_pred = pred_rel[:, 1:] - pred_rel[:, :-1]  # [B,T-1,J,3]
                v_gt = gt_rel[:, 1:] - gt_rel[:, :-1]
                ve = torch.norm(v_pred - v_gt, dim=-1).mean(dim=-1)  # [B,T-1]
                ve_valid = ve[vel_mask]
                total_mpjve += ve_valid.mean().item() * ve_valid.numel()
                total_vel_frames += ve_valid.numel()

        it += 1
        if it >= max_batches:
            break

        with torch.no_grad():
            # 1) coarse baseline（不经过 refiner）
            z_seq, x_coarse = model.get_coarse_prior(radar_seq)  # x_coarse: [B*T,27,3]
            x_coarse = x_coarse.view(B, T, 27, 3)
        
            # 2) 一步 refiner 的输出（残差）
            radar_raw = radar_seq.flatten(0,1)                     # [B*T,128,6]
            t1 = torch.ones((B*T,1), device=radar_seq.device)
            v_pred = model.refiner(x_coarse.flatten(0,1), t1, x_coarse.flatten(0,1), radar_raw)  # [B*T,27,3]
            v_pred = v_pred.view(B, T, 27, 3)
        
            # 3) refined
            x_ref = x_coarse + v_pred
        
            # 统一 root-rel（确保口径一致）
            gt = skeleton_seq
            x_coarse_rel = x_coarse - x_coarse[:,:,0:1,:]
            x_ref_rel    = x_ref    - x_ref[:,:,0:1,:]
            gt_rel       = gt       - gt[:,:,0:1,:]
        
            # 只在有效帧统计（valid_mask=True）
            m = valid_mask.bool()
            def mpjpe_mm(a,b):
                return (torch.norm(a-b, dim=-1).mean(dim=-1)[m].mean() * 1000).item()
        
            print("coarse MPJPE(mm):", mpjpe_mm(x_coarse_rel, gt_rel))
            print("v_pred max(abs) (m):", v_pred.abs().max().item())
            print("refined MPJPE(mm):", mpjpe_mm(x_ref_rel, gt_rel))


    denom = max(total_frames, 1)
    out = {
        "mpjpe": total_mpjpe / denom,
        "pck@50": total_pck / denom,
    }
    if compute_vel and total_vel_frames > 0:
        out["mpjve"] = total_mpjve / total_vel_frames
    return out

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
    save_path = "/code/radarFMv2/checkpoints_refinev8"
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
    ckpt_stage1 = "/code/radarFMv2/checkpoints_stage1v4/best_model.pt"
    model.load_pretrained_stage1(ckpt_stage1, freeze=True)

    # ckpt_path = "/code/radarFMv2/checkpoints_refinev7/best_refiner.pt"
    # # ckpt_path = "./checkpoints_refine/best_refiner.pt"
    # checkpoint = torch.load(ckpt_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # steps_list = [1, 2, 5, 10, 15, 20]

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
    #         steps=s,
    #         do_k_sampling=True, k_samples=5, sample_noise_level=0.01,
    #         do_traj_metrics=True, traj_batches=2
            
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

    # --- 4. 优化器配置 ---
    # 仅更新需要梯度的参数（即 Refiner 部分）
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
    best_mpjpe = 1e9

    for epoch in range(1, 251):
        model.train()
        total_epoch_loss = 0.0
    
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for radar_seq, skeleton_seq, valid_mask in pbar:
            radar_seq = radar_seq.to(device, non_blocking=True)
            skeleton_seq = skeleton_seq.to(device, non_blocking=True)
            valid_mask = valid_mask.to(device, non_blocking=True).bool()  # [B,T], True=真实帧
    
            optimizer.zero_grad(set_to_none=True)
    
            # 混合精度训练
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # ✅ 关键：把 valid_mask 传入 loss
                loss = model.compute_fm_loss(radar_seq, skeleton_seq)
    
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
    
            total_epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
        # --- 验证与保存 ---
        if epoch % 2 == 0 or epoch == 1:
            metrics = quick_validate(
                dataloader=val_loader,
                model=model,
                device=device,
                steps=2,
                max_batches=5,
                pck_thr=50.0,
                compute_vel=True
            )
            print(f"[Epoch {epoch}] quick val: "
                  f"MPJPE {metrics['mpjpe']:.2f}mm | "
                  f"PCK@50 {metrics['pck@50']:.3f} | "
                  + (f"MPJVE {metrics['mpjve']:.2f}" if 'mpjve' in metrics else ""))
            val_mpjpe = metrics['mpjpe']
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