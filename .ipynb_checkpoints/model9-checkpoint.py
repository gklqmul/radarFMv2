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
from tools import _masked_dyn_sum, compute_auc_pck, compute_mpjpe, compute_pampjpe, compute_pck, compute_spatial_structure_corr, diff1, diff2, diff3, draw_skeleton_3d, plot_skeleton, batch_procrustes_align, evaluate_sequence



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
    """
    输出 v(x_t, t | radar, x0) 作为 ODE 速度场 dx/dt
    """
    def __init__(self, num_joints=27, radar_in_channels=6, embed_dim=512,
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
        core_joints = [0, 1, 2, 3, 18, 22]
        scales[core_joints] = 0.6
        self.end_effectors = [9, 10, 16, 17, 20, 21, 24, 25, 26]
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
                 num_joints=27, parent_list=PARENT_LIST, refine_embed_dim=512):
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
        self.end_effectors = [9, 10, 16, 17, 21, 24, 25, 26]

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
    save_path = '/code/radarFMv2/checkpoints_basemodel'
    # save_path = "./checkpoints_refine"
    os.makedirs(save_path, exist_ok=True)
    
    dataset = RadarDiffusionDataset(
        root_dir='/code/radarFMv2/dataset', 
        # root_dir='./dataset',
        sample_level='sequence', 
        num_joints=27
    )
    
    # # 单机运行，直接使用常规 DataLoader
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
        parent_list=PARENT_LIST
    ).to(device)

    # # 加载你那个 98mm 的 Stage 1 权重
    # # freeze=True 意味着我们只训练 Refiner，保持 Stage 1 不动
    # ckpt_stage1 = "/code/radarFMv2/checkpoints_baseline1/best_model.pt"
    # model.load_pretrained_stage1(ckpt_stage1, freeze=True)

    ckpt_path = "/code/radarFMv2/checkpoints_basemodel/best_refiner.pt"
    # # ckpt_path = "./checkpoints_refine/best_refiner.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)

    # occlusion_scenarios = {
    # "Top_Half":    ((0.0, 0.75, 0.0),  (3.0, 1.5, 3.0)),   # 遮挡上方
    # "Bottom_Half": ((0.0, -0.75, 0.0), (3.0, 1.5, 3.0)),  # 遮挡下方
    # "Left_Side":   ((-0.75, 0.0, 0.0), (1.5, 3.0, 3.0)),  # 遮挡左侧
    # "Right_Side":  ((0.75, 0.0, 0.0),  (1.5, 3.0, 3.0)),  # 遮挡右侧
    # }
    
    # drop_ratios = [0.9,0.8, 0.7, 0.6, 0.5,0.4,0.3, 0.2,0.1]
    # steps_list = [5]
    
    # # 用于存储所有结果，方便最后统一输出表格
    # final_results = []
    
    # # 开始实验循环
    # for label, (b_center, b_size) in occlusion_scenarios.items():
    #     print(f"\n>>> 正在运行场景: {label}...")
        
    #     for d_ratio in drop_ratios:
    #         # 动态生成当前遮挡条件下的验证集
    #         v_set = dataset.get_val_set(
    #             box_size=b_size, 
    #             occl_prob=1.0, 
    #             drop_ratio=d_ratio, 
    #             box_center=b_center
    #         )
    #         # v_set.visualize_occlusion()
    #         val_loader = DataLoader(v_set, batch_size=16, shuffle=False, collate_fn=collate_fn_for_cross_modal)
    
    #         for s in steps_list:
    #             # 仅在特定的 step 开启可视化，节省资源
    #             # current_vis_dir = os.path.join(save_path, f"vis_{label}_r{d_ratio}_s{s}") if s == 1 and d_ratio==0.4 else None
    #             current_vis_dir = None
    #             metrics = evaluate_sequence(
    #                 dataloader=val_loader,
    #                 model=model,
    #                 device=device,
    #                 vis_dir=current_vis_dir,
    #                 vis_edges=EDGES,
    #                 num_vis_samples=5,
    #                 steps=s
    #             )
                
    #             # 保存这一行数据
    #             result_entry = {
    #                 "Scenario": label,
    #                 "Drop_Ratio": d_ratio,
    #                 "Steps": s,
    #                 **metrics  # 自动展开所有指标 (如 MPJPE, PCK 等)
    #             }
    #             final_results.append(result_entry)
    
    # # --- 核心：打印 Excel 专用表格格式 ---
    # print("\n" + "="*60)
    # print("实验完成！请复制下方内容粘贴至 Excel:")
    # print("="*60)
    
    # if final_results:
    #     # 提取所有指标的 Key 作为表头
    #     headers = list(final_results[0].keys())
    #     print("\t".join(headers)) # 打印表头，Tab分隔
        
    #     for row in final_results:
    #         values = []
    #         for h in headers:
    #             v = row[h]
    #             if isinstance(v, float):
    #                 values.append(f"{v:.4f}") # 保留4位小数
    #             else:
    #                 values.append(str(v))
    #         print("\t".join(values)) # 打印每一行，Tab分隔
        
    # steps_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    steps_list = [5, 10]

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
            steps=s,
            # joint_names=[f"joint_{i}" for i in range(27)]
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
    
    # for epoch in range(1, 100):
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
    #             loss = model.compute_fm_loss(radar_seq, skeleton_seq)
           
    #         scaler.scale(loss).backward()
    #         scaler.unscale_(optimizer)
    #         # 梯度裁剪防止 Refiner 训练早期震荡
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         scaler.step(optimizer)
    #         scaler.update()
            
    #         total_epoch_loss += loss.item()
    #         pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    #     # --- 6. 验证与保存 ---
    #     if epoch % 1 == 0 or epoch == 1:
    #         # 调用你优化后的 evaluate_sequence 函数
    #         metrics = evaluate_sequence(
    #             dataloader=val_loader,
    #             model=model,
    #             device=device,
    #             vis_edges=EDGES,
    #             num_vis_samples=3,
    #             steps=1  # 验证时使用 10 步 ODE 积分
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