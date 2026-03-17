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
        
class SingleFrameFlowTransformer(nn.Module):
    """
    Refiner: 基于 Flow Matching 修正骨架
    Input: x_t [B, 17, 3], z_radar [B, 64, 256]
    Output: v_pred [B, 17, 3]
    """
    def __init__(self, num_joints=17, radar_embed_dim=256, embed_dim=512):
        super().__init__()
        self.num_joints = num_joints
        
        if radar_embed_dim != embed_dim:
            self.radar_proj = nn.Linear(radar_embed_dim, embed_dim)
        else:
            self.radar_proj = nn.Identity()

        # Embedding 层
        self.joint_embed = nn.Linear(3, embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.coarse_embed = nn.Linear(3, embed_dim)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=8, 
            dim_feedforward=2048,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        self.vel_head = nn.Linear(embed_dim, 3)

    def forward(self, x_t, tau, z_radar, x_coarse):
        # x_t: [B, 17, 3]
        # tau: [B, 1]
        # z_radar: [B, 64, 256] <-- 注意这里是 256
        
        # 1. 投影雷达特征 [B, 64, 256] -> [B, 64, 512]
        # 此时 z_radar 变成了 Key/Value 兼容的维度
        z_radar = self.radar_proj(z_radar) 

        # 2. 构造 Query
        h_xt = self.joint_embed(x_t)            # [B, 17, 512]
        h_tau = self.time_embed(tau).unsqueeze(1) # [B, 1, 512]
        h_coarse = self.coarse_embed(x_coarse)  # [B, 17, 512]
        
        query = h_xt + h_tau + h_coarse         # [B, 17, 512]
        
        # 3. Cross-Attention
        # Query(17, 512) <-> Key(64, 512)
        refined_feat = self.transformer(tgt=query, memory=z_radar) # [B, 17, 512]
        
        # 4. 预测速度
        vt = self.vel_head(refined_feat) # [B, 17, 3]
        return vt


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
            radar_embed_dim=radar_embed_dim,
            embed_dim=refine_embed_dim
        )
        
        self.parent_dict = parent_list

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
        
        # 预测速度向量 v_t
        v_pred = self.refiner(x_t, t, z_seq, x_coarse)
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
            v_pred = self.refiner(x_curr, t_tensor, z_seq, x_coarse)
            
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


# ==========================================
# 3. 工具函数
# ==========================================

def calculate_mpjpe(pred, gt):
    # pred, gt: [B, 17, 3]
    return torch.norm(pred - gt, dim=-1).mean().item()

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
    针对 Flow Matching 优化的序列评估函数 - 已恢复 GIF 生成功能
    """
    model.eval()
    total_mpjpe, total_pampjpe, total_auc = 0.0, 0.0, 0.0
    total_pck = {th: 0.0 for th in pck_thresholds}
    total_mpjve, total_vel_frames, total_samples = 0.0, 0, 0
    
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
    
    has_visualized = 0
    pbar = tqdm(dataloader, desc=f"Eval (Steps={steps})")

    for batch_idx, batch in enumerate(dataloader):
        radar_seq = batch['radar_cond']
        skeleton_seq = batch['pointcloud']
        radar_seq = radar_seq.to(device).float()     # [B, T, 128, 6]
        skeleton_seq = skeleton_seq.to(device).float() # [B, T, 17, 3]
        B, T, N, C = radar_seq.shape
        
        # 1. 识别并提取有效帧掩码
        is_valid_mask = (skeleton_seq.abs().sum(dim=(2, 3)) > 1e-6)
        
        # 2. 调用模型推理 (返回 pred 为 [B, T, 17, 3])
        pred = model.inference(radar_seq, steps=steps) 
        
        # 3. 转换 mm 并 Root-Relative
        pred_mm = pred.cpu() * 1000.0
        gt_mm = skeleton_seq.cpu() * 1000.0
        pred_rel = pred_mm - pred_mm[:, :, 0:1, :]
        gt_rel = gt_mm - gt_mm[:, :, 0:1, :]

        # 4. 指标计算
        mask = is_valid_mask.cpu() 
        if mask.any():
            v_pred, v_gt = pred_rel[mask], gt_rel[mask]
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

            # 速度指标 MPJVE
            vel_mask = mask[:, 1:] & mask[:, :-1]
            if vel_mask.any():
                ve = compute_mpjve(pred_rel, gt_rel)
                v_ve = ve[vel_mask]
                total_mpjve += v_ve.mean().item() * v_ve.numel()
                total_vel_frames += v_ve.numel()

            # --- 5. 恢复的可视化逻辑 (GIF & HTML) ---
            if vis_dir and has_visualized < num_vis_samples and vis_edges:
                # 找到当前 Batch 中有有效数据的样本索引
                valid_bs = [i for i in range(B) if mask[i].any()]
                
                if valid_bs:
                    # 1. 随机锁定一个样本 b
                    b = random.choice(valid_bs)
                    # 获取该样本下所有有效的帧索引 (全长)
                    valid_indices = torch.where(mask[b])[0] 
                    
                    if len(valid_indices) > 0:
                        # --- A. 生成全长 GIF ---
                        # 提取该样本的全长数据 [T_valid, 17, 3]
                        gt_seq_full = gt_rel[b, valid_indices].numpy()
                        pred_seq_full = pred_rel[b, valid_indices].numpy()
                        num_total_frames = gt_seq_full.shape[0]

                        fig = plt.figure(figsize=(10, 5))
                        ax_gt = fig.add_subplot(121, projection="3d")
                        ax_pr = fig.add_subplot(122, projection="3d")

                        def update(idx):
                            # 清除上一帧，防止重叠
                            ax_gt.cla()
                            ax_pr.cla()
                            actual_frame_idx = valid_indices[idx].item()
                            draw_skeleton_3d(ax_gt, gt_seq_full[idx], vis_edges, "green", f"GT Frame {actual_frame_idx}")
                            draw_skeleton_3d(ax_pr, pred_seq_full[idx], vis_edges, "red", f"Pred Frame {actual_frame_idx}")
                            return []

                        # interval 设小一点让播放更流畅
                        ani = animation.FuncAnimation(fig, update, frames=num_total_frames, interval=100)
                        gif_name = f"Sample{has_visualized}_B{batch_idx}_S{b}_full_seq.gif"
                        ani.save(os.path.join(vis_dir, gif_name), writer="pillow", fps=10)
                        plt.close(fig)

                        # --- B. 生成相同样本的 10 帧 HTML ---
                        # 从全长索引中均匀抽取 10 帧，或者取中间连续 10 帧
                        num_html = 10
                        if num_total_frames > num_html:
                            # 选中间一段连续的 10 帧
                            start_f = num_total_frames // 4
                            html_indices = valid_indices[start_f : start_f + num_html]
                        else:
                            html_indices = valid_indices

                        for t_idx in html_indices:
                            t = t_idx.item()
                            html_name = f"Sample{has_visualized}_B{batch_idx}_f{t:03d}.html"
                            plot_skeleton(
                                gt_joints=gt_rel[b, t].numpy(),
                                pred_joints=pred_rel[b, t].numpy(),
                                edges=vis_edges,
                                frame_id=f"Sample{has_visualized}-Frame{t}",
                                out_html=os.path.join(vis_dir, html_name)
                            )

                        # 完成一个样本的全套可视化，计数加 1
                        has_visualized += 1

        pbar.set_postfix({"mpjpe": f"{total_mpjpe/max(total_samples, 1):.2f}mm"})

    # 汇总
    denom = max(total_samples, 1)
    final = {
        "mpjpe": total_mpjpe / denom,
        "pa_mpjpe": total_pampjpe / denom,
        "auc_pck": total_auc / denom,
    }
    for th in pck_thresholds:
        final[f"pck@{int(th)}"] = total_pck[th] / denom
    if total_vel_frames > 0:
        final["mpjve"] = total_mpjve / total_vel_frames

    return final

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
    # 注意：输入应该是 numpy 数组 (17, 3)
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
    save_path = "/code/mmfi/checkpoints_baseline7"
    os.makedirs(save_path, exist_ok=True)
    
   # --- 2. 数据准备 ---
    dataset_root = '/code/mmfi/combined'
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
        parent_list=PARENT,
        refine_embed_dim=512
    ).to(device)

    # ckpt_path = "/code/mmfi/checkpoints_refinev3/best_refiner.pt"
    # checkpoint = torch.load(ckpt_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])

    # steps_list = [1, 2, 5, 10, 15, 20, 25, 30]

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

    # 加载你那个 98mm 的 Stage 1 权重
    # freeze=True 意味着我们只训练 Refiner，保持 Stage 1 不动
    ckpt_stage1 = "/code/mmfi/checkpoints_stage1v3/best_model.pt"
    model.load_pretrained_stage1(ckpt_stage1, freeze=True)

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
    best_mpjpe = float("inf")
    
    for epoch in range(1, 251):
        model.train()
        total_epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            radar_seq = batch['radar_cond']
            skeleton_seq = batch['pointcloud']
            radar_seq = radar_seq.to(device).float()     # [B, T, 128, 6]
            skeleton_seq = skeleton_seq.to(device).float() # [B, T, 17, 3]

            optimizer.zero_grad(set_to_none=True)

            # 使用混合精度训练
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # 调用模型内置的 FM Loss 计算函数
                loss = model.compute_fm_loss(radar_seq, skeleton_seq)

            scaler.scale(loss).backward()
            # 梯度裁剪防止 Refiner 训练早期震荡
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_epoch_loss += loss.item()
         

        # --- 6. 验证与保存 ---
        if epoch % 2 == 0 or epoch == 1:
            # 调用你优化后的 evaluate_sequence 函数
            metrics = evaluate_sequence(
                dataloader=val_loader,
                model=model,
                device=device,
                vis_dir=None,
                vis_edges=EDGES,
                num_vis_samples=1,
                steps=5  # 验证时使用 10 步 ODE 积分
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