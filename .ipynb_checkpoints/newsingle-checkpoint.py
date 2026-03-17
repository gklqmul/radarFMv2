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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# 假设 dataset.py 在同级目录下，请确保它里面的 _process_single_sample 已经是修复后的版本（输出6通道）
from dataset import DataReader, RadarDiffusionDataset
from flowmodels import collate_fn_for_cross_modal

import torch.distributed as dist
import os

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
# ==========================================
# 1. 常量定义
# ==========================================
EDGES_27 = [
    (0,1), (1,2), (2,3), (3,26), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (8,10),
    (3,11), (11,12), (12,13), (13,14), (14,15), (15,16), (15,17),
    (0,18), (18,19), (19,20), (20,21), (0,22), (22,23), (23,24), (24,25)
]
PARENT = {
    1:0, 2:1, 3:2, 26:3,
    4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:8,
    11:3, 12:11, 13:12, 14:13, 15:14, 16:15, 17:15,
    18:0, 19:18, 20:19, 21:20,
    22:0, 23:22, 24:23, 25:24
}

class CoarseSkeletonHead(nn.Module):
    def __init__(self, latent_dim, num_joints, parent):
        super().__init__()
        self.num_joints = num_joints
        self.parent = parent

        # 可学习的关节查询量：每个关节都有自己的特征向量
        self.joint_queries = nn.Parameter(torch.randn(1, num_joints, latent_dim))
        
        # 交叉注意力：Query是关节，Key/Value是雷达点云特征
        self.attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=8, batch_first=True)
        
        # 独立回归头
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 4) # dx, dy, dz, length
        )

    def forward(self, z_sequence):
        B_T = z_sequence.shape[0]
        # 1. 关节特征提取
        queries = self.joint_queries.expand(B_T, -1, -1)
        joint_features, _ = self.attention(queries, z_sequence, z_sequence)
        
        # 2. 预测偏移
        raw = self.regressor(joint_features)
        dir_raw = raw[:, 1:, :3] 
        len_raw = raw[:, 1:, 3]

        direction = F.normalize(dir_raw, dim=-1, eps=1e-6)
        length = F.softplus(len_raw)
        
        # 3. FK 递归计算坐标
        offsets = torch.zeros(B_T, self.num_joints, 3, device=z_sequence.device)
        offsets[:, 1:] = direction * length.unsqueeze(-1)

        joints = torch.zeros_like(offsets)
        for j in range(1, self.num_joints):
            p = self.parent[j]
            joints[:, j] = joints[:, p] + offsets[:, j]

        return joints, offsets, length

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
        

class RadarStage1Model(nn.Module):
    def __init__(self, in_channels=6, radar_embed_dim=256, num_latents=64,
                 num_joints=27, parent_list=None):
        super().__init__()

        # 复用你之前的 Encoder
        from flowmodels import TimeAwareCompressedRadarEncoder, TemporalAdapter
        
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

    def forward(self, radar_seq):
        """
        radar_seq: [B, T, 128, 6]
        return: x0 [B, T, 27, 3]
        """
        B, T, N, C = radar_seq.shape
        radar_flat = radar_seq.view(B * T, N, C)
        
        # 1. 基础特征提取 [B*T, 64, 256]
        z = self.encoder(radar_flat)
        
        # 2. 时序增强 [B*T, 64, 256]
        z = self.temporal_adapter(z, B, T)
        
        # 3. 关节注意力预测 (不再取 mean)
        x0, _, _ = self.coarse_head(z)

        return x0.view(B, T, 27, 3)


def stage1_loss(pred, gt, parent_dict):
    valid_mask = (gt.abs().sum(dim=(2, 3)) > 1e-6)
    if not valid_mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # 1. 基础位姿误差 (米)
    dist = torch.norm(pred - gt, dim=-1)
    loss_mpjpe = dist[valid_mask].mean()

    # 2. 几何结构误差
    bone_len_losses = []
    bone_dir_losses = []
    
    for j, p in parent_dict.items():
        vec_pred = pred[:, :, j] - pred[:, :, p]
        vec_gt = gt[:, :, j] - gt[:, :, p]
        
        # 长度误差 (MSE)
        len_pred = torch.norm(vec_pred, dim=-1)
        len_gt = torch.norm(vec_gt, dim=-1)
        bone_len_losses.append(F.mse_loss(len_pred[valid_mask], len_gt[valid_mask]))
        
        # 方向误差 (1 - CosSim)
        cos_sim = F.cosine_similarity(vec_pred, vec_gt, dim=-1)
        bone_dir_losses.append(1.0 - cos_sim[valid_mask].mean())

    # --- 关键改动：取平均值，不要直接 sum ---
    loss_geom = torch.stack(bone_len_losses).mean() + torch.stack(bone_dir_losses).mean()

    # 3. 时序速度误差
    if pred.shape[1] > 1:
        vel_pred = pred[:, 1:] - pred[:, :-1]
        vel_gt = gt[:, 1:] - gt[:, :-1]
        # 给末端关节的速度误差加点权重，减少手脚乱抖
        loss_vel = F.mse_loss(vel_pred, vel_gt)
    else:
        loss_vel = 0.0

    # 4. 最终加权：保持 MPJPE 是大头
    # 这样总 Loss 应该在 0.1 ~ 0.3 左右，更容易观察
    total_loss = loss_mpjpe * 1.0 + loss_geom * 0.5 + loss_vel * 0.1
    
    return total_loss


@torch.no_grad()
def eval_stage1(dataloader, model, device):
    model.eval()

    total_err = 0.0
    total_cnt = 0

    for radar_seq, skeleton_seq in dataloader:
        radar_seq = radar_seq.to(device)
        skeleton_seq = skeleton_seq.to(device)

        pred = model(radar_seq)  # [B, T, 27, 3]

        valid_mask = (skeleton_seq.abs().sum(dim=(2, 3)) > 1e-6)

        err = torch.norm(pred - skeleton_seq, dim=-1).mean(dim=-1)  # [B, T]
        err = err[valid_mask]

        total_err += err.sum().item()
        total_cnt += err.numel()

    return (total_err / max(total_cnt, 1)) * 1000.0  # → mm

def eval_stage1_with_viz(dataloader, model, device, save_gif=True, gif_name="stage1_check.gif"):
    model.eval()
    total_err = 0.0
    total_cnt = 0
    
    # 用于可视化的存储
    viz_sample = None

    with torch.no_grad():
        for i, (radar_seq, skeleton_seq) in enumerate(dataloader):
            radar_seq = radar_seq.to(device)
            skeleton_seq = skeleton_seq.to(device)

            # pred shape: [B, T, 27, 3]
            pred = model(radar_seq)  

            # 计算误差
            valid_mask = (skeleton_seq.abs().sum(dim=(2, 3)) > 1e-6)
            err = torch.norm(pred - skeleton_seq, dim=-1).mean(dim=-1)
            err_valid = err[valid_mask]

            total_err += err_valid.sum().item()
            total_cnt += err_valid.numel()

            # 只取第一个 batch 的第一个序列做可视化
            if save_gif and i == 0:
                # radar_seq 通常是 [B, T, N, C]，假设 C 的前3位是 XYZ
                viz_sample = {
                    'radar': radar_seq[15].cpu().numpy(),
                    'pred': pred[15].cpu().numpy(),
                    'gt': skeleton_seq[15].cpu().numpy()
                }

    mpjpe = (total_err / max(total_cnt, 1)) * 1000.0

    if save_gif and viz_sample is not None:
        create_skeleton_radar_gif(viz_sample, gif_name)

    return mpjpe

def create_skeleton_radar_gif(viz_sample, gif_name):
    radar_data = viz_sample['radar']  # [T, N, C]
    pred_data = viz_sample['pred']    # [T, 27, 3]
    gt_data = viz_sample['gt']        # [T, 27, 3]
    T = pred_data.shape[0]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 定义骨骼连接关系 (使用你之前的 PARENT 结构)
    # PARENT = {1:0, 2:1, ...}
    connections = [(k, v) for k, v in PARENT.items()]

    def update(frame):
        ax.clear()
        # 1. 绘制雷达点云 (散点)
        # 假设 radar_data 前三维是 XYZ
        r_pts = radar_data[frame]
        # 过滤掉全 0 的点
        mask = np.abs(r_pts[:, :3]).sum(axis=1) > 1e-6
        ax.scatter(r_pts[mask, 0], r_pts[mask, 1], r_pts[mask, 2], 
                   s=2, c='gray', alpha=0.5, label='Radar Points')

        # 2. 绘制预测骨架 (红色)
        p_pts = pred_data[frame]
        ax.scatter(p_pts[:, 0], p_pts[:, 1], p_pts[:, 2], c='red', s=20)
        for c1, c2 in connections:
            ax.plot([p_pts[c1, 0], p_pts[c2, 0]], 
                    [p_pts[c1, 1], p_pts[c2, 1]], 
                    [p_pts[c1, 2], p_pts[c2, 2]], color='red', linewidth=2)

        # 3. 绘制 GT 骨架 (蓝色，可选，用于对比)
        g_pts = gt_data[frame]
        if np.abs(g_pts).sum() > 1e-6:
            ax.scatter(g_pts[:, 0], g_pts[:, 1], g_pts[:, 2], c='blue', s=10, alpha=0.3)
            for c1, c2 in connections:
                ax.plot([g_pts[c1, 0], g_pts[c2, 0]], 
                        [g_pts[c1, 1], g_pts[c2, 1]], 
                        [g_pts[c1, 2], g_pts[c2, 2]], color='blue', linewidth=1, linestyle='--', alpha=0.3)

        # 设置固定坐标轴范围（根据你的数据场景调整，例如 -1 到 1 或者 0 到 2000）
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        ax.set_title(f"Frame {frame} | Stage 1 Check")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    ani = animation.FuncAnimation(fig, update, frames=T, interval=100)
    ani.save(gif_name, writer='pillow')
    plt.close()
    print(f"Visualization saved to {gif_name}")


def main_eval():
    # -------------------------
    # Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # Dataset
    # -------------------------
    dataset = RadarDiffusionDataset(
        root_dir="./dataset",
        sample_level="sequence",
        num_joints=27
    )

    val_set = dataset.get_val_set()

    val_loader = DataLoader(
        val_set,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn_for_cross_modal,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # -------------------------
    # Model
    # -------------------------
    model = RadarStage1Model(
        in_channels=6,
        radar_embed_dim=256,
        num_latents=64,
        num_joints=27,
        parent_list=PARENT
    ).to(device)

    # -------------------------
    # Load checkpoint (your format)
    # -------------------------
    ckpt_path = "./checkpoints_crossv1/coarse_head_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    model.encoder.load_state_dict(ckpt["radar_encoder_state_dict"])
    model.coarse_head.load_state_dict(ckpt["coarse_head_state_dict"])

    # -------------------------
    # Eval
    # -------------------------
    model.eval()
    with torch.no_grad():
        mpjpe = eval_stage1_with_viz(val_loader, model, device)

    print("mpjpe", mpjpe)


if __name__ == "__main__":
    main_eval()
    
# def main():
#     assert torch.cuda.is_available(), "CUDA 不可用"

#     local_rank = setup_ddp()
#     world_size = dist.get_world_size()
#     device = torch.device(f"cuda:{local_rank}")

#     if local_rank == 0:
#         print(f"Using {world_size} GPUs with DDP")

#     # -------------------------
#     # Dataset & Sampler
#     # -------------------------
#     dataset = RadarDiffusionDataset(
#         root_dir="./dataset",
#         sample_level="sequence",
#         num_joints=27
#     )

#     train_set = dataset.get_train_set()
#     val_set   = dataset.get_val_set()

#     train_sampler = torch.utils.data.distributed.DistributedSampler(
#         train_set, shuffle=True
#     )
#     val_sampler = torch.utils.data.distributed.DistributedSampler(
#         val_set, shuffle=False
#     )

#     train_loader = DataLoader(
#         train_set,
#         batch_size=4,          # ⚠️ 每张卡的 batch
#         sampler=train_sampler,
#         collate_fn=collate_fn_for_cross_modal,
#         num_workers=0,
#         pin_memory=True
#     )

#     val_loader = DataLoader(
#         val_set,
#         batch_size=4,
#         sampler=val_sampler,
#         collate_fn=collate_fn_for_cross_modal,
#         num_workers=0,
#         pin_memory=True
#     )

#     # -------------------------
#     # Model
#     # -------------------------
#     model = RadarStage1Model(
#         in_channels=6,
#         radar_embed_dim=256,
#         num_latents=64,
#         num_joints=27,
#         parent_list=PARENT
#     ).to(device)

#     model = torch.nn.parallel.DistributedDataParallel(
#         model,
#         device_ids=[local_rank],
#         output_device=local_rank,
#         find_unused_parameters=False
#     )
#     mpjpe = eval_stage1_with_viz(val_loader,model, device)
#     print("mpjpe",mpjpe)
    
#     # optimizer = torch.optim.AdamW(
#     #     model.parameters(),
#     #     lr=2e-4,
#     #     weight_decay=1e-4
#     # )

#     # best_mpjpe = float("inf")

#     # -------------------------
#     # Training Loop
#     # -------------------------
#     # for epoch in range(1, 81):
#     #     model.train()
#     #     train_sampler.set_epoch(epoch)

#     #     total_loss = torch.tensor(0.0, device=device)
#     #     total_frames = torch.tensor(0.0, device=device)

#     #     for radar_seq, skeleton_seq in train_loader:
#     #         radar_seq = radar_seq.to(device, non_blocking=True)
#     #         skeleton_seq = skeleton_seq.to(device, non_blocking=True)

#     #         optimizer.zero_grad(set_to_none=True)

#     #         with torch.autocast("cuda"):
#     #             pred = model(radar_seq)
#     #             loss = stage1_loss(pred, skeleton_seq, PARENT)

#     #         loss.backward()
#     #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#     #         optimizer.step()

#     #         B, T = radar_seq.shape[:2]
#     #         total_loss += loss.detach() * (B * T)
#     #         total_frames += (B * T)

#     #     # -------- 同步训练 loss --------
#     #     dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
#     #     dist.all_reduce(total_frames, op=dist.ReduceOp.SUM)

#     #     if local_rank == 0:
#     #         print(
#     #             f"[Epoch {epoch}] "
#     #             f"Train Loss: {(total_loss / total_frames).item():.6f} m"
#     #         )

#     #     # -------------------------
#     #     # Validation
#     #     # -------------------------
#     #     if epoch % 3 == 0:
#     #         mpjpe = eval_stage1(val_loader, model.module, device)

#     #         mpjpe_tensor = torch.tensor(mpjpe, device=device)
#     #         dist.all_reduce(mpjpe_tensor, op=dist.ReduceOp.SUM)
#     #         mpjpe = mpjpe_tensor.item() / world_size

#     #         if local_rank == 0:
#     #             print(f"[Epoch {epoch}] Val MPJPE: {mpjpe:.2f} mm")

#     #             if mpjpe < best_mpjpe:
#     #                 best_mpjpe = mpjpe
#     #                 torch.save(
#     #                     {
#     #                         "radar_encoder_state_dict": model.module.encoder.state_dict(),
#     #                         "coarse_head_state_dict": model.module.coarse_head.state_dict(),
#     #                     },
#     #                     "./checkpoints_crossv1/coarse_head_best.pt"
#     #                 )
#     #                 print(f"✅ New Best MPJPE: {best_mpjpe:.2f} mm")

#     dist.destroy_process_group()


# if __name__ == "__main__":
#     main()