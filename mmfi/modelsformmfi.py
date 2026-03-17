import os
import yaml
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

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


from mmfi_lib.mmfidataset import make_dataset, make_dataloader, DataReader


# 基于图片 image_ad92b9.png 的拓扑结构

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
    主模型封装类
    """
    def __init__(self, in_channels=6, radar_embed_dim=256, num_latents=64, 
                 num_joints=17, parent_list=None, refine_embed_dim=512):
        super().__init__()
        
        self.encoder = TimeAwareCompressedRadarEncoder(
            in_channels=in_channels,
            embed_dim=radar_embed_dim,
            num_latents=num_latents
        )
        
        self.coarse_head = CoarseSkeletonHead(
            latent_dim=radar_embed_dim, 
            num_joints=num_joints, 
            parent=parent_list
        )
        
        self.refiner = SingleFrameFlowTransformer(
            num_joints=num_joints,
            radar_embed_dim=radar_embed_dim,
            embed_dim=refine_embed_dim
        )

    def load_pretrained_stage1(self, ckpt_path, freeze=True):
        if not os.path.exists(ckpt_path):
            print(f"Checkpint {ckpt_path} not found, skipping load.")
            return
            
        print(f"Loading Stage 1 weights from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # 加载逻辑需要根据实际 checkpoint 里的 key 调整
        # 这里假设你分别保存了 state dict
        if "radar_encoder_state_dict" in checkpoint:
            try:
                self.encoder.load_state_dict(checkpoint["radar_encoder_state_dict"])
                self.coarse_head.load_state_dict(checkpoint["coarse_head_state_dict"])
            except Exception as e:
                print(f"Warning: Loading failed (shape mismatch?): {e}")
                return

        if freeze:
            for param in self.encoder.parameters(): param.requires_grad = False
            for param in self.coarse_head.parameters(): param.requires_grad = False
            self.encoder.eval()
            self.coarse_head.eval()

    def get_coarse_prior(self, radar_input):
        """
        Input: [B, 128, 6]
        Output: z_seq [B, 64, 256], x_coarse [B, 17, 3]
        """
        # 1. 得到压缩序列特征 [B, 64, 256]
        z_seq = self.encoder(radar_input)
        
        # 2. 聚合得到全局特征 [B, 256] 用于 Coarse Head
        z_global = z_seq.mean(dim=1)
        
        # 3. 得到 x0
        x_coarse, _, _ = self.coarse_head(z_global)
        
        return z_seq, x_coarse

    def forward(self, x_t, t, radar_input):
        # 1. 提取特征和起点
        z_seq, x_coarse = self.get_coarse_prior(radar_input)
        
        # 2. 预测速度 (自动处理 256 -> 512 的映射)
        v_pred = self.refiner(x_t, t, z_seq, x_coarse)
        return v_pred

    @torch.no_grad()
    def inference_single_frame(self, radar_input, steps=10):
        """单帧推理"""
        self.eval()
        B = radar_input.shape[0]
        device = radar_input.device
        
        z_seq, x_coarse = self.get_coarse_prior(radar_input)
        x_curr = x_coarse.clone()
        dt = 1.0 / steps
        
        for i in range(steps):
            t_val = i / steps
            t_tensor = torch.full((B, 1), t_val, device=device).float()
            v_pred = self.refiner(x_curr, t_tensor, z_seq, x_coarse)
            x_curr = x_curr + v_pred * dt
            
        return x_curr


# ==========================================
# 3. 工具函数
# ==========================================

def calculate_mpjpe(pred, gt):
    # pred, gt: [B, 17, 3]
    return torch.norm(pred - gt, dim=-1).mean().item()

@torch.no_grad()
def evaluate_sequence(
    dataloader, reader, model, device, 
    vis_dir=None, vis_edges=None, num_vis_samples=2
):
    model.eval()
    total_mpjpe = 0.0
    total_samples = 0
    
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
    
   
    target_batch = random.randint(0, len(dataloader) - 1)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Eval Batches")):
        radar_seq = batch['radar_cond']
        skeleton_seq = batch['pointcloud']
        
        radar_seq = radar_seq.to(device).float()
        skeleton_seq = skeleton_seq.to(device).float()
        
        # 1. 获取维度
        B, T, N, C = radar_seq.shape

        is_valid_mask = (skeleton_seq.abs().sum(dim=(2, 3)) > 1e-6).view(-1)
        
        # 2. 展平进行推理
        radar_flat = radar_seq.view(B * T, N, C)
        gt_flat    = skeleton_seq.view(B * T, 17, 3) 

        with torch.no_grad():
            pred_flat = model.inference_single_frame(radar_flat, steps=20)

        
        # 3. 反归一化 (Flat 状态下进行)
        pred_mm = reader.denormalize_pointcloud(pred_flat.cpu()) 
        gt_mm   = reader.denormalize_pointcloud(gt_flat.cpu())

        # 4. 去中心化 (Root Relative)
        pred_root = pred_mm[:, 0:1, :]
        gt_root   = gt_mm[:, 0:1, :]
        
        pred_mm_rel = pred_mm - pred_root
        gt_mm_rel   = gt_mm - gt_root

        mask_cpu = is_valid_mask.cpu()
        valid_pred = pred_mm_rel[mask_cpu] # [N_valid, 17, 3]
        valid_gt   = gt_mm_rel[mask_cpu]   # [N_valid, 17, 3]

        if valid_gt.shape[0] > 0:
            mpjpe = calculate_mpjpe(valid_pred, valid_gt)
            total_mpjpe += mpjpe * valid_gt.shape[0]
            total_samples += valid_gt.shape[0]

        # 5. 计算指标
        # calculate_mpjpe 返回的是当前 batch 所有帧的平均值
        # mpjpe = calculate_mpjpe(pred_mm_rel, gt_mm_rel) 
        
        # # 【修正】这里样本总数是 B*T，不是 B
        # total_mpjpe += mpjpe * (B * T)
        # total_samples += (B * T)
        
        # 6. 可视化逻辑
        if vis_dir is not None and batch_idx == target_batch and vis_edges is not None:
            print(f"[Eval] Generating visualizations...")
            
            # --- Reshape 回序列 ---
            # 使用刚才算好的 _rel (相对坐标) 进行可视化，这样骨架不会乱飞
            vis_pred_seq = pred_mm_rel.view(B, T, 17, 3).numpy() 
            vis_gt_seq   = gt_mm_rel.view(B, T, 17, 3).numpy()   
            
            limit = min(B, num_vis_samples)
            
            for b in range(limit):
                # 【修正】这里要用 reshaped 后的 vis_pred_seq，而不是 flat 的 pred_mm
                sample_pred = vis_pred_seq[b] # [T, 17, 3]
                sample_gt   = vis_gt_seq[b]   # [T, 17, 3]
                
                # 随机选几帧
                if T > 10:
                    selected_indices = np.linspace(0, T-1, 10, dtype=int)
                else:
                    selected_indices = range(T)

                for t in selected_indices:
                    gt_sum = np.sum(np.abs(sample_gt[t]))
                    if gt_sum < 1e-3: # 如果和几乎为 0
                        print(f"[Info] Batch {batch_idx} Sample {b} Frame {t}: GT is all zeros (Padding). Skipping plot.")
                        continue # 跳过这一帧，不画图
                    file_name = f"batch{batch_idx}_sample{b}_frame{t:03d}.html"
                    save_path = os.path.join(vis_dir, file_name)
                    
                    # 此时 sample_gt[t] 才是第 t 帧 [17, 3]
                    plot_skeleton(
                        gt_joints=sample_gt[t],
                        pred_joints=sample_pred[t],
                        edges=vis_edges,
                        frame_id=t,
                        out_html=save_path
                    )
            
            print(f"[Eval] Visualization saved to {vis_dir}")

    return total_mpjpe / max(total_samples, 1)

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
    # ==============================
    # 0. 初始化 DDP
    # ==============================
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = (local_rank == 0)

    if is_main_process:
        print("Using DDP with 2 GPUs")

    # ==============================
    # 1. 数据 & 配置
    # ==============================
    save_path = "./checkpoints_cross_mmfi"
    dataset_root = "./combined"
    config_file = "./config.yaml"

    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    reader = DataReader(cache_path="state_cacheB.pt")
    train_dataset, val_dataset = make_dataset(dataset_root, config)

    # --- Distributed Sampler ---
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        **config["train_loader"]
    )

    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        **config["validation_loader"]
    )

    # ==============================
    # 2. 模型
    # ==============================
    model = RadarPoseRefiner(
        in_channels=6,
        radar_embed_dim=256,
        num_latents=64,
        num_joints=17,
        parent_list=PARENT,
        refine_embed_dim=512
    ).to(device)

    # 加载 Stage 1
    model.load_pretrained_stage1("./coarse_head_best.pt", freeze=True)

    # --- DDP 包装 ---
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )

    # ==============================
    # 3. 优化器（只优化 Refiner）
    # ==============================
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    best_mpjpe = float("inf")

    for epoch in range(1, 102):
        # ⚠️ 必须：让每个 epoch shuffle 不同
        train_sampler.set_epoch(epoch)

        model.train()
        # 冻结 Stage 1（DDP 下必须用 module）
        model.module.encoder.eval()
        model.module.coarse_head.eval()

        epoch_loss = 0.0
        samples = 0

        for batch in tqdm(train_loader, disable=not is_main_process):
            radar_seq = batch["radar_cond"].to(device).float()
            skeleton_seq = batch["pointcloud"].to(device).float()

            B, T = radar_seq.shape[:2]
            is_valid_frame = (skeleton_seq.abs().sum(dim=(2, 3)) > 1e-6).float()
            mask_flat = is_valid_frame.view(B * T, 1)

            radar_flat = radar_seq.view(B * T, 128, 6)
            x_1_flat   = skeleton_seq.view(B * T, 17, 3)

            # -------- Stage 1（无梯度）--------
            with torch.no_grad():
                z_seq_flat, x_0_flat = model.module.get_coarse_prior(radar_flat)

            # -------- Flow Matching --------
            t_rand = torch.rand(B * T, 1, device=device)
            x_t = (1 - t_rand.unsqueeze(-1)) * x_0_flat + t_rand.unsqueeze(-1) * x_1_flat
            v_target = x_1_flat - x_0_flat

            v_pred = model.module.refiner(x_t, t_rand, z_seq_flat, x_0_flat)

            loss_unreduced = F.mse_loss(v_pred, v_target, reduction='none') # [B*T, 17, 3]
            loss_per_frame = loss_unreduced.mean(dim=(1, 2))
            loss = (loss_per_frame * mask_flat.squeeze()).sum() / (mask_flat.sum() + 1e-6)
            # loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            valid_samples_count = mask_flat.sum().item()
            epoch_loss += loss.item() * valid_samples_count
            samples += valid_samples_count

        avg_loss = epoch_loss / max(samples, 1)

        if is_main_process:
            print(f"Epoch {epoch} | Train Loss: {avg_loss:.6f}")

        # ==============================
        # 5. 验证（只在 rank=0）
        # ==============================
        if is_main_process and epoch % 4 == 0:
            val_mpjpe = evaluate_sequence(
                dataloader=val_loader,
                reader=reader,
                model=model.module,   # ⚠️ 注意这里
                device=device,
                vis_dir=f"{save_path}/vis_results/epoch_{epoch}",
                vis_edges=EDGES,
                num_vis_samples=1,
            )

            print(f"Epoch {epoch} | Val MPJPE: {val_mpjpe:.2f} mm")

            if val_mpjpe < best_mpjpe:
                best_mpjpe = val_mpjpe
                torch.save(
                    model.module.state_dict(),
                    f"{save_path}/best_model.pt"
                )
                print(f"New Best MPJPE: {best_mpjpe:.2f} mm")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()