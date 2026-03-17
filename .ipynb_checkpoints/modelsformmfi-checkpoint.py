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

# 假设 dataset.py 在同级目录下，请确保它里面的 _process_single_sample 已经是修复后的版本（输出6通道）
from dataset import DataReader
from mmfi_lib.mmfi import make_dataset, make_dataloader
from flowmodels import collate_fn_for_cross_modal

# ==========================================
# 1. 常量定义
# ==========================================
EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

PARENT = {
    1: 0,
    2: 0,
    3: 1,
    4: 2,
    6: 5,
    11: 5,
    12: 6,
    7: 5,
    9: 7,
    8: 6,
    10: 8,
    13: 11,
    15: 13,
    14: 12,
    16: 14
}

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
    Input: x_t [B, 27, 3], z_radar [B, 64, 256]
    Output: v_pred [B, 27, 3]
    """
    def __init__(self, num_joints=27, radar_embed_dim=256, embed_dim=512):
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
        # x_t: [B, 27, 3]
        # tau: [B, 1]
        # z_radar: [B, 64, 256] <-- 注意这里是 256
        
        # 1. 投影雷达特征 [B, 64, 256] -> [B, 64, 512]
        # 此时 z_radar 变成了 Key/Value 兼容的维度
        z_radar = self.radar_proj(z_radar) 

        # 2. 构造 Query
        h_xt = self.joint_embed(x_t)            # [B, 27, 512]
        h_tau = self.time_embed(tau).unsqueeze(1) # [B, 1, 512]
        h_coarse = self.coarse_embed(x_coarse)  # [B, 27, 512]
        
        query = h_xt + h_tau + h_coarse         # [B, 27, 512]
        
        # 3. Cross-Attention
        # Query(27, 512) <-> Key(64, 512)
        refined_feat = self.transformer(tgt=query, memory=z_radar) # [B, 27, 512]
        
        # 4. 预测速度
        vt = self.vel_head(refined_feat) # [B, 27, 3]
        return vt


class RadarPoseRefiner(nn.Module):
    """
    主模型封装类
    """
    def __init__(self, in_channels=6, radar_embed_dim=256, num_latents=64, 
                 num_joints=27, parent_list=None, refine_embed_dim=512):
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
        Output: z_seq [B, 64, 256], x_coarse [B, 27, 3]
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
    # pred, gt: [B, 27, 3]
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
    
    has_visualized = False

    # 1. 给 Dataloader 加进度条，不想等死必须要看进度
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Eval Batches")):
        radar_seq, skeleton_seq = batch
        # radar_seq: [B, T, 128, 6]
        
        radar_seq = radar_seq.to(device).float()
        skeleton_seq = skeleton_seq.to(device).float()
        
        B, T, N, C = radar_seq.shape
        
        radar_flat = radar_seq.view(B * T, N, C)
        
        with torch.no_grad():
            # inference_single_frame 内部会自动广播 steps
            # 输出: [B*T, 27, 3]
            pred_flat = model.inference_single_frame(radar_flat, steps=20)
        
        # 3. Reshape 回去: [B*T, 27, 3] -> [B, T, 27, 3]
        pred_seq = pred_flat.view(B, T, 27, 3)
        

        pred_mm = reader.denormalize_pointcloud(pred_seq.cpu())
        gt_mm = reader.denormalize_pointcloud(skeleton_seq.cpu())
        
        mpjpe = calculate_mpjpe(pred_mm, gt_mm) 
        total_mpjpe += mpjpe * B
        total_samples += B
        
        # --- 4. 可视化逻辑 (只在第一个 Batch 执行) ---
        if vis_dir is not None and not has_visualized and vis_edges is not None:
            print(f"[Eval] Generating visualizations for first {num_vis_samples} samples...")
            
            # 遍历这个 Batch 中的前几个样本
            limit = min(B, num_vis_samples)
            for b in range(limit):
                # 获取该样本的所有帧
                sample_pred = pred_mm[b].numpy() # [T, 27, 3]
                sample_gt   = gt_mm[b].numpy()   # [T, 27, 3]
                selected_indices = np.random.choice(T, size=10, replace=False)
                selected_indices.sort()
                # 遍历该样本的时间步 (跳帧画图，避免太多)
                for t in selected_indices:
                    file_name = f"batch{batch_idx}_sample{b}_frame{t:03d}.html"
                    save_path = os.path.join(vis_dir, file_name)
                    
                    plot_skeleton(
                        gt_joints=sample_gt[t],
                        pred_joints=sample_pred[t],
                        edges=vis_edges,
                        frame_id=t,
                        out_html=save_path
                    )
            
            has_visualized = True # 标记为已完成，后续 Batch 不再画图
            print(f"[Eval] Visualization saved to {vis_dir}")

    return total_mpjpe / max(total_samples, 1)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    save_path = "./checkpoints_cross_mmfi"
    
    reader = DataReader(cache_path="state_cacheB.pt") 
    train_dataset, val_dataset = make_dataset(dataset_root, config)

    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])
    
   
    # 2. 实例化完整模型
    model = RadarPoseRefiner(
        in_channels=6,         # (x,y,z,d,s,time)
        radar_embed_dim=256,   # Encoder 输出维度
        num_latents=64,
        num_joints=27,
        parent_list=PARENT,
        refine_embed_dim=512   # Refiner 内部维度
    ).to(device)

    # 3. 尝试加载预训练 Stage 1 (如果存在)
    # 如果没有，Refiner 训练会极其困难，建议先训练 Coarse 部分
    model.load_pretrained_stage1("./coarse_head_best.pt", freeze=True)

    # # 4. 优化器 (只优化 Refiner)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    # state_dict = torch.load("./refiner_epoch_50.pt", map_location=device, weights_only=True)

    # 4. 将权重载入模型
    # strict=True (默认) 会严格检查每一个 key 是否匹配
    # 如果报错 "Missing keys" 或 "Unexpected keys"，请看后面的【常见问题】
    # msg = model.load_state_dict(state_dict, strict=True)
    # print("Load result:", msg)

    # 5. 切换到评估模式
    # 这非常重要！它会固定 BatchNorm 和 Dropout，否则推理结果会也是随机的
    # model.eval()
    
    # 5. 训练
    for epoch in range(1, 102):
        model.train() # 设置 dropout 状态
        epoch_loss = 0.0
        samples = 0
        
        for batch in train_loader:
            radar_seq, skeleton_seq = batch
            radar_seq = radar_seq.to(device).float()     # [B, T, 128, 6]
            skeleton_seq = skeleton_seq.to(device).float() # [B, T, 27, 3]
            
            B, T = radar_seq.shape[:2]
            
            # --- 展平序列为 Batch 进行训练 ---
            # 因为我们的模型是 SingleFrame，可以直接把 T 维度并入 B 维度
            # 这样效率比写 for t in range(T) 高得多
            radar_flat = radar_seq.view(B * T, 128, 6)   # [B*T, 128, 6]
            x_1_flat = skeleton_seq.view(B * T, 27, 3)   # [B*T, 27, 3] (GT)
            
            # 获取 x_0 (无需梯度)
            with torch.no_grad():
                _, x_0_flat = model.get_coarse_prior(radar_flat)
            
            # Flow Matching 采样
            t_rand = torch.rand(B * T, 1, device=device)
            t_broad = t_rand.unsqueeze(-1) # [B*T, 1, 1]
            
            # 插值得到 x_t
            x_t_flat = (1 - t_broad) * x_0_flat + t_broad * x_1_flat
            
            # 目标速度
            v_target = x_1_flat - x_0_flat
            
            # 预测
            v_pred = model(x_t_flat, t_rand, radar_flat)
            
            loss = F.mse_loss(v_pred, v_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * (B * T)
            samples += (B * T)
            
        avg_loss = epoch_loss / samples
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.6f}")
        
        # 简单验证
        if epoch % 20 == 0:
            val_mpjpe = evaluate_sequence(
            dataloader=val_loader,
            reader=reader,
            model=model,
            device=device, 
            vis_dir=f"{save_path}/vis_results/epoch_{epoch}",  # 每个 epoch 存一个文件夹
            vis_edges=EDGES_27,                      # 你的骨骼连接列表
            num_vis_samples=1,                       # 只画前2个动作样本
           )

            print(f"Epoch {epoch} | Val MPJPE: {val_mpjpe:.2f} mm")
            
            # 保存
            torch.save(model.state_dict(), f"{save_path}/refiner_epoch_{epoch}.pt")

if __name__ == "__main__":
    main()