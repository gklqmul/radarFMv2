import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import plotly.graph_objects as go
import random
from datetime import datetime
from tqdm.auto import tqdm

# 假设 dataset.py 在同级目录下
from dataset import DataReader, RadarDiffusionDataset
from modelv6 import collate_fn_for_cross_modal


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

# EDGES = [
#     # Right Leg (Hip -> RHip -> RKnee -> RAnkle)
#     (0, 1), (1, 2), (2, 3),
    
#     # Left Leg (Hip -> LHip -> LKnee -> LAnkle)
#     (0, 4), (4, 5), (5, 6),
    
#     # Spine & Head (Hip -> Spine -> Neck -> Head -> Site)
#     (0, 7), (7, 8), (8, 9), (9, 10),
    
#     # Left Arm (Neck -> LShoulder -> LElbow -> LWrist)
#     (8, 11), (11, 12), (12, 13),
    
#     # Right Arm (Neck -> RShoulder -> RElbow -> RWrist)
#     (8, 14), (14, 15), (15, 16)
# ]

# PARENT = {
#     # Right Leg
#     1: 0,
#     2: 1,
#     3: 2,
    
#     # Left Leg
#     4: 0,
#     5: 4,
#     6: 5,
    
#     # Spine & Head
#     7: 0,
#     8: 7,
#     9: 8,
#     10: 9,
    
#     # Left Arm
#     11: 8,
#     12: 11,
#     13: 12,
    
#     # Right Arm
#     14: 8,
#     15: 14,
#     16: 15
# }

def calculate_mpjpe(pred_skeleton, gt_skeleton):
    """ 计算平均每个关节位置误差 (单位应与输入一致，通常是 mm) """
    # pred_skeleton: [N, J, 3]
    # gt_skeleton:   [N, J, 3]
    
    # 直接计算欧氏距离
    dist = torch.norm(pred_skeleton - gt_skeleton, dim=-1) # [N, J]
    
    # 对所有样本和所有关节求平均
    return dist.mean().item()

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
        

def bone_length_loss(pred_lengths, gt_joints, parent):
    """
    pred_lengths: [B, J]      (root=0 可忽略)
    gt_joints:    [B, J, 3]
    parent:       list[int]  parent[0] = -1
    """
    B, J, _ = gt_joints.shape
    device = gt_joints.device

    loss = 0.0
    count = 0

    for j in range(1, J):  # skip root
        p = parent[j]
        if p < 0:
            continue

        gt_len = torch.norm(
            gt_joints[:, j] - gt_joints[:, p],
            dim=-1
        )  # [B]

        pred_len = pred_lengths[:, j - 1]

        loss += F.mse_loss(pred_len, gt_len)
        count += 1

    return loss / max(count, 1)


def position_loss(pred, gt):
    return F.mse_loss(pred, gt)


def hierarchy_direction_loss(pred, gt, parent):
    loss = 0.0
    for j in range(1, pred.shape[1]):
        p = parent[j]
        vp = pred[:, j] - pred[:, p]
        vg = gt[:, j] - gt[:, p]

        vp = F.normalize(vp, dim=-1)
        vg = F.normalize(vg, dim=-1)

        loss += (1 - (vp * vg).sum(dim=-1)).mean()
    return loss / (pred.shape[1] - 1)


def offset_direction_loss(pred_offsets, gt_joints, parent):
    """
    pred_offsets: [B, J, 3]
    gt_joints:    [B, J, 3]
    """
    loss = 0.0
    count = 0

    for j in range(1, gt_joints.shape[1]):
        p = parent[j]
        if p < 0:
            continue

        gt_offset = gt_joints[:, j] - gt_joints[:, p]
        pred_offset = pred_offsets[:, j]

        gt_dir = F.normalize(gt_offset, dim=-1)
        pred_dir = F.normalize(pred_offset, dim=-1)

        loss += 1.0 - (gt_dir * pred_dir).sum(dim=-1).mean()
        count += 1

    return loss / max(count, 1)



def train_one_epoch_coarse(dataloader, reader, radar_encoder, coarse_head, optimizer, device):
    """
    单帧 coarse skeleton 训练
    所有计算都在归一化空间
    """
    radar_encoder.train()
    coarse_head.train()
    total_loss = 0.0
    num_samples = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Eval Batches")):
        radar_seq = batch['radar_cond']
        skeleton_seq = batch['pointcloud']

        radar_seq = radar_seq.to(device).float()     # [B, T, 128, 6]
        skeleton_seq = skeleton_seq.to(device).float() # [B, T, 27, 3]
            
        B, T = radar_seq.shape[:2]
        radar_flat = radar_seq.view(B * T, 128, 6)   # [B*T, 128, 6]
        gt_joints = skeleton_seq.view(B * T, 27, 3)   # [B*T, 27, 3] (GT)

        # if batch_idx == 0:
        #     data_max = gt_joints.abs().max().item()
        #     print(f"\n[DEBUG CHECK] Batch 0 GT Max Value: {data_max:.4f}")
            
        #     # 如果最大值 > 10，说明实际上就是毫米单位（无论你是否调用了归一化）
        #     # if data_max > 10.0:
        #     #     print(f"⚠️ 检测到输入数据为毫米级 (>10)，自动启用 /1000 缩放！")
        #     #     AUTO_SCALER = 1000.0
        #     # else:
        #     #     print(f"✅ 输入数据范围正常 (归一化空间)。")
    
        optimizer.zero_grad()
        z_seq = radar_encoder(radar_flat)
        z = z_seq.mean(dim=1)
            
        pred_joints, pred_offsets, pred_lengths = coarse_head(z)

        # print(f"GT min/max: {gt_joints.min().item()}, {gt_joints.max().item()}")
        # print(f"Pred min/max: {pred_joints.min().item()}, {pred_joints.max().item()}")

        loss_pos = F.mse_loss(pred_joints, gt_joints)
    
        # =========================
        # Loss 2: offset direction
        # =========================
        loss_dir = offset_direction_loss(
            pred_offsets,
            gt_joints, 
            PARENT
        )
    
        # =========================
        # Loss 3: bone length
        # =========================
        loss_len = bone_length_loss(
            pred_lengths.squeeze(-1),
            gt_joints,
            PARENT
        )
    
        # =========================
        # Total
        # =========================
        # print("loss dir", loss_dir)
        # print("loss_len", loss_len)
        # print("loss_pos", loss_pos)
        loss = (
            loss_pos
            # + 0.5 * loss_dir
            # + 0.1 * loss_len
        )
    
        loss.backward()

        torch.nn.utils.clip_grad_norm_(list(radar_encoder.parameters()) + list(coarse_head.parameters()), max_norm=1.0)
        optimizer.step()
    
        total_loss += loss.item() * gt_joints.shape[0]
        num_samples += gt_joints.shape[0]
        
        # print("total_loss", total_loss, num_samples)
        
    print("all_loss", total_loss/ num_samples)
    return total_loss / num_samples


@torch.no_grad()
def evaluate_coarse_generation(dataloader, reader, radar_encoder, coarse_head, device):
    """
    验证 / 计算 MPJPE
    输出 MPJPE 单位为 mm (反归一化)
    """
    radar_encoder.eval()
    coarse_head.eval()
    total_mpjpe = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Eval Batches")):
        radar_seq = batch['radar_cond']
        skeleton_seq = batch['pointcloud']
        radar_seq = radar_seq.to(device).float()       # [B, T, 128, 6]
        skeleton_seq = skeleton_seq.to(device).float()

        B, T = radar_seq.shape[:2]
        radar_flat = radar_seq.view(B * T, 128, 6)   # [B*T, 128, 6]
        gt_skeleton = skeleton_seq.view(B * T, 27, 3)

        # ---- 前向（归一化空间）----
        z_seq = radar_encoder(radar_flat)
        z = z_seq.mean(dim=1)   
        pred_skeleton, pred_offsets, pred_lengths = coarse_head(z)     # [B, J, 3]

        # ---- 反归一化回物理空间 ----

        # pred_mm_flat = reader.denormalize_pointcloud(pred_skeleton.cpu())
        # gt_mm_flat   = reader.denormalize_pointcloud(gt_skeleton.cpu())
        pred_mm_flat = pred_skeleton.cpu() * 1000
        gt_mm_flat = gt_skeleton.cpu() * 1000
        pred_root = pred_mm_flat[:, 0:1, :]
        gt_root   = gt_mm_flat[:, 0:1, :]
        
        pred_mm_rel = pred_mm_flat - pred_root
        gt_mm_rel   = gt_mm_flat - gt_root

        # ---- MPJPE ----
        mpjpe = calculate_mpjpe(pred_mm_rel, gt_mm_rel)
        total_mpjpe += mpjpe * gt_skeleton.shape[0]
        total_samples += gt_skeleton.shape[0]
        # print("total_mpjpe",total_mpjpe, "total_samples",total_samples)
        
    return total_mpjpe / total_samples

def load_coarse_model(
    ckpt_path,
    radar_encoder,
    coarse_head,
    device
):
    ckpt = torch.load(ckpt_path, map_location=device)

    radar_encoder.load_state_dict(ckpt["radar_encoder_state_dict"])
    coarse_head.load_state_dict(ckpt["coarse_head_state_dict"])

    radar_encoder.eval()
    coarse_head.eval()

    print(f"Loaded coarse model from epoch {ckpt['epoch']}, "
          f"best MPJPE = {ckpt['best_val_mpjpe']:.2f} mm")

def plot_skeleton(gt_joints, pred_joints, edges, frame_id=0, out_html=None):
    """
    gt_joints: np.array (V,3)  ground-truth joints
    pred_joints: np.array (V,3) predicted joints
    edges: list of (i,j) tuples
    frame_id: int
    out_html: str or None
    """
    gt_joints   -= gt_joints[0]
    pred_joints -= pred_joints[0]
    def make_bone_lines(joints, bones):
        xs, ys, zs = [], [], []
        for (i,j) in bones:
            xs += [float(joints[i,0]), float(joints[j,0]), None]
            ys += [float(joints[i,1]), float(joints[j,1]), None]
            zs += [float(joints[i,2]), float(joints[j,2]), None]
        return xs, ys, zs

    gt_scatter = go.Scatter3d(
        x=gt_joints[:,0], y=gt_joints[:,1], z=gt_joints[:,2],
        mode='markers', marker=dict(size=4, color='blue'), name='GT joints'
    )
    pred_scatter = go.Scatter3d(
        x=pred_joints[:,0], y=pred_joints[:,1], z=pred_joints[:,2],
        mode='markers', marker=dict(size=4, color='red'), name='Recon joints'
    )

    gt_xs, gt_ys, gt_zs = make_bone_lines(gt_joints, edges)
    pred_xs, pred_ys, pred_zs = make_bone_lines(pred_joints, edges)

    gt_bones = go.Scatter3d(x=gt_xs, y=gt_ys, z=gt_zs, mode='lines',
                            line=dict(color='blue', width=3), name='GT bones')
    pred_bones = go.Scatter3d(x=pred_xs, y=pred_ys, z=pred_zs, mode='lines',
                              line=dict(color='red', width=3), name='Recon bones')

    layout = go.Layout(
        title=f"Skeleton Comparison (frame {frame_id})",
        scene=dict(aspectmode='auto'),
        legend=dict(x=0.02, y=0.98)
    )

    fig = go.Figure(data=[gt_bones, gt_scatter, pred_bones, pred_scatter], layout=layout)
    if out_html:
        fig.write_html(out_html, auto_open=False)
        print("Saved interactive skeleton comparison to:", out_html)
    return fig
    
@torch.no_grad()
def visualize_coarse_on_val(
    dataloader,
    reader,
    radar_encoder,
    coarse_head,
    device,
    out_dir="vis_coarse_eval",
    num_samples=5
):
    import os
    os.makedirs(out_dir, exist_ok=True)

    radar_encoder.eval()
    coarse_head.eval()

    sample_count = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Eval Batches")):
        radar_seq = batch['radar_cond']
        skeleton_seq = batch['pointcloud']
        radar_seq = radar_seq.to(device).float()       # [B, T, 128, 6]
        skeleton_seq = skeleton_seq.to(device).float()

        B, T = radar_seq.shape[:2]
        radar_flat = radar_seq.view(B * T, 128, 6)   # [B*T, 128, 6]
        gt_skeleton = skeleton_seq.view(B * T, 27, 3)
        
        z_seq = radar_encoder(radar_flat)  
        z = z_seq.mean(dim=1)
        # z = z_seq.max(dim=1)[0] # [B, D]
        print(f"z variance: {z.var().item()}")
        pred_joints, _, _ = coarse_head(z)

        # pred_mm_flat = reader.denormalize_pointcloud(pred_joints.cpu())
        # gt_mm_flat   = reader.denormalize_pointcloud(gt_skeleton.cpu())
        pred_mm_flat = pred_joints.cpu() * 1000
        gt_mm_flat = gt_skeleton.cpu() * 1000
        
        pred_root = pred_mm_flat[:, 0:1, :]
        gt_root   = gt_mm_flat[:, 0:1, :]
        
        pred_mm_rel = pred_mm_flat - pred_root
        gt_mm_rel   = gt_mm_flat - gt_root
        
        variance = torch.var(pred_mm_flat, dim=0).mean()

        print(f"Prediction Variance: {variance.item():.6f}")
        
        if variance < 1000:
            print("🚨 警告：模型发生了 Mode Collapse (模式坍塌)！它正在输出平均姿态！")
        else:
            print("✅ 正常：模型对不同的雷达输入有不同的响应。")

        B = pred_joints.shape[0]

        k_per_batch = 2  

        indices = list(range(B))
        random.shuffle(indices)
        
        for b in indices[:k_per_batch]:
            if sample_count >= num_samples:
                return
        
            pred = pred_joints[b].cpu().numpy()
            gt   = gt_skeleton[b].cpu().numpy()
        
            out_html = os.path.join(
                out_dir,
                f"sample_{sample_count:03d}.html"
            )
        
            plot_skeleton(
                gt_joints=gt,
                pred_joints=pred,
                edges=EDGES_27,
                frame_id=sample_count,
                out_html=out_html
            )
        
            print(f"[VIS] saved: {out_html}")
            sample_count += 1


# def main_train_coarse():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("Using device:", device)
#     best_val_mpjpe = float("inf")

#     # ----------------- 数据 -----------------
#     reader = DataReader(cache_path="state_cache.pt")
#     dataset = RadarDiffusionDataset(root_dir='./dataset', reader=reader, sample_level='sequence', num_joints=27)
#     train_loader = DataLoader(dataset.get_train_set(), batch_size=4, shuffle=True, collate_fn=collate_fn_for_cross_modal)
#     val_loader   = DataLoader(dataset.get_val_set(), batch_size=4, shuffle=False, collate_fn=collate_fn_for_cross_modal)

#     # ----------------- 模型 -----------------
#     radar_feat_dim = 6
#     latent_dim = 256
#     num_joints = 27
    
#     radar_encoder = TimeAwareCompressedRadarEncoder(
#             in_channels=radar_feat_dim,
#             embed_dim=latent_dim,
#             num_latents=64
#         ).to(device)
#     coarse_head = CoarseSkeletonHead(latent_dim, num_joints, PARENT).to(device)
    
#     optimizer = torch.optim.Adam(
#         list(radar_encoder.parameters()) + list(coarse_head.parameters()), 
#         lr=1e-3
#     )

#     save_path = "./saveforever/coarse_head_best.pt"
#     # load_coarse_model(
#     #     "./coarse_head_best.pt",
#     #     radar_encoder,
#     #     coarse_head,
#     #     device
#     # )

   
#     # visualize_coarse_on_val(
#     #     val_loader,
#     #     reader,
#     #     radar_encoder,
#     #     coarse_head,
#     #     device,
#     #     out_dir="vis_coarse_eval",
#     #     num_samples=10
#     # )

#     # ----------------- 训练循环 -----------------
#     for epoch in range(1, 201):
#         train_loss = train_one_epoch_coarse(train_loader, reader, radar_encoder, coarse_head, optimizer, device)
#         val_mpjpe = evaluate_coarse_generation(val_loader, reader, radar_encoder, coarse_head, device)

#         print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val MPJPE: {val_mpjpe:.2f} mm")

#         if val_mpjpe < best_val_mpjpe:
#             best_val_mpjpe = val_mpjpe
#             torch.save({
#                 "epoch": epoch,
#                 "best_val_mpjpe": best_val_mpjpe,
#                 "radar_encoder_state_dict": radar_encoder.state_dict(),
#                 "coarse_head_state_dict": coarse_head.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#             }, save_path)
#             print(f"✅ Best coarse model saved (MPJPE = {best_val_mpjpe:.2f} mm)")
def main_train_coarse():
    assert torch.cuda.is_available(), "CUDA 不可用"
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")

    device = torch.device("cuda:0")
    best_val_mpjpe = float("inf")

    # ----------------- 模型 -----------------
    radar_feat_dim = 6
    latent_dim = 256
    num_joints = 27

    radar_encoder = TimeAwareCompressedRadarEncoder(
        in_channels=radar_feat_dim,
        embed_dim=latent_dim,
        num_latents=64
    )

    coarse_head = CoarseSkeletonHead(
        latent_dim,
        num_joints,
        PARENT
    )
    print("正在检查模型健康状况...")
    for name, param in coarse_head.named_parameters():
        if param.abs().max() > 10.0: # 正常初始化的权重绝对不会超过 2.0
            print(f"❌ 致命错误: 参数 {name} 已经被污染 (Max={param.abs().max()})！")
            print("请立刻重启 Python 内核 (Restart Kernel) 或检查是否加载了坏的 checkpoint。")
            return # 直接退出，别跑了
    print("✅ 模型权重健康，准备开始训练。")
    

    # ----------------- 数据 -----------------
    reader = DataReader(cache_path="state_cache.pt")
    dataset = RadarDiffusionDataset(
        root_dir='./dataset',
        reader=reader,
        sample_level='sequence',
        num_joints=27
    )

    train_loader = DataLoader(
        dataset.get_train_set(),
        batch_size=4 * num_gpus,   # ✅ 总 batch size
        shuffle=True,
        collate_fn=collate_fn_for_cross_modal,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset.get_val_set(),
        batch_size=4 * num_gpus,
        shuffle=False,
        collate_fn=collate_fn_for_cross_modal,
        num_workers=4,
        pin_memory=True
    )

    

    # ✅ DataParallel
    radar_encoder = torch.nn.DataParallel(radar_encoder).to(device)
    coarse_head   = torch.nn.DataParallel(coarse_head).to(device)

    optimizer = torch.optim.Adam(
        list(radar_encoder.parameters()) +
        list(coarse_head.parameters()),
        lr=1e-3
    )

    save_path = "./saveforever/coarse_head_best.pt"

    # ----------------- 训练循环 -----------------
    for epoch in range(1, 201):
        train_loss = train_one_epoch_coarse(
            train_loader,
            reader,
            radar_encoder,
            coarse_head,
            optimizer,
            device
        )

        val_mpjpe = evaluate_coarse_generation(
            val_loader,
            reader,
            radar_encoder,
            coarse_head,
            device
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MPJPE: {val_mpjpe:.2f} mm"
        )

        if val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = val_mpjpe
            torch.save({
                "epoch": epoch,
                "best_val_mpjpe": best_val_mpjpe,
                # ⚠️ 注意保存 module.state_dict()
                "radar_encoder_state_dict": radar_encoder.module.state_dict(),
                "coarse_head_state_dict": coarse_head.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, save_path)

            print(f"✅ Best coarse model saved (MPJPE = {best_val_mpjpe:.2f} mm)")


if __name__ == "__main__":
    main_train_coarse()
