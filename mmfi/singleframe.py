import os
import torch
import tqdm
import yaml
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import plotly.graph_objects as go
import random
from datetime import datetime

from mmfi_lib.mmfidataset import make_dataset, make_dataloader, DataReader

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
def calculate_mpjpe(pred_skeleton, gt_skeleton):

    """ 计算平均每个关节位置误差 """

    # 处理输入：支持 (B, T, D) 或 (B, T, J, 3) 格式

    if pred_skeleton.ndim == 3:

        B, T, D = pred_skeleton.shape

        pred = pred_skeleton.reshape(B, T, -1, 3)

        gt = gt_skeleton.reshape(B, T, -1, 3)
    else:  # ndim == 4
        pred = pred_skeleton
        gt = gt_skeleton

    dist = torch.norm(pred - gt, dim=-1) # [B, T, J]

    return dist.mean().item()

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

    for batch_idx, batch in enumerate(dataloader):
        radar_seq = batch['radar_cond']
        skeleton_seq = batch['pointcloud']
        radar_seq = radar_seq.to(device).float()     # [B, T, 128, 6]
        skeleton_seq = skeleton_seq.to(device).float() # [B, T, 17, 3]
            
        B, T = radar_seq.shape[:2]
        radar_flat = radar_seq.view(B * T, 128, 6)   # [B*T, 128, 6]
        gt_joints = skeleton_seq.view(B * T, 17, 3)   # [B*T, 17, 3] (GT)
    
        optimizer.zero_grad()
        z_seq = radar_encoder(radar_flat)   
        z = z_seq.mean(dim=1)# [B, D]
    
        pred_joints, pred_offsets, pred_lengths = coarse_head(z)
        # pred_joints  : [B, J, 3]
        # pred_offsets : [B, J, 3]
        # pred_lengths : [B, J, 1]
    
        # =========================
        # Loss 1: joint position
        # =========================
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
        loss = (
            loss_pos
            + 0.2 * loss_dir
            + 0.1 * loss_len
        )
    
        loss.backward()
        optimizer.step()
    
        total_loss += loss.item() * gt_joints.shape[0]
        num_samples += gt_joints.shape[0]

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

    for batch_idx, batch in enumerate(dataloader):
        radar_seq = batch['radar_cond']
        skeleton_seq = batch['pointcloud']
        radar_seq = radar_seq.to(device).float()       # [B, T, 128, 6]
        skeleton_seq = skeleton_seq.to(device).float()

        B, T = radar_seq.shape[:2]
        radar_flat = radar_seq.view(B * T, 128, 6)   # [B*T, 128, 6]
        gt_skeleton = skeleton_seq.view(B * T, 17, 3)

        # ---- 前向（归一化空间）----
        z_seq = radar_encoder(radar_flat)   
        z = z_seq.mean(dim=1)# [B, D]
        pred_skeleton, pred_offsets, pred_lengths = coarse_head(z)     # [B, J, 3]

        # ---- 反归一化回物理空间 ----
        pred_skeleton_mm = reader.denormalize_pointcloud(pred_skeleton.cpu())
        root_joint = pred_skeleton_mm[:, 0:1, :] 
        
        # 核心操作：所有点减去根节点
        pred_skeleton = pred_skeleton_mm - root_joint
        
        gt_skeleton_mm   = reader.denormalize_pointcloud(gt_skeleton.cpu())
        

        # ---- MPJPE ----
        mpjpe = calculate_mpjpe(pred_skeleton, gt_skeleton_mm)
        total_mpjpe += mpjpe * gt_skeleton.shape[0]
        total_samples += gt_skeleton.shape[0]

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

    for batch_idx, batch in enumerate(dataloader):
        radar_seq = batch['radar_cond']
        skeleton_seq = batch['pointcloud']
        radar_seq = radar_seq.to(device).float()       # [B, T, 128, 6]
        skeleton_seq = skeleton_seq.to(device).float()

        B, T = radar_seq.shape[:2]
        radar_flat = radar_seq.view(B * T, 128, 6)   # [B*T, 128, 6]
        gt_skeleton = skeleton_seq.view(B * T, 17, 3)
        
        z_seq = radar_encoder(radar_flat)   
        z = z_seq.mean(dim=1)# [B, D]
        print(f"z variance: {z.var().item()}")
        pred_joints, _, _ = coarse_head(z)

        
        pred_joints = reader.denormalize_pointcloud(pred_joints.cpu())
        root = pred_joints[:,0:1,:]
        pred_skeleton = pred_joints - root
        gt_skeleton = reader.denormalize_pointcloud(gt_skeleton.cpu())

        B = pred_joints.shape[0]

        k_per_batch = 2  

        indices = list(range(B))
        random.shuffle(indices)
        
        for b in indices[:k_per_batch]:
            if sample_count >= num_samples:
                return
        
            pred = pred_skeleton[b].numpy()
            gt   = gt_skeleton[b].numpy()
        
            out_html = os.path.join(
                out_dir,
                f"sample_{sample_count:03d}.html"
            )
        
            plot_skeleton(
                gt_joints=gt,
                pred_joints=pred,
                edges=EDGES,
                frame_id=sample_count,
                out_html=out_html
            )
        
            print(f"[VIS] saved: {out_html}")
            sample_count += 1


def main_train_coarse():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    best_val_mpjpe = float("inf")

    # ----------------- 数据 -----------------
    save_path = "./checkpoints_cross_mmfi"
    dataset_root = './combined'
    config_file = './config.yaml'
    with open(config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    reader = DataReader(cache_path="state_cacheB.pt") 
    train_dataset, val_dataset = make_dataset(dataset_root, config)

    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])

    # ----------------- 模型 -----------------
    radar_feat_dim = 6
    latent_dim = 256
    num_joints = 17
    
    radar_encoder = TimeAwareCompressedRadarEncoder(
            in_channels=radar_feat_dim,
            embed_dim=latent_dim,
            num_latents=64
        ).to(device)
    coarse_head = CoarseSkeletonHead(latent_dim, num_joints, PARENT).to(device)
    
    optimizer = torch.optim.Adam(
        list(radar_encoder.parameters()) + list(coarse_head.parameters()), 
        lr=1e-3
    )

    save_path = "./coarse_head_best.pt"
    load_coarse_model(
        "./coarse_head_best.pt",
        radar_encoder,
        coarse_head,
        device
    )

    # ---------------- 可视化 ----------------
    visualize_coarse_on_val(
        val_loader,
        reader,
        radar_encoder,
        coarse_head,
        device,
        out_dir="vis_coarse_eval",
        num_samples=10
    )

    # for epoch in range(1, 100):
    #     train_loss = train_one_epoch_coarse(train_loader, reader, radar_encoder, coarse_head, optimizer, device)
    #     val_mpjpe = evaluate_coarse_generation(val_loader, reader, radar_encoder, coarse_head, device)

    #     print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val MPJPE: {val_mpjpe:.2f} mm")

    #     if val_mpjpe < best_val_mpjpe:
    #         best_val_mpjpe = val_mpjpe
    #         torch.save({
    #             "epoch": epoch,
    #             "best_val_mpjpe": best_val_mpjpe,
    #             "radar_encoder_state_dict": radar_encoder.state_dict(),
    #             "coarse_head_state_dict": coarse_head.state_dict(),
    #             "optimizer_state_dict": optimizer.state_dict(),
    #         }, save_path)
    #         print(f"✅ Best coarse model saved (MPJPE = {best_val_mpjpe:.2f} mm)")


if __name__ == "__main__":
    main_train_coarse()
