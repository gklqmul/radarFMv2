import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from datetime import datetime

# 假设 dataset.py 在同级目录下
from dataset import DataReader, RadarDiffusionDataset
from flowmodels import collate_fn_for_cross_modal

EDGES_27 = [
    (0,1), (1,2), (2,3), (3,26), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (8,10),
    (3,11), (11,12), (12,13), (13,14), (14,15), (15,16), (15,17),
    (0,18), (18,19), (19,20), (20,21), (0,22), (22,23), (23,24), (24,25)
]
PARENT = {
    1:0, 2:1, 3:2,          # spine
    4:3, 5:4, 6:5, 7:6,     # right arm
    11:3,12:11,13:12,14:13,# left arm
    18:0,19:18,20:19,21:20,# right leg
    22:0,23:22,24:23,25:24 # left leg
}



class RadarEncoder(nn.Module):
    def __init__(self, radar_feat_dim=5, hidden_dim=256, latent_dim=256):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(radar_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        self.gru = nn.GRU(128, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, latent_dim)
        self.null_token = nn.Parameter(torch.randn(1, 1, latent_dim))

    def forward(self, radar_seq, force_null=False, p_uncond=0.0):
        B, T, P, F_dim = radar_seq.shape
        if force_null:
            return self.null_token.expand(B, T, -1)

        snr = radar_seq[..., -1:] 
        weights = torch.softmax(snr, dim=2) 
        
        point_feats = self.point_encoder(radar_seq.reshape(B*T, P, F_dim))
        point_feats = point_feats.reshape(B, T, P, 128)
        time_feats = (point_feats * weights).sum(dim=2) 
        
        z, _ = self.gru(time_feats) 
        z = self.fc(z) 

        if self.training and p_uncond > 0:
            mask = torch.rand(B, 1, 1, device=z.device) < p_uncond
            z = torch.where(mask, self.null_token, z)
        return z
        
class FlowMatchingModel(nn.Module):
    def __init__(self, num_joints, latent_dim, hidden_dim=512):
        super().__init__()
        self.num_joints = num_joints

        # ✅ Joint identity embedding
        self.joint_emb = nn.Embedding(num_joints, 32)

        self.input_dim = (
            num_joints * (3 + 32)  # xyz + identity
            + latent_dim
            + 1                    # tau
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_joints * 3)
        )

    def forward(self, X_t, t, z):
        B = X_t.shape[0]

        x = X_t.view(B, self.num_joints, 3)

        joint_ids = torch.arange(
            self.num_joints, device=X_t.device
        )
        joint_feat = self.joint_emb(joint_ids)            # [J,32]
        joint_feat = joint_feat.unsqueeze(0).repeat(B,1,1)

        x = torch.cat([x, joint_feat], dim=-1)            # [B,J,3+32]
        x = x.view(B, -1)

        inp = torch.cat([x, z, t], dim=-1)
        return self.mlp(inp)

def hierarchical_direction_loss(pred, gt, parent_map):
    """
    pred, gt: [B, J, 3]  (real space)
    """
    loss = 0.0
    count = 0

    for j, p in parent_map.items():
        v_pred = pred[:, j] - pred[:, p]
        v_gt   = gt[:, j]   - gt[:, p]

        v_pred = F.normalize(v_pred, dim=-1)
        v_gt   = F.normalize(v_gt, dim=-1)

        loss += 1.0 - (v_pred * v_gt).sum(dim=-1).mean()
        count += 1

    return loss / count
def get_adj_matrix(num_joints, edges, device):
    # 1. 创建单位矩阵（每个节点自环，关注自身特征）
    adj = torch.eye(num_joints, device=device)
    # 2. 根据骨骼连接填入 1
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1 
    # 3. 归一化（GCN的标准操作，防止特征值爆炸）
    d_inv_sqrt = torch.pow(adj.sum(1), -0.5)
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

def calculate_bone_loss(pred_skeleton, gt_skeleton, edges):
    """
    pred_skeleton: [B, J, 3] (物理空间 mm)
    """
    loss = 0
    for i, j in edges:
        pred_len = torch.norm(pred_skeleton[:, i] - pred_skeleton[:, j], dim=-1)
        gt_len = torch.norm(gt_skeleton[:, i] - gt_skeleton[:, j], dim=-1)
        loss += F.l1_loss(pred_len, gt_len)
    return loss / len(edges)

def calculate_temporal_loss(v_pred_seq, epsilon=0.01):

    """ 模块 3: Temporal Smoothness """

    if v_pred_seq.shape[1] < 2: return 0.0

    v_diff = v_pred_seq[:, 1:] - v_pred_seq[:, :-1]

    accel_norm = torch.norm(v_diff, dim=-1)

    loss_temp = torch.clamp(accel_norm - epsilon, min=0).mean()

    return loss_temp


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

def build_adj_matrix(num_joints, edges, device):
    adj = torch.eye(num_joints, device=device) 
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1
    
    row_sum = adj.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    
    return adj_normalized

def train_one_epoch(dataloader, reader, radar_encoder, fm_model, optimizer, device):
    radar_encoder.train()
    fm_model.train()
    total_loss = 0.0

    for batch in dataloader:
        radar_seq, skeleton_seq = batch
        radar_seq = radar_seq.to(device).float()
        skeleton_seq = skeleton_seq.to(device).float()
        B, T, J, _ = skeleton_seq.shape
        
        optimizer.zero_grad()
        z_motion = radar_encoder(radar_seq, p_uncond=0.1)

        loss_fm = 0.0
        loss_bone = 0.0
        loss_hier = 0.0

        for t_idx in range(T):
            x1_norm = skeleton_seq[:, t_idx].reshape(B, -1) 
            x0 = torch.randn_like(x1_norm) 
            tau = torch.rand(B, 1, device=device)
            xt = (1 - tau) * x0 + tau * x1_norm
            v_gt = x1_norm - x0 
            
            v_pred = fm_model(xt, tau, z_motion[:, t_idx])
            loss_fm += F.mse_loss(v_pred, v_gt)
            x1_pred_norm = (xt + v_pred).view(B, J, 3)
            x1_pred_real = reader.denormalize_pointcloud(x1_pred_norm)
            x1_gt_real   = reader.denormalize_pointcloud(skeleton_seq[:, t_idx])
            
            loss_bone += calculate_bone_loss(x1_pred_real, x1_gt_real, EDGES_27)
            loss_hier += hierarchical_direction_loss(
                x1_pred_real, x1_gt_real, PARENT
            )

        

        # 这里的 0.001 是权衡 MSE(归一化) 和 Bone(mm) 数量级的系数，需根据实际情况微调
        total_batch_loss = (
            loss_fm / T
            + 0.01 * loss_bone / T
            + 0.05 * loss_hier / T
        )
        
        total_batch_loss.backward()
        optimizer.step()
        total_loss += total_batch_loss.item() * B

    return total_loss / len(dataloader.dataset)

@torch.no_grad()

def validate(dataloader, reader, radar_encoder, fm_model, device):

    radar_encoder.eval()
    fm_model.eval()
    total_mpjpe = 0.0

    if len(dataloader.dataset) == 0:
        print("Warning: Validation dataset is empty!")
        return 0.0

    for batch in dataloader:
        radar_seq, skeleton_seq = batch
        radar_seq, skeleton_seq = radar_seq.to(device).float(), skeleton_seq.to(device).float()
        pred_skeleton = generate_with_cfg(radar_seq, fm_model, radar_encoder, device, w=1)

        B, T = pred_skeleton.shape[:2]
        num_joints = pred_skeleton.shape[-1] // 3
        pred_skeleton = pred_skeleton.view(B, T, num_joints, 3)
        pred_skeleton = reader.denormalize_pointcloud(pred_skeleton.cpu())
        skeleton_seq = reader.denormalize_pointcloud(skeleton_seq.cpu())
        mpjpe = calculate_mpjpe(pred_skeleton, skeleton_seq)
        total_mpjpe += mpjpe * radar_seq.shape[0]
    return total_mpjpe / len(dataloader.dataset)


@torch.no_grad()
def generate_with_cfg(radar_seq, fm_model, radar_encoder, device, w=2.5, steps=100):
    B, T, P, _ = radar_seq.shape
    num_joints = 27
    radar_encoder.eval()
    fm_model.eval()
    
    z_cond = radar_encoder(radar_seq, p_uncond=0)
    z_null = radar_encoder(radar_seq, force_null=True)

    skeleton_gen = []
    dt = 1.0 / steps

    for t_idx in range(T):
        # 每一帧独立生成
        xt = torch.randn((B, num_joints*3), device=device)
        for s in range(steps):
            tau = torch.full((B, 1), s * dt, device=device)
            v_uncond = fm_model(xt, tau, z_null[:, t_idx])
            v_cond = fm_model(xt, tau, z_cond[:, t_idx])
            v_final = v_uncond + w * (v_cond - v_uncond)
            xt = xt + v_final * dt
        skeleton_gen.append(xt.unsqueeze(1)) # [B, 1, J*3]

    return torch.cat(skeleton_gen, dim=1)

def evaluate_generation(
    dataloader,
    reader,
    radar_encoder,
    fm_model,
    device,
    w=1.5,
    vis_dir="vis_gen",
    num_vis_frames=5,
    sample_idx=0
):
    import os, numpy as np
    os.makedirs(vis_dir, exist_ok=True)

    radar_encoder.eval()
    fm_model.eval()

    total_mpjpe = 0.0
    total_samples = 0
    vis_done = False

    if len(dataloader.dataset) == 0:
        print("Warning: Evaluation dataset is empty!")
        return 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            radar_seq, skeleton_seq = batch
            radar_seq = radar_seq.to(device).float()
            skeleton_seq = skeleton_seq.to(device).float()

            # ===== 1. 生成 =====
            pred_skeleton = generate_with_cfg(
                radar_seq, fm_model, radar_encoder, device, w=w
            )

            # ===== 2. reshape =====
            B, T = pred_skeleton.shape[:2]
            J = pred_skeleton.shape[-1] // 3
            pred_skeleton = pred_skeleton.view(B, T, J, 3)

            # ===== 3. 反归一化 =====
            pred_skeleton = reader.denormalize_pointcloud(pred_skeleton.cpu())
            gt_skeleton   = reader.denormalize_pointcloud(skeleton_seq.cpu())

            # ===== 3.5 多帧可视化（只做一次）=====
            if not vis_done:
                frame_ids = np.linspace(
                    0, T - 1, num=num_vis_frames, dtype=int
                )

                for f in frame_ids:
                    out_html = os.path.join(
                        vis_dir,
                        f"batch{batch_idx}_sample{sample_idx}_frame{f}.html"
                    )

                    plot_skeleton(
                        gt_joints=gt_skeleton[sample_idx, f].numpy(),
                        pred_joints=pred_skeleton[sample_idx, f].numpy(),
                        edges=EDGES_27,
                        frame_id=f,
                        out_html=out_html
                    )

                vis_done = True

            # ===== 4. MPJPE =====
            mpjpe = calculate_mpjpe(pred_skeleton, gt_skeleton)
            total_mpjpe += mpjpe * B
            total_samples += B

    return total_mpjpe / total_samples


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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    best_val_mpjpe = float("inf")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = os.path.join("./checkpoint", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")
   
    best_ckpt_path = os.path.join(save_dir, "best_model.pt")
    adj_matrix = get_adj_matrix(27, EDGES_27, device)
    radar_feat_dim = 5
    num_joints = 27
    latent_dim = 256
    
    
    reader = DataReader(cache_path="state_cache.pt")
    dataset = RadarDiffusionDataset(root_dir='../dataset', reader=reader, sample_level='sequence', num_joints=num_joints)
    train_dataloader = DataLoader(dataset.get_train_set(), batch_size=32, shuffle=True, collate_fn=collate_fn_for_cross_modal)
    val_dataloader = DataLoader(dataset.get_val_set(), batch_size=32, shuffle=False, collate_fn=collate_fn_for_cross_modal)
    test_dataloader = DataLoader(dataset.get_test_set(), batch_size=32, shuffle=False, collate_fn=collate_fn_for_cross_modal)    

    # 模型与优化器
    radar_encoder = RadarEncoder(radar_feat_dim, latent_dim=latent_dim).to(device)
    fm_model = FlowMatchingModel(num_joints, latent_dim).to(device)
    # fm_model = GraphFlowModel(num_joints, latent_dim, adj_matrix).to(device)
    optimizer = optim.Adam(list(radar_encoder.parameters()) + list(fm_model.parameters()), lr=1e-3)

    # 训练循环
    for epoch in range(1, 200):
        train_loss = train_one_epoch(
            train_dataloader, reader,
            radar_encoder, fm_model,
            optimizer, device
        )

        val_mpjpe = validate(
            val_dataloader, reader,
            radar_encoder, fm_model,
            device
        )

        print(f"Epoch {epoch:03d} | Loss: {train_loss:.6f} | Val MPJPE: {val_mpjpe:.4f} m")

        if val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = val_mpjpe
            torch.save({
                "epoch": epoch,
                "best_val_mpjpe": best_val_mpjpe,
                "radar_encoder_state_dict": radar_encoder.state_dict(),
                "fm_model_state_dict": fm_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, best_ckpt_path)

            print(f"✅ Best model saved at epoch {epoch}, MPJPE = {best_val_mpjpe:.4f} m")


    ckpt = torch.load(best_ckpt_path, map_location=device)

    radar_encoder.load_state_dict(ckpt["radar_encoder_state_dict"])
    fm_model.load_state_dict(ckpt["fm_model_state_dict"])

    print(f"Loaded best model from epoch {ckpt['epoch']} "
        f"(Val MPJPE = {ckpt['best_val_mpjpe']:.4f} m)")
    sample_radar, _ = next(iter(val_dataloader))
    gen_mpjpe = evaluate_generation(
        test_dataloader,
        reader,
        radar_encoder,
        fm_model,
        device,
        w=1.5,
        vis_dir=save_dir+"/vis",
        num_vis_frames=8
    )

    print(f"Generation MPJPE: {gen_mpjpe:.4f} m")

if __name__ == "__main__":
    main()