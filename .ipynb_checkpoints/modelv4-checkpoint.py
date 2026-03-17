from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import DataReader, RadarDiffusionDataset
from flowmodels import collate_fn_for_cross_modal
from models import get_adj_matrix

EDGES_27 = [
    (0,1), (1,2), (2,3), (3,26), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (8,10),
    (3,11), (11,12), (12,13), (13,14), (14,15), (15,16), (15,17),
    (0,18), (18,19), (19,20), (20,21), (0,22), (22,23), (23,24), (24,25)
]

class RadarEncoder(nn.Module):
    def __init__(self, radar_feat_dim=5, hidden_dim=256):
        super().__init__()

        # ---------- 1. 空间分支（geometry + semantics） ----------
        self.spatial_encoder = nn.Sequential(
            nn.Linear(radar_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )

        self.spatial_attn = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # ---------- 2. Doppler 分支（motion-specific） ----------
        self.doppler_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim)
        )

        self.doppler_attn = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # ---------- 3. 融合 + 时序 ----------
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, radar_seq, h0=None):
        """
        radar_seq: [B, T, P, 5] = (x,y,z,doppler,snr)
        """

        B, T, P, _ = radar_seq.shape

        doppler = radar_seq[..., 3:4]   # [B, T, P, 1]
        snr = radar_seq[..., 4:5]       # [B, T, P, 1]

        # ================= 空间分支 =================
        spatial_feats = self.spatial_encoder(radar_seq)  # [B,T,P,H]

        spatial_attn_in = torch.cat([spatial_feats, snr], dim=-1)
        spatial_frame = torch.sum(
            spatial_feats,
            dim=2
        )  # [B,T,H]
        # spatial_w = F.softmax(
        #     self.spatial_attn(spatial_attn_in).squeeze(-1),
        #     dim=2
        # )

        # spatial_frame = torch.sum(
        #     spatial_feats * spatial_w.unsqueeze(-1),
        #     dim=2
        # )  # [B,T,H]

        # ================= Doppler 分支 =================
        doppler_feats = self.doppler_encoder(doppler)  # [B,T,P,H]

        doppler_attn_in = torch.cat([doppler_feats, snr], dim=-1)
        doppler_w = F.softmax(
            self.doppler_attn(doppler_attn_in).squeeze(-1),
            dim=2
        )

        doppler_frame = torch.sum(
            doppler_feats * doppler_w.unsqueeze(-1),
            dim=2
        )  # [B,T,H]

        # ================= 融合 =================
        frame_feats = self.fusion(
            torch.cat([spatial_frame, doppler_frame], dim=-1)
        )  # [B,T,H]

        # ================= 时序（tracking） =================
        z_seq, h_n = self.gru(frame_feats, h0)

        return z_seq, h_n


class ST_GraphFlowModel(nn.Module):
    def __init__(self, num_joints, adj_matrix, hidden_dim=256):
        super().__init__()
        self.num_joints = num_joints
        self.register_buffer('adj', adj_matrix)

        # ----------- Joint & time embedding -----------
        self.joint_embed = nn.Linear(3, hidden_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ----------- Radar projection（可学习对齐）-----------
        self.radar_proj = nn.Linear(hidden_dim, hidden_dim)

        # ----------- Spatial GCN -----------
        self.gcn_layer = nn.Linear(hidden_dim, hidden_dim)

        # ----------- Temporal Conv -----------
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        # ----------- Output -----------
        self.node_out = nn.Linear(hidden_dim, 3)

    def forward(self, xt_seq, tau_seq, z_seq, h_n):
        """
        xt_seq: [B, T, J*3]
        tau_seq: [B, T, 1]
        z_seq: [B, T, H]      radar frame-level features
        h_n: [1, B, H]        radar track-level latent
        """

        B, T, _ = xt_seq.shape
        J = self.num_joints
        H = z_seq.shape[-1]

        # ----------- reshape joints -----------
        x = xt_seq.view(B, T, J, 3)

        # ----------- joint embedding -----------
        h = self.joint_embed(x)          # [B, T, J, H]

        # ----------- time embedding -----------
        t_emb = self.time_mlp(tau_seq).unsqueeze(2)  # [B, T, 1, H]
        h = h + t_emb

        # ======================================================
        # =============== Radar conditioning (方案 1) ==========
        # ======================================================

        # ---- frame-wise radar conditioning ----
        z_seq_proj = self.radar_proj(z_seq)          # [B, T, H]
        h = h + z_seq_proj[:, :, None, :]             # [B, T, J, H]

        # ---- track-wise global radar conditioning ----
        h_radar = h_n.squeeze(0)                      # [B, H]
        h_radar = self.radar_proj(h_radar)            # [B, H]
        h_radar = h_radar[:, None, None, :]           # [B,1,1,H]
        h_radar = h_radar.expand(B, T, J, H)          # [B,T,J,H]

        h = h + h_radar

        # ----------- Spatial GCN -----------
        # [J, J] @ [B, T, J, H] -> [B, T, J, H]
        h_gcn = torch.einsum('jk,btkh->btjh', self.adj, h)
        h = h + F.relu(self.gcn_layer(h_gcn))

        # ----------- Temporal Conv -----------
        # [B,T,J,H] -> [B*J, H, T]
        h_temp = h.permute(0, 2, 3, 1).reshape(B * J, H, T)
        h_temp = self.temporal_conv(h_temp)

        # back to [B, T, J, H]
        h = h_temp.view(B, J, H, T).permute(0, 3, 1, 2)

        # ----------- Output -----------
        v = self.node_out(h)   # [B, T, J, 3]

        return v.view(B, T, -1)

def calculate_bone_loss(pred_skeleton, gt_skeleton, edges):
    """
    pred_skeleton, gt_skeleton: [N, J, 3]  (physical space)
    """
    loss = 0.0
    for i, j in edges:
        pred_len = torch.norm(pred_skeleton[:, i] - pred_skeleton[:, j], dim=-1)
        gt_len = torch.norm(gt_skeleton[:, i] - gt_skeleton[:, j], dim=-1)
        loss += F.l1_loss(pred_len, gt_len)
    return loss / len(edges)


# -------------------------
# Bone direction loss
# -------------------------
def calculate_direction_loss(pred_skeleton, gt_skeleton, edges, eps=1e-6):
    loss = 0.0
    for i, j in edges:
        v_pred = pred_skeleton[:, i] - pred_skeleton[:, j]
        v_gt = gt_skeleton[:, i] - gt_skeleton[:, j]

        v_pred = v_pred / (torch.norm(v_pred, dim=-1, keepdim=True) + eps)
        v_gt = v_gt / (torch.norm(v_gt, dim=-1, keepdim=True) + eps)

        loss += (1 - (v_pred * v_gt).sum(dim=-1)).mean()
    return loss / len(edges)


# -------------------------
# Temporal smoothness loss (weak!)
# -------------------------
def calculate_temporal_loss(v_pred_seq):
    """
    v_pred_seq: [B, T, J*3]
    """
    if v_pred_seq.shape[1] < 2:
        return torch.tensor(0.0, device=v_pred_seq.device)

    accel = v_pred_seq[:, 1:] - v_pred_seq[:, :-1]
    return torch.mean(accel ** 2)


def calculate_mpjpe(pred_skeleton, gt_skeleton):

    if pred_skeleton.ndim == 3:
        B, T, D = pred_skeleton.shape

        pred = pred_skeleton.reshape(B, T, -1, 3)

        gt = gt_skeleton.reshape(B, T, -1, 3)

    else: 

        pred = pred_skeleton
        gt = gt_skeleton

    dist = torch.norm(pred - gt, dim=-1) # [B, T, J]
    return dist.mean().item()

# 在训练循环中
def fm_criterion(
    v_pred,
    v_gt,
    xt,
    tau,
    skeleton_seq,
    reader,
    edges,
    lambda_temp=0.01,
    lambda_bone=0.1,
    lambda_dir=0.1,
    use_temp=True,
):
    """
    v_pred: [B, T, J*3]
    v_gt:   [B, T, J*3]
    xt:     [B, T, J*3]
    tau:    [B, T, 1]
    skeleton_seq: [B, T, J, 3] (normalized GT)
    """

    B, T, D = v_pred.shape
    J = D // 3

    # ----------- FM loss (core) -----------
    loss_fm = F.mse_loss(v_pred, v_gt)

    # ----------- reconstruct x1 -----------
    x1_pred = xt + tau * v_pred
    x1_pred = x1_pred.view(B, T, J, 3)

    x1_real_pred = reader.denormalize_pointcloud(x1_pred)
    x1_real_gt = reader.denormalize_pointcloud(skeleton_seq)

    # ----------- physical losses -----------
    loss_bone = calculate_bone_loss(
        x1_real_pred.view(-1, J, 3),
        x1_real_gt.view(-1, J, 3),
        edges,
    )

    loss_dir = calculate_direction_loss(
        x1_real_pred.view(-1, J, 3),
        x1_real_gt.view(-1, J, 3),
        edges,
    )

    # ----------- temporal loss (weak regularizer) -----------
    if use_temp:
        loss_temp = calculate_temporal_loss(v_pred)
    else:
        loss_temp = torch.tensor(0.0, device=v_pred.device)

    total_loss = (
        loss_fm
        + lambda_temp * loss_temp
        + lambda_bone * loss_bone
        + lambda_dir * loss_dir
    )

    loss_dict = {
        "fm": loss_fm.item(),
        "bone": loss_bone.item(),
        "dir": loss_dir.item(),
        "temp": loss_temp.item(),
    }

    return total_loss, loss_dict

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

def train_one_epoch(dataloader, reader, radar_encoder, fm_model, optimizer, device, edges):
    radar_encoder.train()
    fm_model.train()

    total_loss = 0.0
    total_mjpe = 0.0
    total_samples = 0

    for batch in dataloader:
        radar_seq, skeleton_seq = batch
        radar_seq = radar_seq.to(device).float()
        skeleton_seq = skeleton_seq.to(device).float()
        B, T, J, _ = skeleton_seq.shape

        optimizer.zero_grad()

        # 1. Radar encoder
        z_seq, h_n = radar_encoder(radar_seq)

        # CFG
        if torch.rand(1) < 0.15:
            z_seq = torch.zeros_like(z_seq)

        # 2. Flow matching construction
        x1 = skeleton_seq.view(B, T, -1)
        x0 = torch.randn_like(x1) * 0.2

        tau = torch.rand(B, 1, 1, device=device).repeat(1, T, 1)
        xt = (1 - tau) * x0 + tau * x1
        v_gt = x1 - x0

        # 3. Predict velocity
        v_pred = fm_model(xt, tau, z_seq, h_n)

        # 4. FM loss (ONLY training objective)
        loss_fm = F.mse_loss(v_pred, v_gt)

        # 5. Physical regularization
        x1_pred = (xt + (1 - tau) * v_pred).view(B, T, J, 3)
        x1_real_pred = reader.denormalize_pointcloud(x1_pred)
        x1_real_gt = reader.denormalize_pointcloud(skeleton_seq)
        print("v_gt range:", v_gt.min().item(), v_gt.max().item())
        print("v_pred range:", v_pred.min().item(), v_pred.max().item())


        loss_bone = calculate_bone_loss(
            x1_real_pred.view(-1, J, 3),
            x1_real_gt.view(-1, J, 3),
            edges
        )

        total_batch_loss = loss_fm + 0.5 * loss_bone

        # 6. Backprop
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(fm_model.parameters(), 1.0)
        optimizer.step()

        # 7. MJPE (metric ONLY)
        with torch.no_grad():
            mjpe = calculate_mpjpe(x1_real_pred, x1_real_gt)

        # 8. Accumulate
        total_loss += total_batch_loss.item() * B
        total_mjpe += mjpe * B
        total_samples += B

    return {
        "loss": total_loss / total_samples,
        "mjpe": total_mjpe / total_samples
    }

@torch.no_grad()
def validate(dataloader, reader, radar_encoder, fm_model, device, num_joints=27):
    radar_encoder.eval()
    fm_model.eval()
    total_mpjpe = 0.0

    if len(dataloader.dataset) == 0:
        print("Warning: Validation dataset is empty!")
        return 0.0

    for batch in dataloader:
        radar_seq, skeleton_seq = batch
        radar_seq = radar_seq.to(device).float()
        skeleton_seq = skeleton_seq.to(device).float()  # [B, T, J, 3]

        # 生成预测
        pred_skeleton = generate_with_cfg(
            radar_seq, fm_model, radar_encoder, device,
            w=1, steps=100, num_joints=num_joints
        )

        # 反归一化
        pred_skeleton_real = reader.denormalize_pointcloud(pred_skeleton.cpu())
        gt_skeleton_real = reader.denormalize_pointcloud(skeleton_seq.cpu())

        # 计算 MPJPE
        mpjpe = calculate_mpjpe(pred_skeleton_real, gt_skeleton_real)
        total_mpjpe += mpjpe * radar_seq.shape[0]

    final_mpjpe = total_mpjpe / len(dataloader.dataset)
    return final_mpjpe


@torch.no_grad()
def generate_with_cfg(radar_seq, fm_model, radar_encoder, device, w=1, steps=100, num_joints=27):
    """
    radar_seq: [B, T, P, 5]
    返回: [B, T, J, 3]
    """
    B, T, P, _ = radar_seq.shape

    # Radar encoder
    z_cond, h_cond = radar_encoder(radar_seq)
    z_null = torch.zeros_like(z_cond)
    h_null = torch.zeros_like(h_cond)

    dt = 1.0 / steps
    # 初始噪声与训练时一致
    xt = torch.randn((B, T, num_joints * 3), device=device) * 0.2

    for s in range(steps):
        tau = torch.full((B, T, 1), s * dt, device=device)

        # unconditioned
        v_uncond = fm_model(xt, tau, z_null, h_null)

        # conditioned
        v_cond = fm_model(xt, tau, z_cond, h_cond)

        # CFG 引导
        v_final = v_uncond + w * (v_cond - v_uncond)

        # 更新 xt
        xt = xt + v_final * dt

    return xt.view(B, T, num_joints, 3)


if __name__ == "__main__":
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
    batch_size = 16
    
    reader = DataReader(cache_path="state_cache.pt")
    dataset = RadarDiffusionDataset(root_dir='../dataset', reader=reader, sample_level='sequence', num_joints=num_joints)
    train_dataloader = DataLoader(dataset.get_train_set(), batch_size=batch_size, shuffle=True, collate_fn=collate_fn_for_cross_modal)
    val_dataloader = DataLoader(dataset.get_val_set(), batch_size=batch_size, shuffle=False, collate_fn=collate_fn_for_cross_modal)
    test_dataloader = DataLoader(dataset.get_test_set(), batch_size=batch_size, shuffle=False, collate_fn=collate_fn_for_cross_modal)    

    # 模型与优化器
    radar_encoder = RadarEncoder(radar_feat_dim).to(device)
    fm_model = ST_GraphFlowModel(num_joints, adj_matrix, latent_dim).to(device)

    optimizer = optim.Adam(list(radar_encoder.parameters()) + list(fm_model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

    # 训练循环
    for epoch in range(1, 200):
        train_logs = train_one_epoch(
            train_dataloader, 
            reader,
            radar_encoder, 
            fm_model,
            optimizer, 
            device,
            EDGES_27  # <--- 加上这个实参
        )
    
        train_loss = train_logs["loss"]
        train_mjpe = train_logs["mjpe"]
    
        val_mpjpe = validate(
            val_dataloader, reader,
            radar_encoder, fm_model,
            device
        )
    
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Train MJPE: {train_mjpe:.4f} mm | "
            f"Val MJPE: {val_mpjpe:.4f} mm"
        )
    
        # --------------------------
        # 保存最优模型
        # --------------------------
        if val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = val_mpjpe
            torch.save({
                "epoch": epoch,
                "best_val_mpjpe": best_val_mpjpe,
                "radar_encoder_state_dict": radar_encoder.state_dict(),
                "fm_model_state_dict": fm_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, best_ckpt_path)
    
            print(f"✅ Best model saved at epoch {epoch}, Val MPJPE = {best_val_mpjpe:.4f} mm")
    
        # --------------------------
        # 学习率调度
        # --------------------------
        scheduler.step(val_mpjpe)


    # ckpt = torch.load(best_ckpt_path, map_location=device)

    # radar_encoder.load_state_dict(ckpt["radar_encoder_state_dict"])
    # fm_model.load_state_dict(ckpt["fm_model_state_dict"])

    # print(f"Loaded best model from epoch {ckpt['epoch']} "
    #     f"(Val MPJPE = {ckpt['best_val_mpjpe']:.4f} m)")
    # sample_radar, _ = next(iter(val_dataloader))
    # gen_mpjpe = evaluate_generation(
    #     val_dataloader,
    #     reader,
    #     radar_encoder,
    #     fm_model,
    #     device,
    #     w=1.5,
    #     vis_dir=save_dir+"/vis",
    #     num_vis_frames=8
    # )

    # print(f"Generation MPJPE: {gen_mpjpe:.4f} m")