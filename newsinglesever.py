import os
import torch
import torch.nn as nn
import torch.distributed as dist
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

EDGES_27 = [
    (0, 1), (1, 2), (2, 3), (3, 26), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (8, 10),
    (3, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (15, 17),
    (0, 18), (18, 19), (19, 20), (20, 21), (0, 22), (22, 23), (23, 24), (24, 25)
]

PARENT = {
    1: 0, 2: 1, 3: 2, 26: 3,
    4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 8,
    11: 3, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 15,
    18: 0, 19: 18, 20: 19, 21: 20,
    22: 0, 23: 22, 24: 23, 25: 24
}

# ==========================================
# 2. 模型组件定义
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class RadarStage1Model(nn.Module):
    """
    radar -> encoder -> temporal_adapter -> coarse_head/direct_head
    输入只有 6 维 + valid_mask
    """
    def __init__(
        self,
        in_channels=6,
        radar_embed_dim=256,
        num_latents=64,
        num_joints=27,
        parent_list=None,
        use_direct_head=False,
    ):
        super().__init__()
        assert in_channels == 6, "Expect radar_cond dim=6"

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

        self.use_direct_head = use_direct_head
        self.num_joints = num_joints

    def forward(self, radar_seq, valid_mask=None):
        """
        radar_seq:  [B, T, 128, 6]
        valid_mask: [B, T] bool, True=真实帧（强烈建议传）
        return: x0 [B, T, J, 3]
        """
        B, T, N, C = radar_seq.shape
        assert C == 6, f"radar_seq last dim should be 6, got {C}"

        radar_flat = radar_seq.view(B * T, N, C)

        # encoder: [BT,64,embed]
        z = self.encoder(radar_flat)

        # temporal adapter: mask padding frames
        z = self.temporal_adapter(z, B, T, valid_mask=valid_mask)

        z_global = z.mean(dim=1)  # [BT,embed]

        if self.use_direct_head:
            x0 = self.direct_head(z_global)       # [BT,J,3]
        else:
            x0, _, _ = self.coarse_head(z_global) # [BT,J,3]

        return x0.view(B, T, self.num_joints, 3)


# ==========================================
# 3. 损失函数与评估函数
# ==========================================
def stage1_loss(pred, gt, valid_mask):
    """
    pred, gt: [B, T, J, 3] (meters, root-relative)
    valid_mask: [B, T] bool
    """
    valid_mask = valid_mask.bool()
    # [B,T,J]
    err = torch.norm(pred - gt, dim=-1)
    # [B,T]
    err = err.mean(dim=-1)
    err = err[valid_mask]
    if err.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    return err.mean()


@torch.no_grad()
def eval_stage1(dataloader, model, device, epoch, edges,
                save_dir="/code/radarFMv2/checkpoints_stage1v2/vis_results",
                max_batches=None):
    """
    返回：MPJPE(mm)，只统计 valid_mask=True 的帧
    dataloader 必须返回 (radar_seq, skeleton_seq, valid_mask)
    """
    model.eval()
    total_err, total_cnt = 0.0, 0
    plotted = False
    os.makedirs(save_dir, exist_ok=True)

    for i, (radar_seq, skeleton_seq, valid_mask) in enumerate(dataloader):
        radar_seq = radar_seq.to(device).float()
        skeleton_seq = skeleton_seq.to(device).float()
        valid_mask = valid_mask.to(device).bool()  # [B,T]

        # Stage1 forward（让时序模块忽略 padding）
        try:
            pred = model(radar_seq, valid_mask=valid_mask)  # [B,T,J,3]
        except TypeError:
            pred = model(radar_seq)

        # 误差：meters
        err = torch.norm(pred - skeleton_seq, dim=-1).mean(dim=-1)  # [B,T]
        err_valid = err[valid_mask]
        if err_valid.numel() > 0:
            total_err += err_valid.sum().item()
            total_cnt += err_valid.numel()

        # 可视化：挑一个真实帧来画
        if (not plotted) and valid_mask.any():
            b = 0
            # 找 b=0 的第一帧有效帧
            idx = torch.where(valid_mask[b])[0]
            if idx.numel() > 0:
                t = int(idx[0].item())
                out_path = os.path.join(save_dir, f"epoch_{epoch}_vis.html")
                plot_skeleton(
                    skeleton_seq[b, t].detach().cpu().numpy(),
                    pred[b, t].detach().cpu().numpy(),
                    edges,
                    epoch,
                    out_path
                )
                plotted = True

        if max_batches is not None and (i + 1) >= max_batches:
            break

    # 返回 mm
    mean_mpjpe_m = total_err / max(total_cnt, 1)
    return mean_mpjpe_m * 1000.0


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

def eval_stage1_with_viz(dataloader, model, device, save_gif=True,
                         gif_name="stage1v2_check.gif", html_name="coord_checkv2.html"):
    model.eval()
    total_err = 0.0
    total_cnt = 0
    viz_sample = None

    with torch.no_grad():
        for i, (radar_seq, skeleton_seq, valid_mask) in enumerate(dataloader):
            radar_seq = radar_seq.to(device).float()
            skeleton_seq = skeleton_seq.to(device).float()
            valid_mask = valid_mask.to(device).bool()

            pred = model(radar_seq, valid_mask=valid_mask)  # [B,T,27,3]

            err = torch.norm(pred - skeleton_seq, dim=-1).mean(dim=-1)  # [B,T]
            err_valid = err[valid_mask]
            total_err += err_valid.sum().item()
            total_cnt += err_valid.numel()

            if save_gif and i == 0 and radar_seq.size(0) > 0:
                b = 0
                viz_sample = {
                    'radar': radar_seq[b].detach().cpu().numpy(),
                    'pred': pred[b].detach().cpu().numpy(),
                    'gt': skeleton_seq[b].detach().cpu().numpy(),
                    'valid_mask': valid_mask[b].detach().cpu().numpy()
                }

    mpjpe = (total_err / max(total_cnt, 1)) * 1000.0

    if save_gif and viz_sample is not None:
        # 建议选一个有效帧画
        vm = viz_sample['valid_mask']
        t = int(np.where(vm)[0][0]) if vm.any() else 0
        save_single_frame_html(viz_sample, frame_idx=t, save_path=html_name)

    return mpjpe

    
def save_single_frame_html(viz_sample, frame_idx=0, save_path="coordinate_check.html"):
    """
    专门用于检查坐标对齐情况的交互式 HTML 可视化
    """
    radar_pts = viz_sample['radar'][frame_idx]  # [N, C]
    pred_joints = viz_sample['pred'][frame_idx] # [27, 3]
    gt_joints = viz_sample['gt'][frame_idx]     # [27, 3]

    # 1. 提取雷达点云坐标 (过滤无效点)
    # mask = np.abs(radar_pts[:, :3]).sum(axis=1) > 1e-6
    r_coords = radar_pts[:, :3]

    # 2. 构造雷达散点图
    traces = []
    traces.append(go.Scatter3d(
        x=r_coords[:, 0], y=r_coords[:, 1], z=r_coords[:, 2],
        mode='markers',
        marker=dict(size=1.5, color='gray', opacity=0.5),
        name='Radar Points (Original)'
    ))

    # 3. 构造 GT 骨架连线
    def add_skeleton_trace(joints, color, name, show_line=True):
        # 散点
        traces.append(go.Scatter3d(
            x=joints[:, 0], y=joints[:, 1], z=joints[:, 2],
            mode='markers',
            marker=dict(size=3, color=color),
            name=f'{name} Joints'
        ))
        # 连线
        if show_line:
            line_x, line_y, line_z = [], [], []
            for start_node, end_node in PARENT.items():
                line_x += [joints[start_node, 0], joints[end_node, 0], None]
                line_y += [joints[start_node, 1], joints[end_node, 1], None]
                line_z += [joints[start_node, 2], joints[end_node, 2], None]
            traces.append(go.Scatter3d(
                x=line_x, y=line_y, z=line_z,
                mode='lines',
                line=dict(color=color, width=4),
                name=f'{name} Skeleton'
            ))

    add_skeleton_trace(gt_joints, 'blue', 'GT')
    add_skeleton_trace(pred_joints, 'red', 'Pred')

    # 4. 布局设置
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            # 保持坐标比例 1:1:1，非常重要，否则看不出缩放问题
            aspectmode='data' 
        ),
        title=f"Coordinate System Consistency Check (Frame {frame_idx})"
    )

    fig.write_html(save_path)
    print(f"Interactive coordinate check saved to: {save_path}")
    
def create_skeleton_radar_gif(viz_sample, gif_name):
    radar_data = viz_sample['radar']  # [T, N, C]
    pred_data = viz_sample['pred']    # [T, 27, 3]
    gt_data = viz_sample['gt']        # [T, 27, 3]
    T = pred_data.shape[0]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 定义骨骼连接关系
    connections = [(k, v) for k, v in PARENT.items()]

    def update(frame):
        ax.clear()
        
        # 1. 绘制雷达点云
        r_pts = radar_data[frame]
        # 降低过滤阈值，确保点云能显示出来
        mask = np.abs(r_pts[:, :3]).sum(axis=1) > 0 
        
        if mask.any():
            ax.scatter(r_pts[mask, 0], r_pts[mask, 1], r_pts[mask, 2], 
                       s=2, c='gray', alpha=0.5, label='Radar Points')
            
            # 调试打印：检查每一帧雷达点云的坐标中心
            if frame == 0:
                print(f"Frame 0 Radar Center: {r_pts[mask, :3].mean(axis=0)}")
                print(f"Frame 0 Pred Skeleton Center: {pred_data[frame].mean(axis=0)}")

        # 2. 绘制预测骨架 (红色)
        p_pts = pred_data[frame]
        ax.scatter(p_pts[:, 0], p_pts[:, 1], p_pts[:, 2], c='red', s=20, label='Pred')
        for c1, c2 in connections:
            ax.plot([p_pts[c1, 0], p_pts[c2, 0]], 
                    [p_pts[c1, 1], p_pts[c2, 1]], 
                    [p_pts[c1, 2], p_pts[c2, 2]], color='red', linewidth=2)

        # 3. 绘制 GT 骨架 (蓝色)
        g_pts = gt_data[frame]
        if np.abs(g_pts).sum() > 1e-6:
            ax.scatter(g_pts[:, 0], g_pts[:, 1], g_pts[:, 2], c='blue', s=10, alpha=0.3, label='GT')
            for c1, c2 in connections:
                ax.plot([g_pts[c1, 0], g_pts[c2, 0]], 
                        [g_pts[c1, 1], g_pts[c2, 1]], 
                        [g_pts[c1, 2], g_pts[c2, 2]], color='blue', linewidth=1, linestyle='--', alpha=0.3)

        # --- 去掉坐标限制 ---
        # ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1]) 
        
        ax.set_title(f"Frame {frame} | Stage 1 Full View\n(Checking Coord Alignment)")
        ax.set_xlabel("X (mm/m)"); ax.set_ylabel("Y (mm/m)"); ax.set_zlabel("Z (mm/m)")
        ax.legend()

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
    ckpt_path = "./checkpoints_stage1v2/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    
    model.eval()
    with torch.no_grad():
        mpjpe = eval_stage1_with_viz(val_loader, model, device, save_gif=True, gif_name="./checkpoints_stage1v2/stage1v3_check.gif")
    print("mpjpe", mpjpe)

if __name__ == "__main__":
    main_eval()

    
def main():
    # --- 1. 设备与目录准备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "/code/radarFMv2/checkpoints_stage1v5"
    os.makedirs(save_dir, exist_ok=True)
    
    # --- 2. 数据准备 ---
    dataset = RadarDiffusionDataset(root_dir="/code/radarFMv2/dataset", num_joints=27)
    train_set = dataset.get_train_set()
    val_set = dataset.get_val_set()

    train_loader = DataLoader(
        train_set, batch_size=8, shuffle=True, 
        collate_fn=collate_fn_for_cross_modal, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=8, shuffle=False, 
        collate_fn=collate_fn_for_cross_modal, num_workers=4, pin_memory=True
    )

    # --- 3. 模型初始化 ---
    model = RadarStage1Model(
        in_channels=6, radar_embed_dim=256, num_latents=64,
        num_joints=27, parent_list=PARENT
    ).to(device)

    # --- 4. 优化器、混合精度与调度器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler()
    # 动态调整学习率：如果5次验证误差不降，LR减半
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # --- 5. 断点续传逻辑 (Resume) ---
    start_epoch = 1
    best_mpjpe = float('inf')
    resume_path = os.path.join(save_dir, "latest.pt")

    # --- 6. 训练循环 ---
    for epoch in range(start_epoch, 201):
        model.train()
        total_loss = 0
        
        for radar_seq, skeleton_seq, valid_mask in train_loader:
            radar_seq = radar_seq.to(device)
            skeleton_seq = skeleton_seq.to(device)
            valid_mask = valid_mask.to(device).bool()
        
            optimizer.zero_grad(set_to_none=True)
        
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(radar_seq, valid_mask=valid_mask)
                loss = stage1_loss(pred, skeleton_seq, valid_mask)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

        # --- 7. 每 5 个 Epoch 验证并保存 ---
        if epoch % 2 == 0 or epoch == 1:
            mpjpe = eval_stage1(val_loader, model, device, epoch, EDGES_27)
            scheduler.step(mpjpe) # 更新学习率调度器
            
            print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Val MPJPE: {mpjpe:.2f}mm")

            # 构建保存字典
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mpjpe': best_mpjpe,
                'current_mpjpe': mpjpe
            }

            # 永远保存最新的，防止服务器宕机
            torch.save(checkpoint_data, os.path.join(save_dir, "latest.pt"))

            # 如果误差创新低，保存最好的模型
            if mpjpe < best_mpjpe:
                best_mpjpe = mpjpe
                checkpoint_data['best_mpjpe'] = best_mpjpe
                torch.save(checkpoint_data, os.path.join(save_dir, "best_model.pt"))
                print(f"⭐ 发现更好模型，已保存至 best_model.pt")

    print("🎉 训练完成！")


# if __name__ == "__main__":
#     main()