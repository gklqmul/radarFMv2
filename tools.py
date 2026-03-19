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

def compute_dynamic_mask(gt_rel, valid_mask, threshold=10.0):
    """
    gt_rel: [B, T, J, 3]  (mm)
    valid_mask: [B, T]
    threshold: motion threshold (mm)

    return:
        dyn_mask: [B, T] bool
    """

    # frame displacement
    disp = torch.norm(gt_rel[:,1:] - gt_rel[:,:-1], dim=-1)   # [B,T-1,J]

    # mean joint displacement
    mean_disp = disp.mean(dim=-1)                             # [B,T-1]

    dyn = mean_disp > threshold                               # [B,T-1]

    # pad first frame
    dyn = torch.cat([torch.zeros_like(dyn[:,:1]), dyn], dim=1)

    # combine with valid mask
    dyn_mask = dyn & valid_mask

    return dyn_mask
    
def _masked_dyn_sum(x, m):
    """
    x: [B, T', J, 3]  (or [B,T',J]) representing diff vectors or errors
    m: [B, T'] bool mask for valid diffs
    Return: (sum_scalar, count) where scalar is sum of per-(b,t) mean-joint L2.
    """
    # L2 over xyz -> [B,T',J]
    if x.dim() == 4:
        x = torch.norm(x, dim=-1)
    # mean over joints -> [B,T']
    if x.dim() == 3:
        x = x.mean(dim=-1)

    m_f = m.float()
    sum_val = (x * m_f).sum()
    cnt = m_f.sum()
    return sum_val, cnt

def diff1(x):  # [B,T,J,3] -> [B,T-1,J,3]
    return x[:, 1:] - x[:, :-1]

def diff2(x):  # [B,T,J,3] -> [B,T-2,J,3]
    return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]

def diff3(x):  # [B,T,J,3] -> [B,T-3,J,3]
    return x[:, 3:] - 3 * x[:, 2:-1] + 3 * x[:, 1:-2] - x[:, :-3]
    
def masked_mean(x, mask, eps=1e-8):
    """
    x: [B,T,J,3] or [B,T,J] etc.
    mask: [B,T] bool
    returns scalar
    """
    # 先把 x 压成 [B,T] 标量（按你的定义）
    # 这里假设 x 是向量，先L2 -> [B,T,J]，再mean joint -> [B,T]
    if x.dim() == 4:            # [B,T,J,3]
        x = torch.norm(x, dim=-1)        # [B,T,J]
    if x.dim() == 3:            # [B,T,J]
        x = x.mean(dim=-1)               # [B,T]

    x = x * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return x.sum() / denom 
def compute_spatial_structure_corr(pred_rel, gt_rel):
    """
    pred_rel, gt_rel: (M, J, 3)
    return: scalar correlation averaged over frames
    """
    J = pred_rel.shape[1]

    def pairwise_dist(x):
        diff = x.unsqueeze(2) - x.unsqueeze(1)  # (M, J, J, 3)
        return torch.norm(diff, dim=-1)         # (M, J, J)

    D_pred = pairwise_dist(pred_rel)
    D_gt = pairwise_dist(gt_rel)

    # 去掉对角线
    mask = ~torch.eye(J, dtype=torch.bool, device=pred_rel.device)
    D_pred = D_pred[:, mask]
    D_gt = D_gt[:, mask]

    # 计算 Pearson correlation
    pred_mean = D_pred.mean(dim=1, keepdim=True)
    gt_mean = D_gt.mean(dim=1, keepdim=True)

    pred_centered = D_pred - pred_mean
    gt_centered = D_gt - gt_mean

    numerator = (pred_centered * gt_centered).sum(dim=1)
    denominator = torch.sqrt(
        (pred_centered ** 2).sum(dim=1) *
        (gt_centered ** 2).sum(dim=1)
    ) + 1e-8

    corr = numerator / denominator  # (M,)

    return corr.mean()


def compute_jitter_metrics(pred_rel, valid_mask):
    # pred_rel: [B, T, J, 3] (mm)
    # valid_mask: [B, T] bool
    
    # velocity/accel/jerk existence masks
    # m_v = valid_mask[:, 1:] & valid_mask[:, :-1]           # [B, T-1]
    dyn_mask = compute_dynamic_mask(pred_rel, valid_mask)
    m_v = dyn_mask[:,1:] & dyn_mask[:,:-1]
    m_a = m_v[:, 1:] & m_v[:, :-1]                         # [B, T-2]
    m_j = m_a[:, 1:] & m_a[:, :-1]                         # [B, T-3]

    v = diff1(pred_rel)  # [B, T-1, J, 3]
    a = diff2(pred_rel)  # [B, T-2, J, 3]
    j = diff3(pred_rel)  # [B, T-3, J, 3]

    out = {}
    if m_v.any():
        out["mpjv"] = masked_mean(v, m_v).item()           # 预测自身速度幅值
    if m_a.any():
        out["mpja"] = masked_mean(a, m_a).item()           # 预测自身加速度幅值（抖动）
    if m_j.any():
        out["mpjj"] = masked_mean(j, m_j).item()           # 预测自身jerk幅值（抖动）
    return out

def compute_dynamic_errors(pred_rel, gt_rel, valid_mask):
    # 与GT的速度/加速度/jerk误差
    dyn_mask = compute_dynamic_mask(gt_rel, valid_mask)

    m_v = dyn_mask[:,1:] & dyn_mask[:,:-1]
    # m_v = valid_mask[:, 1:] & valid_mask[:, :-1]
    m_a = m_v[:, 1:] & m_v[:, :-1]
    m_j = m_a[:, 1:] & m_a[:, :-1]

    ev = diff1(pred_rel) - diff1(gt_rel)
    ea = diff2(pred_rel) - diff2(gt_rel)
    ej = diff3(pred_rel) - diff3(gt_rel)

    out = {}
    if m_v.any():
        out["mpjve"] = masked_mean(ev, m_v).item()
    if m_a.any():
        out["mpjae"] = masked_mean(ea, m_a).item()
    if m_j.any():
        out["mpjje"] = masked_mean(ej, m_j).item()
    return out
    
def compute_mpjpe(pred, gt):
    """
    pred, gt: [M, J, 3]  (mm)
    return: float (mm)
    """
    return torch.norm(pred - gt, dim=-1).mean()
    
def batch_procrustes_align(pred, gt, eps=1e-8):


    muX = pred.mean(dim=1, keepdim=True)
    muY = gt.mean(dim=1, keepdim=True)

    X0 = pred - muX
    Y0 = gt - muY

    H = torch.matmul(X0.transpose(1,2), Y0)

    U, S, Vt = torch.linalg.svd(H)
    V = Vt.transpose(1,2)

    detR = torch.det(V @ U.transpose(1,2))
    sign = torch.ones_like(detR)
    sign[detR < 0] = -1

    Z = torch.eye(3, device=pred.device).unsqueeze(0).repeat(pred.shape[0],1,1)
    Z[:,2,2] = sign
    R = V @ Z @ U.transpose(1,2)

    varX = (X0**2).sum(dim=(1,2)) + eps
    
    scale = (S[:,0] + S[:,1] + sign*S[:,2]) / varX
    
    X_aligned = scale.view(-1,1,1) * (X0 @ R.transpose(1,2)) + muY
 

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
    
def compute_bone_length_mae(pred_rel, gt_rel, edges):
    """
    pred_rel, gt_rel: (M, J, 3)
    edges: list[(i,j)]
    return: scalar MAE over all bones
    """
    device = pred_rel.device
    idx_i = torch.tensor([e[0] for e in edges], device=device)
    idx_j = torch.tensor([e[1] for e in edges], device=device)

    d_pred = torch.norm(pred_rel[:, idx_i] - pred_rel[:, idx_j], dim=-1)  # (M,E)
    d_gt   = torch.norm(gt_rel[:, idx_i]   - gt_rel[:, idx_j], dim=-1)
    return (d_pred - d_gt).abs().mean()

def compute_bone_length_var(x_rel, edges):
    device = x_rel.device
    idx_i = torch.tensor([e[0] for e in edges], device=device)
    idx_j = torch.tensor([e[1] for e in edges], device=device)
    d = torch.norm(x_rel[:, idx_i] - x_rel[:, idx_j], dim=-1)  # (M,E)
    return d.var(dim=0).mean()  # across frames, then average bones

def compute_mjae(seq_pred, seq_gt):
    """
    seq_pred, seq_gt: (B, T, J, 3) root-relative mm
    return: (B, T-2) per-frame acceleration error (mean over joints)
    """
    a_pred = seq_pred[:, 2:] - 2*seq_pred[:, 1:-1] + seq_pred[:, :-2]
    a_gt   = seq_gt[:, 2:]   - 2*seq_gt[:, 1:-1]   + seq_gt[:, :-2]
    err = torch.norm(a_pred - a_gt, dim=-1).mean(dim=-1)  # (B, T-2)
    return err

def compute_jerk_energy(seq):
    """
    seq: (B, T, J, 3) root-relative mm
    return: scalar jerk energy (lower = smoother)
    """
    # third difference
    j = seq[:, 3:] - 3*seq[:, 2:-1] + 3*seq[:, 1:-2] - seq[:, :-3]
    return (j**2).sum(dim=-1).mean()

def best_of_k_mpjpe(samples, gt_rel):
    """
    samples: (K, M, J, 3)  K samples for each frame
    gt_rel:  (M, J, 3)
    return: scalar best-of-K MPJPE
    """
    # mpjpe per sample
    errs = torch.norm(samples - gt_rel.unsqueeze(0), dim=-1).mean(dim=-1)  # (K,M)
    best = errs.min(dim=0).values  # (M,)
    return best.mean()

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
        
def plot_skeleton(gt_joints, pred_joints, edges, aligned=None, frame_id=0, out_html=None):
    """
    重叠显示 GT, Pred 和 Aligned (如果存在)
    输入：均应为 numpy 数组 (J, 3) 或可转为 numpy 的对象
    """
    # 1. 基础预处理：全部归一化到 Root-Relative
    gt_joints = np.array(gt_joints) - np.array(gt_joints)[0]
    pred_joints = np.array(pred_joints) - np.array(pred_joints)[0]
    
    data_list = []

    # 辅助函数：生成该姿态的 trace (点+线)
    def add_pose_traces(joints, color, name):
        # 散点
        scatter = go.Scatter3d(
            x=joints[:,0], y=joints[:,1], z=joints[:,2],
            mode='markers', marker=dict(size=4, color=color), 
            name=name, legendgroup=name
        )
        # 线段
        xs, ys, zs = [], [], []
        for (i, j) in edges:
            xs += [float(joints[i,0]), float(joints[j,0]), None]
            ys += [float(joints[i,1]), float(joints[j,1]), None]
            zs += [float(joints[i,2]), float(joints[j,2]), None]
        
        lines = go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines', 
            line=dict(color=color, width=3),
            name=name, legendgroup=name, showlegend=False
        )
        return [scatter, lines]

    # 2. 添加 GT 和 Pred
    data_list.extend(add_pose_traces(gt_joints, 'blue', 'GT'))
    data_list.extend(add_pose_traces(pred_joints, 'red', 'Pred'))

    # 3. 如果有 Aligned 则添加
    if aligned is not None:
        aligned_joints = np.array(aligned) - np.array(aligned)[0]
        data_list.extend(add_pose_traces(aligned_joints, 'green', 'Aligned'))

    # 4. 绘图
    fig = go.Figure(data=data_list)
    fig.update_layout(
        scene=dict(aspectmode='data'), 
        title=f"Frame {frame_id}: GT (Blue) vs Pred (Red) vs Aligned (Green)",
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    if out_html:
        fig.write_html(out_html)
    else:
        fig.show()



def _extract_radar_xyz_frame(radar_frame):
    """
    radar_frame:
        possible shapes:
            [N, 3]
            [N, C] where first 3 dims are xyz
            [H, W, 3] / others -> flatten to [-1, 3]
    return:
        xyz: [K, 3] numpy array
    """
    if hasattr(radar_frame, "detach"):
        radar_frame = radar_frame.detach().cpu().float().numpy()
    else:
        radar_frame = np.asarray(radar_frame, dtype=np.float32)

    if radar_frame.ndim == 2:
        if radar_frame.shape[-1] < 3:
            raise ValueError(f"Radar frame last dim must be >=3, got {radar_frame.shape}")
        xyz = radar_frame[:, :3]

    elif radar_frame.ndim >= 3:
        radar_frame = radar_frame.reshape(-1, radar_frame.shape[-1])
        if radar_frame.shape[-1] < 3:
            raise ValueError(f"Radar frame last dim must be >=3, got {radar_frame.shape}")
        xyz = radar_frame[:, :3]

    else:
        raise ValueError(f"Unsupported radar frame shape: {radar_frame.shape}")

    # 去掉全零 / 非法点
    valid = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 1e-8)
    xyz = xyz[valid]
    return xyz


def save_gt_radar_frame_jpg(
    gt_joints_mm,
    radar_xyz,
    edges,
    save_path,
    frame_id="frame10",
    elev=18,
    azim=-58,
    dpi=420,
):
    """
    gt_joints_mm: [J,3] numpy
    radar_xyz:    [N,3] numpy
    edges:        list[(i,j)]
    save_path:    .jpg
    """

    gt_joints_mm = np.asarray(gt_joints_mm, dtype=np.float32)
    radar_xyz = np.asarray(radar_xyz, dtype=np.float32)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    # -----------------------
    # radar point cloud
    # -----------------------
    if radar_xyz.shape[0] > 0:
        ax.scatter(
            radar_xyz[:, 0],
            radar_xyz[:, 1],
            radar_xyz[:, 2],
            s=5,
            alpha=0.55,
            depthshade=False,
        )

    # -----------------------
    # GT skeleton joints
    # -----------------------
    ax.scatter(
        gt_joints_mm[:, 0],
        gt_joints_mm[:, 1],
        gt_joints_mm[:, 2],
        s=70,
        depthshade=False,
    )

    # bones
    for i, j in edges:
        xs = [gt_joints_mm[i, 0], gt_joints_mm[j, 0]]
        ys = [gt_joints_mm[i, 1], gt_joints_mm[j, 1]]
        zs = [gt_joints_mm[i, 2], gt_joints_mm[j, 2]]
        ax.plot(xs, ys, zs, linewidth=2.6)

    # -----------------------
    # 视角：侧前方
    # -----------------------
    ax.view_init(elev=elev, azim=azim)

    # -----------------------
    # 等比例坐标范围
    # 用 skeleton + radar 共同定范围
    # -----------------------
    if radar_xyz.shape[0] > 0:
        all_pts = np.concatenate([gt_joints_mm, radar_xyz], axis=0)
    else:
        all_pts = gt_joints_mm

    xyz_min = all_pts.min(axis=0)
    xyz_max = all_pts.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    span = (xyz_max - xyz_min).max()

    # 给一点边距
    span = max(span, 800.0)  # mm
    half = span / 2.0 * 1.15

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    # 标签
    ax.set_xlabel("X (mm)", labelpad=14)
    ax.set_ylabel("Y (mm)", labelpad=14)
    ax.set_zlabel("Z (mm)", labelpad=14)
    ax.set_title(f"GT + Radar Point Cloud ({frame_id})", pad=18, fontsize=18)

    # 提高观感
    ax.grid(True)
    ax.set_box_aspect((1, 1, 1))

    # 超高清 jpg
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, format="jpg", dpi=dpi, bbox_inches="tight", pil_kwargs={"quality": 100})
    plt.close(fig)

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap


# =========================================================
# 1) Radar / plotting utils
# =========================================================

def extract_radar_xyz_snr_frame(radar_frame):
    """
    radar_frame: [N,6] = (x, y, z, doppler, snr, t)
    return:
        xyz: [K,3]
        snr: [K]
    """
    if hasattr(radar_frame, "detach"):
        radar_frame = radar_frame.detach().cpu().float().numpy()
    else:
        radar_frame = np.asarray(radar_frame, dtype=np.float32)

    if radar_frame.ndim != 2 or radar_frame.shape[-1] < 5:
        raise ValueError(f"Expected radar_frame [N,6]-like, got {radar_frame.shape}")

    xyz = radar_frame[:, :3]
    snr = radar_frame[:, 4]

    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(snr)
    xyz = xyz[valid]
    snr = snr[valid]
    return xyz, snr


def normalize_snr_for_vis(snr, eps=1e-6, q_low=1.0, q_high=99.0):
    """
    Robust normalization to [0,1] with log compression.
    """
    snr = np.asarray(snr, dtype=np.float32)
    snr = np.log1p(np.maximum(snr, 0.0))
    lo = np.percentile(snr, q_low)
    hi = np.percentile(snr, q_high)
    snr = np.clip(snr, lo, hi)
    return (snr - lo) / (hi - lo + eps)


def get_paper_blue_cmap():
    """
    Paper-style blue colormap for radar points.
    """
    colors = [
        (0.92, 0.96, 1.00),
        (0.72, 0.85, 0.97),
        (0.45, 0.68, 0.91),
        (0.18, 0.45, 0.78),
        (0.05, 0.20, 0.52),
    ]
    return LinearSegmentedColormap.from_list("paper_blue", colors)


def remap_xyz_for_plot(xyz):
    """
    Original convention:
        x: horizontal plane
        y: up
        z: depth

    Plot convention:
        plot_x = x
        plot_y = z (depth)
        plot_z = y (up)
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    return np.stack([xyz[:, 0], xyz[:, 2], xyz[:, 1]], axis=1)


def set_axes_equal_3d(ax, xyz, scale_margin=1.10):
    xyz = np.asarray(xyz, dtype=np.float32)
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    span = (xyz_max - xyz_min).max()
    if span < 1e-8:
        span = 1.0
    half = 0.5 * span * scale_margin

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))


def style_3d_axes_for_paper(ax, unit="mm", show_grid=True):
    ax.set_xlabel(f"X ({unit})", labelpad=8, fontsize=11)
    ax.set_ylabel(f"Z-depth ({unit})", labelpad=8, fontsize=11)
    ax.set_zlabel(f"Y-up ({unit})", labelpad=8, fontsize=11)

    ax.tick_params(axis="both", which="major", labelsize=9, pad=2)
    ax.tick_params(axis="z", which="major", labelsize=9, pad=2)

    try:
        ax.xaxis.pane.set_facecolor((1, 1, 1, 0.0))
        ax.yaxis.pane.set_facecolor((1, 1, 1, 0.0))
        ax.zaxis.pane.set_facecolor((1, 1, 1, 0.0))

        ax.xaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
        ax.yaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
        ax.zaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
    except Exception:
        pass

    if show_grid:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    else:
        ax.grid(False)


def draw_skeleton_on_ax(
    ax,
    joints_mm,
    edges,
    color="#2F2F2F",
    linewidth=2.0,
    alpha=1.0,
    joint_size=28,
):
    """
    joints_mm: [J,3], root-relative, mm
    """
    pts = remap_xyz_for_plot(joints_mm)

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        s=joint_size,
        c=color,
        alpha=alpha,
        depthshade=False,
    )

    for i, j in edges:
        ax.plot(
            [pts[i, 0], pts[j, 0]],
            [pts[i, 1], pts[j, 1]],
            [pts[i, 2], pts[j, 2]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )



# =========================================================
# 2) Metrics utils
# =========================================================

def init_metric_accumulator(pck_thresholds, num_joints=None, device=None):
    acc = {
        # frame-weighted
        "sum_mpjpe": 0.0,
        "sum_pampjpe": 0.0,
        "sum_auc": 0.0,
        "sum_pck": {th: 0.0 for th in pck_thresholds},
        "sum_ssc": 0.0,
        "sum_bone_mae": 0.0,
        "n_frames": 0,

        # per-joint
        "sum_joint_mpjpe": torch.zeros(num_joints, device=device) if num_joints is not None else None,
        "sum_joint_pampjpe": torch.zeros(num_joints, device=device) if num_joints is not None else None,
        "n_joint_frames": 0,

        # diff-weighted
        "sum_mpjve": 0.0,
        "n_vel": 0,

        "sum_mjae": 0.0,
        "n_acc": 0,

        # prediction-only motion
        "sum_mpjv_pred": 0.0,
        "n_mpjv": 0,

        "sum_mpja_pred": 0.0,
        "n_mpja": 0,

        "sum_mpjj_pred": 0.0,
        "n_mpjj": 0,

        # dynamic errors
        "sum_mpjve_dyn": 0.0,
        "n_mpjve_dyn": 0,

        "sum_mpjae_dyn": 0.0,
        "n_mpjae_dyn": 0,

        "sum_mpjje_dyn": 0.0,
        "n_mpjje_dyn": 0,

        # bone variance
        "sum_bone_var_pred": 0.0,
        "sum_bone_var_gt": 0.0,
        "n_bone_var": 0,
    }
    return acc


def add_metrics_inplace(acc, cur, pck_thresholds):
    acc["sum_mpjpe"] += cur["sum_mpjpe"]
    acc["sum_pampjpe"] += cur["sum_pampjpe"]
    acc["sum_auc"] += cur["sum_auc"]
    acc["sum_ssc"] += cur["sum_ssc"]
    acc["sum_bone_mae"] += cur["sum_bone_mae"]
    acc["n_frames"] += cur["n_frames"]

    for th in pck_thresholds:
        acc["sum_pck"][th] += cur["sum_pck"][th]

    if acc["sum_joint_mpjpe"] is not None and cur["sum_joint_mpjpe"] is not None:
        acc["sum_joint_mpjpe"] += cur["sum_joint_mpjpe"]
        acc["sum_joint_pampjpe"] += cur["sum_joint_pampjpe"]
        acc["n_joint_frames"] += cur["n_joint_frames"]

    acc["sum_mpjve"] += cur["sum_mpjve"]
    acc["n_vel"] += cur["n_vel"]

    acc["sum_mjae"] += cur["sum_mjae"]
    acc["n_acc"] += cur["n_acc"]

    acc["sum_mpjv_pred"] += cur["sum_mpjv_pred"]
    acc["n_mpjv"] += cur["n_mpjv"]

    acc["sum_mpja_pred"] += cur["sum_mpja_pred"]
    acc["n_mpja"] += cur["n_mpja"]

    acc["sum_mpjj_pred"] += cur["sum_mpjj_pred"]
    acc["n_mpjj"] += cur["n_mpjj"]

    acc["sum_mpjve_dyn"] += cur["sum_mpjve_dyn"]
    acc["n_mpjve_dyn"] += cur["n_mpjve_dyn"]

    acc["sum_mpjae_dyn"] += cur["sum_mpjae_dyn"]
    acc["n_mpjae_dyn"] += cur["n_mpjae_dyn"]

    acc["sum_mpjje_dyn"] += cur["sum_mpjje_dyn"]
    acc["n_mpjje_dyn"] += cur["n_mpjje_dyn"]

    acc["sum_bone_var_pred"] += cur["sum_bone_var_pred"]
    acc["sum_bone_var_gt"] += cur["sum_bone_var_gt"]
    acc["n_bone_var"] += cur["n_bone_var"]


def average_metric_accumulators(acc_list, pck_thresholds, num_joints, device):
    """
    Average multiple metric accumulators at metric level.
    Counts are assumed identical across samples for the same batch.
    """
    if len(acc_list) == 0:
        return init_metric_accumulator(pck_thresholds, num_joints=num_joints, device=device)

    out = init_metric_accumulator(pck_thresholds, num_joints=num_joints, device=device)

    out["sum_mpjpe"] = float(np.mean([a["sum_mpjpe"] for a in acc_list]))
    out["sum_pampjpe"] = float(np.mean([a["sum_pampjpe"] for a in acc_list]))
    out["sum_auc"] = float(np.mean([a["sum_auc"] for a in acc_list]))
    out["sum_ssc"] = float(np.mean([a["sum_ssc"] for a in acc_list]))
    out["sum_bone_mae"] = float(np.mean([a["sum_bone_mae"] for a in acc_list]))

    out["n_frames"] = acc_list[0]["n_frames"]

    for th in pck_thresholds:
        out["sum_pck"][th] = float(np.mean([a["sum_pck"][th] for a in acc_list]))

    if acc_list[0]["sum_joint_mpjpe"] is not None:
        out["sum_joint_mpjpe"] = torch.stack([a["sum_joint_mpjpe"] for a in acc_list], dim=0).mean(dim=0)
        out["sum_joint_pampjpe"] = torch.stack([a["sum_joint_pampjpe"] for a in acc_list], dim=0).mean(dim=0)
        out["n_joint_frames"] = acc_list[0]["n_joint_frames"]

    out["sum_mpjve"] = float(np.mean([a["sum_mpjve"] for a in acc_list]))
    out["n_vel"] = acc_list[0]["n_vel"]

    out["sum_mjae"] = float(np.mean([a["sum_mjae"] for a in acc_list]))
    out["n_acc"] = acc_list[0]["n_acc"]

    out["sum_mpjv_pred"] = float(np.mean([a["sum_mpjv_pred"] for a in acc_list]))
    out["n_mpjv"] = acc_list[0]["n_mpjv"]

    out["sum_mpja_pred"] = float(np.mean([a["sum_mpja_pred"] for a in acc_list]))
    out["n_mpja"] = acc_list[0]["n_mpja"]

    out["sum_mpjj_pred"] = float(np.mean([a["sum_mpjj_pred"] for a in acc_list]))
    out["n_mpjj"] = acc_list[0]["n_mpjj"]

    out["sum_mpjve_dyn"] = float(np.mean([a["sum_mpjve_dyn"] for a in acc_list]))
    out["n_mpjve_dyn"] = acc_list[0]["n_mpjve_dyn"]

    out["sum_mpjae_dyn"] = float(np.mean([a["sum_mpjae_dyn"] for a in acc_list]))
    out["n_mpjae_dyn"] = acc_list[0]["n_mpjae_dyn"]

    out["sum_mpjje_dyn"] = float(np.mean([a["sum_mpjje_dyn"] for a in acc_list]))
    out["n_mpjje_dyn"] = acc_list[0]["n_mpjje_dyn"]

    out["sum_bone_var_pred"] = float(np.mean([a["sum_bone_var_pred"] for a in acc_list]))
    out["sum_bone_var_gt"] = float(np.mean([a["sum_bone_var_gt"] for a in acc_list]))
    out["n_bone_var"] = acc_list[0]["n_bone_var"]

    return out


def compute_metrics_for_prediction(
    pred,
    skeleton_seq,
    valid_mask,
    edge_i=None,
    edge_j=None,
    pck_thresholds=(50.0, 100.0),
    auc_max_threshold=80.0,
    auc_step=5.0,
):
    """
    pred: [B,T,J,3] in meters
    skeleton_seq: [B,T,J,3] in meters

    Returns one metric accumulator dict for this prediction.
    """
    device = pred.device
    J = pred.shape[2]
    out = init_metric_accumulator(pck_thresholds, num_joints=J, device=device)

    pred_mm = pred * 1000.0
    gt_mm = skeleton_seq * 1000.0

    pred_rel = pred_mm - pred_mm[:, :, 0:1, :]
    gt_rel = gt_mm - gt_mm[:, :, 0:1, :]

    v_pred = pred_rel[valid_mask]  # [M,J,3]
    v_gt = gt_rel[valid_mask]      # [M,J,3]
    M = v_gt.shape[0]

    if M == 0:
        return out

    # Frame-level metrics
    mpjpe_val = compute_mpjpe(v_pred, v_gt).item()

    aligned_pred = batch_procrustes_align(v_pred, v_gt)
    pampjpe_val = compute_mpjpe(aligned_pred, v_gt).item()

    auc_val = compute_auc_pck(
        v_pred, v_gt, max_threshold=auc_max_threshold, step=auc_step
    ).item()

    pck_vals = {th: compute_pck(v_pred, v_gt, th).item() for th in pck_thresholds}

    out["sum_mpjpe"] += mpjpe_val * M
    out["sum_pampjpe"] += pampjpe_val * M
    out["sum_auc"] += auc_val * M
    out["sum_ssc"] += compute_spatial_structure_corr(v_pred, v_gt).item() * M
    out["n_frames"] += M

    for th in pck_thresholds:
        out["sum_pck"][th] += pck_vals[th] * M

    # Per-joint
    joint_err = torch.norm(v_pred - v_gt, dim=-1)
    joint_pa_err = torch.norm(aligned_pred - v_gt, dim=-1)
    out["sum_joint_mpjpe"] += joint_err.sum(dim=0)
    out["sum_joint_pampjpe"] += joint_pa_err.sum(dim=0)
    out["n_joint_frames"] += M

    # Bone MAE
    if edge_i is not None:
        d_pred = torch.norm(v_pred[:, edge_i] - v_pred[:, edge_j], dim=-1)
        d_gt = torch.norm(v_gt[:, edge_i] - v_gt[:, edge_j], dim=-1)
        bone_mae_val = (d_pred - d_gt).abs().mean().item()
        out["sum_bone_mae"] += bone_mae_val * M

    # Diff-based metrics
    m_v = valid_mask[:, 1:] & valid_mask[:, :-1]   # [B,T-1]
    m_a = m_v[:, 1:] & m_v[:, :-1]                 # [B,T-2]
    m_j = m_a[:, 1:] & m_a[:, :-1]                 # [B,T-3]

    if m_v.any():
        v_err = diff1(pred_rel) - diff1(gt_rel)
        s, c = _masked_dyn_sum(v_err, m_v)
        out["sum_mpjve"] += s.item()
        out["n_vel"] += int(c.item())

        out["sum_mpjve_dyn"] += s.item()
        out["n_mpjve_dyn"] += int(c.item())

        v_mag = diff1(pred_rel)
        s, c = _masked_dyn_sum(v_mag, m_v)
        out["sum_mpjv_pred"] += s.item()
        out["n_mpjv"] += int(c.item())

    if m_a.any():
        a_err = diff2(pred_rel) - diff2(gt_rel)
        s, c = _masked_dyn_sum(a_err, m_a)
        out["sum_mjae"] += s.item()
        out["n_acc"] += int(c.item())

        out["sum_mpjae_dyn"] += s.item()
        out["n_mpjae_dyn"] += int(c.item())

        a_mag = diff2(pred_rel)
        s, c = _masked_dyn_sum(a_mag, m_a)
        out["sum_mpja_pred"] += s.item()
        out["n_mpja"] += int(c.item())

    if m_j.any():
        j_err = diff3(pred_rel) - diff3(gt_rel)
        s, c = _masked_dyn_sum(j_err, m_j)
        out["sum_mpjje_dyn"] += s.item()
        out["n_mpjje_dyn"] += int(c.item())

        j_mag = diff3(pred_rel)
        s, c = _masked_dyn_sum(j_mag, m_j)
        out["sum_mpjj_pred"] += s.item()
        out["n_mpjj"] += int(c.item())

    # Bone temporal variance
    if edge_i is not None and M > 1:
        d_pred = torch.norm(v_pred[:, edge_i] - v_pred[:, edge_j], dim=-1)
        d_gt = torch.norm(v_gt[:, edge_i] - v_gt[:, edge_j], dim=-1)
        out["sum_bone_var_pred"] += d_pred.var(dim=0, unbiased=False).mean().item()
        out["sum_bone_var_gt"] += d_gt.var(dim=0, unbiased=False).mean().item()
        out["n_bone_var"] += 1

    return out


def select_best_prediction_oracle(pred_list, skeleton_seq):
    """
    pred_list: list of [B,T,J,3]
    skeleton_seq: [B,T,J,3]

    Returns:
        pred_best: [B,T,J,3]
    """
    all_preds = torch.stack(pred_list, dim=0)      # [N,B,T,J,3]
    gt_expanded = skeleton_seq.unsqueeze(0)        # [1,B,T,J,3]
    dist = torch.norm(all_preds - gt_expanded, dim=-1).mean(dim=-1)  # [N,B,T]
    best_idx = torch.argmin(dist, dim=0)           # [B,T]

    N, B, T, J, C = all_preds.shape
    idx_reshaped = best_idx.view(1, B, T, 1, 1).expand(1, B, T, J, C)
    pred_best = torch.gather(all_preds, 0, idx_reshaped).squeeze(0)
    return pred_best


def finalize_metric_accumulator(acc, pck_thresholds, joint_names=None):
    denom = max(acc["n_frames"], 1)

    final = {
        "mpjpe": acc["sum_mpjpe"] / denom,
        "pa_mpjpe": acc["sum_pampjpe"] / denom,
        "auc_pck": acc["sum_auc"] / denom,
        "ssc": acc["sum_ssc"] / denom,
    }

    for th in pck_thresholds:
        final[f"pck@{int(th)}"] = acc["sum_pck"][th] / denom

    if acc["n_vel"] > 0:
        final["mpjve"] = acc["sum_mpjve"] / acc["n_vel"]

    if acc["n_acc"] > 0:
        final["mjae"] = acc["sum_mjae"] / acc["n_acc"]

    # bone stats
    final["bone_mae"] = acc["sum_bone_mae"] / denom
    if acc["n_bone_var"] > 0:
        final["bone_var_pred"] = acc["sum_bone_var_pred"] / acc["n_bone_var"]
        final["bone_var_gt"] = acc["sum_bone_var_gt"] / acc["n_bone_var"]

    # motion mags
    if acc["n_mpjv"] > 0:
        final["mpjv_pred"] = acc["sum_mpjv_pred"] / acc["n_mpjv"]
    if acc["n_mpja"] > 0:
        final["mpja_pred"] = acc["sum_mpja_pred"] / acc["n_mpja"]
    if acc["n_mpjj"] > 0:
        final["mpjj_pred"] = acc["sum_mpjj_pred"] / acc["n_mpjj"]

    # dyn errors
    if acc["n_mpjve_dyn"] > 0:
        final["mpjve_dyn"] = acc["sum_mpjve_dyn"] / acc["n_mpjve_dyn"]
    if acc["n_mpjae_dyn"] > 0:
        final["mpjae_dyn"] = acc["sum_mpjae_dyn"] / acc["n_mpjae_dyn"]
    if acc["n_mpjje_dyn"] > 0:
        final["mpjje_dyn"] = acc["sum_mpjje_dyn"] / acc["n_mpjje_dyn"]

    # per-joint
    if acc["sum_joint_mpjpe"] is not None and acc["n_joint_frames"] > 0:
        joint_mpjpe = (acc["sum_joint_mpjpe"] / acc["n_joint_frames"]).detach().cpu().tolist()
        joint_pa_mpjpe = (acc["sum_joint_pampjpe"] / acc["n_joint_frames"]).detach().cpu().tolist()

        final["joint_mpjpe"] = joint_mpjpe
        final["joint_pa_mpjpe"] = joint_pa_mpjpe

        if joint_names is not None and len(joint_names) == len(joint_mpjpe):
            final["joint_mpjpe_dict"] = {
                name: val for name, val in zip(joint_names, joint_mpjpe)
            }
            final["joint_pa_mpjpe_dict"] = {
                name: val for name, val in zip(joint_names, joint_pa_mpjpe)
            }

    return final

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def extract_radar_xyz_snr_frame(radar_frame):
    """
    radar_frame: [N,6] = (x, y, z, doppler, snr, t)
    return:
        xyz: [K,3]
        snr: [K]
    """
    if hasattr(radar_frame, "detach"):
        radar_frame = radar_frame.detach().cpu().float().numpy()
    else:
        radar_frame = np.asarray(radar_frame, dtype=np.float32)

    if radar_frame.ndim != 2 or radar_frame.shape[-1] < 5:
        raise ValueError(f"Expected radar_frame [N,6]-like, got {radar_frame.shape}")

    xyz = radar_frame[:, :3]
    snr = radar_frame[:, 4]

    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(snr)
    xyz = xyz[valid]
    snr = snr[valid]
    return xyz, snr


def normalize_snr_for_vis(snr, eps=1e-6, q_low=1.0, q_high=99.0):
    snr = np.asarray(snr, dtype=np.float32)
    snr = np.log1p(np.maximum(snr, 0.0))
    lo = np.percentile(snr, q_low)
    hi = np.percentile(snr, q_high)
    snr = np.clip(snr, lo, hi)
    return (snr - lo) / (hi - lo + eps)


def get_paper_blue_cmap():
    colors = [
        (0.92, 0.96, 1.00),
        (0.72, 0.85, 0.97),
        (0.45, 0.68, 0.91),
        (0.18, 0.45, 0.78),
        (0.05, 0.20, 0.52),
    ]
    return LinearSegmentedColormap.from_list("paper_blue", colors)


def remap_xyz_for_plot(xyz):
    """
    Original convention:
        x: horizontal plane
        y: up
        z: depth

    Plot convention:
        plot_x = x
        plot_y = z (depth)
        plot_z = -y (flip vertical so the head is upward in the figure)
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    return np.stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]], axis=1)


def set_axes_equal_3d(ax, xyz, scale_margin=1.10):
    xyz = np.asarray(xyz, dtype=np.float32)
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    span = (xyz_max - xyz_min).max()
    if span < 1e-8:
        span = 1.0
    half = 0.5 * span * scale_margin

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))


def style_3d_axes_for_paper(ax, unit="mm", show_grid=True):
    ax.set_xlabel(f"X ({unit})", labelpad=8, fontsize=11)
    ax.set_ylabel(f"Z-depth ({unit})", labelpad=8, fontsize=11)
    ax.set_zlabel(f"Y-up ({unit})", labelpad=8, fontsize=11)

    ax.tick_params(axis="both", which="major", labelsize=9, pad=2)
    ax.tick_params(axis="z", which="major", labelsize=9, pad=2)

    try:
        ax.xaxis.pane.set_facecolor((1, 1, 1, 0.0))
        ax.yaxis.pane.set_facecolor((1, 1, 1, 0.0))
        ax.zaxis.pane.set_facecolor((1, 1, 1, 0.0))

        ax.xaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
        ax.yaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
        ax.zaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
    except Exception:
        pass

    if show_grid:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    else:
        ax.grid(False)


def draw_skeleton_on_ax(
    ax,
    joints_mm,
    edges,
    color="#2F2F2F",
    linewidth=2.0,
    alpha=1.0,
    joint_size=28,
):
    """
    joints_mm: [J,3], in mm
    """
    pts = remap_xyz_for_plot(joints_mm)

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        s=joint_size,
        c=color,
        alpha=alpha,
        depthshade=False,
    )

    for i, j in edges:
        ax.plot(
            [pts[i, 0], pts[j, 0]],
            [pts[i, 1], pts[j, 1]],
            [pts[i, 2], pts[j, 2]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )


def save_gt_radar_overlay(
    radar_frame,
    gt_joints_mm,
    edges,
    save_path,
    elev=18,
    azim=-48,
    dpi=500,
    add_colorbar=True,
):
    """
    图1: GT skeleton + radar point cloud

    radar_frame: [N,6], unit = meter
    gt_joints_mm: [J,3], unit = mm
    """
    radar_xyz_m, radar_snr = extract_radar_xyz_snr_frame(radar_frame)

    # radar: m -> mm
    radar_xyz_mm = radar_xyz_m * 1000.0

    radar_plot = remap_xyz_for_plot(radar_xyz_mm)
    gt_plot = remap_xyz_for_plot(gt_joints_mm)

    snr_norm = normalize_snr_for_vis(radar_snr)
    cmap = get_paper_blue_cmap()

    fig = plt.figure(figsize=(7.2, 6.4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    sc = None
    if radar_plot.shape[0] > 0:
        sc = ax.scatter(
            radar_plot[:, 0],
            radar_plot[:, 1],
            radar_plot[:, 2],
            c=snr_norm,
            cmap=cmap,
            s=8,
            alpha=0.85,
            linewidths=0.0,
            depthshade=False,
        )

    # GT skeleton on top: orange
    draw_skeleton_on_ax(
        ax=ax,
        joints_mm=gt_joints_mm,
        edges=edges,
        color="#E68613",   # 论文里更稳的橙色，不是刺眼纯橙
        linewidth=2.8,
        alpha=1.0,
        joint_size=34,
    )

    all_pts = [gt_plot]
    if radar_plot.shape[0] > 0:
        all_pts.append(radar_plot)
    all_pts = np.concatenate(all_pts, axis=0)

    set_axes_equal_3d(ax, all_pts, scale_margin=1.15)
    ax.view_init(elev=elev, azim=azim)
    style_3d_axes_for_paper(ax, unit="mm", show_grid=True)

    if add_colorbar and sc is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
        cbar.set_label("Normalized SNR", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        save_path,
        format="jpg",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": 100},
    )
    plt.close(fig)
    
def save_gt_pred_samples_overlay(
    gt_joints_mm,
    pred_joints_list_mm,
    edges,
    save_path,
    elev=18,
    azim=-48,
    dpi=500,
):
    """
    图2: GT skeleton + multiple inference skeletons

    gt_joints_mm: [J,3], unit = mm
    pred_joints_list_mm: list of [J,3], unit = mm
    """
    gt_plot = remap_xyz_for_plot(gt_joints_mm)

    fig = plt.figure(figsize=(7.2, 6.4), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    # Softer paper-style colors for inference samples
    pred_colors = [
        "#9E9E9E",  # soft gray
        "#8FB9E8",  # light blue
        "#9CCFA3",  # light green
        "#B8A6D9",  # soft lavender
        "#AEBFD1",  # gray-blue
    ]

    for k, pred_joints_mm in enumerate(pred_joints_list_mm):
        draw_skeleton_on_ax(
            ax=ax,
            joints_mm=pred_joints_mm,
            edges=edges,
            color=pred_colors[k % len(pred_colors)],
            linewidth=1.8,
            alpha=0.72,
            joint_size=20,
        )

    # GT on top: orange
    draw_skeleton_on_ax(
        ax=ax,
        joints_mm=gt_joints_mm,
        edges=edges,
        color="#E68613",
        linewidth=3.0,
        alpha=1.0,
        joint_size=36,
    )

    all_pts = [gt_plot]
    for pred_joints_mm in pred_joints_list_mm:
        all_pts.append(remap_xyz_for_plot(pred_joints_mm))
    all_pts = np.concatenate(all_pts, axis=0)

    set_axes_equal_3d(ax, all_pts, scale_margin=1.15)
    ax.view_init(elev=elev, azim=azim)
    style_3d_axes_for_paper(ax, unit="mm", show_grid=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(
        save_path,
        format="jpg",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": 100},
    )
    plt.close(fig)

def save_two_frame_figures(
    radar_frame,
    gt_joints_mm,
    pred_joints_list_mm,
    edges,
    save_dir,
    file_stem,
    elev=18,
    azim=-48,
    dpi=500,
):
    """
    一次调用保存两张图:
      1) GT + radar
      2) GT + 5 predictions
    """
    os.makedirs(save_dir, exist_ok=True)

    save_gt_radar_overlay(
        radar_frame=radar_frame,
        gt_joints_mm=gt_joints_mm,
        edges=edges,
        save_path=os.path.join(save_dir, f"{file_stem}_gt_radar.jpg"),
        elev=elev,
        azim=azim,
        dpi=dpi,
        add_colorbar=True,
    )

    save_gt_pred_samples_overlay(
        gt_joints_mm=gt_joints_mm,
        pred_joints_list_mm=pred_joints_list_mm,
        edges=edges,
        save_path=os.path.join(save_dir, f"{file_stem}_gt_preds.jpg"),
        elev=elev,
        azim=azim,
        dpi=dpi,
    )

# =========================================================
# 3) Main evaluation function
# =========================================================

@torch.no_grad()
def evaluate_sequence(
    dataloader,
    model,
    device,
    vis_dir=None,
    vis_edges=None,
    num_vis_samples=2,     # kept for compatibility; not used in the old gif/html sense
    steps=5,
    pck_thresholds=(50.0, 100.0),
    auc_max_threshold=80.0,
    auc_step=5.0,
    sample_mode="mean",    # "single" | "mean" | "best"
    num_samples=5,
    joint_names=None,
):
    """
    Evaluation for sequence prediction.

    Supported sampling modes:
        - sample_mode="single":
            one stochastic inference run
        - sample_mode="mean":
            run N stochastic inferences, compute metrics for each sample separately,
            then average metrics at metric level
        - sample_mode="best":
            oracle best-of-N using GT at frame level (upper bound only)

    Conventions:
    - Inputs/GT are in meters; converted to millimeters for reporting
    - All pose metrics are computed on root-relative joints (joint 0 as root)
    - Valid frames are those where GT is non-zero

    Visualization:
    - Saves a single JPG per chosen sequence/frame
    - Each figure overlays:
        radar point cloud + GT skeleton + all sampled predictions
    - No aligned prediction is drawn
    """
    model.eval()

    if sample_mode not in {"single", "mean", "best"}:
        raise ValueError(f"Unknown sample_mode={sample_mode}, choose from ['single', 'mean', 'best']")

    # Save directory for frame visualizations
    save_frame_dir = "/code/radarFMv2/gt_radar_frame10_vis"
    os.makedirs(save_frame_dir, exist_ok=True)

    # Edges
    edge_i = edge_j = None
    if vis_edges:
        edge_i = torch.tensor([e[0] for e in vis_edges], device=device, dtype=torch.long)
        edge_j = torch.tensor([e[1] for e in vis_edges], device=device, dtype=torch.long)

    # Global metric accumulator
    global_acc = None

    target_frame = 10
    vis_count = 0

    pbar = tqdm(dataloader, desc=f"Eval ({sample_mode}, steps={steps}, N={num_samples})")

    for batch_idx, (radar_seq, skeleton_seq) in enumerate(pbar):
        radar_seq = radar_seq.to(device).float()       # [B,T,...]
        skeleton_seq = skeleton_seq.to(device).float() # [B,T,J,3]

        B, T, J = skeleton_seq.shape[0], skeleton_seq.shape[1], skeleton_seq.shape[2]

        if global_acc is None:
            global_acc = init_metric_accumulator(
                pck_thresholds=pck_thresholds,
                num_joints=J,
                device=device,
            )

        # Valid mask
        valid_mask = (skeleton_seq.abs().sum(dim=(2, 3)) > 1e-6)  # [B,T]
        if not valid_mask.any():
            continue

        # -------------------------------------------------
        # 1) Generate prediction samples
        # -------------------------------------------------
        if sample_mode == "single":
            pred_samples = [model.inference(radar_seq, steps=steps)]
        else:
            pred_samples = [model.inference(radar_seq, steps=steps) for _ in range(num_samples)]

        # -------------------------------------------------
        # 2) Visualization for frame 10
        #    Save only a few examples to avoid huge dumps
        # -------------------------------------------------
        if vis_edges is not None and T > target_frame and vis_count < num_vis_samples:
            valid_bs = [b for b in range(B) if valid_mask[b, target_frame].item()]
            for b in valid_bs:
                if vis_count >= num_vis_samples:
                    break

                gt_frame_mm = skeleton_seq[b, target_frame].detach().cpu().numpy() * 1000.0
                gt_frame_mm = gt_frame_mm - gt_frame_mm[0:1]

                pred_joints_list_mm = []
                for p in pred_samples:
                    pred_frame_mm = p[b, target_frame].detach().cpu().numpy() * 1000.0
                    pred_frame_mm = pred_frame_mm - pred_frame_mm[0:1]
                    pred_joints_list_mm.append(pred_frame_mm)

                radar_frame = radar_seq[b, target_frame]  # [N,6]

                save_name = f"batch{batch_idx:04d}_sample{b:02d}_frame{target_frame:03d}.jpg"
                save_path = os.path.join(save_frame_dir, save_name)

                save_two_frame_figures(
                    radar_frame=radar_frame,
                    gt_joints_mm=gt_frame_mm,
                    pred_joints_list_mm=pred_joints_list_mm,
                    edges=vis_edges,
                    save_dir=save_frame_dir,
                    file_stem=f"batch{batch_idx:04d}_sample{b:02d}_frame{target_frame:03d}",
                    elev=18,
                    azim=-48,
                    dpi=500,
                )

                vis_count += 1

        # -------------------------------------------------
        # 3) Compute metrics
        # -------------------------------------------------
        if sample_mode == "single":
            batch_acc = compute_metrics_for_prediction(
                pred=pred_samples[0],
                skeleton_seq=skeleton_seq,
                valid_mask=valid_mask,
                edge_i=edge_i,
                edge_j=edge_j,
                pck_thresholds=pck_thresholds,
                auc_max_threshold=auc_max_threshold,
                auc_step=auc_step,
            )

        elif sample_mode == "mean":
            # metric-level mean, NOT prediction mean
            per_sample_accs = []
            for pred in pred_samples:
                cur = compute_metrics_for_prediction(
                    pred=pred,
                    skeleton_seq=skeleton_seq,
                    valid_mask=valid_mask,
                    edge_i=edge_i,
                    edge_j=edge_j,
                    pck_thresholds=pck_thresholds,
                    auc_max_threshold=auc_max_threshold,
                    auc_step=auc_step,
                )
                per_sample_accs.append(cur)

            batch_acc = average_metric_accumulators(
                acc_list=per_sample_accs,
                pck_thresholds=pck_thresholds,
                num_joints=J,
                device=device,
            )

        elif sample_mode == "best":
            pred_best = select_best_prediction_oracle(pred_samples, skeleton_seq)
            batch_acc = compute_metrics_for_prediction(
                pred=pred_best,
                skeleton_seq=skeleton_seq,
                valid_mask=valid_mask,
                edge_i=edge_i,
                edge_j=edge_j,
                pck_thresholds=pck_thresholds,
                auc_max_threshold=auc_max_threshold,
                auc_step=auc_step,
            )

        add_metrics_inplace(global_acc, batch_acc, pck_thresholds)

        running_mpjpe = global_acc["sum_mpjpe"] / max(global_acc["n_frames"], 1)
        pbar.set_postfix({"mpjpe": f"{running_mpjpe:.2f}mm"})

    if global_acc is None:
        # empty dataloader fallback
        return {
            "sample_mode": sample_mode,
            "num_samples": num_samples if sample_mode != "single" else 1,
            "mpjpe": 0.0,
            "pa_mpjpe": 0.0,
            "auc_pck": 0.0,
            "ssc": 0.0,
        }

    final = finalize_metric_accumulator(
        acc=global_acc,
        pck_thresholds=pck_thresholds,
        joint_names=joint_names,
    )

    final["sample_mode"] = sample_mode
    final["num_samples"] = num_samples if sample_mode != "single" else 1

    return final