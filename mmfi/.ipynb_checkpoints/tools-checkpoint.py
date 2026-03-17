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