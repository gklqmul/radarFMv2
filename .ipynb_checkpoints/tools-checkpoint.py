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




@torch.no_grad()
def evaluate_sequence(
    dataloader, model, device,
    vis_dir=None, vis_edges=None, num_vis_samples=2,
    steps=10,
    pck_thresholds=(50.0, 100.0),
    auc_max_threshold=80.0,
    auc_step=5.0,
    sample_mode="best",       # "single" | "mean" | "best"
    num_samples=5,
    joint_names=None,
):
    """
    Evaluation for sequence prediction.

    Supported sampling modes:
        - sample_mode="single":
            one stochastic inference run
        - sample_mode="mean":
            run N times and average hypotheses
        - sample_mode="best":
            oracle best-of-N using GT at frame level (upper bound only)

    Conventions:
    - Inputs/GT are in meters; we convert to millimeters for reporting.
    - All pose metrics are computed on root-relative joints (joint 0 as root).
    - Valid frames are those where GT is non-zero (mask from skeleton_seq).

    Metrics:
    Frame-weighted (by #valid frames M):
        - mpjpe, pa_mpjpe, auc_pck, pck@th, ssc, bone_mae
        - joint_mpjpe, joint_pa_mpjpe
    Diff-weighted (by #valid diffs):
        - mpjve, mjae
        - mpjv_pred, mpja_pred, mpjj_pred
        - mpjve_dyn, mpjae_dyn, mpjje_dyn
    Sequence-stat:
        - bone_var_pred, bone_var_gt

    Notes:
    - "best" mode uses GT to choose the best sample for each frame, so it is oracle
      and should only be used as an upper-bound analysis.
    """
    model.eval()

    if sample_mode not in {"single", "mean", "best"}:
        raise ValueError(f"Unknown sample_mode={sample_mode}, choose from ['single', 'mean', 'best']")

    # -----------------------
    # Accumulators
    # -----------------------
    # frame-weighted accumulators (sum over valid frames)
    sum_mpjpe = 0.0
    sum_pampjpe = 0.0
    sum_auc = 0.0
    sum_pck = {th: 0.0 for th in pck_thresholds}
    sum_ssc = 0.0
    sum_bone_mae = 0.0
    n_frames = 0  # total valid frames M across dataset

    # per-joint accumulators
    sum_joint_mpjpe = None
    sum_joint_pampjpe = None
    n_joint_frames = 0

    # diff-weighted accumulators (sum over valid diffs)
    sum_mpjve = 0.0
    n_vel = 0

    sum_mjae = 0.0
    n_acc = 0

    # prediction-only motion magnitude
    sum_mpjv_pred, n_mpjv = 0.0, 0
    sum_mpja_pred, n_mpja = 0.0, 0
    sum_mpjj_pred, n_mpjj = 0.0, 0

    # dynamic errors vs GT (explicit)
    sum_mpjve_dyn, n_mpjve_dyn = 0.0, 0
    sum_mpjae_dyn, n_mpjae_dyn = 0.0, 0
    sum_mpjje_dyn, n_mpjje_dyn = 0.0, 0

    # bone variance (per-batch mean over bones)
    sum_bone_var_pred = 0.0
    sum_bone_var_gt = 0.0
    n_bone_var = 0

    # -----------------------
    # Visualization setup
    # -----------------------
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
    has_visualized = 0

    # edges for bone metrics / visualization
    edge_i = edge_j = None
    if vis_edges:
        edge_i = torch.tensor([e[0] for e in vis_edges], device=device, dtype=torch.long)
        edge_j = torch.tensor([e[1] for e in vis_edges], device=device, dtype=torch.long)

    pbar = tqdm(dataloader, desc=f"Eval ({sample_mode}, steps={steps}, N={num_samples})")

    for batch_idx, (radar_seq, skeleton_seq) in enumerate(pbar):
        radar_seq = radar_seq.to(device).float()       # [B,T,...]
        skeleton_seq = skeleton_seq.to(device).float() # [B,T,J,3] in meters

        B, T, J = skeleton_seq.shape[0], skeleton_seq.shape[1], skeleton_seq.shape[2]

        if sum_joint_mpjpe is None:
            sum_joint_mpjpe = torch.zeros(J, device=device)
            sum_joint_pampjpe = torch.zeros(J, device=device)

        # -----------------------
        # 1) Valid mask from GT
        # -----------------------
        valid_mask = (skeleton_seq.abs().sum(dim=(2, 3)) > 1e-6)  # [B,T] bool
        if not valid_mask.any():
            continue

        # -----------------------
        # 2) Inference under chosen sampling mode
        # -----------------------
        if sample_mode == "single":
            pred = model.inference(radar_seq, steps=steps)  # [B,T,J,3]

        else:
            all_preds = []
            for _ in range(num_samples):
                p = model.inference(radar_seq, steps=steps)  # [B,T,J,3]
                all_preds.append(p)

            # [N,B,T,J,3]
            all_preds = torch.stack(all_preds, dim=0)

            if sample_mode == "mean":
                pred = all_preds.mean(dim=0)  # [B,T,J,3]

            elif sample_mode == "best":
                # Oracle frame-wise best-of-N using GT
                gt_expanded = skeleton_seq.unsqueeze(0)  # [1,B,T,J,3]
                dist = torch.norm(all_preds - gt_expanded, dim=-1).mean(dim=-1)  # [N,B,T]
                best_idx = torch.argmin(dist, dim=0)  # [B,T]

                idx_reshaped = best_idx.view(1, B, T, 1, 1).expand(1, B, T, J, 3)
                pred = torch.gather(all_preds, 0, idx_reshaped).squeeze(0)  # [B,T,J,3]

        # -----------------------
        # 3) Convert to mm + root-relative
        # -----------------------
        pred_mm = pred * 1000.0
        gt_mm = skeleton_seq * 1000.0

        pred_rel = pred_mm - pred_mm[:, :, 0:1, :]
        gt_rel = gt_mm - gt_mm[:, :, 0:1, :]

        aligned_pred_rel_full = torch.zeros_like(pred_rel)

        # -----------------------
        # 4) Frame-level metrics on valid frames only
        # -----------------------
        v_pred = pred_rel[valid_mask]  # [M,J,3]
        v_gt = gt_rel[valid_mask]      # [M,J,3]
        M = v_gt.shape[0]
        if M == 0:
            continue

        # overall MPJPE / PA-MPJPE / PCK / AUC
        mpjpe_val = compute_mpjpe(v_pred, v_gt).item()

        aligned_pred = batch_procrustes_align(v_pred, v_gt)   # [M,J,3]
        aligned_pred_rel = aligned_pred - aligned_pred[:, 0:1, :]

        pampjpe_val = compute_mpjpe(aligned_pred, v_gt).item()
        auc_val = compute_auc_pck(v_pred, v_gt, max_threshold=auc_max_threshold, step=auc_step).item()
        pck_vals = {th: compute_pck(v_pred, v_gt, th).item() for th in pck_thresholds}

        sum_mpjpe += mpjpe_val * M
        sum_pampjpe += pampjpe_val * M
        sum_auc += auc_val * M
        for th in pck_thresholds:
            sum_pck[th] += pck_vals[th] * M
        n_frames += M

        # per-joint MPJPE / PA-MPJPE
        joint_err = torch.norm(v_pred - v_gt, dim=-1)           # [M,J]
        joint_pa_err = torch.norm(aligned_pred - v_gt, dim=-1)  # [M,J]

        sum_joint_mpjpe += joint_err.sum(dim=0)
        sum_joint_pampjpe += joint_pa_err.sum(dim=0)
        n_joint_frames += M

        # SSC
        ssc_val = compute_spatial_structure_corr(v_pred, v_gt).item()
        sum_ssc += ssc_val * M

        # bone_mae (mean abs bone length error) on valid frames
        if edge_i is not None:
            d_pred = torch.norm(v_pred[:, edge_i] - v_pred[:, edge_j], dim=-1)  # [M,E]
            d_gt = torch.norm(v_gt[:, edge_i] - v_gt[:, edge_j], dim=-1)        # [M,E]
            bone_mae_val = (d_pred - d_gt).abs().mean().item()
            sum_bone_mae += bone_mae_val * M

        # -----------------------
        # 5) Diff-based metrics (only on consecutive valid frames)
        # -----------------------
        m_v = valid_mask[:, 1:] & valid_mask[:, :-1]   # [B,T-1]
        m_a = m_v[:, 1:] & m_v[:, :-1]                 # [B,T-2]
        m_j = m_a[:, 1:] & m_a[:, :-1]                 # [B,T-3]

        # velocity error vs GT
        if m_v.any():
            v_err = diff1(pred_rel) - diff1(gt_rel)  # [B,T-1,J,3]
            s, c = _masked_dyn_sum(v_err, m_v)
            sum_mpjve += s.item()
            n_vel += int(c.item())

            sum_mpjve_dyn += s.item()
            n_mpjve_dyn += int(c.item())

            v_mag = diff1(pred_rel)
            s, c = _masked_dyn_sum(v_mag, m_v)
            sum_mpjv_pred += s.item()
            n_mpjv += int(c.item())

        # acceleration error vs GT
        if m_a.any():
            a_err = diff2(pred_rel) - diff2(gt_rel)  # [B,T-2,J,3]
            s, c = _masked_dyn_sum(a_err, m_a)
            sum_mjae += s.item()
            n_acc += int(c.item())

            sum_mpjae_dyn += s.item()
            n_mpjae_dyn += int(c.item())

            a_mag = diff2(pred_rel)
            s, c = _masked_dyn_sum(a_mag, m_a)
            sum_mpja_pred += s.item()
            n_mpja += int(c.item())

        # jerk error vs GT
        if m_j.any():
            j_err = diff3(pred_rel) - diff3(gt_rel)  # [B,T-3,J,3]
            s, c = _masked_dyn_sum(j_err, m_j)
            sum_mpjje_dyn += s.item()
            n_mpjje_dyn += int(c.item())

            j_mag = diff3(pred_rel)
            s, c = _masked_dyn_sum(j_mag, m_j)
            sum_mpjj_pred += s.item()
            n_mpjj += int(c.item())

        # -----------------------
        # 6) Bone length temporal variance (per-batch)
        # -----------------------
        if edge_i is not None and M > 1:
            d_pred = torch.norm(v_pred[:, edge_i] - v_pred[:, edge_j], dim=-1)  # [M,E]
            d_gt = torch.norm(v_gt[:, edge_i] - v_gt[:, edge_j], dim=-1)        # [M,E]
            bone_var_pred = d_pred.var(dim=0, unbiased=False).mean().item()
            bone_var_gt = d_gt.var(dim=0, unbiased=False).mean().item()
            sum_bone_var_pred += bone_var_pred
            sum_bone_var_gt += bone_var_gt
            n_bone_var += 1

        # -----------------------
        # 7) Visualization
        # -----------------------
        if vis_dir and has_visualized < num_vis_samples and vis_edges:
            valid_bs = [i for i in range(B) if valid_mask[i].any().item()]

            if valid_bs:
                b = random.choice(valid_bs)
                valid_indices = torch.where(valid_mask[b])[0]  # (T_valid,)

                if valid_indices.numel() > 0:
                    gt_seq_full = gt_rel[b, valid_indices].detach().cpu().numpy()
                    pred_seq_full = pred_rel[b, valid_indices].detach().cpu().numpy()

                    aligned_pred_rel_full[valid_mask] = aligned_pred_rel
                    aligned_seq_full = aligned_pred_rel_full[b, valid_indices].detach().cpu().numpy()
                    num_total_frames = gt_seq_full.shape[0]

                    fig = plt.figure(figsize=(10, 5))
                    ax_gt = fig.add_subplot(121, projection="3d")
                    ax_pr = fig.add_subplot(122, projection="3d")

                    def update(idx):
                        ax_gt.cla()
                        ax_pr.cla()
                        actual_frame_idx = int(valid_indices[idx].item())
                        draw_skeleton_3d(ax_gt, gt_seq_full[idx], vis_edges, "green", f"GT Frame {actual_frame_idx}")
                        draw_skeleton_3d(ax_pr, pred_seq_full[idx], vis_edges, "red", f"Pred Frame {actual_frame_idx}")
                        draw_skeleton_3d(ax_pr, aligned_seq_full[idx], vis_edges, "blue", f"Aligned Pred Frame {actual_frame_idx}")
                        return []

                    ani = animation.FuncAnimation(fig, update, frames=num_total_frames, interval=100)
                    gif_name = f"Sample{has_visualized}_B{batch_idx}_S{b}_full_seq.gif"
                    ani.save(os.path.join(vis_dir, gif_name), writer="pillow", fps=10)
                    plt.close(fig)

                    num_html = 10
                    if num_total_frames > num_html:
                        start_f = num_total_frames // 4
                        html_indices = valid_indices[start_f:start_f + num_html]
                    else:
                        html_indices = valid_indices

                    for t_idx in html_indices:
                        t = int(t_idx.item())
                        html_name = f"Sample{has_visualized}_B{batch_idx}_f{t:03d}.html"
                        plot_skeleton(
                            gt_joints=gt_rel[b, t].detach().cpu().numpy(),
                            pred_joints=pred_rel[b, t].detach().cpu().numpy(),
                            edges=vis_edges,
                            aligned=aligned_pred_rel_full[b, t].detach().cpu().numpy(),
                            frame_id=f"Sample{has_visualized}-Frame{t}",
                            out_html=os.path.join(vis_dir, html_name)
                        )

                    has_visualized += 1

        pbar.set_postfix({"mpjpe": f"{sum_mpjpe / max(n_frames, 1):.2f}mm"})

    # -----------------------
    # 8) Final aggregation
    # -----------------------
    denom = max(n_frames, 1)
    final = {
        "sample_mode": sample_mode,
        "num_samples": num_samples if sample_mode != "single" else 1,
        "mpjpe": sum_mpjpe / denom,
        "pa_mpjpe": sum_pampjpe / denom,
        "auc_pck": sum_auc / denom,
        "ssc": sum_ssc / denom,
    }

    for th in pck_thresholds:
        final[f"pck@{int(th)}"] = sum_pck[th] / denom

    if n_vel > 0:
        final["mpjve"] = sum_mpjve / n_vel

    if n_acc > 0:
        final["mjae"] = sum_mjae / n_acc

    if edge_i is not None:
        final["bone_mae"] = sum_bone_mae / denom
        if n_bone_var > 0:
            final["bone_var_pred"] = sum_bone_var_pred / n_bone_var
            final["bone_var_gt"] = sum_bone_var_gt / n_bone_var

    if n_mpjv > 0:
        final["mpjv_pred"] = sum_mpjv_pred / n_mpjv
    if n_mpja > 0:
        final["mpja_pred"] = sum_mpja_pred / n_mpja
    if n_mpjj > 0:
        final["mpjj_pred"] = sum_mpjj_pred / n_mpjj

    if n_mpjve_dyn > 0:
        final["mpjve_dyn"] = sum_mpjve_dyn / n_mpjve_dyn
    if n_mpjae_dyn > 0:
        final["mpjae_dyn"] = sum_mpjae_dyn / n_mpjae_dyn
    if n_mpjje_dyn > 0:
        final["mpjje_dyn"] = sum_mpjje_dyn / n_mpjje_dyn

    # per-joint metrics
    if n_joint_frames > 0:
        joint_mpjpe = (sum_joint_mpjpe / n_joint_frames).detach().cpu().tolist()
        joint_pa_mpjpe = (sum_joint_pampjpe / n_joint_frames).detach().cpu().tolist()

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