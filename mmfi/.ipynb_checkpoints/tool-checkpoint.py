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
import yaml

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
PARENT_LIST = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9,  8,  11, 12, 8,  14, 15]

def calculate_mpjpe(pred, gt):
    # pred, gt: [B, 27, 3]
    return torch.norm(pred - gt, dim=-1).mean().item()

def batch_procrustes_align(pred, gt, eps=1e-8):
    """
    Procrustes alignment (scale+rotation+translation) for each sample.
    pred, gt: [M, J, 3]
    return: pred_aligned [M, J, 3]
    """
    # subtract mean
    muX = pred.mean(dim=1, keepdim=True)  # [M,1,3]
    muY = gt.mean(dim=1, keepdim=True)
    X0 = pred - muX
    Y0 = gt - muY

    # compute covariance
    # [M,3,3] = [M,3,J] @ [M,J,3]
    H = torch.matmul(X0.transpose(1, 2), Y0)

    # SVD
    U, S, Vt = torch.linalg.svd(H)  # U:[M,3,3], Vt:[M,3,3]
    V = Vt.transpose(1, 2)

    # handle reflection
    detR = torch.det(torch.matmul(V, U.transpose(1, 2)))  # [M]
    sign = torch.ones_like(detR)
    sign[detR < 0] = -1.0

    # build correction matrix
    Z = torch.eye(3, device=pred.device, dtype=pred.dtype).unsqueeze(0).repeat(pred.shape[0], 1, 1)
    Z[:, 2, 2] = sign

    R = torch.matmul(torch.matmul(V, Z), U.transpose(1, 2))  # [M,3,3]

    # scale
    varX = (X0 ** 2).sum(dim=(1, 2)) + eps  # [M]
    scale = (S * Z.diagonal(dim1=-2, dim2=-1)).sum(dim=1) / varX  # [M]

    # aligned
    X_aligned = scale.view(-1, 1, 1) * torch.matmul(X0, R) + muY
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

def eval_stage1_with_viz(dataloader, model, device, save_dir="/code/mmfi/vis_results"):
    """
    参数:
        save_dir: 所有的可视化结果都会保存在这个目录下
    """
    model.eval()
    total_err = 0.0
    total_cnt = 0
    viz_sample = None
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # 数据搬运到设备
            radar_seq = batch['radar_cond'].to(device).float()     # [B, T, 128, 6]
            skeleton_seq = batch['pointcloud'].to(device).float()  # [B, T, 17, 3]

            # 模型推理
            pred = model(radar_seq)  

            # 计算误差 (MPJPE)
            valid_mask = (skeleton_seq.abs().sum(dim=(2, 3)) > 1e-6)
            err = torch.norm(pred - skeleton_seq, dim=-1).mean(dim=-1)
            err_valid = err[valid_mask]

            total_err += err_valid.sum().item()
            total_cnt += err_valid.numel()

            # 采样第一个 Batch 的第一个序列用于后续可视化
            if i == 0:
                viz_sample = {
                    'radar': radar_seq[0].cpu().numpy(),
                    'pred': pred[0].cpu().numpy(),
                    'gt': skeleton_seq[0].cpu().numpy()
                }
                
    mpjpe = (total_err / max(total_cnt, 1)) * 1000.0

    # --- 开始可视化逻辑 ---
    if viz_sample is not None:
        try:
            print(f"--- Starting Visualizations in {save_dir} ---")
            
            # 1. 保存 GIF 动画
            gif_path = os.path.join(save_dir, "sequence_animation.gif")
            create_skeleton_radar_gif(viz_sample, gif_path)
            
            # 2. 多帧 HTML 采样 (保存 5 帧)
            T = viz_sample['radar'].shape[0]
            # 均匀选取 5 个帧索引，例如序列长度 60，会选 [0, 14, 29, 44, 59]
            sample_indices = [int(x) for x in np.linspace(0, T - 1, 5)]
            
            html_subfolder = os.path.join(save_dir, "frames_html")
            os.makedirs(html_subfolder, exist_ok=True)
            
            for idx in sample_indices:
                frame_filename = os.path.join(html_subfolder, f"frame_{idx:03d}_check.html")
                save_single_frame_html(viz_sample, frame_idx=idx, save_path=frame_filename)
            
            print(f"Successfully saved GIF and {len(sample_indices)} HTML frames.")
            
        except Exception as e:
            print(f"Visualization failed during eval: {e}")
            import traceback
            traceback.print_exc()

    return mpjpe

    
def save_single_frame_html(viz_sample, frame_idx=0, save_path="coordinate_check.html"):
    """
    专门用于检查坐标对齐情况的交互式 HTML 可视化
    """
    radar_pts = viz_sample['radar'][frame_idx]  # [N, C]
    pred_joints = viz_sample['pred'][frame_idx] # [17, 3]
    gt_joints = viz_sample['gt'][frame_idx]     # [17, 3]

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
    pred_data = viz_sample['pred']    # [T, 17, 3]
    gt_data = viz_sample['gt']        # [T, 17, 3]
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