import torch
import torch.nn.functional as F
import random
from dataset import DataReader, RadarDiffusionDataset # 你的数据加载代码

# ==========================================
# 1. 准备环境 & 加载数据集
# ==========================================
print("正在加载数据集...")
reader = DataReader(cache_path="state_cache.pt") 
dataset = RadarDiffusionDataset(
    root_dir='./dataset', 
    reader=reader, 
    sample_level='sequence', 
    num_joints=27
)
print(f"数据集加载完毕，共有 {len(dataset)} 个序列。")

# ==========================================
# 2. 随机提取 20 帧真实数据
# ==========================================
def get_random_batch_from_dataset(dataset, reader, num_samples=20):
    """
    从 dataset 中随机抽取 num_samples 帧，并反归一化为真实物理坐标
    """
    collected_frames = []
    
    # 循环直到凑够 20 帧
    while len(collected_frames) < num_samples:
        # 1. 随机选一个序列
        seq_idx = random.randint(0, len(dataset) - 1)
        sample = dataset[seq_idx]
        
        # sample['pointcloud'] 是骨架数据 [T, 27, 3]
        skeleton_seq = sample['pointcloud'] 
        
        # 2. 在这个序列里随机选一帧 (或者多帧)
        T = skeleton_seq.shape[0]
        if T == 0: continue
        
        frame_idx = random.randint(0, T - 1)
        frame_data = skeleton_seq[frame_idx] # [27, 3]
        
        # 转为 Tensor (以防是 numpy)
        if not isinstance(frame_data, torch.Tensor):
            frame_data = torch.from_numpy(frame_data)
            
        collected_frames.append(frame_data)

    # 堆叠成 [20, 27, 3]
    batch_tensor = torch.stack(collected_frames[:num_samples]).float()
    
    # 3. !!! 关键步骤 !!! 反归一化
    # 我们需要在真实物理空间（米/毫米）测试骨长，而不是在归一化空间
    # reader.denormalize_pointcloud 通常接受 CPU Tensor
    real_world_batch = reader.denormalize_pointcloud(batch_tensor)
    
    return real_world_batch

# 获取数据
print("正在随机抽取 20 帧...")
real_gt_data = get_random_batch_from_dataset(dataset, reader, num_samples=20)
print(f"提取数据形状: {real_gt_data.shape}") # 应该是 [20, 27, 3]

# ==========================================
# 3. 之前的测试逻辑 (适配你的 27 关节)
# ==========================================

# 你的 Parent 列表
PARENT_DICT = {
    1:0, 2:1, 3:2, 26:3,
    4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:8,
    11:3, 12:11, 13:12, 14:13, 15:14, 16:15, 17:15,
    18:0, 19:18, 20:19, 21:20,
    22:0, 23:22, 24:23, 25:24
}
parents = [-1] * 27
for child, parent in PARENT_DICT.items():
    parents[child] = parent

def gram_schmidt_rotation(start_pos, end_pos):
    B = start_pos.shape[0]
    device = start_pos.device
    y_axis = F.normalize(end_pos - start_pos + 1e-8, dim=-1)
    ref_axis = torch.tensor([0, 0, 1], device=device).float().expand(B, 3)
    x_axis = F.normalize(torch.cross(y_axis, ref_axis, dim=-1) + 1e-8, dim=-1)
    z_axis = F.normalize(torch.cross(x_axis, y_axis, dim=-1) + 1e-8, dim=-1)
    return torch.stack([x_axis, y_axis, z_axis], dim=-1)

def xyz_to_rotation_and_length(xyz):
    B, J, _ = xyz.shape
    device = xyz.device
    global_rots = torch.zeros(B, J, 3, 3, device=device)
    local_rots = torch.zeros(B, J, 3, 3, device=device)
    bone_lengths = torch.zeros(B, J, device=device)
    global_rots[:, 0] = torch.eye(3, device=device)
    local_rots[:, 0] = torch.eye(3, device=device)
    for i in range(1, J):
        p = parents[i]
        bone_lengths[:, i] = torch.norm(xyz[:, i] - xyz[:, p], dim=-1)
        global_rots[:, i] = gram_schmidt_rotation(xyz[:, p], xyz[:, i])
        R_parent_inv = global_rots[:, p].transpose(-1, -2)
        local_rots[:, i] = torch.matmul(R_parent_inv, global_rots[:, i])
    return local_rots, bone_lengths

def rotation_and_length_to_xyz(local_rots, bone_lengths):
    B, J, _, _ = local_rots.shape
    device = local_rots.device
    recons_xyz = torch.zeros(B, J, 3, device=device)
    global_rots = torch.zeros(B, J, 3, 3, device=device)
    global_rots[:, 0] = local_rots[:, 0]
    v_canonical = torch.tensor([0., 1., 0.], device=device).view(1, 3, 1)
    for i in range(1, J):
        p = parents[i]
        global_rots[:, i] = torch.matmul(global_rots[:, p], local_rots[:, i])
        offset = torch.matmul(global_rots[:, i], v_canonical) * bone_lengths[:, i].view(B, 1, 1)
        recons_xyz[:, i] = recons_xyz[:, p] + offset.squeeze(-1)
    return recons_xyz

# ==========================================
# 4. 执行测试
# ==========================================
print("\n[测试 A] 数学闭环验证 (使用动态骨长)...")
rotations, dynamic_lengths = xyz_to_rotation_and_length(real_gt_data)
recons_A = rotation_and_length_to_xyz(rotations, dynamic_lengths)

# 根节点对齐后计算误差
gt_centered = real_gt_data - real_gt_data[:, 0:1, :]
recons_A_centered = recons_A - recons_A[:, 0:1, :]
error_A = torch.norm(gt_centered - recons_A_centered, dim=-1).mean()
print(f"-> 闭环误差 (应该极小): {error_A.item():.6f}")

print("\n[测试 B] 刚体约束验证 (使用平均骨长)...")
# 算出这 20 个样本的平均体型 (假设他们体型接近，或者我们想看如果不接近会发生什么)
# 如果这 20 帧来自不同志愿者，这个测试误差会偏大，这是符合预期的
avg_lengths = dynamic_lengths.mean(dim=0, keepdim=True).expand(20, -1)
recons_B = rotation_and_length_to_xyz(rotations, avg_lengths)
recons_B_centered = recons_B - recons_B[:, 0:1, :]

error_B = torch.norm(gt_centered - recons_B_centered, dim=-1).mean()
print(f"-> 刚体化误差: {error_B.item():.6f}")

print("\n------------------------------")
print("结果判定：")
if error_A < 1e-4:
    print("✅ 测试A通过：IK算法完美适配你的数据集。")
else:
    print("❌ 测试A失败：请检查 Parents 列表是否正确，或数据是否存在 NaN。")

print(f"关于测试B (刚体误差): {error_B.item():.6f}")
print("这个数值反映了：如果你强制把所有人变成了'标准体型'，会产生多大的坐标偏差。")
print("在训练 Refiner 时，我们会传入 dynamic_lengths (GT)，所以训练时这个误差实际上是 0。")