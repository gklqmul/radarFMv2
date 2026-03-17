import os
import random
from copy import copy
from typing import Counter
import torch.nn.functional as F
import plotly.graph_objects as go
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import h5py
from tqdm.auto import tqdm


class ValSubsetWithSpatialOcclusion(Dataset):
    def __init__(
        self,
        subset,
        occl_prob=1.0,
        box_size=(0.4, 0.4, 0.4),
        box_center=None,
        drop_ratio=None, 
        keep_min_points=1, # 设为1，允许极度遮挡而不触发重置
        seed=0,
    ):
        self.subset = subset
        self.occl_prob = occl_prob
        self.box_size = box_size
        self.drop_ratio = drop_ratio
        self.keep_min_points = keep_min_points
        self.rng = random.Random(seed)
        self.box_center = box_center

    def __len__(self):
        return len(self.subset)

    @torch.no_grad()
    def _apply_spatial_block(self, radar_seq):
        """
        逻辑：Box 位置固定，但每一帧在 Box 内部丢弃哪些点是独立随机的。
        """
        T, N, D = radar_seq.shape
        xyz = radar_seq[..., :3]
        device = radar_seq.device
        
        # --- 1. 确定遮挡中心 (在整个序列 T 内固定) ---
        if self.box_center is not None:
            center = torch.tensor(self.box_center, device=device, dtype=radar_seq.dtype)
        else:
            # 如果没给中心，从第一帧随机挑一个点作为中心
            xyz_f0 = xyz[0].reshape(-1, 3)
            if xyz_f0.numel() == 0: return radar_seq
            idx = self.rng.randrange(xyz_f0.shape[0])
            center = xyz_f0[idx]

        sx, sy, sz = self.box_size
        half_size = torch.tensor([sx, sy, sz], device=device) * 0.5
        lower, upper = center - half_size, center + half_size

        # --- 2. 识别 Box 内部的点 (T, N) ---
        is_in_box = (
            (xyz[..., 0] >= lower[0]) & (xyz[..., 0] <= upper[0]) &
            (xyz[..., 1] >= lower[1]) & (xyz[..., 1] <= upper[1]) &
            (xyz[..., 2] >= lower[2]) & (xyz[..., 2] <= upper[2])
        )

        out = radar_seq.clone()
        
        # --- 3. 逐帧处理遮挡 ---
        for t in range(T):
            # 获取当前帧在 Box 内的点的索引
            in_box_indices = torch.where(is_in_box[t])[0]
            out_box_indices = torch.where(~is_in_box[t])[0]
            
            if self.drop_ratio is not None and in_box_indices.numel() > 0:
                # 为了可重现，为每一帧设置独立但确定的种子
                # 种子由基础种子 + 帧号组成
                gen = torch.Generator(device=device)
                gen.manual_seed(self.rng.randint(0, 10000) + t)
                
                # 计算需要丢弃的数量
                num_in = in_box_indices.numel()
                num_drop = int(num_in * self.drop_ratio)
                
                # 随机打乱 Box 内点的索引，取前 num_drop 个作为丢弃点
                perm = torch.randperm(num_in, generator=gen, device=device)
                dropped_in_idx = in_box_indices[perm[:num_drop]]
                kept_in_idx = in_box_indices[perm[num_drop:]]
                
                # 这一帧最终保留的点 = Box 外的点 + Box 内没被丢弃的点
                kept_idx = torch.cat([out_box_indices, kept_in_idx])
            else:
                # 如果没给比例，默认 Box 内全丢 (100% drop)
                kept_idx = out_box_indices

            # --- 4. 补齐逻辑：保持点数 N ---
            if kept_idx.numel() == 0:
                # 极端情况：点全被删了，用全 0 填充
                out[t] = torch.zeros((N, D), device=device)
                continue

            if kept_idx.numel() >= N:
                sel = kept_idx[:N]
            else:
                # 随机重复存活点补齐到 N
                repeat_num = N - kept_idx.numel()
                # 补齐也需要可重现
                gen_pad = torch.Generator(device=device).manual_seed(t + 999)
                rand_idx = torch.randint(0, kept_idx.numel(), (repeat_num,), 
                                         device=device, generator=gen_pad)
                sel = torch.cat([kept_idx, kept_idx[rand_idx]])

            out[t] = radar_seq[t, sel]

        return out
        
    def visualize_occlusion(self, num_samples=2, save_dir="/code/radarFMv2/vis_occl", frame_idx=20):
        import plotly.graph_objects as go
        import numpy as np
        import os

        ratio_subdir = str(self.drop_ratio) if self.drop_ratio is not None else "fixed"
        current_save_dir = os.path.join(str(save_dir), ratio_subdir)
        os.makedirs(current_save_dir, exist_ok=True)
        
        indices = self.rng.sample(range(len(self)), min(num_samples, len(self)))
        
        for idx in indices:
            # 1. 获取完全原始的数据
            original_sample = self.subset[idx]
            radar_raw = original_sample["radar_cond"] # [T, N, D]
            gt_joints = original_sample["pointcloud"][frame_idx].cpu().numpy()
            
            T, N, D = radar_raw.shape
            xyz = radar_raw[frame_idx, :, :3].cpu().numpy()
            
            # 2. 定位 Box (使用和 apply_spatial_block 一致的逻辑)
            if self.box_center is not None:
                center = np.array(self.box_center)
            else:
                # 模拟从第一帧选中心
                center = radar_raw[frame_idx, self.rng.randrange(N), :3].cpu().numpy()

            sx, sy, sz = self.box_size
            low, high = center - np.array([sx, sy, sz])*0.5, center + np.array([sx, sy, sz])*0.5

            # 3. 手动重现丢弃逻辑
            # 找出当前帧哪些点在 Box 内
            mask_in_box = (xyz[:,0] >= low[0]) & (xyz[:,0] <= high[0]) & \
                          (xyz[:,1] >= low[1]) & (xyz[:,1] <= high[1]) & \
                          (xyz[:,2] >= low[2]) & (xyz[:,2] <= high[2])
            
            in_box_idx = np.where(mask_in_box)[0]
            out_box_points = xyz[~mask_in_box]
            
            # 在 Box 内部模拟随机丢弃
            if self.drop_ratio is not None and len(in_box_idx) > 0:
                # 模拟 apply_spatial_block 里的随机种子
                np.random.seed(self.rng.randint(0, 10000) + frame_idx)
                perm = np.random.permutation(len(in_box_idx))
                num_drop = int(len(in_box_idx) * self.drop_ratio)
                
                dropped_idx = in_box_idx[perm[:num_drop]]
                kept_in_idx = in_box_idx[perm[num_drop:]]
                
                dropped_points = xyz[dropped_idx]
                kept_in_points = xyz[kept_in_idx]
            else:
                dropped_points = xyz[in_box_idx]
                kept_in_points = np.array([]).reshape(0, 3)

            # --- 4. 绘图 ---
            fig = go.Figure()

            # 绿色：安全区点 (Box 外)
            fig.add_trace(go.Scatter3d(x=out_box_points[:,0], y=out_box_points[:,1], z=out_box_points[:,2],
                                     mode='markers', marker=dict(size=2, color='green', opacity=0.5), name='Safe Area'))
            
            # 黄色：Box 内幸存点
            if len(kept_in_points) > 0:
                fig.add_trace(go.Scatter3d(x=kept_in_points[:,0], y=kept_in_points[:,1], z=kept_in_points[:,2],
                                         mode='markers', marker=dict(size=3, color='orange'), name='Box-Kept'))

            # 红色：被删除的点 (可视化它们原本的位置)
            if len(dropped_points) > 0:
                fig.add_trace(go.Scatter3d(x=dropped_points[:,0], y=dropped_points[:,1], z=dropped_points[:,2],
                                         mode='markers', marker=dict(size=3, color='red', symbol='x'), name='Box-Dropped'))

            # 蓝色：GT 关节点
            fig.add_trace(go.Scatter3d(x=gt_joints[:,0], y=gt_joints[:,1], z=gt_joints[:,2],
                                     mode='markers', marker=dict(size=5, color='blue'), name='GT Joints'))

            # 绘制 Box 线框
            # (为了代码简洁，这里省略了具体的 bx, by, bz 坐标定义，参考之前回复即可)
            
            fig.update_layout(title=f"Occlusion Analysis | Drop Ratio: {self.drop_ratio}", scene=dict(aspectmode='data'))
            fig.write_html(os.path.join(current_save_dir, f"debug_idx{idx}.html"))
            
    def __getitem__(self, idx):
        sample = self.subset[idx]
        if "radar_cond" in sample and sample["radar_cond"].dim() == 3:
            if self.rng.random() <= self.occl_prob:
                sample = dict(sample)
                # 关键：我们只修改输入的条件 radar_cond，不修改 Ground Truth
                sample["radar_cond"] = self._apply_spatial_block(sample["radar_cond"])
        return sample
        
class RadarDiffusionDataset(Dataset):

    JOINT_CONFIG = {
        17: [0,18,22,2,3,12,13,15,5,6,8,26,19,21,23,25,27],
        24: [0,18,22,2,3,12,13,15,5,6,8,26,19,21,23,25,27,
            20,24,4,11,1,10,17],
        27: list(range(27))
    }

    def __init__(self, root_dir, scale_mode='global_unit', transform=None,sample_level='sequence',num_joints=27, parents=None):
        super().__init__()
        assert num_joints in (17, 24, 27), 'Only 17, 24, 27 joints are supported.'
        assert scale_mode in ('global_unit', None), 'Only global_unit mode is implemented.'
        self.root_dir = root_dir
        self.scale_mode = scale_mode
        self.transform = transform
        self.sample_level = sample_level
        self.num_joints = num_joints
        self.joint_indices = self.JOINT_CONFIG[num_joints]

        self.pointclouds_data = [] 
        self.stats = None
        self.label_counter = Counter()

        if parents is None:
            self.parents = [-1, 0, 1,2,3,4,5,6,7,8,8,3,11,12,13,14,15,15,0,18,19,20,0,22,23,24,3]

        self._load()
       
        self._split_indices() 
    
    def _split_indices(self):
        total_size = len(self.pointclouds_data)
        indices = list(range(total_size))
        
        # --- 关键步骤：随机打乱索引 ---
        # 建议设置一个随机种子(seed)，保证每次运行代码时，划分的结果是一致的
        # 否则你每次重启训练，验证集都在变，实验结果就没法复现了
        random.seed(42) 
        random.shuffle(indices)
        # 0.9不许改了
        train_size = int(0.9 * total_size)
        # train_size = 0
        
        # 现在这里的切片就是随机抽取的了
        self.train_indices = indices[:train_size]
        self.val_indices = indices[train_size:]
        
        

    def _collect_data_paths(self):
        datapath = []
        for env in ['env1','env2']:
            env_path = os.path.join(self.root_dir, env, 'subjects')
            if not os.path.exists(env_path):
                continue
            for subject in os.listdir(env_path):
                aligned_path = os.path.join(env_path, subject, 'aligned')
                if not os.path.exists(aligned_path):
                    continue
                for action in os.listdir(aligned_path):
                    action_path = os.path.join(aligned_path, action)
                    if not os.path.isdir(action_path):
                        continue
                    radar_files = [f for f in os.listdir(action_path) if f.endswith('.h5')]
                    skeleton_files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
                    if radar_files and skeleton_files:
                        datapath.append((os.path.join(action_path, radar_files[0]),
                                         os.path.join(action_path, skeleton_files[0])))
        return datapath

    def _process_single_sample(self, radar_path, skeleton_path):
        raw_radar_frames = []
        with h5py.File(radar_path, 'r') as f:
            for name in sorted(f["frames"].keys()):
                frame = np.array(f["frames"][name])
                raw_radar_frames.append(frame)
                
        skeleton_data = np.load(skeleton_path).astype(np.float32)

        assert len(raw_radar_frames) == len(skeleton_data)
        
        processed_data = []
        total_frames = len(raw_radar_frames)

        for i in range(total_frames):
            
            # --- 修改开始：显式处理前后帧并添加时间标签 ---
            temp_frames_list = []
            
            # 遍历上一帧(-1)、当前帧(0)、下一帧(1)
            for t_offset in [-1, 0, 1]:
                target_idx = i + t_offset
                
                # 边界检查：确保不越界
                if 0 <= target_idx < total_frames:
                    frame = raw_radar_frames[target_idx]
                    
                    # 1. 先进行清洗和特征选择 (保留原始的列选择逻辑)
                    # 假设 [5, 1, 6, 3, 7] 对应 x, y, z, doppler, snr 已经调整了
                    temp_cleaned = self._clean_and_expend(frame)[:, [5, 1, 6, 3, 7]]
                    # temp_cleaned[:, 0] = temp_cleaned[:, 0] * -1  # <--- 核心：反转雷达横向坐标
                    cleaned_frame = temp_cleaned
                    # 2. 创建时间标签列 (Time Index Channel)
                    # 形状为 [N, 1]，值为 -1.0, 0.0 或 1.0
                    time_col = np.full((cleaned_frame.shape[0], 1), float(t_offset))
                    
                    # 3. 拼接：变成 [N, 6]
                    frame_with_time = np.hstack([cleaned_frame, time_col])
                    temp_frames_list.append(frame_with_time)
            
            # 4. 堆叠这三帧 (vstack)
            if len(temp_frames_list) > 0:
                stacked_points = np.vstack(temp_frames_list)
            else:
                # 理论上不会进入这里，除非 raw_radar_frames 为空
                stacked_points = np.empty((0, 6))
            
            # 5. 去重 (Unique)
            # 注意：加入了 Time Index 后，即使空间坐标相同的点，
            # 如果来自不同帧，它们现在也是“不同”的点了（因为第6列不同），会被保留下来
            if len(stacked_points) > 0:
                stacked_points = np.unique(stacked_points, axis=0)
            # --- 修改结束 ---

            # 下采样/补齐到 128 点
            # 注意：你的 self._process_point_cloud 需要能处理 6 列数据
            processed_points = self._process_point_cloud(stacked_points)
            
            label = int(radar_path.split('action')[-1][:2]) - 1
            root_coord_mm = (skeleton_data[i][18].copy() + skeleton_data[i][22].copy()) / 2
            chest_coord_mm = (skeleton_data[i][1].copy()+skeleton_data[i][3].copy()) / 2
    
            skeleton_data[i][0]= root_coord_mm
            skeleton_data[i][2]= chest_coord_mm
            joint_data_rel = (skeleton_data[i] - root_coord_mm) / 1000
            radar_xyz = processed_points[:, :3] - root_coord_mm / 1000

            # 3. 拼回剩余的 3 维特征 (Doppler, SNR, Time)
            if processed_points.shape[1] > 3:
                radar_extra = processed_points[:, 3:]
                radar_data_rel = np.concatenate([radar_xyz, radar_extra], axis=-1)
            else:
                radar_data_rel = radar_xyz
            skeleton_data26= joint_data_rel[self.joint_indices]
            pointcloud = skeleton_data26
            
            processed_data.append({
                'radar_cond': torch.from_numpy(radar_data_rel).float(), # 现在维度是 [128, 6]
                'action_cond': torch.tensor(label).long(),
                'pointcloud': torch.from_numpy(pointcloud).float(),
                'id': f'{os.path.basename(radar_path)}_{i}',
                'root-shift': torch.from_numpy(root_coord_mm).float(),
            })
            self.label_counter[label] += 1
            
        return processed_data
    
    def _clean_and_expend(self, radar_data):
        if radar_data.shape[0] == 0 or len(radar_data.shape) != 2 or radar_data.shape[1] < 8:
            return np.zeros((1, 8))
        x = radar_data[:, 5]
        y = radar_data[:, 1]
        z = radar_data[:, 6]
        valid_mask = (x >= -1.5) & (x <= 1.5) & (z >= 1.0) & (z <= 4.5) & (y >= -1) & (y <= 2.0)
        if not np.any(valid_mask):
            return np.zeros((1, 8))
        return radar_data[valid_mask]

    def _process_point_cloud(self, points, max_points=128):
        if points.size == 0:
            points = np.zeros((0, 6))
            
        num_features = points.shape[1]
        
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            return points[indices]
        
        # 3. 补齐 (太少了就补)
        if len(points) < max_points:
            pad_size = max_points - len(points)
            indices = np.random.choice(len(points), pad_size, replace=True)
            padding = points[indices]
            return np.vstack([points, padding])
            
        return points

    def _load(self):
        datapath = self._collect_data_paths()
        print('Loading data...')
        for radar_path, skeleton_path in tqdm(datapath):
            seq = self._process_single_sample(radar_path, skeleton_path)
            if self.sample_level == 'frame': 
                for frame in seq:
                    self.pointclouds_data.append(frame)
            else:
                self.pointclouds_data.append(seq)

        print(f'Total {len(self.pointclouds_data)} samples loaded.')
     
    def __len__(self):
        return len(self.pointclouds_data)

    def __getitem__(self, idx):

        if self.sample_level == 'sequence':
            seq = self.pointclouds_data[idx]

            pcs = torch.stack([f['pointcloud'] for f in seq])   # (T, 26, 3)
            radars = torch.stack([f['radar_cond'] for f in seq]) # (T, 512, D)
            labels = torch.stack([f['action_cond'] for f in seq])
            # gt_rot_matrix, gt_lengths = xyz_to_rotation_and_length(pcs.float(), self.parents)
            # gt_rot_6d = gt_rot_matrix[..., :2].flatten(start_dim=-2)
            
            return {
                'pointcloud': pcs,
                'radar_cond': radars,
                'action_cond': labels,
                # 'gt_rot': gt_rot_6d,       # [T, 27, 6] Refiner 的 Target (旋转)
                # 'gt_len': gt_lengths / 1000 ,      # [T, 27]    用于刚体约束 FK 计算
                
                'length': pcs.shape[0],
                'id': seq[0]['id'].split('_')[0],
            }

        else:  # frame-level mode
            frame = self.pointclouds_data[idx]
            pcs = frame['pointcloud']              # (26,3)
            radars = frame['radar_cond']           # (512,D)
            label = frame['action_cond']           # scalar

            return {
                'pointcloud': pcs,
                'radar_cond': radars,
                'action_cond': label,
                'length': 1,
                'id': frame['id'],
            }

    
    def get_train_set(self):
        """返回 PyTorch Subset 对象作为训练集。"""
        return Subset(self, self.train_indices)

    def get_val_set(self):
        """返回 PyTorch Subset 对象作为验证集。"""
        return Subset(self, self.val_indices)
    
    # def get_val_set(self,box_size,occl_prob,drop_ratio,box_center):
    #     base = Subset(self, self.val_indices)
    #     # ✅ only val is corrupted
    #     return ValSubsetWithSpatialOcclusion(
    #         base,
    #         occl_prob=occl_prob,                 # 每个样本都遮挡（你也可以设 0.5）
    #         box_size=box_size,       # 遮挡块大小（米）
    #         drop_ratio=drop_ratio,               # 或者设成 0.3 之类，强制约 30% 点被遮挡
    #         box_center=box_center,
    #         keep_min_points=1,
    #         seed=0
    #     )


class DataReader:

    def __init__(self, cache_path="stats_cache.ptA"):
        self.cache_path = cache_path
        self.stats = None

    def compute_statistics(self, dataset):

        if os.path.exists(self.cache_path):
            print(f"[DataReader] Loaded cached stats: {self.cache_path}")
            self.stats = torch.load(self.cache_path)
            return self.stats

        print("[DataReader] Computing global statistics...")

        all_skeleton_points = []
        all_radar_points = []

        for item in dataset.pointclouds_data:

            # ------------- sequence-level 输入 ----------------
            if isinstance(item, list):  
                seq = item
                for frame in seq:
                    pc = frame['pointcloud']           # (26,3)
                    radar = frame['radar_cond']        # (512, D)
                    all_skeleton_points.append(pc.reshape(-1, 3))
                    all_radar_points.append(radar.reshape(-1, radar.shape[-1]))

            # ------------- frame-level 输入 -------------------
            else:  
                frame = item
                pc = frame['pointcloud']               # (26,3)
                radar = frame['radar_cond']            # (512, D)
                all_skeleton_points.append(pc.reshape(-1, 3))
                all_radar_points.append(radar.reshape(-1, radar.shape[-1]))

        all_skeleton_points = torch.cat(all_skeleton_points, dim=0)
        all_radar_points = torch.cat(all_radar_points, dim=0)

        sk_mean = all_skeleton_points.mean(dim=0)
        sk_std = all_skeleton_points.std(dim=0)

        radar_mean = all_radar_points.mean(dim=0)
        radar_std = all_radar_points.std(dim=0)

        print(f"[DataReader] Skeleton Mean: {sk_mean}, Std: {sk_std}")
        print(f"[DataReader] Radar Mean: {radar_mean}, Std: {radar_std}")

        self.stats = {
            "sk_mean": sk_mean,
            "sk_std": sk_std,
            "radar_mean": radar_mean,
            "radar_std": radar_std,
        }

        torch.save(self.stats, self.cache_path)
        print(f"[DataReader] Stats saved to {self.cache_path}")

        return self.stats


    # -----------------------------------------------------------
    #   PointCloud: 支持 2D (26,3) 和 3D (T,26,3)
    # -----------------------------------------------------------
    def normalize_pointcloud(self, pc):

        sk_mean = self.stats['sk_mean']
        sk_std = self.stats['sk_std']

        # sequence-level: (T,26,3)
        if pc.ndim == 3:
            return (pc - sk_mean.view(1,1,3)) / sk_std.view(1,1,3)

        # frame-level: (26,3)
        elif pc.ndim == 2:
            return (pc - sk_mean.view(1,3)) / sk_std.view(1,3)

        else:
            raise ValueError(f"Unsupported pc ndim: {pc.ndim}")

    def denormalize_pointcloud(self, pc:torch.Tensor):

        device = pc.device

        sk_mean = self.stats['sk_mean'].to(device)
        sk_std = self.stats['sk_std'].to(device)

        if pc.ndim == 3:
            return pc * sk_std.view(1,1,3) + sk_mean.view(1,1,3)
        elif pc.ndim == 2:
            return pc * sk_std.view(1,3) + sk_mean.view(1,3)
        elif pc.ndim == 4:
            return pc * sk_std.view(1,1,1,3) + sk_mean.view(1,1,1,3)
        else:
            raise ValueError(f"Unsupported pc ndim: {pc.ndim}")


    # -----------------------------------------------------------
    #   Radar: 支持 2D (512,D) 和 3D (T,512,D)
    # -----------------------------------------------------------
    def normalize_radar(self, radar):

        radar_mean = self.stats['radar_mean']
        radar_std = self.stats['radar_std']

        if radar.ndim == 3:
            return (radar - radar_mean.view(1,1,-1)) / radar_std.view(1,1,-1)

        elif radar.ndim == 2:
            return (radar - radar_mean.view(1,-1)) / radar_std.view(1,-1)

        else:
            raise ValueError(f"Unsupported radar ndim: {radar.ndim}")

    def denormalize_radar(self, radar):

        radar_mean = self.stats['radar_mean']
        radar_std = self.stats['radar_std']

        if radar.ndim == 3:
            return radar * radar_std.view(1,1,-1) + radar_mean.view(1,1,-1)

        elif radar.ndim == 2:
            return radar * radar_std.view(1,-1) + radar_mean.view(1,-1)

        else:
            raise ValueError(f"Unsupported radar ndim: {radar.ndim}")


# import os
# import h5py
# import numpy as np
# import torch
# from torch.utils.data import Dataset, Subset
# from collections import Counter
# from tqdm import tqdm


# class RadarDiffusionDataset(Dataset):
#     """
#     生成用 Dataset：
#       - 每帧雷达点云：stack(-1,0,+1) + unique -> 得到 N 点
#       - 下采样/补齐到 128 点
#       - radar_cond 增加 density 通道：真实有效点比例（不受 padding 影响）
#       - skeleton 做 root-relative（m -> /1000）
#     """

#     JOINT_CONFIG = {
#         17: [0, 18, 22, 2, 3, 12, 13, 15, 5, 6, 8, 26, 19, 21, 23, 25, 27],
#         24: [0, 18, 22, 2, 3, 12, 13, 15, 5, 6, 8, 26, 19, 21, 23, 25, 27,
#              20, 24, 4, 11, 1, 10, 17],
#         27: list(range(27))
#     }

#     def __init__(
#         self,
#         root_dir,
#         scale_mode='global_unit',
#         transform=None,
#         sample_level='sequence',
#         num_joints=27,
#         parents=None,
#         max_points=128,
#         # density 相关
#         density_mode="ratio",   # "ratio" or "count"
#         density_clip=True,      # 是否 clip 到 [0,1]（ratio 下有意义）
#     ):
#         super().__init__()
#         assert num_joints in (17, 24, 27), 'Only 17, 24, 27 joints are supported.'
#         assert scale_mode in ('global_unit', None), 'Only global_unit mode is implemented.'
#         assert density_mode in ("ratio", "count")

#         self.root_dir = root_dir
#         self.scale_mode = scale_mode
#         self.transform = transform
#         self.sample_level = sample_level
#         self.num_joints = num_joints
#         self.joint_indices = self.JOINT_CONFIG[num_joints]
#         self.max_points = int(max_points)

#         self.density_mode = density_mode
#         self.density_clip = density_clip

#         self.pointclouds_data = []
#         self.stats = None
#         self.label_counter = Counter()

#         if parents is None:
#             self.parents = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 3, 11, 12, 13, 14, 15, 15, 0, 18, 19, 20, 0, 22, 23, 24, 3]
#         else:
#             self.parents = parents

#         self._load()
#         self._split_indices()

#     def _split_indices(self):
#         total_size = len(self.pointclouds_data)
#         indices = list(range(total_size))

#         train_size = int(0.8 * total_size)
#         val_size = int(0.1 * total_size)

#         self.train_indices = indices[:train_size]
#         self.val_indices = indices[train_size:train_size + val_size]
#         self.test_indices = indices[train_size + val_size:]

#     def _collect_data_paths(self):
#         datapath = []
#         for env in ['env1', 'env2']:
#             env_path = os.path.join(self.root_dir, env, 'subjects')
#             if not os.path.exists(env_path):
#                 continue
#             for subject in os.listdir(env_path):
#                 aligned_path = os.path.join(env_path, subject, 'aligned')
#                 if not os.path.exists(aligned_path):
#                     continue
#                 for action in os.listdir(aligned_path):
#                     action_path = os.path.join(aligned_path, action)
#                     if not os.path.isdir(action_path):
#                         continue
#                     radar_files = [f for f in os.listdir(action_path) if f.endswith('.h5')]
#                     skeleton_files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
#                     if radar_files and skeleton_files:
#                         datapath.append((
#                             os.path.join(action_path, radar_files[0]),
#                             os.path.join(action_path, skeleton_files[0])
#                         ))
#         return datapath

#     def _process_single_sample(self, radar_path, skeleton_path):
#         raw_radar_frames = []
#         with h5py.File(radar_path, 'r') as f:
#             for name in sorted(f["frames"].keys()):
#                 frame = np.array(f["frames"][name])
#                 raw_radar_frames.append(frame)

#         skeleton_data = np.load(skeleton_path).astype(np.float32)
#         assert len(raw_radar_frames) == len(skeleton_data)

#         processed_data = []
#         total_frames = len(raw_radar_frames)

#         for i in range(total_frames):
#             # -------- 1) 三帧堆叠 + time index --------
#             temp_frames_list = []
#             for t_offset in [-1, 0, 1]:
#                 target_idx = i + t_offset
#                 if 0 <= target_idx < total_frames:
#                     frame = raw_radar_frames[target_idx]

#                     # 清洗 + 特征选择 -> [N,5] : (x,y,z,doppler,snr)
#                     temp_cleaned = self._clean_and_expend(frame)[:, [5, 1, 6, 3, 7]]
#                     cleaned_frame = temp_cleaned

#                     # time channel -> [N,1] with {-1,0,1}
#                     time_col = np.full((cleaned_frame.shape[0], 1), float(t_offset), dtype=np.float32)

#                     # concat -> [N,6]
#                     frame_with_time = np.hstack([cleaned_frame.astype(np.float32), time_col])
#                     temp_frames_list.append(frame_with_time)

#             if len(temp_frames_list) > 0:
#                 stacked_points = np.vstack(temp_frames_list)
#             else:
#                 stacked_points = np.empty((0, 6), dtype=np.float32)

#             # unique
#             if stacked_points.shape[0] > 0:
#                 stacked_points = np.unique(stacked_points, axis=0)

#             # -------- 2) 先算 density（只看补齐前真实点数）--------
#             real_n = int(stacked_points.shape[0])
#             if self.density_mode == "ratio":
#                 density = real_n / float(self.max_points)
#                 if self.density_clip:
#                     density = float(np.clip(density, 0.0, 1.0))
#             else:  # "count"
#                 density = float(real_n)

#             # -------- 3) 下采样/补齐到 128 点（返回 [128,6]）--------
#             processed_points = self._process_point_cloud(stacked_points, max_points=self.max_points)

#             # -------- 4) 计算 skeleton root/chest 并做 root-relative + meter --------
#             label = int(radar_path.split('action')[-1][:2]) - 1

#             root_coord_mm = (skeleton_data[i][18].copy() + skeleton_data[i][22].copy()) / 2
#             chest_coord_mm = (skeleton_data[i][1].copy() + skeleton_data[i][3].copy()) / 2

#             skeleton_data[i][0] = root_coord_mm
#             skeleton_data[i][2] = chest_coord_mm

#             # joint in meters, root-relative
#             joint_data_rel = (skeleton_data[i] - root_coord_mm) / 1000.0  # (J,3) in meters

#             # radar xyz also root-relative (meters)
#             radar_xyz = processed_points[:, :3] - (root_coord_mm / 1000.0)

#             # extras: doppler,snr,time  (still in original scale)
#             if processed_points.shape[1] > 3:
#                 radar_extra = processed_points[:, 3:]  # (128,3)
#                 radar_data_rel = np.concatenate([radar_xyz, radar_extra], axis=-1)  # (128,6)
#             else:
#                 radar_data_rel = radar_xyz  # (128,3)

#             # -------- 5) 增加 density 通道：把 density 复制到每个点 --------
#             # radar_data_rel: (128,6) -> (128,7)
#             density_col = np.full((radar_data_rel.shape[0], 1), density, dtype=np.float32)
#             radar_data_rel = np.concatenate([radar_data_rel.astype(np.float32), density_col], axis=-1)

#             # joints select
#             joints_sel = joint_data_rel[self.joint_indices]  # (num_joints,3)
#             pointcloud = joints_sel

#             processed_data.append({
#                 'radar_cond': torch.from_numpy(radar_data_rel).float(),   # [128, 7]
#                 'action_cond': torch.tensor(label).long(),
#                 'pointcloud': torch.from_numpy(pointcloud).float(),       # [num_joints,3]
#                 'id': f'{os.path.basename(radar_path)}_{i}',
#                 'root-shift': torch.from_numpy(root_coord_mm).float(),
#                 'density': torch.tensor(density).float(),                 # 额外保留一个 scalar（可选）
#                 'real_npoints': torch.tensor(real_n).long(),              # 可选：调试/分桶用
#             })
#             self.label_counter[label] += 1

#         return processed_data

#     def _clean_and_expend(self, radar_data):
#         if radar_data.shape[0] == 0 or len(radar_data.shape) != 2 or radar_data.shape[1] < 8:
#             return np.zeros((0, 8), dtype=np.float32)

#         x = radar_data[:, 5]
#         y = radar_data[:, 1]
#         z = radar_data[:, 6]
#         valid_mask = (x >= -1.5) & (x <= 1.5) & (z >= 1.0) & (z <= 4.5) & (y >= -1.0) & (y <= 2.0)
#         if not np.any(valid_mask):
#             return np.zeros((0, 8), dtype=np.float32)

#         return radar_data[valid_mask].astype(np.float32)

#     def _process_point_cloud(self, points, max_points=128):
#         """
#         points: [N,6]  (xyz,doppler,snr,time)
#         return: [max_points,6]
#         padding 使用全 0（不会影响 density，因为 density 已在 padding 前统计）
#         """
#         if points.size == 0:
#             points = np.zeros((0, 6), dtype=np.float32)

#         num_features = points.shape[1] if points.ndim == 2 and points.shape[1] > 0 else 6

#         if len(points) > max_points:
#             indices = np.random.choice(len(points), max_points, replace=False)
#             return points[indices].astype(np.float32)

#         if len(points) < max_points:
#             pad_size = max_points - len(points)
#             padding = np.zeros((pad_size, num_features), dtype=np.float32)
#             out = np.vstack([points.astype(np.float32), padding])
#             return out

#         return points.astype(np.float32)

#     def _load(self):
#         datapath = self._collect_data_paths()
#         print('Loading data...')
#         for radar_path, skeleton_path in tqdm(datapath):
#             seq = self._process_single_sample(radar_path, skeleton_path)
#             if self.sample_level == 'frame':
#                 for frame in seq:
#                     self.pointclouds_data.append(frame)
#             else:
#                 self.pointclouds_data.append(seq)

#         print(f'Total {len(self.pointclouds_data)} samples loaded.')

#     def __len__(self):
#         return len(self.pointclouds_data)

#     def __getitem__(self, idx):
#         if self.sample_level == 'sequence':
#             seq = self.pointclouds_data[idx]

#             pcs = torch.stack([f['pointcloud'] for f in seq], dim=0)      # (T, num_joints, 3)
#             radars = torch.stack([f['radar_cond'] for f in seq], dim=0)   # (T, 128, 7)
#             labels = torch.stack([f['action_cond'] for f in seq], dim=0)  # (T,)

#             # 可选：你也可以把 density 序列返回出来，用于可视化/分析
#             # dens = torch.stack([f['density'] for f in seq], dim=0)       # (T,)

#             return {
#                 'pointcloud': pcs,
#                 'radar_cond': radars,
#                 'action_cond': labels,
#                 'length': pcs.shape[0],
#                 'id': seq[0]['id'].split('_')[0],
#                 # 'density_seq': dens,   # 可选
#             }

#         # frame-level mode（你原代码这里没真正返回 radar/pointcloud，我保留你的结构）
#         frame = self.pointclouds_data[idx]
#         return {
#             'action_cond': frame['action_cond'],
#             'length': 1,
#             'id': frame['id'],
#         }

#     def get_train_set(self):
#         return Subset(self, self.train_indices)

#     def get_val_set(self):
#         return Subset(self, self.val_indices)

#     def get_test_set(self):
#         return Subset(self, self.test_indices)

