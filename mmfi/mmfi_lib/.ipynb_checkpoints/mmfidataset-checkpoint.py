
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def decode_config(config):
    all_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14',
                    'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28',
                    'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
    all_actions = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14',
                   'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27']
    train_form = {}
    val_form = {}
    # Limitation to actions (protocol)
    if config['protocol'] == 'protocol1':  # Daily actions
        actions = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
    elif config['protocol'] == 'protocol2':  # Rehabilitation actions:
        actions = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A15', 'A16', 'A24', 'A25', 'A26']
    else:
        actions = all_actions
    # Limitation to subjects and actions (split choices)
    if config['split_to_use'] == 'random_split':
        rs = config['random_split']['random_seed']
        ratio = config['random_split']['ratio']
        for action in actions:
            np.random.seed(rs)
            idx = np.random.permutation(len(all_subjects))
            idx_train = idx[:int(np.floor(ratio*len(all_subjects)))]
            idx_val = idx[int(np.floor(ratio*len(all_subjects))):]
            subjects_train = np.array(all_subjects)[idx_train].tolist()
            subjects_val = np.array(all_subjects)[idx_val].tolist()
            for subject in all_subjects:
                if subject in subjects_train:
                    if subject in train_form:
                        train_form[subject].append(action)
                    else:
                        train_form[subject] = [action]
                if subject in subjects_val:
                    if subject in val_form:
                        val_form[subject].append(action)
                    else:
                        val_form[subject] = [action]
            rs += 1
    elif config['split_to_use'] == 'cross_scene_split':
        subjects_train = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                          'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20',
                          'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']
        subjects_val = ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
        for subject in subjects_train:
            train_form[subject] = actions
        for subject in subjects_val:
            val_form[subject] = actions
    elif config['split_to_use'] == 'cross_subject_split':
        subjects_train = config['cross_subject_split']['train_dataset']['subjects']
        subjects_val = config['cross_subject_split']['val_dataset']['subjects']
        for subject in subjects_train:
            train_form[subject] = actions
        for subject in subjects_val:
            val_form[subject] = actions
    else:
        subjects_train = config['manual_split']['train_dataset']['subjects']
        subjects_val = config['manual_split']['val_dataset']['subjects']
        actions_train = config['manual_split']['train_dataset']['actions']
        actions_val = config['manual_split']['val_dataset']['actions']
        for subject in subjects_train:
            train_form[subject] = actions_train
        for subject in subjects_val:
            val_form[subject] = actions_val

    dataset_config = {'train_dataset': {'modality': config['modality'],
                                        'split': 'training',
                                        'data_form': train_form
                                        },
                      'val_dataset': {'modality': config['modality'],
                                      'split': 'validation',
                                      'data_form': val_form}}
    return dataset_config
    
class MMFi_Database:
    def __init__(self, data_root):
        self.data_root = data_root
        self.scenes = {}
        self.subjects = {}
        self.actions = {}
        self.modalities = {}
        self.load_database()

    def load_database(self):
        for scene in sorted(os.listdir(self.data_root)):
            if scene.startswith("."):
                continue
            self.scenes[scene] = {}
            for subject in sorted(os.listdir(os.path.join(self.data_root, scene))):
                if subject.startswith("."):
                    continue
                self.scenes[scene][subject] = {}
                self.subjects[subject] = {}
                for action in sorted(os.listdir(os.path.join(self.data_root, scene, subject))):
                    if action.startswith("."):
                        continue
                    self.scenes[scene][subject][action] = {}
                    self.subjects[subject][action] = {}
                    if action not in self.actions.keys():
                        self.actions[action] = {}
                    if scene not in self.actions[action].keys():
                        self.actions[action][scene] = {}
                    if subject not in self.actions[action][scene].keys():
                        self.actions[action][scene][subject] = {}
                    for modality in ['infra1', 'infra2', 'depth', 'rgb', 'lidar', 'mmwave', 'wifi-csi']:
                        data_path = os.path.join(self.data_root, scene, subject, action, modality)
                        self.scenes[scene][subject][action][modality] = data_path
                        self.subjects[subject][action][modality] = data_path
                        self.actions[action][scene][subject][modality] = data_path
                        if modality not in self.modalities.keys():
                            self.modalities[modality] = {}
                        if scene not in self.modalities[modality].keys():
                            self.modalities[modality][scene] = {}
                        if subject not in self.modalities[modality][scene].keys():
                            self.modalities[modality][scene][subject] = {}
                        if action not in self.modalities[modality][scene][subject].keys():
                            self.modalities[modality][scene][subject][action] = data_path


class MMFi_Dataset(Dataset):
    def __init__(self, data_base, data_unit, modality, split, data_form):
        self.data_base = data_base
        self.data_unit = data_unit
        self.modality = modality.split('|')
        for m in self.modality:
            assert m in ['rgb', 'infra1', 'infra2', 'depth', 'lidar', 'mmwave', 'wifi-csi']
        self.split = split
        self.data_source = data_form
        self.data_list = self.load_data()


    def get_scene(self, subject):
        if subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
            return 'E01'
        elif subject in ['S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']:
            return 'E02'
        elif subject in ['S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']:
            return 'E03'
        elif subject in ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']:
            return 'E04'
        else:
            raise ValueError('Subject does not exist in this dataset.')

    def get_data_type(self, mod):
        if mod in ["rgb", 'infra1', "infra2"]:
            return ".npy"
        elif mod in ["lidar", "mmwave"]:
            return ".bin"
        elif mod in ["depth"]:
            return ".png"
        elif mod in ["wifi-csi"]:
            return ".mat"
        else:
            raise ValueError("Unsupported modality.")

    def load_data(self):
        data_info = []
        for subject, actions in self.data_source.items():
            for action in actions:
                
                #Path 1: Sequence 模式 (读取整个文件夹)
                if self.data_unit == 'sequence':
                    data_dict = {
                        'modality': self.modality,
                        'scene': self.get_scene(subject),
                        'subject': subject,
                        'action': action,
                        'gt_path': os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, 'ground_truth.npy')
                    }
                    
                    # 检查 GT 是否存在，没有 GT 没法训练
                    if not os.path.exists(data_dict['gt_path']):
                        continue

                    # 检查各模态的文件夹是否存在
                    is_valid_sequence = True
                    for mod in self.modality:
                        mod_path = os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, mod)
                        data_dict[mod+'_path'] = mod_path
                        
                        if not os.path.isdir(mod_path):
                            is_valid_sequence = False
                            break
                    
                    if is_valid_sequence:
                        data_info.append(data_dict)

                # Path 2: Frame 模式 (单帧训练)
                elif self.data_unit == 'frame':
                    frame_num = 297
                    # 遍历 0 到 296
                    for idx in range(frame_num):
                        # 1. 先检查这一帧的 GT 是否越界 (虽然 GT 通常是完整的，但保险起见)
                        gt_file = os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, 'ground_truth.npy')
                        if not os.path.exists(gt_file):
                            continue

                        data_dict = {
                            'modality': self.modality,
                            'scene': self.get_scene(subject),
                            'subject': subject,
                            'action': action,
                            'gt_path': gt_file,
                            'idx': idx
                        }
                        
                        # 2. 检查每个模态对应的具体文件 (frameXXX.bin) 是否存在
                        mod = 'mmwave'  # 目前只有 mmwave 是按帧存储的
                        file_name = "frame{:03d}".format(idx+1) + self.get_data_type(mod)
                        file_path = os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, mod, file_name)
                        data_dict[mod+'_path'] = file_path

                        if os.path.exists(file_path):
                           
                           data_info.append(data_dict)
                else:
                    raise ValueError('Unsupport data unit!')
                    
        return data_info

    def read_dir(self, dir_path):
        _, mod = os.path.split(dir_path)        
        if mod != 'mmwave':

             pass 

        data_list = []
        valid_indices = []  # 记录这到底是第几帧 (0-based)，用于同步 GT

        # 遍历所有可能的帧名
        target_frames = 297
        for i in range(target_frames):
            idx = i + 1
            # mmwave 是 .bin 文件
            file_name = "frame{:03d}.bin".format(idx)
            file_path = os.path.join(dir_path, file_name)
            
            if os.path.exists(file_path):
                try:
                    # 读取 mmwave bin 文件
                    with open(file_path, 'rb') as f:
                        raw_data = f.read()
                        data = np.frombuffer(raw_data, dtype=np.float64).copy().reshape(-1, 5)
                        
                    data_list.append(data)
                    valid_indices.append(i) # 记录这是原来的第 i 帧
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
        
        return data_list, valid_indices
        
    def _process_mmwave_sequence(self, raw_frames_list, max_points=128, target_len=297):
        processed_sequence = []
        total_frames = len(raw_frames_list)
        for i in range(total_frames):
            temp_frames_list = []
            
            # 1. 堆叠 [-1, 0, 1] 三帧
            for t_offset in [-1, 0, 1]:
                target_idx = i + t_offset
                if 0 <= target_idx < total_frames:
                    frame = raw_frames_list[target_idx]
                    if frame.shape[0] > 0 and frame.shape[1] >= 5:
                        # 转换顺序为 X, Y, Z, Dop, SNR (基于你之前的调整 [1, 2, 0, 3, 4])
                        cleaned_frame = frame[:, [1, 2, 0, 3, 4]]
                        time_col = np.full((cleaned_frame.shape[0], 1), float(t_offset))
                        frame_with_time = np.hstack([cleaned_frame, time_col])
                        temp_frames_list.append(frame_with_time)

            # 2. 合并点云
            if len(temp_frames_list) > 0:
                stacked_points = np.vstack(temp_frames_list)
                
                # --- [新增] 限制 X, Y, Z 的范围 (ROI Filtering) ---
                # stacked_points 列索引：0:X, 1:Y, 2:Z, 3:Dop, 4:SNR, 5:Time
                # mask = (
                #     (stacked_points[:, 0] >= -1.0) & (stacked_points[:, 0] <= 1.0) &  # X 限制
                #     (stacked_points[:, 1] >= -1.0) & (stacked_points[:, 1] <= 1.5) &  # Y 限制
                #     (stacked_points[:, 2] >= 1.5)  & (stacked_points[:, 2] <= 4.0)    # Z 限制
                # )
                # stacked_points = stacked_points[mask]
                # ----------------------------------------------
            else:
                stacked_points = np.zeros((0, 6))

            # 3. 采样或补齐 (采样现在只针对范围内的点)
            processed_points = self._sample_or_pad(stacked_points, max_points)
            processed_sequence.append(processed_points)
        
        return torch.from_numpy(np.array(processed_sequence)).float()

    def _sample_or_pad(self, points, max_points=128):
        # 只有 Time Index 也是 6 列，所以这里 points 应该是 N x 6
        if points.size == 0:
            return np.zeros((max_points, 6))
            
        num_points = points.shape[0]
        if num_points > max_points:
            indices = np.random.choice(num_points, max_points, replace=False)
            return points[indices]
        elif len(points) < max_points:
            pad_size = max_points - len(points)
            indices = np.random.choice(len(points), pad_size, replace=True)
            padding = points[indices]
            return np.vstack([points, padding])
        return points

    #for mmwave original format is (y,x,z,dop,snr)没有进行对齐以及去噪
    #for GT orginal (x,y,z)
    def __getitem__(self, idx):
        item = self.data_list[idx]

        # 1. 准备 GT 数据 (T, J, 3)
        gt_numpy = np.load(item['gt_path']) # 形状通常是 (297, 17, 3)
        
        root_joint = gt_numpy[:, 0:1, :]

        gt_temp = gt_numpy - root_joint
        gt_full = torch.from_numpy(gt_temp).float()

        try:
            action_id = int(item['action'].replace('A', '')) - 1
        except:
            action_id = 0 # Fallback

        # 3. 读取 mmWave 数据列表
        raw_mmwave_list = []
        target_len = 297

        if 'mmwave' in item['modality']:
            path_to_read = item['mmwave_path']
            
            # --- Sequence 模式 ---
            if os.path.isdir(path_to_read):
                # 读取文件夹得到 list of (N, 5) numpy arrays
                mmwave_list, valid_indices = self.read_dir(path_to_read)
                
                # 对齐 GT：只取有效的 GT 帧
                if len(valid_indices) > 0:
                    gt_valid = gt_full[valid_indices]
                else:
                    gt_valid = torch.zeros((0, gt_full.shape[1], 3))
                
                # 补齐列表到 297 帧
                pad_len = target_len - len(mmwave_list)
                if pad_len > 0:
                    # --- 修改 mmwave_list 部分 ---
                    if len(mmwave_list) > 0:
                        # 获取最后一帧的内容
                        last_frame = mmwave_list[-1]
                        for _ in range(pad_len):
                            # 注意：如果 frame 是 array，直接 append 引用即可；
                            # 如果后面有修改需求，建议用 last_frame.copy()
                            mmwave_list.append(last_frame)
                    else:
                        # 万一 mmwave_list 是空的，还是只能用 0 填充，或者抛出异常
                        empty_frame = np.zeros((0, 5), dtype=np.float32)
                        for _ in range(pad_len):
                            mmwave_list.append(empty_frame)
                
                    # --- 修改 GT 部分 ---
                    # 获取 GT 的最后一帧 (形状为 [1, joints, 3])
                    last_gt_frame = gt_valid[-1:] 
                    # 将最后一帧重复 pad_len 次
                    gt_pad = last_gt_frame.repeat(pad_len, 1, 1)
                    # 拼接
                    gt_final = torch.cat([gt_valid, gt_pad], dim=0)
                else:
                    # 如果读多了，截断
                    mmwave_list = mmwave_list[:target_len]
                    gt_final = gt_valid[:target_len]
                
                raw_mmwave_list = mmwave_list
            
            # --- Frame 模式 (如果需要兼容) ---
            else:
                # 这里为了简化，Frame模式也尽量伪装成 Sequence 长度为 1 或 297
                pass 
                
        root_tensor = torch.from_numpy(gt_numpy[:, 0:1, :]).float()
        radar_tensor = self._process_mmwave_sequence(raw_mmwave_list, max_points=128, target_len=target_len)
        radar_cond = radar_tensor.clone() # 如果是 numpy
        radar_cond[:, :, :3] = radar_tensor[:, :, :3] - root_tensor
        
        # -------------------------------------------------------------
        # [返回] 严格构造模型需要的字典
        # -------------------------------------------------------------
        return {
            # 1. 雷达数据：归一化 + 维度固定 (297, 512, 6)
            'radar_cond': radar_cond,
            
            # 2. 骨架数据：归一化 + 维度固定 (297, 17, 3)
            'pointcloud': gt_final,
            
            # 3. 动作标签：必须是 LongTensor
            'action_cond': torch.tensor(action_id).long(),
            
            # 4. 其他元数据
            'length': target_len,
            'id': f"{item['subject']}_{item['action']}_{idx}"
        }
    
    def read_frame(self, frame):
        _mod, _frame = os.path.split(frame)
        _, mod = os.path.split(_mod)
        if mod in ['infra1', 'infra2', 'rgb']:
            data = np.load(frame)
        elif mod == 'lidar':
            with open(frame, 'rb') as f:
                raw_data = f.read()
                data = np.frombuffer(raw_data, dtype=np.float64)
                data = data.reshape(-1, 3)
        elif mod == 'mmwave':
            with open(frame, 'rb') as f:
                raw_data = f.read()
                data = np.frombuffer(raw_data, dtype=np.float64)
                data = data.copy().reshape(-1, 5)
                # data = data[:, :3]
        else:
            raise ValueError('Found unseen modality in this dataset.')
        return data

    def __len__(self):
        return len(self.data_list)

    


def make_dataset(dataset_root, config):
    database = MMFi_Database(dataset_root)
    config_dataset = decode_config(config)
  
    train_dataset = MMFi_Dataset(database, config['data_unit'], **config_dataset['train_dataset'])
    val_dataset = MMFi_Dataset(database, config['data_unit'], **config_dataset['val_dataset'])
    return train_dataset, val_dataset


def make_dataloader(dataset, is_training, generator, batch_size):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        drop_last=is_training,
        generator=generator
    )
    return loader


