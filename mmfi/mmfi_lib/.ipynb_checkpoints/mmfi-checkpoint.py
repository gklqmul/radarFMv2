import os
import scipy.io as scio
import glob
import cv2
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
                        

                            # [关键修改] 先检查存在性，再加入列表
                        if os.path.exists(file_path):
                           
                           data_info.append(data_dict)
                else:
                    raise ValueError('Unsupport data unit!')
                    
        return data_info

    def read_dir(self, dir_path):
        """
        策略：
        1. 顺序尝试读取 frame001 到 frame297。
        2. 如果文件存在 -> 读取并加入列表，记录索引。
        3. 如果文件不存在 -> 直接跳过（不占位）。
        """
        _, mod = os.path.split(dir_path)
        
        
        if mod != 'mmwave':
             # 简单兼容其他模态的逻辑 (如果有的话)
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
                        # 注意：原始代码可能是 float64，reshape 成 (-1, 5)
                        data = np.frombuffer(raw_data, dtype=np.float64).copy().reshape(-1, 5)
                        
                    data_list.append(data)
                    valid_indices.append(i) # 记录这是原来的第 i 帧
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    # 读坏了也当做不存在，直接跳过
                    continue
            # 文件不存在 -> 直接 continue，什么都不做
        
        return data_list, valid_indices

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 1. 加载完整的 GT (297, label_dim)
        gt_numpy = np.load(item['gt_path'])
        gt_full = torch.from_numpy(gt_numpy)
        
        sample = {
            'modality': item['modality'],
            'scene': item['scene'],
            'subject': item['subject'],
            'action': item['action']
        }
        
        # 如果有 idx (Frame模式), 存下来方便 collate_fn 使用
        if 'idx' in item:
            sample['idx'] = item['idx']

        # 2. 处理 mmWave 数据
        if 'mmwave' in item['modality']:
            path_to_read = item['mmwave_path']
            
            # =================================================
            # 分支 A: Sequence 模式 (路径是文件夹)
            # =================================================
            if os.path.isdir(path_to_read):
                # 1. 读取文件夹 (会自动跳过不存在的帧)
                mmwave_list, valid_indices = self.read_dir(path_to_read)
                
                # 2. 对齐 GT (只保留有效帧对应的 GT)
                if len(valid_indices) > 0:
                    gt_valid = gt_full[valid_indices]
                else:
                    # 如果整个文件夹都读不到数据 (极端情况)
                    gt_valid = torch.empty((0, gt_full.shape[1]))

                # 3. 末尾填充 (补齐到 297 帧)
                target_len = 297
                pad_len = target_len - len(mmwave_list)
                
                if pad_len > 0:
                    # 补空数据帧
                    empty_frame = np.zeros((0, 5), dtype=np.float32) # float32更通用
                    for _ in range(pad_len):
                        mmwave_list.append(empty_frame)
                    
                    # 补 GT (补0) - 确保维度与 gt_valid 一致
                    if gt_valid.dim() == 3:
                        gt_pad_block = torch.zeros((pad_len, gt_valid.shape[1], gt_valid.shape[2]), dtype=gt_valid.dtype)
                    else:
                        gt_pad_block = torch.zeros((pad_len, gt_valid.shape[1]), dtype=gt_valid.dtype)
                    gt_final = torch.cat([gt_valid, gt_pad_block], dim=0)
                else:
                    gt_final = gt_valid

                sample['input_mmwave'] = mmwave_list
                sample['output'] = gt_final

            # =================================================
            # 分支 B: Frame 模式 (路径是文件)
            # =================================================
            else:
                # 1. 直接读取这一帧
                try:
                    with open(path_to_read, 'rb') as f:
                        raw_data = f.read()
                        # 确保使用 float32 或 float64 (取决于你的bin文件生成方式)
                        # 这里为了保险，通常雷达数据是 float64 存储的，如果读出来全是0请改 float32
                        data = np.frombuffer(raw_data, dtype=np.float64).copy().reshape(-1, 5)
                except Exception as e:
                    print(f"Error reading frame {path_to_read}: {e}")
                    data = np.zeros((0, 5), dtype=np.float32)

                # 2. 放入列表 (为了配合 collate_fn 处理参差不齐的点数)
                sample['input_mmwave'] = [data]
                
                # 3. 获取对应的 GT (使用 load_data 存好的 idx)
                frame_idx = item['idx']
                # 保持维度 (1, D)
                sample['output'] = gt_full[frame_idx].unsqueeze(0)

        # 处理其他模态 (如果需要的话)
        else:
            sample['output'] = gt_full

        return sample
    
    def read_frame(self, frame):
        _mod, _frame = os.path.split(frame)
        _, mod = os.path.split(_mod)
        if mod in ['infra1', 'infra2', 'rgb']:
            data = np.load(frame)
        elif mod == 'depth':
            data = cv2.imread(frame, cv2.IMREAD_UNCHANGED) * 0.001
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
        elif mod == 'wifi-csi':
            data = scio.loadmat(frame)['CSIamp']
            data[np.isinf(data)] = np.nan
            for i in range(10):  # 32
                temp_col = data[:, :, i]
                nan_num = np.count_nonzero(temp_col != temp_col)
                if nan_num != 0:
                    temp_not_nan_col = temp_col[temp_col == temp_col]
                    temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
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


def collate_fn_padd(batch):
    '''
    专门处理包含 mmWave (参差不齐点云) 的 Batch
    '''
    # 1. 基础信息收集 (Meta Info)
    batch_data = {
        'modality': batch[0]['modality'],
        'scene': [sample['scene'] for sample in batch],
        'subject': [sample['subject'] for sample in batch],
        'action': [sample['action'] for sample in batch],
        # 如果有 idx 就拿，没有就 None
        'idx': [sample['idx'] for sample in batch] if 'idx' in batch[0] else None
    }
    
    # 2. 处理 Output (GT)
    # 你的 GT 已经是 (297, 14) 这种整齐的 Tensor 了，直接 Stack 即可
    _output = [sample['output'] for sample in batch]
    batch_data['output'] = torch.stack(_output)

    # 3. 处理 Input (多模态)
    for mod in batch_data['modality']:
        if mod in ['mmwave', 'lidar']:
            # --- 针对参差不齐的点云数据的特殊处理 ---
            
            # 第一步：找出这个 Batch 里所有帧中，最大的点数是多少
            # batch -> sample -> frame -> shape[0]
            max_points = 0
            for sample in batch:
                data_list = sample['input_' + mod] # list of numpy arrays
                for frame in data_list:
                    if frame.shape[0] > max_points:
                        max_points = frame.shape[0]
            
            # 如果全是空帧，至少给 1 个点，防止 shape 为 0 报错
            if max_points == 0: max_points = 1
            
            # 第二步：创建一个全 0 的大张量 (Batch, 297, Max_Points, 5)
            batch_size = len(batch)
            num_frames = 297 # 你固定的帧数
            feature_dim = 5  # mmwave 特征数
            
            padded_tensor = torch.zeros((batch_size, num_frames, max_points, feature_dim), dtype=torch.float32)
            
            # 第三步：把数据填进去 (Scatter / Copy)
            for b_i, sample in enumerate(batch):
                data_list = sample['input_' + mod]
                # data_list 长度肯定是 297 (你之前 read_dir 保证的)
                for t_i, frame_data in enumerate(data_list):
                    num_pts = frame_data.shape[0]
                    if num_pts > 0:
                        # 把 numpy 转 tensor 填入对应位置
                        # [第b个样本, 第t帧, 前num_pts个点, :]
                        padded_tensor[b_i, t_i, :num_pts, :] = torch.from_numpy(frame_data).float()
            
            batch_data['input_' + mod] = padded_tensor
            
        else:
            # --- 针对 RGB/Depth 等整齐数据的处理 ---
            _input = []
            for sample in batch:
                data = sample['input_' + mod]
                if isinstance(data, np.ndarray):
                    tensor = torch.from_numpy(data).float()
                else:
                    tensor = torch.tensor(data, dtype=torch.float32)
                _input.append(tensor)
            
            # 直接堆叠 (Batch, 297, H, W, C)
            batch_data['input_' + mod] = torch.stack(_input)

    return batch_data


def make_dataloader(dataset, is_training, generator, batch_size, collate_fn_padd = collate_fn_padd):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_padd,
        shuffle=is_training,
        drop_last=is_training,
        generator=generator
    )
    return loader


