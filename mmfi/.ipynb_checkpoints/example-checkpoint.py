import os
import argparse

import yaml
import numpy as np
import torch

from mmfi_lib.mmfi import make_dataset, make_dataloader
from mmfi_lib.evaluate import calulate_error



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code implementation with MMFi dataset and library")
    parser.add_argument("dataset_root", type=str, help="Root of Dataset")
    parser.add_argument("config_file", type=str, help="Configuration YAML file")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    train_dataset, val_dataset = make_dataset(dataset_root, config)

    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])


    print("开始检查 DataLoader 输出...")

    for batch_idx, batch_data in enumerate(train_loader):
        print(f"\n=== 正在检查第 {batch_idx} 个 Batch ===")

        if 'input_mmwave' in batch_data:
            mmwave = batch_data['input_mmwave']
            print(f"\n[mmWave Info]")
            print(f"  -> Shape: {mmwave.shape}") 

            non_zero_count = torch.count_nonzero(mmwave)
            total_elements = mmwave.numel()
            sparsity = 1.0 - (non_zero_count / total_elements)
            
            print(f"  -> 全0检查: 非零元素有 {non_zero_count} 个 (总共 {total_elements})")
            print(f"  -> 稀疏度: {sparsity:.4f} (mmWave数据通常很高，比如0.9以上，但不能是1.0)")
            
            if non_zero_count > 0:
                # 打印第一个样本，第1帧，前3个点的数据看看
                print(f"  -> 数据采样 (Sample 0, Frame 0, First 3 points):")
                print(mmwave[0, 0, :3, :])
                
                # 打印某一帧的最大点数看看 (验证是否正确 padding)
                # 找到点数最多的一帧
                points_per_frame = torch.count_nonzero(mmwave[:,:,:,0], dim=2) # 粗略估计
                print(f"  -> 这个Batch里单帧最大点数: {mmwave.shape[2]}")
                print(f"  -> 实际某帧点数统计: {points_per_frame[0, :10]} ... (前10帧)")
            else:
                print("  ⚠️ 警告：数据全是 0！请检查 read_dir 路径或 bin 文件读取逻辑。")

        # 2. 检查 GT (Output)
        if 'output' in batch_data:
            gt = batch_data['output']
            print(f"\n[GT Info]")
            print(f"  -> Shape: {gt.shape}")
            # 预期: (Batch_Size, 297, GT_Dim)
            
            # 检查 GT 是否全 0 (补零填充的部分应该是 0，但有效部分不应该是)
            print(f"  -> 数据采样 (Sample 0, Frame 0):")
            print(gt[0, 0, :])
            
            # 检查 GT 的有效长度 (通过检查是否全0行)
            # 假设 GT 是 (B, 297, D)
            non_zero_frames = (torch.sum(torch.abs(gt), dim=2) > 0).sum(dim=1)
            print(f"  -> 每个样本的有效 GT 帧数: {non_zero_frames}")
            # 这里应该能看到类似 290, 297, 250 这种数字，说明对齐逻辑生效了

        # 只看一个 Batch 就够了，看完直接退出
        break
        




