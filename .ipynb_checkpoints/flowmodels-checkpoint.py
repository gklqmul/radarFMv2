import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Optional, Callable, Tuple
from scipy.spatial import procrustes

# def collate_fn_for_cross_modal(batch): skeleton_sequences = [] radar_sequences = [] for item in batch: # 骨架序列 (Z_S 的目标) skeleton_sequences.append(item['pointcloud']) # 雷达序列 (Z_R 的输入) radar_sequences.append(item['radar_cond']) # 找到最大序列长度 max_len = max(s.shape[0] for s in skeleton_sequences) # 填充序列到相同长度 padded_skeletons = [] padded_radars = [] for skel, radar in zip(skeleton_sequences, radar_sequences): T = skel.shape[0] if T < max_len: # 填充到 max_len pad_len = max_len - T # 对于骨骼: (T, num_joints, 3) -> (max_len, num_joints, 3) skel_pad = torch.cat([skel, skel[-1:].repeat(pad_len, 1, 1)], dim=0) # 对于雷达: (T, num_points, feat_dim) -> (max_len, num_points, feat_dim) radar_pad = torch.cat([radar, radar[-1:].repeat(pad_len, 1, 1)], dim=0) padded_skeletons.append(skel_pad) padded_radars.append(radar_pad) else: padded_skeletons.append(skel) padded_radars.append(radar) # 堆叠成张量 [B, T, ...] skeleton_batch = torch.stack(padded_skeletons, dim=0) radar_batch = torch.stack(padded_radars, dim=0) return radar_batch, skeleton_batch
def collate_fn_for_cross_modal(batch):
    skeleton_sequences, radar_sequences, lengths = [], [], []
    for item in batch:
        skeleton_sequences.append(item['pointcloud'])
        radar_sequences.append(item['radar_cond'])
        lengths.append(item['length'])  # dataset 已经给了 length

    max_len = max(lengths)

    padded_skeletons, padded_radars = [], []
    for skel, radar, L in zip(skeleton_sequences, radar_sequences, lengths):
        if L < max_len:
            pad_len = max_len - L
            skel_pad = torch.cat([skel, skel[-1:].repeat(pad_len, 1, 1)], dim=0)
            radar_pad = torch.cat([radar, radar[-1:].repeat(pad_len, 1, 1)], dim=0)
        else:
            skel_pad, radar_pad = skel, radar
        padded_skeletons.append(skel_pad)
        padded_radars.append(radar_pad)

    skeleton_batch = torch.stack(padded_skeletons, dim=0)
    radar_batch = torch.stack(padded_radars, dim=0)

    lengths = torch.tensor(lengths, dtype=torch.long)
    # valid_mask: True 表示真实帧
    B = len(batch)
    T = max_len
    valid_mask = torch.arange(T).unsqueeze(0).repeat(B, 1) < lengths.unsqueeze(1)

    return radar_batch, skeleton_batch


class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int = 64):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj = nn.Sequential(
            nn.Linear(2 * (emb_dim // 2), emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, t: torch.Tensor):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        B = t.shape[0]
        device = t.device
        half = self.emb_dim // 2
        freqs = torch.exp(torch.arange(half, device=device) * -(math.log(10000.0) / half))
        args = t * freqs.unsqueeze(0) * 2 * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(emb)


class FlowField(nn.Module):
    def __init__(self, latent_dim: int, cond_dim: Optional[int] = None,
                 hidden_dim: int = 384, time_emb_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim if cond_dim is not None else latent_dim
        self.time_emb = TimeEmbedding(time_emb_dim)
        inp = latent_dim + time_emb_dim + self.cond_dim
        self.net = nn.Sequential(
            nn.Linear(inp, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        te = self.time_emb(t)  # (B, time_emb_dim)
        inp = torch.cat([x, te, cond], dim=-1)
        return self.net(inp)


class FlowMatcher:
    def __init__(self,
                 flow_field: nn.Module,
                 skeleton_encoder: nn.Module,
                 radar_encoder: nn.Module,
                 skeleton_decoder: Optional[nn.Module] = None,
                 device: str = 'cuda',
                 alpha_fn: Callable = lambda t: t,
                 alpha_dot_fn: Callable = lambda t: torch.ones_like(t),
                 latent_proj: bool = True,
                 latent_dim: Optional[int] = None):
        """
        flow_field: FlowField instance (expects forward signature flow(x, t, cond))
        skeleton_encoder: sequence -> z_s (list of tensors)
        radar_encoder: sequence -> z_r (list of tensors)
        skeleton_decoder: z -> sequence (decoder expects skeleton-latent space)
        latent_proj: whether to use learnable linear proj to align latents
        latent_dim: target flow latent dim (if None uses flow_field.latent_dim)
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.flow = flow_field.to(self.device)
        self.skel_enc = skeleton_encoder.to(self.device)
        self.rad_enc = radar_encoder.to(self.device)
        self.skel_dec = skeleton_decoder.to(self.device) if skeleton_decoder is not None else None

        self.alpha = alpha_fn
        self.alpha_dot = alpha_dot_fn

        # if latent_proj True, we will create projection layers;
        # but we do lazy init (we don't know encoder output dims yet).
        self.use_proj = bool(latent_proj)
        self.latent_dim = latent_dim if latent_dim is not None else getattr(self.flow, 'latent_dim', None)

        # projection layers (lazy created) - shapes unknown until we see encoder outputs
        self.proj_r = None       # radar_raw_dim -> flow_latent_dim
        self.proj_s = None       # skel_raw_dim  -> flow_latent_dim
        self.proj_s_inv = None   # flow_latent_dim -> skel_raw_dim (for decoding)
        # normalization after projection
        self.post_proj_norm = None  # LayerNorm over flow_latent_dim

        # store raw latent dims when discovered
        self._z_r_dim = None
        self._z_s_dim = None

    def _get_valid_frames(self, z_padded: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
            """
            辅助函数：将填充后的 Z 序列转换为扁平化的有效帧集合 (N_valid, D_latent)。
            """
            B, T_max, D_z = z_padded.shape
            mask = torch.zeros(B, T_max, device=z_padded.device).bool()
            
            # 构造掩码
            for i, length in enumerate(lengths.cpu()):
                mask[i, :length] = True
                
            return z_padded.reshape(-1, D_z)[mask.reshape(-1)]
    # -------------------------
    # Lazy initialize projection & norm layers (call once dims known)
    # -------------------------
    def _lazy_init_projs(self, z_s_dim: int, z_r_dim: int):
        """
        Create self.proj_r, self.proj_s, self.proj_s_inv, self.post_proj_norm
        z_s_dim: dimensionality of skeleton encoder raw output
        z_r_dim: dimensionality of radar encoder raw output
        """
        target_dim = self.latent_dim if self.latent_dim is not None else getattr(self.flow, 'latent_dim', None)
        if target_dim is None:
            raise ValueError("latent_dim not set and flow_field.latent_dim missing; provide latent_dim on init.")

        self._z_s_dim = int(z_s_dim)
        self._z_r_dim = int(z_r_dim)

        # create linear projections
        self.proj_r = nn.Linear(self._z_r_dim, target_dim).to(self.device)
        self.proj_s = nn.Linear(self._z_s_dim, target_dim).to(self.device)
        # inverse map to recover skeleton-latent from flow-latent for decoder
        self.proj_s_inv = nn.Linear(target_dim, self._z_s_dim).to(self.device)

        # normalization layer after projection (helps matching)
        self.post_proj_norm = nn.LayerNorm(target_dim).to(self.device)

        # remember latent_dim
        self.latent_dim = target_dim


    def compute_and_cache_latents(self, dataloader: DataLoader, out_dir: str,
                                  overwrite: bool = False, batch_mode: bool = False,
                                  batch_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute skeleton latents (zs_raw) and radar latents (zr_raw) for the dataset.
        Returns zs, zr as tensors on self.device with shape (N, D_raw).
        Important: does NOT apply projection; saves raw encoder outputs so that
        projection layers (learnable) can be trained jointly later.
        """
        os.makedirs(out_dir, exist_ok=True)
        zs_path = os.path.join(out_dir, 'zs_flat.pt')
        zr_path = os.path.join(out_dir, 'zr_flat.pt')

        if os.path.exists(zs_path) and os.path.exists(zr_path) and not overwrite:
            print(f"[FlowMatcher] Loading cached latents from {out_dir}")
            zs = torch.load(zs_path)
            zr = torch.load(zr_path)
            # move to device on return
            return zs.to(self.device), zr.to(self.device)

        # self.flow.eval()
        self.skel_enc.eval()
        self.rad_enc.eval()

        zs_flat_list = []
        zr_flat_list = []

        for skeleton_seqs, radar_seqs in tqdm(dataloader, desc="Computing & Flattening latents"):
            
            # 转移到 GPU
            radar_seqs = [r.to(self.device) for r in radar_seqs]
            skeleton_seqs = [s.to(self.device) for s in skeleton_seqs]

            with torch.no_grad():
                # E_S 和 E_R 返回 Z 序列和 lengths
                Z_S_padded, lengths = self.skel_enc(skeleton_seqs) 
                Z_R_padded, _ = self.rad_enc(radar_seqs) 
            
            # 扁平化: 只保留有效帧
            zs_flat = self._get_valid_frames(Z_S_padded, lengths)
            zr_flat = self._get_valid_frames(Z_R_padded, lengths)

            zs_flat_list.append(zs_flat)
            zr_flat_list.append(zr_flat)
            
            # 懒惰初始化投影层 (只需一次)
            if self.use_proj and (self.proj_r is None or self.proj_s is None):
                self._lazy_init_projs(z_s_dim=zs_flat.shape[-1], z_r_dim=zr_flat.shape[-1])
        
        # 堆叠所有批次的有效帧
        zs = torch.cat(zs_flat_list, dim=0).to('cpu')
        zr = torch.cat(zr_flat_list, dim=0).to('cpu')

        torch.save(zs, zs_path)
        torch.save(zr, zr_path)
        print(f"[FlowMatcher] Saved flat latents to {out_dir}")
        return zs.to(self.device), zr.to(self.device)

    def train_from_latents(self, zs_raw: torch.Tensor, zr_raw: torch.Tensor,
                           batch_size: int = 64, epochs: int = 20, lr: float = 1e-4,
                           weight_decay: float = 0.0, noise_scale: float = 0.0,
                           grad_clip: Optional[float] = None, verbose: bool = True,
                           val_loader=None, reader=None, val_num=8, save_path='best_flow.pth',
                           lambda_mom: float = 1.0):
        """
        zs_raw, zr_raw: raw encoder outputs (N, D_s_raw) and (N, D_r_raw) (torch tensors)
        Training uses dynamic projection:
          z_r_proj = norm(proj_r(zr_raw))
          z_s_proj = norm(proj_s(zs_raw))
        The flow learns mapping in projected (flow) latent space.
        """
        device = self.device
        zs_raw = zs_raw.to(device).float()
        zr_raw = zr_raw.to(device).float()
        N = zs_raw.shape[0]
        assert zr_raw.shape[0] == N

        # lazy init projections if still None (use shapes from given tensors)
        if self.use_proj and (self.proj_r is None or self.proj_s is None):
            self._lazy_init_projs(z_s_dim=zs_raw.shape[1], z_r_dim=zr_raw.shape[1])

        # build optimizer including flow + projection params + inverse proj
        params = list(self.flow.parameters())
        if self.proj_r is not None:
            params += list(self.proj_r.parameters())
        if self.proj_s is not None:
            params += list(self.proj_s.parameters())
        if self.proj_s_inv is not None:
            params += list(self.proj_s_inv.parameters())
        if self.post_proj_norm is not None:
            params += list(self.post_proj_norm.parameters()) if any(p.requires_grad for p in self.post_proj_norm.parameters()) else []

        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

        best_val = float('inf')
        for ep in range(epochs):
            perm = torch.randperm(N)
            total_loss = 0.0
            seen = 0
            pbar = tqdm(range(0, N, batch_size), desc=f"FM train ep{ep}") if verbose else range(0, N, batch_size)
            for start in pbar:
                batch_idx = perm[start: start + batch_size]
                z_s_batch_raw = zs_raw[batch_idx].to(device)  # (B, D_s_raw)
                z_r_batch_raw = zr_raw[batch_idx].to(device)  # (B, D_r_raw)

                if self.proj_r is not None:
                    z_r = self.proj_r(z_r_batch_raw)
                else:
                    z_r = z_r_batch_raw

                if self.proj_s is not None:
                    z_s = self.proj_s(z_s_batch_raw)
                else:
                    z_s = z_s_batch_raw

                # post-projection normalization (LayerNorm)
                if self.post_proj_norm is not None:
                    z_r = self.post_proj_norm(z_r)
                    z_s = self.post_proj_norm(z_s)

                B = z_r.shape[0]
                t = torch.rand(B, device=device)
                a = self.alpha(t).unsqueeze(-1)

                x_t = (1.0 - a) * z_r + a * z_s
                if noise_scale > 0.0:
                    x_t = x_t + noise_scale * torch.randn_like(x_t)

                alpha_p = self.alpha_dot(t).unsqueeze(-1)
                v_target = alpha_p * (z_s - z_r)

                v_pred = self.flow(x_t, t, z_r)

                fm_loss = F.mse_loss(v_pred, v_target)

                # simple moment matching regularizer to encourage proj_r and proj_s to align
                mom_loss = ((z_r.mean(dim=0) - z_s.mean(dim=0))**2).mean() + \
                           ((z_r.std(dim=0) - z_s.std(dim=0))**2).mean()

                loss = fm_loss + lambda_mom * mom_loss

                opt.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()

                total_loss += loss.item() * B
                seen += B
                if verbose:
                    pbar.set_postfix({'batch_mse': fm_loss.item(), 'mom': mom_loss.item()})

            avg = total_loss / seen if seen > 0 else 0.0
            print(f"[FlowMatcher] Epoch {ep} avg loss: {avg:.6e}")

            if val_loader is not None:
                decoded_denorm, mse_list, mean_mse = validate_from_radar(self, val_loader, reader=reader, num_samples=val_num)
                print(f"[FlowMatcher] Validation mean MSE (epoch {ep}): {mean_mse:.6f}")
                if mean_mse < best_val:
                    best_val = mean_mse
                    torch.save(self.flow.state_dict(), save_path)
                    # also save projection weights
                    if self.proj_r is not None:
                        torch.save(self.proj_r.state_dict(), save_path + '.proj_r.pth')
                    if self.proj_s is not None:
                        torch.save(self.proj_s.state_dict(), save_path + '.proj_s.pth')
                    if self.proj_s_inv is not None:
                        torch.save(self.proj_s_inv.state_dict(), save_path + '.proj_s_inv.pth')
                    print(f"[FlowMatcher] New best flow + projs saved to {save_path} (val {best_val:.6f})")

    def eval(self):
        """将所有内部 PyTorch 模块设置为评估模式。"""
        self.flow.eval()
        self.skel_enc.eval()
        self.rad_enc.eval()
        
        # 必须检查这些可选模块是否存在
        if self.skel_dec is not None:
            self.skel_dec.eval()
        if self.proj_r is not None:
            self.proj_r.eval()
        if self.proj_s is not None:
            self.proj_s.eval()
        if self.proj_s_inv is not None:
            self.proj_s_inv.eval()
        if self.post_proj_norm is not None:
            self.post_proj_norm.eval() 
            
        return self # 允许链式调用 (e.g., fm.eval().to(device))

    def sample(self, z_r_input: torch.Tensor, steps: int = 10000, method: str = 'rk4') -> torch.Tensor:
        """
        z_r_input: (B, D_r_raw) or (B, flow_latent_dim)
        returns: z_s_hat in flow latent space (B, flow_latent_dim) on device
        """
        self.flow.eval()
        device = self.device

        if not torch.is_tensor(z_r_input):
            z_r_input = torch.tensor(z_r_input, dtype=torch.float32)

        z_r_input = z_r_input.to(device).float()

        # If projections exist and input is raw radar latent, project+norm
        if self.proj_r is not None and z_r_input.shape[-1] == self._z_r_dim:
            z_r = self.post_proj_norm(self.proj_r(z_r_input.to(device)))
        else:
            # assume already in flow-latent space
            z_r = z_r_input

        z = z_r.clone()
        B = z.shape[0]
        dt = 1.0 / steps
        t = torch.zeros(B, device=device)

        if method == 'euler':
            for i in range(steps):
                with torch.no_grad():
                    v = self.flow(z, t, z_r)
                    z = z + dt * v
                    t = t + dt
            return z

        elif method == 'rk4':
            for i in range(steps):
                with torch.no_grad():
                    # Z_R 作为条件 C
                    k1 = self.flow(z, t, z_r)
                    k2 = self.flow(z + 0.5 * dt * k1, t + 0.5 * dt, z_r)
                    k3 = self.flow(z + 0.5 * dt * k2, t + 0.5 * dt, z_r)
                    k4 = self.flow(z + dt * k3, t + dt, z_r)
                    z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                    t = t + dt
            return z
        else:
            raise ValueError("Unknown integration method")

    # -------------------------
    # inference from raw radar sequence (encoder inside)
    # -------------------------
    def infer_from_radar_seq(self, radar_seq, steps: int = 10000, method: str = 'rk4'):
        """
        radar_seq: 单个雷达序列 (T, N, D_feat)
        返回: 重构的骨架序列 (T, V, 3) on CPU
        """
        self.flow.eval()
        self.rad_enc.eval()
        if self.skel_dec is not None:
            self.skel_dec.eval()

        if not torch.is_tensor(radar_seq):
            radar_seq = torch.tensor(radar_seq, dtype=torch.float32)

        # encode radar -> z_r_raw
        with torch.no_grad():
            z_r_raw = self.rad_enc([radar_seq.to(self.device)])  # (1, D_r_raw)
            z_r_raw = z_r_raw.to(self.device)

            # ensure projections initialized
            if self.use_proj and (self.proj_r is None or self.proj_s_inv is None):
                # need to know raw dims: create projections from observed shapes
                # assume compute_and_cache_latents was called earlier and populated dims;
                if self._z_r_dim is None or self._z_s_dim is None:
                    # fallback: infer from z_r_raw and a dummy skeleton encode (if possible)
                    # run a dummy skeleton encode to get z_s_dim (if skel_enc works on some input)
                    try:
                        dummy_pc = torch.zeros_like(radar_seq).to(self.device)  # not perfect but just to trigger
                        z_s_dummy = self.skel_enc([dummy_pc])
                        self._z_s_dim = z_s_dummy.shape[-1]
                    except Exception:
                        pass
                self._lazy_init_projs(z_s_dim=self._z_s_dim or 1, z_r_dim=self._z_r_dim or z_r_raw.shape[-1])

            # project radar -> flow-latent
            if self.proj_r is not None and z_r_raw.shape[-1] == self._z_r_dim:
                z_r = self.post_proj_norm(self.proj_r(z_r_raw))
            else:
                z_r = z_r_raw

            z_s_hat_flow = self.sample(z_r, steps=steps, method=method)  # (1, flow_latent_dim)

            # map flow-latent back to skeleton-latent (if inverse proj exists)
            if self.proj_s_inv is not None:
                z_s_hat_raw = self.proj_s_inv(z_s_hat_flow)
            else:
                z_s_hat_raw = z_s_hat_flow

            # decode skeleton sequence
            T = radar_seq.shape[0]
            decoded = self.skel_dec(z_s_hat_raw, [T])[0]  # (T, V, 3) on device
            return decoded.detach().cpu()

    def save_flow(self, path: str):
        torch.save(self.flow.state_dict(), path)
        if self.proj_r is not None:
            torch.save(self.proj_r.state_dict(), path + '.proj_r.pth')
        if self.proj_s is not None:
            torch.save(self.proj_s.state_dict(), path + '.proj_s.pth')
        if self.proj_s_inv is not None:
            torch.save(self.proj_s_inv.state_dict(), path + '.proj_s_inv.pth')
        if self.post_proj_norm is not None:
            try:
                torch.save(self.post_proj_norm.state_dict(), path + '.norm.pth')
            except Exception:
                pass

    def load_flow(self, path: str, map_location: Optional[str] = None):
        st = torch.load(path, map_location=map_location)
        self.flow.load_state_dict(st)
        # try load projs if present
        try:
            if os.path.exists(path + '.proj_r.pth'):
                self.proj_r.load_state_dict(torch.load(path + '.proj_r.pth', map_location=map_location))
            if os.path.exists(path + '.proj_s.pth'):
                self.proj_s.load_state_dict(torch.load(path + '.proj_s.pth', map_location=map_location))
            if os.path.exists(path + '.proj_s_inv.pth'):
                self.proj_s_inv.load_state_dict(torch.load(path + '.proj_s_inv.pth', map_location=map_location))
            if os.path.exists(path + '.norm.pth') and self.post_proj_norm is not None:
                self.post_proj_norm.load_state_dict(torch.load(path + '.norm.pth', map_location=map_location))
        except Exception:
            pass


def compute_p_mjpe(pred_seq: np.ndarray, target_seq: np.ndarray) -> float:
    """
    pred_seq: (T, V, 3)
    target_seq: (T, V, 3)
    returns: mean P-MJPE over frames
    """
    T, V, _ = pred_seq.shape
    errors = []

    for t in range(T):
        pred_frame = pred_seq[t]
        target_frame = target_seq[t]

        # Procrustes alignment
        mtx1, mtx2, disparity = procrustes(target_frame, pred_frame)
        errors.append(np.linalg.norm(mtx1 - mtx2, axis=-1).mean())

    return float(np.mean(errors))

def skeleton_mse(original_seqs, recon_seqs):
    """
    original_seqs: list of (T, V, 3)
    recon_seqs: same shape as original
    """
    mse_list = []
    for orig, recon in zip(original_seqs, recon_seqs):
        T = min(orig.shape[0], recon.shape[0])
        mse = ((orig[:T] - recon[:T].cpu())**2).mean().item()
        mse_list.append(mse)
    return mse_list

def validate_from_radar(fm: FlowMatcher, val_loader, reader, num_samples: int = 8, steps: int = 10000, method: str = 'rk4'):
    """
    Seq-to-Seq 验证：使用 DataLoader，逐批次运行 Flow Matching/ODE，并在物理空间计算 MSE。

    Args:
        fm: FlowMatcher 实例 (包含所有编码器、Flow Field 和投影层)。
        val_loader: 验证集 DataLoader，返回 (radar_seqs, skeleton_seqs, labels_seqs)。
        reader: DataReader 实例 (包含 denormalize_pointcloud)。
        num_samples: 限制评估的样本数量。
        steps: ODE 求解的步数。
        method: ODE 求解方法 ('rk4' 或 'euler')。
        
    Returns:
        decoded_seqs_batch_one (list): 第一个批次的重构序列 (用于可视化)。
        total_mse_list (list): 所有样本的 MSE 列表。
        mean_mse (float): 所有样本的平均 MSE。
    """
    device = fm.device
    fm.eval()  # 设置 FlowMatcher 及其所有内部模块为评估模式

    total_mse_list = []
    num_processed_samples = 0
    decoded_seqs_batch_one = None

    # 确保 steps 是整数，防止 ODE 求解器出错
    steps = int(steps)

    with torch.no_grad():
        # tqdm 用于显示进度条
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating FM E2E")):
            
            # 假设 collate_fn_for_cross_modal 返回 (radar_seqs, skeleton_seqs, _)
            skeleton_seqs, radar_seqs = batch
            B_current = len(radar_seqs)

            # --- 1. 批次数量控制和数据转移 ---
            
            # 达到 num_samples 限制 (如果当前批次超出，则截断)
            if num_processed_samples >= num_samples: 
                break
            
            if num_processed_samples + B_current > num_samples:
                 B_current = num_samples - num_processed_samples
                 radar_seqs = radar_seqs[:B_current]
                 skeleton_seqs = skeleton_seqs[:B_current]
            
            if B_current == 0:
                break
                
            # 将序列张量转移到 GPU
            radar_seqs = [r.to(device) for r in radar_seqs]
            skeleton_seqs = [s.to(device) for s in skeleton_seqs]

            # 2. 编码 Z_R 和 Z_S (目标) 序列
            # E_R/E_S 返回 Z 序列和 lengths
            Z_R_padded, lengths = fm.rad_enc(radar_seqs)
            Z_S_target_padded, _ = fm.skel_enc(skeleton_seqs)
            
            B, T_max, D_z = Z_R_padded.shape

            # --- 3. 扁平化和 Flow 转换 (ODE 求解) ---
            
            # 扁平化：从 (B, T, D) -> (B*T, D)
            mask = torch.zeros(B, T_max, device=device).bool()
            for i, length in enumerate(lengths):
                mask[i, :length] = True
            mask_flat = mask.reshape(-1)

            Z_R_flat = Z_R_padded.reshape(-1, D_z)[mask_flat]
            Z_S_target_flat = Z_S_target_padded.reshape(-1, D_z)[mask_flat] # 真实目标 (用于 MSE 计算的 Ground Truth)
            
            # 投影 Z_R -> Flow Latent Space (Z_R 是 Z0 和 C)
            Z_R_flow = Z_R_flat
            if fm.proj_r is not None:
                Z_R_flow = fm.post_proj_norm(fm.proj_r(Z_R_flow)) 
            
            # ODE 求解 (fm.sample 接收 Z_R_flow (N_valid, D) 并返回 Z_S_flow (N_valid, D))
            Z_S_flow_flat = fm.sample(Z_R_flow, steps=steps, method=method)
            
            # 逆投影 Flow Latent -> Skeleton Raw Latent
            Z_S_raw_flat = Z_S_flow_flat
            if fm.proj_s_inv is not None:
                Z_S_raw_flat = fm.proj_s_inv(Z_S_flow_flat)
            
            # 4. 还原维度 (用于解码器输入)
            Z_S_gen_seq_padded = torch.zeros_like(Z_R_padded) 
            Z_S_gen_seq_padded.reshape(-1, D_z)[mask_flat] = Z_S_raw_flat # 填充有效帧

            # 5. 解码 (Seq-to-Seq Decoder)
            decoded_seqs = fm.skel_dec(Z_S_gen_seq_padded, lengths.tolist()) 

            # --- 6. 指标计算 (物理空间 MSE) ---
            
            # 将目标序列移动到 CPU (用于去归一化)
            target_seqs_cpu = [s.cpu() for s in skeleton_seqs]
            
            batch_mse_list = []
            for i in range(len(decoded_seqs)):
                dec_seq = decoded_seqs[i].cpu()
                orig_seq = target_seqs_cpu[i] 

                # 确保两个序列长度一致 (已由解码器裁剪)
                T = dec_seq.shape[0] # 解码器输出的序列长度是可靠的

                # 去归一化
                dec_denorm = reader.denormalize_pointcloud(dec_seq)
                orig_denorm = reader.denormalize_pointcloud(orig_seq[:T]) # 仅取原始序列的前T帧

                # 计算 MSE (通常对应 MJPE 的平方)
                mse = ((orig_denorm - dec_denorm) ** 2).mean().item()
                batch_mse_list.append(mse)
            
            total_mse_list.extend(batch_mse_list)
            num_processed_samples += B_current
            
            if batch_idx == 0:
                decoded_seqs_batch_one = decoded_seqs # 保存第一个批次的结果 (用于外部可视化)
            
    # --- 7. 最终结果 ---
    mean_mse = np.mean(total_mse_list) if total_mse_list else float('inf')

    print(f"[Validation] Total Processed Samples: {num_processed_samples}")
    print(f"[Validation] Mean MSE: {mean_mse:.6f}")

    return decoded_seqs_batch_one, total_mse_list, mean_mse