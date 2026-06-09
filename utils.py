import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from einops import rearrange
from config import Config as config

def rotate_half(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: (B, T, E)
    # cos, sin: (T, E)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

def precompute_2d_rope(h, w, dim, device=config.DEVICE, theta=10000.0):
    # dim is dim_head
    half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, 2).float() / half)).to(device)
    
    y = torch.arange(h, device=device)
    x = torch.arange(w, device=device)
    
    args_y = torch.outer(y, freqs)
    args_x = torch.outer(x, freqs)
    
    def get_components(args):
        cos = args.cos().repeat_interleave(2, dim=-1)
        sin = args.sin().repeat_interleave(2, dim=-1)
        return cos, sin

    cos_y, sin_y = get_components(args_y)
    cos_x, sin_x = get_components(args_x)
    
    cos = torch.cat([cos_y.unsqueeze(1).repeat(1, w, 1), 
                     cos_x.unsqueeze(0).repeat(h, 1, 1)], dim=-1)
    sin = torch.cat([sin_y.unsqueeze(1).repeat(1, w, 1), 
                     sin_x.unsqueeze(0).repeat(h, 1, 1)], dim=-1)
    
    return rearrange(cos, 'h w d -> 1 1 (h w) d'), rearrange(sin, 'h w d -> 1 1 (h w) d')

@torch.no_grad()
def update_ema(student_model: nn.Module, target_model: nn.Module, tau: float):
    for p_student, p_target in zip(student_model.parameters(), target_model.parameters()):
        p_target.data.mul_(tau).add_(p_student.data, alpha=1.0 - tau)

def compute_dmin_2d(mask_grid: torch.Tensor, p: int = 1) -> torch.Tensor:
    """
    Computes d_min for a batch of flat mask configurations.
    Args:
        mask_grid: Tensor of shape (B_flat, H, W) where 1=Masked, 0=Context.
    Returns:
        d_min: Tensor of shape (B_flat, H, W)
    """
    B_flat, H, W = mask_grid.shape
    device = mask_grid.device

    # 1. Compute static pairwise distances between all grid coordinates
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    coords = torch.stack([y, x], dim=-1).float().view(-1, 2)  # (H*W, 2)
    dists_grid = torch.cdist(coords, coords, p=p)             # (H*W, H*W)

    # 2. Flatten mask grid to (B_flat, 1, H*W) for broadcasting
    mask_flat = mask_grid.view(B_flat, 1, -1)

    # 3. Penalize context patch channels with infinity so they are ignored by min()
    # (1 - mask_flat) is 1 at context positions, adding 1e9 to their distance values
    dists_masked = dists_grid.unsqueeze(0) + (~mask_flat).float() * 1e9

    # 4. Find the minimum distance to any valid masked token
    d_min_flat = dists_masked.min(dim=-1).values  # (B_flat, H*W)
    return d_min_flat.view(B_flat, H, W)