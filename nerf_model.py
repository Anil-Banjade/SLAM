from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    if isinstance(x, tuple):
        raise TypeError("Input to positional_encoding must be a torch.Tensor, not a tuple.")
    freq_bands = 2.0 ** torch.arange(num_freqs, dtype=torch.float32, device=x.device)
    enc = [x]
    for freq in freq_bands:
        enc.append(torch.sin(freq * x))
        enc.append(torch.cos(freq * x))
    return torch.cat(enc, dim=-1)


class NeRF(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_rays(
    H: int,
    W: int,
    focal: float,
    c2w: torch.Tensor,
    device: torch.device | str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(c2w, np.ndarray):
        c2w = torch.tensor(c2w, dtype=torch.float32, device=device)
    c2w = c2w.to(device)
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy",
    )
    dirs = torch.stack(
        [
            (i - W * 0.5) / focal,
            -(j - H * 0.5) / focal,
            -torch.ones_like(i),
        ],
        -1,
    )

    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def sample_points(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    N_samples: int,
    near: float = 2.0,
    far: float = 6.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = rays_o.device
    t_vals = torch.linspace(near, far, steps=N_samples, device=device)
    t_vals = t_vals.expand(list(rays_d.shape[:-1]) + [N_samples])
    points = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., :, None]
    return points, t_vals


def render_volume_density(
    raw: torch.Tensor,
    t_vals: torch.Tensor,
    rays_d: torch.Tensor,
) -> torch.Tensor:
    rgb = torch.sigmoid(raw[..., :3])
    sigma = F.softplus(raw[..., 3])
    dists = t_vals[..., 1:] - t_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    alpha = 1.0 - torch.exp(-sigma * dists)
    T = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], -1),
        -1,
    )[..., :-1]
    weights = alpha * T
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    return rgb_map


@dataclass
class NeRFRenderConfig:
    H: int
    W: int
    focal: float
    #N_samples: int = 64
    N_samples:int=32
    #chunk: int = 1024 * 32
    chunk:int=512
    device: str = "cuda"


@torch.no_grad()
def render_image(
    model: NeRF,
    pose_c2w: torch.Tensor | np.ndarray,
    conf: NeRFRenderConfig,
) -> np.ndarray:
    device = torch.device(conf.device)
    rays_o, rays_d = get_rays(conf.H, conf.W, conf.focal, pose_c2w, device=device)
    rays_o = rays_o.reshape(-1, 3).to(device)
    rays_d = rays_d.reshape(-1, 3).to(device)

    all_rgb = []
    for i in range(0, rays_o.shape[0], conf.chunk):
        rays_o_chunk = rays_o[i : i + conf.chunk]
        rays_d_chunk = rays_d[i : i + conf.chunk]

        pts, t_vals = sample_points(
            rays_o_chunk, rays_d_chunk, conf.N_samples, near=2.0, far=6.0
        )
        embedded = positional_encoding(pts, num_freqs=10)
        raw = model(embedded)
        rgb_chunk = render_volume_density(raw, t_vals, rays_d_chunk)
        all_rgb.append(rgb_chunk)

    rgb = torch.cat(all_rgb, dim=0)
    rgb_image = rgb.reshape(conf.H, conf.W, 3).cpu().numpy()
    rgb_image = np.clip(rgb_image, 0.0, 1.0)
    return rgb_image

