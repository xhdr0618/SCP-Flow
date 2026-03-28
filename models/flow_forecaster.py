import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_feature(batch_size: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(batch_size, dim, device=device, dtype=dtype)


class TauEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=tau.device, dtype=tau.dtype) * -scale)
        angles = tau[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.embed_dim:
            emb = F.pad(emb, (0, self.embed_dim - emb.shape[-1]))
        return self.proj(emb)


class ROIEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        hidden = max(out_dim // 2, 16)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x).flatten(1)
        return self.proj(feat)


class StructureConditionEncoder(nn.Module):
    def __init__(self, out_dim: int, roi_channels: int = 3):
        super().__init__()
        self.out_dim = out_dim
        self.scalar_proj = nn.Sequential(
            nn.Linear(2, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.disc_encoder = ROIEncoder(roi_channels, out_dim)
        self.polar_encoder = ROIEncoder(roi_channels, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        struct_cond: Optional[Dict[str, torch.Tensor]],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if struct_cond is None:
            return _zero_feature(batch_size, self.out_dim, device, dtype)

        vcdr = struct_cond.get("vcdr")
        ocod = struct_cond.get("ocod")
        if vcdr is None:
            vcdr = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        else:
            vcdr = vcdr.to(device=device, dtype=dtype).view(batch_size, 1)
        if ocod is None:
            ocod = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        else:
            ocod = ocod.to(device=device, dtype=dtype).view(batch_size, 1)

        struct_feat = self.scalar_proj(torch.cat([vcdr, ocod], dim=-1))

        disc_roi = struct_cond.get("disc_roi")
        if disc_roi is not None:
            struct_feat = struct_feat + self.disc_encoder(disc_roi.to(device=device, dtype=dtype))

        polar_roi = struct_cond.get("polar_roi")
        if polar_roi is not None:
            struct_feat = struct_feat + self.polar_encoder(polar_roi.to(device=device, dtype=dtype))

        return self.norm(struct_feat)


class FiLMResBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        groups = min(32, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.film = nn.Linear(cond_dim, channels * 4)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale1, bias1, scale2, bias2 = self.film(cond).chunk(4, dim=-1)

        h = self.norm1(x)
        h = h * (1 + scale1[:, :, None, None]) + bias1[:, :, None, None]
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = h * (1 + scale2[:, :, None, None]) + bias2[:, :, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class ProgressionFlowForecaster(nn.Module):
    def __init__(
        self,
        latent_channels: int = 4,
        cond_dim: int = 128,
        hidden_channels: int = 128,
        tau_embed_dim: int = 128,
        struct_dim: int = 128,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.cond_dim = cond_dim
        self.tau_embed = TauEmbedding(tau_embed_dim)
        self.structure_encoder = StructureConditionEncoder(struct_dim)

        self.prev_latent_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(latent_channels, cond_dim),
            nn.SiLU(),
        )

        fused_dim = cond_dim * 2 + tau_embed_dim + cond_dim + struct_dim
        self.cond_fuser = nn.Sequential(
            nn.Linear(fused_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.input_proj = nn.Conv2d(latent_channels * 2, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                FiLMResBlock(hidden_channels, hidden_channels),
                FiLMResBlock(hidden_channels, hidden_channels),
                FiLMResBlock(hidden_channels, hidden_channels),
            ]
        )
        self.velocity_head = nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, padding=1)

        self.interval_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.structure_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 2),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        tau: torch.Tensor,
        ch: torch.Tensor,
        cp: Optional[torch.Tensor] = None,
        struct_cond: Optional[Dict[str, torch.Tensor]] = None,
        prev_latent: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = z_t.shape[0]
        device = z_t.device
        dtype = z_t.dtype

        if cp is None:
            cp = torch.zeros_like(ch)
        if prev_latent is None:
            prev_latent = torch.zeros_like(z_t)

        tau_feat = self.tau_embed(tau.to(dtype=dtype))
        prev_feat = self.prev_latent_pool(prev_latent)
        struct_feat = self.structure_encoder(struct_cond, batch_size, device, dtype)

        cond_feat = self.cond_fuser(torch.cat([ch, cp, tau_feat, prev_feat, struct_feat], dim=-1))
        x = self.input_proj(torch.cat([z_t, prev_latent], dim=1))
        for block in self.blocks:
            x = block(x, cond_feat)

        velocity = self.velocity_head(x)
        tau_remain = (1.0 - tau).view(batch_size, 1, 1, 1)
        pred_target = z_t + tau_remain * velocity
        pred_interval = self.interval_head(cond_feat).squeeze(-1)
        pred_uncertainty = F.softplus(self.uncertainty_head(cond_feat).squeeze(-1)) + 1e-4
        pred_structure = self.structure_head(cond_feat)

        return {
            "velocity": velocity,
            "pred_target": pred_target,
            "pred_interval": pred_interval,
            "pred_uncertainty": pred_uncertainty,
            "pred_structure": pred_structure,
        }
