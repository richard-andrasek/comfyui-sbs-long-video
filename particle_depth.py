from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F


def detect_particle_mask(
    frames_bhwc: torch.Tensor,
    brightness_threshold: float = 0.85,
    saturation_threshold: float = 0.25,
    min_area: int = 1,
    max_area: int = 200,
    mask_blur: float = 1.0,
) -> torch.Tensor:
    """Find bright, low-saturation particle-like regions; returns Bx1xHxW float masks."""
    if not 0 <= brightness_threshold <= 1 or not 0 <= saturation_threshold <= 1:
        raise ValueError("brightness_threshold and saturation_threshold must be in [0, 1]")
    if min_area < 1 or max_area < min_area:
        raise ValueError("particle area limits must satisfy 1 <= min_area <= max_area")
    if mask_blur < 0 or not math.isfinite(mask_blur):
        raise ValueError("particle_mask_blur must be finite and non-negative")

    rgb = frames_bhwc[..., :3].float().clamp(0, 1)
    maximum = rgb.max(dim=-1).values
    minimum = rgb.min(dim=-1).values
    saturation = torch.where(maximum > 1e-6, (maximum - minimum) / maximum.clamp_min(1e-6), 0)
    brightness = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    candidates = (brightness > brightness_threshold) & (saturation < saturation_threshold)

    # OpenCV is available in most ComfyUI installations. Use connected components
    # when present to reject broad highlights and isolated single-pixel noise.
    try:
        import cv2
        cleaned = []
        for candidate in candidates.detach().cpu().numpy().astype(np.uint8):
            count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
            keep = np.zeros_like(candidate, dtype=np.float32)
            for label in range(1, count):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if min_area <= area <= max_area:
                    keep[labels == label] = 1.0
            cleaned.append(keep)
        mask = torch.from_numpy(np.stack(cleaned)).to(device=frames_bhwc.device, dtype=torch.float32).unsqueeze(1)
    except ImportError:
        # Dependency-free fallback retains candidates; connected-component area
        # filtering is applied when OpenCV is available in the ComfyUI runtime.
        mask = candidates.float().unsqueeze(1)

    if mask_blur > 0:
        radius = max(1, int(math.ceil(mask_blur * 2)))
        coords = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
        kernel = torch.exp(-0.5 * (coords / mask_blur).square())
        kernel = kernel / kernel.sum()
        mask = F.conv2d(mask, kernel.view(1, 1, 1, -1), padding=(0, radius))
        mask = F.conv2d(mask, kernel.view(1, 1, -1, 1), padding=(radius, 0)).clamp(0, 1)
    return mask


def apply_particle_depth(
    scene_depth_b1hw: torch.Tensor,
    particle_mask_b1hw: torch.Tensor,
    *,
    depth_min: float = 0.02,
    depth_max: float = 1.0,
    offset_min: float = 0.05,
    offset_max: float = 0.30,
    strength: float = 1.0,
    mode: str = "relative",
    frame_index_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend synthetic particle depth into normalized scene depth; also return synthetic map."""
    if not (0 <= depth_min <= depth_max <= 1):
        raise ValueError("particle depth limits must satisfy 0 <= min <= max <= 1")
    if not (0 <= offset_min <= offset_max <= 1):
        raise ValueError("particle offset limits must satisfy 0 <= min <= max <= 1")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("particle_depth_strength must be finite and in [0, 1]")
    if mode not in ("relative", "absolute"):
        raise ValueError("particle_depth_mode must be 'relative' or 'absolute'")
    if scene_depth_b1hw.shape != particle_mask_b1hw.shape:
        raise ValueError("scene depth and particle mask must have matching shapes")

    batch, _, height, width = scene_depth_b1hw.shape
    device = scene_depth_b1hw.device
    offsets = []
    for index in range(batch):
        # Coarse seeded noise gives depth variation across flakes without pixel noise;
        # indexing by source frame keeps results stable across chunk sizes.
        generator = torch.Generator(device="cpu").manual_seed(91027 + frame_index_start + index)
        noise = torch.rand((1, 1, max(2, height // 32), max(2, width // 32)), generator=generator)
        noise = F.interpolate(noise, size=(height, width), mode="bilinear", align_corners=False).to(device)
        offsets.append(offset_min + noise * (offset_max - offset_min))
    offset = torch.cat(offsets, dim=0)
    if mode == "relative":
        synthetic = (scene_depth_b1hw - offset).clamp(depth_min, depth_max)
    else:
        absolute_noise = (offset - offset_min) / max(offset_max - offset_min, 1e-6)
        synthetic = (depth_min + absolute_noise * (depth_max - depth_min)).clamp(depth_min, depth_max)
    effective_mask = particle_mask_b1hw.to(
        device=scene_depth_b1hw.device,
        dtype=scene_depth_b1hw.dtype,
    ).clamp(0, 1) * strength
    combined = scene_depth_b1hw * (1 - effective_mask) + synthetic * effective_mask
    return combined, synthetic
