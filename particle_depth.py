from __future__ import annotations

import logging
import math
import warnings

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)
_WARNED_NO_CV2 = False


def _gaussian_kernel(sigma: float, device) -> tuple[torch.Tensor, int]:
    radius = max(1, int(math.ceil(sigma * 2)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coords / sigma).square())
    return kernel / kernel.sum(), radius


def _area_filter_cv2(candidate_bool: torch.Tensor, min_area: int, max_area: int, cv2) -> torch.Tensor:
    cand = candidate_bool.to("cpu", torch.uint8).numpy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    valid = (stats[:, cv2.CC_STAT_AREA] >= min_area) & (stats[:, cv2.CC_STAT_AREA] <= max_area)
    valid[0] = False
    keep = valid[labels]
    return torch.from_numpy(keep).to(candidate_bool.device)


def _area_filter_gpu(candidate_bool: torch.Tensor, max_area: int) -> torch.Tensor:
    global _WARNED_NO_CV2
    if not _WARNED_NO_CV2:
        LOGGER.warning("OpenCV is unavailable; particle area filtering uses an approximate GPU fallback and does not enforce min_area")
        _WARNED_NO_CV2 = True
    x = candidate_bool.float()[None, None]
    r = max(1, int(math.sqrt(max_area) // 2))
    k = 2 * r + 1
    opened = -F.max_pool2d(-x, k, 1, r)
    opened = F.max_pool2d(opened, k, 1, r)
    big = F.max_pool2d(opened, 3, 1, 1)
    return (x * (1 - big))[0, 0] > 0.5


@torch.inference_mode()
def detect_particle_mask(
    frames_bhwc: torch.Tensor,
    brightness_threshold: float = 0.85,
    saturation_threshold: float = 0.25,
    min_area: int = 1,
    max_area: int = 200,
    mask_blur: float = 1.0,
) -> torch.Tensor:
    """Find bright, low-saturation particle-like regions; return Bx1xHxW float16 masks."""
    if not 0 <= brightness_threshold <= 1 or not 0 <= saturation_threshold <= 1:
        raise ValueError("brightness_threshold and saturation_threshold must be in [0, 1]")
    if min_area < 1 or max_area < min_area:
        raise ValueError("particle area limits must satisfy 1 <= min_area <= max_area")
    if mask_blur < 0 or not math.isfinite(mask_blur):
        raise ValueError("particle_mask_blur must be finite and non-negative")
    if frames_bhwc.ndim != 4 or frames_bhwc.shape[-1] < 3:
        raise ValueError("frames must have shape BxHxWxC with at least 3 channels")
    try:
        import cv2
    except ImportError:
        cv2 = None

    b, h, w, _ = frames_bhwc.shape
    device = frames_bhwc.device
    out = torch.empty((b, 1, h, w), dtype=torch.float16, device=device)
    if mask_blur > 0:
        kernel, radius = _gaussian_kernel(mask_blur, device)
        horizontal = kernel.view(1, 1, 1, -1)
        vertical = kernel.view(1, 1, -1, 1)

    for i in range(b):
        frame = frames_bhwc[i, ..., :3]
        if frame.dtype == torch.uint8:
            rgb = frame.float().div_(255.0)
        else:
            rgb = frame.float().clamp(0, 1)
        mx = rgb.amax(-1)
        mn = rgb.amin(-1)
        sat = (mx - mn) / mx.clamp_min(1e-6)
        lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        cand = (lum > brightness_threshold) & (sat < saturation_threshold)
        del rgb, mx, mn, sat, lum

        keep = _area_filter_cv2(cand, min_area, max_area, cv2) if cv2 is not None else _area_filter_gpu(cand, max_area)
        m = keep.float()[None, None]
        if mask_blur > 0:
            m = F.conv2d(m, horizontal, padding=(0, radius))
            m = F.conv2d(m, vertical, padding=(radius, 0)).clamp_(0, 1)
        out[i] = m[0].to(torch.float16)
        del cand, keep, m
    return out


@torch.inference_mode()
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
    micro_batch: int = 8,
    return_synthetic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Blend synthetic particle depth into normalized scene depth using bounded scratch memory."""
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
    if micro_batch < 1:
        raise ValueError("micro_batch must be >= 1")

    batch, _, h, w = scene_depth_b1hw.shape
    device, dtype = scene_depth_b1hw.device, scene_depth_b1hw.dtype
    combined = torch.empty_like(scene_depth_b1hw)
    synthetic_out = torch.empty_like(scene_depth_b1hw) if return_synthetic else None
    ch, cw = max(2, h // 32), max(2, w // 32)

    for start in range(0, batch, micro_batch):
        end = min(start + micro_batch, batch)
        coarse = []
        for i in range(start, end):
            generator = torch.Generator(device="cpu").manual_seed(91027 + frame_index_start + i)
            coarse.append(torch.rand((1, 1, ch, cw), generator=generator))
        noise = F.interpolate(torch.cat(coarse).to(device), size=(h, w), mode="bilinear", align_corners=False).to(dtype)
        offset = noise.mul_(offset_max - offset_min).add_(offset_min)
        scene = scene_depth_b1hw[start:end]
        if mode == "relative":
            synth = (scene - offset).clamp_(depth_min, depth_max)
        else:
            synth = (offset - offset_min).div_(max(offset_max - offset_min, 1e-6)).mul_(depth_max - depth_min).add_(depth_min).clamp_(depth_min, depth_max)
        mask = particle_mask_b1hw[start:end].to(device=device, dtype=dtype).clamp_(0, 1).mul_(strength)
        combined[start:end] = torch.lerp(scene, synth, mask)
        if synthetic_out is not None:
            synthetic_out[start:end] = synth
    return combined, synthetic_out
