from __future__ import annotations

import math

import torch

try:
    from .depth_runner import finite_quantile, is_scene_start
except ImportError:
    from depth_runner import finite_quantile, is_scene_start


SCENE_Z_SHIFT_QUANTILE = 0.01  # p1 of normalized depth is placed on the screen plane for each scene.


class AdaptiveZeroParallax:
    """Per-scene zero parallax: puts each scene's p1 depth on the screen plane (a pure shift)."""

    def __init__(self, base_shift: float, scene_starts: list[int], stride: int = 1):
        self.base_shift = float(base_shift)
        self.scene_starts = sorted(set(int(x) for x in scene_starts)) or [0]
        self.stride = max(1, int(stride))
        self.applied = self.base_shift

    def shifts(
        self,
        frame_indices: list[int],
        depth_b1hw: torch.Tensor,
        depth_powers: float | list[float],
    ) -> list[float]:
        """depth_powers must be the final per-frame powers (after adaptive depth power)."""
        if not isinstance(depth_powers, list):
            depth_powers = [float(depth_powers)] * len(frame_indices)
        results = []
        for item, frame_index in enumerate(frame_indices):
            # Measured once, on the scene's first frame; every later frame reuses self.applied.
            if is_scene_start(self.scene_starts, int(frame_index), self.stride):
                p1 = finite_quantile(depth_b1hw[item], SCENE_Z_SHIFT_QUANTILE)
                if math.isfinite(p1):
                    power = max(float(depth_powers[item]), 1e-3)  # same floor as the renderer
                    scene_z_shift = min(1.0, max(0.0, p1)) ** power
                    self.applied = min(1.0, max(0.0, scene_z_shift + self.base_shift))
            results.append(self.applied)
        return results
