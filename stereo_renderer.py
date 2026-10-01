from __future__ import annotations

import torch
import torch.nn.functional as F


class GpuStereoRenderer:
    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def render(
        self,
        frames_bhwc: torch.Tensor,
        depth_b1hw: torch.Tensor,
        disparity_ratio: float,
        layout: str = "sbs",
        depth_power: float = 1.0,
    ) -> torch.Tensor:
        frames = frames_bhwc.permute(0, 3, 1, 2).contiguous().to(self.device)
        depth = depth_b1hw.to(self.device)
        height, width = depth.shape[-2:]
        disparity_span = float(disparity_ratio) * float(width)
        disparity = depth.clamp(0, 1).pow(max(depth_power, 1e-3)) * disparity_span

        base_grid = self._base_grid(height, width, frames.shape[0], frames.device)
        disp_norm = disparity.squeeze(1) * (2.0 / max(width - 1, 1))

        left_grid = base_grid.clone()
        right_grid = base_grid.clone()
        left_grid[..., 0] = left_grid[..., 0] + disp_norm
        right_grid[..., 0] = right_grid[..., 0] - disp_norm

        left = F.grid_sample(frames, left_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        right = F.grid_sample(frames, right_grid, mode="bilinear", padding_mode="zeros", align_corners=True)

        left_mask = self._oob_mask(left_grid)
        right_mask = self._oob_mask(right_grid)
        left = self._fill_holes(left, left_mask)
        right = self._fill_holes(right, right_mask)

        if layout == "top_bottom":
            stacked = torch.cat([left, right], dim=2)
        else:
            stacked = torch.cat([left, right], dim=3)
        return stacked.permute(0, 2, 3, 1).clamp(0, 1)

    @staticmethod
    def _base_grid(height: int, width: int, batch: int, device: torch.device) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, height, device=device)
        xs = torch.linspace(-1.0, 1.0, width, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid.unsqueeze(0).repeat(batch, 1, 1, 1)

    @staticmethod
    def _oob_mask(grid: torch.Tensor) -> torch.Tensor:
        x = grid[..., 0]
        y = grid[..., 1]
        mask = (x < -1.0) | (x > 1.0) | (y < -1.0) | (y > 1.0)
        return mask.unsqueeze(1)

    @staticmethod
    def _fill_holes(image: torch.Tensor, mask: torch.Tensor, iterations: int = 4) -> torch.Tensor:
        filled = image
        for _ in range(iterations):
            neighbor_sum = F.avg_pool2d(filled, kernel_size=3, stride=1, padding=1)
            filled = torch.where(mask, neighbor_sum, filled)
        return filled
