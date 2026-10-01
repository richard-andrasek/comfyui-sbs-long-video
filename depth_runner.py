from __future__ import annotations

import importlib.util
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


MODEL_CANDIDATES = {
    "da3_small": ["depth-anything/DA3-SMALL", "depth-anything/DA3-SMALL-1.1"],
    "da3_base": ["depth-anything/DA3-BASE", "depth-anything/DA3-BASE-1.1"],
    "da3_large": ["depth-anything/DA3-LARGE", "depth-anything/DA3-LARGE-1.1"],
}

# The official Depth Anything V3 package is the preferred backend.
# When it is not installed, fall back to the stable Hugging Face V2 checkpoints
# that are supported by AutoModelForDepthEstimation in current transformers.
HF_MODEL_CANDIDATES = {
    "da3_small": ["depth-anything/Depth-Anything-V2-Small-hf"],
    "da3_base": ["depth-anything/Depth-Anything-V2-Base-hf"],
    "da3_large": ["depth-anything/Depth-Anything-V2-Large-hf"],
}

def refine_depth_edges(
    frames_bchw: torch.Tensor,
    depth_b1hw: torch.Tensor,
    radius: int = 2,
    strength: float = 8.0,
) -> torch.Tensor:
    """
    Edge-aware depth smoothing.

    Smooths depth locally while reducing smoothing across strong RGB edges.

    frames_bchw: [B, 3, H, W], RGB in [0, 1]
    depth_b1hw:  [B, 1, H, W], normalized depth in [0, 1]
    """
    if radius <= 0 or strength <= 0:
        return depth_b1hw

    # Keep RGB and depth on the same device/dtype.
    frames_bchw = frames_bchw.to(
        device=depth_b1hw.device,
        dtype=depth_b1hw.dtype,
    )

    kernel_size = radius * 2 + 1

    # Convert RGB to luminance.
    gray = (
        frames_bchw[:, 0:1] * 0.299
        + frames_bchw[:, 1:2] * 0.587
        + frames_bchw[:, 2:3] * 0.114
    )

    # Estimate local RGB variance.
    local_mean = F.avg_pool2d(
        gray,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )

    local_sq_mean = F.avg_pool2d(
        gray * gray,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )

    local_variance = (
        local_sq_mean - local_mean * local_mean
    ).clamp_min(0.0)

    # Strong RGB edges -> edge_weight approaches 0.
    # Flat regions -> edge_weight approaches 1.
    edge_weight = torch.exp(-strength * local_variance)

    # Local depth average.
    depth_mean = F.avg_pool2d(
        depth_b1hw,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )

    # Smooth mostly where the RGB image is locally flat.
    refined = (
        depth_b1hw * edge_weight
        + depth_mean * (1.0 - edge_weight)
    )

    return refined.clamp(0.0, 1.0)


class DepthAnythingRunner:
    def __init__(
        self,
        model_name: str = "da3_small",
        inference_resolution: Optional[int] = 518,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model_name = model_name
        self.inference_resolution = int(inference_resolution) if inference_resolution else None
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = None
        self.model = None
        self._pipeline = None
        self._api_backend = None
        self._model_id = None

    def load(self) -> None:
        if self.model is not None or self._pipeline is not None or self._api_backend is not None:
            return

        if importlib.util.find_spec("depth_anything_3") is not None:
            self._load_official_api()
            return

        self._load_transformers_fallback()
        if self.model is not None or self._pipeline is not None:
            return

        raise RuntimeError(
            "Depth Anything V3 could not be loaded through either the official "
            "'depth_anything_3' package or the Hugging Face V2 fallback. "
            "Install the official package with: "
            "pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
        )

    def _load_official_api(self) -> None:
        from depth_anything_3.api import DepthAnything3

        errors = []
        for candidate in MODEL_CANDIDATES.get(self.model_name, [self.model_name]):
            try:
                model = DepthAnything3.from_pretrained(candidate)
                model = model.to(device=self.device)
                if self.device.type == "cuda":
                    try:
                        model = model.to(dtype=self.dtype)
                    except Exception:
                        pass
                self._api_backend = model
                self._model_id = candidate
                return
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")

        message = "\n".join(errors) if errors else "No model candidates available"
        raise RuntimeError(f"Failed to load Depth Anything model '{self.model_name}'. {message}")

    def _load_transformers_fallback(self) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation, pipeline
        except Exception:
            return

        errors = []
        for candidate in HF_MODEL_CANDIDATES.get(self.model_name, [self.model_name]):
            try:
                processor = AutoImageProcessor.from_pretrained(candidate)
                model = AutoModelForDepthEstimation.from_pretrained(candidate)
                model = model.to(device=self.device)
                if self.device.type == "cuda":
                    try:
                        model = model.to(dtype=self.dtype)
                    except Exception:
                        pass
                self.processor = processor
                self.model = model
                self._model_id = candidate
                return
            except Exception as exc:
                errors.append(f"{candidate} (direct): {exc}")

            try:
                pipe_device = 0 if self.device.type == "cuda" else -1
                self._pipeline = pipeline(task="depth-estimation", model=candidate, device=pipe_device)
                self._model_id = candidate
                return
            except Exception as exc:
                errors.append(f"{candidate} (pipeline): {exc}")

        if errors:
            message = "\n".join(errors)
            raise RuntimeError(
                f"Failed to load Depth Anything model '{self.model_name}' via transformers fallback. {message}"
            )

    def infer(
        self,
        frames_bchw: torch.Tensor,
        invert_depth: bool = False,
        edge_refine: bool = False,
        edge_radius: int = 2,
        edge_strength: float = 8.0,
    ) -> torch.Tensor:

        self.load()
        original_size = frames_bchw.shape[-2:]
        inference_frames = self._prepare_inference_frames(frames_bchw)

        if self._api_backend is not None:
            depth = self._infer_official_api(inference_frames, invert_depth=invert_depth)
        elif self._pipeline is not None:
            depth = self._infer_pipeline(inference_frames, invert_depth=invert_depth)
        else:
            depth = self._infer_model(inference_frames, invert_depth=invert_depth)

        #return self._restore_depth_size(depth, original_size)

        depth = self._restore_depth_size(depth, original_size)

        if edge_refine:
            depth = refine_depth_edges(
                frames_bchw=frames_bchw,
                depth_b1hw=depth,
                radius=edge_radius,
                strength=edge_strength,
            )

        return depth

        

    def _prepare_inference_frames(self, frames_bchw: torch.Tensor) -> torch.Tensor:
        if not self.inference_resolution:
            return frames_bchw

        target_h, target_w = self._target_size(*frames_bchw.shape[-2:])
        if (target_h, target_w) == frames_bchw.shape[-2:]:
            return frames_bchw

        return F.interpolate(frames_bchw, size=(target_h, target_w), mode="bilinear", align_corners=False)

    def _restore_depth_size(self, depth: torch.Tensor, original_size: tuple[int, int]) -> torch.Tensor:
        if depth.shape[-2:] == original_size:
            return depth
        return F.interpolate(depth, size=original_size, mode="bilinear", align_corners=False)

    def _target_size(self, height: int, width: int) -> tuple[int, int]:
        if not self.inference_resolution:
            return height, width

        longest_side = max(height, width)
        if longest_side <= self.inference_resolution:
            return height, width

        scale = self.inference_resolution / float(longest_side)
        target_h = max(1, int(round(height * scale)))
        target_w = max(1, int(round(width * scale)))
        return target_h, target_w

    def _infer_official_api(self, frames_bchw: torch.Tensor, invert_depth: bool) -> torch.Tensor:
        frames = (frames_bchw.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        result = self._api_backend.inference(list(frames))
        depth = torch.from_numpy(np.asarray(result.depth)).float().unsqueeze(1)
        depth = depth.to(frames_bchw.device)
        depth = self._normalize_depth(depth)
        return 1.0 - depth if invert_depth else depth

    def _infer_model(self, frames_bchw: torch.Tensor, invert_depth: bool) -> torch.Tensor:
        from PIL import Image

        frames = (frames_bchw.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        pil_frames = [Image.fromarray(frame) for frame in frames]
        inputs = self.processor(images=pil_frames, return_tensors="pt")
        converted_inputs = {}
        for key, value in inputs.items():
            if torch.is_floating_point(value):
                converted_inputs[key] = value.to(self.device, dtype=self.dtype)
            else:
                converted_inputs[key] = value.to(self.device)
        with torch.inference_mode():
            predicted = self.model(**converted_inputs).predicted_depth.unsqueeze(1)
        depth = self._normalize_depth(predicted)
        return 1.0 - depth if invert_depth else depth

    def _infer_pipeline(self, frames_bchw: torch.Tensor, invert_depth: bool) -> torch.Tensor:
        from PIL import Image
        outputs = []
        frames = (frames_bchw.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        for frame in frames:
            result = self._pipeline(Image.fromarray(frame))
            depth = result["depth"]
            if not isinstance(depth, torch.Tensor):
                depth = torch.as_tensor(depth)
            depth = depth.float().unsqueeze(0).unsqueeze(0)
            outputs.append(depth)
        stacked = torch.cat(outputs, dim=0)
        stacked = self._normalize_depth(stacked)
        return 1.0 - stacked if invert_depth else stacked

    @staticmethod
    def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
        flat = depth.flatten(2)
        mins = flat.min(dim=-1).values.view(-1, 1, 1, 1)
        maxs = flat.max(dim=-1).values.view(-1, 1, 1, 1)
        return (depth - mins) / (maxs - mins + 1e-6)
