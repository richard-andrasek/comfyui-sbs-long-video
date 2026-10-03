from __future__ import annotations

import importlib.util
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .depth_refinement import refine_depth_edges, fast_global_smoother
except ImportError:
    from depth_refinement import refine_depth_edges, fast_global_smoother


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

# Internal defaults for temporal percentile normalization.
TEMPORAL_DEPTH_EMA = 0.95
TEMPORAL_DEPTH_LOW_PERCENTILE = 2.0
TEMPORAL_DEPTH_HIGH_PERCENTILE = 98.0


class DepthAnythingRunner:
    def __init__(
        self,
        model_name: str = "da3_small",
        inference_resolution: Optional[int] = 518,
        device: Optional[torch.device] = None,
        depth_normalization_method: str = "simple",
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
        self.depth_normalization_method = depth_normalization_method
        self._depth_range_ema: Optional[tuple[float, float]] = None
        self._previous_cut_luma: Optional[torch.Tensor] = None

        if depth_normalization_method not in ("simple", "ema"):
            raise ValueError("depth_normalization_method must be 'simple' or 'ema'")

    def reset_temporal_normalization(self) -> None:
        """Clear temporal depth and scene-cut state before a new video."""
        self._depth_range_ema = None
        self._previous_cut_luma = None

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
        edge_refine_method: str = "none",
    ) -> torch.Tensor:

        self.load()
        original_size = frames_bchw.shape[-2:]
        inference_frames = self._prepare_inference_frames(frames_bchw)

        if self._api_backend is not None:
            depth = self._infer_official_api(inference_frames)
        elif self._pipeline is not None:
            depth = self._infer_pipeline(inference_frames)
        else:
            depth = self._infer_model(inference_frames)

        if self.depth_normalization_method == "simple":
            # Preserve established behavior: normalize and invert at model
            # output size, then resize to the original video dimensions.
            depth = self._normalize_depth(depth)
            if invert_depth:
                depth = 1.0 - depth
            depth = self._restore_depth_size(depth, original_size)
        else:
            depth = self._restore_depth_size(depth, original_size)
            depth = self._normalize_depth_temporal(frames_bchw, depth)
            if invert_depth:
                depth = 1.0 - depth

        if edge_refine_method != "none":
            if edge_refine_method == "simple":
                depth = refine_depth_edges(
                    frames_bchw,
                    depth,
                )
            elif edge_refine_method == "fgs":
                depth = fast_global_smoother(
                    frames_bchw,
                    depth,
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

    def _infer_official_api(self, frames_bchw: torch.Tensor) -> torch.Tensor:
        frames = (frames_bchw.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        result = self._api_backend.inference(list(frames))
        depth = torch.from_numpy(np.asarray(result.depth)).float().unsqueeze(1)
        depth = depth.to(frames_bchw.device)
        return depth

    def _infer_model(self, frames_bchw: torch.Tensor) -> torch.Tensor:
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
        return predicted

    def _infer_pipeline(self, frames_bchw: torch.Tensor) -> torch.Tensor:
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
        return stacked

    def _normalize_depth_temporal(
        self,
        frames_bchw: torch.Tensor,
        depth_b1hw: torch.Tensor,
        cut_threshold: float = 0.25,
    ) -> torch.Tensor:
        """Normalize each raw depth frame using EMA-smoothed percentile bounds."""
        if frames_bchw.shape[0] != depth_b1hw.shape[0]:
            raise ValueError("Frame and depth batch sizes must match")

        normalized = []
        alpha = TEMPORAL_DEPTH_EMA
        for index in range(depth_b1hw.shape[0]):
            depth = depth_b1hw[index:index + 1]
            frame = frames_bchw[index:index + 1].to(device=depth.device, dtype=torch.float32)
            gray = (frame[:, 0:1] * 0.299 + frame[:, 1:2] * 0.587 + frame[:, 2:3] * 0.114)
            cut_luma = F.interpolate(gray, size=(32, 32), mode="area").detach()

            is_cut = False
            if self._previous_cut_luma is not None:
                is_cut = float((cut_luma - self._previous_cut_luma).abs().mean().item()) > cut_threshold
            self._previous_cut_luma = cut_luma
            if is_cut:
                self._depth_range_ema = None

            finite = torch.isfinite(depth)
            values = depth[finite]
            if values.numel() == 0:
                if self._depth_range_ema is None:
                    normalized.append(torch.nan_to_num(depth, nan=0.0, posinf=1.0, neginf=0.0))
                else:
                    low_ema, high_ema = self._depth_range_ema
                    safe = torch.nan_to_num(depth, nan=low_ema, posinf=high_ema, neginf=low_ema)
                    normalized.append(((safe - low_ema) / (high_ema - low_ema)).clamp(0.0, 1.0))
                continue

            q = torch.tensor(
                [TEMPORAL_DEPTH_LOW_PERCENTILE / 100.0, TEMPORAL_DEPTH_HIGH_PERCENTILE / 100.0],
                device=values.device,
                dtype=torch.float32,
            )
            low, high = torch.quantile(values.float(), q).tolist()
            if high - low <= 1e-6:
                if self._depth_range_ema is None:
                    normalized.append(self._normalize_depth(depth))
                else:
                    low_ema, high_ema = self._depth_range_ema
                    safe = torch.nan_to_num(depth, nan=low_ema, posinf=high_ema, neginf=low_ema)
                    normalized.append(((safe - low_ema) / (high_ema - low_ema)).clamp(0.0, 1.0))
                continue

            if self._depth_range_ema is None:
                low_ema, high_ema = low, high
            else:
                old_low, old_high = self._depth_range_ema
                low_ema = alpha * old_low + (1.0 - alpha) * low
                high_ema = alpha * old_high + (1.0 - alpha) * high
            if high_ema - low_ema <= 1e-6:
                if self._depth_range_ema is None:
                    normalized.append(self._normalize_depth(depth))
                else:
                    old_low, old_high = self._depth_range_ema
                    safe = torch.nan_to_num(depth, nan=old_low, posinf=old_high, neginf=old_low)
                    normalized.append(((safe - old_low) / (old_high - old_low)).clamp(0.0, 1.0))
                continue

            self._depth_range_ema = (low_ema, high_ema)
            safe_depth = torch.nan_to_num(depth, nan=low_ema, posinf=high_ema, neginf=low_ema)
            normalized.append(((safe_depth - low_ema) / (high_ema - low_ema)).clamp(0.0, 1.0))

        return torch.cat(normalized, dim=0)

    @staticmethod
    def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
        flat = depth.flatten(2)
        mins = flat.min(dim=-1).values.view(-1, 1, 1, 1)
        maxs = flat.max(dim=-1).values.view(-1, 1, 1, 1)
        return (depth - mins) / (maxs - mins + 1e-6)
