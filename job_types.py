from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional

try:
    from .config import JOB_TYPE, RENDER_TYPE
except ImportError:
    from config import JOB_TYPE, RENDER_TYPE


@dataclass
class StereoVideoJob:
    source_video_path: str
    width: int
    height: int
    source_fps: float
    target_fps: float
    every_nth: int
    audio_mode: str
    stereo_layout: str
    depth_model: str
    depth_use_source_resolution: bool
    depth_inference_resolution: int
    depth_edge_refine_method: str
    chunk_size: int
    depth_power: float
    invert_depth: bool
    disparity_percent: float = 0.0
    temp_dir: Optional[str] = None
    frame_count: int = 0
    source_duration: float = 0.0
    output_depth_video: bool = False
    depth_normalization_method: str = "simple"
    global_depth_max: float = 750.0
    enable_particle_depth: bool = False
    particle_brightness_threshold: float = 0.85
    particle_saturation_threshold: float = 0.25
    particle_min_area: int = 1
    particle_max_area: int = 200
    particle_depth_min: float = 0.02
    particle_depth_max: float = 1.0
    particle_depth_offset_min: float = 0.05
    particle_depth_offset_max: float = 0.30
    particle_mask_blur: float = 1.0
    particle_depth_strength: float = 1.0
    particle_depth_mode: str = "relative"
    particle_debug_video: bool = False

    def to_handle(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["type"] = JOB_TYPE
        payload["version"] = 1
        return payload

    @classmethod
    def from_handle(cls, handle: Dict[str, Any]) -> "StereoVideoJob":
        if not isinstance(handle, dict) or handle.get("type") != JOB_TYPE:
            raise ValueError("Invalid stereo video job handle")
        payload = dict(handle)
        payload.pop("type", None)
        payload.pop("version", None)
        payload.pop("depth_mode", None)
        payload.pop("depth_video_path", None)
        if "disparity_percent" not in payload:
            payload["disparity_percent"] = payload.pop("disparity_ratio", 0.0) * 100.0
        else:
            payload.pop("disparity_ratio", None)
        if "depth_edge_refine_method" not in payload:
            payload["depth_edge_refine_method"] = "simple" if payload.pop("depth_edge_refine", False) else "none"
        payload.pop("depth_edge_radius", None)
        payload.pop("depth_edge_strength", None)
        payload.setdefault("output_depth_video", False)
        payload.setdefault("depth_normalization_method", "simple")
        if payload.get("depth_normalization_method") == "ema":
            payload["depth_normalization_method"] = "simple"
        payload.setdefault("global_depth_max", 750.0)
        particle_defaults = {
            "enable_particle_depth": False,
            "particle_brightness_threshold": 0.85,
            "particle_saturation_threshold": 0.25,
            "particle_min_area": 1,
            "particle_max_area": 200,
            "particle_depth_min": 0.02,
            "particle_depth_max": 1.0,
            "particle_depth_offset_min": 0.05,
            "particle_depth_offset_max": 0.30,
            "particle_mask_blur": 1.0,
            "particle_depth_strength": 1.0,
            "particle_depth_mode": "relative",
            "particle_debug_video": False,
        }
        for key, value in particle_defaults.items():
            payload.setdefault(key, value)
        global_depth_max = float(payload["global_depth_max"])
        if not math.isfinite(global_depth_max) or global_depth_max <= 0:
            raise ValueError("global_depth_max must be finite and greater than zero")
        payload["global_depth_max"] = global_depth_max
        # Ignore temporal tuning fields written by earlier development builds.
        payload.pop("depth_temporal_ema", None)
        payload.pop("depth_temporal_low_percentile", None)
        payload.pop("depth_temporal_high_percentile", None)
        return cls(**payload)


@dataclass
class StereoVideoRender:
    temp_video_path: str
    audio_source_path: Optional[str]
    audio_start_seconds: float
    audio_duration_seconds: float
    audio_mode: str
    cleanup_dir: Optional[str]
    frame_count: int
    fps: float
    width: int
    height: int
    stereo_layout: str
    temp_depth_video_path: Optional[str] = None
    temp_particle_debug_path: Optional[str] = None

    def to_handle(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["type"] = RENDER_TYPE
        payload["version"] = 1
        return payload

    @classmethod
    def from_handle(cls, handle: Dict[str, Any]) -> "StereoVideoRender":
        if not isinstance(handle, dict) or handle.get("type") != RENDER_TYPE:
            raise ValueError("Invalid stereo video render handle")
        payload = dict(handle)
        payload.pop("type", None)
        payload.pop("version", None)
        payload.setdefault("temp_depth_video_path", None)
        payload.setdefault("temp_particle_debug_path", None)
        return cls(**payload)
