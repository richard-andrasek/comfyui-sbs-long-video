from __future__ import annotations

from dataclasses import asdict, dataclass
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
    depth_mode: str
    depth_video_path: Optional[str]
    depth_model: str
    depth_use_source_resolution: bool
    depth_inference_resolution: int
    depth_edge_refine_method: str
    depth_edge_radius: int
    depth_edge_strength: float
    chunk_size: int
    disparity_px: float
    depth_power: float
    invert_depth: bool
    disparity_ratio: float = 0.0
    temp_dir: Optional[str] = None
    frame_count: int = 0
    source_duration: float = 0.0

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
        payload.setdefault("disparity_ratio", 0.0)
        if "depth_edge_refine_method" not in payload:
            payload["depth_edge_refine_method"] = "simple" if payload.pop("depth_edge_refine", False) else "none"
        payload.setdefault("depth_edge_radius", 2)
        payload.setdefault("depth_edge_strength", 8.0)
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
        return cls(**payload)
