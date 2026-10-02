from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

import numpy as np
import torch

try:
    from .config import DEFAULTS
    from .video_io import ffmpeg_path
except ImportError:
    from config import DEFAULTS
    from video_io import ffmpeg_path


class StreamingVideoEncoder:
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = DEFAULTS.output_codec,
        crf: int = DEFAULTS.output_crf,
        preset: str = DEFAULTS.output_preset,
        pixel_format: str = DEFAULTS.pixel_format,
    ) -> None:
        self.output_path = output_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.codec = codec
        self.crf = int(crf)
        self.preset = preset
        self.pixel_format = pixel_format
        self.process: Optional[subprocess.Popen] = None

    def __enter__(self) -> "StreamingVideoEncoder":
        command = [
            ffmpeg_path(),
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.8f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            self.codec,
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-pix_fmt",
            self.pixel_format,
            self.output_path,
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        return self

    def write_frames(self, frames_bhwc: np.ndarray) -> None:
        if not self.process or not self.process.stdin:
            raise RuntimeError("Encoder process not started")
        frames_uint8 = np.clip(frames_bhwc * 255.0, 0, 255).astype(np.uint8)
        self.process.stdin.write(frames_uint8.tobytes())

    def __exit__(self, exc_type, exc, tb) -> None:
        stderr = ""
        if self.process and self.process.stdin:
            self.process.stdin.close()
        if self.process and self.process.stderr:
            stderr = self.process.stderr.read().decode("utf-8", errors="ignore")
        if self.process:
            return_code = self.process.wait(timeout=10)
            self.process = None
            if exc_type is None and return_code != 0:
                raise RuntimeError(f"ffmpeg encode failed: {stderr.strip()}")



class StreamingDepthVideoEncoder:
    """Stream normalized depth as 8-bit grayscale FFV1 in Matroska."""

    def __init__(self, output_path: str, width: int, height: int, fps: float) -> None:
        self.output_path = output_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.process: Optional[subprocess.Popen] = None

    def __enter__(self) -> "StreamingDepthVideoEncoder":
        command = [
            ffmpeg_path(), "-y", "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", f"{self.width}x{self.height}", "-r", f"{self.fps:.8f}",
            "-i", "pipe:0", "-an", "-c:v", "ffv1", "-level", "3",
            "-pix_fmt", "gray", self.output_path,
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        return self

    def write_depth(self, depth_b1hw: torch.Tensor) -> None:
        if not self.process or not self.process.stdin:
            raise RuntimeError("Depth encoder process not started")
        frames = depth_b1hw.detach().squeeze(1).clamp(0, 1).mul(255).round().to(torch.uint8).cpu().contiguous().numpy()
        self.process.stdin.write(frames.tobytes())

    def __exit__(self, exc_type, exc, tb) -> None:
        stderr = ""
        if self.process and self.process.stdin:
            self.process.stdin.close()
        if self.process and self.process.stderr:
            stderr = self.process.stderr.read().decode("utf-8", errors="ignore")
        if self.process:
            return_code = self.process.wait(timeout=10)
            self.process = None
            if exc_type is None and return_code != 0:
                raise RuntimeError(f"FFV1 depth encode failed: {stderr.strip()}")

def _stereo_metadata_args(layout: str) -> list[str]:
    """FFmpeg metadata for YouTube 3D recognition. layout is 'sbs' or 'top_bottom'."""
    mode = "left_right" if layout == "sbs" else "top_bottom"
    return ["-metadata:s:v:0", f"stereo_mode={mode}"]


def finalize_mux(
    rendered_video_path: str,
    output_path: str,
    audio_source_path: Optional[str],
    audio_start_seconds: float,
    audio_duration_seconds: float,
    audio_mode: str,
    audio_codec: str = DEFAULTS.audio_codec,
    stereo_layout: str = "sbs",
) -> str:
    stereo_meta = _stereo_metadata_args(stereo_layout)
    if audio_mode == "none" or not audio_source_path:
        command = [
            ffmpeg_path(),
            "-y",
            "-i",
            rendered_video_path,
            "-c",
            "copy",
            *stereo_meta,
            output_path,
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"ffmpeg remux failed: {completed.stderr.strip()}")
        return output_path

    command = [
        ffmpeg_path(),
        "-y",
        "-ss",
        f"{audio_start_seconds:.8f}",
        "-t",
        f"{audio_duration_seconds:.8f}",
        "-i",
        audio_source_path,
        "-i",
        rendered_video_path,
        "-map",
        "1:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        audio_codec,
        *stereo_meta,
        "-shortest",
        output_path,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {completed.stderr.strip()}")
    return output_path
