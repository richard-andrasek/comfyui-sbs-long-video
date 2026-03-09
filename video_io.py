from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator, Optional

import numpy as np

try:
    from .config import DEFAULTS
except ImportError:
    from config import DEFAULTS

try:
    import folder_paths
except ImportError:
    class _FolderPaths:
        @staticmethod
        def get_input_directory() -> str:
            return os.getcwd()

        @staticmethod
        def get_output_directory() -> str:
            return os.getcwd()

        @staticmethod
        def get_temp_directory() -> str:
            return tempfile.gettempdir()

    folder_paths = _FolderPaths()


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    has_audio: bool


@dataclass(frozen=True)
class SelectedVideoSpec:
    metadata: VideoMetadata
    start_frame: int
    end_frame: int
    every_nth: int
    target_fps: float
    selected_frame_count: int
    start_seconds: float
    selected_duration_seconds: float


VIDEO_EXTENSIONS = ("webm", "mp4", "mkv", "gif", "mov", "avi", "m4v")


def list_input_videos(include_none: bool = False) -> list[str]:
    try:
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.isdir(input_dir):
            for entry in os.listdir(input_dir):
                full_path = os.path.join(input_dir, entry)
                if os.path.isfile(full_path) and entry.lower().split(".")[-1] in VIDEO_EXTENSIONS:
                    files.append(entry)
        files = sorted(files)
        if include_none:
            return files if files else ["none"]
        return files
    except Exception:
        return ["none"] if include_none else []


def resolve_input_path(path: str) -> str:
    if not path:
        raise ValueError("Path must not be empty")
    if os.path.isabs(path):
        return path
    candidate = os.path.join(folder_paths.get_input_directory(), path)
    return candidate if os.path.exists(candidate) else os.path.abspath(path)


def resolve_output_path(path: str) -> str:
    if not path:
        raise ValueError("Output path must not be empty")
    if os.path.isabs(path):
        output_path = path
    else:
        output_path = os.path.join(folder_paths.get_output_directory(), path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return output_path


def resolve_output_prefix(prefix: str, extension: str) -> str:
    if not prefix:
        raise ValueError("Filename prefix must not be empty")
    if not extension:
        raise ValueError("Output extension must not be empty")

    output_dir = folder_paths.get_output_directory()
    os.makedirs(output_dir, exist_ok=True)

    normalized_extension = extension.lower().lstrip(".")
    base_prefix = os.path.basename(prefix).strip()
    if not base_prefix:
        raise ValueError("Filename prefix must contain at least one non-separator character")

    counter = 1
    while True:
        filename = f"{base_prefix}_{counter:05d}.{normalized_extension}"
        output_path = os.path.join(output_dir, filename)
        if not os.path.exists(output_path):
            return output_path
        counter += 1


def get_temp_dir(prefix: str = DEFAULTS.temp_prefix) -> str:
    return tempfile.mkdtemp(prefix=f"{prefix}_", dir=folder_paths.get_temp_directory())


def cleanup_temp_dir(path: Optional[str]) -> None:
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _find_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} not found on PATH")
    return path


def ffmpeg_path() -> str:
    return _find_binary("ffmpeg")


def ffprobe_path() -> str:
    return _find_binary("ffprobe")


def _parse_rate(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    return float(Fraction(value))


def probe_video(path: str) -> VideoMetadata:
    resolved = resolve_input_path(path)
    command = [
        ffprobe_path(),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        resolved,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not video_stream:
        raise RuntimeError(f"No video stream found in {resolved}")
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    fps = _parse_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    duration = float(video_stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    nb_frames = video_stream.get("nb_frames")
    if nb_frames and nb_frames.isdigit():
        frame_count = int(nb_frames)
    elif fps > 0 and duration > 0:
        frame_count = int(round(fps * duration))
    else:
        frame_count = 0
    return VideoMetadata(
        path=resolved,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        has_audio=audio_stream is not None,
    )


def select_video_range(
    metadata: VideoMetadata,
    start_frame: int = 0,
    end_frame: int = 0,
    every_nth: int = 1,
) -> SelectedVideoSpec:
    if metadata.fps <= 0:
        raise RuntimeError(f"Invalid FPS reported for {metadata.path}")
    if every_nth < 1:
        raise ValueError("every_nth must be >= 1")
    total_frames = metadata.frame_count or int(round(metadata.duration * metadata.fps))
    if total_frames <= 0:
        raise RuntimeError(f"Could not determine frame count for {metadata.path}")
    start = max(0, int(start_frame))
    stop = total_frames if not end_frame or end_frame <= 0 else min(int(end_frame), total_frames)
    if stop <= start:
        raise ValueError("end_frame must be greater than start_frame")
    selected = int(math.ceil((stop - start) / every_nth))
    target_fps = metadata.fps / every_nth
    start_seconds = start / metadata.fps
    duration_seconds = selected / target_fps
    return SelectedVideoSpec(
        metadata=metadata,
        start_frame=start,
        end_frame=stop,
        every_nth=every_nth,
        target_fps=target_fps,
        selected_frame_count=selected,
        start_seconds=start_seconds,
        selected_duration_seconds=duration_seconds,
    )


class FFmpegChunkDecoder:
    def __init__(
        self,
        spec: SelectedVideoSpec,
        chunk_size: int,
        pixel_format: str = "rgb24",
    ) -> None:
        self.spec = spec
        self.chunk_size = max(1, int(chunk_size))
        self.pixel_format = pixel_format
        self.channels = 3 if pixel_format == "rgb24" else 1
        self.frame_bytes = spec.metadata.width * spec.metadata.height * self.channels
        self._process: Optional[subprocess.Popen] = None

    def _command(self) -> list[str]:
        selector_terms = []
        if self.spec.start_frame > 0:
            selector_terms.append(f"gte(n\\,{self.spec.start_frame})")
        if self.spec.every_nth > 1:
            selector_terms.append(f"not(mod(n-{self.spec.start_frame}\\,{self.spec.every_nth}))")
        vf_parts = []
        if selector_terms:
            vf_parts.append(f"select='{ '*'.join(selector_terms) }'")
        vf_parts.append("setpts=N/FRAME_RATE/TB")
        return [
            ffmpeg_path(),
            "-v",
            "error",
            "-i",
            self.spec.metadata.path,
            "-vf",
            ",".join(vf_parts),
            "-frames:v",
            str(self.spec.selected_frame_count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            self.pixel_format,
            "-vsync",
            "0",
            "pipe:1",
        ]

    def __enter__(self) -> "FFmpegChunkDecoder":
        self._process = subprocess.Popen(
            self._command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.frame_bytes * self.chunk_size,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._process and self._process.stdout:
            self._process.stdout.close()
        if self._process and self._process.poll() is None:
            self._process.kill()
        if self._process:
            self._process.wait(timeout=2)
        self._process = None

    def __iter__(self) -> Iterator[np.ndarray]:
        if not self._process or not self._process.stdout:
            raise RuntimeError("Decoder process not started")
        while True:
            chunk = self._process.stdout.read(self.frame_bytes * self.chunk_size)
            if not chunk:
                break
            frame_count = len(chunk) // self.frame_bytes
            usable = frame_count * self.frame_bytes
            if usable == 0:
                break
            frames = np.frombuffer(chunk[:usable], dtype=np.uint8)
            shape = (frame_count, self.spec.metadata.height, self.spec.metadata.width, self.channels)
            yield frames.reshape(shape)
        if self._process.poll() not in (0, None):
            stderr = ""
            if self._process.stderr:
                stderr = self._process.stderr.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg decode failed: {stderr.strip()}")
