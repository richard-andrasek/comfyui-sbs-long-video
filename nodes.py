from __future__ import annotations

import os
from contextlib import ExitStack
from typing import Optional

import numpy as np
import torch

try:
    from .config import CATEGORY, DEFAULTS, JOB_TYPE, RENDER_TYPE
    from .depth_runner import DepthAnythingRunner
    from .encoder import StreamingVideoEncoder, finalize_mux
    from .job_types import StereoVideoJob, StereoVideoRender
    from .stereo_renderer import GpuStereoRenderer
    from .video_io import (
        FFmpegChunkDecoder,
        cleanup_temp_dir,
        get_temp_dir,
        list_input_videos,
        probe_video,
        resolve_output_prefix,
        select_video_spec,
    )
except ImportError:
    from config import CATEGORY, DEFAULTS, JOB_TYPE, RENDER_TYPE
    from depth_runner import DepthAnythingRunner
    from encoder import StreamingVideoEncoder, finalize_mux
    from job_types import StereoVideoJob, StereoVideoRender
    from stereo_renderer import GpuStereoRenderer
    from video_io import (
        FFmpegChunkDecoder,
        cleanup_temp_dir,
        get_temp_dir,
        list_input_videos,
        probe_video,
        resolve_output_prefix,
        select_video_spec,
    )

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None


def _status_text(job: StereoVideoJob) -> str:
    return (
        f"{job.frame_count} frames at {job.target_fps:.3f} fps, "
        f"{job.width}x{job.height}, depth={job.depth_mode}, layout={job.stereo_layout}"
    )


class StereoVideoSource:
    DESCRIPTION = "Builds a long-video stereo conversion job with optional depth refinement: none, simple, or Fast Global Smoother (FGS)."
    OUTPUT_TOOLTIPS = [
        "Internal job handle for StereoVideoConvert.",
        "Human-readable summary of the selected source clip and render settings.",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        input_videos = list_input_videos()
        optional_videos = list_input_videos(include_none=True)
        return {
            "required": {
                "source_video": (input_videos if input_videos else ["none"], {"tooltip": "Main input clip to convert into stereo."}),
                "stereo_layout": (["sbs", "top_bottom"], {"default": "sbs", "tooltip": "Stereo arrangement for the output video."}),
                "use_depth_video": ("BOOLEAN", {"default": False, "tooltip": "Use an uploaded depth reference clip instead of estimating depth automatically."}),
                "depth_video": (optional_videos, {"tooltip": "Optional external depth video. It must match the source clip in frame count when enabled."}),
                "depth_model": (["da3_small", "da3_base", "da3_large"], {"default": "da3_small", "tooltip": "Depth model size used when automatic depth estimation is enabled."}),
                "depth_use_source_resolution": ("BOOLEAN", {"default": True, "tooltip": "Run depth inference at the video's original frame resolution instead of a manual lower resolution."}),
                "depth_inference_resolution": ("INT", {"default": DEFAULTS.depth_inference_size, "min": 128, "max": 2048, "tooltip": "Manual longest-side resolution for depth inference when source-resolution mode is off."}),
                "every_nth": ("INT", {"default": 1, "min": 1, "tooltip": "Frame skipping factor. Higher values render faster previews and lower the output FPS."}),
                "chunk_size": ("INT", {"default": DEFAULTS.chunk_size, "min": 1, "max": 64, "tooltip": "Frames processed per batch. Higher values improve throughput but use more RAM and VRAM."}),
                "disparity_ratio": ("FLOAT", {"default": DEFAULTS.disparity_ratio, "min": 0.0, "max": 0.25, "step": 0.0005, "tooltip": "Stereo separation as a fraction of image width. This keeps the 3D strength more consistent across different resolutions."}),
                "disparity_px": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 256.0, "step": 0.5, "tooltip": "Legacy fallback pixel disparity. Used only when disparity_ratio is 0 for older workflows."}),
                "depth_power": ("FLOAT", {"default": 0.30, "min": 0.1, "max": 4.0, "step": 0.05, "tooltip": "Depth response curve. Higher values exaggerate near/far separation."}),
                "invert_depth": ("BOOLEAN", {"default": True, "tooltip": "Flip the depth map if the scene appears inside-out."}),
                "audio_mode": (["copy", "none"], {"default": "copy", "tooltip": "Copy source audio into the final muxed video, or output video only."}),
                "depth_edge_refine_method": (
                    ["none", "simple", "fgs"],
                    {
                        "default": "none",
                        "tooltip": "Depth refinement method: none disables refinement, simple uses local RGB edge-aware smoothing, and fgs uses Fast Global Smoother based refinement.",
                    },
                ),
                "depth_edge_radius": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 16,
                        "tooltip": "Radius of the edge-aware depth refinement neighborhood.",
                    },
                ),
                "depth_edge_strength": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.1,
                        "max": 100.0,
                        "step": 0.5,
                        "tooltip": "Strength of RGB-edge protection during depth refinement.",
                    },
                ),
            },
        }

    RETURN_TYPES = (JOB_TYPE, "STRING")
    RETURN_NAMES = ("video_job", "summary")
    FUNCTION = "build_job"
    CATEGORY = CATEGORY

    def build_job(
        self,
        source_video: str,
        stereo_layout: str,
        use_depth_video: bool,
        depth_model: str,
        depth_use_source_resolution: bool,
        every_nth: int,
        chunk_size: int,
        depth_inference_resolution: int,
        disparity_ratio: float,
        disparity_px: float,
        depth_power: float,
        invert_depth: bool,
        audio_mode: str,
        depth_video: str = "none",
        depth_edge_refine_method: str = "none",
        depth_edge_radius: int = 2,
        depth_edge_strength: float = 8.0,
    ):
        if source_video == "none":
            raise ValueError("No input video found. Place a video in ComfyUI's input directory and select it here.")

        metadata = probe_video(source_video)
        spec = select_video_spec(metadata, every_nth=every_nth)

        depth_mode = "external_depth_video" if use_depth_video else "depth_anything_v3"
        depth_path: Optional[str] = None
        if use_depth_video:
            if not depth_video or depth_video == "none":
                raise ValueError("Select a depth video when depth_mode is external_depth_video")
            depth_metadata = probe_video(depth_video)
            depth_spec = select_video_spec(depth_metadata, every_nth=every_nth)
            if depth_spec.selected_frame_count != spec.selected_frame_count:
                raise ValueError("Depth video selection does not match source frame count")
            depth_path = depth_metadata.path

        job = StereoVideoJob(
            source_video_path=metadata.path,
            width=metadata.width,
            height=metadata.height,
            source_fps=metadata.fps,
            target_fps=spec.target_fps,
            every_nth=spec.every_nth,
            audio_mode=audio_mode,
            stereo_layout=stereo_layout,
            depth_mode=depth_mode,
            depth_video_path=depth_path,
            depth_model=depth_model,
            depth_use_source_resolution=depth_use_source_resolution,
            depth_inference_resolution=depth_inference_resolution,
            depth_edge_refine_method=depth_edge_refine_method,
            depth_edge_radius=depth_edge_radius,
            depth_edge_strength=depth_edge_strength,
            chunk_size=chunk_size,
            disparity_px=disparity_px,
            disparity_ratio=disparity_ratio,
            depth_power=depth_power,
            invert_depth=invert_depth,
            temp_dir=None,
            frame_count=spec.selected_frame_count,
            source_duration=spec.selected_duration_seconds,
        )
        return (job.to_handle(), _status_text(job))


class StereoVideoConvert:
    DESCRIPTION = "Runs decode, depth estimation, stereo reprojection, and streaming encode into a temporary video."
    OUTPUT_TOOLTIPS = [
        "Internal rendered-video handle for StereoVideoMuxOutput.",
        "Human-readable status text describing the temporary render output.",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_job": (JOB_TYPE, {"tooltip": "Job handle produced by StereoVideoSource."})}}

    RETURN_TYPES = (RENDER_TYPE, "STRING")
    RETURN_NAMES = ("rendered_video", "summary")
    FUNCTION = "convert"
    CATEGORY = CATEGORY

    def convert(self, video_job):
        job = StereoVideoJob.from_handle(video_job)
        spec = select_video_spec(
            probe_video(job.source_video_path),
            every_nth=job.every_nth,
        )

        render_width = job.width * 2 if job.stereo_layout == "sbs" else job.width
        render_height = job.height if job.stereo_layout == "sbs" else job.height * 2
        temp_dir = get_temp_dir()
        temp_video_path = os.path.join(temp_dir, "rendered_video.mp4")

        runner = None
        if job.depth_mode == "depth_anything_v3":
            runner = DepthAnythingRunner(
                model_name=job.depth_model,
                inference_resolution=None if job.depth_use_source_resolution else job.depth_inference_resolution,
            )
        renderer = GpuStereoRenderer(device=runner.device if runner else None)
        progress = ProgressBar(job.frame_count) if ProgressBar is not None else None

        processed_frames = 0
        try:
            with ExitStack() as stack:
                source_decoder = stack.enter_context(FFmpegChunkDecoder(spec, chunk_size=job.chunk_size))
                depth_decoder = None
                if job.depth_mode == "external_depth_video":
                    depth_spec = select_video_spec(
                        probe_video(job.depth_video_path),
                        every_nth=job.every_nth,
                    )
                    depth_decoder = stack.enter_context(FFmpegChunkDecoder(depth_spec, chunk_size=job.chunk_size))
                    depth_iter = iter(depth_decoder)
                encoder = stack.enter_context(
                    StreamingVideoEncoder(
                        output_path=temp_video_path,
                        width=render_width,
                        height=render_height,
                        fps=job.target_fps,
                    )
                )

                for source_chunk in source_decoder:
                    frames = torch.from_numpy(source_chunk).float() / 255.0
                    if job.depth_mode == "depth_anything_v3":
                        depth = runner.infer(
                            frames.permute(0, 3, 1, 2),
                            invert_depth=job.invert_depth,
                            edge_refine_method=job.depth_edge_refine_method,
                            edge_radius=job.depth_edge_radius,
                            edge_strength=job.depth_edge_strength,
                        )
                    else:
                        try:
                            depth_chunk = next(depth_iter)
                        except StopIteration as exc:
                            raise RuntimeError("Depth video ended before source video") from exc
                        depth = self._depth_from_video_chunk(depth_chunk, invert_depth=job.invert_depth)

                    rendered = renderer.render(
                        frames_bhwc=frames,
                        depth_b1hw=depth,
                        disparity_px=job.disparity_px,
                        disparity_ratio=job.disparity_ratio,
                        layout=job.stereo_layout,
                        depth_power=job.depth_power,
                    )
                    encoder.write_frames(rendered.detach().cpu().numpy())
                    processed_frames += rendered.shape[0]
                    if progress is not None:
                        progress.update(rendered.shape[0])
        except Exception:
            cleanup_temp_dir(temp_dir)
            raise

        render = StereoVideoRender(
            temp_video_path=temp_video_path,
            audio_source_path=job.source_video_path if job.audio_mode == "copy" else None,
            audio_start_seconds=0.0,
            audio_duration_seconds=spec.selected_duration_seconds,
            audio_mode=job.audio_mode,
            cleanup_dir=temp_dir,
            frame_count=processed_frames,
            fps=job.target_fps,
            width=render_width,
            height=render_height,
            stereo_layout=job.stereo_layout,
        )
        return (render.to_handle(), f"Rendered {processed_frames} frames to temp video")

    @staticmethod
    def _depth_from_video_chunk(depth_chunk: np.ndarray, invert_depth: bool) -> torch.Tensor:
        if depth_chunk.shape[-1] == 3:
            mono = depth_chunk.mean(axis=-1, keepdims=True)
        else:
            mono = depth_chunk
        depth = torch.from_numpy(mono).float() / 255.0
        depth = depth.permute(0, 3, 1, 2).contiguous()
        return 1.0 - depth if invert_depth else depth


class StereoVideoMuxOutput:
    DESCRIPTION = "Finalizes the rendered temp video, optionally muxes audio, and saves the output with an auto-incremented filename."
    OUTPUT_TOOLTIPS = ["Final saved output path."]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rendered_video": (RENDER_TYPE, {"tooltip": "Rendered temp-video handle produced by StereoVideoConvert."}),
                "filename_prefix": ("STRING", {"default": "stereo_output", "tooltip": "Prefix for the final saved filename. The node appends an incrementing counter automatically."}),
                "output_format": (["mp4", "webm", "mkv"], {"default": "mp4", "tooltip": "Final file extension/container written to ComfyUI's output directory."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "mux"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def mux(self, rendered_video, filename_prefix, output_format):
        render = StereoVideoRender.from_handle(rendered_video)
        output_path = resolve_output_prefix(filename_prefix, output_format)
        try:
            finalized = finalize_mux(
                rendered_video_path=render.temp_video_path,
                output_path=output_path,
                audio_source_path=render.audio_source_path,
                audio_start_seconds=render.audio_start_seconds,
                audio_duration_seconds=render.audio_duration_seconds,
                audio_mode=render.audio_mode,
                stereo_layout=render.stereo_layout,
            )
        finally:
            cleanup_temp_dir(render.cleanup_dir)
        return (finalized,)


NODE_CLASS_MAPPINGS = {
    "StereoVideoSource": StereoVideoSource,
    "StereoVideoConvert": StereoVideoConvert,
    "StereoVideoMuxOutput": StereoVideoMuxOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StereoVideoSource": "Stereo Video Source",
    "StereoVideoConvert": "Stereo Video Convert",
    "StereoVideoMuxOutput": "Stereo Video Mux Output",
}
