from __future__ import annotations

import os
import shutil
from contextlib import ExitStack
from typing import Optional

import torch

try:
    from .config import CATEGORY, DEFAULTS, JOB_TYPE, RENDER_TYPE
    from .depth_runner import DepthAnythingRunner
    from .encoder import StreamingDepthVideoEncoder, StreamingVideoEncoder, finalize_mux
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
    from encoder import StreamingDepthVideoEncoder, StreamingVideoEncoder, finalize_mux
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
        f"{job.width}x{job.height}, depth_model={job.depth_model}, layout={job.stereo_layout}"
    )


class StereoVideoSource:
    DESCRIPTION = "Builds a long-video stereo conversion job using automatic Depth Anything depth estimation, with optional depth refinement."
    OUTPUT_TOOLTIPS = [
        "Internal job handle for StereoVideoConvert.",
        "Human-readable summary of the selected source clip and render settings.",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        input_videos = list_input_videos()
        return {
            "required": {
                "source_video": (input_videos if input_videos else ["none"], {"tooltip": "Main input clip to convert into stereo."}),
                "stereo_layout": (["sbs", "top_bottom"], {"default": "sbs", "tooltip": "Stereo arrangement for the output video."}),
                "output_depth_video": ("BOOLEAN", {"default": False, "tooltip": "Save the generated depth as a lossless grayscale video. Allowed for clips under 60 seconds or when preview_run is enabled."}),
                "depth_model": (["da3_small", "da3_base", "da3_large"], {"default": "da3_small", "tooltip": "Depth model size used for automatic depth estimation."}),
                "depth_use_source_resolution": ("BOOLEAN", {"default": True, "tooltip": "Run depth inference at the video's original frame resolution instead of a manual lower resolution."}),
                "depth_inference_resolution": ("INT", {"default": DEFAULTS.depth_inference_size, "min": 128, "max": 2048, "tooltip": "Manual longest-side resolution for depth inference when source-resolution mode is off."}),
                "preview_run": ("BOOLEAN", {"default": False, "tooltip": "Render a storyboard preview by processing every 30th frame."}),
                "chunk_size": ("INT", {"default": DEFAULTS.chunk_size, "min": 1, "max": 64, "tooltip": "Frames processed per batch. Higher values improve throughput but use more RAM and VRAM."}),
                "disparity_percent": ("FLOAT", {"default": DEFAULTS.disparity_percent, "min": 0.0, "max": 25.0, "step": 0.05, "tooltip": "Stereo separation as a percentage of image width. This keeps the 3D strength more consistent across different resolutions."}),
                "depth_power": ("FLOAT", {"default": 0.30, "min": 0.1, "max": 4.0, "step": 0.05, "tooltip": "Depth response curve. Higher values exaggerate near/far separation."}),
                "invert_depth": ("BOOLEAN", {"default": True, "tooltip": "Flip the inferred depth map if the scene appears inside-out."}),
                "audio_mode": (["copy", "none"], {"default": "copy", "tooltip": "Copy source audio into the final muxed video, or output video only."}),
                "depth_edge_refine_method": (
                    ["none", "simple", "fgs"],
                    {
                        "default": "none",
                        "tooltip": "Depth refinement method: none disables refinement, simple uses local RGB edge-aware smoothing, and fgs uses Fast Global Smoother based refinement.",
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
        output_depth_video: bool,
        depth_model: str,
        depth_use_source_resolution: bool,
        preview_run: bool,
        chunk_size: int,
        depth_inference_resolution: int,
        disparity_percent: float,
        depth_power: float,
        invert_depth: bool,
        audio_mode: str,
        depth_edge_refine_method: str = "none",
    ):
        if source_video == "none":
            raise ValueError("No input video found. Place a video in ComfyUI's input directory and select it here.")

        # Translate the preview toggle into the internal source-frame stride.
        every_nth = 30 if preview_run else 1
        metadata = probe_video(source_video)
        spec = select_video_spec(metadata, every_nth=every_nth)
        source_duration = metadata.duration
        if source_duration <= 0 and metadata.frame_count > 0 and metadata.fps > 0:
            source_duration = metadata.frame_count / metadata.fps
        if source_duration <= 0:
            source_duration = spec.selected_duration_seconds
        if output_depth_video and not preview_run and source_duration >= 60.0:
            raise ValueError("Generated depth-video output is limited to source clips under 60 seconds. Enable preview_run to allow depth output for a longer clip.")

        job = StereoVideoJob(
            source_video_path=metadata.path,
            width=metadata.width,
            height=metadata.height,
            source_fps=metadata.fps,
            target_fps=spec.target_fps,
            every_nth=spec.every_nth,
            audio_mode=audio_mode,
            stereo_layout=stereo_layout,
            depth_model=depth_model,
            depth_use_source_resolution=depth_use_source_resolution,
            depth_inference_resolution=depth_inference_resolution,
            depth_edge_refine_method=depth_edge_refine_method,
            chunk_size=chunk_size,
            disparity_percent=disparity_percent,
            depth_power=depth_power,
            invert_depth=invert_depth,
            temp_dir=None,
            frame_count=spec.selected_frame_count,
            source_duration=source_duration,
            output_depth_video=bool(output_depth_video),
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
        temp_depth_video_path = os.path.join(temp_dir, "generated_depth.mkv") if job.output_depth_video else None

        runner = None
        runner = DepthAnythingRunner(
            model_name=job.depth_model,
            inference_resolution=None if job.depth_use_source_resolution else job.depth_inference_resolution,
        )
        renderer = GpuStereoRenderer(device=runner.device)
        progress = ProgressBar(job.frame_count) if ProgressBar is not None else None

        processed_frames = 0
        try:
            with ExitStack() as stack:
                source_decoder = stack.enter_context(FFmpegChunkDecoder(spec, chunk_size=job.chunk_size))
                depth_encoder = stack.enter_context(
                    StreamingDepthVideoEncoder(
                        output_path=temp_depth_video_path,
                        width=job.width,
                        height=job.height,
                        fps=job.target_fps,
                    )
                ) if temp_depth_video_path else None
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
                    depth = runner.infer(
                        frames.permute(0, 3, 1, 2),
                        invert_depth=job.invert_depth,
                        edge_refine_method=job.depth_edge_refine_method,
                    )

                    if depth_encoder is not None:
                        depth_encoder.write_depth(depth)
                    rendered = renderer.render(
                        frames_bhwc=frames,
                        depth_b1hw=depth,
                        disparity_percent=job.disparity_percent,
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
            temp_depth_video_path=temp_depth_video_path,
        )
        return (render.to_handle(), f"Rendered {processed_frames} frames to temp video")



class StereoVideoMuxOutput:
    DESCRIPTION = "Finalizes the rendered temp video, optionally muxes audio, and saves the output with an auto-incremented filename."
    OUTPUT_TOOLTIPS = ["Final saved stereo video path.", "Final saved generated depth-video path, or empty when depth export was not enabled."]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rendered_video": (RENDER_TYPE, {"tooltip": "Rendered temp-video handle produced by StereoVideoConvert."}),
                "filename_prefix": ("STRING", {"default": "stereo_output", "tooltip": "Prefix for the final saved filename. The node appends an incrementing counter automatically."}),
                "output_format": (["mp4", "webm", "mkv"], {"default": "mp4", "tooltip": "Final file extension/container written to ComfyUI's output directory."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "depth_output_path")
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
            depth_finalized = ""
            if render.temp_depth_video_path:
                depth_finalized = resolve_output_prefix(f"{filename_prefix}_depth", "mkv")
                shutil.move(render.temp_depth_video_path, depth_finalized)
        finally:
            cleanup_temp_dir(render.cleanup_dir)
        return (finalized, depth_finalized)


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
