from __future__ import annotations

import math
import os
import shutil
from contextlib import ExitStack
from typing import Optional

import torch

try:
    from .config import CATEGORY, DEFAULTS, JOB_TYPE, RENDER_TYPE, PARTICLE_EFFECTS_CONFIG_TYPE
    from .depth_runner import DepthAnythingRunner
    from .encoder import StreamingDepthVideoEncoder, StreamingVideoEncoder, finalize_mux
    from .job_types import StereoVideoJob, StereoVideoRender
    from .stereo_renderer import GpuStereoRenderer
    from .particle_depth import detect_particle_mask, apply_particle_depth
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
    from config import CATEGORY, DEFAULTS, JOB_TYPE, RENDER_TYPE, PARTICLE_EFFECTS_CONFIG_TYPE
    from depth_runner import DepthAnythingRunner
    from encoder import StreamingDepthVideoEncoder, StreamingVideoEncoder, finalize_mux
    from job_types import StereoVideoJob, StereoVideoRender
    from stereo_renderer import GpuStereoRenderer
    from particle_depth import detect_particle_mask, apply_particle_depth
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


MAX_CHUNK_PIXELS = 64_000_000
DEBUG_SUB = 4

_DEFAULT_PARTICLE_EFFECTS = {
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


class ParticleEffectsConfig:
    DESCRIPTION = "Configures optional particle depth effects for Stereo Video Source."
    RETURN_TYPES = (PARTICLE_EFFECTS_CONFIG_TYPE,)
    RETURN_NAMES = ("particle_effects_config",)
    FUNCTION = "build_config"
    CATEGORY = CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "brightness_threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
            "saturation_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            "particle_min_area": ("INT", {"default": 1, "min": 1, "max": 10000}),
            "particle_max_area": ("INT", {"default": 200, "min": 1, "max": 100000}),
            "particle_depth_min": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
            "particle_depth_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "particle_depth_offset_min": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            "particle_depth_offset_max": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
            "particle_mask_blur": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "particle_depth_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "particle_depth_mode": (["relative", "absolute"], {"default": "relative"}),
            "particle_debug_video": ("BOOLEAN", {"default": False}),
        }}

    def build_config(
        self, brightness_threshold, saturation_threshold,
        particle_min_area, particle_max_area, particle_depth_min, particle_depth_max,
        particle_depth_offset_min, particle_depth_offset_max, particle_mask_blur,
        particle_depth_strength, particle_depth_mode, particle_debug_video,
    ):
        values = dict(_DEFAULT_PARTICLE_EFFECTS)
        values.update({
            "particle_brightness_threshold": float(brightness_threshold),
            "particle_saturation_threshold": float(saturation_threshold),
            "particle_min_area": int(particle_min_area),
            "particle_max_area": int(particle_max_area),
            "particle_depth_min": float(particle_depth_min),
            "particle_depth_max": float(particle_depth_max),
            "particle_depth_offset_min": float(particle_depth_offset_min),
            "particle_depth_offset_max": float(particle_depth_offset_max),
            "particle_mask_blur": float(particle_mask_blur),
            "particle_depth_strength": float(particle_depth_strength),
            "particle_depth_mode": particle_depth_mode,
            "particle_debug_video": bool(particle_debug_video),
        })
        _validate_particle_effects(values)
        values["type"] = PARTICLE_EFFECTS_CONFIG_TYPE
        values["version"] = 1
        return (values,)


def _validate_particle_effects(values):
    if not 0 <= values["particle_brightness_threshold"] <= 1 or not 0 <= values["particle_saturation_threshold"] <= 1:
        raise ValueError("Particle brightness and saturation thresholds must be in [0, 1]")
    if values["particle_min_area"] < 1 or values["particle_max_area"] < values["particle_min_area"]:
        raise ValueError("Particle area limits must satisfy 1 <= min_area <= max_area")
    if not 0 <= values["particle_depth_min"] <= values["particle_depth_max"] <= 1:
        raise ValueError("Particle depth limits must satisfy 0 <= min <= max <= 1")
    if not 0 <= values["particle_depth_offset_min"] <= values["particle_depth_offset_max"] <= 1:
        raise ValueError("Particle depth offsets must satisfy 0 <= min <= max <= 1")
    if not math.isfinite(values["particle_mask_blur"]) or values["particle_mask_blur"] < 0:
        raise ValueError("Particle mask blur must be finite and non-negative")
    if not math.isfinite(values["particle_depth_strength"]) or not 0 <= values["particle_depth_strength"] <= 1:
        raise ValueError("Particle depth strength must be in [0, 1]")
    if values["particle_depth_mode"] not in ("relative", "absolute"):
        raise ValueError("Particle depth mode must be 'relative' or 'absolute'")


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
                "chunk_size": ("INT", {"default": DEFAULTS.chunk_size, "min": 1, "max": 64, "tooltip": "Frames processed per batch. Higher values improve throughput but use more RAM and VRAM; the value may be capped by resolution."}),
                "disparity_percent": ("FLOAT", {"default": DEFAULTS.disparity_percent, "min": 0.0, "max": 25.0, "step": 0.05, "tooltip": "Stereo separation as a percentage of image width. This keeps the 3D strength more consistent across different resolutions."}),
                "depth_power": ("FLOAT", {"default": 0.30, "min": 0.1, "max": 4.0, "step": 0.05, "tooltip": "Depth response curve. Higher values exaggerate near/far separation."}),
                "invert_depth": ("BOOLEAN", {"default": True, "tooltip": "Flip the inferred depth map if the scene appears inside-out."}),
                "audio_mode": (["copy", "none"], {"default": "copy", "tooltip": "Copy source audio into the final muxed video, or output video only."}),
                "depth_normalization_method": (
                    ["simple", "global"],
                    {"default": "global", "tooltip": "Depth normalization: Depth must be normalized to fit within a boundary. GLOBAL is generally better with no performance cost.  SIMPLE uses a basic, independent per-frame min/max. This may result in some scenes with too much/little depth; GLOBAL uses a depth maximum trusting that DepthAnything is accurate, but requires manual tuning. If it goes over that max, it will scale per-frame as a fallback."},
                ),
                "global_depth_max": ("FLOAT", {"default": 750.0, "min": 0.01, "max": 1000000.0, "step": 10.0, "tooltip": "Global depth maximum (default: 750). Used only with global depth normalization. Watch the console for p99 warnings; if you see many, increase this value."}),
                "depth_edge_refine_method": (
                    ["none", "simple", "fgs"],
                    {
                        "default": "none",
                        "tooltip": "Depth refinement method: none disables refinement, simple uses local RGB edge-aware smoothing, and fgs uses Fast Global Smoother based refinement.",
                    },
                ),
                "enable_particle_depth": ("BOOLEAN", {"default": False, "tooltip": "Enable the particle depth settings from the optional Particle Effects Config input."}),
            },
            "optional": {
                "particle_effects_config": (PARTICLE_EFFECTS_CONFIG_TYPE, {"tooltip": "Optional particle effects settings. Defaults are used when unconnected."}),
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
        depth_normalization_method: str = "simple",
        global_depth_max: float = 750.0,
        enable_particle_depth: bool = False,
        particle_effects_config=None,
    ):
        if depth_normalization_method not in ("simple", "global"):
            raise ValueError("depth_normalization_method must be 'simple' or 'global'")
        if not math.isfinite(float(global_depth_max)) or global_depth_max <= 0:
            raise ValueError("global_depth_max must be finite and greater than zero")
        particle_effects = dict(_DEFAULT_PARTICLE_EFFECTS)
        if isinstance(particle_effects_config, dict) and particle_effects_config.get("type") == PARTICLE_EFFECTS_CONFIG_TYPE:
            particle_effects.update({key: value for key, value in particle_effects_config.items() if key in particle_effects})
        _validate_particle_effects(particle_effects)
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
            depth_normalization_method=depth_normalization_method,
            global_depth_max=global_depth_max,
            enable_particle_depth=bool(enable_particle_depth),
            **particle_effects,
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

    @torch.inference_mode()
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
        temp_particle_debug_path = os.path.join(temp_dir, "particle_depth_debug.mp4") if job.particle_debug_video else None

        runner = DepthAnythingRunner(
            model_name=job.depth_model,
            inference_resolution=None if job.depth_use_source_resolution else job.depth_inference_resolution,
            depth_normalization_method=job.depth_normalization_method,
            global_depth_max=job.global_depth_max,
        )
        renderer = GpuStereoRenderer(device=runner.device)
        progress = ProgressBar(job.frame_count) if ProgressBar is not None else None

        processed_frames = 0
        try:
            with ExitStack() as stack:
                effective_chunk = max(1, min(job.chunk_size, MAX_CHUNK_PIXELS // (job.width * job.height)))
                if effective_chunk != job.chunk_size:
                    print(f"[StereoVideo] chunk_size capped {job.chunk_size} -> {effective_chunk} for {job.width}x{job.height}")
                source_decoder = stack.enter_context(FFmpegChunkDecoder(spec, chunk_size=effective_chunk))
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
                debug_encoder = stack.enter_context(
                    StreamingVideoEncoder(
                        output_path=temp_particle_debug_path,
                        width=job.width * 5,
                        height=job.height,
                        fps=job.target_fps,
                    )
                ) if temp_particle_debug_path else None

                for source_chunk in source_decoder:
                    frames_u8 = torch.from_numpy(source_chunk).to(runner.device, non_blocking=True)
                    frames = frames_u8.float().div_(255.0)
                    del source_chunk
                    depth = runner.infer(
                        frames.permute(0, 3, 1, 2),
                        invert_depth=job.invert_depth,
                        edge_refine_method=job.depth_edge_refine_method,
                        source_frame_index_start=processed_frames * job.every_nth,
                        source_frame_stride=job.every_nth,
                        source_fps=job.source_fps,
                    ).to(runner.device)
                    scene_depth = depth if job.particle_debug_video else None

                    particle_mask = synthetic_depth = None
                    if job.enable_particle_depth or debug_encoder is not None:
                        particle_mask = detect_particle_mask(
                            frames_u8,
                            brightness_threshold=job.particle_brightness_threshold,
                            saturation_threshold=job.particle_saturation_threshold,
                            min_area=job.particle_min_area,
                            max_area=job.particle_max_area,
                            mask_blur=job.particle_mask_blur,
                        )
                        if job.enable_particle_depth:
                            depth, synthetic_depth = apply_particle_depth(
                                depth, particle_mask,
                                depth_min=job.particle_depth_min,
                                depth_max=job.particle_depth_max,
                                offset_min=job.particle_depth_offset_min,
                                offset_max=job.particle_depth_offset_max,
                                strength=job.particle_depth_strength,
                                mode=job.particle_depth_mode,
                                frame_index_start=processed_frames * job.every_nth,
                                return_synthetic=debug_encoder is not None,
                            )
                        else:
                            synthetic_depth = depth

                    if debug_encoder is not None:
                        def gray_u8(depth_map):
                            gray = (depth_map.clamp(0, 1) * 255).to(torch.uint8).squeeze(1).unsqueeze(-1)
                            return gray.expand(-1, -1, -1, 3)
                        for start in range(0, frames.shape[0], DEBUG_SUB):
                            end = min(start + DEBUG_SUB, frames.shape[0])
                            panel = torch.cat((
                                frames_u8[start:end],
                                gray_u8(scene_depth[start:end]),
                                gray_u8(particle_mask[start:end].float()),
                                gray_u8(synthetic_depth[start:end]),
                                gray_u8(depth[start:end]),
                            ), dim=2)
                            debug_encoder.write_frames(panel.cpu().numpy())
                            del panel

                    if depth_encoder is not None:
                        depth_encoder.write_depth(depth)
                    rendered = renderer.render(
                        frames_bhwc=frames,
                        depth_b1hw=depth,
                        disparity_percent=job.disparity_percent,
                        layout=job.stereo_layout,
                        depth_power=job.depth_power,
                    )
                    rendered_u8 = (rendered.clamp(0, 1) * 255).to(torch.uint8)
                    encoder.write_frames(rendered_u8.cpu().numpy())
                    chunk_frame_count = rendered.shape[0]
                    processed_frames += chunk_frame_count
                    if progress is not None:
                        progress.update(chunk_frame_count)
                    del frames, frames_u8, depth, scene_depth, particle_mask, synthetic_depth, rendered, rendered_u8
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
            temp_particle_debug_path=temp_particle_debug_path,
        )
        return (render.to_handle(), f"Rendered {processed_frames} frames to temp video")



class StereoVideoMuxOutput:
    DESCRIPTION = "Finalizes the rendered temp video, optionally muxes audio, and saves the output with an auto-incremented filename."
    OUTPUT_TOOLTIPS = ["Final saved stereo video path.", "Final saved generated depth-video path, or empty when depth export was not enabled.", "Final saved five-panel particle diagnostic video, or empty when disabled."]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rendered_video": (RENDER_TYPE, {"tooltip": "Rendered temp-video handle produced by StereoVideoConvert."}),
                "filename_prefix": ("STRING", {"default": "stereo_output", "tooltip": "Prefix for the final saved filename. The node appends an incrementing counter automatically."}),
                "output_format": (["mp4", "webm", "mkv"], {"default": "mp4", "tooltip": "Final file extension/container written to ComfyUI's output directory."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_path", "depth_output_path", "particle_debug_path")
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
            debug_finalized = ""
            if render.temp_depth_video_path:
                depth_finalized = resolve_output_prefix(f"{filename_prefix}_depth", "mkv")
                shutil.move(render.temp_depth_video_path, depth_finalized)
            if render.temp_particle_debug_path:
                debug_finalized = resolve_output_prefix(f"{filename_prefix}_particle_debug", "mp4")
                shutil.move(render.temp_particle_debug_path, debug_finalized)
        finally:
            cleanup_temp_dir(render.cleanup_dir)
        return (finalized, depth_finalized, debug_finalized)


NODE_CLASS_MAPPINGS = {
    "ParticleEffectsConfig": ParticleEffectsConfig,
    "StereoVideoSource": StereoVideoSource,
    "StereoVideoConvert": StereoVideoConvert,
    "StereoVideoMuxOutput": StereoVideoMuxOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ParticleEffectsConfig": "Particle Effects Config",
    "StereoVideoSource": "Stereo Video Source",
    "StereoVideoConvert": "Stereo Video Convert",
    "StereoVideoMuxOutput": "Stereo Video Mux Output",
}
