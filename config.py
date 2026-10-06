from dataclasses import dataclass


JOB_TYPE = "STEREO_VIDEO_JOB"
RENDER_TYPE = "STEREO_VIDEO_RENDER"
PARTICLE_EFFECTS_CONFIG_TYPE = "particle_effects_config"
CATEGORY = "Stereo Video"


@dataclass(frozen=True)
class PluginDefaults:
    chunk_size: int = 4
    depth_inference_size: int = 512
    disparity_percent: float = 100.0 * 22.0 / 1920.0
    depth_power: float = 1.0
    output_codec: str = "libx264"
    output_crf: int = 19
    output_preset: str = "medium"
    pixel_format: str = "yuv420p"
    audio_codec: str = "aac"
    temp_prefix: str = "stereo_video"


DEFAULTS = PluginDefaults()
