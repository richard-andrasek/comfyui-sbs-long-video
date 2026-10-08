# ComfyUI Stereo Long Video

`ComfyUI Stereo Long Video` is a video-first custom node package for converting flat (2D) video into 3D stereoscopic output inside ComfyUI. Using the Depth Anything 3 AI model for depth estimation on a monocular (flat) input video. Resulting in a SBS or top/bottom stereo video that can be used on 3D displays. 

Instead of loading the entire video and pushing it through image-batch workflows, this project treats video as a stream:

- `ffmpeg` is used to break apart the video into chunks
- Depth is estimated per chunk
- Each frame is converted to 3D using the depth (reprojected)
- Frames are streamed into a temporary file
- Once all frames are complete, the audio is added back to the final 3D video

![screenshot](screenshot/nodes.png)

## Why This Exists

Most 2D-to-3D video workflows either process video as large image batches or rely on video-native depth models that can require substantial system RAM or VRAM, especially at higher resolutions.

This project targets a different use case: a memory-efficient SBS video converter for ComfyUI that can still run on low VRAM and RAM machines. It decodes video in chunks, runs per-frame depth estimation with Depth Anything, renders stereo on the GPU, and streams frames into the encoder instead of keeping the whole clip in memory.

That design involves a tradeoff. Video-oriented depth models can deliver stronger temporal consistency, but they often demand more VRAM for higher-resolution inputs. This project instead prioritizes practical long-video conversion on modest hardware while still producing useful depth estimation and stereo output for higher-resolution material.

To improve this, there are options for higher fidelity 3D processing, including Fast Global Smoothing to improve edge processing (increases processing time roughly 50%) and global depth normalization (no increase in processing time, but has a manual configuration).  These options improve quality to nearly the level of streaming-oriented techniques, without the significant hardware overhead.

Conversely, options exist to dramatically improve generation time.  Edge refinement can be turned off or set to "simple" and inference resolution can be reduced to 12, allowing for both faster processing and reduced VRAM usage.  Finally, a "preview_mode" option allows for a storyboard mode where every 30th frame is rendered (roughly one new frame per second).  This allows you to convert the full video very quickly for use as testing the 3d mode. (Take the duration of the preview times 30 to get an estimate of the full processing time.)

## Installation

Clone the repo into your ComfyUI `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/richard-andrasek/comfyui-sbs-long-video
```

Install the Python dependencies into the same environment that runs ComfyUI:

```bash
cd /path/to/ComfyUI
python -m pip install -r custom_nodes/comfyui-sbs-long-video/requirements.txt
```

System requirements:

- `ffmpeg` and `ffprobe` must be available on `PATH`
- `torch` must be installed in the ComfyUI environment
- a CUDA-capable GPU is strongly recommended for practical runtime

Optional depth backend:

- if the official Depth Anything 3 package is available, install it in the ComfyUI environment:

```bash
python -m pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
```

- otherwise the plugin falls back to `transformers`-compatible checkpoints

After installing dependencies, restart ComfyUI.

## Nodes

Since this processing is different than any other ComfyUI plugin, this repo currently ships three custom nodes:

- `StereoVideoSource`
- `StereoVideoConvert`
- `StereoVideoMuxOutput`

Due to the batching nature, these are incompatable with other ComfyUI nodes (such as the Video Helper Suite)

## Workflow

1. Add `Stereo Video Source` and choose a source clip from ComfyUI's input directory.
2. Choose a Depth Anything 3 model and inference settings for automatic depth estimation.
3. Send the job into `Stereo Video Convert` to run decode, depth, stereo rendering, and temporary video encode.
4. Optionally enable `output_depth_video` to save generated depth as a lossless 8-bit grayscale FFV1/MKV file. This is allowed for sources shorter than 60 seconds, or when `preview_run` is enabled. Finish with `Stereo Video Mux Output` to save the final file into ComfyUI's output directory.

## How Files Are Produced

This plugin works with ComfyUI's standard directories:

- Source clips are read from the ComfyUI input directory.
- `StereoVideoConvert` writes a temporary rendered video into the ComfyUI temp directory.
- `StereoVideoMuxOutput` writes the final video into the ComfyUI output directory. When requested, it also writes the generated depth video there as `<prefix>_depth_00001.mkv` (auto-incremented).

Final filenames are auto-incremented:

- `prefix_00001.mp4`
- `prefix_00002.mp4`
- `prefix_00001.webm`
- `prefix_00001.mkv`

Available final container choices:

- `mp4`
- `webm`
- `mkv`

Audio behavior:

- `audio_mode = copy` trims and muxes audio from the original source clip
- `audio_mode = none` writes a video-only file

Preview and batch behavior:

- `preview_run` processes every 30th frame for a quick preview; disabled processes every frame
- `chunk_size` controls how many frames are processed at once
- `filename_prefix` controls the base name of each final export

## Depth Estimation

Depth is estimated automatically with the selected Depth Anything model.

## Node Reference

### StereoVideoSource

Builds the video job definition.

- `source_video`: main input clip
- `stereo_layout`: `sbs` or `top_bottom`
- `depth_model`: built-in depth model size
- `depth_use_source_resolution`: run depth at source resolution when enabled
- `depth_inference_resolution`: manual inference size when source-resolution mode is disabled
- `output_depth_video`: optionally save the generated normalized depth maps to a lossless 8-bit grayscale FFV1/MKV file; permitted only for source clips under 60 seconds unless `preview_run` is enabled. White/black direction follows `invert_depth`.
- `preview_run`: when enabled, processes every 30th frame for a quick preview
- `chunk_size`: frames processed per chunk
- `disparity_percent`: stereo separation as a percentage of image width (for example, `1.5` for 1.5%)
- `depth_power`: depth response curve before reprojection
- `invert_depth`: flips the inferred depth map
- `depth_normalization_method`: `simple` uses per-frame min/max; `global` clamps depth to a fixed shared maximum; `adaptive global` uses the same hard clamp and gradually raises the maximum if the initial value is too low
- `global_depth_max`: initial depth ceiling for `global` and `adaptive global` modes (default `850`); frames whose p99 exceeds it produce a console warning
- `audio_mode`: `copy` or `none`

### StereoVideoConvert

Executes the job.

- `video_job`: internal handle from `StereoVideoSource`
- returns a rendered temp-video handle for the output node
- when generated-depth export is enabled, streams each inferred depth chunk to a temporary FFV1 file alongside the stereo render

### StereoVideoMuxOutput

Writes the final deliverable.

- `filename_prefix`: base name for the export
- `output_format`: `mp4`, `webm`, or `mkv`
- returns the final stereo path and, when enabled, the generated depth-video path
