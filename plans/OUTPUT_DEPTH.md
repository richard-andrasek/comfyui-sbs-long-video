# Plan: Save the Generated Depth Video

## Goal

When `use_depth_video` is false, optionally save the depth maps produced by the selected built-in model (including `da3_large`) as a depth video. Keep the current chunked processing model: infer depth for one source chunk, encode those depth frames immediately, and discard the tensors. Do not collect the full video of depth tensors in memory.

The saved depth should be the same normalized, inverted (if selected), and edge-refined map that is passed to `GpuStereoRenderer`. This makes the artifact useful for inspecting the actual conversion input and reusing it as an external depth video.

## Existing flow

1. `StereoVideoSource.build_job` stores `depth_model`, source-resolution/inference-resolution settings, `invert_depth`, and refinement method in `StereoVideoJob`. `use_depth_video=False` sets `depth_mode="depth_anything_v3"`.
2. `StereoVideoConvert.convert` creates `DepthAnythingRunner(model_name=job.depth_model, ...)`. The model name `da3_large` resolves to the DA3-LARGE candidate in `depth_runner.py` (with the existing V2 fallback behavior if the official DA3 package is unavailable).
3. The source is decoded in `chunk_size` batches. For each batch, `runner.infer(...)` returns a `[B,1,H,W]` depth tensor. The runner applies normalization, optional inversion, restores source dimensions, and applies the selected edge refinement.
4. That same tensor is passed to `renderer.render(...)`; only the rendered stereo frames are currently sent to `StreamingVideoEncoder`.
5. `StereoVideoMuxOutput.mux` finalizes the stereo video, then deletes the render’s temporary directory. Therefore an auxiliary depth file placed in that directory must be copied/finalized before cleanup.

Depths from `runner.infer` are already available at source frame dimensions. No additional decode or inference pass is needed. Preview mode also applies its frame stride before inference; the depth output should use `job.target_fps` and contain `job.frame_count` selected frames, just like the stereo render.

### Duration limit for depth export

The infrastructure already reads clip duration and frame rate in `probe_video` using `ffprobe` (`VideoMetadata.duration`, `fps`, and `frame_count`). `select_video_spec` derives selected frame count, target frame rate, and selected duration; `StereoVideoJob` carries `source_duration`, `target_fps`, and `frame_count`. Use the probed source duration to enforce the export limit in `StereoVideoSource.build_job`, before the expensive inference/render begins.

Depth-video output is permitted only when `preview_run` is true or the source video is shorter than 60 seconds. Reject a request to export generated depth when `preview_run` is false and source duration is 60 seconds or longer, with a clear error explaining the limit and the preview option. Apply this condition only to generated depth export; it must not prevent ordinary stereo conversion or use of a supplied external depth video.

Important: current `preview_run` behavior selects every 30th frame and divides the output frame rate by 30. It does not truncate the source to a short opening excerpt; the preview still spans roughly the full source duration with sparse frames. Thus the exception allows full-duration, low-frame-rate depth preview export for long sources. Preserve the existing preview semantics unless the user separately requests a short excerpt.

## Proposed user-facing behavior

- Add an `output_depth_video` boolean to `StereoVideoSource` (default `false` to preserve current storage and runtime behavior). Permit generated-depth export only if `preview_run=true` or the source duration is under 60 seconds; otherwise fail early with a clear message.
- Only enable it for automatic depth mode. If `use_depth_video=True`, the supplied clip is already the depth source, so do not create a second generated-depth file. The UI can disable the toggle conditionally if supported, or the job builder can reject/ignore it with a clear summary.
- Use an explicit format choice or a single lossless default. Depth is data, not ordinary color footage: the current `StreamingVideoEncoder` uses H.264, 8-bit `yuv420p`, CRF 19. That is lossy and can alter edge values and quantize depth. Prefer a lossless grayscale representation (for example FFV1 in MKV with a documented pixel format/bit depth), or a lossless PNG image sequence if reliable grayscale video encoding is not supported in the target FFmpeg build. Avoid silently using the ordinary stereo encoder settings for depth.
- Save the depth artifact in ComfyUI's output directory with an auto-incremented filename related to the mux output, e.g. `stereo_output_depth_00001.mkv`. Return its path from the output node, or expose it in the node's UI result/summary so users can find it. Keep the stereo output as the node's primary output.

## Implementation steps

1. **Job and node input:** Add `output_depth_video: bool` to `StereoVideoSource.INPUT_TYPES` and `build_job`, and persist it in `StereoVideoJob` handles. Maintain compatibility with already-created workflow handles by defaulting the field to `false` in `from_handle` when it is absent (and update versioning only if the project requires it).
2. **Depth encoder:** Add a depth-specific streaming encoder in `encoder.py`, or generalize `StreamingVideoEncoder` to accept a pixel format/codec safely. Its input contract should accept `[B,1,H,W]` float depth in `[0,1]`, convert to the selected grayscale integer format, and stream to FFmpeg. Do not route depth through RGB H.264/yuv420 settings. Decide and document bit depth/container and verify the produced file can be decoded by the existing `FFmpegChunkDecoder` (or add a grayscale decode path if needed).
3. **Parallel chunk write:** In `StereoVideoConvert.convert`, create the optional depth encoder in the existing `ExitStack`, with `width=job.width`, `height=job.height`, and `fps=job.target_fps`. For every automatically inferred chunk, write `depth` to that encoder before the tensor is otherwise released. Do not write a depth video in external-depth mode. Make sure encoder context exits cleanly on successful completion and that the existing exception cleanup removes incomplete temporary output.
4. **Render handle:** Carry the temporary generated-depth path and relevant metadata (frame count, fps, width, height, encoding) in `StereoVideoRender`. Its handle loader should default missing new fields for backward compatibility. Keep this file inside the render's temp directory until the output node runs.
5. **Final output node:** When a depth path exists, choose an incremented depth output path in the output directory and finalize/copy the depth artifact before `cleanup_temp_dir`. Do not pass it through `finalize_mux`, which adds stereo metadata and optionally source audio. If audio is ever requested for depth, treat that as a separate explicit feature. Ensure cleanup still happens in `finally` if depth export or stereo mux fails.
6. **Discoverability:** Update output return values/summary and README/workflow documentation to state that depth export is optional, is generated only in automatic mode, and is a grayscale depth map (describe whether white means near or far after the `invert_depth` setting). If the ComfyUI output node conventions require `UI` results to expose extra paths, follow those conventions while preserving the existing output path.
7. **Resource/error handling:** The encoder must be closed and its FFmpeg exit status checked. On any inference/encoding failure, clean the temporary directory as today. Ensure no full-video depth tensor is retained. Account for preview stride when setting frame rate and expected frame count. Enforce the 60-second export gate described above, while allowing full-duration sparse preview export when `preview_run=true`.

## Details and pitfalls

- The tensor written must be the post-`invert_depth` and post-refinement `depth` used by the renderer. Saving raw model predictions instead would make the artifact disagree with the produced stereo video.
- `_normalize_depth` currently normalizes each frame independently. This plan preserves that behavior; it does not add temporal normalization or change depth values.
- The renderer's `depth_power` and `disparity_percent` are stereo-rendering controls. Do not bake those into the exported depth map: retaining normalized depth lets the external-depth workflow apply its own controls.
- Model inference may run at a smaller configured resolution, but the runner upsamples depth back to the source frame size before returning it. Export at that restored size to align frame-for-frame with the source video.
- Grayscale depth with lossy chroma subsampling or a color-range conversion is undesirable. Confirm FFmpeg pixel format, bit depth, range metadata, and downstream reader behavior. A lossless container/file format is important for a reusable depth reference.
- If output-file creation is deferred to `StereoVideoMuxOutput`, its output prefix is only known there. Keep the generated file in the temporary job directory from Convert and move/copy it to its final auto-incremented name before deleting that directory.
- The source audio is unrelated to the depth video; do not add it. The depth video should be silent and have the selected frame rate and exactly the processed frame count.

## Verification checklist

- With auto depth, `da3_large`, and export enabled: verify a depth file is created with the expected dimensions, fps, and frame count.
- Verify decoded depth frames are grayscale, non-constant, and match the dimensions/count of the stereo input frames. Compare a few decoded pixels or a frame image against the tensor sent to the renderer, allowing only the documented lossless integer quantization.
- Verify the exported depth video can be selected later as `depth_video` and the stereo render can complete with it.
- Verify disabled export creates no depth artifact and retains the existing stereo behavior.
- Verify external-depth mode creates no redundant generated-depth file.
- Verify depth export is accepted for a source under 60 seconds when `preview_run=false`, and rejected at 60 seconds or longer in that mode. Verify `preview_run=true` allows export for a long source and preserves the existing sparse-frame/full-duration behavior.
- Force or simulate encoder failure and confirm no stale temporary files remain and normal output cleanup still runs.



