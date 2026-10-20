# Migration Plan: FFmpeg Scene Detection to PySceneDetect AdaptiveDetector

## Goal

Replace the ffmpeg `select='gt(scene,...)'` scene detector with PySceneDetect's Python API and `AdaptiveDetector`. Continue detecting scenes once when a job is built, store scene starts as source frame numbers on the job, and leave the existing depth-stat logging flow and rendered output unchanged.

The target implementation will use PySceneDetect's `detect(video_path, AdaptiveDetector(...))` API. It returns `(scene_start, scene_end)` pairs whose `FrameTimecode` values expose frame numbers. Take the first frame of each scene after the first as a cut; always include source frame 0. Use frame numbers directly—do not round seconds using the repository's approximate `source_fps`—so cuts stay aligned with the decoder's source-frame indexing.

## Current behavior to preserve

- `StereoVideoSource.build_job()` resolves/probes the source video and currently calls `detect_scene_starts()` in `video_io.py` once.
- `FFmpegChunkDecoder` selects source frames `0, every_nth, 2 * every_nth, ...`; `every_nth` is 30 for preview and 1 otherwise. Indices are zero-based and relative to the source stream.
- `StereoVideoJob.scene_starts` carries sorted, unique source-frame indices to the conversion node.
- `DepthAnythingRunner.infer()` matches scene starts to sampled inference items using `source_frame_index_start`, `source_frame_stride`, and `bisect_right`. Keep that matching logic and its frame-0 behavior.
- At each matching sampled frame, the INFO trace logs only the source frame, timestamp, and raw model depth statistics. Keep that format.
- Scene detection errors must never abort a conversion; fall back to `[0]` and log a warning/exception.
- Do not change depth inference, normalization, rendering, encoding, node inputs, or UI behavior.

## Implementation steps

1. **Add the package dependency**
   - Add `scenedetect>=0.7,<0.8` to `requirements.txt` (the package name is `scenedetect`; the Python import is also `scenedetect`).
   - The upper bound follows PySceneDetect's current API guidance while its API is pre-1.0. Confirm the ComfyUI Python runtime meets the package's supported Python version during implementation.

2. **Replace the detector in `video_io.py`**
   - Remove the ffmpeg scene-detection subprocess, showinfo parsing, ffprobe scene-time conversion, and the old `SCENE_THRESHOLD` constant. Keep the ffmpeg/ffprobe helpers used elsewhere in the module.
   - Import `detect` and `AdaptiveDetector` from `scenedetect` inside the helper (or at module scope if package import behavior is known to be safe for ComfyUI startup).
   - Call `detect(resolved_source_path, AdaptiveDetector(...), show_progress=False)` exactly once per job. Do not use the CLI or spawn a separate scenedetect process.
   - Read the start frame number from each returned scene's start `FrameTimecode` (`frame_num`); discard the first scene start if it is 0, then add 0 explicitly, deduplicate, sort, and return `list[int]`.
   - Treat an empty scene list as `[0]`. Catch dependency/import, decode, and detection failures; log and return `[0]` so scene logging remains best-effort.
   - Choose and document explicit `AdaptiveDetector` settings in code rather than surfacing them as new node inputs. Use explicit `adaptive_threshold=3.0`, `min_scene_len=15`, `window_width=2`, and `min_content_val=10.0`. A controlled dark-to-dark test missed its known cut with the default `min_content_val=15.0` and detected it at `10.0`; the two user clips still need validation. Record all four settings together so the behavior is reproducible. Do not translate the retired ffmpeg threshold `0.2` into an AdaptiveDetector threshold; they are different metrics.

3. **Update job-start scene logging in `nodes.py`**
   - Preserve the existing one-time INFO summary and first-few-cut preview, but remove references to the ffmpeg scene threshold.
   - Identify the detector and its configured settings in the summary (for example, `PySceneDetect AdaptiveDetector` plus the explicit settings), along with scene count and initial cut frame numbers/timestamps.
   - Continue using `index / metadata.fps` only for human-readable timestamps. Do not use that rounded timestamp to derive or alter frame indices.

4. **Keep the job and runner interface stable**
   - `scene_starts` stays a sorted list of zero-based source-frame indices in `StereoVideoJob` and its serialized handle.
   - Do not add a detector configuration UI/input. Do not change the runner's sampled-frame matching, raw-depth statistics, or log shape.
   - Confirm there are no remaining imports or logs that refer to `SCENE_THRESHOLD` or ffmpeg scene parsing.

## Verification plan

1. Install the requirements in the same Python environment used by ComfyUI; verify `from scenedetect import detect, AdaptiveDetector` imports successfully.
2. Run detection on the two known 30-second dark clips. Compare returned cut frame numbers against their known scene boundaries and report missed and extra cuts for each clip. Inspect extracted frames immediately before and at each reported cut to confirm that the frame number is the first frame of the new scene.
3. Run detection against a short synthetic clip with known hard cuts as a controlled indexing check. Include a dark-to-dark cut as well as a high-contrast cut if practical.
4. Verify preview mapping with stride 30: for each cut, the first sampled index at or after that cut should be the frame logged by `DepthAnythingRunner.infer()`. Include cuts on a sample boundary and between sample boundaries.
5. Verify one INFO summary is emitted at job creation and the raw-stat scene trace remains in the requested compact format. Force/mock detector failure and verify the job still carries `[0]` and conversion proceeds.
6. Confirm the rendered output path is untouched by the migration: scene detection only reads the source and supplies frame indices to the existing logging flow.

## Acceptance criteria

- PySceneDetect is declared in `requirements.txt` and `AdaptiveDetector` runs once per built job.
- Scene starts are zero-based frame numbers compatible with the existing decoder and runner, including preview stride 30.
- The two dark clips are checked with missed/extra cuts documented; frame extraction confirms reported boundaries.
- Detection failure falls back to frame 0 without aborting a conversion.
- Depth statistics remain raw-only and rendering behavior remains unchanged.

## API references

- [PySceneDetect installation and package names](https://github.com/Breakthrough/PySceneDetect/blob/main/website/pages/download.md)
- [Python API: `detect()` return values and version guidance](https://www.scenedetect.com/docs/latest/api.html)
- [AdaptiveDetector parameters and behavior](https://www.scenedetect.com/docs/latest/api/detectors.html)
