# Temporal Depth Normalization Plan

## Goal

Reduce visible frame-to-frame changes in stereo separation caused by normalizing every predicted depth frame to its own minimum and maximum. Keep depth in the renderer's expected `[0, 1]` range while making the mapping change gradually over time.

## Current behavior

- `DepthAnythingRunner._normalize_depth()` in `depth_runner.py` computes a separate min and max for each frame and maps that frame to `[0, 1]`.
- The official API, Transformers model, and Transformers pipeline paths all call that normalization before returning depth. Inversion (`invert_depth`) is also applied there.
- `StereoVideoConvert.convert()` processes a video in chunks and calls the same runner for each chunk. The runner currently has no temporal state, so a temporal method must persist across calls/chunks.
- `GpuStereoRenderer.render()` converts normalized depth into disparity using `depth ** depth_power`. Changes in the normalized range therefore change stereo strength directly.
- The optional depth-video output is written from the depth returned by the runner, so it should reflect the final temporally normalized depth used for rendering.

## Recommended approach

Add a depth normalization method dropdown with `simple` and `ema` options. Keep `simple` as the default and preserve its current behavior exactly: each frame is independently normalized using its own min and max. `ema` uses temporally smoothed robust range bounds to reduce pumping.

1. Refactor each inference backend to return its raw predicted depth tensor. In `simple` mode, pass it through the existing per-frame min/max normalization unchanged. In `ema` mode, keep it as floating point on the runner's device for temporal normalization.
2. In `ema` mode, estimate configurable low/high percentiles of raw depth independently for every frame (suggested defaults: 2nd and 98th). Percentile bounds are less sensitive than min/max to isolated bad pixels and particles.
3. Maintain an EMA of those two bounds in frame order. For frame `t`, update `low_ema = alpha * low_ema + (1-alpha) * low_t` and likewise for `high_ema`; normalize that frame using the updated bounds and clamp to `[0, 1]`. Suggested default `alpha=0.95`, configurable in the NOTES.md range of 0.9–0.99. Initialize the EMA directly from the first frame's bounds rather than blending against arbitrary zeros.
4. Apply `invert_depth` after either normalization method. Then run the existing optional spatial edge refinement. This ordering keeps inversion semantics intuitive and ensures refinement operates on the final normalized range.
5. Store the EMA bounds on the runner, not in a local call variable, so state carries across decoder chunks. Create a fresh runner for each conversion as the current node does, and expose an explicit state reset method for reuse and tests.

The percentile calculation should be performed independently per frame, not across the whole chunk. A direct `torch.quantile` implementation is the clearest initial version; profile representative video sizes before optimizing. If quantile cost or device synchronization is material, replace it with a documented approximate histogram method without changing the interface.

## Scene changes and stability safeguards

In `ema` mode, stale bounds can carry across a scene cut and flatten or stretch the new scene's depth. Add a cut reset so new content can initialize fresh bounds. Recommended first implementation: compare consecutive guide-frame luminance with a cheap mean absolute difference (optionally downsampled) and reset the bounds when it exceeds a conservative threshold. Reset before normalizing the first frame after a detected cut. Document that very fast camera motion can also trigger the reset. Keep cut detection optional or conservative so ordinary motion does not continually defeat temporal smoothing. `simple` mode has no temporal state to reset.

Validate each frame's estimates before updating state: ignore non-finite values, protect the range with a small epsilon, and fall back to the previous valid bounds (or existing per-frame normalization when no valid state exists) for constant/invalid maps. Ensure `high > low` before division.

## User-facing controls and data flow

- Add a `depth_normalization_method` dropdown to `StereoVideoSource` with `simple` and `ema` options, defaulting to `simple` to preserve existing renders. `simple` means the current per-frame min/max normalization; `ema` means temporal percentile normalization.
- Keep the EMA smoothing factor and percentile bounds as internal defaults: `0.95`, 2nd percentile, and 98th percentile. Do not expose separate controls for these values.
- Keep scene-cut detection and its threshold internal; expose only the normalization method dropdown.
- Carry settings through `StereoVideoJob` and its handle. In `from_handle()`, provide defaults for old workflow handles so existing saved workflows continue to load.
- Pass settings into `DepthAnythingRunner` or its inference call. Ensure the mode is consistent across all three inference backends.
- Update the node description/tooltips and `README.md` with a concise explanation that temporal normalization stabilizes depth range over time and may react to cuts.

## Files likely to change

- `depth_runner.py`: separate raw inference from normalization; implement percentile bounds, EMA state, inversion order, and reset/cut behavior.
- `nodes.py`: expose the method dropdown and EMA controls, then pass them through the conversion path.
- `job_types.py`: serialize settings and preserve compatibility with older handles.
- `README.md`: explain the option and defaults.
- `test_depth_refine.py` or a dedicated normalization test module: cover normalization behavior and state.

## Verification plan

When implementation is requested, add focused tests for:

1. `simple` mode matching existing independent min/max normalization for each backend's raw output.
2. `ema` mode mapping percentile endpoints to the expected range and clamping outliers.
3. EMA reducing bound changes for alternating frame ranges, including state continuity when frames are split across calls/chunks.
4. First-frame initialization, reset behavior, constant maps, non-finite values, and inversion ordering in `ema` mode.
5. Backward compatibility when deserializing existing job handles without the new fields; missing method should resolve to `simple`.

Then run the relevant test module and perform a short preview render with depth export both enabled and disabled. Compare per-frame depth percentiles and disparity summaries before/after, and inspect cuts and fast camera motion for lag or sudden depth jumps.

## Rollout recommendation

Ship with `simple` as the default. After visual comparison on several clips (static scene, gradual camera movement, hard cuts, and particle-heavy footage), decide whether to recommend `ema` more prominently. Keep `simple` available as a fallback because model output scale and content can differ by backend.


