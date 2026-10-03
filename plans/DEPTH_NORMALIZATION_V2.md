# Global Depth Normalization V2 Plan

## Goal

Replace the `ema` normalization mode with a predictable global depth scale. Use a configurable reference maximum of `750` depth units by default, and adapt only a frame whose 99th-percentile depth exceeds that value. Preserve `simple` mode unchanged for existing workflows.

## Current behavior and context

- `DepthAnythingRunner` currently supports `simple` and `ema` normalization. `ema` tracks smoothed per-frame percentile bounds and detects scene cuts.
- The normalization method is selected in `StereoVideoSource`, serialized through `StereoVideoJob`, and applied by `DepthAnythingRunner.infer()` before optional edge refinement and stereo rendering.
- Stereo rendering expects normalized depth in `[0, 1]`; that value feeds the disparity response curve.
- The existing depth-video export receives the normalized depth returned by the runner.
- The `ema` path and its internal state, cut detection, and tests should be removed when implementing this V2 plan.

## Proposed behavior

Replace the `ema` option with `global`, leaving the method dropdown as `simple` / `global`:

- `simple`: retain the current independent per-frame min/max normalization exactly.
- `global`: use the configured global reference maximum, `750` by default, with a per-frame p99 fallback for unusually large depth values.

For each frame in `global` mode:

1. Run inference and obtain the raw depth map. Resize it to the source frame dimensions before measuring percentiles so all backends use the map that will feed refinement and rendering.
2. Compute that frame's p99 from finite raw depth values. Do not pool values across frames or chunks: the statistic and any fallback scaling are frame-local.
3. Choose the normalization maximum:
   - If `p99 <= global_depth_max`, use `global_depth_max`.
   - If `p99 > global_depth_max`, use that frame's p99 as the maximum for this frame, and emit a warning containing the global frame number, estimated timestamp, measured p99, and configured maximum.
4. Normalize from zero using `depth / chosen_max`, then clamp to `[0, 1]`. Thus values above the chosen maximum saturate at `1`. In the fallback case, the frame's p99 maps to `1`; values above p99 are clipped.
5. Apply `invert_depth` after normalization, then apply the existing optional edge refinement. Export and render this same normalized depth.

This uses `750` as the stable common reference for ordinary frames. Frames whose p99 exceeds the reference expand their own range to avoid clipping most of that frame's depth structure; the warning makes frequent fallback use visible so the user can increase the reference value.

## User-facing controls

- Change the method choices from `simple` / `ema` to `simple` / `global`.
- Add a positive floating-point input such as `global_depth_max`, default `750`.
- Set the input description/tooltip to explicitly state that the default is 750, that the console will show warnings when a frame's p99 exceeds the value, and that users should watch the console for warnings and raise the value if they see multiple warnings.
- Carry `global_depth_max` through `StereoVideoJob` serialization. Existing saved handles without a method should continue to select `simple`; handles from the intermediate `ema` implementation should safely map to `simple` or another explicitly documented compatibility choice. Do not silently interpret `ema` as `global` because their output behavior is materially different.
- Validate that `global_depth_max` is finite and greater than zero before conversion starts.
- Remove the EMA smoothing, percentile-bound, previous-frame luminance, reset, and scene-cut state. The p99 itself remains a fixed internal percentile; it is not an additional user control.

## Warning and frame indexing

Emit a Python warning-level log record for every frame whose p99 exceeds the configured maximum. Include the frame index, p99 value, and configured maximum in the message. Maintain the frame index across decoder chunks so logs point to the correct source frame. Use the module logger rather than `print`, allowing ComfyUI's console logging to display warnings consistently.

If warning volume proves excessive in real clips, a later change can aggregate consecutive exceedances while preserving count and maximum p99. The first implementation should report each exceeded frame as requested.

## Files expected to change during implementation

- `depth_runner.py`: remove EMA and cut-detection state; implement per-frame p99 global normalization, clamping, and warning logging; preserve `simple` behavior and backend support.
- `nodes.py`: expose the `simple` / `global` choice and the `global_depth_max` input with its requested description; validate and pass both settings.
- `job_types.py`: serialize method and maximum; handle old and intermediate workflow payloads safely.
- `README.md`: document the two methods, default global maximum, and warning guidance.
- `plans/DEPTH_NORMALIZATION.md`: mark the prior EMA proposal as superseded or link to this V2 plan so the repository does not present incompatible guidance.
- `test_depth_normalization.py` or a replacement focused test module: remove EMA-state cases and cover the global behavior.

## Verification plan

Add focused checks for:

1. `simple` mode preserving the prior per-frame min/max result.
2. `global` mode using `750` when p99 is at or below the configured limit.
3. A frame with p99 above the limit using its own p99 as the scale, clamping values above p99, and producing a warning with the frame index and p99.
4. Per-frame behavior when frames are batched together or split across decoder chunks; one frame's p99 must not affect another's normalization.
5. Constant maps, non-finite depth values, inversion order, and invalid maximum values.
6. Backward compatibility for saved job handles, including payloads containing the old `ema` method and development-only EMA fields.

Run the focused tests, Python compilation, and a short preview with depth export enabled. Inspect console warnings and confirm exported depth and stereo output both use the same normalized map. Compare a clip where all frame p99 values remain below 750 with one that produces repeated exceedances.

## Rollout

Keep `simple` as the node default to preserve existing renders. Let users opt into `global` after choosing a suitable reference maximum for their model and footage. The initial recommendation is `750`; users should raise it when the console reports a large number of p99 warnings.
