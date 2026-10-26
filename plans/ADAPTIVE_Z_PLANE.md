# Adaptive Z-Plane (`adaptive_zero_parallax`)

## Overview

Add a UI-facing boolean, `adaptive_zero_parallax` (default `False`), to Stereo Video Source.

When enabled, at the first frame of every detected scene the plugin measures the
**p1 (1st percentile) of the normalized depth**, runs it through that scene's
`depth_power` (the value chosen by `adaptive_depth_power` when that is on), and uses the
result as the **`scene_z_shift`**. `scene_z_shift` is added to the manual `zero_parallax`
slider, and the sum is subtracted from every frame of the scene until the next scene start.

    scene_z_shift           = p1 ** scene_depth_power          (once per scene)
    effective_zero_parallax = clamp(scene_z_shift + zero_parallax, 0.0, 1.0)
    disparity               = (depth ** scene_depth_power - effective_zero_parallax) * disparity_span

### Why p1 goes through `depth_power`: a pure shift

The renderer applies `depth_power` first and subtracts `zero_parallax` afterwards. To make
the p1 depth land exactly on the screen plane (0.0), the value subtracted must be on the
same curve as the depth it is subtracted from, i.e. `p1 ** depth_power`. Because the
subtraction still happens *after* the curve, every pixel moves by the same amount: the
gaps between depths and the total depth range are unchanged. Only the scene's position
relative to the screen changes.

Example: p1 = 0.2, `depth_power` = 0.5, `zero_parallax` = 0.0

| normalized depth | d ** 0.5 | minus 0.447 (0.2 ** 0.5) |
|---|---|---|
| 0.1 (below p1) | 0.316 | -0.131 (in front of screen) |
| 0.2 (p1)       | 0.447 |  0.000 (on the screen plane) |
| 0.3            | 0.548 |  0.101 (same gap as with the toggle off) |
| 1.0            | 1.000 |  0.553 (same total range as with the toggle off) |

Subtracting *before* the curve, `(d - p1) ** power`, was rejected: it re-shapes the curve
(stretches the near field, deepens the scene, produces NaN below p1).

Sign convention: in the renderer's depth space 0 = screen plane / nearest, positive
disparity = behind the screen, so a larger subtraction moves the scene toward the viewer.
This holds when `invert_depth` is correct for the clip.

When disabled, rendering is unchanged (bit-identical to today).

## Decisions (confirmed with Richard)

1. **Pure shift.** `scene_z_shift = p1 ** scene_depth_power`, so p1 becomes 0.0 and the
   depth spread is untouched.
2. **Additive.** `scene_z_shift` is added to the manual `zero_parallax`, so the slider
   becomes an extra forward nudge on top of the per-scene value.
3. **`depth_power` is resolved first.** In the chunk loop, the per-frame `depth_power`
   (from `AdaptiveDepthPower.powers()` when `adaptive_depth_power` is on, otherwise the
   constant `job.depth_power`) is computed *before* the z-shift and passed into it. Both
   features change only at the same scene starts, so within a scene the power used for
   the z-shift is the same power the renderer applies.
4. **Once per scene.** p1 is measured and `scene_z_shift` computed only on a scene-start
   frame. Every other frame reuses the stored value, so it never changes within a scene.
   The state lives on one object created before the chunk loop, so a scene that spans
   chunk boundaries keeps its value.
5. **Clamp the sum to [0, 1].** Keeps the existing contract of `zero_parallax`
   ("1 = entire range in front") and keeps max |disparity| <= `disparity_span`, so edge
   gaps are never wider than today.
6. **Same restriction as `adaptive_depth_power`:** not allowed with Preview mode
   (`preview_run`). This also guarantees stride 1, so each scene start is a processed
   frame and is hit exactly once.
7. **Which depth:** the final depth tensor handed to the renderer (after normalization,
   `invert_depth`, and edge refinement), before the `depth_power` curve.
8. **Non-finite p1** (e.g. an all-NaN frame): keep the previous scene's value (initially
   just `zero_parallax`), same fallback behavior as `AdaptiveDepthPower`.

## How the existing pieces work (for the implementer)

- **Scene detection:** `video_io.detect_scene_starts()` (PySceneDetect AdaptiveDetector)
  runs in `StereoVideoSource.build_job` and is stored in `job.scene_starts`
  (zero-based source frame indices, always includes 0). It already runs regardless of
  the adaptive toggles. Nothing to change here.
- **Scene-start test:** the same 3-line test appears twice in `depth_runner.py`
  (`AdaptiveDepthPower.powers` and `DepthAnythingRunner.infer`):
  ```python
  left = bisect_right(scene_starts, frame_index - stride)
  right = bisect_right(scene_starts, frame_index)
  is_scene_start = right > left
  ```
  i.e. "a scene start lies in (frame_index - stride, frame_index]". Frame 0 is always a
  scene start, so the first processed frame always computes a value.
- **Adaptive depth power** (`AdaptiveDepthPower` in `depth_runner.py`) is the model to
  copy: it keeps per-scene state, re-evaluates only on a scene-start frame, and returns one
  value per frame in the chunk. Its `applied` value only changes at scene starts.
- **Quantile helper:** `DepthAnythingRunner._raw_p99(depth)` takes finite values,
  subsamples to <= 1M elements, and calls `torch.quantile(values, 0.99)`. We need the same
  thing at 0.01.
- **Renderer:** `stereo_renderer.GpuStereoRenderer.render` currently does
  `disparity = (depth.clamp(0, 1).pow(power) - float(zero_parallax)) * disparity_span`
  with `zero_parallax` as a single float. It needs to accept a per-frame list (like
  `depth_power` already does).

## File changes

### 1. `depth_runner.py`

a. Add a module-level scene-start helper and use it in the two existing places
   (`AdaptiveDepthPower.powers` and `DepthAnythingRunner.infer`) and in the new class:
```python
def is_scene_start(scene_starts: list[int], frame_index: int, stride: int) -> bool:
    """True when a scene start lies in (frame_index - stride, frame_index]."""
    return bisect_right(scene_starts, frame_index - stride) < bisect_right(scene_starts, frame_index)
```

b. Replace the staticmethod `DepthAnythingRunner._raw_p99(depth)` with a module-level
   `finite_quantile(depth, q)` (same body, `0.99` -> `q`). Update the single caller in
   `infer` to `finite_quantile(raw_depth[item_index], 0.99)`.

c. Add a named constant next to `NEAR_SCENE_P99` / `FAR_SCENE_P99`:
```python
SCENE_Z_SHIFT_QUANTILE = 0.01  # p1 of normalized depth is placed on the screen plane for each scene.
```

d. Add the class, directly under `AdaptiveDepthPower`:
```python
class AdaptiveZeroParallax:
    """Per-scene zero parallax: puts each scene's p1 depth on the screen plane (a pure shift)."""

    def __init__(self, base_shift: float, scene_starts: list[int], stride: int = 1):
        self.base_shift = float(base_shift)
        self.scene_starts = sorted(set(int(x) for x in scene_starts)) or [0]
        self.stride = max(1, int(stride))
        self.applied = self.base_shift

    def shifts(
        self,
        frame_indices: list[int],
        depth_b1hw: torch.Tensor,
        depth_powers: float | list[float],
    ) -> list[float]:
        """depth_powers must be the final per-frame powers (after adaptive depth power)."""
        if not isinstance(depth_powers, list):
            depth_powers = [float(depth_powers)] * len(frame_indices)
        results = []
        for item, frame_index in enumerate(frame_indices):
            # Measured once, on the scene's first frame; every later frame reuses self.applied.
            if is_scene_start(self.scene_starts, int(frame_index), self.stride):
                p1 = finite_quantile(depth_b1hw[item], SCENE_Z_SHIFT_QUANTILE)
                if math.isfinite(p1):
                    power = max(float(depth_powers[item]), 1e-3)  # same floor as the renderer
                    scene_z_shift = min(1.0, max(0.0, p1)) ** power
                    self.applied = min(1.0, max(0.0, scene_z_shift + self.base_shift))
            results.append(self.applied)
        return results
```
   - `p1` is clamped to [0, 1] before the power, matching the renderer's
     `depth.clamp(0, 1).pow(power)`, so the value subtracted is exactly the curved p1.
   - Cost: one subsampled `torch.quantile` on one frame per scene change. No extra frames
     or buffers are held, so RAM/VRAM use is unchanged (laptop requirement holds).

### 2. `stereo_renderer.py` (`GpuStereoRenderer.render`)

a. Type `zero_parallax` as `float | torch.Tensor | list[float] = 0.0`.
b. Convert it the same way as `depth_power` and use the tensor in the disparity line:
```python
shift = torch.as_tensor(zero_parallax, device=depth.device, dtype=depth.dtype).reshape(-1, 1, 1, 1)
disparity = (depth.clamp(0, 1).pow(power) - shift) * disparity_span
```
   (`reshape(-1, 1, 1, 1)` also works for a 0-d tensor, so the existing
   `if power.ndim == 0 ... else ...` branch can be collapsed to the same one-liner.)
   The subtraction stays *after* `.pow(power)`; that is what makes it a pure shift.
   With a scalar `0.0` the output is unchanged.

### 3. `job_types.py` (`StereoVideoJob`)

a. Add the field at the end: `adaptive_zero_parallax: bool = False`
b. In `from_handle`: `payload.setdefault("adaptive_zero_parallax", False)`
   so older saved job handles load unchanged.

### 4. `nodes.py`

**`StereoVideoSource.INPUT_TYPES`** - add as the LAST entry, after `zero_parallax`
(ComfyUI stores widget values by position; inserting earlier corrupts saved workflows):
```python
"adaptive_zero_parallax": ("BOOLEAN", {
    "default": False,
    "tooltip": "At each scene change, place the nearest 1% (p1) of that scene's depth on "
               "the screen plane, then add zero_parallax. Shifts the scene without changing "
               "its depth. Requires Preview mode off."}),
```

**`StereoVideoSource.build_job`**
- Add `adaptive_zero_parallax: bool = False` as the last parameter.
- Validation next to the `adaptive_depth_power` check:
  `if adaptive_zero_parallax and preview_run: raise ValueError("adaptive_zero_parallax cannot be used with Preview mode (preview_run); disable Preview mode and run again.")`
- Pass `adaptive_zero_parallax=adaptive_zero_parallax` into `StereoVideoJob(...)`.

**`StereoVideoConvert.convert`**
- Next to the existing adaptive check:
  `if job.adaptive_zero_parallax and job.every_nth != 1: raise ValueError("adaptive_zero_parallax requires Preview mode to be disabled")`
- Import `AdaptiveZeroParallax` alongside `AdaptiveDepthPower` (both import branches).
- Declare `adaptive_zp = None` alongside `adaptive = None` before the `try`, and after
  `adaptive = AdaptiveDepthPower(...)` create it once, before the chunk loop:
```python
if job.adaptive_zero_parallax:
    adaptive_zp = AdaptiveZeroParallax(
        base_shift=job.zero_parallax,
        scene_starts=job.scene_starts,
        stride=job.every_nth,
    )
```
- In the chunk loop, the order must be: infer depth -> resolve `depth_power` -> z-shift.
  Hoist `frame_indices` out of the `if adaptive is not None:` block so both share it:
```python
frame_indices = [processed_frames * job.every_nth + i * job.every_nth for i in range(depth.shape[0])]
depth_power = job.depth_power
if adaptive is not None:
    depth_power = adaptive.powers(frame_indices, runner.last_raw_p99)

zero_parallax = job.zero_parallax
if adaptive_zp is not None:
    # Must come after depth_power: the scene's p1 is raised to its final power.
    zero_parallax = adaptive_zp.shifts(frame_indices, depth, depth_power)
```
  and pass `zero_parallax=zero_parallax` to `renderer.render(...)`.
- The depth-video export is untouched (depth itself is not modified).

### 5. Docs

- `README.md`, after the `zero_parallax` bullet:
  `- adaptive_zero_parallax: at each scene change, shifts the scene so the nearest 1% (p1) of its depth sits on the screen plane, using that scene's depth_power so the shift does not change the depth; zero_parallax is added on top (sum clamped to 0-1). Disabled by default; requires Preview mode off`
- `TUTORIAL.md`: one short paragraph next to the `adaptive_depth_power` paragraph
  explaining the same, and that with `simple` normalization p1 is usually near 0
  (per-frame min/max), so the effect is strongest with `global` / `adaptive global`.

## Not changing

- Scene detection settings or `detect_scene_starts`.
- `AdaptiveDepthPower`'s behavior (only its scene-start test moves into the shared helper).
- Default behavior: toggle defaults to `False`; with it off every code path produces the
  same output as today.
- No new parameters beyond the one toggle (the 0.01 quantile is a named constant, not a
  widget).

## Optional agent-side sanity check (no ComfyUI, CPU only)

From the repo root:
```python
import torch
from depth_runner import AdaptiveZeroParallax
d = torch.linspace(0, 1, 10000).reshape(1, 1, 100, 100).repeat(6, 1, 1, 1)
d[3:] = d[3:] * 0.5 + 0.3                      # frames 3-5: new scene, p1 ~= 0.305
zp = AdaptiveZeroParallax(base_shift=0.05, scene_starts=[0, 3])
print(zp.shifts([0, 1], d[:2], [0.5, 0.5]))    # chunk 1: ~[0.15, 0.15]   (0.01**0.5 + 0.05)
print(zp.shifts([2, 3], d[2:4], [0.5, 0.6]))   # chunk 2: ~[0.15, 0.54]   (0.305**0.6 + 0.05)
print(zp.shifts([4, 5], d[4:], [0.6, 0.6]))    # chunk 3: ~[0.54, 0.54]   unchanged mid-scene
```
The values confirm one measurement per scene and that the value carries across chunk
boundaries. Also run
`python -m py_compile nodes.py depth_runner.py stereo_renderer.py job_types.py`.

## Validation - HUMAN only

1. Restart ComfyUI; confirm `adaptive_zero_parallax` is the last widget on Stereo Video
   Source, defaults off, and an existing saved workflow loads with all other widgets
   holding their old values.
2. Toggle off: render a short clip and confirm it matches a pre-change render.
3. Toggle on + `preview_run` on: confirm the Preview-mode error appears.
4. Toggle on, `zero_parallax` 0.0, on a clip with several cuts: the nearest part of each
   scene sits on the screen plane, and the plane only jumps at cuts (no drift within a
   scene).
5. Same clip with `zero_parallax` 0.1: scenes move further toward the viewer.
6. Pure-shift check: run `analyze_stereo.py` on the same frame with the toggle off and on.
   The disparity distribution moves by `applied * disparity_span` pixels while its spread
   (e.g. p99 - p1) stays the same.
7. Load an older saved job handle (no `adaptive_zero_parallax` key) and confirm it
   renders as before.
8. Long-movie run: RAM/VRAM usage is unchanged versus toggle off.
