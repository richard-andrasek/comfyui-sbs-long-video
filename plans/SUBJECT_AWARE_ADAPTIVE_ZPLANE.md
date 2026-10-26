# Subject-Aware Adaptive Z-Plane

## Goal

Reduce eye fatigue by keeping the subject of each shot (usually an actor's face) at a
stable, comfortable depth just behind the screen, the way studio stereographers
"pull convergence" shot by shot. Today, `adaptive_zero_parallax` puts the nearest 1% (p1)
of each scene on the screen plane. The nearest 1% is often a shoulder, a table edge or
foreground clutter, so the actor lands at a different depth in every shot and the eyes
have to re-converge at every cut.

This feature is an **alternative** to the p1 calculation: at each scene change, run the
YuNet face detector on the scene's first frame. If a face is found, shift the scene so that
face sits **0.1 behind the screen plane**. If no face is found, fall back to the existing p1
calculation unchanged.

## Concept

At each scene start (the same frames `AdaptiveZeroParallax` already uses):

1. Run YuNet on the full-resolution source frame.
2. If one or more faces pass the filters, pick one (see "Multiple faces") and measure
   its depth: the **median** normalized depth inside the central part of the face box.
3. Compute the shift with the same pure-shift math as the adaptive z-plane, but target
   the face instead of p1:

       face_curved             = face_depth ** scene_depth_power
       scene_z_shift           = face_curved - SUBJECT_SCREEN_DEPTH       (SUBJECT_SCREEN_DEPTH = 0.1)
       effective_zero_parallax = clamp(scene_z_shift + zero_parallax, 0.0, 1.0)
       disparity               = (depth ** scene_depth_power - effective_zero_parallax) * disparity_span

   So the face's disparity is `0.1 * disparity_span` pixels behind the screen, regardless
   of where the face was in the raw depth range.

4. If no face is found (or the face depth is not finite), use the existing
   `scene_z_shift = p1 ** scene_depth_power`.
5. As today, the value is held for the whole scene, survives chunk boundaries, and only
   changes at the next scene start.

### Why the 0.1 is measured after `depth_power`

The renderer applies `depth_power` first and subtracts `zero_parallax` afterwards
(see `ADAPTIVE_Z_PLANE.md`). Subtracting `face_depth ** power - 0.1` therefore moves
every pixel by the same amount. Depth spread and depth_power are unchanged, and the face
ends up at exactly 0.1 in the renderer's curved 0.0-1.0 space, i.e.
`0.1 * disparity_span` pixels of positive (behind-screen) parallax.

### Worked example

face_depth = 0.40, p1 = 0.15, depth_power = 0.5, zero_parallax = 0.0

| | p1 mode (today) | subject mode |
|---|---|---|
| scene_z_shift | 0.15 ** 0.5 = 0.387 | 0.40 ** 0.5 - 0.1 = 0.532 |
| face lands at | 0.632 - 0.387 = **0.245** behind | 0.632 - 0.532 = **0.100** behind |
| p1 lands at | 0.000 (on screen) | 0.387 - 0.532 = **-0.145** (in front) |

In subject mode, anything nearer than the face comes out of the screen. That is the
intended studio look, but see "Guard: pop-out limit" below.

### Clamping behaviour

- **Lower clamp (0.0):** if the face is already nearer than 0.1 (`face_curved < 0.1`),
  the shift would be negative, which would push the whole scene *away* from the viewer.
  Clamping at 0 keeps the current contract of `zero_parallax` (never pushes back), so in
  that case the face simply stays where it is, which is between the screen and 0.1.
  Allowing a negative shift is an open decision (see below).
- **Upper clamp (1.0):** unchanged from today.

## Multiple faces

YuNet regularly returns several faces (two-shots, crowds, posters, TVs in the background).
Options:

| # | Rule | Good for | Weak for |
|---|---|---|---|
| A | **Largest face** (box area) | Close-ups, over-the-shoulder (the back of a head is usually not detected, so the speaker facing camera wins), main actor vs extras | A large face on a poster or TV in the background |
| B | **Nearest face** (smallest face depth) | Picks the foreground actor even when framed small | Foreground extra or partially cropped face takes over |
| C | **Most central face** (distance from frame center, or from the upper-center "headroom" point) | Classic centered framing | Rule-of-thirds two-shots; often picks the empty middle |
| D | **Highest confidence** | Simple | Confidence tracks sharpness and pose, not importance |
| E | **Average or median depth of all accepted faces** | Two-shots: puts the plane between both actors so neither is far from it | Crowds; one far face drags the plane back |
| F | **Weighted score**: `area * confidence * center_weight` | Most robust single pick | More constants to tune |
| G | **Largest face, with two-shot blending**: if the 2nd largest face is at least `TWO_SHOT_AREA_RATIO` (e.g. 0.6) of the largest, use the mean depth of those two; otherwise use the largest | Covers both close-ups and balanced two-shots | Slightly more logic |

**Recommendation:** start with **A (largest face)** in Phase 2. It is one line, and it
matches how films frame the subject. If human testing shows two-shots being pulled toward
one actor, move to **G**. Keep the rule in one small function so it can be swapped.

Filters applied before choosing (named constants, not UI widgets):

- `SUBJECT_FACE_SCORE_THRESHOLD`: YuNet confidence (OpenCV default 0.9; start ~0.8).
- `SUBJECT_MIN_FACE_HEIGHT`: ignore faces smaller than a fraction of frame height
  (start ~0.04). This drops background extras and false positives in textures.
- `SUBJECT_FACE_SAMPLE_FRACTION`: sample depth only from the central part of the box
  (start 0.5, i.e. the middle 50% of width and height). YuNet boxes include hair and edge
  background, and the median over the inner region is much more stable.

## Guard: pop-out limit (recommended)

`NOTES.md` records that protruding content (negative parallax) produced bad artifacts with
this depth model. Subject mode deliberately puts everything nearer than the face in front
of the screen, so in a deep shot with a small, far-away face, a lot of the frame could
pop out.

Proposal: limit how far p1 may come in front of the screen:

    max_shift     = p1 ** power + SUBJECT_MAX_POP_OUT        (e.g. 0.15)
    scene_z_shift = min(face_curved - SUBJECT_SCREEN_DEPTH, max_shift)

When the guard is active, the face sits a little deeper than 0.1, which is still far more
stable than p1 mode. This should be logged so testing shows how often it triggers.

## Enabling it (needs a decision, see project rules)

The project rules say not to add node parameters without asking, and changing the type of
an existing widget would corrupt saved workflows (ComfyUI stores widget values by
position, and `adaptive_zero_parallax` saves as True/False).

Options:

1. **New BOOLEAN `subject_aware_zero_parallax`, added as the LAST widget** (default
   `False`). When on, it uses subject mode with the p1 fallback, whether or not
   `adaptive_zero_parallax` is on. Same Preview-mode restriction. *(Recommended: it is
   an explicit toggle, defaults off, and saved workflows are unaffected.)*
2. Turn `adaptive_zero_parallax` into a dropdown `off / nearest (p1) / subject (face)`.
   This is cleaner in the UI but breaks saved workflows unless a migration is written.
3. No new widget: `adaptive_zero_parallax = True` always means subject mode with the p1
   fallback. Fewest widgets, but p1-only mode is no longer selectable.

## Phases

### Phase 0 - Dependency and model file

- **OpenCV:** YuNet runs through `cv2.FaceDetectorYN` (OpenCV >= 4.5.4; the 2023mar
  model needs >= 4.8). OpenCV is already present wherever scene detection works
  (PySceneDetect's default video backend uses it, and `analyze_stereo.py` imports it), but
  it is not listed in `requirements.txt`. Check the version in the ComfyUI venv and add
  `opencv-python>=4.8` to `requirements.txt` (do not add `opencv-contrib-python`
  alongside it; the two packages conflict).
- **Model:** `face_detection_yunet_2023mar.onnx` (~230 KB, MIT license) from the
  `opencv/opencv_zoo` repository. Bundle it at `models/face_detection_yunet_2023mar.onnx`
  so laptops work offline with no first-run download. Confirm `.gitignore` does not
  exclude it, and note the license in the README.
- **Lazy import:** import `cv2` and load the model only when the toggle is on, so users
  with it off are unaffected even if OpenCV is missing or old. If loading fails, log one
  warning and fall back to p1 mode for the whole run rather than failing the job.

### Phase 1 - Face detection helper (`depth_runner.py`)

Put this next to `AdaptiveZeroParallax`, not in a new module (the logic is small):

- A `SubjectFaceDetector` class, created once per run:
  - `__init__`: `cv2.FaceDetectorYN.create(model_path, "", (0, 0), SUBJECT_FACE_SCORE_THRESHOLD)`.
  - `detect(frame_rgb_uint8) -> list[(x, y, w, h, score)]`: convert RGB to BGR, call
    `setInputSize((w, h))` (only when the size changes), `detect()`, return boxes that
    pass the size filter.
- Runs on CPU via OpenCV DNN: no VRAM, and the model's RAM use is negligible. It runs once
  per scene change on a full-resolution frame. Expect tens of ms at 1080p and more at 4K;
  at a few hundred cuts per film this adds seconds, not minutes.
- **Full frame, not downscaled**, as decided, so small faces in wide shots are still found.

### Phase 2 - Subject z-shift (`AdaptiveZeroParallax`)

- Add an optional `face_detector` argument to `__init__` (`None` = today's behaviour,
  bit-identical).
- Extend `shifts(...)` with an optional `frames_rgb_uint8` argument (the chunk's
  `source_chunk` numpy array, already RGB uint8 at source resolution).
- On a scene-start frame:
  1. If a detector is present: detect faces, choose one (rule A), take the median of
     `depth_b1hw[item]` over the inner box region. Depth is already restored to source
     resolution in `DepthAnythingRunner.infer`, so box coordinates map directly. Clip the
     box to frame bounds.
  2. If the face depth is finite: `scene_z_shift = clamp(face, 0, 1) ** power - SUBJECT_SCREEN_DEPTH`,
     apply the pop-out guard, then `applied = clamp(scene_z_shift + base_shift, 0, 1)`.
  3. Otherwise run the existing p1 path unchanged.
- Constants live next to `SCENE_Z_SHIFT_QUANTILE`: `SUBJECT_SCREEN_DEPTH = 0.1`,
  `SUBJECT_FACE_SCORE_THRESHOLD`, `SUBJECT_MIN_FACE_HEIGHT`,
  `SUBJECT_FACE_SAMPLE_FRACTION`, `SUBJECT_MAX_POP_OUT`, and `TWO_SHOT_AREA_RATIO` if
  rule G is adopted.

### Phase 3 - Wiring (`job_types.py`, `nodes.py`)

Assuming option 1 from "Enabling it":

- `StereoVideoJob`: add `subject_aware_zero_parallax: bool = False` at the end, plus
  `payload.setdefault(...)` in `from_handle` so older job handles still load.
- `StereoVideoSource.INPUT_TYPES`: new BOOLEAN as the **last** entry, with a tooltip.
  Add it to `build_job` as the last parameter. Preview-mode validation is the same as
  for `adaptive_zero_parallax`.
- `StereoVideoConvert.convert`: create the `AdaptiveZeroParallax` when either toggle is
  on, passing a `SubjectFaceDetector` only when the subject toggle is on. In the chunk
  loop, pass `source_chunk` into `shifts(...)`. The ordering stays: infer depth, resolve
  `depth_power`, then z-shift.
- Same `every_nth == 1` check as the existing adaptive z-plane.

### Phase 4 - Logging for human tuning

One INFO line per scene start, e.g.:

    Scene @ 1234 (51.4 s): 2 faces, chose largest (h=0.21, score=0.93), face depth 0.41 -> shift 0.54 [subject]
    Scene @ 1310 (54.6 s): 0 faces -> p1 0.12 -> shift 0.35 [p1 fallback]
    Scene @ 1402 (58.4 s): 1 face, pop-out guard limited shift 0.61 -> 0.48 [subject, capped]

This is the main tool for checking behaviour across a whole film without extra outputs.

### Phase 5 - Docs

- `README.md`: one bullet after `adaptive_zero_parallax`.
- `TUTORIAL.md`: a short paragraph on when to use p1 mode vs subject mode, and noting
  that the effect is strongest with `global` / `adaptive global` normalization (as with
  p1 mode).

### Phase 6 - Validation

**Agent-side (no ComfyUI):** `py_compile` on the changed files. Write a small CPU script
that feeds a synthetic depth map with a known face-box depth plus a stub detector
returning that box, and checks that the shift equals `face ** p - 0.1 + base` and that it
falls back to p1 when the stub returns nothing. Optionally, run the real YuNet on one
still image to confirm the model loads.

**Human:**
1. Toggle off: output is identical to a pre-change render, and saved workflows load with
   all widgets in place.
2. Toggle on, a dialogue-heavy clip: the speaking actor sits just behind the screen in
   every shot, and the cuts no longer force a re-convergence.
3. A clip with no people: behaves exactly like p1 mode (check the log says p1 fallback).
4. Two-shots and crowds: judge whether rule A is good enough or rule G is needed.
5. A deep wide shot with a small figure: check the pop-out guard against artifacts.
6. Full-length film: RAM/VRAM are unchanged from toggle off, and the extra run time is
   small.

### Phase 7 (later, optional) - Refinements

Not part of this work; listed so they are not forgotten:

- **Multi-frame detection per scene:** the first frame of a scene can be a blur, a
  transition, or a face turning away. Detect on the first N frames (or first + middle,
  using the chunk lookahead) and take the median face depth.
- **Easing across cuts:** ramp from the old shift to the new shift over ~0.5 s instead of
  jumping. With the subject held at a constant 0.1, the jump at a cut should already be
  much smaller than in p1 mode, so measure before adding this.
- **No-face fallback that understands subjects:** depth x center-prior weighted median
  instead of p1, for shots of animals, vehicles or objects.
- **Tracking within long shots:** re-detect every few seconds within a long scene and
  ease toward the new value, for actors walking toward or away from camera.

## Open decisions for Richard

1. How to enable it: option 1 (new last widget, recommended), 2 (dropdown), or 3
   (replace p1 mode).
2. Multiple faces: rule A to start, with G as the planned upgrade?
3. Pop-out guard: include it in Phase 2, and is 0.15 a reasonable starting cap?
4. Lower clamp: keep `>= 0` (never push a scene backward), or allow a negative shift so
   very near faces are also moved back to 0.1?
5. Model file: bundle it in `models/` (recommended, works offline), or download it on
   first use?

## Not changing

- Scene detection (`detect_scene_starts`, its settings).
- `AdaptiveDepthPower`.
- The p1 calculation itself; it remains the fallback and the behaviour of
  `adaptive_zero_parallax` alone.
- Default output: with the new toggle off, every code path is bit-identical to today.
