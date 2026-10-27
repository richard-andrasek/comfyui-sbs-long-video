# Subject-Aware Adaptive Z-Plane

## Goal

Reduce eye fatigue by keeping the subject of each shot (usually an actor's face) at a
stable, comfortable depth just behind the screen, the way studio stereographers
"pull convergence" shot by shot. Today, `adaptive_zero_parallax` (renamed to `zero_parallax = adaptive` by this plan) puts the nearest 1% (p1)
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
       scene_z_shift           = max(0.0, min(face_curved - SUBJECT_SCREEN_DEPTH, max_shift))
                                 (SUBJECT_SCREEN_DEPTH = 0.1; max_shift is the pop-out guard below)
       effective_zero_parallax = min(scene_z_shift, 1.0)        (constant_zero_parallax is not used)
       disparity               = (depth ** scene_depth_power - effective_zero_parallax) * disparity_span

   So the face's disparity is `0.1 * disparity_span` pixels behind the screen, regardless
   of where the face was in the raw depth range.

4. If no face is found (or the face depth is not finite), use the existing
   `scene_z_shift = p1 ** scene_depth_power`.
5. As today, the value is held for the whole scene, survives chunk boundaries, and only
   changes at the next scene start.

### Why the 0.1 is measured after `depth_power`

The renderer applies `depth_power` first and subtracts the combined shift afterwards
(see `GpuStereoRenderer` in `stereo_renderer.py`). Subtracting `face_depth ** power - 0.1` therefore moves
every pixel by the same amount. Depth spread and depth_power are unchanged, and the face
ends up at exactly 0.1 in the renderer's curved 0.0-1.0 space, i.e.
`0.1 * disparity_span` pixels of positive (behind-screen) parallax.

### Worked example

face_depth = 0.40, p1 = 0.10, depth_power = 0.5

| | p1 mode (today) | subject mode |
|---|---|---|
| scene_z_shift | 0.10 ** 0.5 = 0.316 | 0.40 ** 0.5 - 0.1 = 0.532 |
| face lands at | 0.632 - 0.316 = **0.316** behind | 0.632 - 0.532 = **0.100** behind |
| p1 lands at | 0.000 (on screen) | 0.316 - 0.532 = **-0.216** (in front) |

In subject mode, anything nearer than the face comes out of the screen. That is the
intended studio look, but here p1 would land 0.216 in front, past the 0.15 pop-out limit
(see "Guard: pop-out limit" below). The guard caps the shift at 0.316 + 0.15 = 0.466, so
p1 lands at **-0.150** (in front) and the face at 0.632 - 0.466 = **0.166** behind.

### Clamping behaviour

- **Lower clamp (decided: 0.0):** never move the scene away from the camera. If the face
  is already nearer than 0.1 (`face_curved < 0.1`), the shift would be negative, which
  would push the whole scene *away* from the viewer. In that case the face simply stays
  where it is, between the screen and 0.1.
  - The clamp is `max(0, ...)` on `scene_z_shift`, applied after the pop-out guard.
  - Invariant to verify: in `adaptive` and `subject`, `applied >= 0` for every scene.
    With the current depth normalization a negative shift is unlikely in practice, but
    it is still checked (see Phase 6).
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

**Decision:** use **D (highest confidence)**: of the faces that pass the filters below,
pick the one with the highest YuNet score. Reasoning:

- In an over-the-shoulder shot, the foreground face is mostly the back of a head and
  usually out of focus, so it scores low or is not detected; the in-focus subject facing
  the camera scores high.
- If two faces both score high, they are probably about equally important, so picking
  either is acceptable.
- A wrong pick is limited by the pop-out guard and the clamps, so the cost of a miss is
  small.

Keep the rule in one small function so it can be swapped later if testing shows a need.

Filters applied before choosing (named constants, not UI widgets):

- `SUBJECT_FACE_SCORE_THRESHOLD`: YuNet confidence (OpenCV default 0.9; start ~0.8).
- `SUBJECT_MIN_FACE_HEIGHT`: ignore faces smaller than a fraction of frame height
  (start ~0.04). This drops background extras and false positives in textures.
- `SUBJECT_FACE_SAMPLE_FRACTION`: sample depth only from the central part of the box
  (start 0.5, i.e. the middle 50% of width and height). YuNet boxes include hair and edge
  background, and the median over the inner region is much more stable.

## Guard: pop-out limit (decided: included)

`NOTES.md` records that protruding content (negative parallax) produced bad artifacts with
this depth model. Subject mode deliberately puts everything nearer than the face in front
of the screen, so in a deep shot with a small, far-away face, a lot of the frame could
pop out.

Decision: the guard is part of subject mode (Phase 2). It limits how far p1 may come in
front of the screen, to at most **0.15**. This is a hard limit on p1's final position,
not on how far the scene moves: `adaptive` would put p1 on the screen (shift
`p1 ** power`), and the subject shift may go at most 0.15 beyond that. Because
`constant_zero_parallax` is not added in `subject` mode (see "Enabling it"), nothing
pushes p1 further out after the guard.

    max_shift     = p1 ** power + SUBJECT_MAX_POP_OUT        (SUBJECT_MAX_POP_OUT = 0.15)
    scene_z_shift = max(0.0, min(face_curved - SUBJECT_SCREEN_DEPTH, max_shift))

(the outer `max(0.0, ...)` is the lower clamp, see "Clamping behaviour")

`SUBJECT_MAX_POP_OUT = 0.15` is a named module constant at the **top of `z_plane.py`**
(next to `SCENE_Z_SHIFT_QUANTILE`), so it can be tuned in one place. It is not a node
widget.

When the guard is active, the face sits a little deeper than 0.1, which is still far more
stable than p1 mode. This is logged so testing shows how often it triggers.

## Enabling it (decided): BOOLEAN becomes a dropdown, and both widgets are renamed

On `StereoVideoSource`:

| Today | Becomes |
|---|---|
| `zero_parallax` (FLOAT) | `constant_zero_parallax` (FLOAT, same range and default; used **only** when the dropdown is `constant`) |
| `adaptive_zero_parallax` (BOOLEAN) | `zero_parallax` (dropdown, below) |

The new `zero_parallax` dropdown has three options:

| Option | Behaviour |
|---|---|
| `constant` (**default**) | A fixed shift of `constant_zero_parallax` for the whole video, no per-scene shift. Same as today's `False`. |
| `adaptive` | p1 mode: nearest 1% of each scene on the screen plane. `constant_zero_parallax` is ignored. |
| `subject` | Subject mode from this plan (face at 0.1 behind the screen, p1 at most 0.15 in front), falling back to p1 when no face is found. `constant_zero_parallax` is ignored. |

The three modes are exclusive (decided): the shift is **either** `constant_zero_parallax`,
**or** the `adaptive` shift, **or** the `subject` shift. They are no longer added
together. This is a behaviour change for `adaptive`: today's `True` adds `zero_parallax`
on top of the p1 shift; `adaptive` now matches today's `True` only when `zero_parallax`
was 0.

Notes:

- Both widgets stay in their **current positions** (`constant_zero_parallax` where
  `zero_parallax` is now, the dropdown where `adaptive_zero_parallax` is now).
- **Existing workflows will break, and that is accepted** (decided: the change is big
  enough to warrant it). No migration code: no `web/` fix-up and no `true` / `false`
  mapping. Users re-set these two widgets in old workflows. The example workflows are
  updated (Phase 5).
- `constant` must stay bit-identical to today's `False`, so laptops see no change by
  default.

## Phases

### Phase 0 - Dependency and model file

- **OpenCV:** YuNet runs through `cv2.FaceDetectorYN` (OpenCV >= 4.5.4; the 2023mar
  model needs >= 4.8). OpenCV is already present wherever scene detection works
  (PySceneDetect's default video backend uses it, and `analyze_stereo.py` imports it), but
  it is not listed in `requirements.txt`. Check the version in the ComfyUI venv and add
  `opencv-python>=4.8` to `requirements.txt` (do not add `opencv-contrib-python`
  alongside it; the two packages conflict).
- **Model (decided):** `face_detection_yunet_2023mar.onnx` (~230 KB, MIT license) from
  the `opencv/opencv_zoo` repository. It is **not bundled** in the plugin and not
  downloaded automatically. The user places it in a ComfyUI models folder and loads it
  with a loader node (see "Face detector loader node" below). The README says where to
  get it, where to put it, and its license.
- **Lazy import:** `cv2` is imported and the model is loaded only inside the loader node,
  so users on `constant` or `adaptive` (who don't add the loader) are unaffected even if
  OpenCV is missing or old. If loading fails, the loader node raises an error (no silent
  fallback to p1).

### Face detector loader node (`nodes.py`, `config.py`)

- New node, in the same `CATEGORY`:
  - Class name and `NODE_CLASS_MAPPINGS` key: `SBS_FaceDetectorLoader`. The `SBS_`
    prefix avoids clashes, because the key is shared by every installed node pack and
    a later pack with the same key would silently replace ours.
  - Display name (`NODE_DISPLAY_NAME_MAPPINGS`): "Face Detector Loader".
  - No "Stereo" / "Video" in the name (decided): it loads a still-frame face detector,
    and the detector runs on the 2D source frame, not on stereo output.
- Register a `face_detection` models folder with `folder_paths` (e.g.
  `ComfyUI/models/face_detection/`). The node has one dropdown listing the `.onnx` files
  there.
- Output: a new custom type defined in `config.py` next to `JOB_TYPE` (e.g.
  `FACE_DETECTOR_TYPE = "SBS_FACE_DETECTOR"`), carrying a `SubjectFaceDetector`
  instance.
- If the folder is empty, the file is missing, or OpenCV can't load it, the node raises a
  clear error.

#### Does `StereoVideoSource` need the detector? No.

- `StereoVideoSource` only builds the job handle (a plain dict from `asdict`) and runs
  scene detection. It never looks at faces, and a detector object should not go into the
  handle.
- Face detection runs in `StereoVideoConvert`'s chunk loop, on the decoded frames, so the
  detector is an input to `StereoVideoConvert` only.
- `StereoVideoSource` can't see whether `StereoVideoConvert` has a detector connected,
  so the "subject without a detector" check has to live in `StereoVideoConvert`.

### Phase 1 - Face detection helper (`z_plane.py`)

Put this next to `AdaptiveZeroParallax` in `z_plane.py`, not in a new module (the logic is small):

- A `SubjectFaceDetector` class, created by the loader node from the chosen `.onnx` path:
  - `__init__`: `import cv2` here (not at the top of `z_plane.py`, which `nodes.py`
    always imports), then
    `cv2.FaceDetectorYN.create(model_path, "", (0, 0), SUBJECT_FACE_SCORE_THRESHOLD)`.
  - `detect(frame_rgb_uint8) -> list[(x, y, w, h, score)]`: convert RGB to BGR, call
    `setInputSize((w, h))` (only when the size changes), `detect()`, return boxes that
    pass the size filter.
- Runs on CPU via OpenCV DNN: no VRAM, and the model's RAM use is negligible. It runs once
  per scene change on a full-resolution frame. Expect tens of ms at 1080p and more at 4K;
  at a few hundred cuts per film this adds seconds, not minutes.
- **Runs on the 2D source frame**: the scene's first frame from `source_chunk` (the
  decoded original video), before any stereo rendering, so face boxes line up with the
  depth map pixel for pixel.
- **Full frame, not downscaled**, as decided, so small faces in wide shots are still found.

### Phase 2 - Subject z-shift (`AdaptiveZeroParallax`)

- Remove the `base_shift` argument from `AdaptiveZeroParallax.__init__`
  (`constant_zero_parallax` is no longer added in `adaptive` / `subject`). The p1 path
  becomes `applied = min(1.0, p1_curved)` (p1 is already clamped to 0-1), and the initial
  `self.applied` becomes 0.0.
- Add an optional `face_detector` argument to `__init__` (`None` = p1 behaviour).
- Extend `shifts(...)` with an optional `frames_rgb_uint8` argument (the chunk's
  `source_chunk` numpy array, already RGB uint8 at source resolution).
- On a scene-start frame:
  1. If a detector is present: detect faces, choose the highest-confidence face (rule D), take the median of
     `depth_b1hw[item]` over the inner box region. Depth is already restored to source
     resolution in `DepthAnythingRunner.infer`, so box coordinates map directly. Clip the
     box to frame bounds.
  2. If the face depth is finite: `scene_z_shift = clamp(face, 0, 1) ** power - SUBJECT_SCREEN_DEPTH`,
     apply the pop-out guard, then the lower clamp `scene_z_shift = max(0.0, scene_z_shift)`,
     then `applied = min(1.0, scene_z_shift)`.
  3. Otherwise run the p1 path (as above, without `base_shift`).
- Constants live at the top of `z_plane.py`, next to `SCENE_Z_SHIFT_QUANTILE`:
  `SUBJECT_SCREEN_DEPTH = 0.1`, `SUBJECT_MAX_POP_OUT = 0.15`,
  `SUBJECT_FACE_SCORE_THRESHOLD`, `SUBJECT_MIN_FACE_HEIGHT`,
  and `SUBJECT_FACE_SAMPLE_FRACTION`.

### Phase 3 - Wiring (`job_types.py`, `nodes.py`)

Using the dropdown from "Enabling it":

- Renames everywhere (`job_types.py`, `nodes.py`, `config.py` `DEFAULTS`, log and error
  messages): the float `zero_parallax` -> `constant_zero_parallax`, and
  `adaptive_zero_parallax` -> `zero_parallax`. Search the repo (including `tests/` and
  `test_*.py`) for both old names. **Exception:** leave the `zero_parallax` argument of
  `GpuStereoRenderer` in `stereo_renderer.py` as it is; it receives the final combined
  shift (`applied`), not the user's widget value.
- `StereoVideoJob`: `constant_zero_parallax: float` (was `zero_parallax`), and
  `zero_parallax: str = "constant"` (was `adaptive_zero_parallax: bool = False`).
  `from_handle` defaults `zero_parallax` to `"constant"` and rejects anything outside
  the three options. No mapping of old boolean values.
- `StereoVideoSource.INPUT_TYPES`: rename the FLOAT to `constant_zero_parallax`, and
  replace the BOOLEAN with `zero_parallax`:
  `(["constant", "adaptive", "subject"], {"default": "constant", "tooltip": ...})`, both
  in their current positions. Update `build_job`'s parameter names, types and defaults
  to match.
- Validation in `build_job` and `convert`: the Preview-mode (`every_nth == 1`) check
  applies to `adaptive` and `subject`, not to `constant`.
- `StereoVideoConvert.INPUT_TYPES`: add `face_detector` (type `FACE_DETECTOR_TYPE`) as
  an **optional** input, after `video_job`. It must be optional so `constant` /
  `adaptive` work without a loader node connected. Add it as the last
  parameter of `convert` (`face_detector=None`).
- `StereoVideoConvert.convert`, **first thing**, before probing, decoding or loading the
  depth model: if the mode is `subject` and `face_detector` is `None`, raise a
  `ValueError` so the node fails, e.g. *"zero_parallax is 'subject' but no face
  detector is connected. Connect a Face Detector Loader node to
  StereoVideoConvert's face_detector input."* If a detector is connected but the mode is
  not `subject`, ignore it.
- `StereoVideoConvert.convert`: the shift passed to the renderer is chosen by an if:
  `constant` -> `job.constant_zero_parallax`; `adaptive` / `subject` -> the result of
  `AdaptiveZeroParallax.shifts(...)`, with no `constant_zero_parallax` added.
- `StereoVideoConvert.convert`: create the `AdaptiveZeroParallax` for `adaptive` or
  `subject`, passing the `face_detector` only for `subject`. In the chunk loop, pass
  `source_chunk` into `shifts(...)`. The ordering stays: infer depth, resolve
  `depth_power`, then z-shift.

### Phase 4 - Logging for human tuning

One INFO line per scene start, e.g.:

    Scene @ 1234 (51.4 s): 2 faces, chose highest score (h=0.21, score=0.93), face depth 0.41 -> shift 0.54 [subject]
    Scene @ 1310 (54.6 s): 0 faces -> p1 0.12 -> shift 0.35 [p1 fallback]
    Scene @ 1402 (58.4 s): 1 face, pop-out guard limited shift 0.61 -> 0.48 [subject, capped]

This is the main tool for checking behaviour across a whole film without extra outputs.

### Phase 5 - Docs

All user-facing text that describes `adaptive_zero_parallax` as an on/off toggle, or
mentions the float `zero_parallax`, must be updated for the new names and the three
dropdown options:

- **Node descriptions in `nodes.py`:** the `DESCRIPTION` strings shown under the nodes
  in ComfyUI (`StereoVideoSource`, `StereoVideoConvert`, and any other node whose text
  mentions the z-plane), the `tooltip` for the `zero_parallax` dropdown (explain
  `constant` / `adaptive` / `subject`, default `constant`, and that `adaptive` and
  `subject` require Preview mode off), and the `constant_zero_parallax` tooltip (used
  only when `zero_parallax` is `constant`; ignored by `adaptive` and `subject`). Also update any error messages that say
  "enabled" or "disabled" for this widget.
- `README.md`: rename the `zero_parallax` bullet to `constant_zero_parallax`, and rewrite
  the `adaptive_zero_parallax` bullet as `zero_parallax` with the three options and the
  default. Add a short note that the renames break existing workflows. Document the new loader node and `StereoVideoConvert`'s `face_detector`
  input: where to download the YuNet `.onnx`, which models folder to put it in, its MIT
  license, and that `subject` fails without it.
- `DESCRIPTION` and tooltips for the new loader node, and an updated
  `StereoVideoConvert` description mentioning the optional face detector input.
- `TUTORIAL.md`: update the "Adaptive zero parallax" section ("With
  `adaptive_zero_parallax` enabled..." becomes `zero_parallax = adaptive`; references to the float become
  `constant_zero_parallax`; remove the text saying `zero_parallax` is added on top, since
  the modes are now exclusive), and add a short
  paragraph on when to use `adaptive` vs `subject`, noting that both work best with
  `global` / `adaptive global` normalization.
- `example_workflows/`: update every workflow for the renamed widgets and the new
  dropdown values.

### Phase 6 - Validation

**Agent-side (no ComfyUI):** `py_compile` on the changed files. Write a small CPU script
that feeds a synthetic depth map with a known face-box depth plus a stub detector
returning that box, and checks that:

- the shift equals `face ** p - 0.1` when the pop-out guard does not trigger;
- when `face ** p - 0.1 > p1 ** p + 0.15`, the shift is capped at `p1 ** p + 0.15`, so p1
  lands exactly 0.15 in front of the screen;
- it falls back to `p1 ** p` when the stub returns nothing;
- **never moves the scene away:** a face nearer than 0.1 (`face ** p < 0.1`) gives
  `applied == 0`, and across a sweep of face depths, p1 values and powers,
  `applied >= 0` in `adaptive` and `subject` modes;
- **modes are exclusive:** in `adaptive` and `subject`, changing `constant_zero_parallax`
  does not change the shift; in `constant`, the shift is exactly
  `constant_zero_parallax`.

Optionally, run the real YuNet on one still image to confirm the model loads.

**Human:**
1. `constant`: output is identical to a pre-change render with the same
   `constant_zero_parallax` value. `adaptive`: output is identical to today's `True` with
   `zero_parallax = 0`.
   The updated example workflows load and run.
2. `subject`, a dialogue-heavy clip: the speaking actor sits just behind the screen in
   every shot, and the cuts no longer force a re-convergence.
3. A clip with no people: behaves exactly like p1 mode (check the log says p1 fallback).
4. Two-shots, over-the-shoulder shots and crowds: check that the highest-confidence pick
   lands on an acceptable face (the log shows each pick's score).
5. A deep wide shot with a small figure: check the pop-out guard against artifacts.
6. Full-length film: RAM/VRAM are unchanged from `constant`, and the extra run time is
   small.
7. The node descriptions and tooltips in ComfyUI match the dropdown.
8. `subject` with no face detector connected: `StereoVideoConvert` fails immediately with
   the clear error, before any decoding. `constant` / `adaptive` with no detector run as
   before.
9. The loader node lists `.onnx` files from the models folder and errors clearly when the
   folder is empty.

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

1. ~~How to enable it~~ **Decided:** dropdown `constant` / `adaptive` / `subject`,
   default `constant` (see "Enabling it").
2. ~~Multiple faces~~ **Decided:** rule D, highest confidence.
3. ~~Pop-out guard~~ **Decided:** included in Phase 2, with `SUBJECT_MAX_POP_OUT = 0.15`
   as a constant at the top of `z_plane.py`.
4. ~~Lower clamp~~ **Decided:** `>= 0`, never move the scene away from the camera;
   verified in Phase 6.
5. ~~Model file~~ **Decided:** loaded by a new loader node and passed to
   `StereoVideoConvert` as an optional `face_detector` input; `subject` without it fails
   the node. The loader is `SBS_FaceDetectorLoader` ("Face Detector Loader").
6. ~~Combining modes~~ **Decided:** the modes are exclusive; `constant_zero_parallax`
   applies only in `constant` (see "Enabling it"). With nothing added after it, the
   pop-out guard is a hard limit on p1.

## Not changing

- Scene detection (`detect_scene_starts`, its settings).
- `AdaptiveDepthPower`.
- The p1 calculation itself; it remains the fallback for `subject` and the behaviour of
  `adaptive`.
- Default output: with the dropdown on `constant`, every code path is bit-identical to
  today.
