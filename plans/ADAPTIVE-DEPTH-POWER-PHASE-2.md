TASK: Add ADAPTIVE depth_power to comfyui-sbs-long-video. When enabled, depth_power is chosen
per scene from the scene's raw-depth p99 and blended between a "near" value (the existing
depth_power UI input) and a "far" value (new UI input adaptive_depth_max). When disabled,
behavior and output must be IDENTICAL to today.

IMPORTANT COMPATIBILITY LIMITATION
- Adaptive depth_power does NOT work when `preview_run` (Preview mode) is enabled. Preview mode
  uses a 30-frame source stride, which is incompatible with applying adaptive power at each
  detected scene boundary; attempting to run with both `adaptive_depth_power` and `preview_run` enabled will throw
  an error. Reject this combination during job validation with a clear `ValueError` explaining
  that adaptive depth power requires Preview mode to be disabled. Do not silently fall back to
  fixed `depth_power` or warn and continue.
- Adaptive depth power remains available for normal runs (`preview_run` disabled). Update any
  user-facing option descriptions or documentation added for this feature to state this limit.

READ FIRST
- nodes.py: StereoVideoSource.INPUT_TYPES / build_job, StereoVideoConvert.convert (the chunk
  loop calls runner.infer(...) then renderer.render(..., depth_power=job.depth_power)).
- job_types.py: StereoVideoJob dataclass + to_handle/from_handle.
- stereo_renderer.py: GpuStereoRenderer.render applies
  depth.clamp(0,1).pow(max(depth_power, 1e-3)) with ONE scalar for the whole chunk.
- depth_runner.py: DepthAnythingRunner.infer and the raw model output (the tensor returned
  by _infer_official_api/_infer_model/_infer_pipeline, before normalization).
- config.py: DEFAULTS and how depth_power's default is defined.
- The EXISTING scene-detection + "Scene start:" depth-stats logging added earlier (it is in
  the working copy, not necessarily on GitHub main; grep for "Scene start" / AdaptiveDetector).
  REUSE its list of scene-start source frame indices and its scene-boundary detection helper.
  If the membership test is inline, factor it into one helper so the logging and this feature
  share it. Do NOT re-implement scene detection. At each detected scene boundary, use the raw
  p99 from the first inferred frame for that scene to choose the scene target immediately. Do
  not collect multiple frames or compute a median.

1) UI + JOB PLUMBING (the only two new UI inputs)
- StereoVideoSource.INPUT_TYPES: append at the END of "required" (so saved workflows do not
  shift widget values):
    "adaptive_depth_power": BOOLEAN, default False, tooltip: "Choose depth_power per scene.
        Close-up scenes use depth_power; deep/scenery scenes use adaptive_depth_max."
    "adaptive_depth_max": FLOAT, default = DEFAULTS.depth_power (so enabling the flag alone
        changes nothing), same min/max/step as depth_power, tooltip: "depth_power used for
        deep/scenery scenes when adaptive_depth_power is on. Ignored otherwise."
- Modify the existing depth_power input only to change the description. No other changes are allowed
  When adaptive is on, depth_power simply serves as the near-scene value.
- build_job: add both params LAST with defaults (False, DEFAULTS.depth_power). If
  adaptive_depth_power is True, validate adaptive_depth_max is finite and > 0 (same style as
  the global_depth_max check), and raise a clear ValueError if preview_run is also True. No
  ordering requirement between depth_power and adaptive_depth_max.
- StereoVideoJob: append fields with defaults after global_depth_max:
    adaptive_depth_power: bool = False
    adaptive_depth_max: float = 0.0
  Pass them from build_job. Old handles still load because both have defaults.

2) HARD-CODED CONSTANTS (module level in depth_runner.py, NOT UI inputs; each with a comment)
    NEAR_SCENE_P99 = 400.0      # raw DA3 depth p99 at/below which a scene gets the near power
    FAR_SCENE_P99 = 450.0       # raw DA3 depth p99 at/above which a scene gets the far power
  Comment that 400/450 were calibrated from logged raw p99 on DA3 output and must be
  revisited if the depth model or backend changes. The p99 is taken once from the first frame
  inferred at each scene change; no sampling window or median is used.

3) THE MAPPER (new class AdaptiveDepthPower in depth_runner.py; no new files)
  __init__(near_power, far_power, scene_starts, source_fps, stride)
  powers(frame_indices: list[int], raw_p99: list[float]) -> list[float]   # one per frame
  Frames arrive in increasing order across chunks, so keep state on the instance:
    target, applied, first_frame_done, and the current scene identity/start frame.
  Per frame, in order:
    a. Detect whether this frame is the first inferred frame at or after a scene cut using the
       shared scene-boundary helper (a cut in (idx - stride, idx] applies to that frame).
    b. At the first frame of the video and at each scene boundary, map that frame raw p99:
           t = clamp((raw_p99 - NEAR_SCENE_P99) / (FAR_SCENE_P99 - NEAR_SCENE_P99), 0, 1)
           t = t * t * (3 - 2 * t)                       # smoothstep, flat at both ends
           target = near_power + t * (far_power - near_power)
       Do not gather samples, average values, or calculate a median. Set applied = target
       immediately on the first frame of the video and at every scene change. This is a full
       swing to the newly mapped scene power with no ramp or slew limit. Hold that power for
       the rest of the scene.
    c. Return applied for this frame.
  Log once per scene, at its first inferred frame, at INFO:
    "Adaptive depth_power: frame <N> time (<S>) p99=<X> calculated depth=<Y>"

4) RAW p99 FROM THE RUNNER
- DepthAnythingRunner.infer: add kwarg want_raw_p99: bool = False. When True, after the model
  call and BEFORE any normalization/resize, compute per batch item the 0.99 quantile of the
  finite raw depth values and store the list on self.last_raw_p99. Use the same tensor and the
  same finite-value filtering as the existing "Scene start" raw-stats logging so the 400/450
  calibration matches. torch.quantile has a ~16M element limit: subsample by a fixed stride to
  <= ~1M elements first. Skip all of this when want_raw_p99 is False.
- Add a tiny read-only property on the runner that says whether the official Depth Anything 3
  backend loaded (self._api_backend is not None). The transformers fallback loads Depth-Anything
  V2, whose raw values are on a different scale, so 400/450 are meaningless there.

5) CONVERT LOOP (nodes.py, StereoVideoConvert.convert)
- After creating the runner and calling runner.load(): if job.adaptive_depth_power and the
  official backend is NOT in use, LOGGER.warning that adaptive depth_power is disabled
  (fallback backend) and run with the fixed job.depth_power.
- Otherwise create adaptive = AdaptiveDepthPower(near_power=job.depth_power,
  far_power=job.adaptive_depth_max, scene_starts=<existing list>, source_fps=job.source_fps,
  stride=job.every_nth).
- In the chunk loop: pass want_raw_p99=(adaptive is not None) to runner.infer. Build
  frame_indices = [processed_frames * job.every_nth + i * job.every_nth for i in range(B)]
  (same formula infer already uses), then powers = adaptive.powers(frame_indices,
  runner.last_raw_p99) and pass powers (list of B floats) to renderer.render as depth_power.
  When adaptive is None pass job.depth_power exactly as today.

6) RENDERER (stereo_renderer.py)
- render(depth_power) must accept a float OR a sequence/tensor of length B. Replace
      depth.clamp(0, 1).pow(max(depth_power, 1e-3))
  with: convert depth_power via torch.as_tensor(..., device=depth.device, dtype=depth.dtype),
  clamp(min=1e-3), reshape to (-1, 1, 1, 1) (a scalar becomes shape (1,1,1,1)), then pow.
  The scalar path must produce numerically identical results to the current code.

7) DOCS + TESTS
- README.md node reference: document adaptive_depth_power and adaptive_depth_max, plus one
  paragraph explaining near = depth_power, far = adaptive_depth_max, and the per-scene p99
  mapping from the first inferred frame at each scene change. State that no frame samples or
  median are used, and state that the full mapped power takes effect immediately at every scene change.
- TUTORIAL.md reference: document adaptive_depth_power and adaptive_depth_max, following the style of
  the document
