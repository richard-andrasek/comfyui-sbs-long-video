OVERVIEW

Add a zero_parallax control to the ComfyUI SBS Long Video project.

zero_parallax is a float from 0.0 to 1.0, applied to the normalized depth
(0-1, after the depth_power curve) before disparity is computed. It is a
fraction of the depth range, not a percentage and not a pixel value.

  0.00 -> current behavior, unchanged
  0.25 -> scene shifts forward by 25% of the depth range
           (the nearest 25% ends up in front of the screen)
  1.00 -> the entire depth range is in front of the screen

Formula (stereo_renderer.py):
  disparity = (depth.clamp(0, 1).pow(depth_power) - zero_parallax) * disparity_span

disparity_percent and depth_power keep working as they do now. The depth map,
the depth_power curve and the total depth spread are not modified; only the
position of that spread relative to the screen plane moves. Negative disparity
is already handled by the existing left(+disp) / right(-disp) grid logic, so
the shift is automatically equal and opposite for the two eyes.

Resolution independence is automatic: disparity_span is already
disparity_percent/100 * single-eye width.

Sign convention (derived from the grid math, verify on a test clip):
positive disparity = behind screen, negative = in front. Depth 0 currently
sits on the screen plane, so subtracting zero_parallax moves the scene forward.

FILE CHANGES

1. config.py
   In PluginDefaults add:
       zero_parallax: float = 0.0

2. nodes.py  (StereoVideoSource)
   a. In INPUT_TYPES["required"], add as the LAST entry (after
      depth_edge_refine_method). Position matters: ComfyUI stores widget
      values by position, so inserting it earlier would corrupt saved workflows.
          "zero_parallax": ("FLOAT", {
              "default": DEFAULTS.zero_parallax, "min": 0.0, "max": 1.0, "step": 0.01,
              "tooltip": "Shifts the whole scene toward the viewer, as a fraction of "
                         "the depth range. 0 = unchanged, 0.25 = nearest 25% of the "
                         "range in front of the screen, 1 = entire range in front. "
                         "Does not change depth strength or depth_power."}),
   b. build_job(): add parameter `zero_parallax: float = 0.0` as the LAST
      parameter, and validate like global_depth_max:
          if not math.isfinite(float(zero_parallax)) or not 0.0 <= zero_parallax <= 1.0:
              raise ValueError("zero_parallax must be between 0.0 and 1.0")
   c. Pass `zero_parallax=zero_parallax` into the StereoVideoJob(...) call.
   d. StereoVideoConvert.convert(): add `zero_parallax=job.zero_parallax` to the
      renderer.render(...) call.

3. job_types.py  (StereoVideoJob)
   a. Add the field with the other defaulted fields (after global_depth_max):
          zero_parallax: float = 0.0
   b. In from_handle(), next to the other setdefault calls:
          payload.setdefault("zero_parallax", 0.0)
          payload["zero_parallax"] = float(payload["zero_parallax"])
      Old saved jobs load with 0.0; new jobs round-trip through asdict() as-is.

4. stereo_renderer.py  (GpuStereoRenderer.render)
   a. Add `zero_parallax: float = 0.0` as the last parameter (keeps existing
      callers working).
   b. Replace the disparity line with:
          disparity = (
              depth.clamp(0, 1).pow(max(depth_power, 1e-3)) - float(zero_parallax)
          ) * disparity_span
   c. Leave everything else untouched: grid construction, grid_sample, _oob_mask,
      _fill_holes and the output layout.
   At zero_parallax == 0.0 the result is bit-identical to the current output.

5. README.md
   Add to the StereoVideoSource reference:
   - `zero_parallax`: moves the scene toward the viewer as a fraction of the
     depth range (0.0-1.0); 0 leaves rendering unchanged.

VALIDATION - HUMAN only--no agent to perform this.

1. Restart ComfyUI; confirm Stereo Video Source shows zero_parallax as the last
   widget, and that an existing saved workflow loads with the other widgets
   still holding their old values and zero_parallax = 0.0.
2. Render a short clip at 0.0 and confirm it matches current output (identical
   frames, or a zero-diff against a pre-change render).
3. Render the same clip at 0.25, 0.5 and 1.0. Confirm the scene moves toward
   the viewer. At 1.0 the whole scene should be in front of the screen. If
   the direction is reversed, flip the sign in the renderer line.
4. Run the disparity-analysis script on each output. Expected: the disparity
   distribution moves by zero_parallax * disparity_span pixels (span =
   disparity_percent/100 * single-eye width) while its spread (e.g. p99 - p1)
   stays unchanged.
5. Confirm the shift is relative to the single-eye width, not the SBS width.
6. Inspect foreground and background objects: relative separation is the same
   as at 0.0; only the position relative to the screen changes.
7. Check borders and disocclusions. Maximum |shift| is never larger than the
   current disparity_span (the range moves from [0, span] to [-zp*span,
   (1-zp)*span]), so exposed-edge width should be no worse than today, and
   around zero_parallax 0.5 it is roughly halved. The existing 3x3 hole
   fill (4 iterations) is unchanged; wide edge gaps with large
   disparity_percent were already a limitation.
8. Load an older saved job handle (no zero_parallax key) and confirm it defaults
   to 0.0 and renders as before.
9. Tune zero_parallax on short clips (preview_run) independently of
   disparity_percent and depth_power before processing a full movie.

EXPECTED RESULT

The scene can be moved in front of the screen without changing depth strength
or the depth curve. The default of 0.0 preserves existing behavior, and
older workflows and saved jobs load unchanged.