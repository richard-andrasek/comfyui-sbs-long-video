text
```

OVERVIEW:

Add a zero-parallax offset control to the ComfyUI SBS Long Video project.

The purpose is to independently control where the generated 3D scene sits relative to the screen plane, without changing the depth map or the overall depth strength.

The new setting, zero_parallax_ratio, will apply a constant horizontal disparity offset to the generated left-eye and right-eye images. The offset will move the scene farther behind the screen or closer to/in front of the screen, depending on its sign.

The existing disparity_ratio and depth_power settings will continue to work as they do now.

A value of zero must preserve the current rendering behavior. Existing workflows and saved jobs must remain compatible.

The offset will be expressed as a fraction of the individual eye-image width, making it resolution-independent.

FILE CHANGES:

1. config.py

Add a zero_parallax_ratio configuration default with a value of 0.0.

This ensures that existing configurations retain their current behavior unless the user explicitly changes the new setting.

2. nodes.py

Add a zero_parallax_ratio floating-point input to the Stereo Video Source node.

Provide a default of 0.0, a reasonable adjustment range, an appropriate step size, and a tooltip explaining the effect of the setting.

Pass the value through the node's execution path into the job configuration and renderer.

Ensure that the new input does not interfere with existing node inputs or saved workflows.

3. job_types.py

Add zero_parallax_ratio to the job data structure.

Provide a default value of 0.0 when loading older saved job data that does not contain the new field.

Ensure that job serialization and deserialization continue to work with both old and new job configurations.

4. stereo_renderer.py

Add zero_parallax_ratio as an optional renderer parameter, defaulting to 0.0.

Calculate the zero-parallax offset in pixels by multiplying the ratio by the width of an individual eye image.

Apply this offset to the existing depth-dependent disparity before generating the left-eye and right-eye sampling coordinates.

The left and right images must receive equal and opposite shifts so that the change produces the intended stereo separation.

A zero offset must leave the existing disparity calculation and pixel-warping behavior unchanged.

Preserve the existing sampling, occlusion handling, hole filling, and output layout.

The offset must change the placement of the scene relative to the screen plane without modifying the depth map or the depth_power response curve.

VALIDATION:

1. Restart ComfyUI and verify that the Stereo Video Source node exposes the new zero_parallax_ratio input.

2. Generate a short test clip with zero_parallax_ratio set to 0.0. Confirm that the result matches the existing rendering behavior.

3. Generate the same clip with a small positive offset, such as 0.0005, and then 0.0010.

4. Generate another clip with a small negative offset to verify that the effect reverses direction.

5. Analyze the resulting SBS images with the disparity-analysis script. Confirm that changing the offset moves the disparity distribution in the expected direction while leaving its overall spread approximately unchanged.

6. Verify that the zero-parallax offset is scaled according to the individual eye-image width, not the full side-by-side output width.

7. Inspect foreground and background objects to confirm that the perceived depth placement changes as expected.

8. Check for newly exposed image borders, disocclusion artifacts, and hole-filling problems caused by the additional shifts.

9. Test an existing workflow and an older saved job to ensure that omitting the new setting defaults to zero and does not break backward compatibility.

10. Once the behavior is verified, use short clips to tune zero_parallax_ratio independently from disparity_ratio and depth_power before processing a full movie.

EXPECTED RESULT:

The user can independently adjust the generated scene's placement relative to the screen plane while retaining control over the total depth strength and depth distribution.

The default setting preserves existing behavior, and older workflows remain compatible.
```