# Notes

This is just for my personal note taking on possible improvements.

Of course, with anything, there's a quality vs time constraint.  So these things should be considered as options, 
not hard requirements.  The pipeline with its basic settings works quite well, but with quality trade-offs

## Priorities

* **Priority 3 — Temporal disparity stabilization:** Reduce stereo shimmer and small
  frame-to-frame disparity changes after depth-to-disparity conversion.

* **Priority 4 — Z-buffer DIBR:** Improve visibility/occlusion handling during stereo
  reprojection and reduce foreground/background overlap artifacts.

* **Priority 5 — Joint bilateral edge refinement:** Add as an alternative to FGS for
  comparison and potentially better handling of certain fine edges and textures.

* **Priority 6 — Hole-filling strategy:** Improve disocclusion filling, particularly after
  implementing Z-buffer DIBR.

* **Priority 7 — Particle-depth correction:** Address isolated rain, snow, dust, or similar
  objects receiving implausible foreground depth. Useful, but relatively specialized.


## Potential Improvements

### Joint bilateral edge refinement

An alternative to the Fast Global Smoother (FGS) already implemented. A joint bilateral
filter would use the RGB frame as a guide while smoothing the DA3 depth map. Nearby
pixels are allowed to influence one another based on both spatial distance and RGB
similarity, reducing depth bleeding across strong object boundaries.

This may produce slightly different results from FGS, particularly around fine details,
hair, and textured surfaces. It is worth implementing as an alternative refinement method
rather than replacing FGS.

Potential controls:
- `depth_edge_method`: `fgs` / `joint_bilateral`
- `depth_edge_sigma_color`: sensitivity to RGB differences
- `depth_edge_sigma_space`: spatial smoothing radius

Implementation should remain entirely in PyTorch/CUDA to avoid CPU transfers.

### Temporal disparity stabilization

Apply temporal smoothing after depth has been converted into stereo disparity. The goal
is to suppress small frame-to-frame changes in disparity that are not perceptually
meaningful but can cause visible stereo shimmer or depth pumping.

An EMA or similar low-pass filter could blend the current disparity with the previous
frame while preserving larger intentional changes.

This should be applied after the depth response curve and disparity calculation so that
it stabilizes the quantity that actually controls horizontal stereo displacement.

Potential controls:
- `temporal_disparity_stability`: on/off
- `temporal_disparity_strength`: approximately 0.05-0.25

Care should be taken around genuine scene changes, cuts, and fast camera movement. A
strong temporal filter could introduce lag or ghosting if it is too aggressive.

### Z-buffer DIBR

Replace or augment the current `grid_sample`-based reprojection with a visibility-aware
Z-buffer DIBR renderer.

When multiple source pixels project onto the same destination pixel, the Z-buffer keeps
the pixel belonging to the nearest visible surface. This more accurately models occlusion
and disocclusion than simple image warping.

This could reduce artifacts where foreground and background surfaces overlap during
stereo reprojection and could improve the quality of newly exposed regions.

The implementation would need to:
1. Project source pixels into the target view.
2. Calculate their target coordinates and depth.
3. Resolve collisions using the nearest-depth sample.
4. Identify pixels with no valid source sample.
5. Pass those holes to the hole-filling stage.

This is a substantially larger renderer change than the depth refinements and should be
tested independently. It may also increase GPU memory usage and processing time.

### Hole-filling strategy

Improve the treatment of pixels that become exposed when the image is shifted into the
left or right stereo view.

The current approach uses repeated local averaging to fill holes. This is fast, but it
does not necessarily understand the geometry of the newly exposed region and can smear
foreground colors into background areas.

Possible improvements include:
- Depth-aware hole filling.
- Filling from the background side of an occlusion boundary.
- Directional/inpainting-style filling based on the reprojection direction.
- Combining Z-buffer visibility information with the existing pooling approach.

This should be considered together with Z-buffer DIBR, since better visibility information
makes it possible to distinguish true disocclusion holes from ordinary missing pixels.

### Particle-depth correction

Detect small isolated foreground objects in the depth map that are likely to be
particles such as rain, snow, dust, smoke fragments, or compression/noise artifacts.

Depth models can occasionally assign an isolated particle an extreme foreground depth
even though it belongs visually to the background. In a stereo conversion this can turn a
tiny particle into a distracting object floating far in front of the scene.

A particle-depth correction stage could identify small isolated depth regions and replace
their depth with an estimate of the surrounding background depth.

This should be optional because legitimate small foreground objects—such as text,
small props, or distant people—could otherwise be incorrectly flattened into the
background.

A useful implementation would probably combine connected-component/area tests with
local depth statistics rather than relying solely on object size.


## Other Possible Improvements:

These are some concepts from SBSCrafter:
https://github.com/cnkanwei/ComfyUI-SBSCrafter

### Softmax splatting / depth-aware forward warp

Replace or augment `grid_sample` with depth-aware forward warping so overlapping pixels
are resolved according to depth, improving occlusion handling and stereo boundaries.

### Depth-adaptive blend radius

Adjust hole/blend feathering based on scene depth rather than using a fixed radius, which
can produce cleaner transitions around foreground objects and disocclusions.

### Reinhard-style ring color matching

Match the color characteristics of filled regions to the surrounding image, reducing
visible seams caused by differences in brightness, contrast, or color.

### small_hole_px

Detect small reprojection holes and cracks separately from larger disocclusions, allowing
them to be filled cheaply without invoking more expensive processing.

### keep_original_left/right

Keep one eye pixel-exact and synthesize only the other eye, reducing processing and
avoiding unnecessary interpolation artifacts in the preserved view.

### Auto-convergence / zero-parallax calculation

Automatically determine a suitable reference depth plane so that the stereo image has
consistent convergence without requiring manual adjustment for every scene.

### Resolution-independent disparity (Percentage based, rather than pixel-based)

Express stereo separation as a percentage of image width so the same setting produces
consistent stereo strength across different output resolutions.

### Particle Depth Fix

Detect small isolated objects such as rain, snow, or dust that receive implausible
foreground depth and replace them with an estimate of the surrounding background depth.
