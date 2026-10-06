# Notes

This is just for my personal note taking on possible improvements.

Of course, with anything, there's a quality vs time constraint.  So these things should be considered as options, 
not hard requirements.  The pipeline with its basic settings works quite well, but with quality trade-offs

## Priorities

* **Priority 4 — Z-buffer DIBR:** Improve visibility/occlusion handling during stereo
  reprojection and reduce foreground/background overlap artifacts.

* **Priority 6 — Hole-filling strategy:** Improve disocclusion filling, particularly after
  implementing Z-buffer DIBR.

* **Priority 7 — Particle-depth correction:** Address isolated rain, snow, dust, or similar
  objects receiving implausible foreground depth. Useful, but relatively specialized.

## THINGS THAT DIDN'T WORK

### EMA / "Flicker reduction"

It's nice in theory and had a practical effect of actually reducing flicker in the generated depth map.  
However, in the long run, the global depth map was far, far more successfull

### Global Depth Normalization across the screen

So, in theory, 3d movies should be able to pop out at you.  But in practice, this process didn't work
so well with the model (Depth Anything V3 on global depth normalization) that I am using. As such, I scrapped that
plan.  (The problem: it gave nasty artifacts when protruding "outwards".)  Real studioes use much better
depth planning or manual depth generation (or true 3d rendering) for such effects

### Convergence Plane

This is related to the convergence plane...the above is what I actually tried to implement and what failed.
In theory, I could give the Convergence Plane a go, allowing a movie to be "shifted outwards" from the screen.
But in my testing, I didn't feel like the juice was worth the squeeze.  The addition of artifacts was highly 
undesirable.  And I didn't feel like fighting through that to get it working, only to have all the movies 
effectively shifted forward only a tiny fraction.  In otherwards, what we have so far is great and this
concept wouldn't have made it better.

### Auto-convergence / zero-parallax calculation

This is directly related, but even worse. The problem is that auto-convergence could cause "depth pumping",
given that this is a frame-to-frame rendering.  As such, this process is more likely to fail the succeed. Skip it.

### Joint bilateral edge refinement

The Fast Global Smoothing does a good job. I'm not convinced that this will be a better option.

### Temporal disparity stabilization

DepthAnythingV3 upgrade solved this.  This is unneeded.

### Softmax splatting / depth-aware forward warp

This is an alternative to Z-buffer DIBR.  However, it could cause artifacts and isn't as good as the other. So, skip it.

## Potential Improvements

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


### 3 Depth Layers of Snow

This is an improvement to the particle-depth correction with increased cost.

Generate:

foreground
midground
background

particle depth.

This should make the stereoscopic effect dramatically better


### Reinhard-style ring color matching

Match the color characteristics of filled regions to the surrounding image, reducing
visible seams caused by differences in brightness, contrast, or color.

