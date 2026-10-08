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


### Softmax splatting / depth-aware forward warp

This is an alternative to Z-buffer DIBR.  However, it could cause artifacts and isn't as good as the other. So, skip it.

## Potential Improvements

### Adaptive Global Depth Scaling

The "global" depth scaling has a problem. If the p99 value goes over the configured value (default 750), the whole frame is
re-distributed based on the p99 value.  

This means that if we have 20 frames with a p99 of 700 and the max is configured for 750, we have no problems. A value of
375 within that 0..750 scale will always be at 0.5.  However, if we hit a frame with a max (or rather, a p99) of 900, though, 
that frame will scale based on the 0..900.  Meaning a value of 375 within that would be 0.42--a rather significant difference.

These high-value frames can cause discomfort over time due to "breathing" of the frames... 
Our example value of 375 can be 0.5, 0.5 0.48, 0.5, 0.42, etc. This "breathing" causes VR discomfort.

To approach this problem, we need to have the global max become an ACTUAL max.  There will be no more values above that max.
Meaning, a value of 800 would be clamped down to 750.  900 becomes 750.

To deal with the fact that this can cause some artifacts, we need to the global max be adaptive as it transitions thorugh a movie.
If we're hitting a p99 of 850 for a long time, the global max should drift from 750 to 775, then 800, and up to 850.
This should be a slow transition to prevent significant changes throughout a scene.

Note that this global max should NEVER DECREASE. As it goes through the movie, an increase in that max should be stable through
the remaining movie.

**NOTE 1: Improve the log:** 
pct_over = over_mask.float().mean().item()
p99 = torch.quantile(depth.flatten(), 0.99).item()
max_depth = depth.max().item()
giving you something like:

frame 141688 (6039.283 s): (P99: 891.500) (max: 1034.721) (>750: 0.183%)

**NOTE 2: (leaky) COUNT:**
To accomplish this adaptation, we don't want to increase the global max every tiem we have a p99 over the configured value.

Instead, we want to count 20 frames.  If we have 20 frames that exceed the p99, then we'll increase the global_max_depth.  
However, we don't want to wait 30 minutes into the movie and increase it...To prevent this, we'll have a "leaky counter".

Specifically:

leaky_ratio = 30;  // 30 good frames will reduce 1 bad frame
if p99 > global_max_depth:
    over_counter += leaky_ratio; 
    max_p99 = max(max_p99, p99);
else:
    over_counter -= 1;
if(over_counter > 20 * leaky_ratio):  //20 bad frames = re-adjust
    adjust_global_max_depth();  

**NOTE 3: Slow adjustment of global_depth_max**
We also don't want our global depth max to go from the initial 750 instantly to 900 (or some extreme value).

To prevent this, we'll have a slow scale out:

global_depth_max += Math.floor((max_p99 - global_depth_max) / 4);

So if we saw a 900 max p99 with a current max of 750 would be:  750 + ((900 - 750) / 4)... 787.

If it's still at 900, the next one will take it up to 815.

If someone sets this global value to 300, for example, within the first second it will go to:
300 + ((725-300)/4) = 406

This will slowly approach reasonable values over the course of several seconds/minutes, staying at this value throughout the rest of the video.

**NOTE 4: Expose this as an option:**
For depth_noramalization_method, the "global" option should be a hard-global, clamping anything over the max.

This new process should be called "adaptive global".  This option should be exposed to the UI as another option in the dropdown list.

### Temporal disparity stabilization

This is mostly solved using DepthAnything V3.  The temporal disparity is quite stable. However, it's not perfectly stable.
I believe this may be contributing to fatigue/vr sickness over time.

To improve this, we'll need scene-aware, EMA-based stabilization of temporal disparity.

Given that this pipeline is heavily frame-based, "temporal" anything can cause an explosion of resources so it might not be
a good fit. Regardless, this is a potential improvement that should be explored.

VR Comfort is at stake, particularly for videos longer than 10 minutes.

**Step 1** Diagnose this using statistics on a frame-to-frame basis, comparing the values across frames to see
how large the shutter is. If it has significant issues, continue

**Step 2** This is where we build a scene-aware temporal disparity stabilization.

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

