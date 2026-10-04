# Tutorial: Convert a 2D Video to Stereo 3D

This project converts an ordinary 2D video into stereoscopic 3D video. It estimates the scene's depth from the original video, then uses that estimate to make the two views look slightly different.

## How the conversion works

A 2D video records the scene from one viewpoint. To create a 3D effect, the workflow first estimates which objects are closer to the camera and which are farther away. It does this for each frame, then refines the estimate down to individual pixels: pixels belonging to a nearby object should be closer than pixels in the distant background. For each frame of a movie, the depth estimator will create a copy of that frame (essentially a picture) and then map out the depth of each pixel.  This becomes, effectively, a second image with black-and-white pixels. Brighter pixels are close; darker pixels are farther. This is called a depth map.

The depth map is not a second camera view. Instead, it is used as a guide for shifting image pixels horizontally. The renderer shifts nearer and farther parts by different amounts to create a left-eye view and a right-eye view. This is why the depth estimator is sometimes called a *monocular depth model*: “monocular” means it estimates depth from a single view, here the original 2D frame. The two generated views are then joined together in a single video (either placing the side-by-side or one on top and one on bottom). (More information on depth-map generation.)[https://depth-anything.github.io/]

Depth boundaries get complicated. In order to create the 3d effect, the renderer has to shift the pixels slightly.  This can "squeeze" the 2d image slightly, revealing pixels that were behind the original 2d image at the edges.  However, the original 2D movie doesn't have information about what was behind that original person/object.  This leaves a small gap that we have to fill with background information.  To fill those gaps, we have to pull pixels from the surrounding background image (kind of like copy-pasting in photoshop).  In the worst case, we won't fill from the surrounding background and it will look like a 2d-cutout of the person peaking out from behind the 3d person.

Furthermore, during this depth estimation, determing what counts as an "edge" isn't always straight forward. Simpler processes will leave a "depth halo" around the foreground object.  This makes it look like there's a ring of the background image around a person/object.  That ring will be at the same depth as the person.  To illustrate, imagine a person standing on a beach with the waves in the background. But there's a ring around them that looks like a cutout where the waves are projected onto that ring. Edge-refinement setting can improve these depth boundaries before rendering, preventing this ring.

To summarize, the overall process is:
* The 2d video is broken into frames and each frame is converted to a depth map
* The depth maps are used to shift pixels and create a 3d stereoscopic effect (resulting in a left-eye view and right-eye view)
* The two views are stitched back together into a single video
* The audio is copied from the source onto the stereoscopic video

# Settings

This walks through the settings for this project.  (Many important settings have been intentionally defaulted to simplify use.)

## Depth settings

### Model and inference resolution

`depth_model` selects `da3_small`, `da3_base`, or `da3_large`. Larger models can provide better estimates but generally need more compute and memory. Start with `da3_large` and decrease if memory or speed is highly limited.

`depth_use_source_resolution` is true/false field. In that mode, depth inference uses each frame at its original resolution. This creates a 1-to-1 depth map in the same size as the original movie. If you turn this off, each frame will be resized (improving speed and memory usage).  The new size is determined by the `depth_inference_resolution` parameter.  This will set the longest side of the inference image. A lower value (for example, 512) will reduce memory use and improves speed, at the cost of fine detail. This will only impact the depth estimation and the final video output is still rendered at the same resolution as the original video.

## Preview run

Enable `preview_run` to process every 30th source frame. This is useful for checking your settings before rendering every frame. The preview video has approximately one-thirtieth the source frame rate, so it will feel like a storyboard, not a smooth video. However, you can also get a rough speed estimate if you multiply the generation time by about 30 to estimate a full-frame run.

## Chunk size

The longer the video is that you are processing, the more memory it uses in the process. To combat this, this repo will break the video into multiple chunks and process each chunk independently.  Therefore, rather than loading 76,800 frames for 2 hour movie, this workflow loads 10 frames, process those 10, and then writes 10 frames to the disk.

`chunk_size` controls how many frames are decoded processed for each batch. The default is 4. A 16GB Vram card may run 12. Larger chunks may improve throughput but use more RAM and VRAM; smaller chunks can help avoid memory pressure, with more per-chunk overhead. If inference runs out of memory, reduce this value first.  This is your biggest lever to control memory usage.

## Tune the stereo effect

`disparity_percent` controls the maximum pixel shift relative to the source frame width. The default is about 1.15%, which corresponds to roughly 22 pixels at 1920 pixels wide for each view's shift at the strongest depth; the two views shift in opposite directions. Higher values make the 3D effect stronger and can also make mismatched edges or eye strain more noticeable. Start with the default and adjust in small steps for your display and content. (Setting this to 0 gives a flat output.)

The `depth_power` control changes how depth affects the shift of each pixel. Increasing the value will make things "more 3d" (increasing the perceived depth). Increase it slowly if the scene looks too flat, or decrease it if the depth feels abnormal. 

`invert_depth` reverses the depth direction. Toggle it if near objects appear behind the scene or the stereo effect otherwise looks inside-out.

## Depth normalization

This whole process uses math.  In order to make the depth map usable, the depth values must be scaled into a 0-to-1 range before they can be used to shift pixels This repo offers two approaches to scaling this depth information:

- `simple` normalizes each frame independently using that frame's minimum and maximum. It needs no tuning, but the same object can receive different apparent depth from frame to frame, which may cause the 3D strength to pulse.  Some scenes may be perfect and others may feel like the 3d rendering is strange.
- `global` normalizes each frame based on a shared `global_depth_max`. This gives steadier depth across frames and scenes. However, the `global_depth_max` needs to suit the model's values and footage. The default is `750`. If the calculated depth exceeds the maximum, this will scale that individual frame similar to the `simple` method as a fallback. Also, this will log a warning in the console.  If watch the console and see this happen frequently, raise `global_depth_max`; if most of the scene appears too shallow, lower it. (Preview mode is helpful if you plan to fine-tune this.)

## Edge refinement

`depth_edge_refine_method` selects how the edges of objects are cleaned up:

- `none` skips refinement. It is the fastest choice.
- `simple` applies a local RGB-guided smoothing to reduce noisy depth while following image edges. It adds a small amount of processing time.
- `fgs` uses Fast Global Smoothing for high quality edge-aware refinement. It's excellent at removing the depth halo, but it can add 50% more processing time.

If the edges are particularly distracting to you, stick with `fgs`. If not, `simple` or `none` are perfectly fine choices.  (Some videos may not need this.)

## Audio and depth export

`audio_mode` is pretty basic. `copy` will duplicate the source audio into the final file; `none` creates a video-only output.

`output_depth_video` will save the generated depth maps as a lossless grayscale MKV alongside the stereo video. These files can get HUGE. To help protect your hard drive, this is limited to source clips shorter than 60 seconds or jobs where `preview_run` is enabled. 

# Recommended Settings

## Limited Computer

The original base project was built for this scenario. I would start with these settings

* depth_model:  da3_large  ("large" is still pretty small.  Try medium if you must.)
* depth_use_source_resolution: true (but false for highly limited systems)
* depth_inference_resolution: 1024 (only matters if "source resolution" is false, 512 may improve performance with lower quality)
* chunk_size:  4 (Lower this if you max out your vram. Raise this if you have more vram)
* depth_edge_refine_method: none (simple is a little slower, fgs is higher quality and a lot slower)

## Higher Quality

This what I use with my RTX 5060ti (16GB vram)

* depth_model:  da3_large
* depth_use_source_resolution: true
* depth_inference_resolution: 1024 (only matters if "source resolution" is false)
* chunk_size:  12
* depth_edge_refine_method: fgs (fgs is much slower but much reduced artifacts)

## Common settings

These settings are for the 3d rendering, so they are common across all hardware:

* disparity_percent:  1.15 (this can be used if the video becomes uncomfortable to watch)
* depth_power:  0.30  (0.35 if you use `fgs`.  Increase if the video feels flat. Decrease if it's strangely "too 3d")
* invert_depth:  true
* audio_mode:  copy
* depth_normalization_method: global ("simple" may be ok for single-shot videos - no processing advantage either way)
* global_depth_max: 750 (watch the console for warnings. If the video feels flat, lower this until you get a lot of warnings and then raise it to stop the warnings)
