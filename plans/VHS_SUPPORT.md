# VHS Video Combine Support Plan

## Goal

Allow a rendered stereo video to flow into Video Helper Suite's `VHS_VideoCombine` node, with VHS owning the output filename, container/codec settings, audio mux, and ComfyUI output preview. This would replace most of `StereoVideoMuxOutput` for users who want VHS's export controls.

## Current project boundary

The current graph is `StereoVideoSource` -> `StereoVideoConvert` -> `StereoVideoMuxOutput`.

- `StereoVideoSource` builds a `StereoVideoJob`, including source audio path, selected frame rate/count, and preview stride.
- `StereoVideoConvert` decodes in chunks, runs depth estimation and stereo rendering, and streams the rendered frames to `StreamingVideoEncoder`. It returns a `STEREO_VIDEO_RENDER` handle pointing to the temporary encoded video; it does not return frame tensors.
- `StereoVideoMuxOutput` picks a filename/container, uses `finalize_mux` to optionally copy source audio, adds stereo-layout metadata, and cleans up the temporary directory.

VHS's `VHS_VideoCombine` has a different contract. Its primary `images` input accepts ComfyUI image/latent data, and its optional `audio` input accepts an `AUDIO` object (a waveform tensor plus sample rate). It expects frame rate, format, filename prefix, and related export options. The VHS implementation also makes an intermediate video before muxing audio.

## Main implementation change

To connect to VHS in the normal ComfyUI graph, `StereoVideoConvert` must expose rendered frames as an `IMAGE` output and source audio as an `AUDIO` output, alongside its summary. The frames should be RGB float tensors in ComfyUI's expected `[frames, height, width, channels]` layout, with values in `[0, 1]`. The audio output should follow the ComfyUI `AUDIO` shape: `{"waveform": tensor, "sample_rate": int}`. The example workflow would connect these two outputs to `VHS_VideoCombine.images` and `.audio`, and connect the VHS filename/format controls in place of the current mux node controls.

This requires changing the convert path: it currently writes rendered chunks directly to an encoder and discards them. It would instead have to retain rendered frames until the node returns. A conventional batched `IMAGE` tensor is straightforward and works with VHS's current implementation, but it can consume substantial host RAM for long or high-resolution clips. A lazy disk-backed image sequence might reduce memory, but VHS indexes the first frame and then iterates all frames; ComfyUI's `IMAGE` conventions and other consumers are tensor-oriented. That approach needs a focused compatibility prototype against the installed VHS version before it can be treated as supported.

Audio also becomes a resource consideration. The current implementation only keeps an audio file path and lets FFmpeg copy the source audio during finalization. VHS expects decoded waveform samples in memory and its combine implementation converts the waveform to bytes before muxing. Supporting VHS therefore needs an audio extraction step that respects the selected clip duration and handles files with no audio stream. It should not decode the full source audio when `audio_mode` is `none`.

## Recommended graph and node responsibilities

1. Keep `StereoVideoSource` as the job/configuration node. Remove `audio_mode` if VHS audio is always connected when desired, or retain it to determine whether the convert node produces an audio output.
2. Update `StereoVideoConvert` to return `IMAGE`, optional `AUDIO`, and a human-readable summary. Its render loop should collect CPU image batches and concatenate them once at the end, rather than keeping GPU tensors alive. This is the first implementation to validate because it uses standard ComfyUI data and the existing VHS API.
3. Make `VHS_VideoCombine` the standard output node in the example workflow. Set its `frame_rate` from the job's selected FPS, and let VHS own filename prefix, output format/codec, and audio muxing.
4. Remove `StereoVideoMuxOutput`, `StereoVideoRender`, `finalize_mux`, and output-prefix code only after parity is confirmed for audio/no-audio, preview stride, SBS/top-bottom layout, and supported output formats.

If long-video low-memory operation must remain a first-class path, do not remove the current streaming route. Keep a reduced, clearly named streaming output (or a source option selecting the export route) for users who need chunked rendering and direct FFmpeg muxing. VHS's current interface is not a streaming frame sink, so using `VHS_VideoCombine` and retaining the current bounded-memory behavior cannot be achieved by simply replacing the mux node.

## Audio compatibility details

- Match the selected render duration and FPS so audio and output frames stay in sync, including `preview_run` where output FPS is reduced.
- For source audio pass-through, extract the source's first audio stream into VHS's expected waveform object, starting at the same clip offset and limiting it to the selected duration.
- Handle silent sources and `audio_mode = none` by leaving the optional VHS audio input disconnected/empty.
- Confirm VHS's mux behavior for formats without audio support and for a waveform shorter or longer than the rendered frames. VHS's combine path can pad or shorten audio based on its format settings, which differs from the current `-shortest`/copy-audio behavior.

## Workflow and migration

- Replace the `StereoVideoMuxOutput` node in `example_workflows/stereo_video.json` with VHS Video Combine and wire the convert node's image/audio outputs.
- Remove the current `output_format` choice only if VHS fully owns format selection; retain `filename_prefix` as a VHS widget.
- Existing workflows using `StereoVideoMuxOutput` need either a migration note or a compatibility period. If the streaming output remains, document it as the low-memory alternative.
- VHS is an optional dependency from this project's perspective. Avoid importing VHS Python modules from the plugin at import time; standard `IMAGE`/`AUDIO` sockets permit a ComfyUI graph connection without making the custom node fail to load when VHS is absent.

## Work sequence

1. Verify the installed ComfyUI version's `AUDIO` representation and the installed VHS `VHS_VideoCombine` behavior, including accepted image tensor shape, frame rate, and audio timing.
2. Prototype a short clip through `StereoVideoConvert` -> VHS Video Combine; confirm image color/range/layout and audio sync for both `sbs` and `top_bottom`.
3. Implement image and audio outputs with correct empty/no-audio handling and preserve the source/job metadata needed for timing.
4. Update the example workflow and README, then decide whether the existing streaming mux node is removed or retained as the low-memory path based on measured memory use and feature parity.
5. Verify export formats, stereo metadata expectations, temporary-file cleanup, and failure cleanup. VHS owns its own encoding and muxing flags, so the current project's stereo metadata insertion may need a small post-processing node or may be dropped if VHS supports equivalent format options.

## Open design decision

The major product choice is whether VHS compatibility becomes the only path (simpler graph, standard VHS controls, higher memory use proportional to rendered frame count) or an additional path alongside the existing streaming output (more node/UI surface, preserves long-video bounded-memory behavior). The project name and current chunked encoder favor keeping the streaming route available unless typical source lengths and resolutions show that collecting all rendered frames is acceptable.