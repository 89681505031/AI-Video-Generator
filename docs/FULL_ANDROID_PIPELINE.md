# Full Android generation path

The Android button now executes the full phone pipeline when a valid 512-class
format-v4 model pack is installed:

1. Android document URI → resized RGB NCTHW `[-1,1]`
2. `vae_encoder.onnx` → posterior moments
3. seeded posterior sampling → guide latent
4. seeded Gaussian video latent + pinned guide slice
5. `mobilei2v_transformer.onnx` + FlowMatchEuler
6. `video_decoder.onnx`
7. decoder native `FloatBuffer` → one RGBA frame at a time
8. `MediaCodec` H.264 + `MediaMuxer` MP4

The decoder output is not copied into a second full Java float array. Frames
are read directly from the ORT native output buffer and immediately submitted
to the MP4 encoder.

## First mobile limit

Format v4 currently requires decoder FP32 output. A 1280×720×17 RGB FP32
decoder output alone is roughly 188 MB, before model memory and codec buffers.
For the first phone-ready path the app therefore rejects profiles above
512×512 at generation time.

The 720p export path remains useful for desktop verification. Phone 720p should
be enabled only after decoder chunking/Turbo-VAED or another proven
memory-saving decode path is integrated.
