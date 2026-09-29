# AI Video Generator

Android-first local AI video generation experiment.

## Current stage: Mobile v0.1

This branch contains the first real Android runtime shell:

- Java 17 Android app
- automatic RAM / SoC / ABI inspection
- adaptive mobile profile selection
- ONNX Runtime Android 1.30.0
- model-pack validation
- GitHub Actions APK build
- no fake AI generation

The APK does **not** bundle large video-model weights yet.

Expected model-pack directory inside the app:

`files/models/mobile_video_v1/`

Expected files:

- `text_encoder.onnx`
- `video_model.onnx`
- `vae_decoder.onnx`
- `tokenizer.json`
- `manifest.json`

## Next implementation stage

1. Select/convert a genuinely mobile video model.
2. Add staged model loading to reduce peak RAM.
3. Add text encoder + denoiser + VAE pipeline.
4. Stream decoded frames instead of keeping the whole clip in RAM.
5. Encode frames to MP4 with Android MediaCodec.
6. Add Image-to-Video after Text-to-Video works.

The desktop Wan2.1 1.3B setup remains a separate high-quality path; it is not copied directly into the APK.
