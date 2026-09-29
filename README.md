# AI Video Generator — Android mobile branch

Branch: `mobile-android-v1`

## What works now

- Native Android app (Java 17)
- RAM / SoC / ABI detection
- ONNX Runtime Android 1.30.0
- Image picker using Android's document provider
- External MobileI2V model-pack installer
- ZIP path-traversal protection
- SHA-256 verification for runtime files
- BASE vs DISTILLED runtime metadata validation
- Atomic model-pack replacement
- Stage-by-stage ONNX session loader to reduce peak RAM
- Deep ONNX check that reports real model input/output names
- Streaming H.264/MP4 encoder based on Android MediaCodec
- Reproducible GitHub Actions debug APK build

No fake AI video is returned while the real tensor pipeline is incomplete.

## Target model

The mobile path is based on the official HUST MobileI2V architecture:

- ~270M MobileDiT denoiser
- Image → Video
- 17-frame clips
- 128-channel video latent
- ×32 video VAE
- optional Qwen2-0.5B text conditioning
- distilled low-step mode when a genuine distilled checkpoint is available

The public base checkpoint is kept outside the APK and must be converted/validated before installation.

## Lite mode for phones

The first on-device target can omit Qwen2. It uses a precomputed FP16 empty-prompt conditioning tensor, which cuts storage and peak RAM. Text-conditioned I2V is an optional larger model pack.

## Current pack ID

`mobile_i2v_v1`

See `docs/MODEL_PACK_SPEC.md`.

## Next engineering step

Connect exact tensor I/O for:

1. source image preprocessing
2. video-VAE image encode
3. official MobileDiT sampling
4. video decode
5. frame-by-frame `H264Mp4Encoder`

The desktop Wan2.1 setup remains a separate high-quality path and is not copied into the APK.
