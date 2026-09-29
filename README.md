# AI Video Generator — Android mobile branch

Branch: `mobile-android-v1`

## What works now

- Native Android app (Java 17)
- RAM / SoC / ABI detection
- ONNX Runtime Android 1.30.0
- Image picker using Android's document provider
- External MobileI2V model-pack installer
- ZIP path-traversal protection
- SHA-256 verification for every model/tokenizer file
- Atomic model-pack replacement
- Stage-by-stage ONNX session loader to reduce peak RAM
- Streaming H.264/MP4 encoder based on Android MediaCodec
- Reproducible GitHub Actions debug APK build

No fake AI video is returned while the real tensor pipeline is incomplete.

## Target model

The mobile path is based on the official HUST MobileI2V architecture:

- lightweight ~270M denoiser
- image-to-video
- 17-frame clips
- distilled low-step inference
- Turbo-VAED decoder path

The official raw checkpoint is large and stays outside the APK. The Android app expects a converted, verified pack described in `docs/MODEL_PACK_SPEC.md`.

## Current pack ID

`mobile_i2v_v1`

Required files:

- `vae_encoder.onnx`
- `qwen2_encoder.onnx`
- `mobilei2v_unet.onnx`
- `turbo_vaed.onnx`
- tokenizer files
- `manifest.json`

## Next engineering step

Implement the exact MobileI2V tensor contract from the official inference code:

1. preprocess source image
2. real Qwen2 tokenization + text embeddings
3. image latent conditioning
4. distilled MobileI2V denoising
5. Turbo-VAED frame decode
6. feed frames directly into `H264Mp4Encoder`

The desktop Wan2.1 setup remains a separate high-quality path and is not copied into the APK.
