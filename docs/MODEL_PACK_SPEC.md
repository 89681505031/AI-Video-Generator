# MobileI2V model pack v1

The Android app deliberately keeps the large neural-network weights outside the APK.

## Pack ID

`mobile_i2v_v1`

## ZIP root

The ZIP must contain these files at its root:

- `vae_encoder.onnx`
- `qwen2_encoder.onnx`
- `mobilei2v_unet.onnx`
- `turbo_vaed.onnx`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `manifest.json`

The app rejects path traversal and installs into a temporary directory first.

## manifest.json

Example:

```json
{
  "pack_id": "mobile_i2v_v1",
  "source": "hustvl/MobileI2V",
  "files": {
    "vae_encoder.onnx": "<64-char sha256>",
    "qwen2_encoder.onnx": "<64-char sha256>",
    "mobilei2v_unet.onnx": "<64-char sha256>",
    "turbo_vaed.onnx": "<64-char sha256>",
    "tokenizer.json": "<64-char sha256>",
    "tokenizer_config.json": "<64-char sha256>",
    "special_tokens_map.json": "<64-char sha256>"
  }
}
```

Every declared SHA-256 is checked before the pack is activated.

## Runtime strategy

The app opens ONNX sessions stage-by-stage and closes each stage after use.
Decoded RGBA frames are intended to be handed directly to `H264Mp4Encoder`, one frame at a time, instead of keeping the complete clip in Java memory.

## Important

The official raw MobileI2V checkpoint is not itself this pack. It must first be converted and validated against the exact tensor inputs/outputs expected by the Android pipeline. The app must not rename arbitrary files to satisfy this contract.
