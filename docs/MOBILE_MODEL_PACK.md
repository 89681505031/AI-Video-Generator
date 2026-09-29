# Mobile model pack contract

Target pack id: `mobile_i2v_v1`

The Android app accepts a ZIP only when all required files are present and every model/tokenizer file matches the SHA-256 recorded in `manifest.json`.

Required files:

- `vae_encoder.onnx`
- `qwen2_encoder.onnx`
- `mobilei2v_unet.onnx`
- `turbo_vaed.onnx`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `manifest.json`

Example manifest shape:

```json
{
  "pack_id": "mobile_i2v_v1",
  "model_family": "MobileI2V",
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

The ZIP is extracted into a temporary app-private directory. Paths are canonicalized to block ZIP traversal. Only after validation succeeds is the new pack atomically swapped into `files/models/mobile_i2v_v1/`.

## Why no model weights are committed

The repository intentionally does not pretend that desktop Wan weights are Android-ready. MobileI2V conversion, tokenizer fidelity, tensor contracts, quantization and device-specific execution-provider testing must be verified before a downloadable pack is published.

## Runtime plan

1. VAE encode selected image.
2. Tokenize prompt with the real tokenizer files.
3. Run Qwen2 text encoder.
4. Run the distilled MobileI2V denoiser with its real tensor contract.
5. Decode with Turbo-VAED.
6. Stream frames to Android MediaCodec instead of storing a whole uncompressed clip in RAM.
7. Benchmark CPU vs NNAPI for the actual model and keep the faster stable provider.
