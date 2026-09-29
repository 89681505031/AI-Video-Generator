# MobileI2V model pack

Pack ID: `mobile_i2v_v1`
Format version: **3**

The Android app keeps large neural-network weights outside the APK. A pack is
installed only after runtime metadata and every SHA-256 pass validation.

## Core files

- `vae_encoder.onnx`
- `mobilei2v_transformer.onnx`
- `video_decoder.onnx`
- `null_condition.bin` — FP16 `[1,1,300,896]`
- `null_attention_mask.bin` — uint8 `[300]`
- `runtime.json`
- `manifest.json`

## Optional text extension

When `runtime.json` has `"text_conditioning": true`, also include:

- `qwen2_encoder.onnx`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

Lite mode uses the precomputed empty-prompt conditioning and mask, so Qwen2 is
not loaded on the phone.

## Runtime contract

Verified upstream values:

- 17 output frames
- 128 latent channels
- spatial VAE downsample ×32
- 3 temporal latent slices
- text max length 300
- caption channels 896
- FlowMatchEuler shift 3.0
- VAE scale factor 0.41407

720p uses latent `[1,128,3,23,40]`; 512×512 uses
`[1,128,3,16,16]`.

## manifest.json

```json
{
  "pack_id": "mobile_i2v_v1",
  "format_version": 3,
  "files": {
    "vae_encoder.onnx": "<sha256>",
    "mobilei2v_transformer.onnx": "<sha256>",
    "video_decoder.onnx": "<sha256>",
    "null_condition.bin": "<sha256>",
    "null_attention_mask.bin": "<sha256>",
    "runtime.json": "<sha256>"
  }
}
```

Every core file except `manifest.json` itself must appear in `files`.

## Packaging

After exporting the ONNX graphs and creating `runtime.json`:

`python tools/package_mobile_model.py /path/to/model_dir --out mobile_i2v_v1.zip`

The packager writes format version 3 and all SHA-256 values.
