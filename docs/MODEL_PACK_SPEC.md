# MobileI2V model pack

Pack ID: `mobile_i2v_v1`
Format version: **4**

The Android app keeps large neural-network weights outside the APK. A pack is
installed only after runtime metadata and every SHA-256 pass validation.

## Core files

- `vae_encoder.onnx`
- `mobilei2v_transformer.onnx`
- `video_decoder.onnx`
- `runtime.json`
- `manifest.json`

## Why Qwen/text files are not in v4

The current official MobileI2V `SanaBlock_cross` source has its
`cross_attn` call commented out. The denoiser therefore does not depend on
caption embeddings, attention mask or guide-image argument. ONNX export prunes
those unused inputs.

Format v4 is intentionally honest about this: `runtime.json` must set
`"text_conditioning": false`.

The UI keeps prompt disabled until a checkpoint/graph with working
cross-attention is introduced and validated.

## Denoiser ONNX contract

Inputs:

- `x` — `[1,128,3,h,w]`
- `timestep` — `[1]`
- `cond_mask` — `[1,3*h*w]`
- `flow_score` — `[1]`

Output:

- `noise` — `[1,128,3,h,w]`

The source-image condition is preserved by pinning temporal latent index zero
after every Euler step.

## Runtime geometry

- 17 output frames
- 128 latent channels
- spatial VAE downsample ×32
- 3 temporal latent slices
- FlowMatchEuler shift 3.0
- VAE scale factor 0.41407

720p uses latent `[1,128,3,23,40]`; 512×512 uses
`[1,128,3,16,16]`.

## manifest.json

```json
{
  "pack_id": "mobile_i2v_v1",
  "format_version": 4,
  "files": {
    "vae_encoder.onnx": "<sha256>",
    "mobilei2v_transformer.onnx": "<sha256>",
    "video_decoder.onnx": "<sha256>",
    "runtime.json": "<sha256>"
  }
}
```

## Packaging

`python tools/package_mobile_model.py /path/to/model_dir --out mobile_i2v_v1.zip`
