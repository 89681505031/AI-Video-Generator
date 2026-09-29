# MobileI2V model pack v2

Pack ID: `mobile_i2v_v1`

The Android app keeps large neural-network weights outside the APK and loads ONNX sessions one stage at a time.

## Lite Image → Video core

Required at ZIP root:

- `vae_encoder.onnx`
- `mobilei2v_transformer.onnx`
- `video_decoder.onnx`
- `null_condition.bin`
- `runtime.json`
- `manifest.json`

For the first phone build, Qwen2 is optional. `null_condition.bin` stores the FP16 empty-prompt conditioning tensor with shape `[1,1,300,896]`. This lets Lite I2V skip the Qwen2 model entirely and reduce storage and peak RAM.

## Optional text extension

When `runtime.json` has `"text_conditioning": true`, also include:

- `qwen2_encoder.onnx`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`

## Official runtime contract

The upstream MobileI2V configuration/code uses:

- model: `Mobiledit_300M_P1_D16` (MobileDiT, not a UNet)
- 17 video frames
- 128 latent channels
- video VAE spatial downsample ×32
- temporal latent count `17 // 8 + 1 = 3`
- Qwen2-0.5B conditioning
- max text length 300
- caption channels 896

At 512×512 the noisy video latent is `[B,128,3,16,16]`.

The denoiser forward contract is conceptually:

```
model(
  x,
  timestep,
  guide_image,
  y,
  cond_mask,
  flow_score,
  mask,
  data_info
) -> noise_prediction
```

## runtime.json

Example BASE Lite profile:

```json
{
  "variant": "base",
  "width": 512,
  "height": 512,
  "frames": 17,
  "latent_channels": 128,
  "vae_downsample_rate": 32,
  "temporal_latents": 3,
  "sampling_steps": 30,
  "text_conditioning": false,
  "text_max_length": 300,
  "caption_channels": 896,
  "source_commit": "8d0a253c766b05a43ba408baf5e8f800a36be8b4"
}
```

`variant` is either `base` or `distilled`. The app refuses a pack that labels itself distilled while declaring more than four sampling steps.

## BASE vs DISTILLED

The official Hugging Face model repository publishes the base `hybrid_371.pth` checkpoint (~1.07 GB). The upstream distillation branch publishes training code/configuration, but its instructions do not currently point to a separately named pretrained distilled-student checkpoint.

The Android app therefore keeps BASE and DISTILLED distinct instead of treating the public base checkpoint as the two-step model.

## Integrity

`manifest.json` contains SHA-256 for every runtime file. Installation happens in a temporary app-private directory. ZIP traversal is blocked, metadata is validated, every declared SHA-256 is checked, then the pack is atomically activated.

Create a pack after conversion with:

```bash
python tools/package_mobile_model.py /path/to/exported_model --out mobile_i2v_v1.zip
```

## RAM strategy

- open one ONNX stage at a time via `StagedOrtRunner`
- unload the stage before opening the next large graph
- feed decoded frames directly to `H264Mp4Encoder`
- avoid keeping the whole uncompressed 17-frame clip in Java memory
