# Export MobileI2V to Android ONNX v4

The exporter lives at `tools/export_mobilei2v_onnx.py`.

## Inputs

You need:

1. a checkout of the official `hustvl/MobileI2V` source
2. a MobileI2V checkpoint such as the public `hybrid_371.pth`
3. the LTX video-VAE directory expected by the upstream config

The public `hybrid_371.pth` is treated as **base**, not as a verified
two-step distilled student.

## 720p base example

```bash
python tools/export_mobilei2v_onnx.py \
  --mobilei2v-root /path/to/MobileI2V \
  --checkpoint /path/to/hybrid_371.pth \
  --vae /path/to/video-vae \
  --out-dir build/mobilei2v-720 \
  --profile 720 \
  --variant base \
  --sampling-steps 30
```

## 512 distilled example

Use this only with a real distilled student checkpoint and the official
distillation source tree:

```bash
python tools/export_mobilei2v_onnx.py \
  --mobilei2v-root /path/to/MobileI2V \
  --checkpoint /path/to/distilled_student.pth \
  --vae /path/to/video-vae \
  --out-dir build/mobilei2v-512 \
  --profile 512 \
  --variant distilled \
  --sampling-steps 2
```

The exporter refuses 512 export when the checked-out `mobiledit.py` still
hard-codes 2760 positions.

## Output

- `vae_encoder.onnx`
- `mobilei2v_transformer.onnx`
- `video_decoder.onnx`
- `runtime.json`

The transformer wrapper deliberately exports only:

- `x`
- `timestep`
- `cond_mask`
- `flow_score`

because current upstream MobileDiT has cross-attention disabled.

## Verify

Install `onnx` and run:

```bash
python tools/verify_mobile_onnx.py build/mobilei2v-720
```

Then package:

```bash
python tools/package_mobile_model.py build/mobilei2v-720 \
  --out mobile_i2v_v1.zip
```

## Important mobile note

This first exporter preserves the official PyTorch graph. It does **not** yet
perform mobile quantization or replace the LTX decoder with Turbo-VAED.
Those optimizations are separate stages and must be benchmarked against the
same exported contract rather than assumed to be lossless.


## Turbo-VAED-LTX decoder

The official Turbo-VAED LTX configuration uses 128 latent channels, matching
MobileI2V/LTX latents. To export a mobile-oriented decoder while keeping the
original LTX encoder:

```bash
python tools/export_mobilei2v_onnx.py \
  --mobilei2v-root /path/to/MobileI2V \
  --checkpoint /path/to/mobilei2v_checkpoint.pth \
  --vae /path/to/video-vae \
  --turbo-vaed-root /path/to/Turbo-VAED \
  --turbo-config /path/to/Turbo-VAED/configs/Turbo-VAED-LTX.json \
  --turbo-checkpoint /path/to/turbo_ltx_decoder.ckpt \
  --out-dir build/mobilei2v-512 \
  --profile 512 \
  --variant distilled \
  --sampling-steps 2
```

All three Turbo arguments are required together. The exporter loads the
checkpoint into `AutoencoderKLTurboVAED.decoder`, exactly matching the
official Turbo-VAED validation path, while the VAE encoder remains LTX.

The resulting `runtime.json` records `"decoder": "turbo-vaed-ltx"` and the
Turbo decoder checkpoint SHA-256.
