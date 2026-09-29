# Android ONNX core — format v4

The Android runtime mirrors the parts of the official MobileI2V graph that
actually affect output.

## vae_encoder.onnx

Input:

- `image` — FP32 `[1,3,1,H,W]`, RGB normalized to `[-1,1]`

Output:

- `posterior_moments` — `[1,256,1,ceil(H/32),ceil(W/32)]`

Android performs posterior sampling and applies scale factor `0.41407`.

## mobilei2v_transformer.onnx

Inputs:

- `x`
- `timestep`
- `cond_mask`
- `flow_score`

Output:

- `noise`

The official source currently computes caption tensors but never consumes them:
the cross-attention call inside `SanaBlock_cross` is commented out. The
`guide_image` forward argument is also not read by MobileDiT. ONNX therefore
prunes those values and Android v4 does not pretend they are active inputs.

The source image still affects every denoising step because Android pins the
first temporal latent slice to the VAE-encoded guide image before sampling and
again after each FlowEuler update.

## Memory policy

Sessions are opened stage-by-stage:

1. VAE encoder
2. MobileDiT denoiser
3. video decoder

This keeps peak RAM below the cost of holding all large graphs at once.
