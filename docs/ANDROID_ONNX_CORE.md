# Android ONNX Lite core

The Android runtime now has a strict ONNX contract for the first two real
neural stages.

## vae_encoder.onnx

Inputs:

- `image` — FP32 `[1,3,1,H,W]`, RGB normalized to `[-1,1]`

Outputs:

- `posterior_moments` — FP16/FP32
  `[1,256,1,ceil(H/32),ceil(W/32)]`

The phone performs the Gaussian posterior sample itself and multiplies by
`0.41407`.

## mobilei2v_transformer.onnx

Inputs:

- `x` — FP32 `[1,128,3,h,w]`
- `timestep` — FP32 `[1]`
- `y` — FP32 `[1,1,300,896]`
- `cond_mask` — FP32 `[1,3*h*w]`
- `flow_score` — FP32 `[1]`
- `attention_mask` — INT64 `[1,300]`

Output:

- `noise` — FP16/FP32 `[1,128,3,h,w]`

The export wrapper is responsible for inserting the extra internal caption
dimension expected by upstream MobileDiT.

## Lite optimization

For the Lite pack, conditional and unconditional text embeddings are identical.
Therefore:

`uncond + cfg * (condition - uncond) = uncond`

The phone runs one denoiser inference per scheduler step instead of two or a
batch of two. This reduces peak memory and compute without changing the Lite
result.

## Memory policy

ONNX sessions remain staged:

1. open VAE encoder
2. produce guide latent and close VAE encoder
3. open MobileDiT once for all scheduler steps
4. close MobileDiT
5. open the decoder only for final video decode

The decoder contract is intentionally kept separate until its mobile
frame/chunk output path is validated, because a full uncompressed 720p
17-frame FP32 tensor is too large to treat casually on a phone.
