# Android source-image and latent preparation

The phone pipeline now mirrors the upstream MobileI2V preprocessing before the
neural-network calls.

## Source image

`ImageTensorPreprocessor`:

1. decodes the selected Android document URI
2. downsamples during decode when possible to control memory
3. resizes to the runtime profile exactly
4. converts RGB channels from `[0,255]` to `[-1,1]`
5. writes contiguous channel-first `NCTHW` with shape
   `[1,3,1,H,W]`

The upstream Python path uses `ToTensor()`, resize and
`Normalize(mean=0.5,std=0.5)`, which is the same channel transform.

## VAE posterior

The exported encoder is expected to expose posterior moments rather than hide
random sampling inside ONNX.

`VaePosteriorSampler` reproduces:

`mean + exp(0.5 * clamp(logvar,-30,20)) * N(0,1)`

and then multiplies by the MobileI2V VAE scale factor `0.41407`.

## Initial video latent

`LatentInitializer` creates Gaussian noise with the runtime latent geometry.
The first temporal latent slice is then replaced with the guide-image latent,
matching the official sampler before denoising begins.
