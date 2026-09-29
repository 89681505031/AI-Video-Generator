# Official MobileI2V tensor and scheduler contract

Values here are verified against the official HUST MobileI2V source.

## Geometry

- output frames: 17
- VAE latent channels: 128
- spatial downsample: ×32
- temporal latent count: `17 / 8 + 1 = 3`
- Qwen2 max sequence length: 300
- caption channels: 896
- VAE scale factor: 0.41407

The upstream 720p inference uses ceiling behavior for latent height. 720 is
not divisible by 32.

### Base 720p

- pixels: 1280×720
- latent spatial: 40×23
- temporal latent: 3
- sequence positions: `3 × 23 × 40 = 2760`

### Distillation 512

- pixels: 512×512
- latent spatial: 16×16
- temporal latent: 3
- sequence positions: `3 × 16 × 16 = 768`

The distillation branch replaces the main branch's hard-coded 2760 repeat
count with `x.shape[1]`, which is required for 512.

## MobileDiT call

`model(x, timestep, guide_image, y, cond_mask, flow_score, mask, data_info)`

The condition mask marks the first temporal latent slice as the source-image
condition.

Classifier-free guidance:

`noise = uncond + cfg_scale × (text - uncond)`

## FlowMatch Euler

MobileI2V uses Diffusers 0.35.2
`FlowMatchEulerDiscreteScheduler(shift=1.0)`.

Deterministic update:

`prev = sample + (sigma_next - sigma) × model_output`

After each Euler update the first temporal latent is pinned back to
`guide_image`.

For a two-step schedule under this exact scheduler:

- sigmas: `[1.0, 0.001, 0.0]`
- timesteps: `[1000.0, 1.0]`

The Android port and JVM tests live in
`FlowMatchEulerScheduler.java` and `FlowMatchEulerSchedulerTest.java`.

## Distillation status

The public base checkpoint must not be mislabeled as a verified two-step
student. A phone pack declares its actual `sampling_steps`, variant and
source commit in `runtime.json`.
