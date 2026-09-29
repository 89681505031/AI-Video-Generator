# Cloud build for the phone model pack

Workflow: `Build MobileI2V 512 Model Pack`

This workflow is intentionally manual. It prepares model files once on GitHub;
video generation itself remains on the Android phone.

The workflow downloads:

- `hustvl/MobileI2V/hybrid_371.pth`
- `hustvl/Turbo-VAED/Turbo-VAED-LTX.pth`
- only the `vae/` subfolder from `Lightricks/LTX-Video`

It clones the official MobileI2V **distillation** source branch because that
branch removes the hard-coded 2760-position assumption and can export 512×512.

The current public MobileI2V checkpoint is still the base model, so the first
pack is explicitly named:

`mobile_i2v_v1_512_base30_turbo.zip`

It uses 30 FlowEuler steps. The workflow does not label it as a two-step
distilled student.

When an official/validated distilled student checkpoint is available, the same
workflow/exporter can be switched to `variant=distilled` and 2–4 steps.
