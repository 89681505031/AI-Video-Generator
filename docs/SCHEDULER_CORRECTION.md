# Scheduler correction

The Android scheduler uses the value from the official MobileI2V configuration:

`scheduler.flow_shift: 1.0`

This value is passed by upstream inference into `FlowEuler`, which constructs
Diffusers `FlowMatchEulerDiscreteScheduler(shift=flow_shift)`.

For a two-step schedule with shift 1.0, the deterministic sigma grid is:

- sigma 0: 1.0
- sigma 1: 0.001
- terminal sigma: 0.0
- timesteps: 1000.0 and 1.0

Earlier development notes used shift 3.0; that value has been removed from the
Android runtime and tests before publishing a model pack.
