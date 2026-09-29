# MobileI2V Lite assets

The Lite Android profile avoids loading Qwen2-0.5B on the phone.

`tools/generate_null_condition.py` follows the upstream MobileI2V empty-prompt path and writes:

- `null_condition.bin` — raw little-endian FP16 tensor, shape `[1,1,300,896]`
- `null_condition.info.json` — source revision, shape, byte count and SHA-256

Expected binary size: **537600 bytes**.

The GitHub Actions workflow `Mobile Lite Model Assets` is manual because downloading Qwen2 and computing this deterministic asset does not need to happen for every Android build.

The script pins the base model `Qwen/Qwen2-0.5B` at revision `91d2aff`; it does not use the Instruct variant.
