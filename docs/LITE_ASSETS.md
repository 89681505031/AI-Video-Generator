# MobileI2V Lite assets

Lite mode avoids loading Qwen2-0.5B on the phone while preserving the
conditioning tensors used by upstream MobileI2V.

`tools/generate_null_condition.py` writes:

- `null_condition.bin` — little-endian FP16, shape `[1,1,300,896]`,
  exactly **537600 bytes**
- `null_attention_mask.bin` — uint8 0/1 mask, shape `[300]`,
  exactly **300 bytes**
- `null_condition.info.json` — pinned source revision, active-token count,
  sizes and SHA-256 values

Both neural conditioning and attention mask come from the same tokenization of
the empty prompt. Keeping the mask matters because MobileDiT removes padded
text tokens before cross-attention.

The workflow `Mobile Lite Model Assets` regenerates and verifies these files
when the generator changes.

Pinned source:

- model: `Qwen/Qwen2-0.5B`
- revision: `91d2aff`
- max length: 300
- hidden size: 896
- tokenizer padding: right
