#!/usr/bin/env python3
"""Generate MobileI2V Lite empty-prompt conditioning and its exact mask.

The upstream MobileI2V path uses AutoTokenizer + Qwen2ForCausalLM.get_decoder()
with max_length=300. Lite mode stores both outputs needed by MobileDiT:
- FP16 text conditioning [1,1,300,896]
- uint8 attention mask [300]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2-0.5B"
MODEL_REVISION = "91d2aff"
MAX_LENGTH = 300
HIDDEN_SIZE = 896
EXPECTED_SHAPE = (1, 1, MAX_LENGTH, HIDDEN_SIZE)
EXPECTED_BYTES = 1 * 1 * MAX_LENGTH * HIDDEN_SIZE * 2
EXPECTED_MASK_BYTES = MAX_LENGTH


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("build/mobile-lite-assets"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
    )
    decoder = model.get_decoder().eval()

    tokens = tokenizer(
        "",
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.inference_mode():
        hidden = decoder(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            use_cache=False,
            return_dict=False,
        )[0]

    if tuple(hidden.shape) != (1, MAX_LENGTH, HIDDEN_SIZE):
        raise RuntimeError(
            f"Unexpected decoder shape {tuple(hidden.shape)}; "
            f"expected {(1, MAX_LENGTH, HIDDEN_SIZE)}"
        )

    condition = hidden.unsqueeze(1).to(torch.float16).cpu().contiguous()
    if tuple(condition.shape) != EXPECTED_SHAPE:
        raise RuntimeError(f"Unexpected final shape: {tuple(condition.shape)}")

    condition_path = args.out_dir / "null_condition.bin"
    condition_array = condition.numpy().astype("<f2", copy=False)
    condition_path.write_bytes(condition_array.tobytes(order="C"))

    if condition_path.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(
            f"Unexpected conditioning size {condition_path.stat().st_size}; "
            f"expected {EXPECTED_BYTES}"
        )

    mask = tokens.attention_mask[0].to(torch.uint8).cpu().contiguous().numpy()
    if tuple(mask.shape) != (MAX_LENGTH,):
        raise RuntimeError(f"Unexpected attention mask shape: {tuple(mask.shape)}")
    if not np.isin(mask, [0, 1]).all():
        raise RuntimeError("Attention mask must contain only 0/1")
    active_tokens = int(mask.sum())
    if active_tokens <= 0:
        raise RuntimeError("Empty prompt unexpectedly has zero active tokens")

    mask_path = args.out_dir / "null_attention_mask.bin"
    mask_path.write_bytes(mask.astype("u1", copy=False).tobytes(order="C"))
    if mask_path.stat().st_size != EXPECTED_MASK_BYTES:
        raise RuntimeError(
            f"Unexpected mask size {mask_path.stat().st_size}; "
            f"expected {EXPECTED_MASK_BYTES}"
        )

    info = {
        "source_model": MODEL_ID,
        "source_revision": MODEL_REVISION,
        "prompt": "",
        "tokenizer_padding_side": "right",
        "max_length": MAX_LENGTH,
        "conditioning_shape": list(EXPECTED_SHAPE),
        "conditioning_dtype": "float16-le",
        "conditioning_size_bytes": condition_path.stat().st_size,
        "conditioning_sha256": sha256(condition_path),
        "attention_mask_shape": [MAX_LENGTH],
        "attention_mask_dtype": "uint8",
        "attention_mask_active_tokens": active_tokens,
        "attention_mask_size_bytes": mask_path.stat().st_size,
        "attention_mask_sha256": sha256(mask_path),
    }
    info_path = args.out_dir / "null_condition.info.json"
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
