#!/usr/bin/env python3
"""Generate MobileI2V Lite empty-prompt conditioning from Qwen2-0.5B.

This follows the upstream MobileI2V text-encoder path:
AutoTokenizer + Qwen2ForCausalLM.get_decoder(), max_length=300.
The final tensor is stored as raw little-endian FP16 [1,1,300,896].
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("build/mobile-lite-assets"))
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

    # MobileI2V's model casts text conditioning to its FP16 model dtype.
    condition = hidden.unsqueeze(1).to(torch.float16).cpu().contiguous()
    if tuple(condition.shape) != EXPECTED_SHAPE:
        raise RuntimeError(f"Unexpected final shape: {tuple(condition.shape)}")

    out = args.out_dir / "null_condition.bin"
    array = condition.numpy().astype("<f2", copy=False)
    out.write_bytes(array.tobytes(order="C"))

    if out.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(
            f"Unexpected file size {out.stat().st_size}; expected {EXPECTED_BYTES}"
        )

    info = {
        "source_model": MODEL_ID,
        "source_revision": MODEL_REVISION,
        "prompt": "",
        "tokenizer_padding_side": "right",
        "max_length": MAX_LENGTH,
        "tensor_shape": list(EXPECTED_SHAPE),
        "tensor_dtype": "float16-le",
        "size_bytes": out.stat().st_size,
        "sha256": sha256(out),
    }
    info_path = args.out_dir / "null_condition.info.json"
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
