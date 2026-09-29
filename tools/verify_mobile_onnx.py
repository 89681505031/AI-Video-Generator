#!/usr/bin/env python3
"""Verify exported ONNX files against Android pack format v4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import onnxruntime as ort

EXPECTED = {
    "vae_encoder.onnx": (
        {"image"},
        {"posterior_moments"},
    ),
    "mobilei2v_transformer.onnx": (
        {"x", "timestep", "cond_mask", "flow_score"},
        {"noise"},
    ),
    "video_decoder.onnx": (
        {"latent"},
        {"video"},
    ),
}


def io_names(model: onnx.ModelProto) -> tuple[set[str], set[str]]:
    initializer_names = {item.name for item in model.graph.initializer}
    inputs = {
        item.name
        for item in model.graph.input
        if item.name not in initializer_names
    }
    outputs = {item.name for item in model.graph.output}
    return inputs, outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    args = ap.parse_args()
    root = args.model_dir.resolve()

    runtime_path = root / "runtime.json"
    if not runtime_path.is_file():
        raise SystemExit("Missing runtime.json")

    runtime = json.loads(runtime_path.read_text("utf-8"))
    if runtime.get("text_conditioning", False):
        raise SystemExit("Android v4 requires text_conditioning=false")

    for name, (expected_inputs, expected_outputs) in EXPECTED.items():
        path = root / name
        if not path.is_file():
            raise SystemExit(f"Missing {name}")

        model = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(model)
        inputs, outputs = io_names(model)

        if inputs != expected_inputs:
            raise SystemExit(
                f"{name} inputs={sorted(inputs)}, expected={sorted(expected_inputs)}"
            )
        if outputs != expected_outputs:
            raise SystemExit(
                f"{name} outputs={sorted(outputs)}, expected={sorted(expected_outputs)}"
            )

        # This is deliberately stronger than onnx.checker: Android failed on
        # a graph that checker accepted because ORT rejected a Transpose perm
        # containing -1. Creating a CPU session exercises ORT's real graph
        # loading/type-inference path before the pack is published.
        session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        session_inputs = {item.name for item in session.get_inputs()}
        session_outputs = {item.name for item in session.get_outputs()}
        if session_inputs != expected_inputs:
            raise SystemExit(
                f"{name} ORT inputs={sorted(session_inputs)}, "
                f"expected={sorted(expected_inputs)}"
            )
        if session_outputs != expected_outputs:
            raise SystemExit(
                f"{name} ORT outputs={sorted(session_outputs)}, "
                f"expected={sorted(expected_outputs)}"
            )

        print(f"OK {name} (onnx.checker + ONNX Runtime load)")
        print("  inputs :", sorted(inputs))
        print("  outputs:", sorted(outputs))

    print("Android ONNX v4 contract: OK")


if __name__ == "__main__":
    main()
