#!/usr/bin/env python3
"""Package exported MobileI2V runtime files into a verified Android ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import zipfile

PACK_ID = "mobile_i2v_v1"
FORMAT_VERSION = 4

CORE = [
    "vae_encoder.onnx",
    "mobilei2v_transformer.onnx",
    "video_decoder.onnx",
    "runtime.json",
]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=pathlib.Path)
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("mobile_i2v_v1.zip"),
    )
    args = ap.parse_args()

    model_dir = args.model_dir.resolve()
    runtime_path = model_dir / "runtime.json"
    if not runtime_path.is_file():
        raise SystemExit("Missing runtime.json")

    runtime = json.loads(runtime_path.read_text("utf-8"))
    if runtime.get("text_conditioning", False):
        raise SystemExit(
            "format v4 requires text_conditioning=false; "
            "current official MobileI2V cross-attention is disabled"
        )

    missing = [name for name in CORE if not (model_dir / name).is_file()]
    if missing:
        raise SystemExit("Missing files: " + ", ".join(missing))

    files = {name: sha256(model_dir / name) for name in CORE}
    manifest = {
        "pack_id": PACK_ID,
        "format_version": FORMAT_VERSION,
        "files": files,
    }

    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        "utf-8",
    )

    with zipfile.ZipFile(
        args.out,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as z:
        for name in CORE + ["manifest.json"]:
            z.write(model_dir / name, arcname=name)

    print(args.out)
    print("files:", len(CORE) + 1)
    print("zip_sha256:", sha256(args.out))


if __name__ == "__main__":
    main()
