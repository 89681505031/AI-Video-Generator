#!/usr/bin/env python3
"""Export official MobileI2V PyTorch weights to the Android ONNX v4 contract.

This script is intentionally strict:
- the public hybrid_371.pth is treated as BASE unless the caller explicitly
  supplies a separate distilled checkpoint;
- 512x512 export refuses an upstream source tree that still hard-codes 2760
  positions;
- caption/mask inputs are omitted because current upstream MobileDiT does not
  consume cross-attention output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception as exc:
        raise SystemExit(
            f"Cannot determine MobileI2V source commit under {root}: {exc}"
        ) from exc


def check_source_compatibility(root: Path, width: int, height: int) -> None:
    source = root / "diffusion/model/nets/mobiledit.py"
    if not source.is_file():
        raise SystemExit(f"Missing upstream source: {source}")

    text = source.read_text("utf-8", errors="replace")

    if width == 512 and height == 512:
        hardcoded = (
            "repeat(1,2760)" in text.replace(" ", "")
            or "repeat(1, 2760)" in text
        )
        dynamic = "repeat(1,x.shape[1])" in text.replace(" ", "")
        if hardcoded and not dynamic:
            raise SystemExit(
                "This MobileI2V source still hard-codes 2760 positions. "
                "Use the official distillation branch for 512x512 export."
            )

    # mobiledit.py also contains older/unused block classes. Only inspect
    # the SanaBlock_cross class instantiated by Mobiledit itself.
    cross_start = text.find("class SanaBlock_cross")
    cross_end = text.find("class SanaBlock_vanila", cross_start)

    if cross_start < 0:
        raise SystemExit("Could not find class SanaBlock_cross in mobiledit.py")

    cross_block = text[
        cross_start : (cross_end if cross_end > cross_start else len(text))
    ]

    active_cross_attention = False
    for line in cross_block.splitlines():
        stripped = line.lstrip()
        if (
            "self.cross_attn(x, y" in stripped
            and not stripped.startswith("#")
        ):
            active_cross_attention = True
            break

    if active_cross_attention:
        raise SystemExit(
            "SanaBlock_cross has active cross-attention. "
            "Android v4 is prompt-free; export a new pack format instead."
        )


class DenoiserWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        cond_mask: torch.Tensor,
        flow_score: torch.Tensor,
    ) -> torch.Tensor:
        # Current upstream MobileDiT computes caption embeddings but the
        # cross-attention call is commented out. A constant dummy caption keeps
        # the PyTorch function callable; ONNX dead-code elimination prunes it.
        batch = x.shape[0]
        dummy_y = torch.zeros(
            (batch, 1, 300, 896),
            dtype=x.dtype,
            device=x.device,
        )

        return self.model(
            x,
            timestep,
            x[:, :, :1],
            dummy_y,
            cond_mask,
            flow_score,
            mask=None,
            data_info=None,
        )


class VaeEncoderWrapper(torch.nn.Module):
    def __init__(self, vae: torch.nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # AutoencoderKLLTXVideo._encode returns concatenated mean/logvar moments.
        return self.vae._encode(image)


class VideoDecoderWrapper(torch.nn.Module):
    def __init__(
        self,
        decoder_model: torch.nn.Module,
        scale_factor: float,
        target_height: int,
        target_width: int,
    ):
        super().__init__()
        self.decoder_model = decoder_model
        self.scale_factor = scale_factor
        self.target_height = target_height
        self.target_width = target_width

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder_model.decode(
            latent / self.scale_factor,
            return_dict=False,
        )[0]
        return decoded[
            :,
            :,
            :17,
            : self.target_height,
            : self.target_width,
        ]


def build_mobiledit(
    root: Path,
    checkpoint_path: Path,
    latent_h: int,
    latent_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    sys.path.insert(0, str(root))

    from diffusion.model.nets.mobiledit import Mobiledit_300M_P1_D16  # noqa: E402

    model = Mobiledit_300M_P1_D16(
        input_height=latent_h,
        input_width=latent_w,
        in_channels=128,
        caption_channels=896,
        model_max_length=300,
        qk_norm=True,
        y_norm=True,
        attn_type="linear",
        ffn_type="glumbconv",
        mlp_ratio=2.5,
        mlp_acts=("silu", "silu", None),
        use_pe=False,
        y_norm_scale_factor=0.01,
        linear_head_dim=32,
        pred_sigma=False,
        use_fp32_attention=True,
        config=None,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("state_dict", checkpoint)

    if "pos_embed" in state:
        del state["pos_embed"]

    missing, unexpected = model.load_state_dict(state, strict=False)

    allowed_missing = {"pos_embed"}
    real_missing = [key for key in missing if key not in allowed_missing]
    if real_missing:
        raise SystemExit(
            "Checkpoint is missing model keys; first entries: "
            + ", ".join(real_missing[:20])
        )
    if unexpected:
        raise SystemExit(
            "Checkpoint has unexpected model keys; first entries: "
            + ", ".join(unexpected[:20])
        )

    return model.eval().to(device=device, dtype=dtype)


def load_vae(
    vae_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    try:
        from diffusers import AutoencoderKLLTXVideo
    except ImportError as exc:
        raise SystemExit(
            "Install diffusers==0.35.2 before exporting the LTX video VAE."
        ) from exc

    vae = AutoencoderKLLTXVideo.from_pretrained(str(vae_path))
    return vae.eval().to(device=device, dtype=dtype)


def load_turbo_vaed_ltx(
    turbo_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    if not turbo_root.is_dir():
        raise SystemExit(f"Turbo-VAED root not found: {turbo_root}")
    if not config_path.is_file():
        raise SystemExit(f"Turbo-VAED config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"Turbo-VAED checkpoint not found: {checkpoint_path}")

    sys.path.insert(0, str(turbo_root))

    try:
        from diffusers_vae.src.diffusers.models.autoencoders.autoencoder_kl_turbo_vaed import (
            AutoencoderKLTurboVAED,
        )
    except ImportError as exc:
        raise SystemExit(
            "Could not import AutoencoderKLTurboVAED from the supplied "
            "Turbo-VAED source tree."
        ) from exc

    config = json.loads(config_path.read_text("utf-8"))
    if int(config.get("latent_channels", -1)) != 128:
        raise SystemExit(
            "Turbo-VAED-LTX config must use latent_channels=128."
        )

    model = AutoencoderKLTurboVAED.from_config(config=config)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    # Official validation_videos.py loads the checkpoint directly into decoder.
    state = checkpoint
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state = checkpoint["state_dict"]
        if "decoder" in checkpoint and isinstance(checkpoint["decoder"], dict):
            state = checkpoint["decoder"]

    missing, unexpected = model.decoder.load_state_dict(
        state,
        strict=False,
    )

    # Some training checkpoints may contain auxiliary alignment heads; decoder
    # export is valid as long as decoder parameters themselves are covered.
    critical_missing = [
        key for key in missing
        if key.startswith(("conv_", "up_blocks", "mid_block", "decoder."))
    ]
    if critical_missing:
        raise SystemExit(
            "Turbo-VAED decoder checkpoint is missing critical keys: "
            + ", ".join(critical_missing[:20])
        )

    if unexpected:
        print(
            "Turbo-VAED checkpoint contains extra keys (ignored):",
            ", ".join(unexpected[:20]),
        )

    return model.eval().to(device=device, dtype=dtype)


def export_graph(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            module,
            args,
            str(path),
            input_names=input_names,
            output_names=output_names,
            opset_version=18,
            do_constant_folding=True,
            export_params=True,
        )

    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"ONNX export did not create {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mobilei2v-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--vae", type=Path, required=True)
    ap.add_argument("--turbo-vaed-root", type=Path)
    ap.add_argument("--turbo-config", type=Path)
    ap.add_argument("--turbo-checkpoint", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--profile",
        choices=["720", "512"],
        default="720",
    )
    ap.add_argument(
        "--variant",
        choices=["base", "distilled"],
        default="base",
    )
    ap.add_argument("--sampling-steps", type=int, default=None)
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    args = ap.parse_args()

    root = args.mobilei2v_root.resolve()
    checkpoint = args.checkpoint.resolve()
    vae_path = args.vae.resolve()
    out_dir = args.out_dir.resolve()

    turbo_args = [
        args.turbo_vaed_root,
        args.turbo_config,
        args.turbo_checkpoint,
    ]
    if any(value is not None for value in turbo_args) and not all(
        value is not None for value in turbo_args
    ):
        raise SystemExit(
            "Turbo decoder requires all three: --turbo-vaed-root, "
            "--turbo-config and --turbo-checkpoint"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if not vae_path.exists():
        raise SystemExit(f"VAE path not found: {vae_path}")

    if args.profile == "720":
        width, height = 1280, 720
    else:
        width, height = 512, 512

    check_source_compatibility(root, width, height)

    steps = args.sampling_steps
    if steps is None:
        steps = 30 if args.variant == "base" else 2

    if steps < 1 or steps > 60:
        raise SystemExit("sampling steps must be between 1 and 60")
    if args.variant == "distilled" and steps > 4:
        raise SystemExit("distilled export must use <=4 sampling steps")

    latent_h = ceil_div(height, 32)
    latent_w = ceil_div(width, 32)
    positions = 3 * latent_h * latent_w

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    if dtype == torch.float16 and device.type != "cuda":
        raise SystemExit(
            "fp16 export requires CUDA for reliable tracing; use --dtype fp32 on CPU"
        )

    model = build_mobiledit(
        root,
        checkpoint,
        latent_h,
        latent_w,
        device,
        dtype,
    )
    denoiser = DenoiserWrapper(model).eval()

    x = torch.randn(
        1,
        128,
        3,
        latent_h,
        latent_w,
        device=device,
        dtype=dtype,
    )
    timestep = torch.tensor([1000.0], device=device, dtype=dtype)
    cond_mask = torch.zeros(
        1,
        positions,
        device=device,
        dtype=dtype,
    )
    cond_mask[:, : latent_h * latent_w] = 1
    flow_score = torch.tensor([2.0], device=device, dtype=dtype)

    export_graph(
        denoiser,
        (x, timestep, cond_mask, flow_score),
        out_dir / "mobilei2v_transformer.onnx",
        ["x", "timestep", "cond_mask", "flow_score"],
        ["noise"],
    )

    del denoiser
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    vae = load_vae(vae_path, device, dtype)

    encoder = VaeEncoderWrapper(vae).eval()
    image = torch.zeros(
        1,
        3,
        1,
        height,
        width,
        device=device,
        dtype=dtype,
    )
    export_graph(
        encoder,
        (image,),
        out_dir / "vae_encoder.onnx",
        ["image"],
        ["posterior_moments"],
    )

    decoder_source = "ltx"
    decoder_model = vae

    if args.turbo_vaed_root is not None:
        decoder_source = "turbo-vaed-ltx"
        decoder_model = load_turbo_vaed_ltx(
            args.turbo_vaed_root.resolve(),
            args.turbo_config.resolve(),
            args.turbo_checkpoint.resolve(),
            device,
            dtype,
        )

    decoder = VideoDecoderWrapper(
        decoder_model,
        scale_factor=0.41407,
        target_height=height,
        target_width=width,
    ).eval()
    latent = torch.zeros(
        1,
        128,
        3,
        latent_h,
        latent_w,
        device=device,
        dtype=dtype,
    )
    export_graph(
        decoder,
        (latent,),
        out_dir / "video_decoder.onnx",
        ["latent"],
        ["video"],
    )

    source_commit = git_commit(root)
    runtime = {
        "variant": args.variant,
        "width": width,
        "height": height,
        "frames": 17,
        "latent_channels": 128,
        "vae_downsample_rate": 32,
        "temporal_latents": 3,
        "sampling_steps": steps,
        "text_conditioning": False,
        "text_max_length": 300,
        "caption_channels": 896,
        "source_commit": source_commit,
        "source_checkpoint_sha256": sha256(checkpoint),
        "profile": args.profile,
        "export_dtype": args.dtype,
        "latent_height": latent_h,
        "latent_width": latent_w,
        "sequence_positions": positions,
        "decoder": decoder_source,
        "turbo_decoder_checkpoint_sha256": (
            sha256(args.turbo_checkpoint.resolve())
            if args.turbo_checkpoint is not None
            else None
        ),
    }
    (out_dir / "runtime.json").write_text(
        json.dumps(runtime, indent=2) + "\n",
        "utf-8",
    )

    print(json.dumps(runtime, indent=2))
    print("Exported:")
    for name in (
        "vae_encoder.onnx",
        "mobilei2v_transformer.onnx",
        "video_decoder.onnx",
        "runtime.json",
    ):
        path = out_dir / name
        print(f"  {name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
