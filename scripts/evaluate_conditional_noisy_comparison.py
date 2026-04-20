from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare provided noisy waveforms vs generated noisy waveforms with a trained conditional model."
    )
    parser.add_argument(
        "--config-path",
        default="exp/nsynth_conditional_16gb_no_wandb.yaml",
        help="Path to experiment config used for model/datamodule instantiation.",
    )
    parser.add_argument(
        "--ckpt-path",
        required=True,
        help="Path to trained checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--metadata-path",
        default="data/nsynth_waveform_box/metadata/metadata.jsonl",
        help="Path to metadata JSONL.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "valid", "test"],
        help="Data split to evaluate.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Number of batches to evaluate.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Sampling steps for conditional generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for synthetic noisy generation.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on.",
    )
    parser.add_argument(
        "--output-path",
        default="logs/eval/conditional_noisy_comparison.json",
        help="Where to store aggregated evaluation metrics.",
    )
    return parser.parse_args()


def add_recipe_noise(clean_wave: torch.Tensor, noise_types: list[str], seed: int) -> torch.Tensor:
    generated = []
    for idx in range(clean_wave.shape[0]):
        wave = clean_wave[idx : idx + 1]
        gen = torch.Generator(device=wave.device).manual_seed(seed + idx)
        recipe = noise_types[idx]

        if recipe == "gaussian":
            noise = torch.randn(wave.shape, generator=gen, device=wave.device, dtype=wave.dtype)
            noisy = torch.clamp(wave + 0.01 * noise, min=-1.0, max=1.0)
        elif recipe == "uniform":
            noise = (torch.rand(wave.shape, generator=gen, device=wave.device, dtype=wave.dtype) * 2.0) - 1.0
            noisy = torch.clamp(wave + 0.01 * noise, min=-1.0, max=1.0)
        elif recipe == "pink":
            white = torch.randn(wave.shape, generator=gen, device=wave.device, dtype=wave.dtype)
            pink = torch.cumsum(white, dim=-1)
            pink = pink / (pink.abs().amax(dim=-1, keepdim=True) + 1e-8)
            noisy = torch.clamp(wave + 0.01 * pink, min=-1.0, max=1.0)
        else:
            noise = torch.randn(wave.shape, generator=gen, device=wave.device, dtype=wave.dtype)
            noisy = torch.clamp(wave + 0.01 * noise, min=-1.0, max=1.0)

        generated.append(noisy)

    return torch.cat(generated, dim=0)


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean((a - b) ** 2).item()


def snr_db(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    signal_power = torch.mean(clean ** 2)
    noise_power = torch.mean((clean - noisy) ** 2) + 1e-12
    return (10.0 * torch.log10(signal_power / noise_power)).item()


def _get_split_loader(datamodule: Any, split: str):
    if split == "train":
        return datamodule.train_dataloader()
    if split == "valid":
        return datamodule.val_dataloader()
    return datamodule.test_dataloader()


def main() -> None:
    args = parse_args()

    cfg = OmegaConf.load(args.config_path)
    cfg.datamodule.metadata_path = args.metadata_path

    datamodule = hydra.utils.instantiate(cfg.datamodule, _convert_="partial")
    datamodule.setup("fit")
    model = hydra.utils.instantiate(cfg.model, _convert_="partial")

    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except RuntimeError as err:
        print(
            "Strict checkpoint load failed. Falling back to strict=False. "
            f"Reason: {err}"
        )
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    model = model.to(args.device)

    loader = _get_split_loader(datamodule, args.split)

    total_samples = 0
    totals = {
        "input_mse_provided_vs_generated": 0.0,
        "input_snr_db_provided": 0.0,
        "input_snr_db_generated": 0.0,
        "output_mse_provided_path": 0.0,
        "output_mse_generated_path": 0.0,
        "output_mse_between_paths": 0.0,
    }

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.num_batches:
                break

            clean = batch["Clean Waveform"].to(args.device)
            provided_noisy = batch["Noisy Waveform"].to(args.device)
            class_ids = batch["Class Id"].to(args.device)
            conditioning = batch.get("Conditioning")
            if conditioning is not None:
                conditioning = conditioning.to(args.device)
            noise_types = batch.get("Noise Type", ["gaussian"] * clean.shape[0])

            generated_noisy = add_recipe_noise(clean, noise_types, seed=args.seed + batch_idx * 1000)

            out_from_provided = model.sample_conditioned(
                noise=provided_noisy,
                num_steps=args.num_steps,
                conditioning=conditioning,
                class_ids=class_ids,
                use_ema_model=True,
            )
            out_from_generated = model.sample_conditioned(
                noise=generated_noisy,
                num_steps=args.num_steps,
                conditioning=conditioning,
                class_ids=class_ids,
                use_ema_model=True,
            )

            bsz = clean.shape[0]
            total_samples += bsz

            totals["input_mse_provided_vs_generated"] += mse(provided_noisy, generated_noisy) * bsz
            totals["input_snr_db_provided"] += snr_db(clean, provided_noisy) * bsz
            totals["input_snr_db_generated"] += snr_db(clean, generated_noisy) * bsz
            totals["output_mse_provided_path"] += mse(out_from_provided, clean) * bsz
            totals["output_mse_generated_path"] += mse(out_from_generated, clean) * bsz
            totals["output_mse_between_paths"] += mse(out_from_provided, out_from_generated) * bsz

    if total_samples == 0:
        raise RuntimeError("No samples evaluated. Check split and metadata path.")

    metrics = {
        "config_path": str(Path(args.config_path).resolve()),
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "metadata_path": str(Path(args.metadata_path).resolve()),
        "split": args.split,
        "num_batches": args.num_batches,
        "num_steps": args.num_steps,
        "total_samples": total_samples,
        "averages": {k: v / total_samples for k, v in totals.items()},
    }

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation metrics to: {output_path}")


if __name__ == "__main__":
    main()
