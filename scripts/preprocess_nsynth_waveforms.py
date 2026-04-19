"""Build a deterministic waveform dataset (clean + noisy) from NSynth JSON+WAV.

Expected extracted NSynth layout:
  <nsynth_root>/nsynth-train/examples.json
  <nsynth_root>/nsynth-train/audio/*.wav
  <nsynth_root>/nsynth-valid/examples.json
  <nsynth_root>/nsynth-valid/audio/*.wav
  <nsynth_root>/nsynth-test/examples.json
  <nsynth_root>/nsynth-test/audio/*.wav

Primary training artifact is waveform tensors and metadata.
Optional spectrogram export is only for QC/inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch  # type: ignore[reportMissingImports]
import torchaudio  # type: ignore[reportMissingImports]


DEFAULT_CLASSES = ["bass", "guitar", "keyboard"]
DEFAULT_SPLIT_COUNTS = {"train": 105, "valid": 30, "test": 15}
DEFAULT_NOISE_RECIPES = ["gaussian", "uniform", "pink"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic clean/noisy waveform tensors and metadata from NSynth."
    )
    parser.add_argument(
        "--nsynth-root",
        required=True,
        help="Directory containing extracted nsynth-train/nsynth-valid/nsynth-test folders.",
    )
    parser.add_argument(
        "--output-root",
        default="./data/nsynth_waveform_processed",
        help="Root directory for generated tensors and metadata.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Instrument families to keep.",
    )
    parser.add_argument("--train-per-class", type=int, default=DEFAULT_SPLIT_COUNTS["train"])
    parser.add_argument("--valid-per-class", type=int, default=DEFAULT_SPLIT_COUNTS["valid"])
    parser.add_argument("--test-per-class", type=int, default=DEFAULT_SPLIT_COUNTS["test"])
    parser.add_argument("--seed", type=int, default=460)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--conditioning",
        choices=["none", "onehot"],
        default="onehot",
        help="Conditioning encoding stored in metadata.",
    )
    parser.add_argument(
        "--noise-recipes",
        nargs="+",
        default=DEFAULT_NOISE_RECIPES,
        choices=DEFAULT_NOISE_RECIPES,
        help="Noise recipes sampled deterministically per clip.",
    )
    parser.add_argument(
        "--export-qc-spectrograms",
        action="store_true",
        help="Optionally save Mel spectrogram tensors for QC only.",
    )
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-mels", type=int, default=128)
    return parser.parse_args()


def stable_seed(seed: int, *parts: str) -> int:
    text = f"{seed}|{'|'.join(parts)}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def load_examples(split_root: Path) -> dict[str, Any]:
    examples_path = split_root / "examples.json"
    if not examples_path.exists():
        raise FileNotFoundError(f"Missing NSynth metadata: {examples_path}")
    with examples_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_examples_for_class(
    examples: dict[str, Any],
    family: str,
    count: int,
    split: str,
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    candidates = [
        (note_id, item)
        for note_id, item in examples.items()
        if item.get("instrument_family_str") == family
    ]
    candidates.sort(key=lambda x: x[0])

    rng = random.Random(stable_seed(seed, split, family))
    rng.shuffle(candidates)

    if len(candidates) < count:
        raise ValueError(
            f"Not enough samples for class '{family}' in split '{split}': "
            f"requested {count}, available {len(candidates)}"
        )
    return candidates[:count]


def ensure_output_tree(root: Path, splits: list[str], classes: list[str], export_qc: bool) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    for split in splits:
        for family in classes:
            (root / "waveforms" / "clean" / split / family).mkdir(parents=True, exist_ok=True)
            (root / "waveforms" / "noisy" / split / family).mkdir(parents=True, exist_ok=True)
            if export_qc:
                (root / "qc" / "spectrograms" / "clean" / split / family).mkdir(
                    parents=True,
                    exist_ok=True,
                )
                (root / "qc" / "spectrograms" / "noisy" / split / family).mkdir(
                    parents=True,
                    exist_ok=True,
                )


def maybe_resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    return torchaudio.functional.resample(waveform, source_rate, target_rate)


def add_noise(waveform: torch.Tensor, noise_type: str, generator: torch.Generator) -> torch.Tensor:
    if noise_type == "gaussian":
        noise = torch.randn(waveform.shape, generator=generator, dtype=waveform.dtype)
        return torch.clamp(waveform + 0.01 * noise, min=-1.0, max=1.0)

    if noise_type == "uniform":
        noise = (torch.rand(waveform.shape, generator=generator, dtype=waveform.dtype) * 2.0) - 1.0
        return torch.clamp(waveform + 0.01 * noise, min=-1.0, max=1.0)

    if noise_type == "pink":
        white = torch.randn(waveform.shape, generator=generator, dtype=waveform.dtype)
        pink = torch.cumsum(white, dim=-1)
        pink = pink / (pink.abs().amax(dim=-1, keepdim=True) + 1e-8)
        return torch.clamp(waveform + 0.01 * pink, min=-1.0, max=1.0)

    raise ValueError(f"Unsupported noise type: {noise_type}")


def one_hot_conditioning(family: str, families: list[str]) -> list[float]:
    vec = [0.0] * len(families)
    vec[families.index(family)] = 1.0
    return vec


def main() -> None:
    args = parse_args()
    nsynth_root = Path(args.nsynth_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    families = list(args.classes)
    splits = ["train", "valid", "test"]
    per_class = {
        "train": args.train_per_class,
        "valid": args.valid_per_class,
        "test": args.test_per_class,
    }

    ensure_output_tree(output_root, splits, families, args.export_qc_spectrograms)

    qc_mel = None
    if args.export_qc_spectrograms:
        qc_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
        )

    metadata_rows: list[dict[str, Any]] = []
    split_ids: dict[str, list[str]] = {split: [] for split in splits}

    for split in splits:
        split_root = nsynth_root / f"nsynth-{split}"
        audio_root = split_root / "audio"
        examples = load_examples(split_root)

        for family in families:
            selected = select_examples_for_class(
                examples=examples,
                family=family,
                count=per_class[split],
                split=split,
                seed=args.seed,
            )

            for note_id, item in selected:
                wav_path = audio_root / f"{note_id}.wav"
                if not wav_path.exists():
                    raise FileNotFoundError(f"Missing wav file: {wav_path}")

                waveform, sr = torchaudio.load(str(wav_path))
                if waveform.size(0) > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveform = maybe_resample(waveform, sr, args.sample_rate).to(torch.float32)

                row_seed = stable_seed(args.seed, split, family, note_id)
                row_rng = random.Random(row_seed)
                noise_type = row_rng.choice(args.noise_recipes)
                noise_gen = torch.Generator().manual_seed(row_seed)

                noisy_wave = add_noise(waveform, noise_type=noise_type, generator=noise_gen)

                clean_path = output_root / "waveforms" / "clean" / split / family / f"{note_id}.pt"
                noisy_path = output_root / "waveforms" / "noisy" / split / family / f"{note_id}.pt"
                torch.save(waveform, clean_path)
                torch.save(noisy_wave, noisy_path)

                qc_clean_path = None
                qc_noisy_path = None
                if qc_mel is not None:
                    clean_qc = qc_mel(waveform)
                    noisy_qc = qc_mel(noisy_wave)
                    qc_clean_path = (
                        output_root
                        / "qc"
                        / "spectrograms"
                        / "clean"
                        / split
                        / family
                        / f"{note_id}.pt"
                    )
                    qc_noisy_path = (
                        output_root
                        / "qc"
                        / "spectrograms"
                        / "noisy"
                        / split
                        / family
                        / f"{note_id}.pt"
                    )
                    torch.save(clean_qc.to(torch.float32), qc_clean_path)
                    torch.save(noisy_qc.to(torch.float32), qc_noisy_path)

                conditioning = (
                    one_hot_conditioning(family, families)
                    if args.conditioning == "onehot"
                    else None
                )
                row = {
                    "sample_id": note_id,
                    "name": item.get("note_str", note_id),
                    "split": split,
                    "class": family,
                    "file_location": str(wav_path),
                    "clean_waveform_path": str(clean_path),
                    "noisy_waveform_path": str(noisy_path),
                    "noise_type": noise_type,
                    "conditioning": conditioning,
                    "sample_rate": args.sample_rate,
                    "qc_clean_spectrogram_path": str(qc_clean_path) if qc_clean_path else None,
                    "qc_noisy_spectrogram_path": str(qc_noisy_path) if qc_noisy_path else None,
                }
                metadata_rows.append(row)
                split_ids[split].append(note_id)

    metadata_path = output_root / "metadata" / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row) + "\n")

    splits_path = output_root / "metadata" / "splits.json"
    with splits_path.open("w", encoding="utf-8") as handle:
        json.dump(split_ids, handle, indent=2)

    first = metadata_rows[0]
    clean_sample = torch.load(first["clean_waveform_path"], map_location="cpu")
    noisy_sample = torch.load(first["noisy_waveform_path"], map_location="cpu")
    shape_report = {
        "sample_id": first["sample_id"],
        "clean_waveform_shape": list(clean_sample.shape),
        "noisy_waveform_shape": list(noisy_sample.shape),
        "conditioning_shape": (
            [len(first["conditioning"])] if first["conditioning"] is not None else None
        ),
        "total_rows": len(metadata_rows),
    }
    shape_path = output_root / "metadata" / "sample_shapes.json"
    with shape_path.open("w", encoding="utf-8") as handle:
        json.dump(shape_report, handle, indent=2)

    print(f"Created waveform dataset under: {output_root}")
    print(f"Metadata file: {metadata_path}")
    print(f"Split file: {splits_path}")
    print(f"Sample shape file: {shape_path}")


if __name__ == "__main__":
    main()
