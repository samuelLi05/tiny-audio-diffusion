"""Download a small Nsynth subset and export it as WAV files.

The script keeps the existing training pipeline simple by materializing a local
folder of WAV files that can be consumed by the current `audio_data_pytorch`
dataset. Nsynth audio is mono at 16kHz, so by default the script duplicates the
waveform to stereo and keeps the repository's audio model configuration intact.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torchaudio
import soundfile as sf
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a small Nsynth subset and export it as WAV files."
    )
    parser.add_argument(
        "--dataset-id",
        default="jg583/NSynth",
        help="Hugging Face dataset ID to download.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--family",
        default="guitar",
        help="Nsynth instrument family to keep, for example guitar or bass.",
    )
    parser.add_argument(
        "--instrument-str",
        default=None,
        help="Optional exact instrument_str filter, for example guitar_electronic_011.",
    )
    parser.add_argument(
        "--instrument-id",
        type=int,
        default=None,
        help="Optional exact integer instrument id filter.",
    )
    parser.add_argument(
        "--source",
        choices=["acoustic", "electronic", "synthetic"],
        default=None,
        help="Optional instrument source filter.",
    )
    parser.add_argument(
        "--pitch-min",
        type=int,
        default=None,
        help="Optional minimum MIDI pitch (inclusive).",
    )
    parser.add_argument(
        "--pitch-max",
        type=int,
        default=None,
        help="Optional maximum MIDI pitch (inclusive).",
    )
    parser.add_argument(
        "--velocity-min",
        type=int,
        default=None,
        help="Optional minimum MIDI velocity (inclusive).",
    )
    parser.add_argument(
        "--velocity-max",
        type=int,
        default=None,
        help="Optional maximum MIDI velocity (inclusive).",
    )
    parser.add_argument(
        "--note-min",
        type=int,
        default=None,
        help="Optional minimum note id (inclusive).",
    )
    parser.add_argument(
        "--note-max",
        type=int,
        default=None,
        help="Optional maximum note id (inclusive).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=512,
        help="Maximum number of audio files to export.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/nsynth_guitar/wav_dataset",
        help="Directory where the WAV files will be written.",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=16000,
        help="Sample rate for exported WAV files.",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=2048,
        help="Shuffle buffer size used when streaming from Hugging Face.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed used for dataset shuffling.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_false",
        dest="streaming",
        help="Disable streaming and load the split into memory first.",
    )
    parser.set_defaults(streaming=True)
    return parser.parse_args()


def ensure_clean_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.wav"):
        path.unlink()
    manifest = output_dir.parent / "manifest.jsonl"
    if manifest.exists():
        manifest.unlink()


def get_parquet_files(dataset_id: str, split: str) -> list[str]:
    api = HfApi()
    repo_files = api.list_repo_files(dataset_id, repo_type="dataset")

    if split == "train":
        return [
            hf_hub_url(dataset_id, filename, repo_type="dataset")
            for filename in repo_files
            if filename.startswith("data/train/") and filename.endswith(".parquet")
        ]

    if split == "validation":
        return [
            hf_hub_url(dataset_id, filename, repo_type="dataset")
            for filename in repo_files
            if filename == "data/validation/validation.parquet"
        ]

    if split == "test":
        return [
            hf_hub_url(dataset_id, filename, repo_type="dataset")
            for filename in repo_files
            if filename == "data/test/test.parquet"
        ]

    raise ValueError(
        f'Unsupported split "{split}". Expected train, validation, or test.'
    )


def to_waveform(audio: Mapping[str, object], target_sample_rate: int) -> torch.Tensor:
    if "array" in audio:
        waveform = torch.as_tensor(audio["array"], dtype=torch.float32)
        sample_rate = int(audio.get("sampling_rate", target_sample_rate))
    elif "path" in audio:
        waveform, sample_rate = torchaudio.load(str(audio["path"]))
    else:
        raise ValueError("Nsynth audio entry did not include an array or path.")

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(
            waveform
        )

    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    return waveform.clamp(-1.0, 1.0)


def in_range(value: int, min_value: int | None, max_value: int | None) -> bool:
    if min_value is not None and value < min_value:
        return False
    if max_value is not None and value > max_value:
        return False
    return True


def matches_filters(example: Mapping[str, object], args: argparse.Namespace) -> bool:
    if example.get("instrument_family_str") != args.family:
        return False

    if args.instrument_str is not None and example.get("instrument_str") != args.instrument_str:
        return False

    if args.instrument_id is not None and int(example.get("instrument", -1)) != args.instrument_id:
        return False

    if args.source is not None and example.get("instrument_source_str") != args.source:
        return False

    pitch = int(example.get("pitch", -1))
    velocity = int(example.get("velocity", -1))
    note = int(example.get("note", -1))

    if not in_range(pitch, args.pitch_min, args.pitch_max):
        return False
    if not in_range(velocity, args.velocity_min, args.velocity_max):
        return False
    if not in_range(note, args.note_min, args.note_max):
        return False

    return True


def iter_filtered_examples(dataset: Iterable[Mapping[str, object]], args: argparse.Namespace):
    for example in dataset:
        if matches_filters(example, args):
            yield example


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_clean_output(output_dir)

    parquet_files = get_parquet_files(args.dataset_id, args.split)
    if not parquet_files:
        raise RuntimeError(
            f'No parquet files found for split "{args.split}" in {args.dataset_id}.'
        )

    dataset = load_dataset(
        "parquet",
        data_files=parquet_files,
        split="train",
        streaming=args.streaming,
    )
    if args.streaming:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
    else:
        dataset = dataset.shuffle(seed=args.seed)

    manifest_path = output_dir.parent / "manifest.jsonl"
    written = 0
    instrument_counts: Counter[str] = Counter()

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for example in iter_filtered_examples(dataset, args):
            waveform = to_waveform(example["audio"], args.target_sample_rate)
            sample_rate = int(args.target_sample_rate)

            file_name = f"{args.family}_{written:05d}.wav"
            file_path = output_dir / file_name
            sf.write(str(file_path), waveform.transpose(0, 1).numpy(), sample_rate)

            manifest_file.write(
                json.dumps(
                    {
                        "file_name": file_name,
                        "family": args.family,
                        "note": example.get("note"),
                        "pitch": example.get("pitch"),
                        "velocity": example.get("velocity"),
                        "instrument": example.get("instrument"),
                        "instrument_str": example.get("instrument_str"),
                        "instrument_family_str": example.get("instrument_family_str"),
                        "instrument_source_str": example.get("instrument_source_str"),
                        "sample_rate": sample_rate,
                    }
                )
                + "\n"
            )

            written += 1
            instrument_counts[str(example.get("instrument_str", "unknown"))] += 1
            if written >= args.max_items:
                break

    if written == 0:
        raise RuntimeError(
            f'No examples matched the requested filters in split "{args.split}".'
        )

    top_instruments = ", ".join(
        [f"{name}:{count}" for name, count in instrument_counts.most_common(5)]
    )
    print(
        f"Exported {written} {args.family} examples to {output_dir} "
        f"and wrote metadata to {manifest_path}."
    )
    if top_instruments:
        print(f"Top instrument_str counts: {top_instruments}")


if __name__ == "__main__":
    main()