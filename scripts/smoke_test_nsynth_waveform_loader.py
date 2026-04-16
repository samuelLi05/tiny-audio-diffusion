from __future__ import annotations

import argparse
from typing import Any

import torch
from torch.utils.data import DataLoader

from main.nsynth_waveform_dataset import NSynthWaveformDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test one sample and one batch from NSynthWaveformDataset."
    )
    parser.add_argument(
        "--metadata-path",
        required=True,
        help="Path to metadata.jsonl produced by preprocess_nsynth_waveforms.py",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "valid", "test"],
        help="Split to test.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    conditioning_values = [item["Conditioning"] for item in batch]
    conditioning = None
    if all(value is not None for value in conditioning_values):
        conditioning = torch.stack(conditioning_values)

    return {
        "Sample_id": [item["Sample_id"] for item in batch],
        "file_location": [item["file_location"] for item in batch],
        "Name": [item["Name"] for item in batch],
        "Conditioning": conditioning,
        "Clean Waveform": torch.stack([item["Clean Waveform"] for item in batch]),
        "Noisy Waveform": torch.stack([item["Noisy Waveform"] for item in batch]),
        "Noise Type": [item["Noise Type"] for item in batch],
        "Split": [item["Split"] for item in batch],
        "Class": [item["Class"] for item in batch],
    }


def main() -> None:
    args = parse_args()
    dataset = NSynthWaveformDataset(args.metadata_path, split=args.split)

    sample = dataset[0]
    print("Single sample check")
    print(f"Sample_id: {sample['Sample_id']}")
    print(f"Clean Waveform shape: {tuple(sample['Clean Waveform'].shape)}")
    print(f"Noisy Waveform shape: {tuple(sample['Noisy Waveform'].shape)}")
    if sample["Conditioning"] is not None:
        print(f"Conditioning shape: {tuple(sample['Conditioning'].shape)}")
    else:
        print("Conditioning shape: None")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))

    print("\nBatch check")
    print(f"Batch size: {len(batch['Sample_id'])}")
    print(f"Clean Waveform batch shape: {tuple(batch['Clean Waveform'].shape)}")
    print(f"Noisy Waveform batch shape: {tuple(batch['Noisy Waveform'].shape)}")
    if batch["Conditioning"] is not None:
        print(f"Conditioning batch shape: {tuple(batch['Conditioning'].shape)}")
    else:
        print("Conditioning batch shape: None")


if __name__ == "__main__":
    main()
