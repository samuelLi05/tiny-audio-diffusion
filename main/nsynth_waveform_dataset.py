from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset


class NSynthWaveformDataset(Dataset):
    """Dataset wrapper for metadata emitted by preprocess_nsynth_waveforms.py."""

    def __init__(self, metadata_path: str, split: Optional[str] = None):
        self.metadata_path = Path(metadata_path).expanduser().resolve()
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        rows: list[dict[str, Any]] = []
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if split is None or row.get("split") == split:
                    rows.append(row)

        if not rows:
            raise ValueError("No rows found for the requested split filter.")

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]

        clean_waveform = torch.load(row["clean_waveform_path"], map_location="cpu")
        noisy_waveform = torch.load(row["noisy_waveform_path"], map_location="cpu")

        conditioning = row.get("conditioning")
        conditioning_tensor = (
            torch.tensor(conditioning, dtype=torch.float32)
            if conditioning is not None
            else None
        )

        return {
            "Sample_id": row["sample_id"],
            "file_location": row["file_location"],
            "Name": row["name"],
            "Conditioning": conditioning_tensor,
            "Clean Waveform": clean_waveform,
            "Noisy Waveform": noisy_waveform,
            "Noise Type": row["noise_type"],
            "Split": row["split"],
            "Class": row["class"],
        }
