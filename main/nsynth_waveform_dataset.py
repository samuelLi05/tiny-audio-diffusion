from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
import torchaudio
from torch.utils.data import Dataset


class NSynthWaveformDataset(Dataset):
    """Dataset wrapper for metadata emitted by preprocess_nsynth_waveforms.py."""

    def __init__(
        self,
        metadata_path: str,
        split: Optional[str] = None,
        *,
        target_sample_rate: Optional[int] = None,
        target_length: Optional[int] = None,
    ):
        self.metadata_path = Path(metadata_path).expanduser().resolve()
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")
        self.dataset_root = self.metadata_path.parent.parent
        self.target_sample_rate = target_sample_rate
        self.target_length = target_length
        self._clean_index: dict[str, Path] = {}
        self._noisy_index: dict[str, Path] = {}

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
        self._build_waveform_index_if_needed()

    def _build_waveform_index_if_needed(self) -> None:
        missing_path_fields = any(
            ("clean_waveform_path" not in row) or ("noisy_waveform_path" not in row)
            for row in self.rows
        )
        if not missing_path_fields:
            return

        clean_root = self.dataset_root / "waveforms" / "clean"
        noisy_root = self.dataset_root / "waveforms" / "noisy"

        if clean_root.exists():
            self._clean_index = {
                path.stem: path
                for path in clean_root.rglob("*.pt")
            }
        if noisy_root.exists():
            self._noisy_index = {
                path.stem: path
                for path in noisy_root.rglob("*.pt")
            }

    def _resolve_waveform_path(self, row: dict[str, Any], key: str) -> Path:
        explicit_path = row.get(key)
        if explicit_path:
            path = Path(explicit_path).expanduser()
            if not path.is_absolute():
                path = (self.dataset_root / path).resolve()
            if path.exists():
                return path

        sample_id = row.get("sample_id")
        split = row.get("split")
        class_name = row.get("class")
        is_clean = key.startswith("clean")

        subdir = "clean" if is_clean else "noisy"
        if sample_id and split and class_name:
            candidate = (
                self.dataset_root
                / "waveforms"
                / subdir
                / str(split)
                / str(class_name)
                / f"{sample_id}.pt"
            )
            if candidate.exists():
                return candidate

        if sample_id:
            index = self._clean_index if is_clean else self._noisy_index
            indexed = index.get(sample_id)
            if indexed is not None and indexed.exists():
                return indexed

        raise FileNotFoundError(
            f"Could not resolve {key} for sample_id={sample_id}. "
            f"Expected explicit metadata path or a local PT under waveforms/{subdir}/..."
        )

    def _resolve_waveform_path_optional(self, row: dict[str, Any], key: str) -> Optional[Path]:
        try:
            return self._resolve_waveform_path(row, key)
        except FileNotFoundError:
            return None

    def _normalize_waveform(
        self,
        wave: torch.Tensor,
        source_sample_rate: Optional[int],
    ) -> torch.Tensor:
        normalized = wave.float()

        if normalized.ndim == 1:
            normalized = normalized.unsqueeze(0)

        if (
            self.target_sample_rate is not None
            and source_sample_rate is not None
            and source_sample_rate != self.target_sample_rate
        ):
            normalized = torchaudio.functional.resample(
                normalized,
                source_sample_rate,
                self.target_sample_rate,
            )

        if self.target_length is not None:
            if normalized.shape[-1] > self.target_length:
                normalized = normalized[..., : self.target_length]
            elif normalized.shape[-1] < self.target_length:
                pad = self.target_length - normalized.shape[-1]
                normalized = torch.nn.functional.pad(normalized, (0, pad))

        return normalized

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]

        clean_path = self._resolve_waveform_path(row, "clean_waveform_path")
        noisy_path = self._resolve_waveform_path_optional(row, "noisy_waveform_path")

        source_sample_rate = row.get("sample_rate")

        clean_waveform = torch.load(clean_path, map_location="cpu")
        noisy_waveform = torch.load(noisy_path, map_location="cpu") if noisy_path else clean_waveform
        clean_waveform = self._normalize_waveform(clean_waveform, source_sample_rate)
        noisy_waveform = self._normalize_waveform(noisy_waveform, source_sample_rate)

        conditioning = row.get("conditioning")
        conditioning_tensor = (
            torch.tensor(conditioning, dtype=torch.float32)
            if conditioning is not None
            else None
        )

        return {
            "Sample_id": row["sample_id"],
            "file_location": row.get("file_location", f"{row['sample_id']}.pt"),
            "Name": row.get("name", row["sample_id"]),
            "Conditioning": conditioning_tensor,
            "Clean Waveform": clean_waveform,
            "Noisy Waveform": noisy_waveform,
            "Noise Type": row.get("noise_type", "gaussian"),
            "Split": row["split"],
            "Class": row["class"],
            "Source Sample Rate": source_sample_rate,
            "Clean Path": str(clean_path),
            "Noisy Path": str(noisy_path) if noisy_path is not None else None,
        }
