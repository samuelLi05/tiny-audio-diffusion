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
import os
import random
import tempfile
import wave
from pathlib import Path
from typing import Any

import requests
import torch  # type: ignore[reportMissingImports]
import torchaudio  # type: ignore[reportMissingImports]


DEFAULT_CLASSES = ["bass", "guitar", "keyboard"]
DEFAULT_SPLIT_COUNTS = {"train": 105, "valid": 30, "test": 15}
DEFAULT_NOISE_RECIPES = ["gaussian", "uniform", "pink"]
DEFAULT_EXTRACTED_BOX_FOLDER_ID = os.environ.get("BOX_NSYNTH_EXTRACTED_FOLDER_ID", "377043004275")
DEFAULT_PREPROCESSED_BOX_FOLDER_ID = os.environ.get("BOX_NSYNTH_PREPROCESSED_FOLDER_ID", "377154702223")
BOX_API_BASE = "https://api.box.com/2.0"
BOX_UPLOAD_BASE = "https://upload.box.com/api/2.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic clean/noisy waveform tensors and metadata from NSynth."
    )
    parser.add_argument(
        "--nsynth-root",
        default=None,
        help="Local extracted NSynth root (contains nsynth-train/nsynth-valid/nsynth-test).",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Local output root for generated waveform tensors and metadata.",
    )
    parser.add_argument(
        "--box-extracted-folder-id",
        default=DEFAULT_EXTRACTED_BOX_FOLDER_ID,
        help="Reference Box folder ID for extracted NSynth source data.",
    )
    parser.add_argument(
        "--box-preprocessed-folder-id",
        default=DEFAULT_PREPROCESSED_BOX_FOLDER_ID,
        help="Reference Box folder ID for generated preprocessed data.",
    )
    parser.add_argument(
        "--box-access-token",
        default=os.environ.get("BOX_ACCESS_TOKEN") or os.environ.get("BOX_DEVELOPER_TOKEN"),
        help="Box OAuth access token or developer token required for Box API access.",
    )
    parser.add_argument(
        "--local-extracted-root",
        default=None,
        help=(
            "Optional local extracted NSynth root to use as source data immediately "
            "(contains nsynth-train/nsynth-valid/nsynth-test). "
            "Outputs are still uploaded to Box preprocessed folder."
        ),
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Instrument families to keep.",
    )
    parser.add_argument(
        "--pitch-min",
        type=int,
        default=48,
        help="Minimum MIDI pitch to keep from NSynth metadata.",
    )
    parser.add_argument(
        "--pitch-max",
        type=int,
        default=72,
        help="Maximum MIDI pitch to keep from NSynth metadata.",
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
    pitch_min: int | None = None,
    pitch_max: int | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    if pitch_min is not None and pitch_max is not None and pitch_min > pitch_max:
        raise ValueError(f"Invalid pitch range: min {pitch_min} is greater than max {pitch_max}")

    candidates = [
        (note_id, item)
        for note_id, item in examples.items()
        if item.get("instrument_family_str") == family
        and (pitch_min is None or int(item.get("pitch", -1)) >= pitch_min)
        and (pitch_max is None or int(item.get("pitch", -1)) <= pitch_max)
    ]
    candidates.sort(key=lambda x: x[0])

    rng = random.Random(stable_seed(seed, split, family))
    rng.shuffle(candidates)

    if len(candidates) < count:
        print(
            f"Warning: Not enough samples for class '{family}' in split '{split}'. "
            f"Requested {count}, but only {len(candidates)} are available after pitch filtering. "
            f"Proceeding with available samples."
        )
        return candidates

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


def load_wav_tensor(wav_path: Path) -> tuple[torch.Tensor, int]:
    """Load a PCM WAV into a float32 tensor shaped [channels, time]."""
    with wave.open(str(wav_path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)

    writable = bytearray(raw)

    if sample_width == 1:
        tensor = torch.frombuffer(writable, dtype=torch.uint8).to(torch.float32)
        tensor = (tensor - 128.0) / 128.0
    elif sample_width == 2:
        tensor = torch.frombuffer(writable, dtype=torch.int16).to(torch.float32)
        tensor = tensor / 32768.0
    elif sample_width == 4:
        tensor = torch.frombuffer(writable, dtype=torch.int32).to(torch.float32)
        tensor = tensor / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width ({sample_width} bytes): {wav_path}")

    tensor = tensor.reshape(-1, channels).transpose(0, 1).contiguous()
    return tensor, sample_rate


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


def box_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def box_list_folder_items(token: str, folder_id: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    limit = 1000
    offset = 0
    while True:
        response = requests.get(
            f"{BOX_API_BASE}/folders/{folder_id}/items",
            headers=box_headers(token),
            params={"limit": limit, "offset": offset, "fields": "id,name,type,size"},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        entries = payload.get("entries", [])
        items.extend(entries)
        if len(entries) < limit:
            break
        offset += limit
    return items


def box_find_child_folder(token: str, parent_folder_id: str, folder_name: str) -> dict[str, Any]:
    for item in box_list_folder_items(token, parent_folder_id):
        if item.get("type") == "folder" and item.get("name") == folder_name:
            return item
    raise FileNotFoundError(
        f"Could not find folder '{folder_name}' under Box folder id {parent_folder_id}."
    )


def box_find_child_file(token: str, parent_folder_id: str, file_name: str) -> dict[str, Any] | None:
    for item in box_list_folder_items(token, parent_folder_id):
        if item.get("type") == "file" and item.get("name") == file_name:
            return item
    return None


def box_download_file(token: str, file_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        f"{BOX_API_BASE}/files/{file_id}/content",
        headers=box_headers(token),
        stream=True,
        timeout=240,
    )
    response.raise_for_status()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def box_search_file_in_folder(token: str, folder_id: str, exact_file_name: str) -> dict[str, Any]:
    response = requests.get(
        f"{BOX_API_BASE}/search",
        headers=box_headers(token),
        params={
            "query": exact_file_name,
            "ancestor_folder_ids": folder_id,
            "type": "file",
            "limit": 50,
            "fields": "id,name,type,size",
        },
        timeout=120,
    )
    response.raise_for_status()
    entries = response.json().get("entries", [])
    for entry in entries:
        if entry.get("type") == "file" and entry.get("name") == exact_file_name:
            return entry
    raise FileNotFoundError(
        f"Could not locate file '{exact_file_name}' in Box folder id {folder_id} using search."
    )


def box_create_folder(token: str, parent_folder_id: str, folder_name: str) -> str:
    response = requests.post(
        f"{BOX_API_BASE}/folders",
        headers={**box_headers(token), "Content-Type": "application/json"},
        json={"name": folder_name, "parent": {"id": str(parent_folder_id)}},
        timeout=120,
    )
    if response.status_code == 409:
        conflict = response.json().get("context_info", {}).get("conflicts", {})
        return str(conflict.get("id"))
    response.raise_for_status()
    return str(response.json()["id"])


def box_get_or_create_folder_path(token: str, root_folder_id: str, relative_parts: list[str]) -> str:
    current = str(root_folder_id)
    for part in relative_parts:
        child = box_find_child_folder(token, current, part) if part else None
        if child is None:
            current = box_create_folder(token, current, part)
        else:
            current = str(child["id"])
    return current


def box_upload_file(token: str, folder_id: str, local_path: Path, remote_name: str) -> str:
    with local_path.open("rb") as file_handle:
        files = {
            "attributes": (
                None,
                json.dumps({"name": remote_name, "parent": {"id": str(folder_id)}}),
                "application/json",
            ),
            "file": (remote_name, file_handle, "application/octet-stream"),
        }
        response = requests.post(
            f"{BOX_UPLOAD_BASE}/files/content",
            headers=box_headers(token),
            files=files,
            timeout=300,
        )

    if response.status_code == 409:
        conflict = response.json().get("context_info", {}).get("conflicts", {})
        file_id = str(conflict.get("id"))
        with local_path.open("rb") as file_handle:
            files = {
                "attributes": (None, json.dumps({"name": remote_name}), "application/json"),
                "file": (remote_name, file_handle, "application/octet-stream"),
            }
            update_response = requests.post(
                f"{BOX_UPLOAD_BASE}/files/{file_id}/content",
                headers=box_headers(token),
                files=files,
                timeout=300,
            )
        update_response.raise_for_status()
        updated = update_response.json().get("entries", [])
        if not updated:
            raise RuntimeError(f"Failed to update Box file: {remote_name}")
        return str(updated[0]["id"])

    response.raise_for_status()
    created = response.json().get("entries", [])
    if not created:
        raise RuntimeError(f"Failed to upload Box file: {remote_name}")
    return str(created[0]["id"])


def build_box_subset_locally(
    token: str,
    extracted_folder_id: str,
    local_nsynth_root: Path,
    families: list[str],
    per_class: dict[str, int],
    seed: int,
    pitch_min: int,
    pitch_max: int,
) -> None:
    splits = ["train", "valid", "test"]
    for split in splits:
        split_name = f"nsynth-{split}"
        split_folder = box_find_child_folder(token, extracted_folder_id, split_name)
        split_folder_id = str(split_folder["id"])

        examples_file = box_find_child_file(token, split_folder_id, "examples.json")
        if examples_file is None:
            raise FileNotFoundError(
                f"Missing examples.json in Box folder '{split_name}' ({split_folder_id})."
            )

        audio_folder = box_find_child_folder(token, split_folder_id, "audio")
        audio_folder_id = str(audio_folder["id"])

        local_split_root = local_nsynth_root / split_name
        local_audio_root = local_split_root / "audio"
        local_split_root.mkdir(parents=True, exist_ok=True)
        local_audio_root.mkdir(parents=True, exist_ok=True)

        local_examples_path = local_split_root / "examples.json"
        box_download_file(token, str(examples_file["id"]), local_examples_path)

        examples = load_examples(local_split_root)
        requested_note_ids: set[str] = set()
        for family in families:
            selected = select_examples_for_class(
                examples=examples,
                family=family,
                count=per_class[split],
                split=split,
                seed=seed,
                pitch_min=pitch_min,
                pitch_max=pitch_max,
            )
            for note_id, _ in selected:
                requested_note_ids.add(note_id)

        print(
            f"Downloading {len(requested_note_ids)} wav files for split {split} "
            f"from Box extracted folder {split_folder_id}"
        )
        for note_id in sorted(requested_note_ids):
            wav_name = f"{note_id}.wav"
            wav_entry = box_search_file_in_folder(token, audio_folder_id, wav_name)
            local_wav_path = local_audio_root / wav_name
            box_download_file(token, str(wav_entry["id"]), local_wav_path)


def run_local_pipeline(
    args: argparse.Namespace,
    nsynth_root: Path,
    output_root: Path,
    include_box_fields: bool = False,
) -> tuple[Path, Path, Path]:
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
    first_clean_sample: torch.Tensor | None = None
    first_noisy_sample: torch.Tensor | None = None

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
                pitch_min=args.pitch_min,
                pitch_max=args.pitch_max,
            )

            for note_id, item in selected:
                wav_path = audio_root / f"{note_id}.wav"
                if not wav_path.exists():
                    raise FileNotFoundError(f"Missing wav file: {wav_path}")

                waveform, sr = load_wav_tensor(wav_path)
                if waveform.size(0) > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                waveform = maybe_resample(waveform, sr, args.sample_rate).to(torch.float32)

                row_seed = stable_seed(args.seed, split, family, note_id)
                row_rng = random.Random(row_seed)
                noise_type = row_rng.choice(args.noise_recipes)
                noise_gen = torch.Generator().manual_seed(row_seed)

                noisy_wave = add_noise(waveform, noise_type=noise_type, generator=noise_gen)

                if first_clean_sample is None:
                    first_clean_sample = waveform.clone()
                    first_noisy_sample = noisy_wave.clone()

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
                    "noise_type": noise_type,
                    "conditioning": conditioning,
                    "sample_rate": args.sample_rate,
                }
                if include_box_fields:
                    row["clean_waveform_id"] = None
                    row["noisy_waveform_id"] = None

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
    if first_clean_sample is None or first_noisy_sample is None:
        raise RuntimeError("Failed to capture sample tensors for shape reporting.")
    shape_report = {
        "sample_id": first["sample_id"],
        "clean_waveform_shape": list(first_clean_sample.shape),
        "noisy_waveform_shape": list(first_noisy_sample.shape),
        "conditioning_shape": (
            [len(first["conditioning"])] if first["conditioning"] is not None else None
        ),
        "total_rows": len(metadata_rows),
    }
    shape_path = output_root / "metadata" / "sample_shapes.json"
    with shape_path.open("w", encoding="utf-8") as handle:
        json.dump(shape_report, handle, indent=2)

    return metadata_path, splits_path, shape_path


def upload_outputs_to_box(
    token: str,
    output_root: Path,
    preprocessed_folder_id: str,
    include_qc: bool,
) -> None:
    folder_id_cache: dict[tuple[str, str], str] = {}
    uploaded_file_ids: dict[str, str] = {}

    def ensure_remote_parent(relative_parent_parts: list[str]) -> str:
        current = str(preprocessed_folder_id)
        for part in relative_parent_parts:
            cache_key = (current, part)
            if cache_key not in folder_id_cache:
                if part:
                    try:
                        child = box_find_child_folder(token, current, part)
                        folder_id_cache[cache_key] = str(child["id"])
                    except FileNotFoundError:
                        folder_id_cache[cache_key] = box_create_folder(token, current, part)
                else:
                    folder_id_cache[cache_key] = current
            current = folder_id_cache[cache_key]
        return current

    for local_file in output_root.rglob("*.pt"):
        relative = local_file.relative_to(output_root)
        parent_parts = list(relative.parts[:-1])
        remote_parent_id = ensure_remote_parent(parent_parts)
        file_id = box_upload_file(token, remote_parent_id, local_file, relative.name)
        uploaded_file_ids[relative.as_posix()] = file_id

    metadata_path = output_root / "metadata" / "metadata.jsonl"
    updated_rows: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)

            row["clean_waveform_id"] = uploaded_file_ids.get(f"waveforms/clean/{row['split']}/{row['class']}/{row['sample_id']}.pt")
            row["noisy_waveform_id"] = uploaded_file_ids.get(f"waveforms/noisy/{row['split']}/{row['class']}/{row['sample_id']}.pt")
            row["box_preprocessed_folder_id"] = str(preprocessed_folder_id)

            updated_rows.append(row)

    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in updated_rows:
            handle.write(json.dumps(row) + "\n")

    metadata_parent = ensure_remote_parent(["metadata"])
    box_upload_file(token, metadata_parent, metadata_path, "metadata.jsonl")
    box_upload_file(token, metadata_parent, output_root / "metadata" / "splits.json", "splits.json")
    box_upload_file(
        token,
        metadata_parent,
        output_root / "metadata" / "sample_shapes.json",
        "sample_shapes.json",
    )


def main() -> None:
    args = parse_args()
    families = list(args.classes)
    per_class = {
        "train": args.train_per_class,
        "valid": args.valid_per_class,
        "test": args.test_per_class,
    }

    local_nsynth_root = Path(args.nsynth_root).expanduser().resolve() if args.nsynth_root else None
    local_output_root = Path(args.output_root).expanduser().resolve() if args.output_root else None

    if local_nsynth_root is not None:
        if local_output_root is None:
            raise ValueError("--output-root is required when --nsynth-root is provided.")
        run_local_pipeline(
            args=args,
            nsynth_root=local_nsynth_root,
            output_root=local_output_root,
            include_box_fields=False,
        )
        print(f"Created waveform dataset under: {local_output_root}")
        print(f"Pitch filter applied: {args.pitch_min} to {args.pitch_max}")
        return

    if not args.box_access_token:
        raise ValueError(
            "--box-access-token (or BOX_ACCESS_TOKEN) is required for Box API access when not using --nsynth-root."
        )

    print(f"Using extracted Box folder id: {args.box_extracted_folder_id}")
    print(f"Using preprocessed Box folder id: {args.box_preprocessed_folder_id}")

    local_extracted_root = (
        Path(args.local_extracted_root).expanduser().resolve()
        if args.local_extracted_root
        else None
    )

    with tempfile.TemporaryDirectory(prefix="nsynth_box_work_") as temp_dir:
        temp_root = Path(temp_dir)
        temp_nsynth_root = temp_root / "extracted"
        temp_output_root = temp_root / "preprocessed"
        temp_nsynth_root.mkdir(parents=True, exist_ok=True)
        temp_output_root.mkdir(parents=True, exist_ok=True)

        if local_extracted_root is not None:
            for split in ["train", "valid", "test"]:
                split_root = local_extracted_root / f"nsynth-{split}"
                if not (split_root / "examples.json").exists():
                    raise FileNotFoundError(
                        f"Missing local extracted metadata: {split_root / 'examples.json'}"
                    )
                if not (split_root / "audio").exists():
                    raise FileNotFoundError(
                        f"Missing local extracted audio folder: {split_root / 'audio'}"
                    )
            source_nsynth_root = local_extracted_root
            print(f"Using local extracted source: {source_nsynth_root}")
        else:
            build_box_subset_locally(
                token=args.box_access_token,
                extracted_folder_id=str(args.box_extracted_folder_id),
                local_nsynth_root=temp_nsynth_root,
                families=families,
                per_class=per_class,
                seed=args.seed,
                pitch_min=args.pitch_min,
                pitch_max=args.pitch_max,
            )
            source_nsynth_root = temp_nsynth_root

        metadata_path, splits_path, shape_path = run_local_pipeline(
            args=args,
            nsynth_root=source_nsynth_root,
            output_root=temp_output_root,
            include_box_fields=True,
        )

        upload_outputs_to_box(
            token=args.box_access_token,
            output_root=temp_output_root,
            preprocessed_folder_id=str(args.box_preprocessed_folder_id),
            include_qc=args.export_qc_spectrograms,
        )

        print("Uploaded processed tensors and metadata to Box preprocessed folder.")
        print(f"Local temporary metadata file: {metadata_path}")
        print(f"Local temporary split file: {splits_path}")
        print(f"Local temporary sample shape file: {shape_path}")


if __name__ == "__main__":
    main()
