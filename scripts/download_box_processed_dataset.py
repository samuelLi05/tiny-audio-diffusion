"""Download a processed NSynth waveform dataset tree from Box.

Expected remote folder layout (single root folder ID):
  metadata/metadata.jsonl
  metadata/splits.json
  waveforms/clean/<split>/<class>/*.pt
  waveforms/noisy/<split>/<class>/*.pt
  qc/... (optional)

The script recursively downloads files and preserves the Box folder structure
under a local destination root.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from box_sdk_gen import BoxClient, BoxDeveloperTokenAuth  # type: ignore[reportMissingImports]
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_BOX_ACCESS_TOKEN = os.environ.get("BOX_ACCESS_TOKEN") or os.environ.get(
    "BOX_DEVELOPER_TOKEN"
)

FOLDER_ID = os.environ.get("FOLDER_ID") or os.environ.get("BOX_FOLDER_ID")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a processed NSynth waveform dataset tree from Box."
    )
    parser.add_argument(
        "--box-folder-id",
        required=FOLDER_ID is None,
        default=FOLDER_ID,
        help="Root Box folder ID containing metadata/ and waveforms/. If not provided, uses FOLDER_ID from .env.",
    )
    parser.add_argument(
        "--destination",
        default=os.environ.get(
            "BOX_NSYNTH_PROCESSED_DIR",
            str(PROJECT_ROOT / "data" / "nsynth_waveform_box"),
        ),
        help="Local destination root for downloaded files.",
    )
    parser.add_argument(
        "--box-access-token",
        default=DEFAULT_BOX_ACCESS_TOKEN,
        help=(
            "Box OAuth/developer token. Defaults to BOX_ACCESS_TOKEN or "
            "BOX_DEVELOPER_TOKEN."
        ),
    )
    parser.add_argument(
        "--exclude-qc",
        action="store_true",
        help="Skip downloading the qc/ subtree to reduce transfer size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local files. By default, existing files are skipped.",
    )
    return parser.parse_args()



def build_box_client(access_token: str) -> BoxClient:
    auth = BoxDeveloperTokenAuth(access_token)
    return BoxClient(auth)


def should_skip_relative_path(relative_path: Path, exclude_qc: bool) -> bool:
    if not exclude_qc:
        return False
    if not relative_path.parts:
        return False
    return relative_path.parts[0] == "qc"



def iter_folder_items(client: BoxClient, folder_id: str) -> Iterable:
    # Box API uses offset pagination; continue until we exhaust entries.
    offset = 0
    limit = 1000
    while True:
        items = client.folders.get_folder_items(folder_id, limit=limit, offset=offset)
        entries = items.entries
        for item in entries:
            yield item

        if not entries:
            break

        offset += len(entries)
        total_count = getattr(items, "total_count", None)
        if total_count is not None and offset >= total_count:
            break



def download_tree(
    client: BoxClient,
    folder_id: str,
    destination_root: Path,
    relative_root: Path,
    *,
    overwrite: bool,
    exclude_qc: bool,
) -> None:
    for item in iter_folder_items(client, folder_id):
        relative_path = relative_root / item.name
        if should_skip_relative_path(relative_path, exclude_qc=exclude_qc):
            continue

        target_path = destination_root / relative_path

        if item.type == "folder":
            target_path.mkdir(parents=True, exist_ok=True)
            download_tree(
                client=client,
                folder_id=item.id,
                destination_root=destination_root,
                relative_root=relative_path,
                overwrite=overwrite,
                exclude_qc=exclude_qc,
            )
            continue

        if item.type != "file":
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and not overwrite:
            print(f"Skipping existing file: {target_path}")
            continue

        print(f"Downloading {relative_path}")
        with target_path.open("wb") as out_file:
            file_content = client.downloads.download_file(item.id)

            if isinstance(file_content, (bytes, bytearray)):
                out_file.write(file_content)
                continue

            if hasattr(file_content, "read"):
                while True:
                    chunk = file_content.read(1024 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)
                continue

            raise TypeError(
                f"Unsupported download response type for {relative_path}: "
                f"{type(file_content).__name__}"
            )


def main() -> None:
    args = parse_args()

    if not args.box_access_token:
        raise ValueError(
            "Missing Box token. Set --box-access-token or BOX_ACCESS_TOKEN/BOX_DEVELOPER_TOKEN."
        )
    if not args.box_folder_id:
        raise ValueError("Missing Box folder ID. Set --box-folder-id or FOLDER_ID/BOX_FOLDER_ID.")

    destination_root = Path(args.destination).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    client = build_box_client(args.box_access_token)
    download_tree(
        client=client,
        folder_id=str(args.box_folder_id),
        destination_root=destination_root,
        relative_root=Path(""),
        overwrite=args.overwrite,
        exclude_qc=args.exclude_qc,
    )

    print(f"Downloaded dataset tree to: {destination_root}")


if __name__ == "__main__":
    main()
