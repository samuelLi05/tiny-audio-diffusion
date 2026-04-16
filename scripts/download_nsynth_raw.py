"""Download the raw NSynth split archives into immutable storage.

This script fetches the official NSynth json/wav tarballs and writes them to a
destination directory that can live inside a Box-synced folder. It does not
extract or modify the archives, which keeps the raw files immutable.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

from box_sdk_gen import BoxClient, BoxDeveloperTokenAuth  # type: ignore[reportMissingImports]


NSYNTH_ARCHIVES = {
    "train": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",
    "valid": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",
    "test": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
}

DEFAULT_BOX_FOLDER_ID = os.environ.get("BOX_NSYNTH_FOLDER_ID", "376635839098")
DEFAULT_BOX_ACCESS_TOKEN = os.environ.get("BOX_ACCESS_TOKEN") or os.environ.get(
    "BOX_DEVELOPER_TOKEN"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the raw NSynth split archives into a destination folder."
    )
    parser.add_argument(
        "--destination",
        default=os.environ.get(
            "BOX_NSYNTH_RAW_DIR",
            r"C:\Users\hoult\Box\Final_Project_Data\nsynth\raw",
        ),
        help=(
            "Local cache directory for the raw archives. When Box upload is enabled, "
            "this directory is only used as a temporary staging area."
        ),
    )
    parser.add_argument(
        "--box-folder-id",
        default=DEFAULT_BOX_FOLDER_ID,
        help="Box folder ID that will receive the uploaded archives.",
    )
    parser.add_argument(
        "--box-access-token",
        default=DEFAULT_BOX_ACCESS_TOKEN,
        help=(
            "Box OAuth access token or developer token. "
            "Defaults to BOX_ACCESS_TOKEN or BOX_DEVELOPER_TOKEN."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=sorted(NSYNTH_ARCHIVES),
        help="NSynth splits to download.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Skipping existing file: {destination}")
        return

    print(f"Downloading {url} -> {destination}")
    with urlopen(url) as response, destination.open("wb") as output_file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output_file.write(chunk)


def build_box_client(access_token: str) -> BoxClient:
    auth = BoxDeveloperTokenAuth(token=access_token)
    return BoxClient(auth=auth)


def box_folder_contains_file(client: BoxClient, folder_id: str, file_name: str, file_size: int) -> bool:
    items = client.folders.get_folder_items(folder_id).entries
    for item in items:
        if item.type == "file" and item.name == file_name and item.size == file_size:
            return True
    return False


def upload_file_to_box(client: BoxClient, folder_id: str, path: Path, file_name: str) -> None:
    file_size = path.stat().st_size
    if box_folder_contains_file(client, folder_id, file_name, file_size):
        print(f"Skipping Box upload for existing file: {file_name}")
        return

    print(f"Uploading {file_name} to Box folder {folder_id} using Box chunked upload")

    with path.open("rb") as file_handle:
        uploaded_file = client.chunked_uploads.upload_big_file(
            file_handle,
            file_name,
            file_size,
            folder_id,
        )
    print(
        f"Uploaded to Box: {uploaded_file.name} (file id {uploaded_file.id}, size {uploaded_file.size})"
    )


def iter_requested_archives(splits: Iterable[str]):
    for split in splits:
        yield split, NSYNTH_ARCHIVES[split]


def main() -> None:
    args = parse_args()
    destination_root = Path(args.destination).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)

    print(f"Destination root: {destination_root}")
    print("Raw archives will be written without extraction or post-processing.")

    box_client = build_box_client(args.box_access_token) if args.box_access_token else None

    with tempfile.TemporaryDirectory(prefix="nsynth_raw_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for split, url in iter_requested_archives(args.splits):
            target = destination_root / Path(url).name
            cached_copy = temp_dir / target.name
            print(f"Preparing {split} split")
            download_file(url, cached_copy)
            if box_client and args.box_folder_id:
                upload_file_to_box(
                    client=box_client,
                    folder_id=str(args.box_folder_id),
                    path=cached_copy,
                    file_name=target.name,
                )
            else:
                cached_copy.replace(target)

    print("Done.")


if __name__ == "__main__":
    main()