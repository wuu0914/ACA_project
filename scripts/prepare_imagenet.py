#!/usr/bin/env python3
"""Prepare the 50,000-image ILSVRC2012 validation set for benchmark.py."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "models/dataset/imagenet"
DOWNLOAD_DIR = ROOT / "deps/.work/imagenet"

COMPETITION = "imagenet-object-localization-challenge"
KAGGLE_ARCHIVE = "imagenet_object_localization_patched2019.tar.gz"
KAGGLE_LABELS = "LOC_val_solution.csv"
KAGGLE_SYNSETS = "LOC_synset_mapping.txt"
EXPECTED_IMAGES = 50_000
EXPECTED_SYNSETS = 1_000


class DatasetError(RuntimeError):
    """An actionable dataset preparation error."""


def path_from_argument(value: str | None) -> Path | None:
    return Path(value).expanduser().resolve() if value else None


def read_synsets(path: Path) -> list[str]:
    # LOC_synset_mapping.txt starts every line with the WordNet ID.  A plain
    # one-ID-per-line synset file is accepted as well.
    synsets = [line.split(maxsplit=1)[0] for line in path.read_text().splitlines() if line]
    if len(synsets) != EXPECTED_SYNSETS or len(set(synsets)) != EXPECTED_SYNSETS:
        raise DatasetError(f"{path} must contain {EXPECTED_SYNSETS} unique synsets")
    invalid = [
        synset
        for synset in synsets
        if len(synset) != 9 or synset[0] != "n" or not synset[1:].isdigit()
    ]
    if invalid:
        raise DatasetError(f"{path} contains an invalid WordNet synset ID")
    return synsets


def read_labels(path: Path, synsets: list[str]) -> list[tuple[str, int, str]]:
    synset_to_label = {synset: label for label, synset in enumerate(synsets)}
    entries: list[tuple[str, int, str]] = []

    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            try:
                image_id = row["ImageId"]
                synset = row["PredictionString"].split()[0]
                label = synset_to_label[synset]
            except (KeyError, IndexError) as error:
                raise DatasetError(f"invalid ImageNet label row in {path}") from error
            if not image_id.startswith("ILSVRC2012_val_") or not image_id.removeprefix(
                "ILSVRC2012_val_"
            ).isdigit():
                raise DatasetError(f"invalid validation image ID in {path}: {image_id}")
            entries.append((f"{image_id}.JPEG", label, synset))

    entries.sort()
    filenames = {filename for filename, _, _ in entries}
    if len(entries) != EXPECTED_IMAGES or len(filenames) != EXPECTED_IMAGES:
        raise DatasetError(f"{path} must describe {EXPECTED_IMAGES} unique images")
    return entries


def validate_dataset(root: Path) -> None:
    synsets_path = root / "aux/synsets.txt"
    labels_path = root / "aux/val.txt"
    if not synsets_path.is_file() or not labels_path.is_file():
        raise DatasetError(f"{root} is missing aux/synsets.txt or aux/val.txt")

    synsets = read_synsets(synsets_path)
    lines = [line.split() for line in labels_path.read_text().splitlines() if line]
    if len(lines) != EXPECTED_IMAGES:
        raise DatasetError(f"{labels_path} must contain {EXPECTED_IMAGES} entries")

    seen: set[str] = set()
    for fields in lines:
        if len(fields) != 2:
            raise DatasetError(f"invalid entry in {labels_path}: {' '.join(fields)}")
        filename, label_text = fields
        try:
            label = int(label_text)
        except ValueError as error:
            raise DatasetError(f"invalid label in {labels_path}: {label_text}") from error
        if not 0 <= label < len(synsets):
            raise DatasetError(f"invalid label in {labels_path}: {label_text}")
        synset = synsets[label]
        if filename in seen:
            raise DatasetError(f"duplicate image in {labels_path}: {filename}")
        seen.add(filename)
        if not (root / "val" / synset / filename).is_file():
            raise DatasetError(f"missing validation image: {synset}/{filename}")

    print(f"OK ImageNet: {EXPECTED_IMAGES} validation images in {root}")


def require_kaggle() -> str:
    kaggle = shutil.which("kaggle")
    if kaggle:
        return kaggle
    raise DatasetError(
        "Kaggle CLI is required when local files are unavailable.\n"
        "Install and authenticate it with:\n"
        "  uv tool install kaggle\n"
        "  kaggle auth login\n"
        f"Then accept the rules at https://www.kaggle.com/competitions/{COMPETITION}/rules"
    )


def unwrap_kaggle_download(filename: str) -> Path | None:
    direct = DOWNLOAD_DIR / filename
    if direct.is_file():
        return direct

    # Some Kaggle CLI versions wrap a requested file in a zip archive.
    for wrapper in DOWNLOAD_DIR.glob("*.zip"):
        try:
            with zipfile.ZipFile(wrapper) as archive:
                members = [name for name in archive.namelist() if Path(name).name == filename]
                if len(members) != 1:
                    continue
                with archive.open(members[0]) as source, direct.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                return direct
        except zipfile.BadZipFile:
            continue
    return None


def kaggle_download(filename: str) -> Path:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    existing = unwrap_kaggle_download(filename)
    if existing:
        print(f"Using cached {existing.relative_to(ROOT)}")
        return existing

    kaggle = require_kaggle()
    print(f"Downloading {filename} from Kaggle")
    command = [
        kaggle,
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-f",
        filename,
        "-p",
        str(DOWNLOAD_DIR),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise DatasetError(
            "Kaggle download failed. Run 'kaggle auth login', accept the competition "
            "rules in the browser, and retry."
        ) from error

    downloaded = unwrap_kaggle_download(filename)
    if not downloaded:
        raise DatasetError(f"Kaggle reported success but {filename} was not found")
    return downloaded


def find_metadata(explicit: Path | None, archive: Path | None, filename: str) -> Path:
    candidates = [explicit]
    if archive:
        candidates.append(archive.parent / filename)
    candidates.append(DOWNLOAD_DIR / filename)
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    return kaggle_download(filename)


def extract_validation_images(
    archive_path: Path,
    output: Path,
    entries: list[tuple[str, int, str]],
) -> None:
    expected = {filename: synset for filename, _, synset in entries}
    extracted: set[str] = set()

    print(f"Extracting {EXPECTED_IMAGES} validation images from {archive_path}")
    try:
        archive = tarfile.open(archive_path, mode="r:*")
    except (tarfile.TarError, OSError) as error:
        raise DatasetError(f"cannot open ImageNet archive: {archive_path}") from error

    with archive:
        for member in archive:
            if not member.isfile():
                continue
            filename = PurePosixPath(member.name).name
            synset = expected.get(filename)
            if not synset:
                continue
            if filename in extracted:
                raise DatasetError(f"duplicate image in archive: {filename}")

            destination = output / "val" / synset / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise DatasetError(f"cannot read image from archive: {member.name}")
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            extracted.add(filename)
            if len(extracted) % 5_000 == 0:
                print(f"  extracted {len(extracted)}/{EXPECTED_IMAGES}")

    if len(extracted) != EXPECTED_IMAGES:
        missing = next(filename for filename in expected if filename not in extracted)
        raise DatasetError(
            f"archive contains {len(extracted)}/{EXPECTED_IMAGES} validation images; "
            f"first missing image: {missing}"
        )


def prepare_dataset(
    output: Path,
    archive: Path | None,
    labels: Path | None,
    synsets: Path | None,
) -> None:
    if output.exists():
        validate_dataset(output)
        return

    # Download the small metadata files first so authentication or competition
    # access errors happen before the large image archive download starts.
    labels = find_metadata(labels, archive, KAGGLE_LABELS)
    synsets = find_metadata(synsets, archive, KAGGLE_SYNSETS)
    if archive is None:
        archive = kaggle_download(KAGGLE_ARCHIVE)
    if not archive.is_file():
        raise DatasetError(f"ImageNet archive does not exist: {archive}")

    synset_list = read_synsets(synsets)
    entries = read_labels(labels, synset_list)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".imagenet-", dir=output.parent))
    try:
        extract_validation_images(archive, staging, entries)
        aux = staging / "aux"
        aux.mkdir()
        (aux / "synsets.txt").write_text("\n".join(synset_list) + "\n")
        (aux / "val.txt").write_text(
            "".join(f"{filename} {label}\n" for filename, label, _ in entries)
        )
        validate_dataset(staging)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"ImageNet is ready at {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate without downloading")
    parser.add_argument(
        "--output",
        type=Path,
        default=path_from_argument(os.getenv("IMAGENET_ROOT")) or DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=path_from_argument(os.getenv("IMAGENET_ARCHIVE")),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=path_from_argument(os.getenv("IMAGENET_LABELS")),
    )
    parser.add_argument(
        "--synsets",
        type=Path,
        default=path_from_argument(os.getenv("IMAGENET_SYNSETS")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.check:
            validate_dataset(args.output.resolve())
        else:
            prepare_dataset(
                args.output.resolve(),
                args.archive.resolve() if args.archive else None,
                args.labels.resolve() if args.labels else None,
                args.synsets.resolve() if args.synsets else None,
            )
    except DatasetError as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
