"""Create a leakage-aware YOLO dataset copy without changing source data."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_PATH = PROJECT_ROOT.parent / "Indian-Sign-Language.v1i.yolov8.zip"
SOURCE_SPLITS = ("train", "valid", "test")
OUTPUT_SPLITS = ("train", "valid", "test")
SPLIT_RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}
PHASH_MAX_DISTANCE = 2
BOX_TOLERANCE = 1e-6
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PREFIX_PATTERN = re.compile(r"^(?:(?:adjusted|flipped|original|bnw|rotated)_)+", re.I)


@dataclass
class Sample:
    archive_image: str
    archive_label: str
    source_split: str
    image_bytes: bytes
    label_bytes: bytes
    classes: set[str]
    file_hash: str
    perceptual_hash: int
    source_key: str
    corrected_label: bytes = b""
    corrected_boxes: int = 0
    corrected_boxes_by_class: dict[str, int] = field(default_factory=dict)


class UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def class_names_from_yaml(data_yaml: bytes) -> list[str]:
    text = data_yaml.decode("utf-8")
    match = re.search(r"(?m)^names:\s*(\[.*\])\s*$", text)
    if not match:
        raise ValueError("Expected a one-line `names: [...]` list in data.yaml")
    names = ast.literal_eval(match.group(1))
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("The data.yaml class names must be a list of strings")
    return names


def source_key(archive_image: str) -> str:
    stem = PurePosixPath(archive_image).stem.split(".rf.", 1)[0]
    while True:
        stripped = PREFIX_PATTERN.sub("", stem)
        if stripped == stem:
            break
        stem = stripped
    return re.sub(r"[^a-z0-9]+", "", stem.casefold())


def perceptual_hash(image_bytes: bytes) -> int:
    with Image.open(io.BytesIO(image_bytes)) as image:
        pixels = np.asarray(
            image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
    coefficients = cv2.dct(pixels)[:8, :8].reshape(-1)
    median = float(np.median(coefficients[1:]))
    value = 0
    for bit in coefficients[1:] > median:
        value = (value << 1) | int(bit)
    return value


def parse_annotation(
    raw: bytes, class_names: list[str], archive_label: str
) -> tuple[bytes, set[str], int, dict[str, int]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 annotation: {archive_label}") from exc

    classes: set[str] = set()
    corrected_by_class: Counter[str] = Counter()
    corrected_lines: list[str] = []
    adjusted = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Malformed annotation: {archive_label}:{line_number}")
        try:
            class_id = int(parts[0])
            center_x, center_y, width, height = map(float, parts[1:5])
        except ValueError as exc:
            raise ValueError(
                f"Malformed annotation: {archive_label}:{line_number}"
            ) from exc
        if not 0 <= class_id < len(class_names):
            raise ValueError(
                f"Class ID out of range: {archive_label}:{line_number} ({class_id})"
            )
        if not all(map(math.isfinite, (center_x, center_y, width, height))):
            raise ValueError(f"Non-finite box: {archive_label}:{line_number}")
        if width <= 0 or height <= 0:
            raise ValueError(f"Non-positive box: {archive_label}:{line_number}")

        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        needs_clip = (
            x1 < -BOX_TOLERANCE
            or y1 < -BOX_TOLERANCE
            or x2 > 1 + BOX_TOLERANCE
            or y2 > 1 + BOX_TOLERANCE
            or center_x < 0
            or center_x > 1
            or center_y < 0
            or center_y > 1
        )
        if needs_clip:
            adjusted += 1
            corrected_by_class[class_names[class_id]] += 1
            x1 = min(1.0, max(0.0, x1))
            y1 = min(1.0, max(0.0, y1))
            x2 = min(1.0, max(0.0, x2))
            y2 = min(1.0, max(0.0, y2))
            if x2 - x1 <= BOX_TOLERANCE or y2 - y1 <= BOX_TOLERANCE:
                raise ValueError(
                    f"Box collapses after clipping: {archive_label}:{line_number}"
                )
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
        classes.add(class_names[class_id])
        if needs_clip:
            corrected_lines.append(
                " ".join(
                    [
                        str(class_id),
                        *(
                            format(value, ".10f").rstrip("0").rstrip(".")
                            for value in (center_x, center_y, width, height)
                        ),
                    ]
                )
            )
        else:
            corrected_lines.append(line)

    ending = "\n" if text.endswith(("\n", "\r")) else ""
    if adjusted == 0:
        return raw, classes, 0, {}
    return (
        ("\n".join(corrected_lines) + ending).encode("utf-8"),
        classes,
        adjusted,
        dict(corrected_by_class),
    )


def load_samples(archive: zipfile.ZipFile, class_names: list[str]) -> list[Sample]:
    expected: dict[str, str] = {}
    for item in archive.infolist():
        path = item.filename.replace("\\", "/")
        if item.is_dir():
            continue
        if not any(path.startswith(f"{split}/") for split in SOURCE_SPLITS):
            continue
        pure_path = PurePosixPath(path)
        if "/images/" not in path or pure_path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        label_path = path.replace("/images/", "/labels/", 1).rsplit(".", 1)[0] + ".txt"
        if label_path in expected.values():
            raise ValueError(f"Duplicate image basename/label path in archive: {label_path}")
        expected[path] = label_path

    samples: list[Sample] = []
    for image_path, label_path in sorted(expected.items()):
        try:
            image_bytes = archive.read(image_path)
            label_bytes = archive.read(label_path)
        except KeyError as exc:
            raise ValueError(f"Image/annotation pair is missing: {exc}") from exc
        corrected, classes, box_count, corrected_by_class = parse_annotation(
            label_bytes, class_names, label_path
        )
        parts = PurePosixPath(image_path).parts
        samples.append(
            Sample(
                archive_image=image_path,
                archive_label=label_path,
                source_split=parts[0],
                image_bytes=image_bytes,
                label_bytes=label_bytes,
                classes=classes,
                file_hash=hashlib.sha256(image_bytes).hexdigest(),
                perceptual_hash=perceptual_hash(image_bytes),
                source_key=source_key(image_path),
                corrected_label=corrected,
                corrected_boxes=box_count,
                corrected_boxes_by_class=corrected_by_class,
            )
        )
    return samples


def duplicate_groups(samples: list[Sample]) -> tuple[list[list[int]], int]:
    groups = UnionFind(len(samples))
    first_by_key: dict[str, int] = {}
    first_by_sha: dict[str, int] = {}
    first_by_phash: dict[int, int] = {}
    for index, sample in enumerate(samples):
        for key, mapping in (
            (sample.source_key, first_by_key),
            (sample.file_hash, first_by_sha),
            (f"{sample.perceptual_hash:016x}", first_by_phash),
        ):
            if key in mapping:
                groups.union(index, mapping[key])
            else:
                mapping[key] = index

    cross_split_candidate_pairs = 0
    candidate_rows: list[tuple[int, int, int]] = []
    by_split: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_split[sample.source_split].append(index)

    packed_hashes = np.asarray([sample.perceptual_hash for sample in samples], dtype=np.uint64)
    byte_popcount = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)
    split_pairs = (("train", "valid"), ("train", "test"), ("valid", "test"))
    for left_split, right_split in split_pairs:
        left_indices = by_split[left_split]
        right_indices = by_split[right_split]
        right_array = np.asarray(right_indices, dtype=np.int64)
        for offset in range(0, len(left_indices), 128):
            left_array = np.asarray(left_indices[offset : offset + 128], dtype=np.int64)
            xor = np.bitwise_xor(
                packed_hashes[left_array, None], packed_hashes[None, right_array]
            )
            distances = byte_popcount[
                xor.view(np.uint8).reshape(xor.shape + (8,))
            ].sum(axis=2)
            for local_left, local_right in np.argwhere(
                distances <= PHASH_MAX_DISTANCE
            ):
                left = int(left_array[local_left])
                right = int(right_array[local_right])
                cross_split_candidate_pairs += 1
                candidate_rows.append(
                    (left, right, int(distances[local_left, local_right]))
                )
                groups.union(left, right)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(samples)):
        members[groups.find(index)].append(index)
    components = sorted(members.values(), key=lambda items: min(items))
    return components, candidate_rows


def assign_splits(
    samples: list[Sample], components: list[list[int]], seed: int
) -> list[str]:
    class_totals = Counter(
        class_name for sample in samples for class_name in sample.classes
    )
    target_class = {
        split: {
            class_name: class_totals[class_name] * ratio
            for class_name in class_totals
        }
        for split, ratio in SPLIT_RATIOS.items()
    }
    total_target = {
        split: len(samples) * ratio for split, ratio in SPLIT_RATIOS.items()
    }

    group_rows = []
    for group_id, members in enumerate(components, start=1):
        class_counts = Counter(
            class_name
            for member in members
            for class_name in samples[member].classes
        )
        rarity = sum(
            count / max(class_totals[class_name], 1)
            for class_name, count in class_counts.items()
        )
        group_rows.append(
            (group_id, members, class_counts, rarity, len(members))
        )

    randomizer = random.Random(seed)
    tie_breaker = {row[0]: randomizer.random() for row in group_rows}
    group_rows.sort(
        key=lambda row: (-row[3], -row[4], tie_breaker[row[0]])
    )

    current_class = {
        split: Counter() for split in OUTPUT_SPLITS
    }
    current_total = Counter()
    assignments: dict[int, str] = {}
    for group_id, members, class_counts, _, group_size in group_rows:
        candidates = []
        for split in OUTPUT_SPLITS:
            weighted_class_deficit = 0.0
            group_class_total = max(sum(class_counts.values()), 1)
            for class_name, count in class_counts.items():
                target = max(target_class[split][class_name], 1.0)
                remaining = (
                    target_class[split][class_name]
                    - current_class[split][class_name]
                ) / target
                weighted_class_deficit += count * remaining
            class_deficit = weighted_class_deficit / group_class_total
            total_deficit = (
                total_target[split] - current_total[split]
            ) / max(total_target[split], 1.0)
            score = 0.75 * class_deficit + 0.25 * total_deficit
            candidates.append((-score, tie_breaker[group_id], split))
        _, _, chosen = min(candidates)
        assignments[group_id] = chosen
        current_total[chosen] += group_size
        current_class[chosen].update(class_counts)

    result = [""] * len(samples)
    for group_id, members, _, _, _ in group_rows:
        for member in members:
            result[member] = assignments[group_id]
    if any(not split for split in result):
        raise RuntimeError("At least one sample was not assigned to a split")
    return result


def ensure_safe_archive_paths(samples: list[Sample]) -> None:
    paths = set()
    for sample in samples:
        for name in (sample.archive_image, sample.archive_label):
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or name in paths:
                raise ValueError(f"Unsafe or duplicate archive path: {name}")
            paths.add(name)


def write_dataset(
    output_dir: Path,
    samples: list[Sample],
    class_names: list[str],
    components: list[list[int]],
    assigned_splits: list[str],
    duplicate_pairs: list[tuple[int, int, int]],
    seed: int,
) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    try:
        group_by_member = {
            member: group_id
            for group_id, component in enumerate(components, start=1)
            for member in component
        }
        group_splits = {
            group_id: assigned_splits[members[0]]
            for group_id, members in enumerate(components, start=1)
        }
        class_images = {split: Counter() for split in OUTPUT_SPLITS}
        class_boxes = {split: Counter() for split in OUTPUT_SPLITS}
        group_source_splits: dict[int, set[str]] = defaultdict(set)
        group_classes: dict[int, set[str]] = defaultdict(set)
        for index, sample in enumerate(samples):
            split = assigned_splits[index]
            group_id = group_by_member[index]
            group_source_splits[group_id].add(sample.source_split)
            group_classes[group_id].update(sample.classes)
            for class_name in sample.classes:
                class_images[split][class_name] += 1
            for line in sample.corrected_label.decode("utf-8").splitlines():
                if line:
                    class_boxes[split][class_names[int(line.split()[0])]] += 1

            image_rel = PurePosixPath(sample.archive_image)
            label_rel = PurePosixPath(sample.archive_label)
            image_destination = (
                output_dir / split / "images" / image_rel.name
            )
            label_destination = (
                output_dir / split / "labels" / label_rel.name
            )
            image_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            image_destination.write_bytes(sample.image_bytes)
            label_destination.write_bytes(sample.corrected_label)

        names_text = json.dumps(class_names, ensure_ascii=False)
        yaml_text = (
            "# Prepared copy; paths are relative to this file.\n"
            "path: .\n"
            "train: train/images\n"
            "val: valid/images\n"
            "test: test/images\n\n"
            f"nc: {len(class_names)}\n"
            f"names: {names_text}\n\n"
            "roboflow:\n"
            "  workspace: ashwin-perumal-s-r\n"
            "  project: indian-sign-language-fkx1f-wlm3w-44qhl\n"
            "  version: 1\n"
            "  license: CC BY 4.0\n"
            "  url: https://universe.roboflow.com/ashwin-perumal-s-r/"
            "indian-sign-language-fkx1f-wlm3w-44qhl/dataset/1\n"
        )
        (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")

        with (output_dir / "leakage_groups.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "group_id",
                    "assigned_split",
                    "image_count",
                    "source_splits",
                    "class_names",
                    "archive_image_paths",
                ]
            )
            for group_id, members in enumerate(components, start=1):
                writer.writerow(
                    [
                        group_id,
                        group_splits[group_id],
                        len(members),
                        ",".join(sorted(group_source_splits[group_id])),
                        ",".join(sorted(group_classes[group_id])),
                        ";".join(samples[index].archive_image for index in members),
                    ]
                )

        with (output_dir / "leakage_candidates.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "left_source_split",
                    "left_archive_image",
                    "left_classes",
                    "right_source_split",
                    "right_archive_image",
                    "right_classes",
                    "phash_hamming_distance",
                    "assigned_split",
                ]
            )
            for left, right, distance in duplicate_pairs:
                writer.writerow(
                    [
                        samples[left].source_split,
                        samples[left].archive_image,
                        ",".join(sorted(samples[left].classes)),
                        samples[right].source_split,
                        samples[right].archive_image,
                        ",".join(sorted(samples[right].classes)),
                        distance,
                        assigned_splits[left],
                    ]
                )

        restored = []
        for sample in samples:
            if sample.source_split == "train":
                extracted_image = (
                    PROJECT_ROOT / "train" / "images" / PurePosixPath(sample.archive_image).name
                )
                if not extracted_image.exists():
                    restored.append(sample.archive_image)
        corrected_total = sum(sample.corrected_boxes for sample in samples)
        original_counts = Counter(sample.source_split for sample in samples)
        output_counts = Counter(assigned_splits)
        report = {
            "source_archive": str(ARCHIVE_PATH),
            "source_archive_sha256": hashlib.sha256(ARCHIVE_PATH.read_bytes()).hexdigest(),
            "source_splits": dict(original_counts),
            "output_splits": dict(output_counts),
            "images_prepared": len(samples),
            "class_names": class_names,
            "class_image_counts_by_output_split": {
                split: dict(sorted(counter.items()))
                for split, counter in class_images.items()
            },
            "class_box_counts_by_output_split": {
                split: dict(sorted(counter.items()))
                for split, counter in class_boxes.items()
            },
            "duplicate_group_count": len(components),
            "groups_with_multiple_images": sum(
                len(members) > 1 for members in components
            ),
            "cross_split_near_duplicate_candidate_pairs": len(duplicate_pairs),
            "near_duplicate_candidate_label_conflict_pairs": sum(
                not (samples[left].classes & samples[right].classes)
                for left, right, _ in duplicate_pairs
            ),
            "near_duplicate_method": (
                "Union samples sharing normalized source filename, exact SHA-256, "
                "or 64-bit grayscale pHash Hamming distance <= "
                f"{PHASH_MAX_DISTANCE} across the source train/valid/test splits."
            ),
            "corrected_out_of_bounds_boxes": corrected_total,
            "corrected_out_of_bounds_boxes_by_class": dict(
                sorted(
                    sum(
                        (
                            Counter(sample.corrected_boxes_by_class)
                            for sample in samples
                        ),
                        Counter(),
                    ).items()
                )
            ),
            "box_correction": (
                "Clamp each normalized xyxy box to [0,1], then recalculate xywh; "
                "source annotations are unchanged."
            ),
            "restored_from_archive_into_copy": restored,
            "random_seed": seed,
            "split_ratios_requested": SPLIT_RATIOS,
            "split_image_counts": dict(output_counts),
            "class_images_by_output_split": {
                split: dict(sorted(counter.items()))
                for split, counter in class_images.items()
            },
            "source_license": "CC BY 4.0 (per included Roboflow metadata)",
            "source_project": (
                "https://universe.roboflow.com/ashwin-perumal-s-r/"
                "indian-sign-language-fkx1f-wlm3w-44qhl/dataset/1"
            ),
            "note": (
                "Image-level labels and gestures have not been validated by an "
                "ISL-competent reviewer. No training or model evaluation was run."
            ),
        }
        (output_dir / "preparation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(output_dir)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "prepared",
        help="New output directory; existing directories are never overwritten.",
    )
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    if not ARCHIVE_PATH.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {ARCHIVE_PATH}")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"Archive CRC check failed for {bad_member}")
        yaml_candidates = [
            name
            for name in archive.namelist()
            if PurePosixPath(name.replace("\\", "/")).name == "data.yaml"
        ]
        if not yaml_candidates:
            raise ValueError("Archive contains no data.yaml")
        class_names = class_names_from_yaml(archive.read(yaml_candidates[0]))
        samples = load_samples(archive, class_names)

    ensure_safe_archive_paths(samples)
    if len(samples) != 6770:
        raise ValueError(
            f"Expected 6770 archived images, found {len(samples)}; stopping."
        )

    print(f"Loaded and checked {len(samples)} archive image/label pairs.", flush=True)
    components, candidate_pairs = duplicate_groups(samples)
    print(
        f"Found {len(components)} leakage groups; "
        f"{len(candidate_pairs)} cross-source-split pHash candidate pairs.",
        flush=True,
    )
    assigned_splits = assign_splits(samples, components, args.seed)
    write_dataset(
        output_dir,
        samples,
        class_names,
        components,
        assigned_splits,
        candidate_pairs,
        args.seed,
    )
    print(f"Prepared dataset copy: {output_dir}", flush=True)
    print(f"Preparation report: {output_dir / 'preparation_report.json'}")
    print(f"Leakage-group manifest: {output_dir / 'leakage_groups.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        raise
