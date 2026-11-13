"""Convert a SROIE style key information dataset to this project's JSONL format.

SROIE stores one JSON label file per receipt image, with the same file name
stem, holding the keys company, address, date and total. For example:

    {"company": "STARBUCKS STORE #10208", "address": "...", "date": "14/03/2015", "total": "4.95"}

The label folder layout differs between copies of the dataset, so the image
folder and the label folder are passed in separately instead of being guessed.
SROIE dates are day-first, so evaluate with --day-first.
"""
import json
import shutil
from pathlib import Path
from typing import Dict

from .data import write_jsonl

KEY_MAP = {"company": "vendor", "date": "date", "total": "total"}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
LABEL_SUFFIXES = (".txt", ".json")

def _index_images(images_dir: Path) -> Dict[str, Path]:
    """Map lowercase file name to the real path, so X002.JPG is found as X002.jpg.

    The real name is kept, because case-insensitive file systems (Windows,
    macOS) would otherwise hand back whatever spelling was asked for.
    """
    return {p.name.lower(): p for p in images_dir.iterdir() if p.is_file()}

def _find_image(index: Dict[str, Path], stem: str):
    for suffix in IMAGE_SUFFIXES:
        found = index.get(f"{stem}{suffix}".lower())
        if found is not None:
            return found
    return None

def convert_sroie(images_dir, labels_dir, out_dir) -> Dict[str, int]:
    """Copy matched images into out_dir/images and write out_dir/data.jsonl.

    Receipts are skipped, not fatal, when the label is not valid JSON, lacks a
    needed key, or has no matching image. The returned counts say how many of
    each, so a bad download shows up immediately.
    """
    images_dir, labels_dir, out_dir = Path(images_dir), Path(labels_dir), Path(out_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"image folder not found: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"label folder not found: {labels_dir}")

    label_files = sorted(
        p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() in LABEL_SUFFIXES
    )
    summary = {
        "converted": 0,
        "skipped_bad_json": 0,
        "skipped_missing_keys": 0,
        "skipped_no_image": 0,
    }
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    image_index = _index_images(images_dir)
    rows = []
    for label_path in label_files:
        try:
            data = json.loads(label_path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            summary["skipped_bad_json"] += 1
            continue
        if not isinstance(data, dict) or any(key not in data for key in KEY_MAP):
            summary["skipped_missing_keys"] += 1
            continue
        image_path = _find_image(image_index, label_path.stem)
        if image_path is None:
            summary["skipped_no_image"] += 1
            continue
        name = f"images/{image_path.name}"
        shutil.copyfile(image_path, out_dir / name)
        rows.append({"image": name, **{new: data[old] for old, new in KEY_MAP.items()}})
        summary["converted"] += 1

    if not rows:
        raise ValueError(f"nothing converted from {labels_dir}: {summary}")
    write_jsonl(out_dir / "data.jsonl", rows)
    return summary
