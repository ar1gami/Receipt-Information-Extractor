"""Dataset format: one JSON object per line.

    {"image": "images/001.jpg", "vendor": "...", "date": "...", "total": "..."}

Image paths are relative to the JSONL file's folder.
"""
import json
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from .normalize import FIELDS

def load_jsonl(path) -> List[dict]:
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e
            if "image" not in row:
                raise ValueError(f"{path}:{line_no}: missing 'image' key")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows

def truth_of(row: dict) -> dict:
    return {f: row.get(f) for f in FIELDS}

def load_image(base: Path, row: dict) -> Image.Image:
    return Image.open(base / row["image"]).convert("RGB")

def load_dataset(path, limit=None) -> List[Tuple[Image.Image, dict]]:
    """Load (image, truth) pairs. With limit, only the first `limit` receipts are read."""
    path = Path(path)
    base = path.parent
    rows = load_jsonl(path)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        rows = rows[:limit]
    return [(load_image(base, row), truth_of(row)) for row in rows]

def write_jsonl(path, rows: List[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
