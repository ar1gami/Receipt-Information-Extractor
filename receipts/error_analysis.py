"""Turn an evaluation report into a sheet for sorting failures by hand.

The useful part of error analysis is reading failures yourself and naming
their causes (blurry photo, crumpled paper, two totals on one receipt, and so
on). This module only builds the sheet and counts what you wrote.
"""
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List
from .normalize import FIELDS

COLUMNS = ["index", "image", "field", "truth", "prediction", "category"]
NO_OUTPUT = "(no usable output)"

def build_error_rows(rows: List[dict], report: Dict, limit: int = 50, seed: int = 0) -> List[dict]:
    """One row per failed (receipt, field) pair, at most `limit` of them.

    rows: the dataset rows, in the same order the report was made from.
    report: the JSON saved by scripts/evaluate.py --out.
    When there are more failures than `limit`, a seeded random sample is taken.
    """
    predictions = report["predictions"]
    if len(predictions) != len(rows):
        raise ValueError(
            f"report has {len(predictions)} predictions but the dataset has {len(rows)} rows"
        )
    failed = report["metrics"]["failed_indices"]
    pairs = [(i, field) for field in FIELDS for i in failed.get(field, [])]
    pairs.sort()
    if limit is not None and len(pairs) > limit:
        pairs = sorted(random.Random(seed).sample(pairs, limit))
    out = []
    for i, field in pairs:
        pred = predictions[i]
        out.append(
            {
                "index": i,
                "image": rows[i]["image"],
                "field": field,
                "truth": "" if rows[i].get(field) is None else str(rows[i][field]),
                "prediction": NO_OUTPUT if pred is None else ("" if pred.get(field) is None else str(pred[field])),
                "category": "",
            }
        )
    return out

def write_sheet(path, error_rows: List[dict]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(error_rows)

def summarize_categories(path) -> Dict[str, Counter]:
    """Count the categories you filled in, per field. Blank categories are reported too."""
    counts: Dict[str, Counter] = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = (row.get("category") or "").strip().lower() or "(unlabeled)"
            counts.setdefault(row["field"], Counter())[label] += 1
    return counts

def format_summary(counts: Dict[str, Counter]) -> str:
    lines = []
    for field in FIELDS:
        if field not in counts:
            continue
        total = sum(counts[field].values())
        lines.append(f"{field} ({total} failures)")
        for label, n in counts[field].most_common():
            lines.append(f"  {n:>3}  {label}")
    return "\n".join(lines)
