"""Per-field evaluation of receipt extraction."""
from typing import Dict, List, Optional

from .normalize import FIELDS, normalize_record


def evaluate(
    predictions: List[Optional[dict]], truths: List[dict], day_first: bool = False
) -> Dict:
    """Compare predictions to ground truth.

    A prediction of None means the model produced no usable output (for
    example, invalid JSON). It counts as wrong for every field and is also
    reported separately as a parse failure.

    Returns a dict with per-field exact-match accuracy, the share of receipts
    where every field was correct, the parse failure rate, and the indices of
    receipts that were wrong (for error analysis).
    """
    if len(predictions) != len(truths):
        raise ValueError(
            f"got {len(predictions)} predictions for {len(truths)} receipts"
        )
    n = len(truths)
    if n == 0:
        raise ValueError("cannot evaluate an empty set")

    correct = {f: 0 for f in FIELDS}
    all_correct = 0
    parse_failures = 0
    failures: Dict[str, List[int]] = {f: [] for f in FIELDS}

    for i, (pred, truth) in enumerate(zip(predictions, truths)):
        gold = normalize_record(truth, day_first)
        if pred is None:
            parse_failures += 1
            for f in FIELDS:
                failures[f].append(i)
            continue
        got = normalize_record(pred, day_first)
        ok_all = True
        for f in FIELDS:
            # A field the truth has but we could not normalize never matches.
            if gold[f] is not None and got[f] == gold[f]:
                correct[f] += 1
            else:
                failures[f].append(i)
                ok_all = False
        if ok_all:
            all_correct += 1

    return {
        "n": n,
        "field_accuracy": {f: correct[f] / n for f in FIELDS},
        "all_fields_accuracy": all_correct / n,
        "parse_failure_rate": parse_failures / n,
        "failed_indices": failures,
    }


def format_table(name: str, result: Dict) -> str:
    """Plain text summary suitable for the README."""
    acc = result["field_accuracy"]
    lines = [
        f"{name} (n={result['n']})",
        f"  vendor accuracy : {acc['vendor']:.1%}",
        f"  date accuracy   : {acc['date']:.1%}",
        f"  total accuracy  : {acc['total']:.1%}",
        f"  all fields      : {result['all_fields_accuracy']:.1%}",
        f"  parse failures  : {result['parse_failure_rate']:.1%}",
    ]
    return "\n".join(lines)
