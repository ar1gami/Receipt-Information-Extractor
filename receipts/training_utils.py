"""Torch free helpers for fine tuning, so they can be unit tested anywhere."""
import json
import random
from typing import List, Sequence, Tuple
from .normalize import FIELDS

IGNORE_INDEX = -100

def build_target(row: dict) -> str:
    """The text the model should produce for one training example."""
    return json.dumps({f: row.get(f) for f in FIELDS})

def mask_prompt_labels(input_ids: Sequence[int], prompt_len: int) -> List[int]:
    """Labels for causal LM training: only the answer tokens contribute to loss.

    Everything up to prompt_len (the image tokens and the question) is set to
    IGNORE_INDEX, so the model is trained to write the answer and not to
    reproduce the prompt.
    """
    if prompt_len < 0 or prompt_len > len(input_ids):
        raise ValueError(f"prompt_len {prompt_len} outside 0..{len(input_ids)}")
    return [IGNORE_INDEX] * prompt_len + list(input_ids[prompt_len:])

def split_rows(rows: List[dict], val_fraction: float, seed: int) -> Tuple[List[dict], List[dict]]:
    """Deterministic train/validation split. Never touches the test set."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if len(rows) < 2:
        raise ValueError("need at least 2 rows to split")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(round(len(rows) * val_fraction)))
    val_idx = set(indices[:n_val])
    train = [r for i, r in enumerate(rows) if i not in val_idx]
    val = [r for i, r in enumerate(rows) if i in val_idx]
    return train, val
