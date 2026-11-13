"""Write a small synthetic receipt dataset (images plus JSONL) for smoke tests.

    python scripts/make_synthetic.py data/synthetic 20
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from receipts.data import write_jsonl  # noqa: E402
from receipts.synth import make_dataset  # noqa: E402

def build(out_dir, n: int, start_seed: int = 0, date_style: str = "iso") -> Path:
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (image, truth) in enumerate(make_dataset(n, start_seed, date_style)):
        name = f"images/{i:03d}.png"
        image.save(out_dir / name)
        rows.append({"image": name, **truth})
    jsonl = out_dir / "data.jsonl"
    write_jsonl(jsonl, rows)
    return jsonl

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    print(build(out, count))
