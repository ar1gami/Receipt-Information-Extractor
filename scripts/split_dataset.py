"""Split one JSONL file into train.jsonl and test.jsonl next to it.

    python scripts/split_dataset.py data/sroie/data.jsonl --test-fraction 0.2

Image paths stay relative, so the two files must sit in the same folder as the
images folder. The split is seeded, so rerunning gives the same test set. The
script refuses to overwrite existing output unless --force is given, so a test
set you already evaluated on is not silently replaced.

Receipts from the same store tend to look alike. If your data has many receipts
per store, check that a store does not appear in both files, or the test score
will look better than it should.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from receipts.data import load_jsonl, write_jsonl  # noqa: E402
from receipts.training_utils import split_rows  # noqa: E402

def split_file(path, test_fraction=0.2, seed=0, force=False):
    path = Path(path)
    train_path = path.parent / "train.jsonl"
    test_path = path.parent / "test.jsonl"
    if not force:
        for p in (train_path, test_path):
            if p.exists():
                raise FileExistsError(f"{p} already exists (use --force to overwrite)")
    rows = load_jsonl(path)
    train, test = split_rows(rows, test_fraction, seed)
    write_jsonl(train_path, train)
    write_jsonl(test_path, test)
    return train_path, test_path, len(train), len(test)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("jsonl")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    train_path, test_path, n_train, n_test = split_file(
        args.jsonl, args.test_fraction, args.seed, args.force
    )
    print(f"{train_path}: {n_train} receipts")
    print(f"{test_path}: {n_test} receipts")
    return n_train, n_test

if __name__ == "__main__":
    main()
