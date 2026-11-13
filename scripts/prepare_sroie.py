"""Convert a downloaded SROIE style dataset into data.jsonl plus an images folder.

    python scripts/prepare_sroie.py --images path/to/images --labels path/to/key_labels --out data/sroie

data folder is out of git (it is listed in .gitignore), to respect terms of use.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from receipts.sroie import convert_sroie  # noqa: E402

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--images", required=True, help="folder with the receipt images")
    parser.add_argument("--labels", required=True, help="folder with the per-receipt JSON label files")
    parser.add_argument("--out", required=True, help="output folder")
    args = parser.parse_args(argv)
    summary = convert_sroie(args.images, args.labels, args.out)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"wrote {Path(args.out) / 'data.jsonl'}")
    return summary

if __name__ == "__main__":
    main()
