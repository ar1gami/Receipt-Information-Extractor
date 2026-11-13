"""Build and summarize an error analysis sheet.

Step 1, after running evaluate.py with --out:
    python scripts/error_report.py make data/test.jsonl reports/tuned.json --out reports/errors_tuned.csv
    
Step 2: open the CSV, look at each image, and type a short category in the
category column (for example: blurry, rotated, two totals, faded, unusual layout).

Step 3: count what you wrote:
    python scripts/error_report.py summarize reports/errors_tuned.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from receipts.data import load_jsonl  # noqa: E402
from receipts.error_analysis import (  # noqa: E402
    build_error_rows,
    format_summary,
    summarize_categories,
    write_sheet,
)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make", help="write the sheet to fill in")
    make.add_argument("dataset")
    make.add_argument("report")
    make.add_argument("--out", required=True)
    make.add_argument("--limit", type=int, default=50)
    make.add_argument("--seed", type=int, default=0)

    summ = sub.add_parser("summarize", help="count the categories you filled in")
    summ.add_argument("sheet")

    args = parser.parse_args(argv)
    if args.command == "make":
        rows = load_jsonl(args.dataset)
        report = json.loads(Path(args.report).read_text())
        error_rows = build_error_rows(rows, report, args.limit, args.seed)
        write_sheet(args.out, error_rows)
        print(f"wrote {len(error_rows)} failures to {args.out}")
        return error_rows
    text = format_summary(summarize_categories(args.sheet))
    print(text)
    return text

if __name__ == "__main__":
    main()
