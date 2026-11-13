"""Build one results table from reports saved by scripts/evaluate.py --out.

    python scripts/compare.py ocr=reports/ocr.json base=reports/base.json \
        tuned=reports/tuned.json int8=reports/int8.json

Each argument is label=path. Rows appear in the order given. Latency columns
are blank for reports made without --latency, and the size column is blank for
the OCR baseline.
"""
import json
import sys
from pathlib import Path

HEADER = ["model", "vendor", "date", "total", "all fields", "parse fail", "median ms", "p95 ms", "size MB"]


def row_for(label: str, report: dict) -> list:
    m = report["metrics"]
    acc = m["field_accuracy"]
    lat = report.get("latency")
    return [
        label,
        f"{acc['vendor']:.1%}",
        f"{acc['date']:.1%}",
        f"{acc['total']:.1%}",
        f"{m['all_fields_accuracy']:.1%}",
        f"{m['parse_failure_rate']:.1%}",
        f"{lat['median_ms']:.0f}" if lat else "",
        f"{lat['p95_ms']:.0f}" if lat else "",
        f"{report['model_size_mb']:.0f}" if report.get("model_size_mb") else "",
    ]


def render(rows: list) -> str:
    table = [HEADER] + rows
    widths = [max(len(r[i]) for r in table) for i in range(len(HEADER))]
    lines = []
    for n, r in enumerate(table):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)).rstrip())
        if n == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def parse_pairs(args: list) -> list:
    pairs = []
    for arg in args:
        if "=" not in arg:
            raise ValueError(f"expected label=path, got {arg!r}")
        label, path = arg.split("=", 1)
        pairs.append((label, Path(path)))
    return pairs


def main(argv=None) -> str:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit(__doc__)
    rows = []
    for label, path in parse_pairs(argv):
        rows.append(row_for(label, json.loads(path.read_text())))
    text = render(rows)
    print(text)
    return text


if __name__ == "__main__":
    main()
