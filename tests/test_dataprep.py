import csv
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from receipts.data import load_dataset, load_jsonl
from receipts.error_analysis import (
    NO_OUTPUT,
    build_error_rows,
    format_summary,
    summarize_categories,
    write_sheet,
)
from receipts.metrics import evaluate
from receipts.sroie import convert_sroie

ROOT = Path(__file__).resolve().parent.parent

def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def make_fake_sroie(tmp_path):
    images = tmp_path / "img"
    labels = tmp_path / "lab"
    images.mkdir()
    labels.mkdir()

    def add_image(name):
        Image.new("RGB", (8, 8), "white").save(images / name)

    good = {"company": "STARBUCKS STORE #10208", "address": "1 MAIN ST", "date": "14/03/2015", "total": "4.95"}
    # 1. normal
    add_image("X001.jpg")
    (labels / "X001.txt").write_text(json.dumps(good))
    # 2. upper case extension and .json label
    add_image("X002.JPG")
    (labels / "X002.json").write_text(json.dumps({**good, "company": "BOOK TA .K SDN BHD"}))
    # 3. png image
    add_image("X003.png")
    (labels / "X003.txt").write_text(json.dumps({**good, "date": "05/03/2015", "total": "RM9.00"}))
    # 4. bad json
    add_image("X004.jpg")
    (labels / "X004.txt").write_text("{not valid json")
    # 5. missing key
    add_image("X005.jpg")
    (labels / "X005.txt").write_text(json.dumps({"company": "A", "date": "01/01/2018"}))
    # 6. no image
    (labels / "X006.txt").write_text(json.dumps(good))
    # 7. label that is valid json but not an object
    add_image("X007.jpg")
    (labels / "X007.txt").write_text("[1, 2, 3]")
    # 8. unrelated file in the label folder
    (labels / "notes.md").write_text("ignore me")
    return images, labels


# SROIE Converter:

def test_convert_sroie_counts_and_content(tmp_path):
    images, labels = make_fake_sroie(tmp_path)
    out = tmp_path / "out"
    summary = convert_sroie(images, labels, out)
    assert summary == {
        "converted": 3,
        "skipped_bad_json": 1,
        "skipped_missing_keys": 2,  # X005 and the non-object X007
        "skipped_no_image": 1,
    }
    rows = load_jsonl(out / "data.jsonl")
    assert [r["image"] for r in rows] == ["images/X001.jpg", "images/X002.JPG", "images/X003.png"]
    assert rows[0] == {
        "image": "images/X001.jpg",
        "vendor": "STARBUCKS STORE #10208",
        "date": "14/03/2015",
        "total": "4.95",
    }
    assert "address" not in rows[0]
    for r in rows:
        assert (out / r["image"]).is_file()

def test_converted_dataset_loads_and_scores_with_day_first(tmp_path):
    images, labels = make_fake_sroie(tmp_path)
    out = tmp_path / "out"
    convert_sroie(images, labels, out)
    data = load_dataset(out / "data.jsonl")
    truths = [t for _, t in data]
    preds = [
        {"vendor": "starbucks store 10208", "date": "2015-03-14", "total": "4.95"},
        {"vendor": "BOOK TA K SDN BHD", "date": "2015-03-14", "total": "4.95"},
        {"vendor": "STARBUCKS STORE #10208", "date": "2015-03-05", "total": "9"},
    ]
    day_first = evaluate(preds, truths, day_first=True)
    assert day_first["field_accuracy"]["date"] == 1.0
    assert day_first["field_accuracy"]["total"] == 1.0  # "RM9.00" is read as 9.00
    # 14/03/2015 can only be day first, so it parses either way. 05/03/2015 is
    # ambiguous, and read month first it becomes 5 May, which is not 5 March.
    month_first = evaluate(preds, truths, day_first=False)
    assert month_first["field_accuracy"]["date"] == pytest.approx(2 / 3)
    assert month_first["failed_indices"]["date"] == [2]

def test_convert_sroie_missing_folders(tmp_path):
    with pytest.raises(FileNotFoundError, match="image folder"):
        convert_sroie(tmp_path / "nope", tmp_path, tmp_path / "o")
    with pytest.raises(FileNotFoundError, match="label folder"):
        convert_sroie(tmp_path, tmp_path / "nope", tmp_path / "o")

def test_convert_sroie_nothing_converted_raises(tmp_path):
    images = tmp_path / "i"
    labels = tmp_path / "l"
    images.mkdir()
    labels.mkdir()
    (labels / "a.txt").write_text("garbage")
    with pytest.raises(ValueError, match="nothing converted"):
        convert_sroie(images, labels, tmp_path / "o")

def test_prepare_sroie_cli(tmp_path, capsys):
    images, labels = make_fake_sroie(tmp_path)
    script = load_script("prepare_sroie")
    summary = script.main(["--images", str(images), "--labels", str(labels), "--out", str(tmp_path / "o")])
    assert summary["converted"] == 3
    printed = capsys.readouterr().out
    assert "converted: 3" in printed and "data.jsonl" in printed

# Split:

def build_synthetic(tmp_path, n=20):
    make_synthetic = load_script("make_synthetic")
    return make_synthetic.build(tmp_path / "ds", n, start_seed=40)

def test_split_creates_disjoint_files_next_to_input(tmp_path):
    split = load_script("split_dataset")
    jsonl = build_synthetic(tmp_path)
    train_path, test_path, n_train, n_test = split.split_file(jsonl, 0.25, seed=3)
    assert (n_train, n_test) == (15, 5)
    assert train_path.parent == jsonl.parent == test_path.parent
    train = load_jsonl(train_path)
    test = load_jsonl(test_path)
    assert not {r["image"] for r in train} & {r["image"] for r in test}
    assert {r["image"] for r in train} | {r["image"] for r in test} == {r["image"] for r in load_jsonl(jsonl)}
    # Image paths still resolve from the new files.
    assert len(load_dataset(test_path)) == 5

def test_split_is_reproducible_and_refuses_to_overwrite(tmp_path):
    split = load_script("split_dataset")
    jsonl = build_synthetic(tmp_path)
    _, test_path, _, _ = split.split_file(jsonl, 0.25, seed=3)
    first = test_path.read_text()
    with pytest.raises(FileExistsError, match="--force"):
        split.split_file(jsonl, 0.25, seed=3)
    assert test_path.read_text() == first  # untouched by the refused call
    split.split_file(jsonl, 0.25, seed=3, force=True)
    assert test_path.read_text() == first
    split.split_file(jsonl, 0.25, seed=4, force=True)
    assert test_path.read_text() != first

def test_split_cli(tmp_path, capsys):
    split = load_script("split_dataset")
    jsonl = build_synthetic(tmp_path, n=10)
    n_train, n_test = split.main([str(jsonl), "--test-fraction", "0.3"])
    assert (n_train, n_test) == (7, 3)
    assert "test.jsonl: 3 receipts" in capsys.readouterr().out


# Error Analysis:

ROWS = [
    {"image": "images/0.png", "vendor": "A", "date": "2026-01-01", "total": "1.00"},
    {"image": "images/1.png", "vendor": "B", "date": "2026-01-02", "total": "2.00"},
    {"image": "images/2.png", "vendor": "C", "date": "2026-01-03", "total": "3.00"},
    {"image": "images/3.png", "vendor": "D", "date": "2026-01-04", "total": "4.00"},
]

def make_report():
    preds = [
        dict(ROWS[0]),  # right
        {"vendor": "X", "date": "2026-01-02", "total": "9.00"},  # vendor and total wrong
        None,  # parse failure, all three wrong
        {"vendor": "D", "date": "2026-02-04", "total": "4.00"},  # date wrong
    ]
    truths = [{k: r[k] for k in ("vendor", "date", "total")} for r in ROWS]
    return {"metrics": evaluate(preds, truths), "predictions": preds}

def test_error_rows_cover_every_failure():
    rows = build_error_rows(ROWS, make_report(), limit=None)
    pairs = [(r["index"], r["field"]) for r in rows]
    # Sorted by receipt index, then field name.
    assert pairs == [
        (1, "total"), (1, "vendor"), (2, "date"), (2, "total"), (2, "vendor"), (3, "date"),
    ]
    by_pair = {(r["index"], r["field"]): r for r in rows}
    assert by_pair[(1, "vendor")]["truth"] == "B" and by_pair[(1, "vendor")]["prediction"] == "X"
    assert by_pair[(2, "total")]["prediction"] == NO_OUTPUT
    assert by_pair[(3, "date")]["image"] == "images/3.png"
    assert all(r["category"] == "" for r in rows)

def test_error_rows_limit_is_seeded_sample_of_real_failures():
    report = make_report()
    everything = {(r["index"], r["field"]) for r in build_error_rows(ROWS, report, limit=None)}
    assert len(everything) == 6
    a = build_error_rows(ROWS, report, limit=3, seed=1)
    b = build_error_rows(ROWS, report, limit=3, seed=1)
    assert a == b
    assert len(a) == 3
    assert {(r["index"], r["field"]) for r in a} <= everything
    # The seed must actually change the sample (checked over several seeds so
    # one lucky coincidence cannot fail the test).
    samples = {
        tuple((r["index"], r["field"]) for r in build_error_rows(ROWS, report, limit=3, seed=s))
        for s in range(10)
    }
    assert len(samples) > 1

def test_error_rows_limit_larger_than_failures_returns_all():
    assert len(build_error_rows(ROWS, make_report(), limit=100)) == 6

def test_error_rows_length_mismatch():
    with pytest.raises(ValueError, match="predictions"):
        build_error_rows(ROWS[:2], make_report())

def test_no_failures_gives_empty_sheet():
    truths = [{k: r[k] for k in ("vendor", "date", "total")} for r in ROWS]
    perfect = {"metrics": evaluate([dict(t) for t in truths], truths), "predictions": [dict(t) for t in truths]}
    assert build_error_rows(ROWS, perfect) == []

def test_sheet_round_trip_and_summary(tmp_path):
    sheet = tmp_path / "errors.csv"
    rows = build_error_rows(ROWS, make_report(), limit=None)
    labels = {(1, "vendor"): "Blurry", (2, "vendor"): " blurry ", (2, "date"): "faded", (3, "date"): "faded"}
    for r in rows:
        r["category"] = labels.get((r["index"], r["field"]), "")
    write_sheet(sheet, rows)
    with sheet.open(newline="", encoding="utf-8") as f:
        assert list(csv.DictReader(f))[0].keys() >= {"index", "image", "field", "truth", "prediction", "category"}
    counts = summarize_categories(sheet)
    assert counts["vendor"]["blurry"] == 2
    assert counts["date"]["faded"] == 2
    assert counts["total"]["(unlabeled)"] == 2
    text = format_summary(counts)
    assert "vendor (2 failures)" in text and "total (2 failures)" in text
    assert text.index("vendor (") < text.index("date (") < text.index("total (")

def test_error_report_cli_round_trip(tmp_path, capsys):
    script = load_script("error_report")
    dataset = tmp_path / "test.jsonl"
    dataset.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n")
    report = tmp_path / "report.json"
    report.write_text(json.dumps(make_report()))
    sheet = tmp_path / "sheet.csv"
    made = script.main(["make", str(dataset), str(report), "--out", str(sheet), "--limit", "50"])
    assert len(made) == 6
    # Fill in the category column 
    with sheet.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["category"] = "layout"
    with sheet.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    text = script.main(["summarize", str(sheet)])
    assert "layout" in text
    assert "6 failures" not in text  # counted per field, not overall
    assert "vendor (2 failures)" in text
