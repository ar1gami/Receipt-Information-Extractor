import pytest

from receipts.metrics import evaluate
from receipts.normalize import (
    normalize_date,
    normalize_record,
    normalize_total,
    normalize_vendor,
)

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("$7.45", "7.45"),
        ("7.45", "7.45"),
        (7.45, "7.45"),
        (7, "7.00"),
        ("CAD 1,234.50", "1234.50"),
        ("1.234,50", "1234.50"),
        ("12,50", "12.50"),
        ("1,234", "1234.00"),
        ("  $ 0.99 ", "0.99"),
        ("-5.00", "-5.00"),
        ("", None),
        (None, None),
        ("abc", None),
        ("$", None),
    ],
)
def test_normalize_total(raw, expected):
    assert normalize_total(raw) == expected

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2026-10-12", "2026-10-12"),
        ("2026/10/12", "2026-10-12"),
        ("12/10/2026", "2026-12-10"),  # month first when both readings are valid
        ("25/10/2026", "2026-10-25"),  # day first when month first is impossible
        ("Oct 12, 2026", "2026-10-12"),
        ("12 October 2026", "2026-10-12"),
        ("  2026-10-12  ", "2026-10-12"),
        ("not a date", None),
        ("", None),
        (None, None),
        ("2026-13-45", None),
    ],
)
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Tim Hortons", "tim hortons"),
        ("TIM HORTONS #1234", "tim hortons 1234"),
        ("Tim  Hortons.", "tim hortons"),
        ("", None),
        (None, None),
        ("!!!", None),
    ],
)
def test_normalize_vendor(raw, expected):
    assert normalize_vendor(raw) == expected

def test_day_first_flag_changes_only_ambiguous_dates():
    assert normalize_date("12/10/2026", day_first=True) == "2026-10-12"
    assert normalize_date("12/10/2026", day_first=False) == "2026-12-10"
    # Unambiguous dates are unaffected by the flag.
    assert normalize_date("25/10/2026", day_first=True) == "2026-10-25"
    assert normalize_date("25/10/2026", day_first=False) == "2026-10-25"
    assert normalize_date("2026-10-12", day_first=True) == "2026-10-12"

def test_evaluate_respects_day_first():
    truths = [{"vendor": "A", "date": "12/10/2026", "total": "1.00"}]
    preds = [{"vendor": "A", "date": "2026-10-12", "total": "1.00"}]
    assert evaluate(preds, truths, day_first=True)["field_accuracy"]["date"] == 1.0
    assert evaluate(preds, truths, day_first=False)["field_accuracy"]["date"] == 0.0

def test_normalize_record_fills_missing_fields():
    assert normalize_record({"vendor": "A"}) == {
        "vendor": "a",
        "date": None,
        "total": None,
    }

TRUTHS = [
    {"vendor": "Tim Hortons", "date": "2026-10-12", "total": "7.45"},
    {"vendor": "Shoppers Drug Mart", "date": "2026-10-13", "total": "23.10"},
    {"vendor": "Metro", "date": "2026-10-14", "total": "88.00"},
    {"vendor": "LCBO", "date": "2026-10-15", "total": "31.25"},
]

def test_perfect_predictions():
    result = evaluate([dict(t) for t in TRUTHS], TRUTHS)
    assert result["all_fields_accuracy"] == 1.0
    assert all(v == 1.0 for v in result["field_accuracy"].values())
    assert result["parse_failure_rate"] == 0.0
    assert all(v == [] for v in result["failed_indices"].values())

def test_formatting_differences_are_not_errors():
    preds = [
        {"vendor": "TIM HORTONS", "date": "Oct 12, 2026", "total": "$7.45"},
        {"vendor": "shoppers drug mart", "date": "2026/10/13", "total": "23.1"},
        {"vendor": "Metro.", "date": "2026-10-14", "total": "88"},
        {"vendor": "LCBO", "date": "2026-10-15", "total": "31.25"},
    ]
    result = evaluate(preds, TRUTHS)
    assert result["all_fields_accuracy"] == 1.0

def test_known_mix_of_errors():
    preds = [
        {"vendor": "Tim Hortons", "date": "2026-10-12", "total": "7.45"},  # all right
        {"vendor": "Shoppers", "date": "2026-10-13", "total": "23.10"},  # vendor wrong
        {"vendor": "Metro", "date": "2026-10-15", "total": "80.00"},  # date + total wrong
        {"vendor": "LCBO", "date": "2026-10-15", "total": "31.25"},  # all right
    ]
    result = evaluate(preds, TRUTHS)
    assert result["field_accuracy"]["vendor"] == pytest.approx(3 / 4)
    assert result["field_accuracy"]["date"] == pytest.approx(3 / 4)
    assert result["field_accuracy"]["total"] == pytest.approx(3 / 4)
    assert result["all_fields_accuracy"] == pytest.approx(2 / 4)
    assert result["failed_indices"]["vendor"] == [1]
    assert result["failed_indices"]["date"] == [2]
    assert result["failed_indices"]["total"] == [2]

def test_none_prediction_counts_as_parse_failure_and_all_wrong():
    preds = [dict(TRUTHS[0]), None, dict(TRUTHS[2]), dict(TRUTHS[3])]
    result = evaluate(preds, TRUTHS)
    assert result["parse_failure_rate"] == pytest.approx(1 / 4)
    assert result["all_fields_accuracy"] == pytest.approx(3 / 4)
    for f in ("vendor", "date", "total"):
        assert result["failed_indices"][f] == [1]

def test_missing_field_in_prediction_is_wrong():
    preds = [dict(t) for t in TRUTHS]
    del preds[0]["total"]
    result = evaluate(preds, TRUTHS)
    assert result["field_accuracy"]["total"] == pytest.approx(3 / 4)
    assert result["field_accuracy"]["vendor"] == 1.0

def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate([dict(TRUTHS[0])], TRUTHS)

def test_empty_set_raises():
    with pytest.raises(ValueError):
        evaluate([], [])

def test_all_none_predictions():
    result = evaluate([None] * 4, TRUTHS)
    assert result["parse_failure_rate"] == 1.0
    assert result["all_fields_accuracy"] == 0.0
    assert all(v == 0.0 for v in result["field_accuracy"].values())
