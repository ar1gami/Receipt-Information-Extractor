import pytest

from receipts.metrics import evaluate
from receipts.ocr_baseline import (
    extract_fields_from_text,
    find_date,
    find_total,
    find_vendor,
    ocr_predict,
)
from receipts.synth import make_dataset, make_receipt
from receipts.vlm_output import parse_model_output


#  rule based extraction on plain text (no OCR involved):

def test_total_prefers_total_line_over_subtotal():
    text = "Milk 3.00\nSubtotal 10.00\nTax 1.30\nTOTAL 11.30\n"
    assert find_total(text) == "11.30"

def test_total_ignores_subtotal_when_it_is_the_only_keyword_line():
    text = "Milk 3.00\nSubtotal 10.00\nTax 1.30\n"
    # No usable total line, falls back to the largest amount.
    assert find_total(text) == "10.00"

def test_total_takes_last_total_line():
    text = "Total 5.00\nGrand Total 7.50\n"
    assert find_total(text) == "7.50"

def test_total_handles_decimal_comma_and_thousands():
    assert find_total("TOTAL 1 234,50") == "1234,50"
    assert find_total("TOTAL 1,234.50") == "1,234.50"

def test_total_none_when_no_amounts():
    assert find_total("hello world") is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Date: 2026-10-12", "2026-10-12"),
        ("Date 10/12/2026 14:03", "10/12/2026"),
        ("12 Oct 2026", "12 Oct 2026"),
        ("October 12, 2026", "October 12, 2026"),
        ("no date here", None),
    ],
)
def test_find_date(text, expected):
    assert find_date(text) == expected

def test_find_vendor_skips_blank_and_symbol_lines():
    assert find_vendor("\n  \n***\nCorner Bakery\n123 Main St") == "Corner Bakery"

def test_find_vendor_none_for_empty():
    assert find_vendor("") is None
    assert find_vendor("12 34 56") is None

def test_extract_fields_from_text_full_receipt():
    text = "Corner Bakery\n123 Main St\nDate: 2026-03-05\nBread 4.00\nTOTAL 4.52\n"
    assert extract_fields_from_text(text) == {
        "vendor": "Corner Bakery",
        "date": "2026-03-05",
        "total": "4.52",
    }


#  VLM output parsing:

def test_parse_clean_json():
    out = parse_model_output('{"vendor": "A", "date": "2026-01-02", "total": "3.50"}')
    assert out == {"vendor": "A", "date": "2026-01-02", "total": "3.50"}

def test_parse_code_fenced_json():
    text = '```json\n{"vendor": "A", "date": null, "total": 3.5}\n```'
    assert parse_model_output(text) == {"vendor": "A", "date": None, "total": "3.5"}

def test_parse_json_surrounded_by_prose():
    text = 'Sure! Here is the result: {"vendor": "A", "date": "x", "total": "1"} Hope that helps.'
    assert parse_model_output(text)["vendor"] == "A"

def test_parse_ignores_extra_keys_and_fills_missing():
    out = parse_model_output('{"vendor": "A", "tax": "1.00"}')
    assert out == {"vendor": "A", "date": None, "total": None}

def test_parse_braces_inside_string_values():
    out = parse_model_output('{"vendor": "A {and} B", "date": null, "total": "2"}')
    assert out["vendor"] == "A {and} B"

def test_parse_skips_broken_block_and_finds_later_one():
    text = '{not json} then {"vendor": "B", "date": null, "total": "9"}'
    assert parse_model_output(text)["vendor"] == "B"

@pytest.mark.parametrize("bad", ["", None, "no json at all", "{unterminated", "[1, 2, 3]", "{broken: json}"])
def test_parse_returns_none_for_unusable(bad):
    assert parse_model_output(bad) is None

def test_parse_empty_object_gives_all_none_record():
    assert parse_model_output("{}") == {"vendor": None, "date": None, "total": None}


#  OCR end to end on synthetic receipts 

def test_synthetic_receipts_are_deterministic():
    _, t1 = make_receipt(7)
    _, t2 = make_receipt(7)
    assert t1 == t2

def test_ocr_reads_one_receipt():
    image, truth = make_receipt(3)
    pred = ocr_predict(image)
    result = evaluate([pred], [truth])
    assert result["all_fields_accuracy"] == 1.0, (pred, truth)

@pytest.mark.parametrize("style", ["iso", "slash"])
def test_ocr_baseline_accuracy_on_clean_synthetic_set(style):
    data = make_dataset(20, start_seed=100, date_style=style)
    truths = [t for _, t in data]
    preds = [ocr_predict(img) for img, _ in data]
    result = evaluate(preds, truths)
    # Clean rendered text should be easy for Tesseract. Require high accuracy
    # on every field so a regression in the rules is caught.
    for field, acc in result["field_accuracy"].items():
        assert acc >= 0.9, (field, acc, result["failed_indices"][field])
