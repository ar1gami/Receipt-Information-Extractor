"""Regression tests for the OCR extraction rules.

Each case is a short made up snippet that mimics a pattern found while reading
failures on the training receipts (tax summary rows, junk first lines, names
split over two lines, dates the scorer could not read). None of it is copied
from a dataset.
"""
import pytest

from receipts.normalize import normalize_date
from receipts.ocr_baseline import find_date, find_total, find_vendor


#  dates the scorer must understand (labels keep the receipt's own format)

@pytest.mark.parametrize(
    "text, day_first, expected",
    [
        ("12-01-19", True, "2019-01-12"),
        ("12-01-19", False, "2019-12-01"),
        ("14 MAR 2018", True, "2018-03-14"),
        ("28 MAR 18", True, "2018-03-28"),
        ("20180304", True, "2018-03-04"),
        ("05.03.2018", True, "2018-03-05"),
        ("05.03.18", True, "2018-03-05"),
        ("24-Mar-2018", True, "2018-03-24"),
        ("25/12/2018", True, "2018-12-25"),
    ],
)
def test_normalize_date_handles_receipt_formats(text, day_first, expected):
    assert normalize_date(text, day_first=day_first) == expected

@pytest.mark.parametrize("text", ["20181304", "32-01-19", "12-13-19x", "14 MAX 2018", "2018"])
def test_normalize_date_rejects_nonsense(text):
    assert normalize_date(text, day_first=True) is None


#  Totals:

def test_tax_summary_row_is_not_the_total():
    text = "Item 4.60\nSUB-TOTAL (EX) : 4.60\nTOTAL TAX : 0.28\nTOTAL ; 4.90\nCASH 4.90\nAMOUNT TAX\nTOTAL: 4.60 0.28"
    assert find_total(text) == "4.90"

def test_total_quantity_and_gst_lines_are_skipped():
    text = "Total Qty 2\nTotal GST 0.00\nTotal 80.90\nCash 100.00"
    assert find_total(text) == "80.90"

def test_inclusive_line_beats_excluding_line():
    text = "Total Sales (Excluding GST) : 31.03\nTotal GST : 1.65\nTotal Sales (Inclusive of GST) : 32.70\nCASH : 40.00"
    assert find_total(text) == "32.70"

def test_total_after_rounding_wins():
    text = (
        "Total (Excluding GST): 21.30\nGST payable (6%): 1.37\n"
        "Total (Inclusive of GST): 23.26\nRounding Adj: -0.01\nTOTAL: 23.25"
    )
    assert find_total(text) == "23.25"

def test_total_after_adjustment_line():
    text = "Sub-total 17.76\nTotal Sales Incl GST 17.76\nTotal After Adj Incl GST 17.70\nCASH 50.00"
    assert find_total(text) == "17.70"

def test_last_total_on_a_line_with_saving_first():
    assert find_total("Item 2 Total with GST @ 6% 77.20\nTotal Saving 0.00 Total 77.20\nCash 100.00") == "77.20"

def test_cash_and_change_lines_never_chosen():
    assert find_total("TOTAL 12.00\nCASH 20.00\nCHANGE 8.00") == "12.00"

def test_amount_payable_and_grand_total_win_over_plain_total():
    assert find_total("Total 10.00\nGrand Total 11.00") == "11.00"
    assert find_total("Total 10.00\nAmount Payable 9.50") == "9.50"


def test_fallback_uses_the_repeated_amount():
    # No line says total. The paid amount shows up twice, the other once.
    assert find_total("Coffee 3.50\nBagel 4.00\nPaid 7.50\nCard 7.50") == "7.50"

def test_no_amount_gives_none():
    assert find_total("thank you\nsee you again") is None

#  Vendors:

def test_junk_first_line_is_skipped_for_the_company_line():
    text = "tan woon yann\nOJC MARKETING SDN BHD\nROC NO: 538358-H\nNO 2 & 4, JALAN BAYU 4,\nBANDAR SERI ALAM"
    assert find_vendor(text) == "OJC MARKETING SDN BHD"

def test_registration_number_is_removed():
    assert find_vendor("TRI SHAAS SDN BHD (728515-H)\nJALAN ABC") == "TRI SHAAS SDN BHD"

def test_name_ending_with_ampersand_joins_the_next_line():
    assert find_vendor("HOME MASTER HARDWARE &\nELECTRICAL\nTel 07-123 4567") == "HOME MASTER HARDWARE & ELECTRICAL"

def test_suffix_on_the_next_line_is_joined():
    assert find_vendor("PASARAYA BORONG PINTAR\nSDN BHD\nNO 1 JALAN X") == "PASARAYA BORONG PINTAR SDN BHD"

def test_suffix_only_line_takes_the_line_before():
    assert find_vendor("LIAN CHI PU TIAN VEGETARIAN\nRESTAURANT SDN BHD\nJALAN X") == "LIAN CHI PU TIAN VEGETARIAN RESTAURANT SDN BHD"

def test_leading_and_trailing_noise_removed():
    assert find_vendor("bee 3 PAGOH REST AND SERVICE AREA,\nTel 03-1234 5678") == "PAGOH REST AND SERVICE AREA"
    assert find_vendor("* THAI DELICIOUS RESTAURANT “a\nTel 03-1234 5678") == "THAI DELICIOUS RESTAURANT"

def test_address_and_heading_lines_are_not_the_vendor():
    text = "TAX INVOICE\nNO.53, JALAN BESAR, 45600 BATANG BERJUNTAI\nTEO HENG STATIONERY & BOOKS\nTel 03-1234 5678"
    assert find_vendor(text) == "TEO HENG STATIONERY & BOOKS"

def test_vendor_falls_back_to_first_line_with_words():
    assert find_vendor("12 34\nabc def\nxyz") == "abc def"

def test_vendor_empty_text():
    assert find_vendor("") is None
    assert find_vendor("  \n 12 \n") is None

#  Dates in OCR text:

def test_invalid_year_is_skipped_for_a_real_date():
    assert find_date("Date: 19/03/2078\nPrinted 13/03/2017 03:00") == "13/03/2017"

def test_only_invalid_date_still_returned():
    assert find_date("Date 19/62/2018") == "19/62/2018"

def test_letter_o_read_for_zero():
    assert find_date("O7/01/18 14:41 Slip") == "07/01/18"

def test_month_name_with_two_digit_year():
    assert find_date("28 Mar 18 18:32:36") == "28 Mar 18"

def test_earliest_real_date_wins_by_position():
    assert find_date("Sold 05/03/2018\nPrinted 06/03/2018") == "05/03/2018"

def test_no_date_gives_none():
    assert find_date("no digits here") is None
