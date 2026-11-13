"""Normalization applied to both predictions and ground truth before comparing.

Without this, "$7.45" vs "7.45" or "12/10/2026" vs "2026-10-12" would count as
errors even though the model read the receipt correctly.
"""
import re
from datetime import datetime
from typing import Optional

FIELDS = ("vendor", "date", "total")

_UNAMBIGUOUS_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y%m%d",
    "%d %b %Y",
    "%d %B %Y",
    "%d %b %y",
    "%d %B %y",
    "%d-%b-%Y",
    "%d-%b-%y",
    "%d/%b/%Y",
    "%d/%b/%y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%b %d %Y",
)
_MONTH_FIRST = ("%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y", "%m/%d/%y", "%m-%d-%y", "%m.%d.%y")
_DAY_FIRST = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y", "%d.%m.%y")


def normalize_total(value) -> Optional[str]:
    """Return the amount as a string with two decimals, or None if unparseable.
    
    Handles currency symbols, thousands separators and a decimal comma.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"[^\d,.\-]", "", text)
    if not re.search(r"\d", text):
        return None
    if "," in text and "." in text:
        # The right-most separator is the decimal one.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        head, _, tail = text.rpartition(",")
        if len(tail) == 2:
            text = head.replace(",", "") + "." + tail
        else:
            text = text.replace(",", "")
    try:
        return f"{float(text):.2f}"
    except ValueError:
        return None

def normalize_date(value, day_first: bool = False) -> Optional[str]:
    """Return an ISO date (YYYY-MM-DD), or None if unparseable.

    Numeric dates like 12/10/2026 are ambiguous. By default they are read
    month first (in North American). Pass day_first=True for datasets that use
    day first dates. If only one reading is a valid date (25/10/2026), that
    reading is used either way.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    numeric = _DAY_FIRST + _MONTH_FIRST if day_first else _MONTH_FIRST + _DAY_FIRST
    for fmt in _UNAMBIGUOUS_FORMATS + numeric:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

def normalize_vendor(value) -> Optional[str]:
    """Lowercase, strip punctuation and collapse whitespace."""
    if value is None:
        return None
    text = re.sub(r"[^a-z0-9 ]", " ", str(value).lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text or None

def normalize_record(record: dict, day_first: bool = False) -> dict:
    """Normalize every field of a record. Missing fields become None."""
    return {
        "vendor": normalize_vendor(record.get("vendor")),
        "date": normalize_date(record.get("date"), day_first=day_first),
        "total": normalize_total(record.get("total")),
    }
