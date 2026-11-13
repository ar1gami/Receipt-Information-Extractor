"""Baseline: Tesseract OCR followed by hand-written extraction rules.

This is the thing the fine-tuned vision-language model has to beat.
"""
import re
from typing import Dict, Optional

from .normalize import normalize_date

_AMOUNT = r"\d{1,3}(?:[,\s]\d{3})*[.,]\d{2}|\d+[.,]\d{2}"
_AMOUNT_RE = re.compile(_AMOUNT)

_DATE_PATTERNS = (
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
    r"\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b",
    r"\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{2}\b",
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
)

_TOTAL_KEYWORD = re.compile(r"\btotal\b|\bgrand\b|amount\s*(?:due|payable)|\bamt\b|\bnett?\b", re.IGNORECASE)
# Words that mean the amount on this line is something other than what was paid.
_NOT_TOTAL = re.compile(
    r"sub\s*-?\s*total|total\s+(?:savings|discount|items?|qty|quantity)|\bqty\b|quantity"
    r"|excl|\btax\b|\bgst\b|\bsst\b|\bsales\s+tax\b|\bchange\b|\bcash\b|\btend|\bsummary\b|"
    r"\bdiscount\b|\bsaving",
    re.IGNORECASE,
)
# A line about tax is fine when it also says the total includes it.
_INCLUSIVE = re.compile(r"\bincl|\binc\b|inclusive|with\s+gst", re.IGNORECASE)
_STRONG_TOTAL = re.compile(
    r"grand\s*total|total\s*(?:\(?rm\)?)?\s*(?:amount|amt)?\s*(?:payable|due|rounded)"
    r"|(?:amount|amt)\s*(?:payable|due)|round(?:ed)?\s*total|total\s+(?:after\s+adj|incl)"
    r"|net+\s*total|total\s*sales\s*incl|\btotal\s*\(?rm\)?\s*[:;]",
    re.IGNORECASE,
)


def _clean_amount(text: str) -> str:
    return re.sub(r"\s", "", text)


def _as_float(a: str) -> float:
    return float(a.replace(",", "."))


_ROUNDING = re.compile(r"round", re.IGNORECASE)
_TOTAL_WORD = re.compile(r"\btotal\b", re.IGNORECASE)


def find_total(text: str) -> Optional[str]:
    """Pick the receipt total.

    Each line that mentions a total and has an amount is scored. Lines that
    say "grand total", "total incl GST", "amount payable" and similar score
    highest, and so does a total after a rounding adjustment. Lines about
    tax, quantity, cash tendered, change or a subtotal are skipped. A plain
    "total" line with two or more amounts is usually a tax summary row, so it
    ranks below single amount lines. Of the best scoring lines, the later one
    wins, because the final total comes after the items. If no line
    qualifies, fall back to the amount that appears most often, then to the
    largest amount.
    """
    best = None  # ((score, line_number), amount)
    seen_rounding = False
    for n, line in enumerate(text.splitlines()):
        if _ROUNDING.search(line):
            seen_rounding = True
        if not _TOTAL_KEYWORD.search(line):
            continue
        # "Total Saving 0.00 Total 77.20": only the last total counts.
        words = list(_TOTAL_WORD.finditer(line))
        if len(words) > 1:
            line = line[words[-1].start():]
        inclusive = bool(_INCLUSIVE.search(line))
        if _NOT_TOTAL.search(line) and not inclusive:
            continue
        amounts = _AMOUNT_RE.findall(line)
        if not amounts:
            continue
        score = 1.0
        if _STRONG_TOTAL.search(line) or inclusive:
            score = 2.0
        elif len(amounts) > 1:
            score = 0.5
        elif seen_rounding:
            score = 2.5
        if re.search(r"grand\s*total|payable|\bdue\b|after\s+(?:adj|round)", line, re.IGNORECASE):
            score = 3.0
        key = (score, n)
        if best is None or key >= best[0]:
            best = (key, _clean_amount(amounts[0] if len(amounts) == 1 or score >= 2 else amounts[-1]))
    if best:
        return best[1]

    amounts = [_clean_amount(a) for a in _AMOUNT_RE.findall(text)]
    if not amounts:
        return None
    counts = {}
    for a in amounts:
        counts[a] = counts.get(a, 0) + 1
    repeated = [a for a, c in counts.items() if c > 1 and _as_float(a) > 0]
    pool = repeated or amounts
    return max(pool, key=_as_float)


def _fix_digit_letters(text: str) -> str:
    """OCR often reads a zero as the letter O at the start of a date part."""
    text = re.sub(r"(?<![A-Za-z0-9])[Oo](?=\d[-/.]\d)", "0", text)
    return re.sub(r"(?<=\d[-/.])[Oo](?=\d)", "0", text)


def find_date(text: str) -> Optional[str]:
    """Return the first date in the text that is a real calendar date.

    Strings that look like dates but are not (month 62, year 2078) are
    skipped, since they are almost always OCR mistakes or other numbers. If
    no real date is found, the first date shaped string is returned instead.
    """
    text = _fix_digit_letters(text)
    found = []
    for pattern in _DATE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            found.append((match.start(), match.group(0)))
    found.sort()
    for _, candidate in found:
        for day_first in (True, False):
            iso = normalize_date(candidate, day_first=day_first)
            if iso and 2000 <= int(iso[:4]) <= 2035:
                return candidate
    return found[0][1] if found else None


_COMPANY_WORDS = re.compile(
    r"\bsdn\b|\bbhd\b|\bs/?b\b|\benterprise|\btrading|\brestaura?n|\brestoran\b|\bmart\b|"
    r"\bmarket\b|\bstore|\bshop|\bhardware|\bstationery|\bpharmacy|\bbakery|\bcafe\b|"
    r"\bkitchen|\bcentre\b|\bcenter\b|\bco\.?\b|\bltd\b|\binc\b|\bcorp|\bresources\b|"
    r"\bindustr|\bholdings|\bgallery|\bmotor|\bmachinery|\bservices?\b|\bbooks?\b|\bsupermarket",
    re.IGNORECASE,
)
_LEGAL_SUFFIX = re.compile(r"\bsdn\.?\s*bhd|\bs/b\b|\bbhd\b", re.IGNORECASE)
# A line that is only the tail of a company name, for example "SDN BHD" or "(M) SDN BHD".
_SUFFIX_ONLY = re.compile(r"^(?:\(?\s*m\s*\)?\s*)?(?:sdn\.?\s*bhd\.?|s/b|bhd\.?)$", re.IGNORECASE)
# A line that starts with the tail of a name, so the name began on the line before.
_STARTS_WITH_TAIL = re.compile(r"^(?:restaurant|restoran|sdn|co\.|\(m\)|bhd|s/b)\b", re.IGNORECASE)
# Lines that are address, contact or document headings, never the store name.
_NOT_VENDOR = re.compile(
    r"\bjalan\b|\blorong\b|\btaman\b|\bbandar\b|\btel\b|\bfax\b|\bgst\b|\btax\b|invoice|"
    r"receipt|\bbill\b|\bcash\b|\bcopy\b|www\.|@|\.com|\bno\.?\s*\d|\bdate\b|\bemail\b|"
    r"\bcashier\b|\b\d{5}\b|\bjohor\b|\bselangor\b|\bkuala\b|\bperak\b|\bpenang\b|"
    r"\b(?:reg|roc|co)\.?\s*no\b|\bsimplified\b",
    re.IGNORECASE,
)
_REG_NUMBER = re.compile(r"\s*[\(\[]\s*\d[\w\- ]{3,}\s*[\)\]]?\s*$")
_VENDOR_SCAN_LINES = 10


def _clean_vendor(line: str) -> str:
    line = _REG_NUMBER.sub("", line.strip())
    line = re.sub(r"^[^A-Za-z0-9(]+", "", line)
    line = re.sub(r"^\d\s+(?=[A-Za-z]{3,})", "", line)
    tokens = line.replace("\u201c", " ").replace("\u201d", " ").replace('"', " ").split()
    # Drop stray lowercase or digit fragments before the first capitalized word.
    for k, token in enumerate(tokens):
        if re.fullmatch(r"[A-Z][A-Z0-9'&.\-]{2,}", token):
            if k and all(re.fullmatch(r"[a-z0-9]{1,4}", t) for t in tokens[:k]):
                tokens = tokens[k:]
            break
    # Drop stray one or two letter fragments at the end, but keep real abbreviations.
    keep = {"CO", "CO.", "S/B", "SB", "(M)", "M", "&", "BHD", "BHD."}
    while len(tokens) > 1 and len(tokens[-1].strip(".,")) <= 2 and tokens[-1].upper() not in keep:
        tokens.pop()
    return " ".join(tokens).strip(" |_-.,:;~")


def _vendor_score(line: str, index: int) -> float:
    letters = re.findall(r"[A-Za-z]", line)
    visible = re.sub(r"\s", "", line)
    if len(letters) < 4 or not visible:
        return -99.0
    score = 0.0
    if _COMPANY_WORDS.search(line):
        score += 3.0
    if _LEGAL_SUFFIX.search(line):
        score += 1.0
    if _NOT_VENDOR.search(line):
        score -= 3.0
    upper = sum(1 for c in letters if c.isupper())
    if upper / len(letters) >= 0.7:
        score += 1.0
    score += 1.5 * (len(letters) / len(visible) - 0.7)
    words = re.findall(r"[A-Za-z]{2,}", line)
    score += min(len(words), 4) * 0.3
    if len(line) > 45:
        score -= 1.0
    return score - 0.15 * index


def find_vendor(text: str) -> Optional[str]:
    """Pick the store name from the top of the receipt.

    The first lines are scored: company words (SDN BHD, ENTERPRISE, MART) and
    mostly capital letters raise the score, address, phone and heading words
    lower it, and earlier lines win ties. A name that ends with "&" or "AND"
    is joined with the next line. If nothing scores well, the first line with
    real words is used.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    best = None
    for index, line in enumerate(lines[:_VENDOR_SCAN_LINES]):
        score = _vendor_score(line, index)
        if best is None or score > best[0]:
            best = (score, index)
    if best is not None and best[0] > -90:
        i = best[1]
        name = _clean_vendor(lines[i])
        if _SUFFIX_ONLY.match(name) and i > 0:
            name = f"{_clean_vendor(lines[i - 1])} {name}"
        elif _STARTS_WITH_TAIL.match(name) and i > 0 and _vendor_score(lines[i - 1], i - 1) > 0:
            name = f"{_clean_vendor(lines[i - 1])} {name}"
        if i + 1 < len(lines):
            following = _clean_vendor(lines[i + 1])
            if re.search(r"(?:&|\band)$", name, re.IGNORECASE) or _SUFFIX_ONLY.match(following):
                name = f"{name} {following}"
        if name:
            return name
    for line in lines:
        if len(re.findall(r"[A-Za-z]", line)) >= 3:
            return line
    return None


def extract_fields_from_text(text: str) -> Dict[str, Optional[str]]:
    return {
        "vendor": find_vendor(text),
        "date": find_date(text),
        "total": find_total(text),
    }


def ocr_predict(image) -> Dict[str, Optional[str]]:
    """Run Tesseract on a PIL image and extract the three fields."""
    import pytesseract

    text = pytesseract.image_to_string(image, config="--psm 6")
    ocr_predict.last_text = text
    return extract_fields_from_text(text)


ocr_predict.last_text = None
