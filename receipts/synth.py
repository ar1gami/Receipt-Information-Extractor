"""Generate clean synthetic receipt images with known ground truth.

For testing the pipeline only. Real evaluation must use real receipts.
"""
import random
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

VENDORS = [
    "Maple Leaf Grocers",
    "Northern Coffee Co",
    "Lakeside Pharmacy",
    "City Hardware",
    "Corner Bakery",
    "Union Bookstore",
]
ITEMS = ["Milk", "Bread", "Coffee", "Notebook", "Batteries", "Soap", "Apples", "Tape"]

_FONT_PATH = "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"

def _font(size: int):
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default(size=size)

def make_receipt(seed: int, date_style: str = "iso") -> Tuple[Image.Image, dict]:
    """Return (image, truth) for one synthetic receipt."""
    rng = random.Random(seed)
    vendor = rng.choice(VENDORS)
    year, month, day = 2026, rng.randint(1, 12), rng.randint(1, 28)
    if date_style == "iso":
        date_text = f"{year}-{month:02d}-{day:02d}"
    elif date_style == "slash":
        date_text = f"{month:02d}/{day:02d}/{year}"
    else:
        raise ValueError(f"unknown date_style {date_style}")

    lines = [vendor, "123 Main Street", f"Date: {date_text}", ""]
    subtotal = 0.0
    for _ in range(rng.randint(2, 5)):
        name = rng.choice(ITEMS)
        price = round(rng.uniform(1.0, 30.0), 2)
        subtotal += price
        lines.append(f"{name:<14}{price:>8.2f}")
    subtotal = round(subtotal, 2)
    tax = round(subtotal * 0.13, 2)
    total = round(subtotal + tax, 2)
    lines += ["", f"{'Subtotal':<14}{subtotal:>8.2f}", f"{'Tax':<14}{tax:>8.2f}",
              f"{'TOTAL':<14}{total:>8.2f}", "", "Thank you"]

    font = _font(26)
    line_h = 36
    width, height = 560, line_h * len(lines) + 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((30, 30 + i * line_h), line, fill="black", font=font)

    truth = {"vendor": vendor, "date": date_text, "total": f"{total:.2f}"}
    return image, truth

def make_dataset(n: int, start_seed: int = 0, date_style: str = "iso") -> List[Tuple[Image.Image, dict]]:
    return [make_receipt(start_seed + i, date_style) for i in range(n)]
