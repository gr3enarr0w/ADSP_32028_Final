"""
schema.py — data contracts + field normalization shared by ingestion, indexing,
retrieval, and the rag.search tool.

The `RagResult` dataclass is the EXACT payload shape the `rag.search` MCP tool
returns (required keys: sku, title, price, rating, brand?, ingredients?, doc_id),
plus `url` and `score` extras for citations/ranking.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_PRICE_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
# recognizes: "16 Fl Oz", "16oz", "16 fl. oz", "16 ounce(s)", "1.5 lb", "500 ml"
_SIZE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(fl\.?\s*oz|fluid ounce|ounce|oz|ml|milliliter|l\b|liter|lb|pound|g\b|gram)",
    re.IGNORECASE,
)
_COUNT_RE = re.compile(r"(\d+)\s*(?:ct|count|wipes|pods|pack)", re.IGNORECASE)

# rough unit -> fluid ounces (mass units approximated as 1g~1ml for cleaning liquids)
_TO_OZ = {
    "oz": 1.0, "fl oz": 1.0, "fl. oz": 1.0, "fluid ounce": 1.0, "ounce": 1.0,
    "ml": 0.033814, "milliliter": 0.033814,
    "l": 33.814, "liter": 33.814,
    "lb": 16.0, "pound": 16.0,
    "g": 0.033814, "gram": 0.033814,
}


def parse_price(value) -> Optional[float]:
    """'$ 12.49' | '12.49' | '$1,234.00' -> float; unparseable -> None."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    m = _PRICE_RE.search(s.replace(",", ""))
    if not m:
        return None
    try:
        v = float(m.group())
        return v if v > 0 else None
    except ValueError:
        return None


def parse_rating(value) -> Optional[float]:
    """'4.6' | '4.6 out of 5 stars' -> 4.6; missing -> None."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    m = _PRICE_RE.search(s)
    if not m:
        return None
    try:
        v = float(m.group())
    except ValueError:
        return None
    return v if 0 < v <= 5 else None


def parse_size_oz(*texts: str) -> Optional[float]:
    """Best-effort volume in fluid ounces from any of the given strings."""
    for text in texts:
        if not text:
            continue
        m = _SIZE_RE.search(str(text))
        if m:
            qty = float(m.group(1))
            unit = m.group(2).lower().replace(".", "").strip()
            unit = {"fl oz": "fl oz"}.get(unit, unit)
            factor = _TO_OZ.get(unit)
            if factor:
                return round(qty * factor, 3)
    return None


def price_per_oz(price: Optional[float], size_oz: Optional[float]) -> Optional[float]:
    if price and size_oz and size_oz > 0:
        return round(price / size_oz, 4)
    return None


def split_features(about: str) -> list[str]:
    """Amazon 'About Product' is '|'-separated; drop the boilerplate 'Make sure this fits...'."""
    if not about:
        return []
    parts = [p.strip(" •-\t") for p in str(about).split("|")]
    out = []
    for p in parts:
        if not p:
            continue
        if p.lower().startswith("make sure this fits"):
            continue
        out.append(p)
    return out


def clean_ingredients(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.lower() in {"nan", "none"} else s


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Product:
    doc_id: str
    sku: str
    asin: str
    title: str
    brand: str
    category: str
    price: Optional[float]
    list_price: Optional[float]
    rating: Optional[float]
    features: list[str]
    ingredients: str
    size_oz: Optional[float]
    price_per_oz: Optional[float]
    url: str
    stock: str

    def embed_text(self, review_snippets: Optional[list[str]] = None) -> str:
        """title + features + top review snippets + ingredients — the text we embed."""
        chunks = [self.title]
        if self.brand:
            chunks.append(f"Brand: {self.brand}")
        if self.features:
            chunks.append("Features: " + "; ".join(self.features))
        if self.ingredients:
            chunks.append("Ingredients: " + self.ingredients)
        if review_snippets:
            chunks.append("Reviews: " + " ".join(review_snippets[:3]))
        return "\n".join(chunks)


@dataclass
class RagResult:
    """Exact return schema of the rag.search tool (+ url/score extras)."""
    sku: str
    title: str
    price: Optional[float]
    rating: Optional[float]
    doc_id: str
    brand: Optional[str] = None
    ingredients: Optional[str] = None
    url: Optional[str] = None
    price_per_oz: Optional[float] = None
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
