"""
ingest.py — Amazon Product Dataset 2020 (Kaggle) -> curated Household-Cleaning
slice -> products.parquet (+ reviews.parquet).

Runs identically on:
  * the SAMPLE csv shipped in data/raw/ (24 rows), and
  * the real Kaggle file `marketing_sample_for_amazon_com-ecommerce__...10k_data.csv`
    (just point RAW_CSV at it, or drop it in data/raw/ and pass its path).

Real-file column names are used throughout; optional columns ("Rating",
"Review Snippets") are used when present and skipped gracefully when absent.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import Config, get_config
from .schema import (
    Product, clean_ingredients, parse_price, parse_rating, parse_size_oz,
    price_per_oz, split_features,
)

# real Kaggle column names (a subset we rely on)
COL = {
    "id": "Uniq Id",
    "title": "Product Name",
    "brand": "Brand Name",
    "asin": "Asin",
    "category": "Category",
    "list_price": "List Price",
    "sell_price": "Selling Price",
    "about": "About Product",
    "ingredients": "Ingredients",
    "sku": "Sku",
    "url": "Product Url",
    "stock": "Stock",
    "size_variant": "Size Quantity Variant",
    "description": "Product Description",
    # optional extras (absent in the real 10k file):
    "rating": "Rating",
    "reviews": "Review Snippets",
}


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    return df


def filter_slice(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep rows whose Category mentions any household-cleaning keyword."""
    if COL["category"] not in df.columns:
        return df
    cat = df[COL["category"]].fillna("").str.lower()
    mask = pd.Series(False, index=df.index)
    for kw in cfg.slice_keywords:
        mask |= cat.str.contains(kw, regex=False)
    filtered = df[mask].copy()
    # Fallback: if the real file has no matching category text, keep everything
    # so the notebook still produces output (user can refine keywords).
    return filtered if len(filtered) else df


def _get(row, key: str, default: str = "") -> str:
    col = COL.get(key)
    if col and col in row and pd.notna(row[col]):
        return str(row[col])
    return default


def build_products(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for i, row in df.reset_index(drop=True).iterrows():
        asin = _get(row, "asin") or f"ROW{i:05d}"
        title = _get(row, "title").strip()
        if not title:
            continue
        price = parse_price(_get(row, "sell_price")) or parse_price(_get(row, "list_price"))
        list_price = parse_price(_get(row, "list_price"))
        size_oz = parse_size_oz(_get(row, "size_variant"), title, _get(row, "about"))
        features = split_features(_get(row, "about"))
        p = Product(
            doc_id=_get(row, "id") or asin or f"doc_{i:05d}",
            sku=_get(row, "sku") or asin,
            asin=asin,
            title=title,
            brand=_get(row, "brand").strip(),
            category=_get(row, "category").strip(),
            price=price,
            list_price=list_price,
            rating=parse_rating(_get(row, "rating")),
            features=features,
            ingredients=clean_ingredients(_get(row, "ingredients")),
            size_oz=size_oz,
            price_per_oz=price_per_oz(price, size_oz),
            url=_get(row, "url"),
            stock=_get(row, "stock"),
        )
        rec = p.__dict__.copy()
        rec["features"] = " | ".join(p.features)  # parquet-friendly flat string
        records.append(rec)
    products = pd.DataFrame.from_records(records)
    return products


def build_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Explode the optional 'Review Snippets' column into a tidy reviews table.

    Returns an empty (but correctly-typed) frame when the column is absent —
    which is the case for the real Kaggle file, so downstream code must treat
    reviews as optional.
    """
    rows = []
    has_reviews = COL["reviews"] in df.columns
    for i, row in df.reset_index(drop=True).iterrows():
        pid = _get(row, "id") or _get(row, "asin") or f"doc_{i:05d}"
        if not has_reviews:
            continue
        raw = _get(row, "reviews")
        if not raw:
            continue
        stars = parse_rating(_get(row, "rating"))
        for snip in [s.strip() for s in raw.split("||") if s.strip()]:
            rows.append({"product_id": pid, "stars": stars, "snippet": snip})
    return pd.DataFrame(rows, columns=["product_id", "stars", "snippet"])


def write_parquet(products: pd.DataFrame, reviews: pd.DataFrame, out_dir: str) -> dict:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pth_products = os.path.join(out_dir, "products.parquet")
    pth_reviews = os.path.join(out_dir, "reviews.parquet")
    products.to_parquet(pth_products, index=False)
    reviews.to_parquet(pth_reviews, index=False)
    return {"products": pth_products, "reviews": pth_reviews}


def run(csv_path: Optional[str] = None, out_dir: Optional[str] = None,
        cfg: Optional[Config] = None) -> dict:
    """End-to-end ingestion. Returns dict with paths + row counts."""
    cfg = cfg or get_config()
    csv_path = csv_path or cfg.raw_csv
    out_dir = out_dir or cfg.processed_dir

    raw = load_raw(csv_path)
    sliced = filter_slice(raw, cfg)
    products = build_products(sliced)
    reviews = build_reviews(sliced)
    paths = write_parquet(products, reviews, out_dir)
    return {
        **paths,
        "n_raw": len(raw),
        "n_slice": len(sliced),
        "n_products": len(products),
        "n_reviews": len(reviews),
        "with_price": int(products["price"].notna().sum()) if len(products) else 0,
        "with_rating": int(products["rating"].notna().sum()) if len(products) else 0,
        "with_ingredients": int((products["ingredients"].str.len() > 0).sum()) if len(products) else 0,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, default=str))
