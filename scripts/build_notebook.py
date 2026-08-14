"""Generate notebooks/01_ingestion.ipynb (CP1 deliverable) programmatically."""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_ingestion.ipynb"

nb = new_notebook()
cells = []

cells.append(new_markdown_cell(
    "# CP1 — Data Ingestion: Amazon Product Dataset 2020 → Household-Cleaning slice\n"
    "\n"
    "**Owner:** Shane · **Deliverable:** Checkpoint 1 ingestion notebook\n"
    "\n"
    "This notebook curates a **Household Cleaning** slice of the *Amazon Product "
    "Dataset 2020* (Kaggle) and writes two tidy tables used by the rest of the "
    "pipeline:\n"
    "\n"
    "| file | schema |\n"
    "|---|---|\n"
    "| `data/processed/products.parquet` | `doc_id, sku, asin, title, brand, category, price, list_price, rating, features, ingredients, size_oz, price_per_oz, url, stock` |\n"
    "| `data/processed/reviews.parquet`  | `product_id, stars, snippet` |\n"
    "\n"
    "It runs **unchanged** on the shipped 24-row `SAMPLE_...csv` and on the real "
    "Kaggle file — point `RAW_CSV` at whichever you have.\n"
    "\n"
    "> The heavy lifting lives in `src/rag/ingest.py` and `src/rag/schema.py` so the "
    "> exact same logic is reused by the build script and the MCP tool. The cells "
    "> below both *call* that module and *show* the key transformations for review."
))

cells.append(new_markdown_cell("## 0. Setup"))
cells.append(new_code_cell(
    "import sys, os\n"
    "sys.path.append(os.path.abspath('../src'))\n"
    "import pandas as pd\n"
    "pd.set_option('display.max_colwidth', 60)\n"
    "\n"
    "from rag.config import get_config\n"
    "from rag import ingest, schema\n"
    "\n"
    "cfg = get_config()\n"
    "# To use the REAL Kaggle file instead of the sample, uncomment and edit:\n"
    "# cfg.raw_csv = '../data/raw/marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv'\n"
    "print('Raw CSV :', cfg.raw_csv)\n"
    "print('Output  :', cfg.processed_dir)"
))

cells.append(new_markdown_cell(
    "## 1. Load raw data\n"
    "We read everything as strings (the Kaggle file mixes types and has missing "
    "values) and inspect the schema."
))
cells.append(new_code_cell(
    "raw = ingest.load_raw(cfg.raw_csv)\n"
    "print('raw shape:', raw.shape)\n"
    "print('columns:', list(raw.columns))\n"
    "raw.head(3)"
))

cells.append(new_markdown_cell(
    "## 2. Filter to the Household-Cleaning slice\n"
    "We keep rows whose `Category` mentions any cleaning/household keyword "
    "(`cfg.slice_keywords`). On the real 10k file this narrows ~10,000 rows down to "
    "the cleaning products; on the sample it is already the slice."
))
cells.append(new_code_cell(
    "print('slice keywords:', cfg.slice_keywords)\n"
    "sliced = ingest.filter_slice(raw, cfg)\n"
    "print(f'{len(sliced)} / {len(raw)} rows kept')\n"
    "sliced[[ingest.COL['title'], ingest.COL['category']]].head()"
))

cells.append(new_markdown_cell(
    "## 3. Field normalization (the interesting part)\n"
    "`schema.py` normalizes the messy Amazon fields into analysis-ready values:\n"
    "\n"
    "* **price** — `'$ 12.49'` → `12.49` (falls back from *Selling Price* to *List Price*)\n"
    "* **size_oz** — parsed from *Size Quantity Variant* / title (`'16 Fl Oz'` → `16.0`, `'500 ml'` → `16.9`)\n"
    "* **price_per_oz** — enables *fair* comparisons across pack sizes\n"
    "* **features** — the `'|'`-separated *About Product* with the boilerplate dropped\n"
    "* **rating** — from the optional *Rating* column (NaN on the real file until reviews are joined)"
))
cells.append(new_code_cell(
    "examples = ['$ 12.49', '1,234.00', '', 'nan']\n"
    "print('parse_price :', [schema.parse_price(x) for x in examples])\n"
    "print('parse_size_oz:', schema.parse_size_oz('16 Fl Oz'), schema.parse_size_oz('500 ml'),\n"
    "      schema.parse_size_oz('1.5 lb'))\n"
    "print('price_per_oz :', schema.price_per_oz(12.49, 16.0))\n"
    "print('features     :', schema.split_features('Make sure this fits... | Plant-based | Streak-free'))"
))

cells.append(new_markdown_cell("## 4. Build the products table"))
cells.append(new_code_cell(
    "products = ingest.build_products(sliced)\n"
    "print('products shape:', products.shape)\n"
    "products[['title','brand','price','size_oz','price_per_oz','rating','ingredients']].head()"
))

cells.append(new_markdown_cell("## 5. Build the reviews table (optional column)"))
cells.append(new_code_cell(
    "reviews = ingest.build_reviews(sliced)\n"
    "print('reviews shape:', reviews.shape, '(empty on the real file — reviews are optional)')\n"
    "reviews.head()"
))

cells.append(new_markdown_cell(
    "## 6. Quick EDA / data-quality checks\n"
    "Sanity-check coverage of the fields the retriever and answerer depend on."
))
cells.append(new_code_cell(
    "def coverage(df):\n"
    "    return pd.Series({\n"
    "        'rows': len(df),\n"
    "        'has_price': df['price'].notna().mean(),\n"
    "        'has_rating': df['rating'].notna().mean(),\n"
    "        'has_size_oz': df['size_oz'].notna().mean(),\n"
    "        'has_ingredients': (df['ingredients'].str.len() > 0).mean(),\n"
    "        'has_features': (df['features'].str.len() > 0).mean(),\n"
    "    })\n"
    "coverage(products)"
))
cells.append(new_code_cell(
    "# Price distribution and the demo-relevant sub-slice: eco stainless steel cleaners < $15\n"
    "print(products['price'].describe()[['min','50%','max']].round(2).to_dict())\n"
    "mask = (products['title'].str.contains('stainless', case=False) &\n"
    "        products['ingredients'].str.contains('plant|coco|citr|glucoside', case=False) &\n"
    "        (products['price'] < 15))\n"
    "products.loc[mask, ['title','brand','price','price_per_oz','rating']].sort_values('rating', ascending=False)"
))

cells.append(new_markdown_cell(
    "## 7. Write parquet outputs\n"
    "These two files are the contract for the indexing step (`build_index.sh`)."
))
cells.append(new_code_cell(
    "paths = ingest.write_parquet(products, reviews, cfg.processed_dir)\n"
    "paths"
))
cells.append(new_code_cell(
    "# One-liner equivalent (what the build script calls):\n"
    "summary = ingest.run()\n"
    "summary"
))

cells.append(new_markdown_cell(
    "---\n"
    "### Handoff notes\n"
    "* **Index step** embeds `title + features + top-3 review snippets + ingredients` "
    "(`Product.embed_text`) and stores metadata (`brand, price, rating, size_oz, "
    "price_per_oz, category, ingredients, sku, url, doc_id`).\n"
    "* **`rag.search`** returns `{sku, title, price, rating, brand, ingredients, doc_id}` "
    "with `url`/`score` extras for citations.\n"
    "* To scale to the full Kaggle catalog, set `RAW_CSV` and (optionally) widen "
    "`cfg.slice_keywords` — nothing else changes."
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print("wrote", OUT)
