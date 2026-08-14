# notebooks/

## `01_ingestion.ipynb` (CP1 deliverable)

The ingestion notebook is generated deterministically from a script and then
executed, so it is a build artifact (kept out of version control). Recreate it
with two commands from the repo root:

```bash
python scripts/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/01_ingestion.ipynb
```

This produces the executed notebook with outputs. Last verified run on the
shipped sample slice:

```
raw shape: (24, 30)      slice kept: 24 / 24
products: (24, 15)       reviews: (72, 3)
coverage: has_price 1.00 · has_rating 1.00 · has_ingredients 1.00 · has_size_oz 1.00
demo sub-slice (eco stainless < $15): Steel-Safe Eco ($12.49, 4.6★), Brushed Metal
  Miracle ($9.99, 4.4★), EcoBright wipes ($8.49, 4.3★), NatureNest ($6.49, 4.4★)
outputs written: data/processed/products.parquet, data/processed/reviews.parquet
```

The notebook simply drives `src/rag/ingest.py` + `src/rag/schema.py` (the same
code the build script and MCP tool use) and shows the key normalization steps
(price parsing, size→oz, price-per-oz, feature/ingredient extraction). To run it
on the full Kaggle file, set `RAW_CSV` in `.env` first (see repo `README_shane.md`).
