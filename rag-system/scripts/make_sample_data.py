"""
make_sample_data.py
-------------------
Generates a small, realistic SAMPLE of the *Amazon Product Dataset 2020* (Kaggle)
restricted to the Household Cleaning slice, so the whole RAG pipeline is runnable
without downloading the full ~10k-row Kaggle CSV.

The sample uses the EXACT column names of the real Kaggle file
(`marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv`)
so `ingest.py` runs identically on the sample and on the real data.

Two columns are *optional extras* that the real 10k file does NOT contain:
  - "Rating"          : average star rating (float)
  - "Review Snippets" : ' || '-separated short review quotes
`ingest.py` uses them if present and degrades gracefully (rating=NaN, no reviews)
when they are absent, which is what happens on the real Kaggle file.

Usage:
    python scripts/make_sample_data.py
Writes:
    data/raw/SAMPLE_amazon_household_cleaning.csv
"""
from __future__ import annotations

import csv
import os
import uuid
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = RAW_DIR / "SAMPLE_amazon_household_cleaning.csv"

# Exact real-file schema (order matters for realism) + 2 optional extras at the end.
COLUMNS = [
    "Uniq Id", "Product Name", "Brand Name", "Asin", "Category", "Upc Ean Code",
    "List Price", "Selling Price", "Quantity", "Model Number", "About Product",
    "Product Specification", "Technical Details", "Shipping Weight",
    "Product Dimensions", "Image", "Variants", "Sku", "Product Url", "Stock",
    "Product Details", "Dimensions", "Color", "Ingredients", "Direction To Use",
    "Is Amazon Seller", "Size Quantity Variant", "Product Description",
    "Rating", "Review Snippets",
]

CAT = "Health & Household | Household Supplies | Cleaning & Household | {sub}"


def about(features):
    # Real dataset stores About Product as 'Make sure this fits...|feat1|feat2|...'
    return "Make sure this fits by entering your model number. | " + " | ".join(features)


# (name, brand, sub-category, list$, sell$, size_oz, eco, material_focus, ingredients, features, rating, reviews)
P = [
    ("Steel-Safe Eco Stainless Steel Cleaner & Polish, 16 oz", "GreenGleam", "Metal Cleaners",
     15.99, 12.49, 16.0, True, "stainless steel",
     "Water, Caprylyl/Capryl Glucoside (plant-based surfactant), Citric Acid, Sodium Citrate, Coconut-derived cleansers, Lavender essential oil",
     ["Plant-based, biodegradable formula safe for kitchen appliances",
      "Removes fingerprints, smudges, and water marks from stainless steel",
      "Streak-free shine with a light lavender scent", "Cruelty-free and phosphate-free"],
     4.6, ["Best stainless cleaner I've used, no streaks || Smells great and cuts fingerprints instantly",
           "Eco-friendly and actually works on my fridge"]),
    ("Brushed Metal Miracle Stainless Cleaner Spray, 12 oz", "PureHome", "Metal Cleaners",
     13.49, 9.99, 12.0, True, "stainless steel",
     "Water, Decyl Glucoside, Sodium Gluconate, Citric Acid, Plant-derived alcohol, Fragrance (natural)",
     ["Eco-friendly stainless steel spray for appliances and sinks",
      "Leaves a protective anti-smudge layer", "No harsh ammonia or bleach", "USDA BioPreferred formula"],
     4.4, ["Great value under $10 || Works but needs a second pass on heavy grime",
           "Love that it's non-toxic around my kids"]),
    ("ProShine Stainless Steel Polish, Aerosol, 15 oz", "ShineMaster", "Metal Cleaners",
     18.99, 16.75, 15.0, False, "stainless steel",
     "Mineral oil, Aliphatic hydrocarbons, Silicone emulsion, Propellant",
     ["Professional-grade oil-based polish for a deep shine",
      "Long-lasting fingerprint resistance", "For commercial and home appliances"],
     4.7, ["Restaurant-quality shine || A little greasy if you use too much",
           "Best for heavy-duty commercial kitchens"]),
    ("EcoBright Stainless & Chrome Cleaner Wipes, 30 ct", "GreenGleam", "Metal Cleaners",
     11.99, 8.49, 8.0, True, "stainless steel",
     "Water, Coco-Glucoside, Citric Acid, Bamboo fiber cloth, Plant-based preservative",
     ["Compostable wipes made from bamboo fiber", "Grab-and-go stainless steel cleaning",
      "Streak-free, plant-based formula", "Great for quick appliance touch-ups"],
     4.3, ["Convenient and eco-friendly || Wipes dry out if you don't seal the lid",
           "Perfect size for my stainless fridge"]),
    ("Sparkle Naturals Glass & Window Cleaner, 26 oz", "Sparkle Naturals", "Glass Cleaners",
     9.99, 6.99, 26.0, True, "glass",
     "Water, Ethanol (plant-derived), Vinegar, Coco-Glucoside, Essential oil blend",
     ["Ammonia-free streak-free glass cleaner", "Plant-based and biodegradable",
      "Safe for mirrors, windows, and glass cooktops"],
     4.5, ["Streak free on windows || Vinegar smell fades fast",
           "Cheaper and greener than the blue stuff"]),
    ("CitrusForce All-Purpose Cleaner Concentrate, 32 oz", "CitrusForce", "All-Purpose Cleaners",
     14.99, 11.25, 32.0, True, "multi-surface",
     "Water, Citrus terpenes, Sodium coco-sulfate, Citric acid, Orange essential oil",
     ["Makes up to 4 gallons when diluted", "Plant-based degreaser for counters and floors",
      "Biodegradable and septic-safe", "Fresh orange scent"],
     4.6, ["A little goes a long way || Great degreaser for the stovetop",
           "Concentrate saves money and plastic"]),
    ("PowerClean Heavy-Duty Degreaser Spray, 24 oz", "PowerClean", "All-Purpose Cleaners",
     12.99, 10.49, 24.0, False, "multi-surface",
     "Water, 2-Butoxyethanol, Sodium hydroxide, Nonionic surfactants, Fragrance",
     ["Industrial-strength degreaser for kitchens and garages",
      "Cuts baked-on grease fast", "Not for use on aluminum"],
     4.2, ["Powerful on oven grease || Strong smell, use gloves",
           "Works when nothing else will"]),
    ("Gentle Suds Plant-Based Dish Soap, 18 oz", "Gentle Suds", "Dish Soap",
     7.49, 5.49, 18.0, True, "dishware",
     "Water, Sodium lauroyl sarcosinate, Lauryl glucoside, Aloe, Sea salt, Fragrance (natural)",
     ["Ultra-concentrated plant-based dish soap", "Tough on grease, gentle on hands",
      "Free of dyes, parabens, and phosphates", "Aloe-infused"],
     4.7, ["Cuts grease and doesn't dry my hands || Nice light scent",
           "Small bottle but very concentrated"]),
    ("Bubble Rush Antibacterial Dish Liquid, 28 oz", "Bubble Rush", "Dish Soap",
     6.99, 4.99, 28.0, False, "dishware",
     "Water, Sodium laureth sulfate, Lauramine oxide, Fragrance, Benzisothiazolinone, Blue 1",
     ["Long-lasting suds for tough loads", "Antibacterial hand-wash formula", "Fresh spring scent"],
     4.1, ["Lots of bubbles || Contains dyes, wish it didn't",
           "Cheap and does the job"]),
    ("FreshBath Eco Bathroom & Tile Cleaner, 24 oz", "FreshBath", "Bathroom Cleaners",
     10.99, 8.99, 24.0, True, "tile and porcelain",
     "Water, Lactic acid, Caprylyl glucoside, Citric acid, Eucalyptus oil",
     ["Plant-based bathroom cleaner for tile, tubs, and sinks", "Dissolves soap scum and hard-water stains",
      "Bleach-free and biodegradable", "Eucalyptus-mint scent"],
     4.4, ["Soap scum gone without bleach fumes || Needs a minute of dwell time",
           "Smells like a spa"]),
    ("ScrubPro Bleach Bathroom Foam, 20 oz", "ScrubPro", "Bathroom Cleaners",
     8.49, 6.29, 20.0, False, "tile and porcelain",
     "Water, Sodium hypochlorite, Sodium hydroxide, Surfactants, Fragrance",
     ["Clinging foam kills 99.9% of germs", "Whitens grout and removes mold stains",
      "Not safe for natural stone"],
     4.3, ["Blasts mildew || Ventilate the room, strong bleach",
           "Grout looks new"]),
    ("WoodLove Natural Wood Floor & Furniture Cleaner, 32 oz", "WoodLove", "Wood Cleaners",
     13.99, 12.99, 32.0, True, "wood",
     "Water, Plant-derived soap, Jojoba oil, Beeswax emulsion, Orange oil",
     ["pH-balanced for sealed hardwood and furniture", "Nourishes wood with jojoba and beeswax",
      "Streak-free, residue-free", "Plant-based and pet-safe"],
     4.6, ["Floors smell amazing and shine || Not for unsealed wood",
           "Safe around my dog"]),
    ("EverGreen Stainless Steel Wipes Refill, 40 ct", "EverGreen", "Metal Cleaners",
     14.49, 13.49, 10.0, True, "stainless steel",
     "Water, Lauryl glucoside, Citric acid, Plant fiber cloth, Rosemary extract",
     ["40-count refillable stainless steel wipes", "Removes grease and fingerprints",
      "Biodegradable cloth, recyclable tub", "Light rosemary scent"],
     4.5, ["Refill saves plastic || Slightly wet, let appliance dry",
           "Great for daily fridge wipe-downs"]),
    ("MegaShine Chrome & Stainless Spray, 22 oz", "MegaShine", "Metal Cleaners",
     16.99, 14.99, 22.0, False, "stainless steel",
     "Water, Isopropyl alcohol, Silicone, Ammonia-free surfactants, Fragrance",
     ["Fast-drying stainless and chrome spray", "Anti-static fingerprint guard",
      "For fixtures, appliances, and cars"],
     4.4, ["Dries fast and shines || Contains silicone, buff well",
           "Good for both kitchen and car"]),
    ("Meadow Fresh Laundry Detergent Pods, 42 ct", "Meadow Fresh", "Laundry",
     18.99, 15.99, 42.0, True, "fabric",
     "Sodium carbonate, Plant-based enzymes, Coconut surfactants, Essential oils, PVA film",
     ["Concentrated plant-based laundry pods", "Dissolves fully in cold water",
      "Free of dyes and optical brighteners", "42 loads"],
     4.5, ["Clean clothes, small footprint || PVA film debate aside, works great",
           "No perfume residue"]),
    ("ClearView Streak-Free Glass Wipes, 35 ct", "ClearView", "Glass Cleaners",
     7.99, 5.99, 7.0, False, "glass",
     "Water, Isopropyl alcohol, Ammonia, Nonionic surfactants, Fragrance",
     ["Pre-moistened glass and mirror wipes", "Streak-free shine on the go",
      "For windows, mirrors, and screens"],
     4.2, ["Handy for mirrors || Ammonia smell is strong",
           "Streak free if you buff"]),
    ("PureHome Multi-Surface Eco Spray, 28 oz", "PureHome", "All-Purpose Cleaners",
     11.49, 9.49, 28.0, True, "multi-surface",
     "Water, Decyl glucoside, Sodium citrate, Lactic acid, Thyme essential oil",
     ["Everyday plant-based multi-surface spray", "Safe for sealed counters, glass, and stainless",
      "Biodegradable, phosphate-free", "Thyme-citrus scent"],
     4.6, ["My go-to for the whole kitchen || Works on stainless too",
           "Non-toxic and effective"]),
    ("ToughGuy Oven & Grill Cleaner, 19 oz", "ToughGuy", "All-Purpose Cleaners",
     9.99, 7.99, 19.0, False, "enamel and metal",
     "Water, Sodium hydroxide, Butoxyethanol, Foaming agents",
     ["Heavy-duty foaming oven and grill cleaner", "Removes carbonized grease",
      "Fume-reduced formula"],
     4.3, ["Melts oven gunk || Wear gloves, caustic",
           "Grill looks brand new"]),
    ("NatureNest Stainless Steel Cleaner, 8 oz travel", "NatureNest", "Metal Cleaners",
     8.99, 6.49, 8.0, True, "stainless steel",
     "Water, Coco-glucoside, Citric acid, Sodium gluconate, Peppermint oil",
     ["Compact eco stainless steel cleaner", "Plant-based, streak-free",
      "TSA-friendly travel size", "Peppermint scent"],
     4.4, ["Perfect travel size || Wish it came bigger",
           "Great for RV stainless"]),
    ("HydroClean Steam Mop Floor Solution, 33 oz", "HydroClean", "Floor Care",
     12.49, 10.99, 33.0, True, "sealed floors",
     "Water, Plant-based surfactants, Sodium citrate, Lemongrass oil",
     ["Streak-free floor solution for steam and spray mops", "Plant-based and residue-free",
      "Safe for sealed hardwood, tile, and laminate"],
     4.5, ["No residue on tile || Lemongrass smell is lovely",
           "Works in my spray mop"]),
    ("BrightWhite Grout & Tile Pen Cleaner, 5 oz", "BrightWhite", "Bathroom Cleaners",
     6.49, 4.49, 5.0, False, "grout",
     "Water, Calcium carbonate, Titanium dioxide, Acrylic polymer",
     ["Restores white grout lines", "Precision applicator tip", "Waterproof once dry"],
     4.0, ["Grout looks new || Tedious on big areas",
           "Cheap fix for stained grout"]),
    ("GreenGleam Cooktop & Glass Stove Cream, 10 oz", "GreenGleam", "Glass Cleaners",
     12.99, 10.49, 10.0, True, "glass ceramic",
     "Water, Calcium carbonate (mild abrasive), Coco-glucoside, Citric acid, Lavender oil",
     ["Non-scratch cream for glass and ceramic cooktops", "Removes burnt-on residue",
      "Plant-based, leaves protective shine"],
     4.6, ["Burnt milk wiped right off || Buff after for best shine",
           "Best cooktop cleaner, eco too"]),
    ("SprayJoy Lemon All-Purpose Cleaner, 32 oz", "SprayJoy", "All-Purpose Cleaners",
     5.99, 3.99, 32.0, False, "multi-surface",
     "Water, Alkyl polyglucoside, Sodium carbonate, Lemon fragrance, Preservative",
     ["Budget everyday cleaner", "Fresh lemon scent", "For counters, floors, and appliances"],
     4.1, ["Cheap and cheerful || Not the greenest but works",
           "Big bottle for the price"]),
    ("AquaPure Eco Toilet Bowl Cleaner Gel, 24 oz", "AquaPure", "Bathroom Cleaners",
     8.99, 6.99, 24.0, True, "porcelain",
     "Water, Citric acid, Caprylyl glucoside, Xanthan gum, Tea tree oil",
     ["Plant-based toilet bowl gel", "Clings to remove stains and hard water",
      "Bleach-free, septic-safe", "Tea tree scent"],
     4.4, ["No bleach fumes and clean bowl || Squeeze bottle could be bigger",
           "Septic-safe and effective"]),
]


def price_str(v):
    return f"$ {v:.2f}"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (name, brand, sub, listp, sellp, oz, eco, material, ingredients, feats, rating, reviews) in enumerate(P):
        asin = f"B0SAMPLE{i:03d}"
        rows.append({
            "Uniq Id": uuid.uuid5(uuid.NAMESPACE_DNS, asin).hex,
            "Product Name": name,
            "Brand Name": brand,
            "Asin": asin,
            "Category": CAT.format(sub=sub),
            "Upc Ean Code": "",
            "List Price": price_str(listp),
            "Selling Price": price_str(sellp),
            "Quantity": "",
            "Model Number": f"{brand[:3].upper()}-{i:03d}",
            "About Product": about(feats),
            "Product Specification": f"Material focus: {material}",
            "Technical Details": "",
            "Shipping Weight": f"{oz/16.0 + 0.3:.1f} pounds",
            "Product Dimensions": "",
            "Image": f"https://images.example.com/{asin}.jpg",
            "Variants": "",
            "Sku": f"SKU-{brand[:4].upper()}-{i:03d}",
            "Product Url": f"https://www.amazon.com/dp/{asin}",
            "Stock": "In Stock" if i % 7 != 0 else "Only 3 left in stock",
            "Product Details": "",
            "Dimensions": "",
            "Color": "",
            "Ingredients": ingredients,
            "Direction To Use": "Spray, wipe with a microfiber cloth, buff dry.",
            "Is Amazon Seller": "Y",
            "Size Quantity Variant": f"{oz:g} Fl Oz" if "ct" not in name.lower() else name.split(",")[-1].strip(),
            "Product Description": f"{'Eco-friendly ' if eco else ''}{material} cleaner. " + " ".join(feats[:2]),
            # optional extras (absent in the real 10k file):
            "Rating": f"{rating:.1f}",
            "Review Snippets": " || ".join(reviews),
        })

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} sample products -> {OUT}")


if __name__ == "__main__":
    main()
