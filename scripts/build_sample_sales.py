"""Build data/sample_sales.csv.

Revenue is quantity × unit_price, rounded to cents.
Profit is revenue × the product margin, rounded to cents, except for a few
rows left blank on purpose so the quality report has something to find.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PRODUCTS = [
    ("Trail Bottle", "Drinkware", 18.00, 0.42),
    ("Summit Mug", "Drinkware", 24.00, 0.38),
    ("Ridge Notebook", "Stationery", 12.50, 0.55),
    ("Harbor Tote", "Bags", 36.00, 0.33),
    ("Cedar Candle", "Home", 16.00, 0.48),
    ("Maple Blanket", "Home", 64.00, 0.29),
    ("North Lamp", "Home", 48.00, 0.31),
    ("Coast Headphones", "Electronics", 79.00, 0.22),
    ("Field Backpack", "Bags", 58.00, 0.35),
    ("Aurora Planner", "Stationery", 22.00, 0.51),
]

MARGINS = {name: margin for name, _category, _price, margin in PRODUCTS}

BASE_QUANTITY = {
    "Trail Bottle": 4,
    "Summit Mug": 3,
    "Ridge Notebook": 5,
    "Harbor Tote": 3,
    "Cedar Candle": 1,
    "Maple Blanket": 2,
    "North Lamp": 2,
    "Coast Headphones": 2,
    "Field Backpack": 6,
    "Aurora Planner": 4,
}

REGIONS = ["North", "South", "East", "West"]


def build() -> pd.DataFrame:
    dates = pd.date_range("2024-01-08", periods=18, freq="30D")
    rows: list[dict[str, object]] = []
    for month_index, stamp in enumerate(dates):
        for product_index, (name, category, price, margin) in enumerate(PRODUCTS):
            quantity = BASE_QUANTITY[name] + (month_index % 3)
            if name == "Cedar Candle":
                quantity = 1
            revenue = round(quantity * price, 2)
            profit = round(revenue * margin, 2)
            rows.append(
                {
                    "order_date": stamp.strftime("%Y-%m-%d"),
                    "product": name,
                    "category": category,
                    "region": REGIONS[(month_index + product_index) % len(REGIONS)],
                    "quantity": quantity,
                    "unit_price": price,
                    "revenue": revenue,
                    "profit": profit,
                    "returned": "Yes" if (month_index * 10 + product_index) % 17 == 0 else "No",
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[5:10, "region"] = pd.NA
    frame.loc[[8, 18, 28, 38, 48], "profit"] = pd.NA
    frame.loc[[12, 22], "returned"] = pd.NA
    outlier = {
        "order_date": "2024-11-15",
        "product": "Trail Bottle",
        "category": "Drinkware",
        "region": "West",
        "quantity": 80,
        "unit_price": 18.00,
        "revenue": round(80 * 18.00, 2),
        "profit": round(round(80 * 18.00, 2) * MARGINS["Trail Bottle"], 2),
        "returned": "No",
    }
    duplicates = frame.iloc[:4].copy()
    frame = pd.concat([frame, pd.DataFrame([outlier]), duplicates], ignore_index=True)
    frame = frame.sort_values(["order_date", "product"]).reset_index(drop=True)
    return frame


def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "data" / "sample_sales.csv"
    frame = build()
    frame.to_csv(destination, index=False)
    print(f"Wrote {len(frame)} rows to {destination}")


if __name__ == "__main__":
    main()
