# src/data_prep.py
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd


RAW_PATH = Path("data/raw/kings_quant_data.xlsx")
OUT_PATH = Path("data/processed/universe.csv")


def _to_float_money(x) -> float | np.nan:
    """Convert strings like '$70', '70', ' $120 ' -> 70.0"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_range(x) -> tuple[float | np.nan, float | np.nan]:
    """
    Parse strings like '75-$85', '$140-$160', '160-180' into (low, high).
    Returns (nan, nan) if not parseable.
    """
    if pd.isna(x):
        return (np.nan, np.nan)
    s = str(x).strip().replace(" ", "")
    s = s.replace("$", "")
    # allow formats like "75-85" or "75-$85"
    m = re.match(r"^(\d+(\.\d+)?)-(\d+(\.\d+)?)$", s)
    if not m:
        return (np.nan, np.nan)
    low = float(m.group(1))
    high = float(m.group(3))
    if high < low:
        low, high = high, low
    return (low, high)


def load_universe(path: Path = RAW_PATH) -> pd.DataFrame:
    # Read everything without assuming headers
    raw = pd.read_excel(path, header=None)

    # Find the row that contains the real header (it includes "Equity")
    header_row = None
    for i in range(min(25, len(raw))):  # search first 25 rows
        row_vals = raw.iloc[i].astype(str).tolist()
        if any(v.strip().lower() == "equity" for v in row_vals):
            header_row = i
            break

    if header_row is None:
        raise ValueError("Could not find header row containing 'Equity'")

    # Set columns from that row
    raw.columns = raw.iloc[header_row].tolist()

    # Data starts after header row
    df = raw.iloc[header_row + 1 :].copy()

    # Drop rows that are completely empty
    df = df.dropna(how="all")

    # Rename to consistent internal names
    df = df.rename(
        columns={
            "Ticker*": "Ticker",
            "Sell Side Target": "SellSideTarget",
            "Price Target": "TargetRange",
        }
    )

    # Keep only expected columns (use safe filtering)
    expected = ["Equity", "Ticker", "Region", "Industry", "Price", "SellSideTarget", "TargetRange"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Found columns: {list(df.columns)}")

    df = df[expected]

    # Clean numeric fields
    df["Price"] = df["Price"].apply(_to_float_money)
    df["SellSideTarget"] = df["SellSideTarget"].apply(_to_float_money)

    # Parse range into low/high
    lows, highs = zip(*df["TargetRange"].apply(_parse_range))
    df["RangeLow"] = lows
    df["RangeHigh"] = highs

    # Signals
    df["target_return"] = df["SellSideTarget"] / df["Price"] - 1.0
    df["range_mid"] = (df["RangeLow"] + df["RangeHigh"]) / 2.0
    df["range_mid_return"] = df["range_mid"] / df["Price"] - 1.0
    df["range_width"] = (df["RangeHigh"] - df["RangeLow"]) / df["Price"]

    # Basic cleaning: drop rows missing core fields
    df = df.dropna(subset=["Ticker", "Industry", "Price", "SellSideTarget"])

    # Standardize text
    df["Region"] = df["Region"].astype(str).str.upper().str.strip()
    df["Industry"] = df["Industry"].astype(str).str.strip()
    df["Equity"] = df["Equity"].astype(str).str.strip()
    df["Ticker"] = df["Ticker"].astype(str).str.strip()

    return df.reset_index(drop=True)

def main() -> None:
    df = load_universe()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
