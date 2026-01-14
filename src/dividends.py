# src/dividends.py
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def _extract_field(data: pd.DataFrame, field: str) -> pd.DataFrame:
    """
    Robustly extract a field from yfinance.download output (handles MultiIndex layouts).
    """
    if not isinstance(data.columns, pd.MultiIndex):
        # single ticker case: columns are fields
        if field not in data.columns:
            raise KeyError(f"Field '{field}' not found in data columns.")
        return data[[field]]

    # multi ticker: columns can be (Field, Ticker) OR (Ticker, Field)
    if field in data.columns.get_level_values(0):
        return data[field]
    if field in data.columns.get_level_values(1):
        return data.xs(field, axis=1, level=1)

    raise KeyError(f"Field '{field}' not found in MultiIndex columns.")


def dividend_yield_ttm(
    tickers: list[str],
    period: str = "1y",
) -> tuple[pd.Series, dict]:
    """
    Compute trailing-12-month dividend yield for each ticker:
        yield = sum(dividends over period) / last close

    Returns:
      yields: pd.Series indexed by ticker (float, in decimals, e.g. 0.03 = 3%)
      meta: dict with 'bad' tickers etc.
    """

    tickers = [t for t in tickers if t != "AXA SA"]

    if len(tickers) == 0:
        return pd.Series(dtype=float), {"bad": [], "good": []}

    # actions=True is key for Dividends
    data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        actions=True,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    div = _extract_field(data, "Dividends")
    close = _extract_field(data, "Close")

    out = {}
    bad = []
    for t in tickers:
        try:
            d = div[t].dropna() if t in div.columns else pd.Series(dtype=float)
            c = close[t].dropna() if t in close.columns else pd.Series(dtype=float)

            if len(c) == 0:
                out[t] = 0.0
                bad.append(t)
                continue

            last_px = float(c.iloc[-1])
            if last_px <= 0:
                out[t] = 0.0
                bad.append(t)
                continue

            div_sum = float(d.sum()) if len(d) else 0.0
            y = div_sum / last_px
            if not np.isfinite(y) or y < 0:
                y = 0.0

            out[t] = float(y)
        except Exception:
            out[t] = 0.0
            bad.append(t)

    yields = pd.Series(out, dtype=float)
    meta = {
        "bad": sorted(set(bad)),
        "good": sorted(set(tickers) - set(bad)),
        "period": period,
    }
    return yields, meta
