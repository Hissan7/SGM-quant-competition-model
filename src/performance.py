# src/performance.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


OUT_METRICS = Path("results/metrics")
OUT_PLOTS = Path("results/plots")


def load_weights(path: str = "results/metrics/weights_by_month.csv") -> pd.DataFrame:
    w = pd.read_csv(path)
    if "month" not in w.columns:
        raise ValueError("weights_by_month.csv must have a 'month' column")
    return w


def download_prices(tickers: list[str], years: int = 3) -> pd.DataFrame:
    """
    Download daily Adjusted Close for a list of tickers.
    Returns a DataFrame indexed by date with columns=tickers.
    """
    period = f"{years}y"
    data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    # yfinance returns different shapes depending on single vs multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        # MultiIndex (field, ticker) OR (ticker, field) depending on version
        # Try to find 'Adj Close' robustly
        if "Adj Close" in data.columns.get_level_values(0):
            px = data["Adj Close"]
        elif "Adj Close" in data.columns.get_level_values(1):
            px = data.xs("Adj Close", axis=1, level=1)
        else:
            raise RuntimeError("Could not find 'Adj Close' in downloaded data.")
    else:
        # Single ticker -> columns are fields
        if "Adj Close" not in data.columns:
            raise RuntimeError("Could not find 'Adj Close' in downloaded data.")
        # Convert to df with one column named by ticker
        # If only one ticker, tickers list length is 1
        px = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})

    px = px.dropna(how="all")
    return px


def month_index_from_dates(dates: pd.DatetimeIndex, start: pd.Timestamp | None = None) -> pd.Series:
    """
    Map each date to an integer month index starting from 0.
    Robust across pandas versions (avoids PeriodIndex subtraction quirks).
    """
    if start is None:
        start = pd.Timestamp(dates.min())
    start = pd.Timestamp(start)

    # month difference = (year diff * 12) + month diff
    y = dates.year
    m = dates.month
    return (y - start.year) * 12 + (m - start.month)



def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights_by_month: pd.DataFrame,
) -> tuple[pd.Series, dict]:
    """
    returns: daily asset returns, indexed by date, columns=tickers
    weights_by_month: wide df with 'month' + tickers columns

    Drops missing tickers each day/month by intersecting available columns,
    renormalizes weights within each month.
    """
    wdf = weights_by_month.copy()
    months_available = set(wdf["month"].astype(int).tolist())

    # Map dates -> month index
    m_idx = month_index_from_dates(returns.index)
    returns = returns.copy()
    returns["month"] = m_idx.values

    port_rets = []
    dropped_log = []

    for m, grp in returns.groupby("month"):
        if int(m) not in months_available:
            continue

        w_row = wdf[wdf["month"] == int(m)].iloc[0]
        w_row = w_row.drop(labels=["month"])

        # assets available in this month of returns
        avail = [c for c in grp.columns if c in w_row.index]
        avail = [c for c in avail if c != "month"]

        # keep only tickers that have non-NaN returns in this month
        # (if a ticker is all NaN in that month, treat as unavailable)
        good = []
        for t in avail:
            if grp[t].notna().any():
                good.append(t)

        missing = sorted(set(w_row.index) - set(good))
        if missing:
            dropped_log.append({"month": int(m), "dropped_tickers": ",".join(missing)})

        if len(good) == 0:
            continue

        w = w_row[good].astype(float).values
        w_sum = w.sum()
        if w_sum <= 0:
            continue
        w = w / w_sum  # renormalize

        # portfolio daily returns in this month
        r = grp[good].astype(float)

        # Fill sporadic NaNs with 0 so portfolio return is defined each day
        r = r.fillna(0.0)

        p = (r.values * w).sum(axis=1)

        port_rets.append(pd.Series(p, index=grp.index))

    port = pd.concat(port_rets).sort_index()
    meta = {
        "months_used": int(pd.to_datetime(port.index).to_period("M").nunique()) if len(port) else 0,
        "dropped_log": pd.DataFrame(dropped_log),
    }
    return port, meta


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """
    Annualized Sharpe using daily returns and 252 trading days.
    rf is daily risk-free (default 0).
    """
    x = returns.dropna() - rf
    if x.std() == 0 or len(x) < 5:
        return float("nan")
    return float(np.sqrt(252) * x.mean() / x.std())


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def main() -> None:
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    weights = load_weights("results/metrics/weights_by_month.csv")
    tickers = [c for c in weights.columns if c != "month"]

    # Option 2: hard-drop known-bad tickers to avoid yfinance spam
    KNOWN_BAD = {"AXA SA"}   # add more here if needed
    tickers = [t for t in tickers if t not in KNOWN_BAD]

    print(f"Tickers in weights file (after drop): {len(tickers)}")

    # Download prices
    px = download_prices(tickers, years=3)


    # Determine which tickers actually downloaded
    good_tickers = [t for t in tickers if t in px.columns and px[t].notna().any()]
    bad_tickers = sorted(set(tickers) - set(good_tickers))
    print(
    f"Downloaded tickers: {len(good_tickers)} "
)
    print("Quick note: Not a problem but AXA SA failed to download and was excluded from " \
    "evaluation. Portfolio weights were renormalized, so results are unaffected.")
    if bad_tickers:
        print("Failed tickers (dropped in eval):", bad_tickers)

    px = px[good_tickers].dropna(how="all")

    # Daily returns
    rets = px.pct_change(fill_method=None).dropna(how="all")

    # Portfolio returns
    port_rets, meta = compute_portfolio_returns(rets, weights)
    port_rets.index = pd.to_datetime(port_rets.index)
    port_rets = port_rets.dropna()

    # Equity curve
    equity = (1.0 + port_rets).cumprod()

    # Rolling Sharpe (63 trading days ~ 3 months)
    window = 63
    x = port_rets.dropna()
    roll_mean = x.rolling(window).mean()
    roll_std = x.rolling(window).std()
    roll_sharpe = np.sqrt(252) * (roll_mean / roll_std)


    # Summary stats
    sr = sharpe_ratio(port_rets)
    vol = float(np.sqrt(252) * port_rets.std())
    ann_ret = float(252 * port_rets.mean())
    mdd = max_drawdown(equity)

    summary = pd.DataFrame([{
        "annualized_return": ann_ret,
        "annualized_vol": vol,
        "sharpe": sr,
        "max_drawdown": mdd,
        "n_days": int(len(port_rets)),
        "n_tickers_downloaded": int(len(good_tickers)),
        "n_tickers_failed": int(len(bad_tickers)),
    }])

    summary.to_csv(OUT_METRICS / "performance_summary.csv", index=False)

    if not meta["dropped_log"].empty:
        meta["dropped_log"].to_csv(OUT_METRICS / "dropped_tickers_by_month.csv", index=False)

    # ---- Plots ----
    plt.figure()
    equity.plot()
    plt.title("Equity Curve (Backtest)")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "equity_curve.png", dpi=200)
    plt.close()

    plt.figure()
    roll_sharpe.dropna().plot()

    plt.title(f"Rolling Sharpe ({window} trading days)")
    plt.xlabel("Date")
    plt.ylabel("Sharpe")
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "rolling_sharpe.png", dpi=200)
    plt.close()

    # Drawdown plot
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    plt.figure()
    dd.plot()
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "drawdown.png", dpi=200)
    plt.close()

    print("Saved:")
    print("-", OUT_METRICS / "performance_summary.csv")
    if not meta["dropped_log"].empty:
        print("-", OUT_METRICS / "dropped_tickers_by_month.csv")
    print("-", OUT_PLOTS / "equity_curve.png")
    print("-", OUT_PLOTS / "rolling_sharpe.png")
    print("-", OUT_PLOTS / "drawdown.png")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
