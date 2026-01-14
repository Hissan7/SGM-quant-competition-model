# src/risk_model.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import yfinance as yf
import contextlib,io

@dataclass(frozen=True)
class RiskModelConfig:
    years: int = 3
    ewma_lambda: float = 0.94      # RiskMetrics-style decay
    min_obs: int = 60              # minimum daily observations
    jitter: float = 1e-6           # makes Sigma PSD for cvxpy stability


def download_adj_close(tickers: list[str], years: int = 3) -> pd.DataFrame:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        data = yf.download(
        tickers=tickers,
        period=f"{years}y",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            px = data["Adj Close"]
        elif "Adj Close" in data.columns.get_level_values(1):
            px = data.xs("Adj Close", axis=1, level=1)
        else:
            raise RuntimeError("Could not find 'Adj Close' in downloaded data.")
    else:
        if "Adj Close" not in data.columns:
            raise RuntimeError("Could not find 'Adj Close' in downloaded data.")
        px = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})

    px = px.dropna(how="all")
    return px


def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    """
    EWMA covariance estimate.
    returns: T x N (daily), demeaned inside
    """
    X = returns.to_numpy(dtype=float)
    X = X - np.nanmean(X, axis=0, keepdims=True)

    # Start from sample covariance (nan-safe via pandas)
    S = np.asarray(returns.cov().to_numpy(), dtype=float)

    for t in range(X.shape[0]):
        x = X[t : t + 1].T  # N x 1
        if np.isnan(x).any():
            # skip days with missing data (we'll handle missing by aligning later)
            continue
        S = lam * S + (1.0 - lam) * (x @ x.T)

    return S


def build_sigma_for_universe(
    tickers: list[str],
    cfg: RiskModelConfig = RiskModelConfig(),
) -> tuple[np.ndarray, dict]:
    """
    Returns an NxN covariance matrix aligned to `tickers`.
    If a ticker has no data, it gets a small standalone diagonal variance.
    """
    px = download_adj_close(tickers, years=cfg.years)

    good = [t for t in tickers if t in px.columns and px[t].notna().any()]
    bad = sorted(set(tickers) - set(good))

    # Keep only good tickers for return estimation
    px_good = px[good].dropna(how="all")
    rets = px_good.pct_change(fill_method=None).dropna(how="all")

    N = len(tickers)

    # Fallback: if too little data, use diagonal covariance
    if len(rets) < cfg.min_obs or len(good) < 2:
        Sigma = np.eye(N) * 0.0001
        return Sigma, {"mode": "fallback_diagonal", "n_obs": int(len(rets)), "bad": bad, "good": good}

    # Covariance on good tickers
    Sigma_good = ewma_cov(rets, lam=cfg.ewma_lambda)
    Sigma_good = Sigma_good + np.eye(Sigma_good.shape[0]) * cfg.jitter

    # Expand to full NxN aligned to original tickers order
    Sigma = np.zeros((N, N), dtype=float)
    idx_map = {t: i for i, t in enumerate(tickers)}
    good_idx = [idx_map[t] for t in good]

    # Place Sigma_good into the right block
    for ii, i in enumerate(good_idx):
        for jj, j in enumerate(good_idx):
            Sigma[i, j] = Sigma_good[ii, jj]

    # Give missing tickers a small diagonal variance
    small = 0.0001
    for t in bad:
        i = idx_map[t]
        Sigma[i, i] = small

    # Ensure diagonals are positive
    for i in range(N):
        if Sigma[i, i] <= 0:
            Sigma[i, i] = small

    meta = {"mode": "ewma", "n_obs": int(len(rets)), "bad": bad, "good": good}
    return Sigma, meta
