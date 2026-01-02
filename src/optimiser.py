# src/optimizer.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import cvxpy as cp

from src.macro_controller import add_buckets, macro_targets


@dataclass(frozen=True)
class OptimizerConfig:
    w_min: float = 0.01
    w_max: float = 0.05
    turnover_max: float = 0.30
    tc_bps: float = 5.0  # transaction cost in basis points per 1.0 turnover (5 bps = 0.05%)
    horizon_months: int = 9

    # Objective weights
    risk_aversion: float = 2.0          # penalize concentration (L2)
    uncertainty_aversion: float = 0.5   # penalize analyst range width
    turnover_penalty: float = 0.1       # extra penalty beyond hard turnover constraint

    # Optional bucket concentration penalty (keeps diversification inside buckets)
    bucket_diversification: float = 0.2


def solve_weights(
    universe: pd.DataFrame,
    month: int,
    w_prev: np.ndarray | None = None,
    cfg: OptimizerConfig = OptimizerConfig(),
) -> tuple[np.ndarray, dict]:
    df = add_buckets(universe).copy().reset_index(drop=True)
    n = len(df)

    mu = df["target_return"].to_numpy(dtype=float)

    width = df["range_width"].to_numpy(dtype=float)
    if np.all(np.isnan(width)):
        width = np.zeros(n)
    else:
        med = np.nanmedian(width)
        width = np.where(np.isnan(width), med, width)

    if w_prev is None:
        w_prev = np.ones(n) / n
        w_prev = np.clip(w_prev, cfg.w_min, cfg.w_max)
        w_prev = w_prev / w_prev.sum()
    else:
        w_prev = np.asarray(w_prev, dtype=float)
        if w_prev.shape != (n,):
            raise ValueError(f"w_prev shape {w_prev.shape} does not match n={n}")

    w = cp.Variable(n)
    turnover = cp.norm1(w - w_prev)

    idx_def = np.where(df["Bucket"].values == "DEFENSIVE")[0]
    idx_gro = np.where(df["Bucket"].values == "GROWTH_AI")[0]
    idx_cyc = np.where(df["Bucket"].values == "CYCLICAL_NEUTRAL")[0]

    w_def = cp.sum(w[idx_def]) if len(idx_def) else 0
    w_gro = cp.sum(w[idx_gro]) if len(idx_gro) else 0
    w_cyc = cp.sum(w[idx_cyc]) if len(idx_cyc) else 0

    # --- Feasibility-aware macro targets (CRITICAL FIX) ---
    mt = macro_targets(month, horizon_months=cfg.horizon_months)

    growth_cap = len(idx_gro) * cfg.w_max
    defensive_cap = len(idx_def) * cfg.w_max

    growth_max_feasible = min(mt.growth_max, growth_cap - 1e-6)
    growth_min_feasible = min(mt.growth_min, growth_cap - 1e-6)
    defensive_min_feasible = min(mt.defensive_min, defensive_cap - 1e-6)

    # ensure min <= max
    growth_min_feasible = min(growth_min_feasible, growth_max_feasible - 1e-6)

    constraints = [
        cp.sum(w) == 1.0,
        w >= cfg.w_min,
        w <= cfg.w_max,
        turnover <= cfg.turnover_max,

        # Macro rotation constraints (feasible)
        w_def >= defensive_min_feasible,
        w_gro >= growth_min_feasible,
        w_gro <= growth_max_feasible,

        # keep cyclical bounded
        w_cyc <= 0.60,
    ]

    concentration_risk = cp.sum_squares(w)

    bucket_risk = 0
    if len(idx_def):
        bucket_risk += cp.sum_squares(w[idx_def])
    if len(idx_gro):
        bucket_risk += cp.sum_squares(w[idx_gro])
    if len(idx_cyc):
        bucket_risk += cp.sum_squares(w[idx_cyc])

    tc = (cfg.tc_bps / 10000.0) * turnover

    robust_return = mu @ w - cfg.uncertainty_aversion * (width @ w)

    objective = cp.Maximize(
        robust_return
        - cfg.risk_aversion * concentration_risk
        - cfg.bucket_diversification * bucket_risk
        - tc
        - cfg.turnover_penalty * turnover
    )

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except cp.error.SolverError:
        prob.solve(solver=cp.OSQP, verbose=False)

    if w.value is None:
        raise RuntimeError(
            f"Optimization failed at month={month}. "
            f"growth_cap={growth_cap:.3f}, def_cap={defensive_cap:.3f}, "
            f"targets=(def_min={mt.defensive_min:.3f}, gro_min={mt.growth_min:.3f}, gro_max={mt.growth_max:.3f})"
        )

    w_opt = np.array(w.value).reshape(-1)
    w_opt = np.clip(w_opt, 0, 1)
    w_opt = w_opt / w_opt.sum()

    info = {
        "status": prob.status,
        "objective": float(prob.value),
        "turnover": float(np.sum(np.abs(w_opt - w_prev))),
        "w_defensive": float(w_opt[idx_def].sum()) if len(idx_def) else 0.0,
        "w_growth_ai": float(w_opt[idx_gro].sum()) if len(idx_gro) else 0.0,
        "w_cyclical": float(w_opt[idx_cyc].sum()) if len(idx_cyc) else 0.0,
        "macro_targets": mt,
        "feasible_targets": {
            "defensive_min": defensive_min_feasible,
            "growth_min": growth_min_feasible,
            "growth_max": growth_max_feasible,
            "growth_cap": float(growth_cap),
        },
    }

    return w_opt, info
