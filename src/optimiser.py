# src/optimiser.py
from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp

from src.macro_controller import add_buckets, macro_targets
from src.ml_signal import fit_predict_mu_sigma

# Suppress CVXPY "Solution may be inaccurate" warnings (SCS can emit these)
warnings.filterwarnings("ignore", message="Solution may be inaccurate*")


@dataclass(frozen=True)
class OptimizerConfig:
    w_min: float = 0.01
    w_max: float = 0.05
    turnover_max: float = 0.30
    tc_bps: float = 5.0  # transaction cost in bps per 1.0 turnover (5 bps = 0.05%)
    horizon_months: int = 9

    # Objective weights
    risk_aversion: float = 2.0
    uncertainty_aversion: float = 0.5
    turnover_penalty: float = 0.1

    # Optional bucket diversification (keeps diversification inside buckets)
    bucket_diversification: float = 0.2

    # Risk model integration
    risk_cov_aversion: float = 0.5  # penalty on portfolio variance (quad_form)
    vol_max: float = 0.10           # annualized vol cap (HARD)

    # Macro slack: only violate macro mins if necessary to satisfy risk cap
    macro_slack_penalty: float = 50.0  # big = only violate macro if necessary

    # dividend consraints 
    use_dividends: bool = True
    dividend_weight: float = 1.0 # 1.0 = full dividend yield contribution


def solve_weights(
    universe: pd.DataFrame,
    month: int,
    w_prev: np.ndarray | None = None,
    cfg: OptimizerConfig = OptimizerConfig(),
    use_macro: bool = True,
    use_uncertainty: bool = True,
    use_ml: bool = True,
    Sigma: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Returns:
      w_opt: optimal weights (N,)
      info: diagnostics dict
    """
    df = add_buckets(universe).copy().reset_index(drop=True)
    n = len(df)

    # --- Expected return (mu) + uncertainty (width) sources ---
    if use_ml:
        mu, sigma_ml, ml_meta = fit_predict_mu_sigma(df, alpha=1.0, n_boot=200)
    else:
        mu = df["target_return"].to_numpy(dtype=float)
        sigma_ml = None
        ml_meta = None

    # --- Dividends: add a dividend yield component to expected return ---
    # Note: mu is treated as "per-period expected return" in your optimiser.
    # We'll convert annual dividend yield to a monthly contribution.
    if cfg.use_dividends and "dividend_yield" in df.columns:
        div_y = df["dividend_yield"].to_numpy(dtype=float)
        div_y = np.nan_to_num(div_y, nan=0.0, posinf=0.0, neginf=0.0)

        # annual yield -> approx monthly yield
        mu = mu + cfg.dividend_weight * (div_y / 12.0)

    if use_uncertainty:
        if use_ml:
            width = sigma_ml
        else:
            width = df["range_width"].to_numpy(dtype=float)
    else:
        width = np.zeros(n)

    # NaN-safe
    if np.all(np.isnan(width)):
        width = np.zeros(n)
    else:
        med = np.nanmedian(width)
        width = np.where(np.isnan(width), med, width)

    # Previous weights (equal-weight start if none)
    if w_prev is None:
        w_prev = np.ones(n) / n
        w_prev = np.clip(w_prev, cfg.w_min, cfg.w_max)
        w_prev = w_prev / w_prev.sum()
    else:
        w_prev = np.asarray(w_prev, dtype=float)
        if w_prev.shape != (n,):
            raise ValueError(f"w_prev shape {w_prev.shape} does not match n={n}")

    # Decision variable
    w = cp.Variable(n)

    # Turnover (L1)
    turnover = cp.norm1(w - w_prev)

    # Bucket indices
    idx_def = np.where(df["Bucket"].values == "DEFENSIVE")[0]
    idx_gro = np.where(df["Bucket"].values == "GROWTH_AI")[0]
    idx_cyc = np.where(df["Bucket"].values == "CYCLICAL_NEUTRAL")[0]

    w_def = cp.sum(w[idx_def]) if len(idx_def) else 0
    w_gro = cp.sum(w[idx_gro]) if len(idx_gro) else 0
    w_cyc = cp.sum(w[idx_cyc]) if len(idx_cyc) else 0

    # Base constraints (always on)
    constraints = [
        cp.sum(w) == 1.0,
        w >= cfg.w_min,
        w <= cfg.w_max,
        turnover <= cfg.turnover_max,
        w_cyc <= 0.60,  # optional sanity
    ]

    # ---- HARD volatility constraint ----
    # annualized variance <= vol_max^2
    var_annual = None
    if Sigma is not None:
        Sigma = np.asarray(Sigma, dtype=float)
        if Sigma.shape != (n, n):
            raise ValueError(f"Sigma shape {Sigma.shape} does not match (n,n)=({n},{n})")

        var_annual = 252.0 * cp.quad_form(w, Sigma)
        constraints += [var_annual <= (cfg.vol_max ** 2)]

    # Caps implied by position bounds
    growth_cap = len(idx_gro) * cfg.w_max
    defensive_cap = len(idx_def) * cfg.w_max
    cyclical_cap = len(idx_cyc) * cfg.w_max

    # Macro targets & feasibility-aware clipping
    mt = None
    defensive_min_feasible = None
    growth_min_feasible = None
    growth_max_feasible = None

    # Macro slack vars default to "off"
    macro_slack_gro = None
    macro_slack_def = None

    if use_macro:
        mt = macro_targets(month, horizon_months=cfg.horizon_months)

        if len(idx_gro) == 0:
            raise ValueError("No GROWTH_AI assets found; cannot enforce growth constraints.")
        if len(idx_def) == 0:
            raise ValueError("No DEFENSIVE assets found; cannot enforce defensive constraints.")

        growth_max_feasible = min(mt.growth_max, max(growth_cap - 1e-6, 0.0))
        growth_min_feasible = min(mt.growth_min, max(growth_cap - 1e-6, 0.0))
        defensive_min_feasible = min(mt.defensive_min, max(defensive_cap - 1e-6, 0.0))

        # Ensure ordering (min <= max) and non-negativity
        growth_max_feasible = max(growth_max_feasible, 0.0)
        growth_min_feasible = max(min(growth_min_feasible, growth_max_feasible - 1e-6), 0.0)
        defensive_min_feasible = max(defensive_min_feasible, 0.0)

        # SOFT mins via slack (macro can bend to satisfy vol cap)
        macro_slack_gro = cp.Variable(nonneg=True)
        macro_slack_def = cp.Variable(nonneg=True)

        constraints += [
            w_gro + macro_slack_gro >= growth_min_feasible,
            w_def + macro_slack_def >= defensive_min_feasible,
            w_gro <= growth_max_feasible,  # keep max hard
        ]

    # Risk proxy: L2 encourages diversification
    concentration_risk = cp.sum_squares(w)

    # Bucket diversification: discourage overloading within a bucket
    bucket_risk = 0
    if len(idx_def):
        bucket_risk += cp.sum_squares(w[idx_def])
    if len(idx_gro):
        bucket_risk += cp.sum_squares(w[idx_gro])
    if len(idx_cyc):
        bucket_risk += cp.sum_squares(w[idx_cyc])

    # Transaction costs approximation
    tc = (cfg.tc_bps / 10000.0) * turnover

    # Objective: maximize robust return - penalties
    robust_return = mu @ w - cfg.uncertainty_aversion * (width @ w)

    # Covariance risk penalty (mean-variance style)
    risk_term = 0
    if Sigma is not None:
        risk_term = cfg.risk_cov_aversion * cp.quad_form(w, Sigma)

    # Macro slack penalty only if macro is enabled
    macro_penalty = 0
    if use_macro and macro_slack_gro is not None and macro_slack_def is not None:
        macro_penalty = cfg.macro_slack_penalty * (macro_slack_gro + macro_slack_def)

    objective = cp.Maximize(
        robust_return
        - cfg.risk_aversion * concentration_risk
        - cfg.bucket_diversification * bucket_risk
        - tc
        - cfg.turnover_penalty * turnover
        - risk_term
        - macro_penalty
    )

    # Solve
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except cp.error.SolverError:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except cp.error.SolverError:
            raise RuntimeError(
                "No conic solver available. Install one of: ecos or scs.\n"
                "Try: pip install ecos scs"
            )

    if w.value is None:
        diag = {}
        if Sigma is not None:
            diag["prev_vol_annual"] = float(np.sqrt(252.0 * (w_prev @ Sigma @ w_prev)))
        diag["macro_targets"] = None if mt is None else (mt.defensive_min, mt.growth_min, mt.growth_max)
        diag["feasible_targets"] = (defensive_min_feasible, growth_min_feasible, growth_max_feasible)

        raise RuntimeError(
            f"Optimization failed at month={month}. "
            f"caps=(growth={growth_cap:.3f}, def={defensive_cap:.3f}), "
            f"use_macro={use_macro}, use_uncertainty={use_uncertainty}, use_ml={use_ml}, "
            f"diag={diag}"
        )

    # Extract weights
    w_opt = np.array(w.value).reshape(-1)

    # Clean numeric noise + enforce strict bounds for nice printing
    w_opt = np.clip(w_opt, cfg.w_min, cfg.w_max)
    w_opt = w_opt / w_opt.sum()

    pred_vol = None
    if Sigma is not None:
        pred_vol = float(np.sqrt(252.0 * (w_opt @ Sigma @ w_opt)))

    macro_slack_vals = {"growth_min_slack": 0.0, "defensive_min_slack": 0.0}
    if use_macro and macro_slack_gro is not None and macro_slack_gro.value is not None:
        macro_slack_vals["growth_min_slack"] = float(macro_slack_gro.value)
    if use_macro and macro_slack_def is not None and macro_slack_def.value is not None:
        macro_slack_vals["defensive_min_slack"] = float(macro_slack_def.value)

    info = {
        "status": prob.status,
        "objective": float(prob.value),
        "turnover": float(np.sum(np.abs(w_opt - w_prev))),
        "w_defensive": float(w_opt[idx_def].sum()) if len(idx_def) else 0.0,
        "w_growth_ai": float(w_opt[idx_gro].sum()) if len(idx_gro) else 0.0,
        "w_cyclical": float(w_opt[idx_cyc].sum()) if len(idx_cyc) else 0.0,
        "variant_flags": {"use_macro": use_macro, "use_uncertainty": use_uncertainty, "use_ml": use_ml},
        "macro_targets": mt,
        "macro_slack": macro_slack_vals,
        "feasible_targets": {
            "growth_cap": float(growth_cap),
            "defensive_cap": float(defensive_cap),
            "cyclical_cap": float(cyclical_cap),
            "defensive_min": None if defensive_min_feasible is None else float(defensive_min_feasible),
            "growth_min": None if growth_min_feasible is None else float(growth_min_feasible),
            "growth_max": None if growth_max_feasible is None else float(growth_max_feasible),
        },
        "pred_vol_annual": pred_vol,
        "vol_max": cfg.vol_max if Sigma is not None else None,
        "ml_meta": ml_meta,
    }

    return w_opt, info
