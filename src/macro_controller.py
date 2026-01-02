# src/macro_controller.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


DEFENSIVE = {
    "Healthcare",
    "Biotechnology",
    "Consumer Staples",
    "Consumer Staples (alchol)",
    "Telecommunications",
    "Insurance",
    "Finance",
}

GROWTH_AI = {
    "Technology",
    "Technology (AI/Computing)",
    "Fintech",
    "Energy/Data-Centers",
    "Industrials Components (Data centers)",
}

CYCLICAL_NEUTRAL = {
    "Oil and Gas",
    "Energy",
    "Consumer Cyclical",
    "Construction Materials",
    "Construction and Engineering",
    "Industrials/Textile",
}

# Anything not listed falls into CYCLICAL_NEUTRAL by default (safe)


def industry_bucket(industry: str) -> str:
    ind = str(industry).strip()
    if ind in DEFENSIVE:
        return "DEFENSIVE"
    if ind in GROWTH_AI:
        return "GROWTH_AI"
    if ind in CYCLICAL_NEUTRAL:
        return "CYCLICAL_NEUTRAL"
    return "CYCLICAL_NEUTRAL"


def rotation_g(month: int, horizon_months: int = 9) -> float:
    """
    Smooth rotation signal in [0,1] over horizon_months.
    month=0 => 0.0 (fully defensive tilt)
    month=horizon_months => ~1.0 (fully growth tilt)
    Uses a logistic curve for smoothness.
    """
    # keep it stable for small universes
    x = (month / max(horizon_months, 1)) * 12 - 6  # map to [-6, +6]
    g = 1.0 / (1.0 + np.exp(-x))
    return float(g)


@dataclass(frozen=True)
class MacroTargets:
    defensive_min: float
    growth_max: float
    growth_min: float


def macro_targets(month: int, horizon_months: int = 9) -> MacroTargets:
    """
    Define portfolio-level constraints that evolve with g(t).

    Early months:
      - high defensive minimum
      - low growth maximum
    Late months:
      - relaxed defensive minimum
      - higher growth minimum/maximum
    """
    g = rotation_g(month, horizon_months)

    # Defensive min starts high (~60%) then fades to ~20%
    defensive_min = 0.60 * (1 - g) + 0.20 * g

    # Growth max starts low (~25%) then rises to ~70%
    growth_max = 0.25 * (1 - g) + 0.70 * g

    # Optional: ensure some growth later (~10% early -> ~35% late)
    growth_min = 0.10 * (1 - g) + 0.35 * g

    return MacroTargets(
        defensive_min=float(defensive_min),
        growth_max=float(growth_max),
        growth_min=float(growth_min),
    )


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Bucket"] = out["Industry"].apply(industry_bucket)
    return out
