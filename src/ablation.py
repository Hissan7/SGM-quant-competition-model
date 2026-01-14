# src/ablation.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.optimiser import solve_weights, OptimizerConfig
from src.macro_controller import add_buckets

OUT_METRICS = Path("results/metrics")
OUT_PLOTS = Path("results/plots")


VARIANTS = {
    "baseline": dict(use_macro=False, use_uncertainty=False, use_ml=False),
    "macro_only": dict(use_macro=True, use_uncertainty=False, use_ml=False),
    "macro_robust": dict(use_macro=True, use_uncertainty=True, use_ml=False),
    "macro_robust_ml": dict(use_macro=True, use_uncertainty=True, use_ml=True),
}


def simulate_variant(df: pd.DataFrame, name: str, months: int, cfg: OptimizerConfig) -> tuple[pd.DataFrame, list[np.ndarray]]:
    flags = VARIANTS[name]
    n = len(df)
    w_prev = None
    rows = []
    weights = []

    for m in range(months):
        w, info = solve_weights(df, month=m, w_prev=w_prev, cfg=cfg, **flags)

        rows.append({
            "variant": name,
            "month": m,
            "status": info["status"],
            "objective": info["objective"],
            "turnover": info["turnover"],
            "w_defensive": info["w_defensive"],
            "w_growth_ai": info["w_growth_ai"],
            "w_cyclical": info["w_cyclical"],
            "concentration": float(np.sum(w*w)),
        })
        weights.append(w)
        w_prev = w

    return pd.DataFrame(rows), weights


def summarize(metrics_all: pd.DataFrame) -> pd.DataFrame:
    g = metrics_all.groupby("variant")
    summary = g.agg(
        mean_turnover=("turnover", "mean"),
        max_turnover=("turnover", "max"),
        mean_growth=("w_growth_ai", "mean"),
        final_growth=("w_growth_ai", lambda s: float(s.iloc[-1])),
        mean_defensive=("w_defensive", "mean"),
        mean_concentration=("concentration", "mean"),
        mean_objective=("objective", "mean"),
    ).reset_index()
    return summary.sort_values("mean_objective", ascending=False)


# def plot_growth_paths(metrics_all: pd.DataFrame) -> None:
#     plt.figure()
#     for name in VARIANTS:
#         sub = metrics_all[metrics_all["variant"] == name]
#         plt.plot(sub["month"], sub["w_growth_ai"], label=name)
#     plt.xlabel("Month")
#     plt.ylabel("Growth/AI weight")
#     plt.title("Growth Allocation Over Time (Ablation)")
#     plt.legend()
#     plt.tight_layout()
#     OUT_PLOTS.mkdir(parents=True, exist_ok=True)
#     plt.savefig(OUT_PLOTS / "ablation_growth_paths.png", dpi=200)
#     plt.close()

def plot_growth_paths(metrics_all: pd.DataFrame) -> None:
    plt.figure()

    order = ["baseline", "macro_only", "macro_robust", "macro_robust_ml"]
    styles = {
        "baseline": dict(marker="o", linewidth=2),
        "macro_only": dict(marker="s", linewidth=2),
        "macro_robust": dict(marker="^", linewidth=2),
        "macro_robust_ml": dict(marker="D", linewidth=2),
    }

    for name in order:
        sub = metrics_all[metrics_all["variant"] == name]
        plt.plot(sub["month"], sub["w_growth_ai"], label=name, **styles[name])

    plt.xlabel("Month")
    plt.ylabel("Growth/AI weight")
    plt.title("Growth Allocation Over Time (Ablation)")
    plt.legend()
    plt.tight_layout()
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PLOTS / "ablation_growth_paths.png", dpi=200)
    plt.close()

def plot_turnover_paths(metrics_all: pd.DataFrame) -> None:
    plt.figure()
    for name in VARIANTS:
        sub = metrics_all[metrics_all["variant"] == name]
        plt.plot(sub["month"], sub["turnover"], label=name)
    plt.axhline(0.30, linestyle="--", label="turnover limit")
    plt.xlabel("Month")
    plt.ylabel("Turnover")
    plt.title("Turnover Over Time (Ablation)")
    plt.legend()
    plt.tight_layout()
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PLOTS / "ablation_turnover_paths.png", dpi=200)
    plt.close()


def main() -> None:
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/processed/universe.csv")
    df = add_buckets(df)

    cfg = OptimizerConfig()
    months = 10

    all_metrics = []
    for name in VARIANTS:
        mdf, _w = simulate_variant(df, name=name, months=months, cfg=cfg)
        all_metrics.append(mdf)

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    summary = summarize(metrics_all)

    pivot = metrics_all.pivot_table(index="month", columns="variant", values="w_growth_ai")
    print("\nGrowth weights by month:\n", pivot.round(6))
    print("\nmacro_only - macro_robust:\n", (pivot["macro_only"] - pivot["macro_robust"]).round(8))


    metrics_all.to_csv(OUT_METRICS / "ablation_monthly.csv", index=False)
    summary.to_csv(OUT_METRICS / "ablation_summary.csv", index=False)

    plot_growth_paths(metrics_all)
    plot_turnover_paths(metrics_all)

    print("Saved:")
    print("-", OUT_METRICS / "ablation_monthly.csv")
    print("-", OUT_METRICS / "ablation_summary.csv")
    print("-", OUT_PLOTS / "ablation_growth_paths.png")
    print("-", OUT_PLOTS / "ablation_turnover_paths.png")
    print()
    print(summary)


if __name__ == "__main__":
    main()
