# src/simulate.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.macro_controller import add_buckets

# IMPORTANT: match your filename
from src.optimiser import solve_weights, OptimizerConfig
# If your file is src/optimizer.py then use:
# from src.optimizer import solve_weights, OptimizerConfig


RESULTS_METRICS = Path("results/metrics")
RESULTS_PLOTS = Path("results/plots")


def run_simulation(
    universe_csv: str = "data/processed/universe.csv",
    months: int = 10,  # months 0..9
    cfg: OptimizerConfig = OptimizerConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(universe_csv)
    df = add_buckets(df)

    n = len(df)
    w_prev = None

    rows = []
    weights_over_time = []

    for m in range(months):
        w, info = solve_weights(df, month=m, w_prev=w_prev, cfg=cfg)

        # Store metrics
        rows.append(
            {
                "month": m,
                "status": info["status"],
                "objective": info["objective"],
                "turnover": info["turnover"],
                "w_defensive": info["w_defensive"],
                "w_growth_ai": info["w_growth_ai"],
                "w_cyclical": info["w_cyclical"],
                "defensive_min": info["macro_targets"].defensive_min,
                "growth_min": info["macro_targets"].growth_min,
                "growth_max": info["macro_targets"].growth_max,
            }
        )

        # Store weights for later inspection
        weights_over_time.append(w)

        # update
        w_prev = w

    metrics = pd.DataFrame(rows)

    weights_df = pd.DataFrame(
        np.vstack(weights_over_time),
        columns=[f"{t}" for t in df["Ticker"].tolist()],
    )
    weights_df.insert(0, "month", list(range(months)))

    return metrics, weights_df


def plot_rotation(metrics: pd.DataFrame) -> None:
    plt.figure()
    plt.plot(metrics["month"], metrics["w_defensive"], label="Defensive weight")
    plt.plot(metrics["month"], metrics["w_growth_ai"], label="Growth/AI weight")
    plt.plot(metrics["month"], metrics["w_cyclical"], label="Cyclical/Neutral weight")
    plt.xlabel("Month")
    plt.ylabel("Portfolio weight")
    plt.title("Macro Rotation (Defensive → Growth)")
    plt.legend()
    plt.tight_layout()

    RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_PLOTS / "rotation.png", dpi=200)
    plt.close()


def plot_turnover(metrics: pd.DataFrame) -> None:
    plt.figure()
    plt.plot(metrics["month"], metrics["turnover"], label="Turnover")
    plt.axhline(0.30, linestyle="--", label="Turnover limit (30%)")
    plt.xlabel("Month")
    plt.ylabel("Turnover (L1 change in weights)")
    plt.title("Turnover Over Time")
    plt.legend()
    plt.tight_layout()

    RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_PLOTS / "turnover.png", dpi=200)
    plt.close()


def main() -> None:
    RESULTS_METRICS.mkdir(parents=True, exist_ok=True)
    RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)

    metrics, weights_df = run_simulation()

    metrics.to_csv(RESULTS_METRICS / "simulation.csv", index=False)
    weights_df.to_csv(RESULTS_METRICS / "weights_by_month.csv", index=False)

    plot_rotation(metrics)
    plot_turnover(metrics)

    print("Saved:")
    print(f"- {RESULTS_METRICS / 'simulation.csv'}")
    print(f"- {RESULTS_METRICS / 'weights_by_month.csv'}")
    print(f"- {RESULTS_PLOTS / 'rotation.png'}")
    print(f"- {RESULTS_PLOTS / 'turnover.png'}")
    print()
    print(metrics[["month", "turnover", "w_defensive", "w_growth_ai", "w_cyclical"]])


if __name__ == "__main__":
    main()
