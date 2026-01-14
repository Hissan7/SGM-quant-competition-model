# src/simulate.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from src.macro_controller import add_buckets
from src.optimiser import solve_weights, OptimizerConfig
from src.risk_model import build_sigma_for_universe, RiskModelConfig
from src.dividends import dividend_yield_ttm

RESULTS_METRICS = Path("results/metrics")
RESULTS_PLOTS = Path("results/plots")


def run_simulation(
    universe_csv: str = "data/processed/universe.csv",
    months: int = 10,  # months 0..9
    cfg: OptimizerConfig = OptimizerConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(universe_csv)
    df = add_buckets(df)

    w_prev = None
    rows = []
    weights_over_time = []

    tickers = df["Ticker"].tolist()
    print("Universe tickers:", len(tickers), tickers)

    div_y, div_meta = dividend_yield_ttm(tickers, period="1y")
    print("Dividend yields meta:", div_meta)
    df["dividend_yield"] = df["Ticker"].map(div_y).fillna(0.0)  

    print("Dividend yields :")
    print(df[["Ticker", "dividend_yield"]].sort_values("dividend_yield", ascending=False))

    Sigma, sigma_meta = build_sigma_for_universe(tickers, RiskModelConfig(years=3))
    print("Sigma shape:", Sigma.shape)
    print("Risk model:", sigma_meta)

    for m in range(months):
        w, info = solve_weights(df, month=m, w_prev=w_prev, cfg=cfg, Sigma=Sigma)

        mt = info.get("macro_targets", None)
        slack = info.get("macro_slack", {}) or {}

        rows.append(
            {
                "month": m,
                "status": info.get("status"),
                "objective": info.get("objective"),
                "turnover": info.get("turnover"),
                "w_defensive": info.get("w_defensive"),
                "w_growth_ai": info.get("w_growth_ai"),
                "w_cyclical": info.get("w_cyclical"),
                "defensive_min": None if mt is None else mt.defensive_min,
                "growth_min": None if mt is None else mt.growth_min,
                "growth_max": None if mt is None else mt.growth_max,
                "pred_vol_annual": info.get("pred_vol_annual", None),
                "vol_max": info.get("vol_max", None),
                "growth_min_slack": float(slack.get("growth_min_slack", 0.0)),
                "defensive_min_slack": float(slack.get("defensive_min_slack", 0.0)),
            }
        )

        weights_over_time.append(w)
        w_prev = w

    metrics = pd.DataFrame(rows)

    weights_df = pd.DataFrame(
        np.vstack(weights_over_time),
        columns=df["Ticker"].tolist(),
    )
    weights_df.insert(0, "month", list(range(months)))

    return metrics, weights_df, df


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


def plot_predicted_vol(metrics: pd.DataFrame) -> None:
    if "pred_vol_annual" not in metrics.columns:
        return

    x = metrics.dropna(subset=["pred_vol_annual"]).copy()
    if len(x) == 0:
        return

    cap = float(x["vol_max"].dropna().iloc[0]) if "vol_max" in x.columns and x["vol_max"].notna().any() else 0.10

    plt.figure()
    plt.plot(x["month"], x["pred_vol_annual"], label="Predicted vol (annualized)")
    plt.axhline(cap, linestyle="--", label=f"Vol cap ({cap:.0%})")

    # Macro slack markers (optional)
    if "growth_min_slack" in x.columns and "defensive_min_slack" in x.columns:
        used = x[(x["growth_min_slack"] > 1e-8) | (x["defensive_min_slack"] > 1e-8)]
        if len(used) > 0:
            plt.scatter(used["month"], used["pred_vol_annual"], label="Macro slack used")

    ax = plt.gca()
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)  # kills 1e-7 + 1e-1
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))  # show as %
    y_top = max(cap * 1.2, float(x["pred_vol_annual"].max()) + 0.01)
    ax.set_ylim(0.0, y_top)

    plt.xlabel("Month")
    plt.ylabel("Volatility")
    plt.title("Predicted Portfolio Volatility (Risk Model)")
    plt.legend()
    plt.tight_layout()

    RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_PLOTS / "predicted_vol.png", dpi=200)
    plt.close()
    #debug statement 
    print(metrics[["month","pred_vol_annual","vol_max","growth_min_slack","defensive_min_slack"]])

def plot_dividends_by_bucket(df: pd.DataFrame) -> None:
    if "dividend_yield" not in df.columns:
        return

    bucket_means = (
        df.groupby("Bucket")["dividend_yield"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure()
    ax = bucket_means.plot(kind="bar")

    ax.set_ylabel("Average Dividend Yield")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_title("Average Dividend Yield by Sector Bucket")

    plt.tight_layout()
    RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_PLOTS / "dividend_by_bucket.png", dpi=200)
    plt.close()


def main() -> None:
    RESULTS_METRICS.mkdir(parents=True, exist_ok=True)
    RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)

    metrics, weights_df, df = run_simulation()

    metrics.to_csv(RESULTS_METRICS / "simulation.csv", index=False)
    weights_df.to_csv(RESULTS_METRICS / "weights_by_month.csv", index=False)

    plot_rotation(metrics)
    plot_turnover(metrics)
    plot_predicted_vol(metrics)
    plot_dividends_by_bucket(df)

    print("Saved:")
    print(f"- {RESULTS_METRICS / 'simulation.csv'}")
    print(f"- {RESULTS_METRICS / 'weights_by_month.csv'}")
    print(f"- {RESULTS_PLOTS / 'rotation.png'}")
    print(f"- {RESULTS_PLOTS / 'turnover.png'}")
    print(f"- {RESULTS_PLOTS / 'predicted_vol.png'}")
    print()
    print("note:Macro minimum relaxed in months 5–9 due to vol cap feasibility.")
    print()
    print(metrics[["month", "turnover", "w_defensive", "w_growth_ai", "w_cyclical"]])


if __name__ == "__main__":
    main()
