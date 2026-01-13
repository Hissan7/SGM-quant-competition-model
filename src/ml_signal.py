# src/ml_signal.py
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Returns:
      X_df: feature dataframe (not yet encoded)
      y: target vector (target_return)
    """
    # Target
    y = df["target_return"].to_numpy(dtype=float)

    # Feature engineering (simple, interpretable)
    # - range_width: analyst disagreement proxy
    # - range_mid_return: midpoint implied return (if available)
    # - region: US/EU
    # - industry: categorical
    X_df = pd.DataFrame(
        {
            "range_width": df["range_width"].astype(float),
            "range_mid_return": df["range_mid_return"].astype(float),
            "region": df["Region"].astype(str),
            "industry": df["Industry"].astype(str),
        }
    )

    # Fill any missing numeric values
    for col in ["range_width", "range_mid_return"]:
        if X_df[col].isna().any():
            X_df[col] = X_df[col].fillna(X_df[col].median())

    return X_df, y


def fit_predict_mu_sigma(
    universe: pd.DataFrame,
    alpha: float = 1.0,
    n_boot: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit a simple ML model (Ridge) to predict target_return and estimate uncertainty
    via bootstrap model dispersion.

    Returns:
      mu_hat: predicted expected return (N,)
      sigma_hat: uncertainty estimate (N,) from bootstrap std-dev
      meta: dict with diagnostics
    """
    df = universe.copy().reset_index(drop=True)
    n = len(df)

    X_df, y = build_features(df)

    # Preprocess: one-hot encode categorical features
    cat_cols = ["region", "industry"]
    num_cols = ["range_width", "range_mid_return"]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = Ridge(alpha=alpha, fit_intercept=True)

    pipe = Pipeline([("pre", pre), ("model", model)])

    # Fit on full sample to get central prediction
    pipe.fit(X_df, y)
    mu_hat = pipe.predict(X_df).astype(float)

    # Bootstrap for uncertainty
    rng = np.random.default_rng(seed)
    preds = np.zeros((n_boot, n), dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        X_b = X_df.iloc[idx]
        y_b = y[idx]
        pipe_b = Pipeline([("pre", pre), ("model", Ridge(alpha=alpha, fit_intercept=True))])
        pipe_b.fit(X_b, y_b)
        preds[b] = pipe_b.predict(X_df)

    sigma_hat = preds.std(axis=0)

    meta = {
        "alpha": alpha,
        "n_boot": n_boot,
        "mu_hat_mean": float(np.mean(mu_hat)),
        "mu_hat_std": float(np.std(mu_hat)),
        "sigma_hat_mean": float(np.mean(sigma_hat)),
        "sigma_hat_std": float(np.std(sigma_hat)),
    }

    return mu_hat, sigma_hat, meta
