# src/validate.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def time_series_cv(
    input_csv: str,
    target_col: str,
    out_metrics: str = "results/cv_metrics.json",
    n_splits: int = 5,
) -> dict:
    df = pd.read_csv(input_csv)

    # Primera columna = timestamp
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)

    features = [
        "hour",
        "dayofweek",
        "month",
        "lag_1",
        "lag_24",
        "roll_mean_24",
        "roll_std_24",
    ]

    missing = [c for c in features + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    X = df[features].copy()
    y = df[target_col].copy()

    tscv = TimeSeriesSplit(n_splits=n_splits)

    folds = []
    maes, rmses = [], []
    maes_base, rmses_base = [], []

    for i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        y_pred_base = y_test.shift(24).dropna()
        y_test_base = y_test.iloc[24:]
        mae_b = mean_absolute_error(y_test_base, y_pred_base)
        rmse_b = np.sqrt(mean_squared_error(y_test_base, y_pred_base))

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        maes.append(mae)
        rmses.append(rmse)
        maes_base.append(mae_b)
        rmses_base.append(rmse_b)

        folds.append(
            {
                "fold": i,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "baseline_naive_24h": {"mae": float(round(mae_b, 3)), "rmse": float(round(rmse_b, 3))},
                "random_forest": {"mae": float(round(mae, 3)), "rmse": float(round(rmse, 3))},
            }
        )

    def _stats(arr):
        return {
            "mean": float(round(float(np.mean(arr)), 3)),
            "std": float(round(float(np.std(arr)), 3)),
            "min": float(round(float(np.min(arr)), 3)),
            "max": float(round(float(np.max(arr)), 3)),
        }

    report = {
        "target_col": target_col,
        "n_splits": n_splits,
        "random_forest": {"mae": _stats(maes), "rmse": _stats(rmses)},
        "baseline_naive_24h": {"mae": _stats(maes_base), "rmse": _stats(rmses_base)},
        "folds": folds,
    }

    Path(out_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", default="data/processed/opsd_preprocessed.csv")
    p.add_argument("--target_col", default="AT_load_actual_entsoe_transparency")
    p.add_argument("--n_splits", type=int, default=5)
    args = p.parse_args()

    rep = time_series_cv(
        input_csv=args.input_csv,
        target_col=args.target_col,
        n_splits=args.n_splits,
    )

    print("Validación temporal completada")
    print(f" - Splits: {rep['n_splits']}")
    print(" - Random Forest (MAE mean±std): "
          f"{rep['random_forest']['mae']['mean']} ± {rep['random_forest']['mae']['std']}")
    print(" - Random Forest (RMSE mean±std): "
          f"{rep['random_forest']['rmse']['mean']} ± {rep['random_forest']['rmse']['std']}")
    print(" - Baseline naive (MAE mean±std): "
          f"{rep['baseline_naive_24h']['mae']['mean']} ± {rep['baseline_naive_24h']['mae']['std']}")
    print(" - Baseline naive (RMSE mean±std): "
          f"{rep['baseline_naive_24h']['rmse']['mean']} ± {rep['baseline_naive_24h']['rmse']['std']}")
    print(" - Métricas guardadas en: results/cv_metrics.json")