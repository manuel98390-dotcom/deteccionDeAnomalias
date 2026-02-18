# src/forecast.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


def train_forecast_model(
    input_csv: str,
    target_col: str,
    out_model: str = "models/forecast_model.joblib",
    out_predictions: str = "results/forecast_predictions.csv",
    out_metrics: str = "results/forecast_metrics.json",
) -> dict:
    df = pd.read_csv(input_csv)

    # Primera columna = timestamp
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)

    if target_col not in df.columns:
        raise ValueError(f"La columna '{target_col}' no existe en el CSV.")

    features = [
        "hour",
        "dayofweek",
        "month",
        "lag_1",
        "lag_24",
        "roll_mean_24",
        "roll_std_24",
    ]

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para forecasting: {missing}")

    X = df[features].copy()
    y = df[target_col].copy()

    # División temporal (80% train, 20% test)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    
    # BASELINE: Naive estacional (t-24)
    
    y_pred_naive = y_test.shift(24)
    y_pred_naive = y_pred_naive.dropna()
    y_test_naive = y_test.iloc[24:]

    mae_naive = mean_absolute_error(y_test_naive, y_pred_naive)
    rmse_naive = np.sqrt(mean_squared_error(y_test_naive, y_pred_naive))

    
    # MODELO: RandomForestRegressor
    
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae_rf = mean_absolute_error(y_test, y_pred)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred))
    
    df_test = df.iloc[split:].copy()
    df_test["prediction_rf"] = y_pred
    df_test["residual_rf"] = df_test[target_col] - df_test["prediction_rf"]

    df_test["prediction_naive_24h"] = y_test.shift(24).values
    df_test["residual_naive_24h"] = df_test[target_col] - df_test["prediction_naive_24h"]

    Path(out_predictions).parent.mkdir(parents=True, exist_ok=True)
    df_test.to_csv(out_predictions, index=False)
    
    feat_importance = dict(zip(features, model.feature_importances_.tolist()))

    feat_importance = dict(sorted(feat_importance.items(), key=lambda kv: kv[1], reverse=True))

    
    # Guardar modelo
    
    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "target_col": target_col,
        },
        out_model,
    )

    improvement_mae = ((mae_naive - mae_rf) / mae_naive) * 100.0 if mae_naive != 0 else 0.0
    improvement_rmse = ((rmse_naive - rmse_rf) / rmse_naive) * 100.0 if rmse_naive != 0 else 0.0

    metrics = {
        "target_col": target_col,
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "baseline_naive_24h": {
            "mae": float(round(mae_naive, 3)),
            "rmse": float(round(rmse_naive, 3)),
        },
        "random_forest": {
            "n_estimators": 300,
            "mae": float(round(mae_rf, 3)),
            "rmse": float(round(rmse_rf, 3)),
        },
        "improvement_vs_baseline_pct": {
            "mae": float(round(improvement_mae, 3)),
            "rmse": float(round(improvement_rmse, 3)),
        },
        "feature_importance": feat_importance,
        "artifacts": {
            "model_path": out_model,
            "predictions_path": out_predictions,
        },
    }

    Path(out_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="data/processed/opsd_preprocessed.csv")
    parser.add_argument("--target_col", default="AT_load_actual_entsoe_transparency")
    args = parser.parse_args()

    rep = train_forecast_model(
        input_csv=args.input_csv,
        target_col=args.target_col,
    )

    print("El modelo se entreno correctamente")
    print(f" - Target: {rep['target_col']}")
    print(f" - Train rows: {rep['rows_train']}")
    print(f" - Test rows: {rep['rows_test']}")
    print(" - Baseline naive (t-24): "
          f"MAE={rep['baseline_naive_24h']['mae']} | RMSE={rep['baseline_naive_24h']['rmse']}")
    print(" - Random Forest: "
          f"MAE={rep['random_forest']['mae']} | RMSE={rep['random_forest']['rmse']}")
    print(" - Mejora vs baseline (%): "
          f"MAE={rep['improvement_vs_baseline_pct']['mae']} | RMSE={rep['improvement_vs_baseline_pct']['rmse']}")
    print(" - Feature importance (top 3):")
    top3 = list(rep["feature_importance"].items())[:3]
    for k, v in top3:
        print(f"   * {k}: {round(v, 4)}")
    print(f" - Modelo: {rep['artifacts']['model_path']}")
    print(f" - Predicciones: {rep['artifacts']['predictions_path']}")
    print(" - Métricas: results/forecast_metrics.json")