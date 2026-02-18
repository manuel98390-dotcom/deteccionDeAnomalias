# src/anomaly.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_anomalies(
    input_csv: str,
    target_col: str,
    out_csv: str = "results/anomalies.csv",
    out_model: str = "models/isolation_forest.joblib",
    out_summary: str = "results/anomaly_summary.json",
    contamination: float = 0.01,
    random_state: int = 42,
) -> dict:
    df = pd.read_csv(input_csv)

    # Primera columna = timestamp
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)

    if target_col not in df.columns:
        raise ValueError(f"La columna '{target_col}' no existe en el CSV.")

    feature_cols = [
        target_col,
        "hour",
        "dayofweek",
        "month",
        "lag_1",
        "lag_24",
        "roll_mean_24",
        "roll_std_24",
    ]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para el modelo: {missing}")

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.loc[X.index].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    # -1 = anomalía, 1 = normal
    pred = model.predict(X_scaled)
    score = model.decision_function(X_scaled)

    df["anomaly_flag"] = (pred == -1).astype(int)
    df["anomaly_score"] = score

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "model": model,
            "feature_cols": feature_cols,
            "target_col": target_col,
        },
        out_model,
    )

    total_rows = int(len(df))
    total_anomalies = int(df["anomaly_flag"].sum())
    anomaly_rate = float((total_anomalies / total_rows) * 100.0) if total_rows else 0.0

    summary = {
        "input_csv": input_csv,
        "output_csv": out_csv,
        "model_path": out_model,
        "target_col": target_col,
        "contamination": contamination,
        "rows_analyzed": total_rows,
        "anomalies_detected": total_anomalies,
        "anomaly_rate_pct": round(anomaly_rate, 3),
    }

    Path(out_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="data/processed/opsd_preprocessed.csv")
    parser.add_argument("--target_col", default="AT_load_actual_entsoe_transparency")
    parser.add_argument("--contamination", type=float, default=0.01)
    args = parser.parse_args()

    report = detect_anomalies(
        input_csv=args.input_csv,
        target_col=args.target_col,
        contamination=args.contamination,
    )

    print("✅ Detección de anomalías completada")
    print(f" - Archivo de entrada: {report['input_csv']}")
    print(f" - Archivo generado: {report['output_csv']}")
    print(f" - Modelo guardado en: {report['model_path']}")
    print(f" - Filas analizadas: {report['rows_analyzed']}")
    print(f" - Anomalías detectadas: {report['anomalies_detected']}")
    print(f" - Porcentaje de anomalías: {report['anomaly_rate_pct']}%")