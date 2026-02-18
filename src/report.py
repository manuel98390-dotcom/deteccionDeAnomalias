# src/report.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_anomalies(
    anomalies_csv: str = "results/anomalies.csv",
    out_png: str = "reports/anomalies_plot.png",
    max_points: int = 20000,
) -> str:
    # Cargar el CSV con anomalías con las anomalias clasificadas
    df = pd.read_csv(anomalies_csv)

    # La primera columna corresponde al timestamp
    ts_col = df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)

    # Identificar automáticamente la columna objetivo
    # Se excluyen las columnas técnicas
    candidate_cols = [
        c for c in df.columns
        if c not in {ts_col, "anomaly_flag", "anomaly_score"}
    ]
    target_col = None
    for c in candidate_cols:
        if "load" in c.lower():
            target_col = c
            break

    if target_col is None:
        target_col = candidate_cols[0]

    if len(df) > max_points:
        step = max(1, len(df) // max_points)
        df_plot = df.iloc[::step].copy()
    else:
        df_plot = df.copy()

    # Filtrar anomalías
    anoms = df_plot[df_plot["anomaly_flag"] == 1]

    # Crear carpeta si no existe
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)

    # Se genera la gráfica 
    plt.figure(figsize=(14, 5))
    plt.plot(df_plot[ts_col], df_plot[target_col], linewidth=1)
    plt.scatter(anoms[ts_col], anoms[target_col], s=12)
    plt.title(f"Anomalías detectadas (Isolation Forest) | Serie: {target_col}")
    plt.xlabel("Tiempo")
    plt.ylabel("Carga / Valor")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

    return out_png


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anomalies_csv", default="results/anomalies.csv")
    parser.add_argument("--out_png", default="reports/anomalies_plot.png")
    parser.add_argument("--max_points", type=int, default=20000)
    args = parser.parse_args()

    out = plot_anomalies(
        anomalies_csv=args.anomalies_csv,
        out_png=args.out_png,
        max_points=args.max_points,
    )

    print("Se generó la gráfica de anomalías")
    print(f" - Imagen guardada en: {out}")