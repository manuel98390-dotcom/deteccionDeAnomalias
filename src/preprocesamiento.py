"""  
    Preprocesa OPSD:
    - Lee CSV con timestamp como índice/columna.
    - Convierte datetime y ordena.
    - Selecciona automáticamente una serie objetivo (por keyword)
      y con baja proporción de faltantes.
    - Rellena faltantes por interpolación temporal.
    - Genera features de tiempo (hour, dow, month) y lags básicos. """

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def preprocess_opsd(
    input_csv: str,
    out_csv: str,
    prefer_keyword: str = "load",
    max_missing_ratio: float = 0.05,
) -> dict:
    df = pd.read_csv(input_csv)

    # Seleccionar columna de tiempo
    time_col = None
    for c in df.columns:
        if "timestamp" in c.lower():
            time_col = c
            break
    if time_col is None:
        raise ValueError("No se encontró columna de timestamp en el dataset")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df.set_index(time_col)

    # Candidatas numéricas
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        raise ValueError("No se encontraron columnas numéricas para analizar.")

    # Preferir columnas con keyword (ej. load) y pocos faltantes
    preferred = [c for c in num_cols if prefer_keyword.lower() in c.lower()]
    candidates = preferred if preferred else num_cols

    # Filtrar por missing ratio
    col_missing = {c: float(df[c].isna().mean()) for c in candidates}
    filtered = [c for c in candidates if col_missing[c] <= max_missing_ratio]

    # Si ninguna pasa el filtro, usamos la "mejor" (menos faltantes)
    target_col = None
    if filtered:
        target_col = sorted(filtered, key=lambda c: col_missing[c])[0]
    else:
        target_col = sorted(candidates, key=lambda c: col_missing[c])[0]

    series = df[[target_col]].copy()

    # Relleno de faltantes (time interpolation)
    series[target_col] = series[target_col].interpolate(method="time").ffill().bfill()

    # Features temporales
    out = series.copy()
    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month

    # Lags y rolling
    out["lag_1"] = out[target_col].shift(1)
    out["lag_24"] = out[target_col].shift(24)
    out["roll_mean_24"] = out[target_col].rolling(24).mean()
    out["roll_std_24"] = out[target_col].rolling(24).std()

    out = out.dropna().copy()

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=True)

    return {
        "input_csv": input_csv,
        "output_csv": out_csv,
        "timestamp_col": time_col,
        "target_col": target_col,
        "rows": int(len(out)),
        "cols": int(out.shape[1]),
        "missing_ratio_target_before": float(df[target_col].isna().mean()),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", default="data/raw/opsd_time_series_60min_singleindex.csv")
    p.add_argument("--out_csv", default="data/processed/opsd_preprocessed.csv")
    p.add_argument("--prefer_keyword", default="load")
    p.add_argument("--max_missing_ratio", type=float, default=0.05)
    args = p.parse_args()

    rep = preprocess_opsd(
        input_csv=args.input_csv,
        out_csv=args.out_csv,
        prefer_keyword=args.prefer_keyword,
        max_missing_ratio=args.max_missing_ratio,
    )

    print("Se finalizó el preprocesamiento de OPSD.")
    for k, v in rep.items():
        print(f" - {k}: {v}")