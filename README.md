# ⚡ Energy Demand Forecasting & Anomaly Detection Pipeline

Pipeline completo de Machine Learning aplicado a sistemas energéticos para:

- Forecasting de demanda eléctrica horaria
- Detección automática de anomalías
- Validación temporal robusta
- Exportación de métricas y modelos listos para producción

Este proyecto simula un escenario real donde una empresa del sector energético necesita monitorear, predecir y detectar comportamientos anómalos en el consumo eléctrico.

## Descripción General

El repositorio implementa un pipeline end-to-end:

1. Ingesta del dataset (Open Power System Data - OPSD)
2. Preprocesamiento de series temporales
3. Ingeniería de características:
   - hour
   - dayofweek
   - month
   - lag_1
   - lag_24
   - rolling mean 24h
   - rolling std 24h
4. Modelo baseline (naive t-24)
5. Modelo de forecasting con Machine Learning
6. Detección de anomalías con modelo no supervisado
7. Validación temporal con múltiples splits
8. Exportación de métricas y serialización de modelos


## Modelos ML Implementados

### Forecasting
- Random Forest Regressor
- 300 árboles
- División temporal 80% entrenamiento / 20% prueba
- Comparación contra baseline naïve (t-24)

### Baseline
- Modelo ingenuo basado en el valor del día anterior a la misma hora (t-24)

### Detección de Anomalías
- Isolation Forest
- contamination = 1%
- Escalado con StandardScaler
- Genera:
  - Flag binario de anomalía
  - Score continuo de anomalía


## Resultados Obtenidos

### Forecasting

Random Forest:
- MAE ≈ 70.6
- RMSE ≈ 103.7

Baseline naive:
- MAE ≈ 528.8
- RMSE ≈ 817.2

Mejora aproximada del modelo ML:
- ~86% reducción de error respecto al baseline

### Validación Temporal (5 splits)

Random Forest:
- MAE: 77.0 ± 14.0
- RMSE: 112.6 ± 26.2

Baseline naive:
- MAE: 563.5 ± 26.1
- RMSE: 865.7 ± 42.6

El modelo demuestra estabilidad y robustez temporal.

### Detección de Anomalías

- Filas analizadas: 50,377
- Anomalías detectadas: 504 (~1%)


## Cómo Ejecutar el Pipeline

### 1 Descargar dataset

python3 -m src.ingest


### 2️ Preprocesar

python3 -m src.preprocess


### 3️ Detectar anomalías

python3 -m src.anomaly --contamination 0.01


### 4️ Entrenar modelo de forecasting

python3 -m src.forecast


### 5️ Validación temporal

python3 -m src.validate


### 6️ Generar gráfica de anomalías

python3 -m src.report


## Tecnologías

- Python 3.12
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib

## Dataset

Open Power System Data (OPSD)
Demanda eléctrica horaria (Austria - AT)
