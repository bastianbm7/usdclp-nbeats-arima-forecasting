# usdclp-nbeats-arima-forecasting

Compara modelos modernos de forecasting — **N-BEATS** y **N-HiTS** (vía [Nixtla/neuralforecast](https://github.com/Nixtla/neuralforecast), Apache-2.0, usado como dependencia) — contra un baseline clásico (**AutoARIMA** y **Naive**, vía [Nixtla/statsforecast](https://github.com/Nixtla/statsforecast)) pronosticando el tipo de cambio **USD/CLP**.

## Pregunta que responde

¿Un modelo moderno de deep learning aporta una mejora real y medible sobre un baseline clásico al pronosticar el tipo de cambio, o la mejora observada es mayormente autocorrelación del precio diario (el valor de hoy predice casi perfecto el de mañana)?

## Metodología

- Backtesting walk-forward (múltiples ventanas de validación, no un solo split train/test).
- Horizonte de pronóstico de varios pasos (no 1 solo paso) — a 1 paso el naive casi siempre "gana" por autocorrelación pura; a varios pasos esa ventaja se diluye.
- Métricas de error en test: RMSE, MAE, MAPE — calculadas para los cuatro modelos (Naive, AutoARIMA, NBEATS, NHITS) sobre las mismas ventanas.

## Resultados

Backtesting walk-forward: 5 ventanas no solapadas de 14 días hábiles cada una (70 días hábiles de evaluación fuera de muestra, jun-sep 2026), sobre la serie completa 2010-2026 (~4.346 observaciones).

| Modelo | RMSE | MAE | MAPE | Mejora RMSE vs. mejor baseline |
|---|---|---|---|---|
| **N-BEATS** | **14.50** | **12.63** | **1.39%** | **+4.7%** |
| Naive | 15.21 | 13.31 | 1.46% | — (mejor baseline) |
| N-HiTS | 15.52 | 13.58 | 1.49% | -2.1% |
| AutoARIMA | 16.52 | 14.42 | 1.58% | -8.6% |

![Predicción vs. real](datos/resultados/prediccion_vs_real.png)
![RMSE por modelo](datos/resultados/rmse_comparacion.png)

**Lectura honesta**: N-BEATS mejora el RMSE ~4.7% sobre el mejor baseline clásico (Naive) — una mejora real pero modesta, no dramática. N-HiTS y AutoARIMA ni siquiera superan al naive en este tramo. Esto confirma la hipótesis de partida: en una serie de tipo de cambio diaria, el naive es un baseline fuerte porque gran parte del error bajo de cualquier modelo es autocorrelación, no señal predictiva real. El gráfico también muestra el patrón esperado de "escalón" dentro de cada ventana de 14 días — todos los modelos pronostican desde un único origen por ventana, así que no tienen más información dentro de la ventana que el punto de partida.

## Datos

Tipo de cambio USD/CLP, serie diaria descargada con [`yfinance`](https://github.com/ranaroussi/yfinance) (ticker `CLP=X`, fuente: Yahoo Finance). Se guarda una copia cruda en `datos/bases/` para reproducibilidad exacta (no depender de que Yahoo siga sirviendo el mismo histórico).

## Estructura

```
codigos/
├── 01_obtener_datos.py          # descarga USD/CLP vía yfinance, split train/test
├── 02_baseline_arima_naive.py   # statsforecast: AutoARIMA + Naive, backtesting walk-forward
├── 03_modelo_nbeats_nhits.py    # neuralforecast: NBEATS + NHITS, mismo esquema de backtesting
├── 04_metricas_comparacion.py   # RMSE/MAE/MAPE por modelo, tabla comparativa
└── 05_visualizacion.py          # predicción vs. real + intervalos + baseline superpuesto

datos/
├── bases/        # CSV crudo de USD/CLP
└── resultados/   # gráficos finales + tabla de métricas
```

## Limitaciones conocidas

- **Autocorrelación, no "poder predictivo"**: con precio de cierre diario, buena parte de cualquier error bajo es autocorrelación, no señal real capturada por el modelo. Mitigado con baseline siempre presente, backtesting walk-forward y horizonte de varios pasos — pero no eliminado del todo.
- **Nixtla/neuralforecast es una dependencia, no un fork**: se usa la implementación de la librería de N-BEATS/N-HiTS (Apache-2.0), no el código original de los autores (Oreshkin et al. / Challu et al.) — la fidelidad al paper es metodológica, no literal.
- **Nivel 1 de proporcionalidad**: sin tests automatizados ni monitoreo — es una pieza de portfolio personal, no un servicio en producción.
- **Dependencia de Yahoo Finance**: `yfinance` puede cambiar de comportamiento — mitigado guardando una copia cruda del CSV descargado.

## Paper(s) ancla

- Oreshkin et al., *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting*, ICLR 2020 — [arXiv:1905.10437](https://arxiv.org/abs/1905.10437)
- Challu et al., *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting* — [arXiv:2201.12886](https://arxiv.org/abs/2201.12886)
