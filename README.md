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
| **N-HiTS** | **12.14** | **9.60** | **1.05%** | **+20.2%** |
| Naive | 15.21 | 13.31 | 1.46% | — (mejor baseline) |
| AutoARIMA | 16.52 | 14.42 | 1.58% | -8.6% |
| N-BEATS | 18.27 | 14.09 | 1.55% | -20.1% |

![Predicción vs. real](datos/resultados/prediccion_vs_real.png)
![RMSE por modelo](datos/resultados/rmse_comparacion.png)

**Lectura honesta**: N-HiTS mejora el RMSE ~20% sobre el mejor baseline clásico (Naive) — una mejora real y grande, no un margen cosmético. N-BEATS, en cambio, termina **peor** que el naive (-20%), y de forma consistente: pierde contra N-HiTS en 4 de las 5 ventanas de backtesting, no por un solo tropiezo puntual. Es un resultado que coincide con la motivación original del paper de N-HiTS (Challu et al.) — fue diseñado explícitamente como una versión más robusta que N-BEATS via interpolación jerárquica multi-tasa — y acá se replica esa diferencia en una serie financiera real.

Esto no era el resultado inicial: la primera versión de este pipeline usaba `scaler_type="identity"` (sin normalizar) y `max_steps=300` para entrenar rápido, lo que llevaba a **ambos** modelos de Nixtla a predecir prácticamente una constante por ventana (desviación estándar del pronóstico ~0.3-0.5 contra ~4-12 del dato real) — un forecast visualmente "chato" que no capturaba ninguna dinámica real, aunque el RMSE agregado se viera razonable. Normalizar la entrada (`scaler_type="standard"`) y entrenar más (`max_steps=1500`) resolvió eso para los modelos neuronales: ahora N-HiTS sigue la forma de la curva real (confirmado por desviación estándar del pronóstico en la misma escala que la real, no solo a ojo), y N-BEATS dejó de ser plano pero sigue sin acertar la dirección.

**Naive y AutoARIMA siguen viéndose en escalón — pero por una razón distinta, y verificada, no una suposición:**
- **Naive** es plano por definición: repite el último valor observado para todo el horizonte. No hay forma de que sea otra cosa sin dejar de ser el método naive.
- **AutoARIMA no está limitado por diseño** — en principio podría producir una curva. Pero al inspeccionar el orden que realmente elige en cada una de las 5 ventanas (`AutoARIMA().fit(y).model_['arma']`), siempre selecciona **ARIMA(0,1,1)**: una caminata aleatoria con un término MA(1) sobre la serie diferenciada. Un MA(1) solo tiene memoria de un paso — matemáticamente, su pronóstico multi-paso colapsa a una constante a partir del segundo paso en adelante. No es una limitación de la implementación: es el propio criterio de selección de AIC diciendo que, estadísticamente, USD/CLP no muestra estructura autorregresiva más allá de una caminata aleatoria con un ajuste de un paso. Forzar un orden más complejo "para que se vea menos plano" sería inflar el modelo sin respaldo estadístico — lo contrario del espíritu de este proyecto.

En resumen: de los cuatro modelos, dos (Naive, AutoARIMA) están matemáticamente forzados a ser (casi) planos por lo que son — no por un error de código — y de los dos que sí tienen libertad para curvarse (N-BEATS, N-HiTS), solo uno la usó bien.

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
