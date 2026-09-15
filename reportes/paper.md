# N-BEATS y N-HiTS contra baselines clásicos para el pronóstico de USD/CLP a múltiples escalas temporales

**Repositorio**: [bastianbm7/usdclp-nbeats-arima-forecasting](https://github.com/bastianbm7/usdclp-nbeats-arima-forecasting)
**Fecha**: septiembre de 2026

## Resumen

Este trabajo compara dos arquitecturas de deep learning para pronóstico de series de tiempo — **N-BEATS** (Oreshkin et al., 2020) y **N-HiTS** (Challu et al., 2023) — contra baselines clásicos (Naive, AutoARIMA, Random Walk with Drift) al pronosticar el tipo de cambio **USD/CLP**, usando las implementaciones de código abierto de [Nixtla](https://github.com/Nixtla) (`neuralforecast` y `statsforecast`, Apache-2.0). El estudio se organiza en cuatro partes: (1) un análisis a escala diaria con horizonte de 14 días hábiles, (2) el mismo análisis resampleado a escala mensual, (3) a escala anual, y (4) un experimento adicional de **aprendizaje online** (`refit=True`, warm-start) con horizontes cortos (1-3 pasos) sobre series diaria, semanal y mensual.

El hallazgo central no es "el deep learning gana" ni "el baseline clásico gana", sino que **la estructura dominante de una serie financiera cambia con la escala de tiempo y con el modo de entrenamiento**, y el modelo ganador cambia con ella — incluyendo un caso de inestabilidad real y no resuelta (aprendizaje online a escala mensual con horizonte de 3 pasos) que se documenta en vez de ocultar.

## 1. Pregunta de investigación

¿Un modelo moderno de deep learning aporta una mejora real y medible sobre un baseline estadístico clásico al pronosticar un tipo de cambio, o la mejora observada es mayormente autocorrelación del precio (el valor de hoy predice casi perfecto el de mañana)? Y si la respuesta depende de la escala de tiempo o de si el modelo se reentrena a medida que llegan datos nuevos — ¿de qué depende exactamente?

## 2. Datos

Tipo de cambio USD/CLP, serie diaria descargada con [`yfinance`](https://github.com/ranaroussi/yfinance) (ticker `CLP=X`, fuente: Yahoo Finance), 2010-01-01 a 2026-09-11 (4.346 observaciones en días hábiles). A partir de esa serie diaria se derivan por resampleo (último valor del período) las series semanal (871 obs.), mensual (200-201 obs.) y anual (16-17 obs., el último año siempre se descarta por estar incompleto). Los datos crudos y procesados están en `datos/bases/`.

## 3. Metodología

- **Backtesting walk-forward**: nunca un solo split train/test. Cada resultado surge de múltiples ventanas de evaluación fuera de muestra.
- **Modelos**: `Naive` y `AutoARIMA` (selección automática de orden ARIMA vía AIC) de `statsforecast`; `RandomWalkWithDrift` (solo en el análisis anual); `NBEATS` y `NHITS` de `neuralforecast`, con `loss=MQLoss()` para obtener intervalos de predicción.
- **Métricas**: RMSE, MAE, MAPE, calculadas con `utilsforecast.losses` sobre las mismas ventanas para todos los modelos de una corrida.
- **Dos modos de entrenamiento**:
  - *Estático* (secciones 4.1-4.3): el modelo se entrena una vez sobre el historial y se usa para predecir todas las ventanas de test.
  - *Online* (sección 4.4): el modelo se reentrena en cada ventana a medida que hay datos nuevos disponibles (`refit=True`), partiendo de los pesos de la ventana anterior en vez de reiniciar (`use_init_models=False`, warm start) — mecanismo nativo de `neuralforecast`, no código de online-learning escrito a mano. `statsforecast` ya hace esto por default.
- **Licencias**: tanto `neuralforecast` como `statsforecast` son Apache-2.0, usados como dependencias (`pip install`), no forkeados — la fidelidad a los papers originales de N-BEATS/N-HiTS es metodológica, no literal (no es el código de los autores).

## 4. Resultados

### 4.1 Escala diaria (entrenamiento estático, h=14 días hábiles)

5 ventanas no solapadas de 14 días hábiles (70 días hábiles de evaluación, jun-sep 2026), sobre los 4.346 días hábiles completos.

| Modelo | RMSE | MAE | MAPE | Mejora vs. mejor baseline |
|---|---|---|---|---|
| **N-HiTS** | **12.14** | 9.60 | 1.05% | **+20.2%** |
| Naive | 15.21 | 13.31 | 1.46% | — |
| AutoARIMA | 16.52 | 14.42 | 1.58% | -8.6% |
| N-BEATS | 18.27 | 14.09 | 1.55% | -20.1% |

N-HiTS mejora el RMSE ~20% sobre el mejor baseline, una mejora real y consistente (gana en 4 de las 5 ventanas contra N-BEATS, no por azar). `Naive` y `AutoARIMA` quedan **matemáticamente forzados** a producir forecasts (casi) planos dentro de cada ventana: `Naive` por definición, y `AutoARIMA` porque selecciona consistentemente **ARIMA(0,1,1)** en las 5 ventanas — un MA(1) sobre la serie diferenciada tiene memoria de un solo paso, así que su pronóstico multi-paso colapsa a una constante a partir del segundo paso. Esto se verificó inspeccionando `AutoARIMA().fit(y).model_['arma']` directamente, no se asumió.

La primera versión de este análisis usaba `scaler_type="identity"` (sin normalizar) y `max_steps=300`, lo que llevaba a **ambos** modelos de Nixtla a predecir casi una constante por ventana (desviación estándar del pronóstico ~0.3-0.5 contra ~4-12 del dato real) — un resultado visualmente chato aunque el RMSE agregado pareciera razonable. Normalizar (`scaler_type="standard"`) y entrenar más (`max_steps=1500`) resolvió esto.

![Predicción vs. real diaria](../datos/resultados/prediccion_vs_real.png)
![RMSE por modelo diaria](../datos/resultados/rmse_comparacion.png)

### 4.2 Escala mensual (entrenamiento estático, h=6 meses)

Cierre de mes, 200 observaciones, 5 ventanas de 6 meses.

| Modelo | RMSE | MAE | MAPE | Mejora vs. mejor baseline |
|---|---|---|---|---|
| **N-BEATS** | **33.09** | 30.82 | 3.28% | **+20.7%** |
| Naive | 41.71 | 35.40 | 3.81% | — |
| N-HiTS | 42.57 | 35.11 | 3.77% | -2.1% |
| AutoARIMA | 44.26 | 38.77 | 4.16% | -6.1% |

Se invierten los papeles respecto a la escala diaria: gana **N-BEATS**, no N-HiTS — ninguno de los dos modelos de Nixtla es sistemáticamente mejor en todas las escalas. Detalle honesto visible en el gráfico: en la ventana de cutoff 2024-12, ambos modelos de Nixtla sobre-extrapolan una racha alcista reciente y proyectan un salto a ~1.050-1.080 que la serie real nunca toca (se queda en ~950-990) — sobre-reacción a momentum de corto plazo que el resultado agregado diluye pero no oculta.

![Predicción vs. real mensual](../datos/resultados/prediccion_vs_real_mensual.png)
![RMSE por modelo mensual](../datos/resultados/rmse_comparacion_mensual.png)

### 4.3 Escala anual (entrenamiento estático, h=1 año, sin redes neuronales)

Cierre de diciembre, 16 observaciones completas (2010-2025), 3 ventanas de 1 año. **A propósito no se incluyen N-BEATS/N-HiTS**: con 16 puntos, entrenar una red neuronal no tiene sustento estadístico real — cualquier resultado sería ruido de una corrida, no una señal reproducible.

| Modelo | RMSE | MAPE | Mejora vs. Naive |
|---|---|---|---|
| **AutoARIMA** | **3.72** | 0.42% | **+87.0%** |
| **RandomWalkWithDrift** | **3.72** | 0.42% | **+87.0%** |
| Naive | 28.64 | 3.24% | — |

El resultado más limpio de los tres: `AutoARIMA` y `RandomWalkWithDrift` dan **el mismo número exacto**, porque AutoARIMA elige por su cuenta un ARIMA(0,1,0) con drift — matemáticamente idéntico a un random walk con tendencia. A escala anual, USD/CLP se comporta como una caminata aleatoria con una depreciación promedio constante; agregar esa tendencia (sin deep learning) le gana al naive plano por 87%. Caveat: ese mismo modelo sobre-extrapola la ventana más reciente (2025→2026, proyecta ~1.030 cuando el real bajó a ~900) — con n=16 cualquier métrica agregada tiene un margen de error grande.

![Predicción vs. real anual](../datos/resultados/prediccion_vs_real_anual.png)
![RMSE por modelo anual](../datos/resultados/rmse_comparacion_anual.png)

### 4.4 Aprendizaje online: horizontes cortos (1-3 pasos) con `refit=True`

Se prueba reentrenar en cada ventana a medida que aparecen datos nuevos (warm start), con horizontes de 1, 2 y 3 pasos, sobre tres escalas (diaria/semanal/mensual) — 9 combinaciones, 10 ventanas de backtesting cada una.

![Heatmap RMSE online learning](../datos/resultados/online_learning_heatmap.png)

| Escala | h | Ganador | RMSE ganador | Mejor baseline | RMSE baseline | Mejora |
|---|---|---|---|---|---|---|
| Diaria | 1 | Naive | 3.45 | — | — | — (nada le gana) |
| Diaria | 2 | Naive | 0.70 | — | — | — (nada le gana) |
| Diaria | 3 | N-HiTS | 10.62 | Naive | 10.73 | +1.0% |
| Semanal | 1 | **N-HiTS** | **2.38** | AutoARIMA | 4.79 | **+50.2%** |
| Semanal | 2 | **N-HiTS** | **5.69** | Naive | 11.74 | **+51.5%** |
| Semanal | 3 | N-HiTS | 4.70 | AutoARIMA | 5.33 | +11.9% |
| Mensual | 1 | N-HiTS | 12.32 | Naive | 15.14 | +18.6% |
| Mensual | 2 | **N-BEATS** | **3.25** | Naive | 29.85 | **+89.1%** |
| Mensual | 3 | — | — | AutoARIMA | 41.07 | **-133% a -167%** (ambas redes fallan) |

**Diaria**: el naive domina los 3 horizontes (con empate técnico de N-HiTS en h=3). A 1-3 días la autocorrelación es tan fuerte que ni el aprendizaje online encuentra ventaja — coherente con la sección 4.1.

**Semanal**: la escala donde el aprendizaje online más aporta. N-HiTS gana los 3 horizontes, con mejoras de +50% y +51% en h=1 y h=2.

**Mensual**: el resultado más interesante. En h=1 y h=2 las redes mejoran mucho al baseline (N-BEATS en h=2 llega a un 89% de mejora). Pero en **h=3 ambas redes fallan catastróficamente** (RMSE de 96-110 contra ~41 del naive) — y se verificó explícitamente que no es un problema de entrenamiento insuficiente: subir el presupuesto de 100 a 300 pasos por reentreno mejoró mucho h=1 y sobre todo h=2, pero **no cambió nada en h=3**. Esta inestabilidad queda documentada como hallazgo abierto, no resuelto — hipótesis de trabajo: la combinación de warm-start (el optimizador de PyTorch Lightning no conserva su estado de Adam entre reentrenos, solo los pesos) con un horizonte relativamente largo sobre una serie corta (200 obs.) puede generar actualizaciones inestables al reiniciar el optimizador sobre pesos ya ajustados.

## 5. Discusión

Cuatro estudios, cuatro respuestas distintas a la misma pregunta:

| Configuración | ¿Le gana algo al naive? | Ganador | Por qué |
|---|---|---|---|
| Diaria, estático, h=14 | Sí, +20% | N-HiTS | Único con libertad real de forma que la aprovechó bien |
| Mensual, estático, h=6 | Sí, +21% | N-BEATS | El "ganador" entre N-BEATS/N-HiTS no es estable entre escalas |
| Anual, estático, h=1 | Sí, +87% | AutoARIMA = RWD | Domina una tendencia determinística de largo plazo, no ruido |
| Diaria/semanal/mensual, online, h=1-3 | Depende mucho | N-HiTS (semanal, sobre todo) | Reentrenar ayuda muchísimo en algunos casos y es inestable en otros |

La lectura de portfolio no es "el deep learning gana siempre" ni "el baseline clásico gana siempre" — es que la estructura dominante de una serie financiera **cambia con la escala de tiempo y con el modo de entrenamiento**, y ni siquiera dar más cómputo garantiza estabilidad (mensual h=3 online). Reportar esto último, en vez de ocultarlo detrás de la configuración que mejor se ve, es la decisión metodológica más importante de todo el trabajo.

## 6. Limitaciones conocidas

- **Autocorrelación, no "poder predictivo"**: con precio de cierre diario, buena parte de cualquier error bajo es autocorrelación, no señal real capturada por el modelo.
- **Nixtla es una dependencia, no un fork**: la fidelidad a los papers originales de N-BEATS/N-HiTS es metodológica, no literal.
- **Nivel 1 de proporcionalidad**: sin tests automatizados ni monitoreo — pieza de portfolio personal, no servicio en producción.
- **Backtesting anual con n=16**: muestra chica; el resultado tiene explicación matemática clara pero no la misma robustez estadística que diario/mensual.
- **Inestabilidad de `mensual h=3` online**: no resuelta; ver sección 4.4.
- **Dependencia de Yahoo Finance**: `yfinance` puede cambiar de comportamiento — mitigado guardando copias crudas de los datos descargados.

## 7. Trabajo futuro

- Investigar la causa raíz de la inestabilidad en `mensual h=3` (ej. inspeccionar la trayectoria de pérdida por reentreno, probar conservando el estado del optimizador entre ventanas).
- Probar `AutoNBEATS`/`AutoNHITS` (búsqueda automática de hiperparámetros vía Ray Tune, ya incluidos en el entorno) en vez de fijar `max_steps`/`scaler_type` a mano.
- Extender el aprendizaje online a horizontes más largos y a la escala anual, si se consigue más historial.
- Replicar el mismo diseño sobre otro par de divisas o un índice bursátil, para ver si la síntesis de la sección 5 se sostiene fuera de USD/CLP.

## 8. Referencias

- Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020). *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting*. ICLR 2020. [arXiv:1905.10437](https://arxiv.org/abs/1905.10437)
- Challu, C., Olivares, K. G., Oreshkin, B. N., Ramirez, F. G., Canseco, M. M., & Dubrawski, A. (2023). *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting*. [arXiv:2201.12886](https://arxiv.org/abs/2201.12886)
- Nixtla. [statsforecast](https://github.com/Nixtla/statsforecast) y [neuralforecast](https://github.com/Nixtla/neuralforecast) (Apache-2.0).

## 9. Extensión: estrategia de trading con Reinforcement Learning (septiembre 2026)

El forecasting de precio (secciones 1-7) responde si un modelo predice bien. Esta extensión responde una pregunta distinta: **¿ese forecast sirve para tomar decisiones de trading que ganen plata, una vez que se cuentan el apalancamiento y el riesgo real?** Es un proyecto separado dentro del mismo repo (Issue [#1](https://github.com/bastianbm7/usdclp-nbeats-arima-forecasting/issues/1)), no una sección más del estudio de forecasting.

### 9.1 Elección de enfoque

Se evaluaron tres opciones para convertir el forecast en una estrategia: (A) una regla fija de umbral sobre la predicción, (B) un framework de backtesting dedicado (backtrader/vectorbt), y (C) un agente de **Reinforcement Learning**. Se eligió la opción C, con **FinRL** (Liu et al., [arXiv:2111.09395](https://arxiv.org/abs/2111.09395); repo [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL), MIT) como ancla metodológica — no como dependencia completa: FinRL está diseñado para carteras multi-activo, así que se construyó un entorno Gym propio y liviano (`gymnasium` + `stable-baselines3`, algoritmo PPO) para un solo par.

### 9.2 Modelo de volatilidad: comparación antes de elegir

Antes de decidir qué modelo de volatilidad alimentaría al agente, se comparó walk-forward (100 ventanas semanales, reentrenando en cada una) contra dos baselines:

| Modelo | RMSE | Mejora vs. Naive |
|---|---|---|
| **GARCH(1,1)** | **0.00653** | **+37.4%** |
| EGARCH(1,1) | 0.00661 | +36.7% |
| Media móvil (8 semanas) | 0.00885 | +15.2% |
| Naive | 0.01044 | — |

![Comparación de modelos de volatilidad](../datos/resultados/rmse_comparacion_volatilidad.png)

Ganó GARCH(1,1), usado desde acá como el forecast de volatilidad del agente y como base del stop-loss ("volatility scaling", la misma técnica de position-sizing que usan los papers de Wood et al. citados más abajo). En el camino se encontraron y corrigieron **2 ticks corruptos de yfinance** en `usdclp_long.csv` (2016-12-22 y 2014-04-10, precio cayendo a ~5 en vez de ~660/~544 por un día) — invisibles para el forecasting de precio de las secciones 1-7, pero catastróficos para cualquier cálculo de volatilidad (retorno de ese día: ±488%). La corrección vive en `01_obtener_datos.py` (función `limpiar_ticks_erroneos`), así que persiste si se vuelve a descargar la serie.

### 9.3 Diseño del agente

**Estado** (7 variables, todas relativas/acotadas — no precio nivel, porque USD/CLP no es estacionario en 16 años): retorno de la última semana, forecast N-HiTS a 1 y 2 semanas (el de aprendizaje online semanal `refit=True`, el resultado más fuerte de la sección 4.4) como desviación % del precio actual, volatilidad GARCH, MACD y RSI normalizados, y la posición del precio dentro de su rango de 12 semanas.

**Acción**: 3 posiciones discretas (largo/plano/corto). **Gestión de riesgo**: capital simulado de $100, tamaño de posición vía *risk sizing* — arriesgar el 3% del capital vigente por operación, no un monto fijo — con el stop-loss a la distancia de la volatilidad GARCH pronosticada (1 desvío) y el take-profit en el precio objetivo de N-HiTS. El stop-loss es *trailing*: sube seguiendo al precio a favor (largo) sin retroceder nunca, para asegurar ganancias parciales sin esperar el take-profit fijo. TP/SL se verifican día a día contra el precio real dentro de la semana, no solo al cierre. Slippage simulado de 0.05% por cambio de posición; comisión en 0 (decisión explícita para esta v1, no una omisión).

### 9.4 Resultados: walk-forward con gestión de riesgo real

Walk-forward de 5 ventanas × 20 semanas de test cada una (100 semanas out-of-sample, 2024-09 a 2026-08), reentrenando el agente en cada ventana — nunca un solo split, consistente con la metodología del resto del proyecto.

| Estrategia | Retorno total | Sharpe anualizado | Max drawdown | Win rate | Operaciones |
|---|---|---|---|---|---|
| **Buy-and-hold (sin apalancar)** | **-0.9%** | **~0.00** | -15.4% | 47.0% | — |
| Umbral simple (Opción A) | -16.0% | -0.94 | -20.8% | 51.5% | 68 |
| **PPO (RL)** | **0.0%** | — | 0.0% | — | **0** |

![Curva de capital](../datos/resultados/walkforward_curva_capital.png)
![Métricas por estrategia](../datos/resultados/walkforward_metricas_por_estrategia.png)
![Puntos de entrada y salida](../datos/resultados/walkforward_puntos_entrada_salida.png)

**El hallazgo no es el que se esperaba, y es el más honesto de reportar tal cual**: el agente de RL, entrenado primero con una recompensa simplificada (retorno % sin apalancar), convergió a una política degenerada de "siempre largo" que, evaluada con apalancamiento real, perdía -52.2% — peor que cualquier baseline. Al reentrenar con la **recompensa real** (la misma economía de apalancamiento + TP/SL del backtest, no una versión simplificada), el agente convergió a una política distinta: **no operar nunca**, en las 5 ventanas, con o sin trailing stop. No es una falla del entrenamiento — es la respuesta racional de un agente que sí "siente" el costo del riesgo: ninguna señal disponible le pareció confiable para arriesgar capital.

### 9.5 Por qué: análisis de las variables del estado

Para entender la decisión del agente, se midió la correlación de cada feature del estado con el retorno real de la semana siguiente (348 semanas completas):

| Feature | Correlación | Acierto de dirección |
|---|---|---|
| retorno_1s | -0.109 | 49.1% |
| vol_garch | -0.067 | 51.1% |
| posicion_en_rango | -0.026 | 43.7% |
| nhits_h1_rel | -0.025 | 52.0% |
| rsi_norm | -0.019 | 51.1% |
| nhits_h2_rel | -0.018 | 50.0% |
| macd_rel | -0.005 | 55.2% |

![Correlación de cada feature con el retorno futuro](../datos/resultados/analisis_features_correlacion.png)

Ninguna variable supera |r|=0.11, y el acierto de dirección de todas ronda el 50% (una moneda) — incluido el forecast de N-HiTS (52.0%), pese a que en la sección 4.4 mostró una mejora real de RMSE en la métrica de *error de forecasting*. Esa es justo la distinción que motivó esta extensión: un modelo puede reducir el error de predicción de forma real y medible (sección 4) sin que eso alcance para anticipar la *dirección* del movimiento con la confiabilidad que una estrategia de trading necesita. El agente de RL no "descartó" una señal fuerte por error — la señal disponible es, de hecho, débil.

### 9.6 Limitaciones de esta extensión

- **Slippage simulado, comisión no**: 0.05% por cambio de posición; una comisión real (aunque sea baja) empeoraría más a las estrategias activas (68-100 operaciones) que al buy-and-hold.
- **Walk-forward de 5 ventanas**: más riguroso que un solo split, pero menos ventanas que las 100 del análisis diario original — cada ventana individual sigue siendo una muestra chica.
- **Un solo activo, un solo agente (PPO)**: no se probó DQN/A2C ni otra arquitectura de red; la conclusión es sobre esta configuración puntual, no sobre "RL para trading" en general.
- **Trailing stop con precios de cierre diario**: aproximación estándar de backtesting sin datos intradía — un trailing stop con datos tick a tick podría comportarse distinto.

### 9.7 Tareas pendientes en el Issue #1

- [ ] Cerrar la Tarea de Notion y el Issue con este hallazgo documentado (no hay más código pendiente de esta fase).
- [ ] Decidir si la Fase 2 (bot en tiempo real / forward-test) sigue en pie — dado el hallazgo de 9.4-9.5, probablemente no se justifica sin antes encontrar una señal con más edge que la de la sección 9.5.

### 9.8 Propuestas a partir de esta conclusión

El hallazgo central (sin edge direccional real, un agente que entiende el riesgo prefiere no operar) abre más preguntas que las que cierra:

1. **Buscar mejor señal antes que mejor agente**: la sección 9.5 sugiere que el cuello de botella es la señal (MACD/RSI/forecast con |r|<0.11), no el algoritmo de decisión. Vale más invertir en features nuevas (ej. variables macro, tasas de interés diferencial USD/CLP, flujos de comercio exterior) que en probar otro agente de RL sobre las mismas 7 variables.
2. **Repetir el diseño en otro activo**: si el mismo agente (recompensa real, mismo stack) encuentra una política no-trivial en otro par o instrumento, ayudaría a distinguir "USD/CLP no tiene edge explotable a esta escala" de "el diseño del agente tiene un problema genérico".
3. **Probar con comisión real distinta de cero**: cuantificar cuánto empeoraría el umbral simple (68 operaciones) con una comisión de, por ejemplo, 0.02-0.05% por operación — barato de correr, cierra un cabo suelto de 9.6.
4. **Considerar la "no-operación" como resultado de portfolio válido**: mostrar honestamente que un agente bien diseñado puede concluir "no juegues" es, en sí mismo, un punto de portfolio interesante — distinto (y más raro de ver) que la mayoría de los proyectos de trading con RL que solo muestran el caso en que "funcionó".

### 9.9 Extensión: resultados del Issue #2 (mejoras al agente)

A partir de la propuesta 1 de la sección 9.8 se probaron 3 cambios técnicos **independientes** sobre el mismo agente/entorno de la sección 9.3-9.4, cada uno evaluado por separado y combinados contra el mismo walk-forward de 5 ventanas × 20 semanas:

1. **Más exploración**: `ent_coef=0.02` en PPO (default de stable-baselines3: 0.0).
2. **Acción continua**: `action_space` de `Discrete(3)` a `Box(-1, 1)`, con el tamaño de la apuesta (notional) escalando proporcional a la magnitud de la acción en vez de solo largo/plano/corto.
3. **Reward shaping**: recompensa como exceso de retorno sobre buy-and-hold sin apalancar, en vez de retorno % crudo.

| Estrategia | Retorno total | Sharpe anualizado | Max drawdown | Win rate | Operaciones |
|---|---|---|---|---|---|
| Buy-and-hold (sin apalancar) | -0.9% | ~0.00 | -15.4% | 47.0% | 100 |
| Umbral simple (Opción A) | -16.0% | -0.94 | -20.8% | 51.5% | 68 |
| **PPO base** (referencia 9.4) | **0.0%** | — | 0.0% | — | **0** |
| **PPO + ent_coef alto** | **0.0%** | — | 0.0% | — | **0** |
| **PPO + exceso sobre buy-and-hold** | **0.0%** | — | 0.0% | — | **0** |
| PPO + acción continua | -16.8% | -1.51 | -20.6% | 31.0% | 100 |
| PPO + las 3 combinadas | -16.8% | -1.36 | -21.0% | 33.0% | 100 |

![Métricas por configuración](../datos/resultados/mejoras_rl_metricas_por_estrategia.png)
![Curva de capital por configuración](../datos/resultados/mejoras_rl_curva_capital.png)

**El resultado es tan honesto de reportar como el de la sección 9.4, y refuerza la misma conclusión en vez de contradecirla.** Más exploración (`ent_coef`) y penalizar la inacción (`exceso_bh`) **no movieron la aguja en absoluto**: con las mismas 5 ventanas y semilla, el agente converge exactamente a la misma política degenerada de no operar nunca (capital final $100.00 en las tres filas, sin diferencia siquiera en el tercer decimal) — la política de "plano" es un atractor lo bastante fuerte en este entorno como para que ni un bonus de entropía 20× más alto que el default ni una recompensa que explícitamente castiga quedarse afuera del mercado lo saquen de ahí en 100.000 pasos de entrenamiento.

Solo la **acción continua** — un cambio estructural, no un hiperparámetro — rompe el atractor y obliga al agente a operar (100 operaciones sobre 100 semanas). Pero operar no ayuda: pierde -16.8%, peor que el umbral simple (-16.0%) y muy por debajo de buy-and-hold (-0.9%), con el Sharpe más negativo de las 7 filas de la tabla. Combinar las 3 mejoras da un resultado prácticamente idéntico a la acción continua sola (-16.8% vs. -16.8%, Sharpe -1.36 vs. -1.51) — evidencia de que `ent_coef` y el reward shaping no aportan nada una vez que el espacio de acción deja de ser el cuello de botella; toda la diferencia la explica forzar al agente a apostar.

**Lectura**: esto no es evidencia de que el diseño del agente esté mal — es evidencia de que el diagnóstico de la sección 9.5 era correcto. Cuando se le impide "no jugar", el agente no encuentra una forma rentable de jugar, porque la señal disponible (|r|<0.11 en las 7 variables del estado) no alcanza para eso. Forzar la acción convierte una abstención racional en una pérdida activa. La propuesta 1 de la sección 9.8 — invertir en mejor señal antes que en mejor agente — queda más respaldada después de este experimento, no menos.

**Limitaciones de este experimento puntual**: un solo valor de `ent_coef` (0.02) y una sola semilla (42, la misma del resto del proyecto) — no se descarta que otro valor de `ent_coef` o promediar sobre varias semillas cambie el resultado del agente base, aunque la acción continua ya muestra que el problema no es exploración insuficiente sino ausencia de señal explotable.

### 9.10 Tareas pendientes en el Issue #2

- [x] Cerrar la Tarea de Notion y el Issue #2 con este hallazgo documentado.
- [x] Evaluar si vale la pena la Tarea "Entrenar agente de RL con datos multi-activo" dado este resultado — ver 9.11/9.12: se priorizó la propuesta 1 de 9.8 (features nuevas / mejor señal) antes que esa Tarea, siguiendo esta misma nota.

### 9.11 Radar-baseline: validar la propuesta 1 de 9.8 antes de construir nada (septiembre 2026)

Antes de invertir en features nuevas, dos preguntas quedaban abiertas: (a) ¿de verdad no hay forma barata de destrabar al agente sin depender de que el PPO converja a "no operar" — podría ser un artefacto del entrenamiento? y (b) ¿qué variables nuevas concretas vale la pena probar? Se investigaron en paralelo (agentes de research, no manual) 18 pares paper+código de estrategias de trading (FX, acciones, ETFs) para responder la segunda pregunta con evidencia en vez de intuición, y se diseñó un experimento independiente del RL para responder la primera.

**Tier 0 — Kelly criterion, sin entrenar ningún agente.** El criterio de Kelly (`f* = μ/σ²`, la fracción de capital que maximiza el crecimiento geométrico esperado) da un segundo veredicto sobre si hay edge explotable, con matemática cerrada en vez de depender de la política que aprenda el PPO. Dos variantes, ambas simuladas con la misma gestión de riesgo real del resto del proyecto (`18_kelly_validacion.py`):

- **Incondicional**: μ y σ² del retorno semanal histórico (ventana de entrenamiento), sin usar ninguna feature.
- **Condicional**: μ predicho por una regresión lineal (OLS, sin look-ahead) sobre las 7 features del estado; σ² = la volatilidad GARCH ya pronosticada para esa semana.

| Estrategia | Retorno total | Sharpe anualizado | Max drawdown | Win rate | Operaciones |
|---|---|---|---|---|---|
| **PPO (RL)** | **0.0%** | — | 0.0% | — | **0** |
| Buy-and-hold | -0.9% | 0.00 | -15.4% | 47.0% | 100 |
| Umbral simple (Opción A) | -16.0% | -0.94 | -20.8% | 51.5% | 68 |
| Kelly incondicional | -51.5% | -3.77 | -52.5% | 29.0% | 100 |
| Kelly condicional (7 features) | -52.9% | -4.12 | -53.6% | 25.0% | 100 |
| Kelly condicional (7 + 5 features nuevas, ver Tier 2) | -53.2% | -4.78 | -52.8% | 25.0% | 100 |

![Curva de capital: Kelly vs. baselines](../datos/resultados/kelly_curva_capital.png)
![Métricas por estrategia: Kelly vs. baselines](../datos/resultados/kelly_metricas_por_estrategia.png)

**No salió lo que se esperaba, y es un resultado mejor por eso.** La hipótesis inicial era que `f*` rondaría 0% (confirmando "no hay edge" con un número chico). En cambio, `f*` incondicional da 1.5-2.2× de apalancamiento en las 5 ventanas — porque USD/CLP tiene una deriva histórica semanal pequeña pero no nula (μ≈0.05-0.09% por semana, consistente con la depreciación de largo plazo de la sección 4.3), y Kelly la apalanca. El problema es que esa deriva **no es una señal estable a frecuencia semanal**: apostarle con el tamaño "óptimo" pierde -51.5% fuera de muestra, peor que el umbral simple y muchísimo peor que no operar. La versión condicional (que además usa las 7 features via regresión) pierde todavía más (-52.9%), y el `f*` implícito antes de acotarlo a ±1 llega a pedir hasta 36× de apalancamiento en algunas semanas — la marca de una regresión ajustando ruido, no de una señal real.

Esto responde la pregunta (a) de arriba: la política de "no operar" del PPO no es un artefacto de que el entrenamiento no encontró el camino — es la decisión correcta dado los datos. Un método completamente distinto (sin redes neuronales, sin exploración, sin hiperparámetros) llega a la misma conclusión por otra vía: **cualquier estrategia que apuesta con este set de variables pierde plata; solo abstenerse conserva capital.**

**Tier 2 — candidatas de features nuevas, con anclaje académico.** El radar identificó 5 candidatas concretas de menor fricción de implementación (`19_features_nuevas_validacion.py`): diferencial de tasas Chile-EE.UU. (`rate_diff`, proxy tasa interbancaria 3m vía FRED — ancla: Filippou, Rapach, Taylor & Zhou 2020, carry trade), retorno y momentum del cobre (`copper_ret_1s`, `copper_mom_4s`, vía `yfinance` — ancla: Chen & Rogoff 2003, "Commodity Currencies", específico para CLP), y momentum del propio USD/CLP normalizado por volatilidad a 4 y 12 semanas (`mom_4s`, `mom_12s` — ancla: Moskowitz, Ooi & Pedersen 2012, "Time Series Momentum").

| Feature | Correlación con retorno futuro | Acierto de dirección |
|---|---|---|
| copper_mom_4s | -0.082 | 43.5% |
| copper_ret_1s | -0.050 | 47.3% |
| mom_4s | -0.042 | 50.3% |
| mom_12s | -0.042 | 50.6% |
| rate_diff | 0.033 | 50.3% |

![Features actuales vs. candidatas nuevas](../datos/resultados/analisis_features_nuevas_correlacion.png)

**Ninguna de las 5 supera el umbral |r|=0.11 de la sección 9.5** — ni siquiera se acercan; la más fuerte (`copper_mom_4s`, -0.082) queda por debajo de 4 de las 7 features originales. El signo de `copper_mom_4s` sí es el esperado económicamente (cobre subiendo ⇒ CLP se aprecia ⇒ USD/CLP baja), lo cual descarta un error de signo/unidades, pero la magnitud es demasiado chica para ser útil a esta frecuencia. Agregar las 5 al set de regresión de Kelly condicional no mejora el resultado del Tier 0 (-53.2% vs. -52.9%, ver tabla arriba) — si acaso, ligeramente peor (más parámetros ajustando el mismo ruido).

**Lectura**: la propuesta 1 de la sección 9.8 ("buscar mejor señal") se probó con las candidatas más baratas y mejor ancladas del radar, y a horizonte **semanal** no aparece señal lineal aprovechable en ninguna de ellas — ni en las 7 originales, ni en las 5 nuevas. Esto no cierra la puerta a esas variables en general: el propio radar señaló que el efecto cobre-CLP es sensible a la frecuencia (Ferraro, Rogoff & Rossi 2015 lo encuentran a frecuencia diaria, no mensual) — cabría repetir este mismo chequeo a frecuencia diaria o mensual antes de descartar cobre/tasas por completo. Lo que sí se puede afirmar con este experimento: al ritmo de decisión actual del proyecto (semanal), ni ensanchar el set de variables con estas 5 candidatas ni cambiar el método de sizing (Kelly en vez de RL) revierte la conclusión de 9.5 y 9.9.

**Caveat de datos**: la tasa de Chile usada es un proxy (tasa interbancaria a 3 meses, serie OECD MEI vía FRED), no la TPM oficial del Banco Central de Chile, y tiene rezago de publicación de 2-4 meses — los meses más recientes del dataset quedan con el último valor conocido (forward-fill), no el dato real de esa semana.

### 9.12 Radar-baseline Tier 1: ¿momentum o Differential Sharpe Ratio destraban al agente?

Aunque el Tier 0/2 (9.11) ya sugería que ninguna variable nueva iba a cambiar la conclusión, se completó igual el Tier 1 — mismo criterio que el Issue #2: agotar variantes razonables con evidencia real antes de concluir, no asumir el resultado de antemano. Dos cambios, esta vez sin necesitar datos externos (`20_agente_rl_ronda2.py`, mismo walk-forward de 5 ventanas × 20 semanas):

1. **Momentum en el estado**: agrega `mom_4s`/`mom_12s` (retorno normalizado por volatilidad a 4 y 12 semanas — ancla: Moskowitz, Ooi & Pedersen 2012) a las 7 features existentes.
2. **Differential Sharpe Ratio como recompensa**: reward recursivo de Moody & Saffell (1998) que penaliza la varianza contra el propio historial reciente del agente, en vez de retorno % crudo o exceso sobre buy-and-hold (ya probado en 9.9).

| Estrategia | Retorno total | Sharpe anualizado | Max drawdown | Win rate | Operaciones |
|---|---|---|---|---|---|
| Buy-and-hold | -0.9% | 0.00 | -15.4% | 47.0% | 100 |
| **PPO + momentum** | **-3.1%** | **-0.73** | **-3.1%** | 0.0% | **2** |
| Umbral simple (Opción A) | -16.0% | -0.94 | -20.8% | 51.5% | 68 |
| **PPO base / + DSR reward / + momentum + DSR** | **0.0%** | — | 0.0% | — | **0** |

![Curva de capital: Tier 1](../datos/resultados/ronda2_rl_curva_capital.png)
![Métricas por estrategia: Tier 1](../datos/resultados/ronda2_rl_metricas_por_estrategia.png)

**Momentum es la única de las 6 variantes probadas hasta ahora (3 del Issue #2 + Kelly + 2 de esta ronda) que mueve al agente sin forzarlo estructuralmente** (a diferencia de la acción continua de 9.9, que lo obliga a operar cada semana). Con acción discreta y recompensa cruda, agregar `mom_4s`/`mom_12s` al estado hizo que el agente tomara **2 operaciones** en 100 semanas de test — pocas, pero ya no cero. El resultado de esas 2 operaciones fue negativo (-3.1%, peor que buy-and-hold) pero muchísimo menos negativo que forzar la acción continua (-16.8%) — es un agente que, con una variable más, encontró 2 momentos donde valía la pena arriesgarse según su propio criterio, no una política degenerada. Es un cambio de comportamiento cualitativamente distinto a todo lo probado hasta ahora, aunque el resultado económico siga sin ser positivo.

**El Differential Sharpe Ratio, en cambio, no lo destraba — ni solo ni combinado con momentum.** Ambas configuraciones con `modo_recompensa="dsr"` convergen a 0 operaciones, exactamente como PPO base. Una hipótesis (no verificada, queda para el futuro): el DSR es una señal muy ruidosa al principio del episodio (la varianza del denominador `B_t - A_t²` es inestable en los primeros pasos — ver nota en `11_entorno_trading_rl.py`), lo que podría hacer que el agente aprenda a evitar esa fuente de ruido quedándose plano en vez de aprovechar la señal de varianza real que el DSR busca capturar.

**Lectura acumulada de 9.9 + 9.11 + 9.12**: de 6 variantes independientes probadas sobre el mismo agente/entorno (más exploración, acción continua, exceso sobre buy-and-hold, Kelly, momentum, DSR), solo dos cambiaron el comportamiento del agente (acción continua y momentum) y ambas perdieron plata igual — ninguna encontró una política rentable. La propuesta 1 de la sección 9.8 queda validada con evidencia, no solo con intuición: **el cuello de botella de este proyecto, a la escala y con los datos disponibles hoy, es la señal — no el agente, no el reward, no el espacio de acción.**

### 9.13 Seguimiento: el efecto del cobre aparece a frecuencia diaria, no semanal

La sección 9.11 dejó un cabo suelto explícito: Ferraro, Rogoff & Rossi (2015) — el paper que ancla la hipótesis de cobre como predictor de CLP — encuentran ese efecto a frecuencia **diaria**, no mensual, y el chequeo de 9.11 se hizo a frecuencia semanal (la que usa el agente). Se repitió el mismo chequeo de correlación (`21_features_nuevas_frecuencias.py`) a diario y a mensual, reusando los mismos datos de cobre/tasas ya descargados.

| Frecuencia | Feature | Correlación con retorno futuro | Supera \|r\|=0.11 |
|---|---|---|---|
| **Diaria** (4.325 obs.) | **`copper_ret_1d`** | **-0.256** | **Sí, más del doble** |
| Diaria | `copper_mom_5d` | -0.165 | Sí |
| Diaria | `copper_mom_20d` | -0.075 | No |
| Diaria | `rate_diff` | 0.004 | No |
| Mensual (197 obs.) | `copper_ret_1m` | 0.105 | No (por poco) |
| Mensual | `rate_diff` | 0.013 | No |
| Mensual | `copper_mom_3m` | 0.004 | No |
| Semanal (9.11, referencia) | `copper_mom_4s` | -0.082 | No |

![Correlación de cobre/tasas por frecuencia](../datos/resultados/analisis_features_nuevas_correlacion_frecuencias.png)

**Es el número más fuerte de toda la investigación de esta noche.** `copper_ret_1d` (retorno diario del cobre, `HG=F` vía `yfinance`) correlaciona -0.256 con el retorno del día siguiente de USD/CLP — más del doble del umbral que ninguna de las 7 variables originales (ni las 5 candidatas semanales de 9.11) logró superar. El signo es el esperado económicamente (cobre sube ⇒ CLP se aprecia ⇒ USD/CLP baja) y `copper_mom_5d` (momentum de 5 días) también supera el umbral (-0.165) — no es un número aislado, hay estructura consistente en la misma dirección a horizontes cortos, que se diluye con la ventana (`copper_mom_20d` ya cae a -0.075, mensual prácticamente a cero). Confirma exactamente el patrón que describe el paper ancla: el efecto cobre-CLP existe pero es de corto plazo, y se pierde al resamplear a semanas o meses — que es justo la escala a la que opera el agente hoy.

**Caveat metodológico real, no cosmético, antes de festejar el número**: no se validó el timestamp exacto de cierre de `HG=F` (COMEX, vía `yfinance`) contra el de `CLP=X` (el ticker que usa `01_obtener_datos.py` para USD/CLP). Si el cierre del cobre que `yfinance` reporta para el "día t" ocurre, en la práctica, más tarde en el reloj que el cierre de CLP=X para ese mismo día calendario, parte de esta correlación podría ser información contemporánea filtrándose como si fuera predictiva (look-ahead sutil), no una relación causal de un día para el otro. Con los datos de walk-forward de otras partes del proyecto (ej. USD/CLP diario en la sección 4.1) esta clase de detalle de alineación temporal ya demostró importar — no se debe asumir que "un día de diferencia" es suficiente sin comprobarlo.

**Lectura**: esto no cambia la conclusión de que el agente semanal actual no tiene señal explotable — sigue sin tenerla, con las variables y a la frecuencia que opera hoy. Pero sí cambia el diagnóstico de fondo: el problema no es necesariamente que "USD/CLP no tiene edge explotable en ningún lado" — es que, a la frecuencia semanal, la señal de cobre (que sí existe a diario) ya se diluyó. Antes de invertir en Tier 3/4 (multi-activo, arquitecturas nuevas), validar el timestamp y, si se confirma, evaluar si vale la pena una versión diaria del agente es probablemente la pista de mayor retorno esperado por hora invertida de todo lo que se investigó esta noche — pero es una decisión de alcance (cambiar la escala de todo el pipeline) que le corresponde a Bastián, no algo para decidir en automático.

### 9.14 Validación del hallazgo diario: timestamp limpio, backtest rentable, y por qué "más monedas" no ayudó

Bastián pidió validar el caveat de timestamp de 9.13 aunque resultara válido, probar si la señal es rentable en un backtest real, y ampliar el dataset a un panel de al menos 10 pares de forex — los tres pasos siguientes.

**Validación de timestamp**: en vez de reconstruir a mano el horario exacto de cierre de `HG=F` (COMEX, huso horario America/New_York) contra `CLP=X` (Europe/London en yfinance, con huecos de fin de semana por ser un par poco líquido), se usó una prueba más directa y decisiva — un escaneo de correlación por rezago:

| Rezago (copper_ret en t vs. retorno CLP en t+rezago) | Correlación |
|---|---|
| -2 | -0.009 |
| -1 | -0.050 |
| **0 (mismo día)** | **-0.021** |
| **+1 (predictivo, el usado en 9.13)** | **-0.254** |
| +2 | -0.069 |
| +3 | -0.032 |

El efecto está **concentrado casi por completo en el rezago +1** — si hubiera contaminación por solapamiento de cierres (información del mismo día filtrándose como si fuera predictiva), se esperaría un efecto fuerte también en el rezago 0. La firma es la de un lead-lag genuino de un día, no un artefacto de alineación de timestamps. No es una auditoría exhaustiva al minuto, pero descarta el mecanismo específico de contaminación que se sospechaba.

**Backtest de rentabilidad** (`22_kelly_diario_cobre.py`, walk-forward de 5 ventanas × 60 días = 300 días de test, ~14 meses):

| Estrategia | Retorno total | Sharpe anualizado | Max drawdown | Operaciones |
|---|---|---|---|---|
| **Kelly diario (con cobre)** | **+86.2%** | **4.57** | **-3.5%** | 300 |
| Umbral cobre (solo signo, sin regresión) | +83.5% | 4.44 | -3.4% | 290 |
| Kelly diario (sin cobre) | +6.0% | 0.47 | -9.7% | 300 |
| Buy-and-hold | -2.7% | -0.13 | -15.4% | 100 |

![Curva de capital: Kelly diario con/sin cobre](../datos/resultados/kelly_diario_cobre_curva_capital.png)

Sharpe >4 es un número que hay que mirar con sospecha, no con entusiasmo automático — se verificó que no viene de una sola ventana con suerte: las 5 ventanas del walk-forward son todas positivas individualmente (ganancia de $5.6 a $23.5 cada una), y que la señal simple ("umbral cobre", sin regresión, solo el signo de `copper_ret_1d`) rinde casi lo mismo que la versión con regresión (Kelly) — la ventaja viene genuinamente de la dirección del cobre, no de ruido que la regresión está sobreajustando. Nota de método: a diferencia del backtest semanal (14), esto NO reproduce el trailing-stop/take-profit intradía — a frecuencia diaria no hay datos más finos que el cierre para chequear eso, así que se simula cierre-a-cierre con el mismo costo de slippage. El periodo de test cubre 2025-07 a 2026-09 — los últimos ~14 meses del dataset, no una muestra aleatoria de toda la historia; no se descarta que el resultado sea parcialmente específico del régimen reciente de cobre (una tendencia fuerte y sostenida) y no necesariamente igual de fuerte en periodos de cobre lateral.

**Corrección importante, encontrada al extender este backtest a los 13 pares del panel**: el +86.2% de la tabla de arriba está inflado por un problema real de la fracción de Kelly calculada — la posición está en el límite de apalancamiento (±1, el 100% del capital) el **97% de los días de test**, tanto para CLP como para USD/PEN (que con este mismo método da un absurdo +860% / Sharpe 9.0, ver `panel_fx_backtest_por_par_metricas.csv`). La causa es matemática, no un bug de código: a frecuencia diaria la varianza del retorno (`vol_realizada²`) es un número muy chico, así que `f*=μ/σ²` explota a valores enormes (hasta 30-40× apalancamiento antes del clip) frente a casi cualquier `μ` predicho — el clip a ±1 esconde que el modelo, sin darse cuenta, está recomendando apostar toda la cuenta todos los días, no dimensionar la apuesta según la confianza real. Ni reducir la fracción de Kelly a 5% del valor calculado cambia esto (sigue saturado el 69% de los días) — el problema es de escala, no de exceso de agresividad puntual.

Separando el tamaño de la apuesta de la dirección (usar tamaño **fijo** del capital en vez del tamaño que sugiere Kelly, solo la dirección que predice la regresión) se obtiene una lectura mucho más honesta de cuánto se ganaría en la práctica:

| Tamaño de apuesta diaria | Retorno total (300 días) | Sharpe anualizado |
|---|---|---|
| 100% del capital (equivalente al backtest de arriba) | +85.6% | 4.53 |
| 20% del capital | +13.3% | 4.53 |
| 10% del capital | +6.5% | 4.53 |
| 3% del capital (similar al riesgo por operación del agente semanal) | +1.9% | 4.53 |

**El Sharpe (4.53) no cambia con el tamaño de la apuesta — es la parte del hallazgo que sí es confiable**, porque es una medida de calidad de la señal día a día, no del apalancamiento acumulado. Pero el retorno total en dólares depende enteramente de cuánto se esté dispuesto a arriesgar, y el titular de "+86%" asumía, sin decirlo explícitamente, apostar el 100% de la cuenta todos los días durante 14 meses seguidos — algo que ningún trader prudente haría. Con un tamaño de riesgo comparable al que ya usa el agente semanal (3% del capital), la ganancia de 14 meses baja a +1.9%: sigue siendo positiva y con buen Sharpe, pero está lejos del titular original. **Lectura correcta: la dirección de la señal es sólida (Sharpe ~4.5 estable a cualquier escala), pero "cuánto se gana" depende 100% de una decisión de gestión de riesgo que todavía no se ha tomado con criterio — no es una cifra que el backtest entregue solo.**

**Panel de 13 pares de forex** (`23_dataset_multi_par_diario.py`, dataset nuevo en `datos/bases/panel_fx_diario.csv`): CLP, MXN, BRL, COP, PEN, ZAR, CAD, AUD, NZD (economías commodity — cobre, petróleo, hierro, lácteos, oro) más JPY, CHF, EUR, GBP (no-commodity, grupo de comparación). Nota técnica: al descargar estos pares frescos desde yfinance apareció el mismo bug de ticks corruptos de un solo día ya documentado en `01_obtener_datos.py` para CLP=X (ej. un valor de "5.46" en vez de ~544) — apareció también en COP=X (22 días), PEN=X, ZAR=X y CHF=X, y se corrigió reusando la misma función `limpiar_ticks_erroneos`, no una nueva.

![Correlación de cobre por par](../datos/resultados/panel_fx_correlacion_cobre_por_par.png)

**El efecto generaliza, y de forma que confirma la teoría en vez de contradecirla**: las 3 monedas commodity más "puras" del panel (AUD, NZD, CAD — economías mineras/agrícolas clásicas) muestran la correlación **más fuerte** de las 13 (0.31 a 0.37, más alta que la propia CLP en 0.25), y USD/JPY (refugio, la menos "commodity" de todas) muestra la **más débil** (0.02) — el patrón se ordena casi exactamente como predice Chen & Rogoff (2003). Matices honestos: GBP/EUR (no-commodity) igual muestran correlación moderada (0.25/0.22) — probablemente un factor más amplio de "riesgo-on / debilidad del dólar" que el cobre también capta, no solo el canal específico de materias primas; y PEN (cobre, como CLP) sale sorprendentemente bajo (0.06) — posible efecto de que el Banco Central de Perú interviene el tipo de cambio activamente, amortiguando la respuesta diaria.

**¿Cuántas operaciones hace el backtest en cada par?** Se repitió el mismo backtest de Kelly diario (entrenado con la historia propia de cada moneda, sin pooling) para los 13 pares — la respuesta es **300 de 300 días de test en los 13 pares, sin excepción** (`panel_fx_backtest_por_par_metricas.csv`). No es que "opera casi siempre": con esta forma de calcular la posición, opera literalmente todos los días — nunca se queda plano, porque el `f*` de Kelly rara vez da exactamente cero. Eso llevó a encontrar el problema de apalancamiento que se corrige arriba: al revisar por qué **USD/PEN** daba un retorno de +860% (Sharpe 9.0, el "mejor" resultado del panel por lejos, a pesar de tener la correlación con cobre más débil de todo el grupo commodity) se confirmó que la posición estaba en el límite de apalancamiento el 97% de los días — el mismo problema que en CLP, solo que amplificado por la combinación particular de aciertos de dirección de PEN en este periodo. **Ese +860% no es un hallazgo real, es el mismo artefacto de apalancamiento máximo constante, no una oportunidad de PEN en particular.**

**¿Ayuda entrenar con el panel completo en vez de solo la historia de CLP?** Se probó, sobre el mismo test de CLP y el mismo esquema de walk-forward (5 reentrenos), un modelo entrenado únicamente con `copper_ret_1d`/`copper_mom_5d`/`retorno_1d`/`macd_rel`/`rsi_norm` de USD/CLP contra el mismo modelo entrenado con esas mismas features pero usando las filas de los **13 pares pooled** (más de 55.000 filas de entrenamiento vs. ~4.000):

| Estrategia | Retorno total | Sharpe anualizado |
|---|---|---|
| Kelly diario (solo CLP) | +83.0% | 4.42 |
| Kelly diario (panel pooled, 13 pares) | +67.9% | 3.77 |

![Curva de capital: pooled vs. solo CLP](../datos/resultados/panel_fx_pooled_vs_single_curva_capital.png)

(Mismo caveat de apalancamiento que arriba aplica a estos dos números — ambos están inflados por apostar ~100% del capital casi todos los días. La comparación *relativa* entre ambos sigue siendo válida porque el mismo sesgo afecta a los dos por igual.)

**No ayudó — el modelo entrenado solo con CLP superó al pooled.** Tiene una explicación directa a la luz del gráfico anterior: la sensibilidad al cobre **varía bastante entre monedas** (0.02 a 0.37), así que agrupar todas las filas en una sola regresión "promedia" el coeficiente de CLP con el de monedas mucho más sensibles (AUD/CAD/NZD) y mucho menos sensibles (JPY) al cobre, diluyendo la calibración específica de CLP en vez de reforzarla. Es la segunda confirmación independiente (junto con la tesis de transfer learning EUR/USD→GBP/USD del radar-baseline, sección 9.11 del Artifact) de que sumar datos de otras monedas **no es una mejora automática** — hace falta un mecanismo más sofisticado que "pooling ingenuo" (ej. el contexto compartido vía cross-attention de X-Trend, que aprende a ponderar qué monedas son relevantes para cada una, en vez de promediar a todas por igual) para que más datos realmente ayuden.

### 9.15 Decisiones pendientes (no tomadas en automático)

Este tramo (9.11-9.14) mezcló trabajo sin supervisión directa (9.11-9.13, priorizando lo de menor costo/reversibilidad) con pasos pedidos explícitamente por Bastián después (9.14: validar el timestamp, probar rentabilidad, ampliar a un panel de 10+ pares). Lo que queda es de mayor alcance — construir algo nuevo, no solo investigar — y se deja explícito para que la decisión de invertir ahí sea de Bastián:

- [x] Repetir el chequeo de correlación de cobre/tasas a frecuencia diaria y mensual (9.13) — hecho: `copper_ret_1d` supera el umbral por más del doble a diario, se diluye a semanal/mensual.
- [x] Validar el timestamp de cierre de `HG=F` vs. `CLP=X` (9.14) — el escaneo de rezagos no muestra contaminación por solapamiento (efecto concentrado en el rezago +1, no en el 0).
- [x] Probar rentabilidad en un backtest real (9.14) — +86.2% / Sharpe 4.57 en 300 días de test, consistente en las 5 ventanas, no explicado por una sola operación. **Corregido después**: ese +86.2% asume apostar el 100% del capital casi todos los días (97% de saturación de apalancamiento) — a un tamaño de riesgo realista (3%, como el agente semanal) el retorno de 14 meses baja a +1.9%. El Sharpe (~4.5) sí es confiable a cualquier escala; el retorno en dólares no lo es sin fijar antes una regla de sizing sensata.
- [x] Ampliar el dataset a un panel de forex de 10+ pares (9.14) — 13 pares en `datos/bases/panel_fx_diario.csv`. El efecto generaliza (más fuerte incluso en AUD/CAD/NZD que en CLP), pero pooling ingenuo del panel completo **empeora** el resultado específico de CLP frente a entrenar solo con su propia historia (mismo caveat de apalancamiento aplica a los dos números, la comparación relativa entre ambos sigue siendo válida).
- [ ] **Antes de cualquier otra cosa**: definir una regla de sizing real (no Kelly sin acotar) para todos los resultados de 9.14 — el candidato más simple es reusar directamente `RIESGO_MAX_PCT` del agente semanal (arriesgar un % fijo del capital por el tamaño del stop, no el `f*` crudo de la regresión). Sin esto, ningún número de retorno de esta sección (CLP, el panel de 13 pares, pooled vs. solo-CLP) es comparable a los backtests del resto del proyecto (14, 17, 20), que sí usan risk sizing acotado.
- [ ] **La pregunta grande que sigue abierta**: ¿vale la pena una versión diaria del agente/pipeline (no solo un backtest de Kelly) que use `copper_ret_1d` como feature? Es un cambio de escala real — regenerar el dataset a diario, rediseñar el TP/SL (a diario no hay datos intradía para el trailing stop actual, se simplificaría a cierre-a-cierre como en 9.14), redefinir walk-forward — del tamaño del Issue #1 original. La evidencia de 9.14 lo respalda (el Sharpe estable ~4.5 sobrevive a la corrección de sizing), pero el periodo de test (14 meses recientes) no cubre múltiples regímenes de cobre — valdría la pena, antes de comprometerse, correr el mismo backtest de 22 sobre ventanas históricas más antiguas (ej. 2015-2018, 2019-2021) para ver si el Sharpe se sostiene fuera del tramo alcista reciente.
- [ ] Si se decide ir a diario: el panel de 13 pares ya construido (9.14) sirve como insumo directo para una versión "contexto compartido" real (estilo X-Trend) en vez de pooling ingenuo — encoder de tendencia entrenado sobre el panel, no una regresión OLS que promedia todo por igual.
- [ ] Decidir si este trabajo (9.11-9.14) se formaliza como un Issue de GitHub retroactivo o se documenta solo en el paper — no se abrió Issue nuevo porque no había uno para el radar-baseline en sí, a diferencia de los Issues #1/#2.
- [ ] Actualizar/cerrar la Tarea de Notion "Entrenar agente de RL con datos multi-activo" a la luz de este hallazgo (sigue pendiente, sin tocar) — el resultado de 9.14 sugiere que la versión "pooling simple" de esa Tarea probablemente no ayudaría; si se retoma, hacerlo con un mecanismo más parecido a X-Trend.
- [x] Sesión cerrada acá (2026-09-15) a pedido de Bastián. Se creó una Tarea nueva en Notion, "Reconstruir pipeline de trading RL a frecuencia diaria (USD/CLP + cobre)" (Estado=Por hacer, Prioridad=Media, proyecto vinculado), para retomar la reconstrucción completa en un chat nuevo — con todo el contexto técnico resumido ahí mismo, apuntando a esta sección del paper.

### 9.16 Cierre de la línea "buscar mejor señal" semanal: SMA/EMA/Bollinger/CCI/ADX tampoco superan el umbral (Issue #4)

Última extensión de la propuesta 1 de 9.8 (Tarea de Notion "Probar medias móviles..."), después de que MACD, RSI, momentum (9.11) y tasas/cobre semanal (9.11) ya hubieran fallado. Se probaron 12 indicadores técnicos adicionales, ninguno usado antes en este proyecto: SMA y EMA relativas al precio a 5/10/20/50 semanas, ancho de banda y `%B` de Bollinger (20 semanas, 2 desv. estándar), y CCI/ADX (20 y 14 semanas respectivamente) — estos dos últimos señalados en la ronda 1 del radar-baseline (vía FinRL/Qlib Alpha158) como indicadores nunca probados acá. CCI y ADX necesitan High/Low, que `01_obtener_datos.py` descarta (solo guarda Close); se descargó un OHLC semanal separado (`datos/bases/usdclp_ohlc_semanal.csv`, cache propio) en vez de tocar el pipeline principal.

**Resultado**: ninguna de las 12 supera |r|=0.11. La más fuerte es `ema_50_rel` con -0.076 — menos de la mitad del umbral, y más débil que 4 de las 7 features originales y que `copper_mom_4s` (-0.082) de la ronda anterior. `cci` y `bb_pctb` quedan prácticamente en cero (0.007 y -0.004). Confirma la expectativa honesta de la propia Tarea: son transformaciones del mismo precio que ya había fallado en sus otras formas (MACD, RSI, momentum), y suavizar más (SMA/EMA/Bollinger) no agrega información nueva, solo la retrasa.

![Correlación SMA/EMA/Bollinger/CCI/ADX vs. todo lo probado hasta ahora](../datos/resultados/analisis_features_sma_ema_correlacion.png)

Con esto se agotan las 12 candidatas de indicadores técnicos derivados del precio semanal de USD/CLP identificadas hasta ahora (7 originales + 5 de la ronda radar-baseline + estas 12 no se solapan, aunque MACD/RSI ya estaban en las 7 originales). Ninguna superó el umbral. Dado el paso 2 condicional de la Tarea ("si alguna supera 0.11, agregarla al PPO") no aplicó, no se entrenó ningún agente nuevo — se cierra la línea sin gasto de cómputo, mismo criterio de "barato antes de caro" que las secciones anteriores. La pista con mayor evidencia real sigue siendo la de 9.13/9.14 (cobre a frecuencia diaria, fuera del alcance semanal de esta línea).

### 9.17 Issue #5: reconstrucción completa del agente de RL a frecuencia diaria

La sección 9.15 dejó una pregunta grande abierta: ¿vale la pena reconstruir el pipeline completo (dataset, entorno, agente) a frecuencia diaria para explotar `copper_ret_1d` (|r|=-0.256, más del doble de cualquier variable semanal), dado que el backtest de Kelly de 9.14 solo se había probado en el tramo reciente (2025-07 a 2026-09) del cobre? El Issue [#5](https://github.com/bastianbm7/usdclp-nbeats-arima-forecasting/issues/5) pidió, explícitamente, validar eso primero antes de comprometerse al trabajo caro.

**Paso 1 — validación histórica (`25_validacion_historica_cobre_diario.py`)**: se repitió exactamente el mismo método de `22_kelly_diario_cobre.py` (Kelly condicional vía OLS + umbral simple, walk-forward de 5 ventanas × 60 días) en 3 periodos históricos adicionales, truncando el dataset diario a distintas fechas de corte en vez de reescribir la lógica de walk-forward (el truncamiento reutiliza `ventanas_walkforward()` sin cambios: el "final" de un dataframe truncado a una fecha vieja pasa a ser el periodo de test de esa época).

| Periodo | Test | Retorno total | Sharpe anualizado |
|---|---|---|---|
| 2015-2016 | 2015-11 a 2016-12 | +192.2% | **5.42** |
| 2017-2018 | 2017-11 a 2018-12 | +63.6% | 3.25 |
| 2020-2021 | 2020-11 a 2021-12 | +37.2% | 2.07 |
| 2025-2026 (referencia, igual a 22) | 2025-07 a 2026-09 | +86.2% | 4.57 |

![Sharpe de la señal de cobre por periodo histórico](../datos/resultados/validacion_historica_cobre_diario_sharpe_por_periodo.png)

**La señal se sostiene fuera del régimen alcista reciente del cobre — de hecho, la ventana más vieja (2015-2016) tiene el Sharpe más alto de las cuatro.** Ningún periodo da Sharpe negativo ni cercano a cero; el rango completo (2.07 a 5.42) está muy por encima del umbral que cualquier variable semanal logró superar en toda la investigación previa (9.5, 9.11-9.13, 9.16). Esto descarta que el hallazgo de 9.14 fuera específico de un régimen de cobre puntual, y respalda avanzar a la reconstrucción completa (paso 2).

**Paso 2 — reconstrucción completa.** Antes de lanzar el walk-forward de NHITS a diario, se midió (no se asumió) el costo real: refit=True en cada día, igual que hace `10_generar_dataset_rl.py` semana a semana, se midió en ~8.9s/ventana — casi idéntico por ventana al semanal, pero con ~12× más puntos de decisión (4.346 días vs. 349 semanas), lo que habría tardado **~10.7 horas** sobre toda la historia. Se optó, en cambio, por una cadencia de refit cada 5 días hábiles (horizonte NHITS h=7 en vez de h=2), reutilizando cada fit para las 5 decisiones diarias siguientes — medido en ~9.8s/refit, cubriendo 5 días cada vez, es decir ~5× más barato por día cubierto. El costo real de generar 1.464 días de decisión (~5.8 años, 2020-12 a 2026-09) con esta cadencia fue de **~52 minutos** (`26_generar_dataset_rl_diario.py`), del mismo orden de magnitud que los ~58 minutos del dataset semanal. El trade-off explícito: dentro de cada bloque de 5 días, el forecast NHITS que ve el agente tiene hasta 4 días de antigüedad (no se recalibra contra el cierre de ayer, sino contra el de hasta 4 días atrás) — GARCH, MACD/RSI y las features de cobre sí se recalculan todos los días (son baratos, no generan el mismo problema de escala).

**Entorno nuevo, no modificado el semanal**: `27_entorno_trading_rl_diario.py` es un archivo separado de `11_entorno_trading_rl.py` (que el Issue [#6](https://github.com/bastianbm7/usdclp-nbeats-arima-forecasting/issues/6), multi-activo FX, trabaja en paralelo sobre el mismo archivo semanal, en otra rama). Mismo `RIESGO_MAX_PCT`/mecanismo de risk sizing que el agente semanal — la corrección que 9.14/9.15 identificaron como necesaria (Kelly sin acotar saturaba el apalancamiento el 97% de los días) queda incorporada desde el diseño, no parcheada después. **Trailing stop repensado**: a una posición que se abre y resuelve entre un cierre y el siguiente no hay un segundo punto intradía para "trailear" (el stop necesita al menos 2 observaciones para moverse) — se simplifica, documentado explícitamente en el código, a un chequeo de take-profit/stop-loss de un solo paso contra el único precio disponible (el cierre de mañana).

**Bug encontrado y corregido, con el mismo criterio de "verificar antes de reportar" que BUG #3 (ver bitácora del proyecto)**: la primera corrida del backtest final dio un resultado catastrófico y sospechoso para "Umbral cobre" (-99.75%, Sharpe -8.26) — exactamente el tipo de número que amerita desconfiar antes de creerlo. La causa: `calcular_salida_dia()` usaba el forecast NHITS (`nhits_h1`) como precio de take-profit sin verificar que estuviera del lado *ganador* de la posición. Para estrategias cuya dirección viene del propio NHITS (el agente semanal, y "Umbral simple" en este mismo script) el take-profit siempre cae del lado correcto por construcción — pero "Umbral cobre" elige dirección según `copper_ret_1d`, una señal sin relación con hacia dónde apunta NHITS, así que el "take profit" podía terminar en el lado perdedor. Confirmado en los datos: 170 de 300 cierres etiquetados "take_profit" tenían PnL promedio **negativo** (-96.2 de los -99.75 puntos totales de pérdida). Corregido: el take-profit solo dispara si de verdad está del lado favorable de la posición; si no, se ignora y solo quedan stop-loss/cierre de día como salida. Como esto también pudo afectar el propio entrenamiento del PPO (el agente puede elegir direcciones que no coincidan con NHITS), se reentrenó el agente completo con el entorno corregido antes de reportar nada.

**Backtest final walk-forward** (`28_backtest_walkforward_diario.py`, mismo esquema que `14_backtest_walkforward_gestion_riesgo.py`: 5 ventanas × 60 días de test, 300 días out-of-sample, 2025-06 a 2026-09, PPO reentrenado en cada ventana):

| Estrategia | Retorno total | Sharpe anualizado | Max drawdown | Win rate | Operaciones |
|---|---|---|---|---|---|
| **Umbral cobre** | **+532.3%** | **4.39** | -13.5% | 60.0% | 290 |
| **PPO (RL diario)** | **+397.1%** | **3.88** | -8.8% | 58.9% | 299 |
| Umbral simple (forecast) | +22.9% | 0.83 | -24.0% | 51.1% | 133 |
| Buy-and-hold | -1.2% | -0.02 | -13.2% | 50.0% | — |

![Curva de capital: backtest walk-forward diario](../datos/resultados/walkforward_diario_curva_capital.png)
![Puntos de entrada/salida del agente PPO diario](../datos/resultados/walkforward_diario_puntos_entrada_salida.png)

**A diferencia de la versión semanal (9.4, 9.9, 9.12), el agente PPO diario SÍ encuentra una política rentable** — con gestión de riesgo real (3% de capital arriesgado por operación, TP/SL, slippage), no una simplificación. Verificación post-fix antes de reportar (mismo criterio que arriba): la razón de cierre "take_profit" ahora tiene PnL promedio positivo en ambas estrategias apalancadas (+2.63 en Umbral cobre, +3.22 en PPO), "stop_loss" negativo como corresponde, y la curva de capital compone gradualmente a lo largo de los 300 días sin saltos discontinuos ni una sola operación explicando el resultado — no es un artefacto de un día con suerte.

**Lectura honesta de la comparación PPO vs. Umbral cobre**: la señal simple (solo el signo de `copper_ret_1d`) le gana al agente entrenado — +532% contra +397%. El PPO no ignora la señal (coincide en dirección con "Umbral cobre" el 72.3% de los días, no una política degenerada ni una copia ciega), pero el ~28% de los días donde diverge — presumiblemente combinando otras variables del estado (NHITS, MACD, RSI, volatilidad) — le resta más de lo que le suma. Es un resultado consistente con lo ya visto en 9.14 (el umbral simple sin regresión rendía casi igual que Kelly condicional): cuando una sola variable concentra la mayor parte de la señal explotable, agregar más complejidad no garantiza mejorar sobre usarla directamente.

**Limitaciones de esta reconstrucción, dichas sin atenuar**:
- **Staleness del forecast NHITS**: hasta 4 días de antigüedad dentro de cada bloque de refit de 5 días (ver arriba) — un trade-off de cómputo explícito, no gratis. `vol_garch`, MACD/RSI y las features de cobre sí son frescos cada día.
- **Un solo periodo de test para el backtest final**: las 300 días de walk-forward (2025-06 a 2026-09) caen dentro del mismo tramo alcista reciente del cobre que el paso 1 ya identificó como el de Sharpe más bajo (2.07-4.57) de los cuatro periodos históricos — el paso 1 usó el método más simple (Kelly/umbral) para cubrir más historia; repetir el backtest completo (dataset NHITS + PPO) en 2015-2016 o 2017-2018 queda pendiente si se quiere la misma confianza sobre la versión final con gestión de riesgo real y PPO.
- **Sin comisión** (solo slippage 0.05%, igual que el resto del proyecto) — con 290-299 operaciones en 300 días, una comisión realista (aunque baja) le pega más a las estrategias activas que al buy-and-hold.
- **Retornos muy altos en términos absolutos** (+397% a +532% en 14 meses): se verificaron activamente antes de reportar (ver el bug encontrado y corregido arriba, y la inspección de PnL por razón de cierre) y el Sharpe (~4) es consistente con el validado en 4 periodos históricos independientes en el paso 1 — pero un track record de 300 días, por más prolijo que sea el walk-forward, sigue siendo corto frente a los 16 años de historia completa del dataset.
- **`Umbral simple (forecast)` no es un tercer punto de comparación limpio para el hallazgo del cobre**: usa el forecast NHITS (staleness de hasta 4 días) para dirección, no `copper_ret_1d` — se mantiene en la tabla por paralelismo con `14`, no porque sea la comparación más relevante para esta sección.

**Decisión de scope**: se avanzó directo del paso 1 al paso 2 completo (dataset, entorno, agente, backtest) en la misma sesión, sin pausar a pedir aprobación intermedia — la validación histórica del paso 1 no mostró ninguna señal de alarma (ningún Sharpe negativo o cercano a cero) que ameritara frenar antes de comprometerse al trabajo caro, criterio ya establecido en la Tarea de origen de este Issue.

## Reproducibilidad

Todo el código está en `codigos/` (scripts `01` a `08` para el forecasting de precio; `09` a `28` para la extensión de trading con RL, incluyendo la reconstrucción a frecuencia diaria del Issue #5 — ver `README.md` del repositorio para el detalle de cada uno y cómo correrlos), y todos los resultados numéricos y gráficos citados en este documento están versionados en `datos/resultados/`.
