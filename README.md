# usdclp-nbeats-arima-forecasting

Compara modelos modernos de forecasting — **N-BEATS** y **N-HiTS** (vía [Nixtla/neuralforecast](https://github.com/Nixtla/neuralforecast), Apache-2.0, usado como dependencia) — contra un baseline clásico (**AutoARIMA** y **Naive**, vía [Nixtla/statsforecast](https://github.com/Nixtla/statsforecast)) pronosticando el tipo de cambio **USD/CLP**.

📄 **[Ver el paper completo](reportes/paper.md)** — resumen, metodología, resultados y discusión de las 4 partes del estudio (diario, mensual, anual, aprendizaje online) en un solo documento. Este README es la referencia práctica (cómo está armado el repo, cómo correrlo); el paper es la narrativa de investigación.

## Pregunta que responde

¿Un modelo moderno de deep learning aporta una mejora real y medible sobre un baseline clásico al pronosticar el tipo de cambio, o la mejora observada es mayormente autocorrelación del precio diario (el valor de hoy predice casi perfecto el de mañana)?

## Metodología

- Backtesting walk-forward (múltiples ventanas de validación, no un solo split train/test).
- Horizonte de pronóstico de varios pasos (no 1 solo paso) — a 1 paso el naive casi siempre "gana" por autocorrelación pura; a varios pasos esa ventaja se diluye.
- Métricas de error en test: RMSE, MAE, MAPE — calculadas para los cuatro modelos (Naive, AutoARIMA, NBEATS, NHITS) sobre las mismas ventanas.
- El mismo análisis se repite a **tres escalas de tiempo** (diaria, mensual, anual) resampleando la misma serie — no son datasets distintos, es la pregunta "¿un modelo moderno mejora al baseline clásico?" respondida en cada horizonte por separado, porque la estructura dominante de la serie cambia con la escala (ver síntesis al final).

## Resultados diarios

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

## Resultados mensuales

Misma comparación, resampleando a cierre de mes (200 observaciones, 2010-01 a 2026-08), backtesting de 5 ventanas de 6 meses cada una.

| Modelo | RMSE | MAE | MAPE | Mejora RMSE vs. mejor baseline |
|---|---|---|---|---|
| **N-BEATS** | **33.09** | **30.82** | **3.28%** | **+20.7%** |
| Naive | 41.71 | 35.40 | 3.81% | — (mejor baseline) |
| N-HiTS | 42.57 | 35.11 | 3.77% | -2.1% |
| AutoARIMA | 44.26 | 38.77 | 4.16% | -6.1% |

![Predicción vs. real mensual](datos/resultados/prediccion_vs_real_mensual.png)
![RMSE por modelo mensual](datos/resultados/rmse_comparacion_mensual.png)

Acá se invierten los papeles: **N-BEATS** es el que gana (+20.7%), no N-HiTS — confirmado que no es casualidad de una ventana (se verificó la varianza del pronóstico dentro de cada ventana, en la misma escala que la real, no un artefacto de predicción plana). Ninguno de los dos modelos de Nixtla es sistemáticamente mejor en todas las escalas; cuál gana depende de la frecuencia. Un detalle honesto que el gráfico deja ver: en la ventana de cutoff 2024-12, ambos modelos de Nixtla sobre-extrapolan una racha alcista reciente y proyectan un salto a ~1.050-1.080 que la serie real nunca llega a tocar (se queda en ~950-990) — un caso real de sobre-reacción a momentum de corto plazo, no oculto en el resultado agregado porque el resto de las ventanas compensa.

## Resultados anuales

Misma comparación, resampleando a cierre de diciembre (16 observaciones completas, 2010-2025; se descarta 2026 por ser un año incompleto), backtesting de 3 años. **A propósito no se incluyen N-BEATS/N-HiTS acá** — con 16 puntos, entrenar una red neuronal no tiene sustento estadístico real (cualquier resultado sería ruido de una corrida, no una señal reproducible); se compara solo contra métodos clásicos diseñados para poca data.

| Modelo | RMSE | MAPE | Mejora RMSE vs. Naive |
|---|---|---|---|
| **AutoARIMA** | **3.72** | **0.42%** | **+87.0%** |
| **RandomWalkWithDrift** | **3.72** | **0.42%** | **+87.0%** |
| Naive | 28.64 | 3.24% | — |

![Predicción vs. real anual](datos/resultados/prediccion_vs_real_anual.png)
![RMSE por modelo anual](datos/resultados/rmse_comparacion_anual.png)

El resultado más limpio de los tres análisis: `AutoARIMA` y `RandomWalkWithDrift` dan **el mismo número exacto**, porque AutoARIMA elige por su cuenta un ARIMA(0,1,0) con drift — que es matemáticamente idéntico a un random walk con tendencia. A escala anual, USD/CLP se comporta como una caminata aleatoria con una depreciación promedio constante, y con solo agregar esa tendencia (nada de deep learning) se le gana al naive plano por 87%. Caveat honesto: ese mismo modelo con drift sobre-extrapola la ventana más reciente (2025→2026), proyectando ~1.030 cuando el real bajó a ~900 — la ganancia agregada de 3 ventanas no significa que el drift sea infalible ventana a ventana, y con n=16 cualquier métrica agregada tiene un margen de error grande.

## Síntesis entre escalas

| Escala | ¿Gana algo al naive? | Qué gana | Por qué |
|---|---|---|---|
| Diaria | Sí, +20% | N-HiTS | Único con libertad real de forma que la aprovechó bien; Naive/AutoARIMA quedan matemáticamente forzados a ser planos (random walk / MA(1) sin memoria) |
| Mensual | Sí, +21% | N-BEATS | Con menos ruido de alta frecuencia, hay más momentum/estructura real para que una red capture — pero el "ganador" entre N-BEATS/N-HiTS cambia según la escala |
| Anual | Sí, +87% | AutoARIMA = RandomWalkWithDrift | A esta escala domina una tendencia determinística (depreciación de largo plazo) más que el ruido — ni hace falta deep learning para capturarla |

La lectura de portfolio no es "el deep learning gana siempre" ni "el baseline clásico gana siempre" — es que **la estructura dominante de una serie financiera cambia con la escala de tiempo**, y el modelo ganador cambia con ella. Eso es más honesto (y más interesante) que un único gráfico con un único ganador.

## Aprendizaje online: horizontes cortos (1-3 pasos) con `refit=True`

Los análisis anteriores entrenan una vez con datos históricos y nunca más se actualizan. Acá se prueba lo contrario — que el modelo se **reentrene en cada ventana a medida que aparecen datos nuevos**, con horizontes bien cortos (1, 2 y 3 pasos), sobre tres escalas (diaria/semanal/mensual). Esto ya está soportado nativo en Nixtla, no hace falta código de online-learning a mano:

- `statsforecast` ya reentrena por ventana **por default** (`refit=True` es el default en su `cross_validation`) — por eso `AutoARIMA` podía elegir un orden distinto en cada ventana en los análisis anteriores.
- `neuralforecast` por default NO lo hace (`refit=False`: entrena una vez, predice todas las ventanas con esos mismos pesos). Activando `refit=True` con `use_init_models=False` (default), cada reentreno **parte de los pesos de la ventana anterior** (warm start) en vez de reiniciar — es, literalmente, la red aprendiendo a medida que llegan datos nuevos.

10 ventanas de backtesting por combinación (3 escalas × 3 horizontes = 9 combos en total).

![Heatmap RMSE online learning](datos/resultados/online_learning_heatmap.png)

**Lectura por escala** (rojo = peor de esa columna, verde = mejor — normalizado por columna, no comparable entre columnas):

- **Diaria (h=1,2,3)**: gana el naive en los tres horizontes, con un margen chico en h=3 para N-HiTS (-1%, prácticamente empate). A horizontes de 1-3 días el ruido/autocorrelación domina tan fuerte que ni el online learning le encuentra la vuelta — coherente con todo lo que ya vimos en la sección diaria original.
- **Semanal (h=1,2,3)**: acá el online learning se luce — **N-HiTS gana los tres horizontes**, y en h=1/h=2 por márgenes grandes (-51% y -52% vs. el mejor baseline). Es la escala donde reentrenar con cada dato nuevo aporta más valor real.
- **Mensual (h=1,2,3)**: resultado mixto y el más interesante de investigar. En h=1 y h=2, N-HiTS/N-BEATS mejoran al baseline (N-BEATS en h=2 llega a RMSE=3.25 contra 29.85 del naive, un salto enorme). Pero en **h=3 ambas redes fallan catastróficamente** (RMSE ~95-110 contra ~41 del naive) — y no es un problema de falta de entrenamiento: subir el presupuesto de 100 a 300 pasos por reentreno mejoró mucho h=1/h=2 (sobre todo h=2) pero **no cambió nada en h=3** (verificado corriendo ambos presupuestos, no asumido). Queda como una inestabilidad real y no resuelta de combinar warm-start + horizonte de 3 meses en una serie tan corta (200 obs.) — se documenta como hallazgo, no se esconde ni se fuerza un ajuste para que desaparezca.

**Por qué esto importa para el proyecto en general**: el mensaje no es "el online learning es mejor" ni "es peor" — es que **ayuda mucho en algunos combos (semanal, sobre todo) y puede volverse inestable en otros (mensual h=3) de forma que más cómputo no arregla**. Eso es más honesto que optimizar solo la configuración que se ve mejor y mostrar únicamente esa.

## Extensión: estrategia de trading con Reinforcement Learning

El forecasting de arriba responde si un modelo predice bien el precio. Una extensión separada (dentro del mismo repo, [Issue #1](https://github.com/bastianbm7/usdclp-nbeats-arima-forecasting/issues/1)) responde una pregunta distinta: **¿ese forecast sirve para tomar decisiones de trading que ganen plata con apalancamiento y riesgo real?** Se probó un agente PPO (`gymnasium` + `stable-baselines3`, con [FinRL](https://github.com/AI4Finance-Foundation/FinRL) como ancla metodológica) con gestión de riesgo real (capital de $100, tamaño de posición vía risk sizing, stop-loss con trailing basado en volatilidad GARCH, take-profit en el forecast de N-HiTS).

**Hallazgo, walk-forward de 100 semanas out-of-sample**: ninguna estrategia activa le ganó a mantener la posición sin apalancar (buy-and-hold: -0.9%; umbral simple: -16.0%; PPO: -52.2% con recompensa simplificada). Al reentrenar al agente con la **economía real** (apalancamiento + TP/SL) como recompensa, en vez de una versión simplificada, aprendió a **no operar nunca** — una respuesta racional dado que ninguna de las 7 variables del estado supera |r|=0.11 de correlación con el retorno real siguiente.

**Issue #2 — ¿se puede destrabar al agente?**: se probaron 3 cambios independientes (más exploración vía `ent_coef`, acción continua en vez de discreta, recompensa como exceso sobre buy-and-hold). Más exploración y reward shaping **no cambiaron nada** — el agente converge a la política de no operar de forma idéntica en las tres configuraciones. Acción continua sí lo obliga a operar, pero pierde -16.8% — peor que no operar. El cuello de botella confirmado es la señal, no el agente.

**Radar-baseline — ¿es la política de "no operar" un artefacto del entrenamiento, y hay una variable/técnica barata que ayude?**: se validó con Kelly criterion (sin entrenar ningún agente) si existe edge explotable — la respuesta es no: la fracción "óptima" de Kelly apalanca la deriva histórica de USD/CLP y pierde -51.5% a -52.9% fuera de muestra, peor que no operar. Se probaron además 5 variables nuevas ancladas en papers concretos (diferencial de tasas Chile-EE.UU., retorno/momentum del cobre, momentum clásico de USD/CLP) a frecuencia semanal — ninguna supera el umbral |r|=0.11 de la sección 9.5 — y 2 cambios al agente (momentum en el estado, reward Differential Sharpe Ratio). Momentum sí rompe el atractor de "no operar" (2 operaciones en 100 semanas, -3.1%) pero sigue sin ser rentable; DSR no lo destraba.

> **⚠️ Errata (2026-09-23) — la línea diaria estaba invalidada.** Versiones anteriores de este README reportaban que el retorno diario del cobre "predecía" el de USD/CLP del día siguiente (r=-0.256) y que estrategias construidas encima (reglas simples, Kelly, PPO diario, AUD/NZD, NOK/ZAR/BRL con otros commodities, Momentum Transformer) alcanzaban Sharpe de 4-5 fuera de muestra. **Era un artefacto de timestamps**: Yahoo etiqueta la barra diaria FX con fecha D pero su precio es el de ~00:00 UTC de D (≈20:00 de Nueva York del día D-1), mientras el cierre de un futuro de commodity con fecha D es su settlement de ~13:00-14:30 ET de D — el "retorno del día siguiente" de la moneda cubría 17 de las 24 horas del retorno del commodity, es decir, era la correlación **contemporánea**. Además los backtests entraban a un precio anterior a la señal y cobraban el costo de transacción una fracción de las veces (una sola vez al cambiar de posición, en vez de ida+vuelta en cada operación, con apalancamientos de 3.8-9.6x). Dos auditorías lo encontraron; la evidencia (barras horarias, dólar observado del BCCh y FRED H.10 como fuente independiente: correlación contemporánea 0.23-0.46, operable ≈ 0) y la corrección están en la [sección 9.35 del paper](reportes/paper.md). Todo se rehízo con un módulo central de alineación temporal (`codigos/alineacion_temporal.py`), costos ida+vuelta por instrumento con sensibilidad (`codigos/costos_y_estadistica.py`), selección de parámetros solo con datos pasados, varias semillas e intervalos de confianza (scripts `58`-`67`).

**Resultado corregido, dicho sin atenuar: ninguna estrategia diaria del proyecto tiene un Sharpe neto de costos distinguible de cero o positivo.** Lo que sí es real es la correlación contemporánea commodity-moneda (ordenada como predice Chen & Rogoff 2003: AUD/NZD/CAD más fuerte que CLP, JPY casi nula; NOK con petróleo, ZAR con platino, BRL con soja) — pero para cuando se conoce el settlement del commodity, la moneda ya se movió.

| Resultado (sección del paper) | Original (inválido) | Corregido (neto de costos) |
|---|---|---|
| Correlación cobre → USD/CLP "día siguiente" (9.13) | -0.256 | operable -0.078 (el -0.254 es contemporáneo) |
| Umbral cobre / Kelly, USD/CLP, 300 días (9.14) | +83.5% / Sharpe 4.44; +86.2% / 4.57 | -28.2% / -2.25; -30.1% / -2.43 (bruto 0.7-0.8, IC95 incluye 0) |
| Validación histórica 4 períodos (9.17 paso 1) | Sharpe 2.07-5.42 | regla del cobre negativa en los 4 |
| PPO diario USD/CLP (9.17) | +397.1% / 3.88 | aprende a casi no operar (8-24 operaciones), Sharpe -0.9 a -2.1 (3 semillas) |
| Holding de N días, TP adaptativo, grilla N×h (9.20-9.26) | Sharpe 1-4, "h2 es malo", N=7/h4 2.73 | ningún resultado distinguible de 0; "h2 es malo" no se replica |
| AUD / NZD / CAD (9.28) | +692% / 4.92; +398% / 3.84; +33% / 0.86 | +47.9% / 1.11 [-0.78, 2.90]; -0.58; 0.07 — PPO negativo en las 3 |
| Screening 90 pruebas + SPA (9.31) | 10 pares "predictivos", SPA p=0.0 | los 10 son contemporáneos; SPA sobre 90 estrategias operables p=0.54 (bruto) |
| Regímenes 2014-16 / 2020 / 2022 (9.32) | 9/10 pares Sharpe > 0 | con selección previa: 0, 0 y 2 pares; ninguno funciona |
| TFT panel (9.33) | -1.65 (forecast desalineado un día) | -3.67 (bruto -0.65); control solo-CLP -3.49 |
| Momentum Transformer (9.34) | Sharpe 5.29 ("3× → 5.42") | bruto 0.81 (5 semillas), neto -0.55 a -1.91 |

La conclusión de toda la extensión de RL vuelve a ser la que ya daba la línea semanal: **con estas señales y a estas frecuencias no hay edge explotable, y un agente que siente el costo real aprende a no operar.**

**[Issue #4](https://github.com/bastianbm7/usdclp-nbeats-arima-forecasting/issues/4) cierra la línea de indicadores técnicos semanales**: SMA/EMA a 5/10/20/50 semanas, Bollinger Bands, CCI y ADX (12 candidatas en total, ninguna probada antes) tampoco superan |r|=0.11 — la más fuerte (`ema_50_rel`, -0.076) queda por debajo de 4 de las 7 features originales. Con esto se agotan las variantes razonables de indicadores derivados del precio semanal; no se entrenó ningún agente nuevo porque el paso condicional de la Tarea (superar el umbral) no aplicó.

**[Issue #5](https://github.com/bastianbm7/usdclp-nbeats-arima-forecasting/issues/5) reconstruyó el pipeline completo a frecuencia DIARIA** (dataset walk-forward con NHITS refiteado cada 5 días, entorno Gym diario `27`, risk sizing de 3% del capital, PPO walk-forward). La infraestructura sigue siendo válida; los resultados originales (+397% / Sharpe 3.88) no. Corregido: el PPO entrenado con costos reales converge a casi no operar y pierde; la regla del cobre pierde dos tercios del capital con spread de 0.15%. Un hallazgo nuevo de la corrección: el TP/SL evaluado con precios de cierre está sesgado a favor (con dirección al azar da Sharpe bruto medio de ~0 a +1.3 según la configuración), por lo que cada tabla corregida trae un control al azar. 📄 **[Secciones 9.17 y 9.35 del paper](reportes/paper.md)**.

**[Issue #6](https://github.com/bastianbm7/usdclp-nbeats-arima-forecasting/issues/6) responde si el problema es específico de USD/CLP o genérico**: se entrenó una política PPO **multi-activo** (USD/CLP + USD/MXN + USD/BRL + USD/COP a la vez, mismo motor de riesgo real, estado condicionado por un one-hot del par activo para evitar el pooling ciego que perjudicó a CLP en la sección 9.14) y se comparó contra el agente solo-CLP en el mismo walk-forward de 5 ventanas × 20 semanas. **El resultado es idéntico, hasta el centavo**: el agente multi-activo converge exactamente a la misma política de "no operar nunca" ($100.00, 0 operaciones) que el agente solo-CLP. Un diagnóstico adicional (la misma política evaluada también en el tramo out-of-sample propio de MXN/BRL/COP, no solo CLP) confirma que el atractor de inacción **no es específico de CLP**: 0 operaciones en las 20 combinaciones posibles (4 pares × 5 ventanas). No hay señal real que extraer de ninguna de las 4 series a esta frecuencia. 📄 **[Ver el detalle completo, con gráficos, en las secciones 9.11-9.18 del paper](reportes/paper.md#911-radar-baseline-validar-la-propuesta-1-de-98-antes-de-construir-nada-septiembre-2026)**.

**Issues [#9](https://github.com/bastianbm7/usdclp-nbeats-arima-forecasting/issues/9), [#10](https://github.com/bastianbm7/usdclp-nbeats-arima-forecasting/issues/10), [#11](https://github.com/bastianbm7/usdclp-nbeats-arima-forecasting/issues/11) y el TP adaptativo** (holding de N días, take-profit por horizonte de NHITS, grilla N×h): rehechos con datos alineados, costos y semillas (la grilla, en un subconjunto de 12 de 35 combinaciones por cómputo). Ninguna configuración tiene Sharpe neto distinguible de cero; los patrones reportados ("h2 es el peor horizonte", "zonas buenas" en N=5-7 y N=14) no se replican. 📄 **[Secciones 9.20-9.26 del paper](reportes/paper.md)**.

**[Issue #12](https://github.com/bastianbm7/usdclp-nbeats-arima-forecasting/issues/12) (AUD, CAD, NZD)** y **[Issue #13](https://github.com/bastianbm7/usdclp-nbeats-arima-forecasting/issues/13) (WTI, oro, platino, soja, hierro × NOK/ZAR/BRL; TFT; Momentum Transformer)**: la "generalización" a otras monedas y commodities era la de la correlación contemporánea. Corregido: correlación operable ≈ 0 en todas; el SPA de Hansen sobre el universo de 90 estrategias operables no rechaza que ninguna le gane a no operar ni siquiera sin costos; con selección solo con datos previos a cada régimen no se elige casi nada y lo elegido no funciona; TFT (con el bug de alineación del forecast corregido y el control solo-CLP agregado) y Momentum Transformer no tienen señal neta de costos. La Fase 0 de liquidez (incluido el hallazgo del WTI negativo genuino del 2020-04-20) no depende de la errata y sigue válida. 📄 **[Secciones 9.28-9.35 del paper](reportes/paper.md)**.

**[Issue #14](https://github.com/bastianbm7/usdclp-nbeats-arima-forecasting/issues/14) — variables nuevas a frecuencia semanal/mensual, después de la errata.** Las variables del propio precio de USD/CLP no dependían del error de timestamp. El cobre semanal sí: re-verificado, pasa de -0.10 a +0.04. Se probaron 18 variables nuevas (VIX, S&P 500, emergentes, acciones chilenas, petróleo, dólar global, bono a 2 años, carry), cada una con la hora real de publicación de su dato, reservando 2025-2026 como hold-out intocable y registrando todas las pruebas. **Ninguna sobrevive la corrección por pruebas múltiples**, y el hold-out no se abrió. La única pista es el nivel del VIX: después de semanas de estrés, las monedas de riesgo tienden a recuperarse (USD/CLP semanal, Sharpe neto 0.66 [0.11, 1.20] con carry y costos). Con una regla fija en otras 11 monedas, la dirección se repite en 8 de 9 monedas de riesgo y es ≈0 en los refugios, pero ningún IC95 excluye el cero. Es una pista débil de cartera, no un bot de CLP. 📄 **[Sección 9.36 del paper](reportes/paper.md)**.

**[Issue #15](https://github.com/bastianbm7/usdclp-nbeats-arima-forecasting/issues/15) — cartera mensual de primas de riesgo FX (carry + momentum) en 14 monedas, incluida CLP.** Cambia la pregunta: en vez de predecir USD/CLP mañana, cosechar primas con respaldo académico a rebalanceo mensual. Spot de FRED H.10 (hora conocida) y CLP de Yahoo limpio; tasas OECD con 2 meses de rezago de publicación; costo = spread ida y vuelta sobre el turnover. Con selección solo con 2000-2024 (21 variantes registradas, Sharpe máximo esperado por azar 0.38): carry neto 0.62 [0.17, 1.10], momentum de 6 meses 0.44 [0.06, 0.84], **combinación 50/50 0.78 [0.36, 1.22]** (bruto 0.85, drawdown máximo -14% con 5% de vol objetivo). Es el primer resultado del proyecto que supera el máximo esperado por azar con un IC95 que excluye el cero, y cae en el rango académico. Hay tres salvedades. El premio se concentra en 2000-2008 (1.59 → 0.47 → 0.15 por subperíodo). Brasil aporta ~40% del retorno bruto y su carry está sobreestimado; sin BRL, 0.52. CLP casi no pesa. **Hold-out (apertura única, 20 meses)**: combinación 0.47 neto [-1.09, 2.74]. Es positivo, pero con 20 meses no discrimina. 📄 **[Sección 9.37 del paper](reportes/paper.md)**.

## Datos

Tipo de cambio USD/CLP, serie diaria descargada con [`yfinance`](https://github.com/ranaroussi/yfinance) (ticker `CLP=X`, fuente: Yahoo Finance). Se guarda una copia cruda en `datos/bases/` para reproducibilidad exacta (no depender de que Yahoo siga sirviendo el mismo histórico).

## Estructura

```
codigos/
├── 01_obtener_datos.py          # descarga USD/CLP vía yfinance, split train/test
├── 02_baseline_arima_naive.py   # statsforecast: AutoARIMA + Naive, backtesting walk-forward (diario)
├── 03_modelo_nbeats_nhits.py    # neuralforecast: NBEATS + NHITS, mismo esquema de backtesting (diario)
├── 04_metricas_comparacion.py   # RMSE/MAE/MAPE por modelo, tabla comparativa (diario)
├── 05_visualizacion.py          # predicción vs. real + intervalos + baseline superpuesto (diario)
├── 06_analisis_mensual.py       # mismo pipeline completo (datos+baseline+NN+métricas+gráficos), resampleado a mensual
├── 07_analisis_anual.py         # mismo pipeline sin NN (no hay data suficiente), resampleado a anual
├── 08_online_learning.py        # refit=True (warm start) x 3 escalas x horizontes 1/2/3 pasos
├── 09_comparacion_modelos_volatilidad.py  # Naive/media movil/GARCH/EGARCH walk-forward, elige el modelo de volatilidad
├── 10_generar_dataset_rl.py     # dataset semanal (forecast N-HiTS + volatilidad GARCH + MACD/RSI/min-max) para el agente
├── 11_entorno_trading_rl.py     # entorno Gym (estado/accion/recompensa con gestion de riesgo real, TP/SL con trailing stop)
├── 12_entrenar_agente_rl.py     # chequeo rapido: un solo split train/test (superado por 14)
├── 13_backtest_estrategia_rl.py # backtest del chequeo rapido de 12 (superado por 14)
├── 14_backtest_walkforward_gestion_riesgo.py  # walk-forward real (5 ventanas), PPO vs buy-and-hold vs umbral simple
├── 15_analisis_features.py      # cuanto se correlaciona cada variable del estado con el retorno futuro real
├── 16_graficos_resultados_rl.py # graficos finales de la extension de RL (lee los CSV ya generados, no recalcula)
├── 17_agente_rl_mejoras.py      # Issue #2: ent_coef, accion continua y reward shaping vs. buy-and-hold, solos y combinados
├── 18_kelly_validacion.py       # radar-baseline Tier 0: Kelly criterion (sin RL) como segundo veredicto sobre si hay edge
├── 19_features_nuevas_validacion.py  # radar-baseline Tier 2: correlacion + Kelly con tasas/cobre/momentum nuevos
├── 20_agente_rl_ronda2.py       # radar-baseline Tier 1: PPO + momentum en el estado / reward Differential Sharpe Ratio
├── 21_features_nuevas_frecuencias.py  # repite el chequeo de correlacion de cobre/tasas a frecuencia diaria y mensual
├── 22_kelly_diario_cobre.py     # valida timestamp (lag-scan) y backtest de rentabilidad del cobre a frecuencia diaria
├── 23_dataset_multi_par_diario.py     # panel de 13 pares FX diarios + generalizacion del efecto cobre + pooled vs. solo CLP
├── 24_features_sma_ema_validacion.py  # Issue #4: correlacion de SMA/EMA/Bollinger/CCI/ADX con el retorno futuro semanal
├── 25_validacion_historica_cobre_diario.py  # Issue #5 paso 1: backtest de 22 en ventanas historicas mas viejas del cobre (2015-2016, 2017-2018, 2020-2021)
├── 26_generar_dataset_rl_diario.py    # Issue #5 paso 2: dataset walk-forward DIARIO (NHITS con refit cada 5 dias, GARCH, MACD/RSI/minmax, copper)
├── 27_entorno_trading_rl_diario.py    # Issue #5: entorno Gym diario (archivo nuevo, no modifica el semanal) - TP/SL simplificado a cierre-a-cierre
├── 28_backtest_walkforward_diario.py  # Issue #5: backtest final diario - PPO vs buy-and-hold vs umbral simple vs umbral cobre
├── 29_dataset_volatilidad_multipar.py    # Issue #6: dataset RL semanal + comparacion de modelos de volatilidad para USD/MXN, USD/BRL, USD/COP
├── 30_entorno_trading_rl_multiactivo.py  # Issue #6: entorno Gym multi-activo (composicion sobre 11, one-hot del par activo) - no modifica 11
├── 31_backtest_walkforward_multiactivo.py  # Issue #6: walk-forward del agente multi-activo vs. solo-CLP + diagnostico por par
├── 32_entorno_trading_rl_diario_multidia.py  # Issue #9: entorno Gym con holding fijo de N dias (trailing stop del agente semanal + fix de TP del agente diario) - Issue #10 le agrega horizonte_tp configurable
├── 33_backtest_walkforward_diario_multidia.py  # Issue #9: walk-forward del holding de N dias (N=1,2,3,5) vs. umbral cobre + coincidencia de direccion
├── 34_features_semanales_validacion.py    # chequeo (descartado): precios de la semana calendario anterior como feature - correlacion decae a la mitad 2010-2018 vs 2018-2026
├── 35_generar_dataset_rl_diario_h3.py     # dataset con nhits_h3 (H=8) para el experimento de TP adaptativo
├── 36_entorno_trading_rl_diario_tp_adaptativo.py  # TP adaptativo: elige h1/h2/h3 segun consistencia del forecast, holding=3 fijo
├── 37_backtest_walkforward_diario_tp_adaptativo.py  # walk-forward de TP adaptativo vs. TP fijo en h1 vs. umbral cobre
├── 38_graficos_entrada_salida_multidia.py  # graficos de entrada/salida por operacion, reconstruye fecha exacta de salida sin reentrenar
├── 39_generar_dataset_rl_diario_h5.py     # Issue #10: dataset con nhits_h1..h5 (H=10) para el barrido de TP por horizonte
├── 40_backtest_walkforward_diario_grilla_nh.py  # Issue #10: grilla TP fijo por horizonte (h1-h5) x holding (N=3,5,7), 15 combinaciones
├── 41_graficos_grilla_nh_h4.py            # graficos de retorno en el tiempo y entrada/salida, h4 fijo comparando N
├── 42_backtest_walkforward_diario_grilla_nh_largo.py  # Issue #11: extiende la grilla a N=10,12,14,20 (mismo dataset/entorno del Issue #10)
├── 43_graficos_resumen_grilla_nh_extendida.py  # Issue #11: Sharpe/drawdown/%stop-loss promedio por N, grilla completa (N=3 a 20)
├── 44_heatmaps_grilla_nh.py      # Issue #11: heatmaps N x h de la grilla completa (retorno, saldo, drawdown, razon de cierre)
├── 45_generar_dataset_rl_diario_aud.py  # Issue #12: dataset walk-forward diario para AUD/USD (mismo patron que 26, fuente: panel de 23)
├── 46_generar_dataset_rl_diario_cad.py  # Issue #12: idem para USD/CAD
├── 47_generar_dataset_rl_diario_nzd.py  # Issue #12: idem para NZD/USD
├── 48_backtest_walkforward_diario_aud.py  # Issue #12: backtest final AUD (mismo esquema que 28) - PPO vs buy-and-hold vs umbral simple vs umbral cobre
├── 49_backtest_walkforward_diario_cad.py  # Issue #12: idem para USD/CAD
├── 50_backtest_walkforward_diario_nzd.py  # Issue #12: idem para NZD/USD
├── 51_verificar_liquidez_commodities_nok.py  # Issue #13 Fase 0: liquidez de WTI/oro/platino/soja/hierro/NOK vs. umbral de litio, re-verifica ticks de BRL/ZAR
├── 52_screening_correlacion_commodities_fx.py  # Issue #13 Fase 1: NOK/ZAR/BRL x 5 commodities, escaneo de rezagos -2..+3, correccion FDR sobre 90 pruebas
├── 53_bootstrap_spa_supervivientes.py  # Issue #13 Fase 1: refuerza los 10 supervivientes FDR (rezago +1) con bootstrap SPA de Hansen
├── 54_walkforward_multiperiodo_supervivientes.py  # Issue #13 Fase 2: valida los 10 pares en 3 regimenes historicos (2014-2016, 2020, 2022), Sharpe por ventana
├── 55_panel_extendido_con_nok.py  # Issue #13 Fase 3: agrega NOK al panel de 13 pares de 9.14 (panel_fx_diario_extendido.csv, 14 monedas)
├── 56_tft_panel_walkforward.py  # Issue #13 Fase 3: TFT sobre el panel de 14 monedas (ID como covariable estatica) vs. baselines de 9.14 - no supera el criterio de exito
├── 57_momentum_transformer_port.py  # Issue #13 Fase 4: port en PyTorch de VSN+atencion+SharpeLoss (kieranjwood/trading-momentum-transformer) - arquitectura reusada por 67; sus RESULTADOS (Sharpe 5.29) quedaron invalidados
│
│   # --- CORRECCION (2026-09-23): artefacto de timestamp + costos + bugs (ver seccion 9.35 del paper) ---
├── alineacion_temporal.py        # modulo central: timestamps reales (barra Yahoo FX D = ~20:00 NY de D-1; settlement de commodity D = ~13-14:30 ET de D), merge estricto "settlement < precio FX", marco senal->operacion ejecutable
├── costos_y_estadistica.py       # spreads ida+vuelta por instrumento (supuestos documentados), Sharpe con IC95 bootstrap de bloques, Sharpe maximo esperado bajo H0 (pruebas multiples)
├── validacion_timestamp/         # scripts originales de la auditoria (t1-t4, fetch) con rutas adaptadas - evidencia de la errata
├── 58_validacion_timestamp.py    # errata, evidencia: hora real de la barra diaria, rezagos CLP en relojes consistentes (horario, dolar observado BCCh), validacion cruzada FRED H.10 vs Yahoo
├── 59_senal_cobre_clp_corregida.py  # rehace 9.13/9.14 (CLP) y 9.17 paso 1: rezagos, Kelly/umbral cobre walk-forward con senal previa a la entrada, signo de train, costos, IC95, validacion historica
├── 60_panel_fx_corregido.py      # rehace el panel de 9.14: panel_fx_diario_alineado.csv (13 pares + NOK), correlacion contemporanea vs operable, backtest por par, pooled vs solo-CLP
├── 61_screening_spa_universo_corregido.py  # rehace 9.31: screening de 90 pruebas alineado + SPA de Hansen sobre el universo de 90 estrategias operables (bruto y neto)
├── 62_regimenes_seleccion_previa_corregido.py  # rehace 9.32: seleccion de pares/signo solo con datos previos a cada regimen
├── 63_realinear_datasets_rl_diarios.py  # realinea (no regenera) los datasets NHITS de 26/35/39/45-47: cobre con merge estricto, y_next desde la serie cruda, sin precios repetidos
├── 64_entrenar_ppo_diario_corregido.py  # reentrena los PPO diarios (9.17, 9.20, 9.23, 9.24/9.26 subconjunto, 9.28) con datos alineados, costos corregidos y 3 semillas - lotes A/B/C/D
├── 65_resumen_rl_diario_corregido.py  # metricas de 64 + baselines (umbral cobre con signo de train, umbral simple con mediana de train, buy-and-hold), IC95, sensibilidad a spread
├── 66_tft_panel_corregido.py     # rehace 9.33: TFT con el forecast realineado (bug de 56) sobre el panel alineado + control solo-CLP en las 5 ventanas
├── 67_momentum_transformer_corregido.py  # rehace 9.34: mismo port de 57 sobre el panel alineado, costos, 5 semillas, sin el argumento de escalado 3x
│
│   # --- DESPUES DE LA ERRATA: protocolo comun + variables nuevas (Issues #14, #17) ---
├── protocolo_evaluacion.py       # hold-out intocable desde 2025-01-01 + registro de todas las pruebas (datos/resultados/registro_pruebas.csv)
├── 68_features_semanales_mensuales_corregido.py  # re-verifica cobre/tasas semanales y mensuales (19/21) con la alineacion estricta
├── 69_variables_externas_clp.py  # descarga VIX, S&P 500, EEM, ECH, WTI, DXY amplio, bono 2 anos, tasas 3m, con la hora real de publicacion de cada dato
├── 70_screening_backtest_variables_clp.py  # screening FDR (24 pruebas, sin hold-out) + backtest walk-forward con carry y costos de cada variable
├── 71_confirmacion_vix_otras_monedas.py  # confirma la pista del VIX con una regla fija en 11 monedas que no formularon la hipotesis
│
│   # --- Issue #15: cartera mensual de primas de riesgo FX (carry + momentum), 14 monedas incl. CLP ---
├── 80_datos_cartera_fx_mensual.py  # spot FRED H.10 (mediodia NY) de 13 monedas + CLP=X Yahoo limpio, tasas 3m OECD con 2 meses de rezago -> datos/bases/cartera_fx_mensual.csv
├── cartera_fx.py                 # motor: retorno en exceso con carry (CIP), carry/TSMOM/combinacion, vol targeting con datos pasados, costo sobre turnover
├── 81_cartera_fx_seleccion_pre_holdout.py  # 21 variantes evaluadas solo con retornos < 2025-01-01, registro de pruebas, eleccion de la configuracion
├── 82_cartera_fx_holdout.py      # apertura UNICA del hold-out (2025-01..2026-08) para la configuracion elegida por 81
├── 83_cartera_fx_stop_take_profit.py  # extension: stop-loss/take-profit diario dentro del mes (33 variantes, sin hold-out) - ninguna mejora el Sharpe
├── 84_cartera_fx_stops_por_posicion_trailing.py  # stop por moneda, trailing por moneda (arrastra el maximo entre meses) y trailing de cartera (27 variantes)
├── 85_cartera_fx_stops_por_grupo.py  # momentum 3m vs 6m; stops solo en la pata larga/corta del carry o en la mitad mas/menos volatil del momentum (25 variantes)
├── 86_cartera_fx_estrategia_patas_largas_baja_vol.py  # estrategia: carry con trailing en la pata larga + momentum solo en monedas de baja vol (evaluada en el mismo periodo que la origino)
└── 87_cartera_fx_holdout_estrategia_a.py  # apertura UNICA del hold-out para la estrategia (a) de 86 + resultados desde 2015 y por anio

datos/
├── bases/        # CSV crudo de USD/CLP
└── resultados/   # gráficos finales + tabla de métricas

reportes/
└── paper.md      # informe de investigación completo (resumen, metodología, resultados, discusión)
```

Nota sobre la corrección del 2026-09-23: los scripts `21` (parte diaria), `22`, `23`, `25`, `26`, `28`, `33`, `35`, `37`, `39`, `40`, `42`, `45`-`50` y `52`-`57` llevan un encabezado `INVALIDADO / SUPERADO` que indica qué script corregido los reemplaza; se conservan sin cambios de lógica como registro de los números originales. Los entornos `27`, `32` y `36` se corrigieron en el lugar (bloques `CORRECCIÓN (2026-09-23)`: spread ida+vuelta en cada operación) porque son librerías que importan los scripts corregidos. El lote de `64` cubre holding de N días, TP adaptativo y el subconjunto de la grilla, por eso no hay un script corregido por cada backtest original.

Nota de numeración: los scripts `45`-`50` (Issue #12, AUD/CAD/NZD diario) y `51`-`57` (Issue #13, commodities nuevos) se desarrollaron en ramas paralelas y no chocaron al fusionar porque Issue #13 arrancó su numeración en `51` a propósito, después de confirmar con `ls codigos/` que `50` ya estaba tomado.

## Limitaciones conocidas

- **Autocorrelación, no "poder predictivo"**: con precio de cierre diario, buena parte de cualquier error bajo es autocorrelación, no señal real capturada por el modelo. Mitigado con baseline siempre presente, backtesting walk-forward y horizonte de varios pasos — pero no eliminado del todo.
- **Nixtla/neuralforecast es una dependencia, no un fork**: se usa la implementación de la librería de N-BEATS/N-HiTS (Apache-2.0), no el código original de los autores (Oreshkin et al. / Challu et al.) — la fidelidad al paper es metodológica, no literal.
- **Nivel 1 de proporcionalidad**: sin tests automatizados ni monitoreo — es una pieza de portfolio personal, no un servicio en producción.
- **Dependencia de Yahoo Finance**: `yfinance` puede cambiar de comportamiento — mitigado guardando una copia cruda del CSV descargado.
- **Backtesting anual con n=16**: 3 ventanas de evaluación sobre 16 observaciones totales es una muestra chica — el resultado (AutoARIMA/RWD >> Naive) es consistente y tiene explicación matemática clara, pero no hay que leerlo con la misma confianza estadística que el diario (4.346 obs.) o el mensual (200 obs.).
- **Online learning mensual a h=3 es inestable**: el `refit=True` con warm start rompe en esa combinación específica (RMSE 2-3x peor que el naive) y subir el presupuesto de entrenamiento no lo arregla — no identificado el mecanismo exacto (hipótesis: optimizer sin estado propio en cada reentreno + horizonte largo relativo a una serie corta), documentado como falla abierta en vez de ocultarlo o forzar un ajuste que la haga desaparecer.

## Paper(s) ancla

- Oreshkin et al., *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting*, ICLR 2020 — [arXiv:1905.10437](https://arxiv.org/abs/1905.10437)
- Challu et al., *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting* — [arXiv:2201.12886](https://arxiv.org/abs/2201.12886)
- Liu et al., *FinRL: Deep Reinforcement Learning Framework for Automated Trading*, ancla de la extensión de RL (sección 9 del paper) — [arXiv:2111.09395](https://arxiv.org/abs/2111.09395)
- Moskowitz, Ooi & Pedersen, *Time Series Momentum*, ancla de la feature de momentum (sección 9.11/9.12) — [SSRN:2089463](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463)
- Chen & Rogoff, *Commodity Currencies*, ancla de las features de cobre (sección 9.11) — *Journal of International Economics*, 2003
- Moody & Saffell, *Reinforcement Learning for Trading*, ancla del reward Differential Sharpe Ratio (sección 9.12) — NeurIPS 1998
