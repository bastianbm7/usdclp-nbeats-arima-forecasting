# INVALIDADO / SUPERADO (2026-09-23) - el holding de N dias: ver script 64 (lote B) + 65.
# Motivo: artefacto de timestamp (la barra diaria Yahoo FX con fecha D es el precio
# de ~20:00 NY de D-1, el cierre de un futuro de commodity con fecha D es su
# settlement de ~13:00-14:30 ET de D; el merge_asof(direction='backward') por
# fecha usaba informacion posterior al precio de entrada) + costos cobrados una
# sola vez al cambiar de posicion en vez de ida+vuelta en cada operacion. Ver
# alineacion_temporal.py, costos_y_estadistica.py y la errata en 9.35 del paper.
# Se conserva sin cambios de logica como registro de los numeros originales; no
# reproduce exactamente sus CSV si se vuelve a correr despues de la correccion de
# costos en 27/32/36.
#
# Issue #9 (Propuesta A): backtest walk-forward del agente PPO diario con
# holding fijo de N dias (dias_holding en VALORES_DIAS_HOLDING), comparado
# contra si mismo a N=1 (el resultado ya conocido de 27/28, Issue #5:
# +397.1%, Sharpe 3.88) y contra "Umbral cobre" evaluado con la MISMA
# economia multi-dia en cada N (para que la comparacion PPO-vs-cobre sea
# apples-to-apples a cada N, no solo el PPO cambiando y el cobre quieto).
#
# Pregunta que responde este script (la pregunta central del Issue #9): al
# alargar el holding, ¿el PPO reduce el 28% de dias donde diverge de la
# señal del cobre (y por lo tanto se acerca a superarla), o el efecto va en
# la otra direccion?
#
# OJO ANUALIZACION DEL SHARPE: con dias_holding>1 cada "operacion" cubre
# dias_holding dias de calendario, no 1 - anualizar con sqrt(252) como en
# 28 (pensado para N=1) sobreestimaria el Sharpe para N>1 (menos
# observaciones por año, pero tratadas como si fueran diarias). Se anualiza
# con sqrt(252/dias_holding) - periodos de trading por año, no dias.
# Verificado que con dias_holding=1 esto se reduce exactamente a sqrt(252).

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

multidia_mod = importlib.import_module("32_entorno_trading_rl_diario_multidia")
entorno_diario_mod = importlib.import_module("27_entorno_trading_rl_diario")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")  # reusa ventanas_walkforward()
kelly_diario_mod = importlib.import_module("22_kelly_diario_cobre")  # reusa posiciones_umbral_cobre()

RESULTADOS_DIR = "../datos/resultados"

CAPITAL_INICIAL = 100.0
RIESGO_MAX_PCT = multidia_mod.RIESGO_MAX_PCT
K_STOP_LOSS = multidia_mod.K_STOP_LOSS
SLIPPAGE_PCT = multidia_mod.SLIPPAGE_PCT

N_WINDOWS_WF = 5
N_TEST_POR_VENTANA = 60  # dias de CALENDARIO por ventana (mismo tamano que 28) - a mayor dias_holding, menos decisiones caben en los mismos 60 dias
TOTAL_TIMESTEPS_PPO = 100_000  # mismo presupuesto que 14/28 para cada N (episodios mas cortos con N alto, pero mismo total de pasos de entrenamiento)
SEED = 42
VALORES_DIAS_HOLDING = [1, 2, 3, 5]


def posiciones_ppo(df_train, df_test, dias_holding):
    env_train = multidia_mod.USDCLPTradingEnvDiarioMultidia(df=df_train, dias_holding=dias_holding)
    modelo = PPO("MlpPolicy", env_train, seed=SEED, verbose=0)
    modelo.learn(total_timesteps=TOTAL_TIMESTEPS_PPO)

    env_test = multidia_mod.USDCLPTradingEnvDiarioMultidia(df=df_test, dias_holding=dias_holding)
    obs, _ = env_test.reset()
    posiciones = []
    terminado = False
    while not terminado:
        accion, _ = modelo.predict(obs, deterministic=True)
        posiciones.append(multidia_mod.POSICION_POR_ACCION[int(accion)])
        obs, _, terminado, _, _ = env_test.step(accion)
    # env_test.df = las decisiones YA construidas y precomputadas (mismos
    # bloques de dias_holding que vio el agente) - se reusa tal cual para
    # simular Umbral cobre con la MISMA economia, en vez de reconstruirla.
    return posiciones, env_test.df


def simular_con_gestion_riesgo(df_decisiones, posiciones, nombre, capital_inicial):
    # df_decisiones ya trae precio_salida_*/razon_cierre_* precomputados por
    # 32_entorno_trading_rl_diario_multidia.py - mismo patron que
    # simular_con_gestion_riesgo() de 28, sin volver a precomputar.
    capital = capital_inicial
    posicion_previa = 0.0
    filas = []
    for i, fila in df_decisiones.reset_index(drop=True).iterrows():
        posicion = posiciones[i]
        entrada = fila["y"]
        if posicion == 0:
            notional, pnl, razon, precio_salida, stop_loss = 0.0, 0.0, "plano", entrada, np.nan
        else:
            direccion = "largo" if posicion > 0 else "corto"
            distancia_riesgo = K_STOP_LOSS * fila["vol_garch"]
            stop_loss = fila[f"stop_loss_{direccion}"]
            precio_salida, razon = fila[f"precio_salida_{direccion}"], fila[f"razon_cierre_{direccion}"]
            notional = (RIESGO_MAX_PCT * capital) / distancia_riesgo if distancia_riesgo > 0 else 0.0
            retorno_pct = posicion * (precio_salida - entrada) / entrada
            costo_slippage = SLIPPAGE_PCT * notional if posicion != posicion_previa else 0.0
            pnl = notional * retorno_pct - costo_slippage
        capital += pnl
        posicion_previa = posicion
        filas.append({
            "ds": fila["ds"], "estrategia": nombre, "precio_entrada": entrada, "precio_salida": precio_salida,
            "posicion": posicion, "notional": notional, "razon_cierre": razon, "pnl": pnl, "capital": capital,
        })
    return pd.DataFrame(filas)


def simular_buy_and_hold_diario(df_test_diario, capital_inicial):
    # Referencia N-agnostica: mismo periodo de calendario, mantenido dia a
    # dia (no bloques de dias_holding) - se calcula UNA vez por ventana y se
    # reusa igual para los 4 valores de dias_holding (el test set de cada
    # ventana, a nivel diario, es identico para todos - ventanas_walkforward
    # corta sobre el dataset diario completo antes de submuestrear).
    capital = capital_inicial
    filas = []
    for _, fila in df_test_diario.reset_index(drop=True).iterrows():
        retorno_dia = (fila["y_next"] - fila["y"]) / fila["y"]
        pnl = capital * retorno_dia
        capital += pnl
        filas.append({"ds": fila["ds"], "estrategia": "Buy-and-hold", "posicion": 1.0, "pnl": pnl, "capital": capital})
    return pd.DataFrame(filas)


def calcular_metricas(resultado, nombre, dias_holding):
    capital_previo = resultado["capital"].shift(1).fillna(CAPITAL_INICIAL)
    r = resultado["pnl"] / capital_previo
    capital = resultado["capital"]
    retorno_total_pct = 100 * (capital.iloc[-1] / CAPITAL_INICIAL - 1)
    periodos_por_anio = 252 / dias_holding  # ver nota de anualizacion al inicio del archivo
    sharpe = (r.mean() / r.std()) * np.sqrt(periodos_por_anio) if r.std() > 0 else np.nan
    drawdown = capital / capital.cummax() - 1
    operaciones = (resultado["posicion"] != 0).sum()
    win_rate = 100 * (resultado.loc[resultado["posicion"] != 0, "pnl"] > 0).mean() if operaciones > 0 else np.nan
    return {
        "dias_holding": dias_holding, "estrategia": nombre, "n_decisiones": len(resultado), "capital_final": capital.iloc[-1],
        "retorno_total_%": retorno_total_pct, "sharpe_anualizado": sharpe, "max_drawdown_%": 100 * drawdown.min(),
        "win_rate_%": win_rate, "operaciones": operaciones,
    }


def coincidencia_direccion(posiciones_ppo, posiciones_cobre):
    # % de dias de decision donde el PPO opero (posicion!=0) Y coincidio en
    # signo con la direccion que sugiere el umbral de cobre ese mismo dia de
    # entrada - misma definicion que el "72.3%" de referencia citado en el
    # Issue (calculado sobre el resultado de dias_holding=1 en 28/paper.md
    # 9.17), extendida aca a cualquier N para comparar si el patron cambia.
    pos_ppo, pos_cobre = np.asarray(posiciones_ppo), np.asarray(posiciones_cobre)
    opera = pos_ppo != 0
    if opera.sum() == 0:
        return np.nan, 0
    coincide = np.sign(pos_ppo[opera]) == np.sign(pos_cobre[opera])
    return 100 * coincide.mean(), int(opera.sum())


def graficar_comparacion(curvas_ppo, curva_bh, path_salida):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(curva_bh["ds"], curva_bh["capital"], label="Buy-and-hold", color="black", linewidth=1.4)
    colores = {1: "steelblue", 2: "mediumseagreen", 3: "darkorange", 5: "crimson"}
    for dias_holding, r in curvas_ppo.items():
        ax.plot(r["ds"], r["capital"], label=f"PPO (holding={dias_holding}d)", color=colores.get(dias_holding), linewidth=1.6)
    ax.axhline(CAPITAL_INICIAL, color="gray", linestyle=":", linewidth=1, label="Capital inicial ($100)")
    ax.set_ylabel("Capital ($)")
    ax.set_title(f"Agente RL diario: holding fijo de N dias vs. buy-and-hold - Issue #9 ({N_WINDOWS_WF}x{N_TEST_POR_VENTANA}d walk-forward)")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df_completo = entorno_diario_mod.cargar_dataset()
    print(f"Dataset diario: {len(df_completo)} dias. Walk-forward: {N_WINDOWS_WF} ventanas x {N_TEST_POR_VENTANA} dias de calendario cada una")
    print(f"dias_holding a evaluar: {VALORES_DIAS_HOLDING}\n")

    capital_bh = CAPITAL_INICIAL
    partes_bh = []
    filas_metricas = []
    curvas_ppo_concat = {}
    coincidencias = []

    for dias_holding in VALORES_DIAS_HOLDING:
        print(f"=== dias_holding={dias_holding} ===")
        capital_ppo, capital_cobre = CAPITAL_INICIAL, CAPITAL_INICIAL
        partes_ppo, partes_cobre = [], []

        for w, (df_train, df_test) in enumerate(wf_mod.ventanas_walkforward(df_completo, N_WINDOWS_WF, N_TEST_POR_VENTANA)):
            pos_ppo, df_decisiones_test = posiciones_ppo(df_train, df_test, dias_holding)
            pos_cobre = list(kelly_diario_mod.posiciones_umbral_cobre(df_decisiones_test))

            r_ppo = simular_con_gestion_riesgo(df_decisiones_test, pos_ppo, "PPO", capital_ppo)
            capital_ppo = r_ppo["capital"].iloc[-1]
            partes_ppo.append(r_ppo)

            r_cobre = simular_con_gestion_riesgo(df_decisiones_test, pos_cobre, "Umbral cobre", capital_cobre)
            capital_cobre = r_cobre["capital"].iloc[-1]
            partes_cobre.append(r_cobre)

            coincide_pct, n_opero = coincidencia_direccion(pos_ppo, pos_cobre)
            coincidencias.append({"dias_holding": dias_holding, "ventana": w + 1, "coincidencia_direccion_%": coincide_pct, "dias_ppo_opero": n_opero, "n_decisiones": len(df_decisiones_test)})

            if dias_holding == VALORES_DIAS_HOLDING[0]:
                r_bh = simular_buy_and_hold_diario(df_test, capital_bh)
                capital_bh = r_bh["capital"].iloc[-1]
                partes_bh.append(r_bh)

            print(f"  ventana {w+1}/{N_WINDOWS_WF}: {len(df_decisiones_test)} decisiones, PPO=${capital_ppo:.2f}, Umbral cobre=${capital_cobre:.2f}, coincidencia direccion={coincide_pct:.1f}% ({n_opero} dias PPO opero)")

        r_ppo_concat = pd.concat(partes_ppo, ignore_index=True)
        r_cobre_concat = pd.concat(partes_cobre, ignore_index=True)
        curvas_ppo_concat[dias_holding] = r_ppo_concat

        filas_metricas.append(calcular_metricas(r_ppo_concat, f"PPO (holding={dias_holding}d)", dias_holding))
        filas_metricas.append(calcular_metricas(r_cobre_concat, f"Umbral cobre (holding={dias_holding}d)", dias_holding))

        r_ppo_concat.to_csv(f"{RESULTADOS_DIR}/walkforward_diario_multidia_ppo_h{dias_holding}_operaciones.csv", index=False)
        r_cobre_concat.to_csv(f"{RESULTADOS_DIR}/walkforward_diario_multidia_cobre_h{dias_holding}_operaciones.csv", index=False)
        print()

    curva_bh_concat = pd.concat(partes_bh, ignore_index=True)
    filas_metricas.append(calcular_metricas(curva_bh_concat, "Buy-and-hold", 1))
    curva_bh_concat.to_csv(f"{RESULTADOS_DIR}/walkforward_diario_multidia_buyhold_operaciones.csv", index=False)

    tabla = pd.DataFrame(filas_metricas).sort_values(["dias_holding", "sharpe_anualizado"], ascending=[True, False]).reset_index(drop=True)
    tabla.to_csv(f"{RESULTADOS_DIR}/walkforward_diario_multidia_metricas.csv", index=False)

    tabla_coincidencia = pd.DataFrame(coincidencias)
    tabla_coincidencia.to_csv(f"{RESULTADOS_DIR}/walkforward_diario_multidia_coincidencia_cobre.csv", index=False)

    graficar_comparacion(curvas_ppo_concat, curva_bh_concat, f"{RESULTADOS_DIR}/walkforward_diario_multidia_curva_capital.png")

    print("\n=== Metricas por dias_holding ===")
    print(tabla.to_string(index=False))
    print("\n=== Coincidencia de direccion PPO vs. Umbral cobre, por ventana ===")
    print(tabla_coincidencia.to_string(index=False))
    # Promedio ponderado por cuantos dias PPO realmente opero en cada
    # ventana (una ventana con 0 dias operados no aporta info de direccion,
    # no debe pesar igual que una con 40 dias operados).
    resumen_coincidencia = tabla_coincidencia.dropna(subset=["coincidencia_direccion_%"]).groupby("dias_holding").apply(
        lambda g: np.average(g["coincidencia_direccion_%"], weights=g["dias_ppo_opero"]) if g["dias_ppo_opero"].sum() > 0 else np.nan
    )
    print("\nCoincidencia promedio ponderada por dias_holding:")
    print(resumen_coincidencia)
