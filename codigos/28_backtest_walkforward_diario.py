# Issue #5, paso 2 (reconstruccion RL diaria) - backtest final: PPO (agente
# diario, 27_entorno_trading_rl_diario.py) vs. buy-and-hold vs. umbral simple
# (forecast NHITS) vs. umbral cobre (la senal de 22_kelly_diario_cobre.py,
# 9.13-9.14 del paper) - mismo esquema que 14_backtest_walkforward_gestion_riesgo.py
# (walk-forward de 5 ventanas, reentrenando PPO en cada una), agregando el
# umbral de cobre como punto de comparacion extra porque es el hallazgo que
# motiva toda esta reconstruccion: la pregunta que responde este script no es
# solo "gana plata el agente" sino "el agente PPO (con gestion de riesgo real,
# forecast NHITS, TP/SL) mejora sobre la senal simple de cobre que ya se
# probo rentable en el paso 1 y en 22/25?".
#
# Anotacion sobre "dias_desde_refit" (columna del dataset, ver 26): el
# forecast NHITS que ve el agente puede tener hasta 4 dias de antiguedad
# (cadencia de refit=5 dias, ver nota de diseno en 26) - la volatilidad GARCH,
# MACD/RSI y las features de cobre SI son frescas cada dia. Se documenta
# explicitamente en el reporte final, no se oculta.

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

entorno_mod = importlib.import_module("27_entorno_trading_rl_diario")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")  # reusa ventanas_walkforward(), generica (no especifica de semanal)
kelly_diario_mod = importlib.import_module("22_kelly_diario_cobre")  # reusa posiciones_umbral_cobre()

RESULTADOS_DIR = "../datos/resultados"

CAPITAL_INICIAL = 100.0
RIESGO_MAX_PCT = entorno_mod.RIESGO_MAX_PCT
K_STOP_LOSS = entorno_mod.K_STOP_LOSS
SLIPPAGE_PCT = entorno_mod.SLIPPAGE_PCT

N_WINDOWS_WF = 5
N_TEST_POR_VENTANA = 60  # mismo tamanio de ventana de test que 22_kelly_diario_cobre.py y 25 (300 dias ~14 meses de test total) - comparabilidad directa con el paso 1
TOTAL_TIMESTEPS_PPO = 100_000  # mismo presupuesto que el agente semanal (14)
SEED = 42


def posiciones_ppo(df_train, df_test):
    env_train = entorno_mod.USDCLPTradingEnvDiario(df=df_train)
    modelo = PPO("MlpPolicy", env_train, seed=SEED, verbose=0)
    modelo.learn(total_timesteps=TOTAL_TIMESTEPS_PPO)

    env_test = entorno_mod.USDCLPTradingEnvDiario(df=df_test)
    obs, _ = env_test.reset()
    posiciones = []
    terminado = False
    while not terminado:
        accion, _ = modelo.predict(obs, deterministic=True)
        posiciones.append(entorno_mod.POSICION_POR_ACCION[int(accion)])
        obs, _, terminado, _, _ = env_test.step(accion)
    return posiciones


def posiciones_umbral_simple(df_test, umbral):
    forecast_rel = (df_test["nhits_h1"] - df_test["y"]) / df_test["y"]
    return list(np.where(forecast_rel > umbral, 1.0, np.where(forecast_rel < -umbral, -1.0, 0.0)))


def simular_con_gestion_riesgo(df_test, posiciones, nombre_estrategia, capital_inicial=CAPITAL_INICIAL):
    # Mismo mecanismo que el entorno (27_entorno_trading_rl_diario.py) - el
    # backtest evalua exactamente la misma economia que el agente entreno a
    # optimizar, no una segunda implementacion que podria divergir.
    df_test = entorno_mod.precomputar_salidas_tp_sl(df_test.reset_index(drop=True), K_STOP_LOSS)

    capital = capital_inicial
    posicion_previa = 0.0
    filas = []
    for i, fila in df_test.iterrows():
        posicion = posiciones[i]
        entrada, vol, take_profit = fila["y"], fila["vol_garch"], fila["nhits_h1"]

        if posicion == 0:
            notional, pnl, razon, precio_salida, stop_loss = 0.0, 0.0, "plano", entrada, np.nan
        else:
            direccion = "largo" if posicion > 0 else "corto"
            distancia_riesgo = K_STOP_LOSS * vol
            stop_loss = fila[f"stop_loss_{direccion}"]
            precio_salida, razon = fila[f"precio_salida_{direccion}"], fila[f"razon_cierre_{direccion}"]
            notional = (RIESGO_MAX_PCT * capital) / distancia_riesgo if distancia_riesgo > 0 else 0.0
            retorno_pct = posicion * (precio_salida - entrada) / entrada
            costo_slippage = SLIPPAGE_PCT * notional if posicion != posicion_previa else 0.0
            pnl = notional * retorno_pct - costo_slippage

        capital += pnl
        posicion_previa = posicion
        filas.append({
            "ds": fila["ds"], "estrategia": nombre_estrategia, "precio_entrada": entrada, "precio_salida": precio_salida,
            "take_profit": take_profit, "stop_loss": stop_loss, "posicion": posicion, "notional": notional,
            "razon_cierre": razon, "pnl": pnl, "capital": capital,
        })
    return pd.DataFrame(filas)


def simular_buy_and_hold_simple(df_test, capital_inicial):
    capital = capital_inicial
    filas = []
    for _, fila in df_test.reset_index(drop=True).iterrows():
        retorno_dia = (fila["y_next"] - fila["y"]) / fila["y"]
        pnl = capital * retorno_dia
        capital += pnl
        filas.append({
            "ds": fila["ds"], "estrategia": "Buy-and-hold", "precio_entrada": fila["y"], "precio_salida": fila["y_next"],
            "take_profit": np.nan, "stop_loss": np.nan, "posicion": 1.0, "notional": capital,
            "razon_cierre": "buy_and_hold", "pnl": pnl, "capital": capital,
        })
    return pd.DataFrame(filas)


def calcular_metricas(resultado, nombre):
    capital_previo = resultado["capital"].shift(1).fillna(CAPITAL_INICIAL)
    r = resultado["pnl"] / capital_previo
    capital = resultado["capital"]
    retorno_total_pct = 100 * (capital.iloc[-1] / CAPITAL_INICIAL - 1)
    sharpe = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else np.nan  # 252, no 52 - anualizacion diaria (mismo criterio que 22_kelly_diario_cobre.py)
    drawdown = capital / capital.cummax() - 1
    operaciones = (resultado["posicion"] != 0).sum()
    win_rate = 100 * (resultado.loc[resultado["posicion"] != 0, "pnl"] > 0).mean() if operaciones > 0 else np.nan
    return {
        "estrategia": nombre, "capital_final": capital.iloc[-1], "retorno_total_%": retorno_total_pct,
        "sharpe_anualizado": sharpe, "max_drawdown_%": 100 * drawdown.min(), "win_rate_%": win_rate, "operaciones": operaciones,
    }


def graficar_curva_capital(resultados, path_salida):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colores = {"Buy-and-hold": "black", "Umbral simple (forecast)": "steelblue", "Umbral cobre": "darkorange", "PPO (RL diario)": "crimson"}
    for nombre, r in resultados.items():
        ax.plot(r["ds"], r["capital"], label=nombre, color=colores.get(nombre), linewidth=1.6)
    ax.axhline(CAPITAL_INICIAL, color="gray", linestyle=":", linewidth=1, label="Capital inicial ($100)")
    ax.set_ylabel("Capital ($)")
    ax.set_title(f"Backtest walk-forward diario ({N_WINDOWS_WF} ventanas x {N_TEST_POR_VENTANA} dias) - Issue #5")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


def graficar_puntos_entrada_salida(resultado_ppo, path_salida):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(resultado_ppo["ds"], resultado_ppo["precio_entrada"], color="black", linewidth=1, label="USD/CLP (diario)", zorder=1)

    colores_razon = {"take_profit": "green", "stop_loss": "red", "cierre_dia": "gray"}
    for _, op in resultado_ppo.iterrows():
        if op["posicion"] == 0:
            continue
        marcador = "^" if op["posicion"] > 0 else "v"
        ax.scatter(op["ds"], op["precio_entrada"], marker=marcador, color="blue", s=45, zorder=3)
        ax.scatter(op["ds"] + pd.Timedelta(hours=12), op["precio_salida"], marker="o", color=colores_razon[op["razon_cierre"]], s=25, zorder=3)

    for etiqueta, color in [("Entrada larga (▲) / corta (▼)", "blue"), ("Salida: take-profit", "green"), ("Salida: stop-loss", "red"), ("Salida: cierre de dia", "gray")]:
        ax.scatter([], [], color=color, label=etiqueta)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Agente PPO diario: puntos de entrada/salida sobre el holdout walk-forward completo")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df_completo = entorno_mod.cargar_dataset()
    print(f"Dataset diario: {len(df_completo)} dias. Walk-forward: {N_WINDOWS_WF} ventanas x {N_TEST_POR_VENTANA} dias de test cada una ({N_WINDOWS_WF*N_TEST_POR_VENTANA} dias de test en total)")

    # Umbral simple: la MEDIANA de |forecast_rel| en todo el dataset - regla de
    # decision "simple" honesta (no un valor adivinado): opera cuando el
    # forecast se desvia mas de lo que se desvia en un dia tipico, lo que
    # deja aproximadamente la mitad de los dias operando. Se calcula UNA vez
    # sobre todo el dataset (no por ventana) para que sea un umbral fijo y
    # comparable entre ventanas, igual de "simple" que el umbral fijo del
    # agente semanal (14, UMBRAL_SIMPLE=0.003 constante).
    forecast_rel_abs = ((df_completo["nhits_h1"] - df_completo["y"]) / df_completo["y"]).abs()
    UMBRAL_SIMPLE = forecast_rel_abs.median()
    print(f"Distribucion de |forecast_rel| (nhits_h1 vs y): mediana={UMBRAL_SIMPLE:.5f}, p75={forecast_rel_abs.quantile(0.75):.5f} -> UMBRAL_SIMPLE={UMBRAL_SIMPLE:.5f} (mediana)\n")

    resultados_por_estrategia = {"PPO (RL diario)": [], "Buy-and-hold": [], "Umbral simple (forecast)": [], "Umbral cobre": []}
    capitales = {k: CAPITAL_INICIAL for k in resultados_por_estrategia}

    for w, (df_train, df_test) in enumerate(wf_mod.ventanas_walkforward(df_completo, N_WINDOWS_WF, N_TEST_POR_VENTANA)):
        print(f"--- Ventana {w+1}/{N_WINDOWS_WF}: train={len(df_train)} dias, test={df_test['ds'].min().date()} a {df_test['ds'].max().date()} ---")
        pos_ppo = posiciones_ppo(df_train, df_test)
        pos_umbral = posiciones_umbral_simple(df_test, UMBRAL_SIMPLE)
        pos_cobre = list(kelly_diario_mod.posiciones_umbral_cobre(df_test))

        for nombre, posiciones in [("PPO (RL diario)", pos_ppo), ("Umbral simple (forecast)", pos_umbral), ("Umbral cobre", pos_cobre)]:
            r = simular_con_gestion_riesgo(df_test, posiciones, nombre, capital_inicial=capitales[nombre])
            capitales[nombre] = r["capital"].iloc[-1]
            resultados_por_estrategia[nombre].append(r)

        r_bh = simular_buy_and_hold_simple(df_test, capital_inicial=capitales["Buy-and-hold"])
        capitales["Buy-and-hold"] = r_bh["capital"].iloc[-1]
        resultados_por_estrategia["Buy-and-hold"].append(r_bh)

        print(f"    capitales tras ventana {w+1}: " + ", ".join(f"{k}=${v:.2f}" for k, v in capitales.items()))

    resultados_concat = {nombre: pd.concat(partes, ignore_index=True) for nombre, partes in resultados_por_estrategia.items()}
    for nombre, r in resultados_concat.items():
        r.to_csv(f"{RESULTADOS_DIR}/walkforward_diario_{nombre.split()[0].lower()}_operaciones.csv", index=False)

    tabla = pd.DataFrame([calcular_metricas(r, nombre) for nombre, r in resultados_concat.items()])
    tabla = tabla.sort_values("sharpe_anualizado", ascending=False).reset_index(drop=True)
    tabla.to_csv(f"{RESULTADOS_DIR}/walkforward_diario_metricas.csv", index=False)

    graficar_curva_capital(resultados_concat, f"{RESULTADOS_DIR}/walkforward_diario_curva_capital.png")
    graficar_puntos_entrada_salida(resultados_concat["PPO (RL diario)"], f"{RESULTADOS_DIR}/walkforward_diario_puntos_entrada_salida.png")

    print(f"\n=== Walk-forward diario completo: {N_WINDOWS_WF*N_TEST_POR_VENTANA} dias de test (out-of-sample), capital inicial ${CAPITAL_INICIAL:.0f}, riesgo {RIESGO_MAX_PCT*100:.0f}%/operacion ===\n")
    print(tabla.to_string(index=False))
