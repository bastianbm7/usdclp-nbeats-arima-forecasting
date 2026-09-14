# Walk-forward multi-ventana (consistente con el resto del proyecto - nunca un
# solo split como el chequeo rapido de 12/13) + capa de gestion de riesgo real:
# capital de $100, tamanio de posicion via risk sizing (arriesgar RIESGO_MAX_PCT
# del capital VIGENTE por operacion, no un monto fijo), stop-loss a la distancia
# de la volatilidad GARCH pronosticada (mismo modelo ganador de
# 09_comparacion_modelos_volatilidad.py - "volatility scaling", la misma idea
# que usan los papers ancla de Wood et al.), take-profit en el precio objetivo
# que ya predice el propio N-HiTS. TP/SL se chequean dia a dia con los precios
# reales dentro de la semana, no solo al cierre semanal.
#
# Reentrena PPO en cada ventana (mismo mecanismo de 12_entrenar_agente_rl.py) -
# lo que cambia entre ventanas es donde corta train/test, no el algoritmo.

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

entorno_mod = importlib.import_module("11_entorno_trading_rl")

DATOS_LARGO = "../datos/bases/usdclp_long.csv"
RESULTADOS_DIR = "../datos/resultados"

CAPITAL_INICIAL = 100.0
RIESGO_MAX_PCT = 0.03
K_STOP_LOSS = 1.0
SLIPPAGE_PCT = entorno_mod.SLIPPAGE_PCT

N_WINDOWS_WF = 5
N_TEST_POR_VENTANA = 20
TOTAL_TIMESTEPS_PPO = 100_000
SEED = 42
UMBRAL_SIMPLE = 0.003


def ventanas_walkforward(df, n_windows=N_WINDOWS_WF, n_test=N_TEST_POR_VENTANA):
    total = len(df)
    for w in range(n_windows):
        fin_test = total - (n_windows - 1 - w) * n_test
        inicio_test = fin_test - n_test
        yield df.iloc[:inicio_test].reset_index(drop=True), df.iloc[inicio_test:fin_test].reset_index(drop=True)


def posiciones_ppo(df_train, df_test):
    env_train = entorno_mod.USDCLPTradingEnv(df=df_train)
    modelo = PPO("MlpPolicy", env_train, seed=SEED, verbose=0)
    modelo.learn(total_timesteps=TOTAL_TIMESTEPS_PPO)

    env_test = entorno_mod.USDCLPTradingEnv(df=df_test)
    obs, _ = env_test.reset()
    posiciones = []
    terminado = False
    while not terminado:
        accion, _ = modelo.predict(obs, deterministic=True)
        posiciones.append(entorno_mod.POSICION_POR_ACCION[int(accion)])
        obs, _, terminado, _, _ = env_test.step(accion)
    return posiciones


def posiciones_umbral_simple(df_test, umbral=UMBRAL_SIMPLE):
    forecast_rel = (df_test["nhits_h1"] - df_test["y"]) / df_test["y"]
    return list(np.where(forecast_rel > umbral, 1.0, np.where(forecast_rel < -umbral, -1.0, 0.0)))


def simular_con_gestion_riesgo(df_test, diario, posiciones, nombre_estrategia, capital_inicial=CAPITAL_INICIAL):
    # Usa el mismo precomputo de salidas TP/SL que el entorno (11_entorno_trading_rl.py)
    # para que la evaluacion sea matematicamente identica a lo que el agente
    # entreno a optimizar - no una segunda implementacion que podria divergir.
    df_test = entorno_mod.precomputar_salidas_tp_sl(df_test.reset_index(drop=True), diario, K_STOP_LOSS)

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
    # Referencia sin apalancar: mantiene el 100% del capital vigente en la
    # posicion, sin TP/SL ni slippage (una sola posicion continua, no se
    # "reabre" cada semana) - a diferencia de las estrategias activas, que si
    # pasan por la capa de gestion de riesgo (apalancamiento + stop-loss).
    capital = capital_inicial
    filas = []
    for _, fila in df_test.reset_index(drop=True).iterrows():
        retorno_semana = (fila["y_next"] - fila["y"]) / fila["y"]
        pnl = capital * retorno_semana
        capital += pnl
        filas.append({
            "ds": fila["ds"], "estrategia": "Buy-and-hold", "precio_entrada": fila["y"], "precio_salida": fila["y_next"],
            "take_profit": np.nan, "stop_loss": np.nan, "posicion": 1.0, "notional": capital,
            "razon_cierre": "buy_and_hold", "pnl": pnl, "capital": capital,
        })
    return pd.DataFrame(filas)


def calcular_metricas(resultado, nombre):
    capital_previo = resultado["capital"].shift(1).fillna(CAPITAL_INICIAL)
    r = resultado["pnl"] / capital_previo  # retorno % real de esa semana, no pnl/capital inicial fijo (distorsiona el Sharpe si el capital se aleja mucho del inicial)
    capital = resultado["capital"]
    retorno_total_pct = 100 * (capital.iloc[-1] / CAPITAL_INICIAL - 1)
    sharpe = (r.mean() / r.std()) * np.sqrt(52) if r.std() > 0 else np.nan
    drawdown = capital / capital.cummax() - 1
    operaciones = (resultado["posicion"] != 0).sum()
    win_rate = 100 * (resultado.loc[resultado["posicion"] != 0, "pnl"] > 0).mean() if operaciones > 0 else np.nan
    return {
        "estrategia": nombre, "capital_final": capital.iloc[-1], "retorno_total_%": retorno_total_pct,
        "sharpe_anualizado": sharpe, "max_drawdown_%": 100 * drawdown.min(), "win_rate_%": win_rate, "operaciones": operaciones,
    }


def graficar_operaciones(resultado_ppo, diario, path_salida):
    fecha_ini, fecha_fin = resultado_ppo["ds"].min(), resultado_ppo["ds"].max() + pd.Timedelta(weeks=1)
    precios = diario[(diario["ds"] >= fecha_ini) & (diario["ds"] <= fecha_fin)]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(precios["ds"], precios["y"], color="black", linewidth=1, label="USD/CLP (diario)", zorder=1)

    colores_razon = {"take_profit": "green", "trailing_stop": "orange", "stop_loss": "red", "cierre_semana": "gray"}
    for _, op in resultado_ppo.iterrows():
        if op["posicion"] == 0:
            continue
        marcador_entrada = "^" if op["posicion"] > 0 else "v"
        ax.scatter(op["ds"], op["precio_entrada"], marker=marcador_entrada, color="blue", s=70, zorder=3)
        fecha_salida = op["ds"] + pd.Timedelta(days=3)  # aprox., solo para separar visualmente entrada/salida
        ax.scatter(fecha_salida, op["precio_salida"], marker="o", color=colores_razon[op["razon_cierre"]], s=40, zorder=3)

    for etiqueta, color in [("Entrada larga (▲) / corta (▼)", "blue"), ("Salida: take-profit", "green"), ("Salida: trailing stop", "orange"), ("Salida: stop-loss", "red"), ("Salida: cierre de semana", "gray")]:
        ax.scatter([], [], color=color, label=etiqueta)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Agente PPO: puntos de entrada/salida sobre el holdout walk-forward completo")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    diario = pd.read_csv(DATOS_LARGO, parse_dates=["ds"])
    df_completo = entorno_mod.cargar_dataset()
    print(f"Dataset: {len(df_completo)} semanas. Walk-forward: {N_WINDOWS_WF} ventanas x {N_TEST_POR_VENTANA} semanas de test cada una ({N_WINDOWS_WF*N_TEST_POR_VENTANA} semanas de test en total)\n")

    resultados_por_estrategia = {"PPO (RL)": [], "Buy-and-hold": [], "Umbral simple (Opcion A)": []}
    capitales = {k: CAPITAL_INICIAL for k in resultados_por_estrategia}

    for w, (df_train, df_test) in enumerate(ventanas_walkforward(df_completo)):
        print(f"--- Ventana {w+1}/{N_WINDOWS_WF}: train={len(df_train)} semanas, test={df_test['ds'].min().date()} a {df_test['ds'].max().date()} ---")
        pos_ppo = posiciones_ppo(df_train, df_test)
        pos_umbral = posiciones_umbral_simple(df_test)

        for nombre, posiciones in [("PPO (RL)", pos_ppo), ("Umbral simple (Opcion A)", pos_umbral)]:
            r = simular_con_gestion_riesgo(df_test, diario, posiciones, nombre, capital_inicial=capitales[nombre])
            capitales[nombre] = r["capital"].iloc[-1]
            resultados_por_estrategia[nombre].append(r)

        r_bh = simular_buy_and_hold_simple(df_test, capital_inicial=capitales["Buy-and-hold"])
        capitales["Buy-and-hold"] = r_bh["capital"].iloc[-1]
        resultados_por_estrategia["Buy-and-hold"].append(r_bh)

    resultados_concat = {nombre: pd.concat(partes, ignore_index=True) for nombre, partes in resultados_por_estrategia.items()}
    for nombre, r in resultados_concat.items():
        r.to_csv(f"{RESULTADOS_DIR}/walkforward_{nombre.split()[0].lower()}_operaciones.csv", index=False)

    tabla = pd.DataFrame([calcular_metricas(r, nombre) for nombre, r in resultados_concat.items()])
    tabla = tabla.sort_values("sharpe_anualizado", ascending=False).reset_index(drop=True)
    tabla.to_csv(f"{RESULTADOS_DIR}/walkforward_gestion_riesgo_metricas.csv", index=False)

    graficar_operaciones(resultados_concat["PPO (RL)"], diario, f"{RESULTADOS_DIR}/walkforward_puntos_entrada_salida.png")

    print(f"\n=== Walk-forward completo: {N_WINDOWS_WF*N_TEST_POR_VENTANA} semanas de test (out-of-sample), capital inicial ${CAPITAL_INICIAL:.0f}, riesgo {RIESGO_MAX_PCT*100:.0f}%/operacion ===\n")
    print(tabla.to_string(index=False))
