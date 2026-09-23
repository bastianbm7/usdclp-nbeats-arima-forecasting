# INVALIDADO / SUPERADO (2026-09-23) - el backtest CAD: ver script 64 (lote A) + 65.
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
# Issue #12, paso 3: mismo experimento que 48_backtest_walkforward_diario_aud.py
# pero sobre el dataset de USD/CAD (46_generar_dataset_rl_diario_cad.py).
# Recordatorio del caveat del Issue: esto testea la senal GENERICA de cobre
# sobre CAD, no el commodity fundamental real de CAD (petroleo) - y CAD tiene
# señales de alerta documentadas (reversion de signo historica del Banco de
# Canada, decoupling reciente con petroleo, radar-baseline 2026-09-17). Un
# resultado negativo aca no cierra la pregunta de si CAD tiene edge via
# petroleo; un resultado positivo tampoco confirma que sea robusto.

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

entorno_mod = importlib.import_module("27_entorno_trading_rl_diario")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")
bt28_mod = importlib.import_module("28_backtest_walkforward_diario")

DATASET_PATH = "../datos/resultados/dataset_entrenamiento_rl_diario_cad.csv"
RESULTADOS_DIR = "../datos/resultados"
NOMBRE_ACTIVO = "USD/CAD"
SUFIJO = "cad"


def posiciones_umbral_cobre_apreciacion(df_test, umbral=0.0):
    # Ver nota en 48_backtest_walkforward_diario_aud.py: el dataset de CAD
    # sale del panel (panel_fx_diario.csv), ya normalizado a "USD por 1 CAD"
    # (cobre sube -> CAD se aprecia -> y SUBE) - signo opuesto al de
    # kelly_diario_mod.posiciones_umbral_cobre (22), que asume la convencion
    # cruda de USD/CLP (y=CLP por USD, cobre sube -> y baja).
    return np.where(df_test["copper_ret_1d"] > umbral, 1.0, np.where(df_test["copper_ret_1d"] < -umbral, -1.0, 0.0))

CAPITAL_INICIAL = bt28_mod.CAPITAL_INICIAL
N_WINDOWS_WF = bt28_mod.N_WINDOWS_WF
N_TEST_POR_VENTANA = bt28_mod.N_TEST_POR_VENTANA
RIESGO_MAX_PCT = bt28_mod.RIESGO_MAX_PCT


def graficar_curva_capital(resultados, path_salida):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colores = {"Buy-and-hold": "black", "Umbral simple (forecast)": "steelblue", "Umbral cobre": "darkorange", "PPO (RL diario)": "crimson"}
    for nombre, r in resultados.items():
        ax.plot(r["ds"], r["capital"], label=nombre, color=colores.get(nombre), linewidth=1.6)
    ax.axhline(CAPITAL_INICIAL, color="gray", linestyle=":", linewidth=1, label="Capital inicial ($100)")
    ax.set_ylabel("Capital ($)")
    ax.set_title(f"Backtest walk-forward diario ({N_WINDOWS_WF} ventanas x {N_TEST_POR_VENTANA} dias) - {NOMBRE_ACTIVO} (Issue #12)")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


def graficar_puntos_entrada_salida(resultado_ppo, path_salida):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(resultado_ppo["ds"], resultado_ppo["precio_entrada"], color="black", linewidth=1, label=f"{NOMBRE_ACTIVO} (diario)", zorder=1)

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
    ax.set_title(f"Agente PPO diario ({NOMBRE_ACTIVO}): puntos de entrada/salida sobre el holdout walk-forward completo")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df_completo = entorno_mod.cargar_dataset(DATASET_PATH)
    print(f"Dataset diario ({NOMBRE_ACTIVO}): {len(df_completo)} dias. Walk-forward: {N_WINDOWS_WF} ventanas x {N_TEST_POR_VENTANA} dias de test cada una ({N_WINDOWS_WF*N_TEST_POR_VENTANA} dias de test en total)")

    forecast_rel_abs = ((df_completo["nhits_h1"] - df_completo["y"]) / df_completo["y"]).abs()
    UMBRAL_SIMPLE = forecast_rel_abs.median()
    print(f"Distribucion de |forecast_rel| (nhits_h1 vs y): mediana={UMBRAL_SIMPLE:.5f}, p75={forecast_rel_abs.quantile(0.75):.5f} -> UMBRAL_SIMPLE={UMBRAL_SIMPLE:.5f} (mediana)\n")

    resultados_por_estrategia = {"PPO (RL diario)": [], "Buy-and-hold": [], "Umbral simple (forecast)": [], "Umbral cobre": []}
    capitales = {k: CAPITAL_INICIAL for k in resultados_por_estrategia}

    for w, (df_train, df_test) in enumerate(wf_mod.ventanas_walkforward(df_completo, N_WINDOWS_WF, N_TEST_POR_VENTANA)):
        print(f"--- Ventana {w+1}/{N_WINDOWS_WF}: train={len(df_train)} dias, test={df_test['ds'].min().date()} a {df_test['ds'].max().date()} ---")
        pos_ppo = bt28_mod.posiciones_ppo(df_train, df_test)
        pos_umbral = bt28_mod.posiciones_umbral_simple(df_test, UMBRAL_SIMPLE)
        pos_cobre = list(posiciones_umbral_cobre_apreciacion(df_test))

        for nombre, posiciones in [("PPO (RL diario)", pos_ppo), ("Umbral simple (forecast)", pos_umbral), ("Umbral cobre", pos_cobre)]:
            r = bt28_mod.simular_con_gestion_riesgo(df_test, posiciones, nombre, capital_inicial=capitales[nombre])
            capitales[nombre] = r["capital"].iloc[-1]
            resultados_por_estrategia[nombre].append(r)

        r_bh = bt28_mod.simular_buy_and_hold_simple(df_test, capital_inicial=capitales["Buy-and-hold"])
        capitales["Buy-and-hold"] = r_bh["capital"].iloc[-1]
        resultados_por_estrategia["Buy-and-hold"].append(r_bh)

        print(f"    capitales tras ventana {w+1}: " + ", ".join(f"{k}=${v:.2f}" for k, v in capitales.items()))

    resultados_concat = {nombre: pd.concat(partes, ignore_index=True) for nombre, partes in resultados_por_estrategia.items()}
    for nombre, r in resultados_concat.items():
        slug = "_".join(nombre.lower().replace("(", "").replace(")", "").split()[:2])
        r.to_csv(f"{RESULTADOS_DIR}/walkforward_diario_{SUFIJO}_{slug}_operaciones.csv", index=False)

    tabla = pd.DataFrame([bt28_mod.calcular_metricas(r, nombre) for nombre, r in resultados_concat.items()])
    tabla = tabla.sort_values("sharpe_anualizado", ascending=False).reset_index(drop=True)
    tabla.to_csv(f"{RESULTADOS_DIR}/walkforward_diario_{SUFIJO}_metricas.csv", index=False)

    graficar_curva_capital(resultados_concat, f"{RESULTADOS_DIR}/walkforward_diario_{SUFIJO}_curva_capital.png")
    graficar_puntos_entrada_salida(resultados_concat["PPO (RL diario)"], f"{RESULTADOS_DIR}/walkforward_diario_{SUFIJO}_puntos_entrada_salida.png")

    print(f"\n=== Walk-forward diario completo ({NOMBRE_ACTIVO}): {N_WINDOWS_WF*N_TEST_POR_VENTANA} dias de test (out-of-sample), capital inicial ${CAPITAL_INICIAL:.0f}, riesgo {RIESGO_MAX_PCT*100:.0f}%/operacion ===\n")
    print(tabla.to_string(index=False))
