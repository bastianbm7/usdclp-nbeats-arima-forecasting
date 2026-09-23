# INVALIDADO / SUPERADO (2026-09-23) - la grilla N x h extendida: ver script 64 (lote D, subconjunto) + 65.
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
# Issue #11: extension directa de la grilla del Issue #10 (40) a ventanas de
# holding mas largas (N=10,12,14,20) - Bastian observo que N=7 (el mas largo
# de la grilla original) tenia el mejor perfil de riesgo (menos stop-loss,
# menor drawdown, mas cierres por trailing-stop/take-profit) y quiere ver si
# el patron se sostiene o llega a un techo al alargar aun mas la ventana.
#
# Reusa TODO sin cambios: mismo dataset (dataset_entrenamiento_rl_diario_h5.csv,
# ya llega a h5, no hace falta regenerar NHITS) y el mismo entorno
# (32_entorno_trading_rl_diario_multidia.py, ya generaliza dias_holding y
# horizonte_tp) - solo cambia el rango de N. Logica identica a
# 40_backtest_walkforward_diario_grilla_nh.py, duplicada en vez de
# parametrizada para no tocar el archivo ya cerrado del Issue #10.
#
# Caveat esperado, documentado desde el diseno (no una sorpresa post-hoc):
# con N_TEST_POR_VENTANA=60 fijo, N=20 deja solo 3 decisiones por ventana x 5
# ventanas = 15 decisiones totales - la combinacion mas ruidosa de toda la
# grilla (original + esta extension).

import importlib
import time

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

multidia_mod = importlib.import_module("32_entorno_trading_rl_diario_multidia")
entorno_diario_mod = importlib.import_module("27_entorno_trading_rl_diario")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")

RESULTADOS_DIR = "../datos/resultados"
DATASET_PATH_H5 = "../datos/resultados/dataset_entrenamiento_rl_diario_h5.csv"

CAPITAL_INICIAL = 100.0
N_WINDOWS_WF = 5
N_TEST_POR_VENTANA = 60
TOTAL_TIMESTEPS_PPO = 100_000
SEED = 42

VALORES_N = [10, 12, 14, 20]
VALORES_H = [1, 2, 3, 4, 5]


def entrenar_y_evaluar(df_train, df_test, dias_holding, horizonte_tp):
    env_train = multidia_mod.USDCLPTradingEnvDiarioMultidia(df=df_train, dias_holding=dias_holding, horizonte_tp=horizonte_tp)
    modelo = PPO("MlpPolicy", env_train, seed=SEED, verbose=0)
    modelo.learn(total_timesteps=TOTAL_TIMESTEPS_PPO)

    env_test = multidia_mod.USDCLPTradingEnvDiarioMultidia(df=df_test, dias_holding=dias_holding, horizonte_tp=horizonte_tp)
    obs, _ = env_test.reset()
    posiciones = []
    terminado = False
    while not terminado:
        accion, _ = modelo.predict(obs, deterministic=True)
        posiciones.append(multidia_mod.POSICION_POR_ACCION[int(accion)])
        obs, _, terminado, _, _ = env_test.step(accion)
    return posiciones, env_test.df


def simular_con_gestion_riesgo(df_decisiones, posiciones, capital_inicial):
    capital = capital_inicial
    posicion_previa = 0.0
    filas = []
    for i, fila in df_decisiones.reset_index(drop=True).iterrows():
        posicion = posiciones[i]
        entrada = fila["y"]
        if posicion == 0:
            notional, pnl, razon, dias_hasta_salida = 0.0, 0.0, "plano", 0
        else:
            direccion = "largo" if posicion > 0 else "corto"
            distancia_riesgo = multidia_mod.K_STOP_LOSS * fila["vol_garch"]
            precio_salida, razon = fila[f"precio_salida_{direccion}"], fila[f"razon_cierre_{direccion}"]
            dias_hasta_salida = fila[f"dias_hasta_salida_{direccion}"]
            notional = (multidia_mod.RIESGO_MAX_PCT * capital) / distancia_riesgo if distancia_riesgo > 0 else 0.0
            retorno_pct = posicion * (precio_salida - entrada) / entrada
            costo_slippage = multidia_mod.SLIPPAGE_PCT * notional if posicion != posicion_previa else 0.0
            pnl = notional * retorno_pct - costo_slippage
        capital += pnl
        posicion_previa = posicion
        filas.append({"ds": fila["ds"], "posicion": posicion, "razon_cierre": razon, "dias_hasta_salida": dias_hasta_salida, "pnl": pnl, "capital": capital})
    return pd.DataFrame(filas)


def calcular_metricas(resultado, dias_holding, horizonte_tp):
    capital_previo = resultado["capital"].shift(1).fillna(CAPITAL_INICIAL)
    r = resultado["pnl"] / capital_previo
    capital = resultado["capital"]
    retorno_total_pct = 100 * (capital.iloc[-1] / CAPITAL_INICIAL - 1)
    periodos_por_anio = 252 / dias_holding
    sharpe = (r.mean() / r.std()) * np.sqrt(periodos_por_anio) if r.std() > 0 else np.nan
    drawdown = capital / capital.cummax() - 1
    operaciones = (resultado["posicion"] != 0).sum()
    con_posicion = resultado[resultado["posicion"] != 0]
    win_rate = 100 * (con_posicion["pnl"] > 0).mean() if operaciones > 0 else np.nan
    dist_razon = resultado["razon_cierre"].value_counts(normalize=True) * 100
    dias_reales_prom = con_posicion["dias_hasta_salida"].mean() if operaciones > 0 else np.nan
    return {
        "N": dias_holding, "h": horizonte_tp, "n_decisiones": len(resultado), "capital_final": capital.iloc[-1],
        "retorno_total_%": retorno_total_pct, "sharpe_anualizado": sharpe, "max_drawdown_%": 100 * drawdown.min(),
        "win_rate_%": win_rate, "operaciones": operaciones, "dias_reales_promedio": dias_reales_prom,
        "%_cierre_periodo": dist_razon.get("cierre_periodo", 0.0), "%_trailing_stop": dist_razon.get("trailing_stop", 0.0),
        "%_take_profit": dist_razon.get("take_profit", 0.0), "%_stop_loss": dist_razon.get("stop_loss", 0.0),
    }


if __name__ == "__main__":
    df_completo = entorno_diario_mod.cargar_dataset(DATASET_PATH_H5)
    print(f"Dataset diario (hasta h5, reusado sin regenerar): {len(df_completo)} dias, {df_completo['ds'].min().date()} a {df_completo['ds'].max().date()}")
    print(f"Grilla extendida: N={VALORES_N} x h={VALORES_H} = {len(VALORES_N) * len(VALORES_H)} combinaciones\n")

    filas_metricas = []
    inicio_total = time.time()
    combo_num, total_combos = 0, len(VALORES_N) * len(VALORES_H)

    for N in VALORES_N:
        for h in VALORES_H:
            combo_num += 1
            t0 = time.time()
            capital = CAPITAL_INICIAL
            partes = []
            for df_train, df_test in wf_mod.ventanas_walkforward(df_completo, N_WINDOWS_WF, N_TEST_POR_VENTANA):
                posiciones, df_dec = entrenar_y_evaluar(df_train, df_test, dias_holding=N, horizonte_tp=h)
                r = simular_con_gestion_riesgo(df_dec, posiciones, capital)
                capital = r["capital"].iloc[-1]
                partes.append(r)
            resultado = pd.concat(partes, ignore_index=True)
            resultado.to_csv(f"{RESULTADOS_DIR}/grilla_nh_largo_N{N}_h{h}_operaciones.csv", index=False)

            metricas = calcular_metricas(resultado, N, h)
            filas_metricas.append(metricas)
            pd.DataFrame(filas_metricas).to_csv(f"{RESULTADOS_DIR}/grilla_nh_largo_metricas.csv", index=False)

            elapsed, elapsed_total = time.time() - t0, time.time() - inicio_total
            print(f"[{combo_num}/{total_combos}] N={N} h={h}: retorno={metricas['retorno_total_%']:.1f}%, "
                  f"sharpe={metricas['sharpe_anualizado']:.2f}, dias_reales={metricas['dias_reales_promedio']:.2f}, "
                  f"n_decisiones={metricas['n_decisiones']} ({elapsed / 60:.1f} min, acumulado {elapsed_total / 60:.1f} min)")

    print(f"\n=== Grilla extendida completa: {total_combos} combinaciones en {(time.time() - inicio_total) / 60:.1f} min ===")
    tabla = pd.DataFrame(filas_metricas).sort_values(["N", "h"]).reset_index(drop=True)
    print(tabla.to_string(index=False))
