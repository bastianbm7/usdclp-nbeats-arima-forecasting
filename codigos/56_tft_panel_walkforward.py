# Issue #13, Fase 3 (Nivel 0): ataca el problema de pooling documentado en
# 9.15 del paper - entrenar con las 13 (ahora 14, con NOK) monedas pooled de
# forma INGENUA (una regresion OLS que promedia el coeficiente de todas las
# monedas por igual) dio PEOR resultado para CLP (+67.9%) que entrenar solo
# con su propia historia (+83.0%), porque la sensibilidad al cobre varia
# demasiado entre monedas para promediarla sin mas (seccion 9.14).
#
# Hipotesis de esta Fase: un modelo con capacidad real de aprender patrones
# ESPECIFICOS por serie (TFT, con el ID de cada moneda -> embedding de
# unique_id nativo de neuralforecast + es_commodity como covariable estatica
# explicita) en vez de una regresion lineal que trata a todas las monedas
# igual, deberia poder aprovechar los datos de las otras 13 monedas SIN
# perjudicar a CLP - a diferencia del pooling ingenuo de 9.14.
#
# Criterio de exito explicito del Issue #13: igualar o superar el +83.0% de
# "solo-CLP" (documentado en 9.15) cuando se entrena con el panel completo -
# no solo evitar que empeore respecto al pooled ingenuo (67.9%).
#
# Metodologia: MISMO esquema de walk-forward que 22/23 (5 ventanas x 60 dias
# de test = 300 dias, cortes identicos a los de CLP en kelly_diario_cobre.py)
# y MISMA formula de Kelly (f = mu_pred / sigma^2, clip +-1) que 22/23 - lo
# UNICO que cambia es de donde sale mu_pred: en vez del coeficiente OLS
# pooled, es el forecast de retorno a 1 dia de un TFT entrenado sobre el
# panel completo (14 monedas), con es_commodity como covariable estatica.
# Reentrena UNA vez por ventana (no diario) - mismo criterio que el resto del
# proyecto (14, 22, 23), no online learning (seccion 8, refit=True, que es
# una linea de investigacion distinta).

import importlib
import time

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT

wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")
kelly_diario_mod = importlib.import_module("22_kelly_diario_cobre")

BASES_DIR = "../datos/bases"
RESULTADOS_DIR = "../datos/resultados"
PANEL_EXTENDIDO = f"{BASES_DIR}/panel_fx_diario_extendido.csv"

N_WINDOWS_WF = kelly_diario_mod.N_WINDOWS_WF          # 5
N_TEST_POR_VENTANA = kelly_diario_mod.N_TEST_POR_VENTANA  # 60
CAPITAL_INICIAL = kelly_diario_mod.CAPITAL_INICIAL
LIMITE_APALANCAMIENTO = kelly_diario_mod.LIMITE_APALANCAMIENTO

HIST_EXOG = ["macd_rel", "rsi_norm", "copper_ret_1d", "copper_mom_5d", "vol_realizada"]
STAT_EXOG = ["es_commodity"]
INPUT_SIZE = 20
MAX_STEPS = 1500  # mismo presupuesto que resolvio el forecast "chato" de NHITS en la seccion 4.1 del paper
SEED = 42


def preparar_panel_neuralforecast(panel):
    cols = ["par", "ds", "retorno_1d"] + HIST_EXOG
    df = panel[cols].rename(columns={"par": "unique_id", "retorno_1d": "y"}).reset_index(drop=True)
    static = panel[["par", "es_commodity"]].drop_duplicates().rename(columns={"par": "unique_id"}).reset_index(drop=True)
    static["es_commodity"] = static["es_commodity"].astype(float)
    return df, static


def entrenar_y_predecir_ventana(panel, fecha_fin_test):
    panel_ventana = panel[panel["ds"] <= fecha_fin_test]
    df, static = preparar_panel_neuralforecast(panel_ventana)

    modelo = TFT(h=1, input_size=INPUT_SIZE, hist_exog_list=HIST_EXOG, stat_exog_list=STAT_EXOG,
                 max_steps=MAX_STEPS, scaler_type="standard", random_seed=SEED, enable_progress_bar=False)
    nf = NeuralForecast(models=[modelo], freq="B")
    cv = nf.cross_validation(df=df, static_df=static, n_windows=N_TEST_POR_VENTANA, step_size=1,
                              refit=False, verbose=False)
    return cv[cv["unique_id"] == "CLP=X"].rename(columns={"TFT": "mu_pred"})[["ds", "mu_pred", "y"]]


def posiciones_desde_forecast(df_test, mu_pred):
    sigma2 = df_test["vol_realizada"].to_numpy(dtype=float) ** 2
    f = np.divide(mu_pred, sigma2, out=np.zeros_like(mu_pred), where=sigma2 > 0)
    return np.clip(f, -LIMITE_APALANCAMIENTO, LIMITE_APALANCAMIENTO)


if __name__ == "__main__":
    panel = pd.read_csv(PANEL_EXTENDIDO, parse_dates=["ds"])
    clp = panel[panel["par"] == "CLP=X"].sort_values("ds").reset_index(drop=True)
    print(f"Panel extendido: {len(panel)} filas, {panel['par'].nunique()} pares "
          f"(incluye NOK). CLP: {len(clp)} obs.")
    print(f"Walk-forward: {N_WINDOWS_WF} ventanas x {N_TEST_POR_VENTANA} dias, TFT h=1 input_size={INPUT_SIZE} "
          f"max_steps={MAX_STEPS}, reentrena 1 vez por ventana (no online).")

    capital = CAPITAL_INICIAL
    partes_resultado, filas_rmse = [], []
    t0_total = time.time()

    for i, (clp_train, clp_test) in enumerate(wf_mod.ventanas_walkforward(clp, N_WINDOWS_WF, N_TEST_POR_VENTANA)):
        fecha_fin_test = clp_test["ds"].iloc[-1]
        print(f"\n--- Ventana {i+1}/{N_WINDOWS_WF}: test hasta {fecha_fin_test.date()} "
              f"({len(clp_train)} obs de entrenamiento disponibles para CLP antes de este corte) ---")
        t0 = time.time()
        forecast_clp = entrenar_y_predecir_ventana(panel, fecha_fin_test)
        print(f"  TFT entrenado y predicho en {time.time()-t0:.0f}s, {len(forecast_clp)} dias pronosticados")

        df_test = clp_test.merge(forecast_clp[["ds", "mu_pred"]], on="ds", how="inner")
        if len(df_test) != len(clp_test):
            print(f"  AVISO: {len(clp_test)-len(df_test)} dias de test sin forecast (desalineacion de calendario "
                  f"entre CLP y el resto del panel) - se evaluan solo los {len(df_test)} dias con forecast valido.")

        rmse = np.sqrt(np.mean((df_test["mu_pred"] - df_test["retorno_1d"]) ** 2))
        filas_rmse.append({"ventana": i + 1, "fecha_fin_test": fecha_fin_test, "rmse_retorno": rmse, "n_obs": len(df_test)})

        posiciones = posiciones_desde_forecast(df_test, df_test["mu_pred"].to_numpy())
        r = kelly_diario_mod.simular(df_test, posiciones, "TFT panel (14 monedas, ID como covariable estatica)", capital)
        capital = r["capital"].iloc[-1]
        partes_resultado.append(r)
        print(f"  Capital al cierre de la ventana {i+1}: ${capital:.2f}")

    print(f"\nTiempo total: {(time.time()-t0_total)/60:.1f} min")

    resultado = pd.concat(partes_resultado, ignore_index=True)
    resultado.to_csv(f"{RESULTADOS_DIR}/fase3_tft_panel_operaciones.csv", index=False)

    metrica = kelly_diario_mod.calcular_metricas(resultado, "TFT panel (14 monedas, ID como covariable estatica)")
    tabla_rmse = pd.DataFrame(filas_rmse)
    tabla_rmse.to_csv(f"{RESULTADOS_DIR}/fase3_tft_panel_rmse_por_ventana.csv", index=False)

    referencia = pd.read_csv(f"{RESULTADOS_DIR}/panel_fx_pooled_vs_single_metricas.csv")
    tabla_final = pd.concat([pd.DataFrame([metrica]), referencia], ignore_index=True).sort_values(
        "sharpe_anualizado", ascending=False).reset_index(drop=True)
    tabla_final.to_csv(f"{RESULTADOS_DIR}/fase3_tft_vs_baselines_metricas.csv", index=False)

    print("\n=== RMSE del forecast de retorno de TFT, por ventana ===")
    print(tabla_rmse.to_string(index=False))

    print("\n=== TFT panel vs. baselines de 9.14 (solo-CLP +83.0%, pooled OLS ingenuo +67.9%) ===")
    print(tabla_final.to_string(index=False))

    baseline_solo_clp = referencia.loc[referencia["estrategia"].str.contains("solo CLP"), "retorno_total_%"].iloc[0]
    print(f"\nCriterio de exito (Issue #13): retorno TFT panel ({metrica['retorno_total_%']:.1f}%) "
          f">= solo-CLP ({baseline_solo_clp:.1f}%)? "
          f"{'SI' if metrica['retorno_total_%'] >= baseline_solo_clp else 'NO'}")
