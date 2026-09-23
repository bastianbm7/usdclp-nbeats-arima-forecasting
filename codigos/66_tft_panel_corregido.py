# CORRECCION (2026-09-23) - rehace 9.33 (Issue #13 Fase 3: TFT sobre el panel).
# Reemplaza para efectos del paper a 56.
#
# Tres correcciones:
#   1. BUG DE ALINEACION DEL FORECAST (56:68-73,102,110-111): la fila de
#      cross_validation con ds=D tiene el pronostico del retorno que TERMINA en
#      D (hecho con informacion hasta el cutoff D-1). 56 la unia por ds a la
#      fila D del test y la aplicaba al retorno D -> D+1 (el de y_next): la
#      posicion de cada dia usaba el pronostico del retorno de AYER. Aca la
#      decision en t usa la fila cuyo cutoff es t (pronostico del retorno t ->
#      t+1, hecho con informacion hasta t). La auditoria (tft_fix.py) mostro que
#      solo realinear esto llevaba el Sharpe de -1.18 a ~+3.8 - pero con el
#      panel SIN corregir el timestamp del cobre, asi que ese +3.8 tambien
#      estaba contaminado. Por eso:
#   2. Panel alineado (60): copper_ret_1d/copper_mom_5d conocidos antes del
#      precio FX de cada fila.
#   3. Costos ida+vuelta (0.15% USD/CLP) sobre |f| cada dia + cota inferior de
#      rotacion; Sharpe con IC95.
# Y se agrega el control "TFT solo-CLP" (misma arquitectura entrenada solo con
# la serie de CLP) en las 5 ventanas: 9.33 lo citaba pero no estaba en el
# codigo (el CSV fase3_control_solo_clp_tft.csv no lo genera ningun script).
#
# Uso: python 66_tft_panel_corregido.py --modo panel|solo_clp  (luego --resumen)

import argparse
import importlib
import os
import time

import numpy as np
import pandas as pd
import torch

ce = importlib.import_module("costos_y_estadistica")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")

BASES_DIR = "../datos/bases"
RESULTADOS_DIR = "../datos/resultados"
PANEL_ALINEADO = f"{BASES_DIR}/panel_fx_diario_alineado.csv"
N_WINDOWS_WF, N_TEST_POR_VENTANA = 5, 60
HIST_EXOG = ["macd_rel", "rsi_norm", "copper_ret_1d", "copper_mom_5d", "vol_realizada"]
STAT_EXOG = ["es_commodity"]
INPUT_SIZE, MAX_STEPS, SEED = 20, 1500, 42  # identicos a 56
SPREAD = ce.SPREAD_IDA_VUELTA["CLP"]


def forecast_ventana(panel, fecha_objetivo, solo_clp):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TFT
    pv = panel[panel["ds"] <= fecha_objetivo]
    if solo_clp:
        pv = pv[pv["par"] == "CLP=X"]
    df = pv[["par", "ds", "retorno_1d"] + HIST_EXOG].rename(columns={"par": "unique_id", "retorno_1d": "y"}).reset_index(drop=True)
    kw = dict(h=1, input_size=INPUT_SIZE, hist_exog_list=HIST_EXOG, max_steps=MAX_STEPS, scaler_type="standard",
              random_seed=SEED, enable_progress_bar=False, enable_model_summary=False, logger=False)
    if solo_clp:
        modelo = TFT(**kw)
        static = None
    else:
        modelo = TFT(stat_exog_list=STAT_EXOG, **kw)
        static = pv[["par", "es_commodity"]].drop_duplicates().rename(columns={"par": "unique_id"}).reset_index(drop=True)
        static["es_commodity"] = static["es_commodity"].astype(float)
    nf = NeuralForecast(models=[modelo], freq="B")
    cv = nf.cross_validation(df=df, static_df=static, n_windows=N_TEST_POR_VENTANA, step_size=1, refit=False, verbose=False)
    return cv[cv["unique_id"] == "CLP=X"].rename(columns={"TFT": "mu_pred"})[["ds", "cutoff", "mu_pred", "y"]]


def correr(modo):
    salida = f"{RESULTADOS_DIR}/correccion_tft_{modo}_forecasts.csv"
    panel = pd.read_csv(PANEL_ALINEADO, parse_dates=["ds"])
    clp = panel[panel["par"] == "CLP=X"].sort_values("ds").reset_index(drop=True)
    clp_wf = clp.iloc[:-1].reset_index(drop=True)  # cada fila de test necesita la fila siguiente (su retorno realizado)
    partes = []
    t0 = time.time()
    for i, (_, test) in enumerate(wf_mod.ventanas_walkforward(clp_wf, N_WINDOWS_WF, N_TEST_POR_VENTANA)):
        idx_fin = clp.index[clp["ds"] == test["ds"].iloc[-1]][0]
        fecha_objetivo = clp["ds"].iloc[idx_fin + 1]
        cv = forecast_ventana(panel, fecha_objetivo, modo == "solo_clp")
        # decision en t (fila de test) <- pronostico cuyo cutoff es t (retorno t -> t+1)
        m = test[["ds", "y", "y_next", "vol_realizada", "retorno_1d", "copper_ret_1d"]].merge(
            cv.rename(columns={"ds": "ds_objetivo", "y": "ret_objetivo_log"}), left_on="ds", right_on="cutoff", how="left")
        m["ventana"] = i + 1
        partes.append(m)
        print(f"  {modo} ventana {i+1}/{N_WINDOWS_WF}: {m['mu_pred'].notna().sum()}/{len(m)} con forecast, {(time.time()-t0)/60:.1f} min", flush=True)
    out = pd.concat(partes, ignore_index=True)
    out.to_csv(salida, index=False)
    print(f"[ok] {salida}")


def resumen():
    filas = []
    for modo in ["panel", "solo_clp"]:
        f = f"{RESULTADOS_DIR}/correccion_tft_{modo}_forecasts.csv"
        if not os.path.exists(f):
            print(f"(falta {f})")
            continue
        d = pd.read_csv(f, parse_dates=["ds"]).dropna(subset=["mu_pred"])
        ret = ((d["y_next"] - d["y"]) / d["y"]).to_numpy()
        # chequeo de alineacion: el objetivo del forecast debe ser el retorno realizado t -> t+1
        chequeo = np.corrcoef(np.expm1(d["ret_objetivo_log"]), ret)[0, 1]
        s2 = d["vol_realizada"].to_numpy() ** 2
        mu = d["mu_pred"].to_numpy()
        pos = np.clip(np.divide(mu, s2, out=np.zeros_like(mu), where=s2 > 0), -1, 1)
        rmse = np.sqrt(np.mean((mu - d["ret_objetivo_log"]) ** 2))
        rmse0 = np.sqrt(np.mean(d["ret_objetivo_log"] ** 2))
        base = {"modelo": f"TFT {modo} (corregido)", "chequeo_alineacion_corr(objetivo,retorno_realizado)": chequeo,
                "rmse": rmse, "rmse_predecir_cero": rmse0, "acierto_direccion_%": 100 * np.mean(np.sign(mu) == np.sign(ret)),
                "corr_forecast_retorno": np.corrcoef(mu, ret)[0, 1], "pct_dias_saturado": 100 * np.mean(np.abs(pos) >= 0.999)}
        for costo, r in [("bruto", ce.retornos_posicion_fija(pos, ret, 0.0)),
                         (f"ida+vuelta {100*SPREAD:.2f}%", ce.retornos_posicion_fija(pos, ret, SPREAD)),
                         (f"rotacion {100*SPREAD:.2f}%", ce.retornos_posicion_fija(pos, ret, SPREAD, modo="rotacion"))]:
            filas.append(ce.metricas_desde_retornos(r, base["modelo"], extra={**base, "costo": costo}))
    t = pd.DataFrame(filas)
    t.to_csv(f"{RESULTADOS_DIR}/correccion_tft_metricas.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["panel", "solo_clp"])
    ap.add_argument("--resumen", action="store_true")
    ap.add_argument("--hilos", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="max_steps=20, salida con sufijo _smoke (valida la logica de alineacion)")
    a = ap.parse_args()
    torch.set_num_threads(a.hilos)
    if a.smoke:
        MAX_STEPS = 20
        RESULTADOS_DIR = f"{RESULTADOS_DIR}/correccion_rl_smoke"
    if a.resumen:
        resumen()
    else:
        correr(a.modo)
