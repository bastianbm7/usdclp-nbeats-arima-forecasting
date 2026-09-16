# Issue #10: extension de 35_generar_dataset_rl_diario_h3.py para llegar
# hasta nhits_h5 - necesario para el barrido completo de take-profit fijo por
# horizonte (h1-h5) x ventana de holding (N=3,5,7).
#
# Mismo motivo que 35 para ser un archivo nuevo en vez de tocar 26/35
# in-place: NHITS no tiene semilla fija, re-correr el walk-forward da h1/h2/h3
# ligeramente distintos a los ya versionados - un archivo nuevo evita
# invalidar silenciosamente la reproducibilidad de lo ya documentado (9.17-
# 9.20 usan el dataset de 26; el experimento de TP adaptativo usa el de 35).
#
# H = REFIT_CADENCIA + 5 (no +3 como en 35): el ultimo dia de decision de
# cada bloque (j=5) necesita h_{j+1}..h_{j+5} = h6..h10 para sus propios
# nhits_h1..h5 - formula general H=REFIT_CADENCIA+max_h, verificada contra
# los dos casos anteriores (max_h=2 -> H=7 en 26; max_h=3 -> H=8 en 35).

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import NHITS

import importlib

features_mod = importlib.import_module("19_features_nuevas_validacion")
dataset26_mod = importlib.import_module("26_generar_dataset_rl_diario")

DATOS_LARGO = "../datos/bases/usdclp_long.csv"
RESULTADOS_DIR = "../datos/resultados"

FREQ = dataset26_mod.FREQ
INPUT_SIZE = dataset26_mod.INPUT_SIZE
REFIT_CADENCIA = dataset26_mod.REFIT_CADENCIA
MAX_H = 5
H = REFIT_CADENCIA + MAX_H  # 10 - ver nota de diseno arriba
MAX_STEPS_REFIT = dataset26_mod.MAX_STEPS_REFIT
SCALER_TYPE = dataset26_mod.SCALER_TYPE
N_REFITS = dataset26_mod.N_REFITS

MINMAX_VENTANA_DIARIA = dataset26_mod.MINMAX_VENTANA_DIARIA
MIN_OBS_GARCH = dataset26_mod.MIN_OBS_GARCH


def generar_forecast_nhits_diario_h5(serie_diaria, n_refits, refit_cadencia, h, max_h):
    trainer_kwargs = dict(enable_progress_bar=False, enable_model_summary=False, logger=False)
    modelo = NHITS(h=h, input_size=INPUT_SIZE, loss=MQLoss(), max_steps=MAX_STEPS_REFIT, scaler_type=SCALER_TYPE, **trainer_kwargs)
    nf = NeuralForecast(models=[modelo], freq=FREQ)
    cv = nf.cross_validation(df=serie_diaria, n_windows=n_refits, step_size=refit_cadencia, refit=True, use_init_models=False)
    cv = cv.sort_values(["cutoff", "ds"]).reset_index(drop=True)

    filas = []
    for cutoff, grupo in cv.groupby("cutoff", sort=False):
        grupo = grupo.sort_values("ds").reset_index(drop=True)
        if len(grupo) < refit_cadencia + max_h:
            continue
        for j in range(1, refit_cadencia + 1):
            fila_hoy = grupo.iloc[j - 1]
            fila = {"ds": fila_hoy["ds"], "y": fila_hoy["y"], "dias_desde_refit": j}
            for k in range(1, max_h + 1):
                fila[f"nhits_h{k}"] = grupo.iloc[j - 1 + k]["NHITS-median"]
            filas.append(fila)
    return pd.DataFrame(filas).sort_values("ds").reset_index(drop=True)


if __name__ == "__main__":
    diario = pd.read_csv(DATOS_LARGO, parse_dates=["ds"])
    if "unique_id" not in diario.columns:
        diario.insert(0, "unique_id", "USDCLP")
    serie_nhits = diario[["unique_id", "ds", "y"]].dropna().reset_index(drop=True)
    print(f"Serie diaria: {len(serie_nhits)} obs, {serie_nhits['ds'].min().date()} a {serie_nhits['ds'].max().date()}")
    print(f"Walk-forward NHITS (hasta h{MAX_H}): {N_REFITS} refits x cada {REFIT_CADENCIA} dias (h={H}) = {N_REFITS*REFIT_CADENCIA} dias de decision")

    print(f"\n[1/4] Forecast NHITS walk-forward hasta h{MAX_H} (~12-15s/refit estimado, ~{N_REFITS*13/60:.0f} min estimados)...")
    forecast_nhits = generar_forecast_nhits_diario_h5(serie_nhits, N_REFITS, REFIT_CADENCIA, H, MAX_H)
    print(f"      {len(forecast_nhits)} dias de decision generados, {forecast_nhits['ds'].min().date()} a {forecast_nhits['ds'].max().date()}")

    print(f"\n[2/4] Volatilidad GARCH(1,1) por dia, horizonte=1 ({len(forecast_nhits)} fits)...")
    vol = dataset26_mod.generar_volatilidad_por_dia(diario, forecast_nhits["ds"])

    print(f"\n[3/4] Indicadores tecnicos (MACD, RSI, min/max de {MINMAX_VENTANA_DIARIA} dias)...")
    con_indicadores = dataset26_mod.calcular_indicadores_tecnicos_diarios(diario)

    print(f"\n[4/4] Features de cobre (copper_ret_1d, copper_mom_5d)...")
    macro = features_mod.cargar_macro()
    con_cobre = pd.merge_asof(con_indicadores.sort_values("ds"), macro[["ds", "copper"]], on="ds", direction="backward")
    con_cobre["copper_ret_1d"] = np.log(con_cobre["copper"] / con_cobre["copper"].shift(1))
    con_cobre["copper_mom_5d"] = (con_cobre["copper"] - con_cobre["copper"].shift(5)) / con_cobre["copper"].shift(5)

    dataset = forecast_nhits.merge(vol, on="ds", how="left")
    dataset = dataset.merge(con_cobre[["ds", "macd", "rsi", "precio_min_ventana", "precio_max_ventana", "copper_ret_1d", "copper_mom_5d"]], on="ds", how="left")
    dataset = dataset.sort_values("ds").reset_index(drop=True)
    dataset["y_next"] = dataset["y"].shift(-1)
    dataset = dataset.dropna().reset_index(drop=True)

    columnas = ["ds", "y", "y_next"] + [f"nhits_h{k}" for k in range(1, MAX_H + 1)] + \
               ["vol_garch", "macd", "rsi", "precio_min_ventana", "precio_max_ventana", "copper_ret_1d", "copper_mom_5d", "dias_desde_refit"]
    dataset = dataset[columnas]
    dataset.to_csv(f"{RESULTADOS_DIR}/dataset_entrenamiento_rl_diario_h5.csv", index=False)

    print(f"\n=== listo: {len(dataset)} dias, {dataset['ds'].min().date()} a {dataset['ds'].max().date()} ===")
    print(dataset.head())
    print(dataset.tail())
