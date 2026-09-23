# Issue #12, paso 1: dataset walk-forward diario para AUD/USD, mismo patron
# que 26_generar_dataset_rl_diario.py (NHITS + GARCH + copper_ret_1d) pero
# para la primera moneda commodity nueva (radar-baseline 2026-09-17: AUD es
# la mejor respaldada - el cobre es su commodity fundamental real, a
# diferencia de CAD/petroleo o NZD/lacteos - y sin señales de alerta de
# inestabilidad como las que tiene CAD documentadas por el Banco de Canada).
#
# Reusa CASI TODA la logica de 26 via importlib (generar_forecast_nhits_diario,
# forecast_volatilidad_garch_diario, generar_volatilidad_por_dia,
# calcular_indicadores_tecnicos_diarios son genericas, no dependen de que la
# serie sea CLP) - mismo criterio que 39 reuso 26. Lo unico que cambia es la
# fuente de la serie de precio (AUD/USD ya limpia y normalizada en
# panel_fx_diario.csv, generado por 23_dataset_multi_par_diario.py - evita
# re-descargar/re-limpiar tickers erroneos de yfinance) y el nombre de salida.

import numpy as np
import pandas as pd

import importlib

generar_mod = importlib.import_module("10_generar_dataset_rl")
features_mod = importlib.import_module("19_features_nuevas_validacion")
dataset26_mod = importlib.import_module("26_generar_dataset_rl_diario")

PANEL_CACHE = "../datos/bases/panel_fx_diario.csv"
RESULTADOS_DIR = "../datos/resultados"
PAR = "AUDUSD=X"
UNIQUE_ID = "AUDUSD"

FREQ = dataset26_mod.FREQ
REFIT_CADENCIA = dataset26_mod.REFIT_CADENCIA
H = dataset26_mod.H
N_REFITS = dataset26_mod.N_REFITS
MINMAX_VENTANA_DIARIA = dataset26_mod.MINMAX_VENTANA_DIARIA


if __name__ == "__main__":
    panel = pd.read_csv(PANEL_CACHE, parse_dates=["ds"])
    diario = panel[panel["par"] == PAR][["ds", "y"]].sort_values("ds").reset_index(drop=True)
    diario.insert(0, "unique_id", UNIQUE_ID)
    serie_nhits = diario[["unique_id", "ds", "y"]].dropna().reset_index(drop=True)
    print(f"Serie diaria ({PAR}): {len(serie_nhits)} obs, {serie_nhits['ds'].min().date()} a {serie_nhits['ds'].max().date()}")
    print(f"Walk-forward NHITS: {N_REFITS} refits x cada {REFIT_CADENCIA} dias (h={H}) = {N_REFITS*REFIT_CADENCIA} dias de decision (~{N_REFITS*REFIT_CADENCIA/252:.1f} anios)")

    print(f"\n[1/4] Forecast NHITS walk-forward (~10s/refit medido en CLP, ~{N_REFITS*10/60:.0f} min estimados)...")
    forecast_nhits = dataset26_mod.generar_forecast_nhits_diario(serie_nhits, N_REFITS, REFIT_CADENCIA, H)
    print(f"      {len(forecast_nhits)} dias de decision generados, {forecast_nhits['ds'].min().date()} a {forecast_nhits['ds'].max().date()}")

    print(f"\n[2/4] Volatilidad GARCH(1,1) por dia, horizonte=1 ({len(forecast_nhits)} fits)...")
    vol = dataset26_mod.generar_volatilidad_por_dia(diario, forecast_nhits["ds"])

    print(f"\n[3/4] Indicadores tecnicos (MACD, RSI, min/max de {MINMAX_VENTANA_DIARIA} dias)...")
    con_indicadores = dataset26_mod.calcular_indicadores_tecnicos_diarios(diario)

    print(f"\n[4/4] Features de cobre (copper_ret_1d, copper_mom_5d) - AUD tambien tiene al cobre como commodity fundamental (radar-baseline)...")
    macro = features_mod.cargar_macro()
    con_cobre = pd.merge_asof(con_indicadores.sort_values("ds"), macro[["ds", "copper"]], on="ds", direction="backward")
    con_cobre["copper_ret_1d"] = np.log(con_cobre["copper"] / con_cobre["copper"].shift(1))
    con_cobre["copper_mom_5d"] = (con_cobre["copper"] - con_cobre["copper"].shift(5)) / con_cobre["copper"].shift(5)

    dataset = forecast_nhits.merge(vol, on="ds", how="left")
    dataset = dataset.merge(con_cobre[["ds", "macd", "rsi", "precio_min_ventana", "precio_max_ventana", "copper_ret_1d", "copper_mom_5d"]], on="ds", how="left")
    dataset = dataset.sort_values("ds").reset_index(drop=True)
    dataset["y_next"] = dataset["y"].shift(-1)
    dataset = dataset.dropna().reset_index(drop=True)

    columnas = ["ds", "y", "y_next", "nhits_h1", "nhits_h2", "vol_garch", "macd", "rsi",
                "precio_min_ventana", "precio_max_ventana", "copper_ret_1d", "copper_mom_5d", "dias_desde_refit"]
    dataset = dataset[columnas]
    dataset.to_csv(f"{RESULTADOS_DIR}/dataset_entrenamiento_rl_diario_aud.csv", index=False)

    print(f"\n=== listo: {len(dataset)} dias, {dataset['ds'].min().date()} a {dataset['ds'].max().date()} ===")
    print(f"Distribucion de dias_desde_refit (staleness del forecast NHITS dentro de cada bloque de {REFIT_CADENCIA}):")
    print(dataset["dias_desde_refit"].value_counts().sort_index())
    print(dataset.head())
    print(dataset.tail())
