# Extension de 26_generar_dataset_rl_diario.py para agregar nhits_h3 (forecast
# a 3 dias) - motivado por la idea de Bastian de elegir el take-profit segun
# que tan consistente/lejos llega la tendencia que ve NHITS (h1 vs h2 vs h3),
# en vez de apuntar siempre a h1 como hace el agente de 9.17-9.20.
#
# Archivo NUEVO en vez de modificar 26 in-place: NHITS no tiene semilla fija
# en ese script, asi que re-correr el walk-forward completo daria valores de
# h1/h2 LIGERAMENTE distintos a los ya versionados en
# dataset_entrenamiento_rl_diario.csv (el que usan 27/32/33 y todo lo
# documentado en las secciones 9.17-9.20 del paper) - sobreescribirlo
# invalidaria silenciosamente la reproducibilidad de esos resultados. Este
# script genera un archivo separado (dataset_entrenamiento_rl_diario_h3.csv).
#
# Unico cambio real sobre 26: H = REFIT_CADENCIA + 3 (no +2) - el ultimo dia
# de decision de cada bloque (j=5) necesita h_{j+1}=h6, h_{j+2}=h7 Y ahora
# h_{j+3}=h8 para su propio nhits_h1/h2/h3. Para j=1..4 (4 de cada 5 dias de
# decision), h3 ya estaba disponible dentro del H=7 original y se descartaba
# sin guardar - la unica razon real para recalcular todo es cubrir tambien
# j=5. El costo marginal de H=8 vs H=7 es chico (un paso mas de forecast por
# refit), pero el costo total sigue siendo el mismo orden de magnitud que 26
# (~50-60 min) porque no se puede reusar el forecast crudo de esa corrida -
# nunca se guardo, solo el dataset final con h1/h2 ya recortados.

import numpy as np
import pandas as pd
from arch import arch_model
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import NHITS

import importlib

generar_mod = importlib.import_module("10_generar_dataset_rl")
features_mod = importlib.import_module("19_features_nuevas_validacion")
kelly_diario_mod = importlib.import_module("22_kelly_diario_cobre")
dataset26_mod = importlib.import_module("26_generar_dataset_rl_diario")  # reusa forecast_volatilidad_garch_diario, generar_volatilidad_por_dia, calcular_indicadores_tecnicos_diarios (sin cambios)

DATOS_LARGO = "../datos/bases/usdclp_long.csv"
RESULTADOS_DIR = "../datos/resultados"

FREQ = dataset26_mod.FREQ
INPUT_SIZE = dataset26_mod.INPUT_SIZE
REFIT_CADENCIA = dataset26_mod.REFIT_CADENCIA
H = REFIT_CADENCIA + 3  # +3 (no +2 como en 26): el ULTIMO dia del bloque (j=5) ahora tambien necesita h_{j+3}=h8 para su propio nhits_h3
MAX_STEPS_REFIT = dataset26_mod.MAX_STEPS_REFIT
SCALER_TYPE = dataset26_mod.SCALER_TYPE
N_REFITS = dataset26_mod.N_REFITS

MINMAX_VENTANA_DIARIA = dataset26_mod.MINMAX_VENTANA_DIARIA
MIN_OBS_GARCH = dataset26_mod.MIN_OBS_GARCH


def generar_forecast_nhits_diario_h3(serie_diaria, n_refits, refit_cadencia, h):
    trainer_kwargs = dict(enable_progress_bar=False, enable_model_summary=False, logger=False)
    modelo = NHITS(h=h, input_size=INPUT_SIZE, loss=MQLoss(), max_steps=MAX_STEPS_REFIT, scaler_type=SCALER_TYPE, **trainer_kwargs)
    nf = NeuralForecast(models=[modelo], freq=FREQ)
    cv = nf.cross_validation(df=serie_diaria, n_windows=n_refits, step_size=refit_cadencia, refit=True, use_init_models=False)
    cv = cv.sort_values(["cutoff", "ds"]).reset_index(drop=True)

    filas = []
    for cutoff, grupo in cv.groupby("cutoff", sort=False):
        grupo = grupo.sort_values("ds").reset_index(drop=True)
        if len(grupo) < refit_cadencia + 3:  # +3 (no +2 como en 26) - ver nota de diseno arriba
            continue
        for j in range(1, refit_cadencia + 1):
            fila_hoy = grupo.iloc[j - 1]
            fila_h1 = grupo.iloc[j]
            fila_h2 = grupo.iloc[j + 1]
            fila_h3 = grupo.iloc[j + 2]
            filas.append({
                "ds": fila_hoy["ds"], "y": fila_hoy["y"], "nhits_h1": fila_h1["NHITS-median"],
                "nhits_h2": fila_h2["NHITS-median"], "nhits_h3": fila_h3["NHITS-median"], "dias_desde_refit": j,
            })
    return pd.DataFrame(filas).sort_values("ds").reset_index(drop=True)


if __name__ == "__main__":
    diario = pd.read_csv(DATOS_LARGO, parse_dates=["ds"])
    if "unique_id" not in diario.columns:
        diario.insert(0, "unique_id", "USDCLP")
    serie_nhits = diario[["unique_id", "ds", "y"]].dropna().reset_index(drop=True)
    print(f"Serie diaria: {len(serie_nhits)} obs, {serie_nhits['ds'].min().date()} a {serie_nhits['ds'].max().date()}")
    print(f"Walk-forward NHITS (con h3): {N_REFITS} refits x cada {REFIT_CADENCIA} dias (h={H}) = {N_REFITS*REFIT_CADENCIA} dias de decision (~{N_REFITS*REFIT_CADENCIA/252:.1f} anios)")

    print(f"\n[1/4] Forecast NHITS walk-forward con h3 (~10-12s/refit estimado, ~{N_REFITS*11/60:.0f} min estimados)...")
    forecast_nhits = generar_forecast_nhits_diario_h3(serie_nhits, N_REFITS, REFIT_CADENCIA, H)
    print(f"      {len(forecast_nhits)} dias de decision generados, {forecast_nhits['ds'].min().date()} a {forecast_nhits['ds'].max().date()}")

    print(f"\n[2/4] Volatilidad GARCH(1,1) por dia, horizonte=1 ({len(forecast_nhits)} fits, rapido)...")
    vol = dataset26_mod.generar_volatilidad_por_dia(diario, forecast_nhits["ds"])

    print(f"\n[3/4] Indicadores tecnicos (MACD, RSI, min/max de {MINMAX_VENTANA_DIARIA} dias) - deberia ser instantaneo...")
    con_indicadores = dataset26_mod.calcular_indicadores_tecnicos_diarios(diario)

    print(f"\n[4/4] Features de cobre (copper_ret_1d, copper_mom_5d, ver 22_kelly_diario_cobre.py)...")
    macro = features_mod.cargar_macro()
    con_cobre = pd.merge_asof(con_indicadores.sort_values("ds"), macro[["ds", "copper"]], on="ds", direction="backward")
    con_cobre["copper_ret_1d"] = np.log(con_cobre["copper"] / con_cobre["copper"].shift(1))
    con_cobre["copper_mom_5d"] = (con_cobre["copper"] - con_cobre["copper"].shift(5)) / con_cobre["copper"].shift(5)

    dataset = forecast_nhits.merge(vol, on="ds", how="left")
    dataset = dataset.merge(con_cobre[["ds", "macd", "rsi", "precio_min_ventana", "precio_max_ventana", "copper_ret_1d", "copper_mom_5d"]], on="ds", how="left")
    dataset = dataset.sort_values("ds").reset_index(drop=True)
    dataset["y_next"] = dataset["y"].shift(-1)
    dataset = dataset.dropna().reset_index(drop=True)

    columnas = ["ds", "y", "y_next", "nhits_h1", "nhits_h2", "nhits_h3", "vol_garch", "macd", "rsi",
                "precio_min_ventana", "precio_max_ventana", "copper_ret_1d", "copper_mom_5d", "dias_desde_refit"]
    dataset = dataset[columnas]
    dataset.to_csv(f"{RESULTADOS_DIR}/dataset_entrenamiento_rl_diario_h3.csv", index=False)

    print(f"\n=== listo: {len(dataset)} dias, {dataset['ds'].min().date()} a {dataset['ds'].max().date()} ===")
    print(f"Distribucion de dias_desde_refit (staleness del forecast NHITS dentro de cada bloque de {REFIT_CADENCIA}):")
    print(dataset["dias_desde_refit"].value_counts().sort_index())
    print(dataset.head())
    print(dataset.tail())
