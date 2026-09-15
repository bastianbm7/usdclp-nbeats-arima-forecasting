# Issue #5, paso 2 (reconstruccion RL diaria) - dataset walk-forward diario,
# mismo patron que 10_generar_dataset_rl.py (semanal) pero a frecuencia diaria,
# sumando copper_ret_1d/copper_mom_5d (el hallazgo de 9.13-9.14 del paper).
#
# DECISION DE DISENO IMPORTANTE (medido, no asumido - el coordinador señalo
# este riesgo antes de que se lanzara este script y tenia razon): refit=True
# de NHITS en CADA dia (step_size=1, igual que 10 hace semana a semana) se
# midio empiricamente en ~8.9s/ventana - igual de rapido por ventana que el
# semanal, pero a diario hay ~12x mas puntos de decision (4346 filas vs 349),
# asi que refit diario estricto sobre toda la historia habria tardado ~10.7h
# (medido con una muestra de 15 ventanas y extrapolado). Se opto por refitear
# cada REFIT_CADENCIA=5 dias habiles en vez de cada dia (h=7 en vez de h=2,
# para tener horizonte suficiente de cubrir el h2 del ultimo dia del bloque),
# reutilizando el mismo fit para las 5 decisiones diarias siguientes - medido
# en ~9.8s/refit x 5 dias cubiertos = ~2s/dia efectivo, ~5x mas barato que
# refit diario estricto. El costo real: la senal de forecast dentro de cada
# bloque de 5 dias queda "stale" hasta 4 dias (se calculo con info de hasta
# 4 dias antes de la decision, no del dia anterior como seria un walk-forward
# estrictamente diario) - trade-off explicito, no un default oculto. Con
# N_REFITS=300 (1500 dias de decision, ~6 anios, mismo orden de magnitud
# temporal que las 350 semanas ~6.7 anios del dataset semanal) el costo total
# medido/estimado de la parte NHITS es ~49 min.
#
# GARCH y los indicadores tecnicos (MACD/RSI/minmax) SI se recalculan cada dia
# (no cada 5) - son baratos (segundos por fit, no genera el mismo problema de
# escala que NHITS) y no hay razon para degradar su frescura.

import numpy as np
import pandas as pd
from arch import arch_model
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import NHITS

import importlib

generar_mod = importlib.import_module("10_generar_dataset_rl")  # reusa calcular_macd/calcular_rsi (parametros 12/26/9 y 14 - convencion ESTANDAR diaria de TA, aca se reusan tal cual, sin retunear)
features_mod = importlib.import_module("19_features_nuevas_validacion")  # reusa cargar_macro() (cache de cobre + tasas)
kelly_diario_mod = importlib.import_module("22_kelly_diario_cobre")  # reusa las formulas de copper_ret_1d/copper_mom_5d, ya validadas en 9.13-9.14

DATOS_LARGO = "../datos/bases/usdclp_long.csv"
RESULTADOS_DIR = "../datos/resultados"

FREQ = "B"
INPUT_SIZE = 30  # mismo valor que la escala "diaria" de 08_online_learning.py
REFIT_CADENCIA = 5  # dias habiles entre cada refit de NHITS - ver nota de diseno arriba
H = REFIT_CADENCIA + 2  # +2 (no +1): el ULTIMO dia del bloque (j=5) necesita h_{j+1}=h6 Y h_{j+2}=h7 para su propio nhits_h1/h2
MAX_STEPS_REFIT = 100
SCALER_TYPE = "standard"
N_REFITS = 300  # 300 x 5 = 1500 dias de decision (~6 anios) - ver nota de diseno

MINMAX_VENTANA_DIARIA = 60  # ~1 trimestre habil, analogo en escala calendario a la ventana de 12 semanas (~84 dias) del agente semanal
MIN_OBS_GARCH = 500


def generar_forecast_nhits_diario(serie_diaria, n_refits, refit_cadencia, h):
    trainer_kwargs = dict(enable_progress_bar=False, enable_model_summary=False, logger=False)
    modelo = NHITS(h=h, input_size=INPUT_SIZE, loss=MQLoss(), max_steps=MAX_STEPS_REFIT, scaler_type=SCALER_TYPE, **trainer_kwargs)
    nf = NeuralForecast(models=[modelo], freq=FREQ)
    cv = nf.cross_validation(df=serie_diaria, n_windows=n_refits, step_size=refit_cadencia, refit=True, use_init_models=False)
    cv = cv.sort_values(["cutoff", "ds"]).reset_index(drop=True)

    filas = []
    for cutoff, grupo in cv.groupby("cutoff", sort=False):
        grupo = grupo.sort_values("ds").reset_index(drop=True)
        if len(grupo) < refit_cadencia + 2:
            continue  # bloque incompleto al final de la serie (menos de h filas disponibles) - se descarta, no se rellena con datos parciales
        for j in range(1, refit_cadencia + 1):  # j = 1..5, dia de decision dentro del bloque
            fila_hoy = grupo.iloc[j - 1]  # y REAL en la fecha de decision (ya conocido, no hay look-ahead: es el precio de HOY)
            fila_h1 = grupo.iloc[j]       # forecast para MAÑANA, calculado en el refit de hace (j) dias
            fila_h2 = grupo.iloc[j + 1]   # forecast para PASADO MAÑANA, mismo refit
            filas.append({
                "ds": fila_hoy["ds"], "y": fila_hoy["y"], "nhits_h1": fila_h1["NHITS-median"],
                "nhits_h2": fila_h2["NHITS-median"], "dias_desde_refit": j,
            })
    return pd.DataFrame(filas).sort_values("ds").reset_index(drop=True)


def forecast_volatilidad_garch_diario(retornos_diarios_hist):
    # Analogo a forecast_volatilidad_garch() de 10_generar_dataset_rl.py, pero
    # horizonte=1 dia (no 5) - la posicion diaria se sostiene de un cierre al
    # siguiente, no una semana, asi que el horizonte de riesgo relevante para
    # el stop-loss es de 1 dia, no 5.
    am = arch_model(retornos_diarios_hist * 100, vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    pronostico = res.forecast(horizon=1, reindex=False)
    varianza_1d = pronostico.variance.values[-1, 0]
    return np.sqrt(varianza_1d) / 100


def generar_volatilidad_por_dia(diario, fechas_decision):
    diario = diario.sort_values("ds").reset_index(drop=True)
    diario["retorno_log"] = np.log(diario["y"] / diario["y"].shift(1))
    retornos = diario.set_index("ds")["retorno_log"].dropna()

    filas = []
    for fecha in fechas_decision:
        hist = retornos[retornos.index <= fecha]
        vol = forecast_volatilidad_garch_diario(hist) if len(hist) >= MIN_OBS_GARCH else np.nan
        filas.append({"ds": fecha, "vol_garch": vol})
    return pd.DataFrame(filas)


def calcular_indicadores_tecnicos_diarios(diario):
    df = diario.copy()
    df["macd"] = generar_mod.calcular_macd(df["y"])
    df["rsi"] = generar_mod.calcular_rsi(df["y"])
    df["precio_min_ventana"] = df["y"].rolling(MINMAX_VENTANA_DIARIA).min()
    df["precio_max_ventana"] = df["y"].rolling(MINMAX_VENTANA_DIARIA).max()
    return df


if __name__ == "__main__":
    diario = pd.read_csv(DATOS_LARGO, parse_dates=["ds"])
    if "unique_id" not in diario.columns:
        diario.insert(0, "unique_id", "USDCLP")
    serie_nhits = diario[["unique_id", "ds", "y"]].dropna().reset_index(drop=True)
    print(f"Serie diaria: {len(serie_nhits)} obs, {serie_nhits['ds'].min().date()} a {serie_nhits['ds'].max().date()}")
    print(f"Walk-forward NHITS: {N_REFITS} refits x cada {REFIT_CADENCIA} dias (h={H}) = {N_REFITS*REFIT_CADENCIA} dias de decision (~{N_REFITS*REFIT_CADENCIA/252:.1f} anios)")

    print(f"\n[1/4] Forecast NHITS walk-forward (~10s/refit medido, ~{N_REFITS*10/60:.0f} min estimados)...")
    forecast_nhits = generar_forecast_nhits_diario(serie_nhits, N_REFITS, REFIT_CADENCIA, H)
    print(f"      {len(forecast_nhits)} dias de decision generados, {forecast_nhits['ds'].min().date()} a {forecast_nhits['ds'].max().date()}")

    print(f"\n[2/4] Volatilidad GARCH(1,1) por dia, horizonte=1 ({len(forecast_nhits)} fits, rapido)...")
    vol = generar_volatilidad_por_dia(diario, forecast_nhits["ds"])

    print(f"\n[3/4] Indicadores tecnicos (MACD, RSI, min/max de {MINMAX_VENTANA_DIARIA} dias) - deberia ser instantaneo...")
    con_indicadores = calcular_indicadores_tecnicos_diarios(diario)

    print(f"\n[4/4] Features de cobre (copper_ret_1d, copper_mom_5d, ver 22_kelly_diario_cobre.py)...")
    macro = features_mod.cargar_macro()
    con_cobre = pd.merge_asof(con_indicadores.sort_values("ds"), macro[["ds", "copper"]], on="ds", direction="backward")
    con_cobre["copper_ret_1d"] = np.log(con_cobre["copper"] / con_cobre["copper"].shift(1))
    con_cobre["copper_mom_5d"] = (con_cobre["copper"] - con_cobre["copper"].shift(5)) / con_cobre["copper"].shift(5)

    dataset = forecast_nhits.merge(vol, on="ds", how="left")
    dataset = dataset.merge(con_cobre[["ds", "macd", "rsi", "precio_min_ventana", "precio_max_ventana", "copper_ret_1d", "copper_mom_5d"]], on="ds", how="left")
    dataset = dataset.sort_values("ds").reset_index(drop=True)
    dataset["y_next"] = dataset["y"].shift(-1)  # precio real del dia siguiente, para reward - mismo patron que el dataset semanal
    dataset = dataset.dropna().reset_index(drop=True)

    columnas = ["ds", "y", "y_next", "nhits_h1", "nhits_h2", "vol_garch", "macd", "rsi",
                "precio_min_ventana", "precio_max_ventana", "copper_ret_1d", "copper_mom_5d", "dias_desde_refit"]
    dataset = dataset[columnas]
    dataset.to_csv(f"{RESULTADOS_DIR}/dataset_entrenamiento_rl_diario.csv", index=False)

    print(f"\n=== listo: {len(dataset)} dias, {dataset['ds'].min().date()} a {dataset['ds'].max().date()} ===")
    print(f"Distribucion de dias_desde_refit (staleness del forecast NHITS dentro de cada bloque de {REFIT_CADENCIA}):")
    print(dataset["dias_desde_refit"].value_counts().sort_index())
    print(dataset.head())
    print(dataset.tail())
