# Genera el dataset semanal que va a alimentar al entorno de RL (Issue #1, Fase 1):
# para cada semana de decision junta el precio real, el forecast de N-HiTS
# walk-forward (refit=True, warm start - mismo mecanismo de 08_online_learning.py,
# pero con muchas mas ventanas porque acá hace falta una secuencia larga para
# entrenar al agente, no solo evaluarlo), el forecast de volatilidad GARCH(1,1)
# (ganador de 09_comparacion_modelos_volatilidad.py), e indicadores tecnicos
# calculados directo sobre el historico (sin fit: MACD, RSI, min/max de ventana).
#
# Unica pieza cara: N-HiTS (~10s/ventana medido). GARCH/MACD/RSI/min-max son
# casi gratis en comparacion (por eso no justifican reducir N_WINDOWS).
#
# h=2 en un solo fit da forecast de t+1 Y t+2 juntos - no hace falta entrenar
# por separado para cada horizonte.

import numpy as np
import pandas as pd
from arch import arch_model
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import NHITS

DATOS_LARGO = "../datos/bases/usdclp_long.csv"
RESULTADOS_DIR = "../datos/resultados"

N_WINDOWS = 350
INPUT_SIZE = 12
MAX_STEPS_REFIT = 100
FREQ = "W"
SCALER_TYPE = "standard"

MACD_RAPIDA, MACD_LENTA, MACD_SENAL = 12, 26, 9
RSI_VENTANA = 14
MINMAX_VENTANA = 12
MIN_OBS_GARCH = 500


def resamplear_semanal(df_diario):
    s = df_diario.set_index("ds")["y"].resample(FREQ).last()
    s = s.iloc[:-1]  # descarta el ultimo periodo, incompleto
    df = s.reset_index()
    df.insert(0, "unique_id", "USDCLP")
    return df.dropna().reset_index(drop=True)


def calcular_macd(precio, rapida=MACD_RAPIDA, lenta=MACD_LENTA, senal=MACD_SENAL):
    ema_rapida = precio.ewm(span=rapida, adjust=False).mean()
    ema_lenta = precio.ewm(span=lenta, adjust=False).mean()
    macd = ema_rapida - ema_lenta
    linea_senal = macd.ewm(span=senal, adjust=False).mean()
    return macd - linea_senal  # histograma, ya centrado en 0


def calcular_rsi(precio, ventana=RSI_VENTANA):
    delta = precio.diff()
    ganancia = delta.clip(lower=0).rolling(ventana).mean()
    perdida = (-delta.clip(upper=0)).rolling(ventana).mean()
    rs = ganancia / perdida.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calcular_indicadores_tecnicos(serie_semanal):
    df = serie_semanal.copy()
    df["macd"] = calcular_macd(df["y"])
    df["rsi"] = calcular_rsi(df["y"])
    df["precio_min_ventana"] = df["y"].rolling(MINMAX_VENTANA).min()
    df["precio_max_ventana"] = df["y"].rolling(MINMAX_VENTANA).max()
    return df


def forecast_volatilidad_garch(retornos_diarios_hist):
    am = arch_model(retornos_diarios_hist * 100, vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    pronostico = res.forecast(horizon=5, reindex=False)
    varianza_5d = pronostico.variance.values[-1].sum()
    return np.sqrt(varianza_5d) / 100


def generar_volatilidad_por_semana(df_diario, fechas_decision):
    diario = df_diario.sort_values("ds").reset_index(drop=True)
    diario["retorno_log"] = np.log(diario["y"] / diario["y"].shift(1))
    retornos = diario.set_index("ds")["retorno_log"].dropna()

    filas = []
    for fecha in fechas_decision:
        hist = retornos[retornos.index <= fecha]
        vol = forecast_volatilidad_garch(hist) if len(hist) >= MIN_OBS_GARCH else np.nan
        filas.append({"ds": fecha, "vol_garch": vol})
    return pd.DataFrame(filas)


def generar_forecast_nhits(serie_semanal, n_windows):
    trainer_kwargs = dict(enable_progress_bar=False, enable_model_summary=False, logger=False)
    modelo = NHITS(h=2, input_size=INPUT_SIZE, loss=MQLoss(), max_steps=MAX_STEPS_REFIT, scaler_type=SCALER_TYPE, **trainer_kwargs)
    nf = NeuralForecast(models=[modelo], freq=FREQ)
    cv = nf.cross_validation(df=serie_semanal, n_windows=n_windows, step_size=1, refit=True, use_init_models=False)

    cv["paso"] = cv.groupby("cutoff").cumcount() + 1
    pivot = cv.pivot(index="cutoff", columns="paso", values="NHITS-median")
    pivot = pivot.rename(columns={1: "nhits_h1", 2: "nhits_h2"}).reset_index()
    pivot = pivot.rename(columns={"cutoff": "ds"})
    return pivot[["ds", "nhits_h1", "nhits_h2"]]


if __name__ == "__main__":
    diario = pd.read_csv(DATOS_LARGO, parse_dates=["ds"])
    semanal = resamplear_semanal(diario)
    print(f"Serie semanal: {len(semanal)} obs, {semanal['ds'].min().date()} a {semanal['ds'].max().date()}")

    print(f"\n[1/3] Indicadores tecnicos (MACD, RSI, min/max) - deberia ser instantaneo...")
    con_indicadores = calcular_indicadores_tecnicos(semanal)

    print(f"[2/3] Forecast N-HiTS walk-forward, {N_WINDOWS} ventanas (~10s/ventana, ~{N_WINDOWS*10/60:.0f} min estimados)...")
    forecast_nhits = generar_forecast_nhits(semanal, N_WINDOWS)

    print(f"[3/3] Forecast de volatilidad GARCH(1,1), {len(forecast_nhits)} semanas (rapido, <1 min)...")
    fechas_decision = forecast_nhits["ds"]
    vol = generar_volatilidad_por_semana(diario, fechas_decision)

    dataset = forecast_nhits.merge(con_indicadores, on="ds", how="left").merge(vol, on="ds", how="left")
    dataset = dataset.sort_values("ds").reset_index(drop=True)
    dataset["y_next"] = dataset["y"].shift(-1)  # precio real de la semana siguiente, para calcular reward en el entorno
    dataset = dataset.dropna().reset_index(drop=True)

    columnas = ["unique_id", "ds", "y", "y_next", "nhits_h1", "nhits_h2", "vol_garch", "macd", "rsi", "precio_min_ventana", "precio_max_ventana"]
    dataset = dataset[columnas]
    dataset.to_csv(f"{RESULTADOS_DIR}/dataset_entrenamiento_rl.csv", index=False)

    print(f"\n=== listo: {len(dataset)} semanas, {dataset['ds'].min().date()} a {dataset['ds'].max().date()} ===")
    print(dataset.head())
    print(dataset.tail())
