# Tarea de Notion "Probar medias moviles (SMA/EMA) a distintos periodos como
# features nuevas del agente RL" - continua la linea "buscar mejor senal" de
# 19_features_nuevas_validacion.py, despues de que MACD, RSI y momentum
# (4/12 semanas) dieran |r| entre 0.005 y 0.042 con el retorno futuro semanal,
# muy por debajo del umbral |r|=0.11 de la seccion 9.5. Prueba SMA/EMA a varios
# periodos (5/10/20/50 semanas) mas Bollinger Bands/CCI/ADX, senalados en la
# ronda 1 del radar-baseline (via FinRL/Qlib Alpha158) como indicadores nunca
# probados en este proyecto.
#
# CCI y ADX necesitan High/Low, que 01_obtener_datos.py descarta (solo guarda
# Close, linea 43-44) - en vez de tocar el pipeline principal, se descarga aca
# un OHLC semanal separado con su propio cache, mismo patron que
# descargar_macro()/cargar_macro() de 19_features_nuevas_validacion.py.
#
# Expectativa honesta (de la propia Tarea de Notion): baja probabilidad de
# encontrar senal nueva - son transformaciones del mismo precio que ya fallo
# en sus otras formas (MACD, RSI, momentum). Se prueba igual porque es barato
# (sin entrenar ningun agente) y cierra formalmente esta linea en vez de
# dejarla como intuicion sin probar.

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

entorno_mod = importlib.import_module("11_entorno_trading_rl")

RESULTADOS_DIR = "../datos/resultados"
BASES_DIR = "../datos/bases"
OHLC_CACHE = f"{BASES_DIR}/usdclp_ohlc_semanal.csv"

TICKER = "CLP=X"
SMA_EMA_PERIODOS = [5, 10, 20, 50]
BB_VENTANA, BB_NUM_STD = 20, 2
CCI_VENTANA = 20
ADX_VENTANA = 14


def descargar_ohlc_semanal():
    df = yf.download(TICKER, start="2009-01-01", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel("Ticker")
    df.index = pd.to_datetime(df.index).tz_localize(None)

    semanal = df[["High", "Low", "Close"]].resample("W").agg({"High": "max", "Low": "min", "Close": "last"})
    semanal = semanal.iloc[:-1]  # descarta la ultima semana incompleta, mismo criterio que resamplear_semanal() de 10_generar_dataset_rl.py
    semanal = semanal.reset_index().rename(columns={"Date": "ds", "High": "high", "Low": "low", "Close": "close"})
    semanal = semanal.dropna().reset_index(drop=True)
    semanal.to_csv(OHLC_CACHE, index=False)
    return semanal


def cargar_ohlc_semanal():
    try:
        ohlc = pd.read_csv(OHLC_CACHE, parse_dates=["ds"])
        print(f"OHLC semanal: usando cache {OHLC_CACHE} ({len(ohlc)} filas, hasta {ohlc['ds'].max().date()})")
        return ohlc
    except FileNotFoundError:
        print("OHLC semanal: sin cache, descargando (yfinance)...")
        return descargar_ohlc_semanal()


def calcular_cci(high, low, close, ventana=CCI_VENTANA):
    precio_tipico = (high + low + close) / 3
    media = precio_tipico.rolling(ventana).mean()
    desviacion_media = precio_tipico.rolling(ventana).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (precio_tipico - media) / (0.015 * desviacion_media.replace(0, np.nan))


def calcular_adx(high, low, close, ventana=ADX_VENTANA):
    # Wilder (1978), suavizado via ewm(alpha=1/ventana) como aproximacion causal
    # estandar del suavizado original (equivalente a rolling con "warm-up").
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    rango_verdadero = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = rango_verdadero.ewm(alpha=1 / ventana, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / ventana, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / ventana, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / ventana, adjust=False).mean()


def agregar_features_nuevas(df, ohlc):
    df = df.sort_values("ds").reset_index(drop=True)
    df = pd.merge_asof(df, ohlc[["ds", "high", "low", "close"]], on="ds", direction="backward")

    for p in SMA_EMA_PERIODOS:
        df[f"sma_{p}_rel"] = (df["y"] - df["y"].rolling(p).mean()) / df["y"]
        df[f"ema_{p}_rel"] = (df["y"] - df["y"].ewm(span=p, adjust=False).mean()) / df["y"]

    media_bb = df["y"].rolling(BB_VENTANA).mean()
    std_bb = df["y"].rolling(BB_VENTANA).std()
    banda_sup = media_bb + BB_NUM_STD * std_bb
    banda_inf = media_bb - BB_NUM_STD * std_bb
    df["bb_pctb"] = (df["y"] - banda_inf) / (banda_sup - banda_inf).replace(0, np.nan)
    df["bb_width"] = (banda_sup - banda_inf) / media_bb

    df["cci"] = calcular_cci(df["high"], df["low"], df["close"]) / 100  # /100: misma escala aprox que el resto de las features (acotadas)
    df["adx"] = calcular_adx(df["high"], df["low"], df["close"]) / 100

    return df.dropna().reset_index(drop=True)


FEATURES_NUEVAS = (
    [f"sma_{p}_rel" for p in SMA_EMA_PERIODOS]
    + [f"ema_{p}_rel" for p in SMA_EMA_PERIODOS]
    + ["bb_pctb", "bb_width", "cci", "adx"]
)


def calcular_correlaciones(df):
    retorno_futuro = (df["y_next"] - df["y"]) / df["y"]
    filas = []
    for feature in FEATURES_NUEVAS:
        corr = df[feature].corr(retorno_futuro)
        acierto = (np.sign(df[feature]) == np.sign(retorno_futuro)).mean()
        filas.append({"feature": feature, "correlacion_con_retorno_futuro": corr, "acierto_direccion_%": 100 * acierto})
    return pd.DataFrame(filas).sort_values("correlacion_con_retorno_futuro", key=abs, ascending=False).reset_index(drop=True)


def graficar_correlaciones(tabla_nuevas, path_salida):
    tabla_7 = pd.read_csv(f"{RESULTADOS_DIR}/analisis_features_correlacion.csv")
    tabla_radar1 = pd.read_csv(f"{RESULTADOS_DIR}/analisis_features_nuevas_correlacion.csv")
    combinado = pd.concat([
        tabla_7[["feature", "correlacion_con_retorno_futuro"]].assign(grupo="7 originales (seccion 9.5)"),
        tabla_radar1[["feature", "correlacion_con_retorno_futuro"]].assign(grupo="tasas/cobre/momentum (radar-baseline)"),
        tabla_nuevas[["feature", "correlacion_con_retorno_futuro"]].assign(grupo="SMA/EMA/Bollinger/CCI/ADX (esta tarea)"),
    ]).sort_values("correlacion_con_retorno_futuro", key=abs)

    color_por_grupo = {
        "7 originales (seccion 9.5)": "steelblue",
        "tasas/cobre/momentum (radar-baseline)": "darkorange",
        "SMA/EMA/Bollinger/CCI/ADX (esta tarea)": "seagreen",
    }
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(combinado["feature"], combinado["correlacion_con_retorno_futuro"], color=[color_por_grupo[g] for g in combinado["grupo"]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.11, color="gray", linewidth=0.6, linestyle="--")
    ax.axvline(-0.11, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Correlacion con el retorno semanal siguiente (y_next vs y)")
    ax.set_title("Todas las features probadas hasta ahora vs. el umbral |r|=0.11\nazul=originales, naranja=radar-baseline ronda 1, verde=SMA/EMA/Bollinger/CCI/ADX")
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ohlc = cargar_ohlc_semanal()
    df_base = entorno_mod.cargar_dataset()
    df = agregar_features_nuevas(df_base, ohlc)
    print(f"\nDataset con features nuevas: {len(df)} semanas ({len(df_base) - len(df)} perdidas por SMA/EMA/BB de 50 semanas + merge OHLC)\n")

    tabla_corr = calcular_correlaciones(df)
    tabla_corr.to_csv(f"{RESULTADOS_DIR}/analisis_features_sma_ema_correlacion.csv", index=False)
    graficar_correlaciones(tabla_corr, f"{RESULTADOS_DIR}/analisis_features_sma_ema_correlacion.png")
    print("=== Correlacion con el retorno futuro (SMA/EMA/Bollinger/CCI/ADX) ===")
    print(tabla_corr.to_string(index=False))

    umbral = 0.11
    supera_umbral = tabla_corr[tabla_corr["correlacion_con_retorno_futuro"].abs() > umbral]
    if len(supera_umbral) > 0:
        print(f"\n{len(supera_umbral)} feature(s) superan |r|={umbral} - agregar al estado del PPO (patron de 20_agente_rl_ronda2.py) en un paso siguiente:")
        print(supera_umbral.to_string(index=False))
    else:
        print(f"\nNinguna feature supera |r|={umbral} - misma conclusion que MACD/RSI/momentum en 19_features_nuevas_validacion.py.")
        print("No se justifica el costo de entrenar el PPO con estas features (paso 2 de la Tarea, condicional a superar el umbral).")
