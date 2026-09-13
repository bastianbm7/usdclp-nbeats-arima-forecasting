# Descarga USD/CLP (ticker CLP=X) via yfinance, guarda copia cruda en datos/bases/,
# y transforma a formato largo unique_id/ds/y que esperan statsforecast/neuralforecast.
#
# Frecuencia: la serie viene en dias habiles (gaps de 1 dia entre semana, 3 dias en
# fin de semana, con algunos feriados sueltos) -> se usa freq="B" (business day) en
# los scripts de modelado, no "D" calendario completo.

import pandas as pd
import yfinance as yf

TICKER = "CLP=X"
UNIQUE_ID = "USDCLP"
FECHA_INICIO = "2010-01-01"
DATOS_BASES = "../datos/bases"


def descargar_serie(start=FECHA_INICIO):
    df = yf.download(TICKER, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel("Ticker")
    return df


def limpiar_ticks_erroneos(df, umbral=0.3):
    # yfinance a veces devuelve un tick corrupto de un solo dia que se revierte al
    # siguiente (visto en CLP=X: 2016-12-22 con y=5.00 en vez de ~660, y 2014-04-10
    # con y=5.46 en vez de ~544) - se detecta como una caida Y rebote > umbral contra
    # AMBOS vecinos (un movimiento real de un solo dia en USD/CLP no se acerca a eso)
    # y se interpola en vez de dejar pasar un retorno de +-488% a los modelos.
    y = df["y"]
    ratio_prev = y / y.shift(1)
    ratio_next = y / y.shift(-1)
    es_tick_malo = (ratio_prev < umbral) & (ratio_next < umbral)
    if es_tick_malo.any():
        fechas_malas = df.loc[es_tick_malo, "ds"].dt.date.tolist()
        print(f"Ticks corruptos detectados y corregidos (interpolados): {fechas_malas}")
        df.loc[es_tick_malo, "y"] = None
        df["y"] = df["y"].interpolate()
    return df


def transformar_a_formato_largo(df_crudo):
    df = df_crudo[["Close"]].reset_index()
    df.columns = ["ds", "y"]
    df.insert(0, "unique_id", UNIQUE_ID)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.dropna(subset=["y"]).sort_values("ds").reset_index(drop=True)
    df = limpiar_ticks_erroneos(df)
    return df


if __name__ == "__main__":
    crudo = descargar_serie()
    crudo.to_csv(f"{DATOS_BASES}/usdclp_raw.csv")
    print(f"Crudo: {crudo.shape[0]} filas, {crudo.index.min().date()} a {crudo.index.max().date()}")

    largo = transformar_a_formato_largo(crudo)
    largo.to_csv(f"{DATOS_BASES}/usdclp_long.csv", index=False)
    print(f"Formato largo: {largo.shape[0]} filas -> {DATOS_BASES}/usdclp_long.csv")
    print(largo.head())
    print(largo.tail())
