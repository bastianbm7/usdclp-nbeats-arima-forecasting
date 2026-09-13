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


def transformar_a_formato_largo(df_crudo):
    df = df_crudo[["Close"]].reset_index()
    df.columns = ["ds", "y"]
    df.insert(0, "unique_id", UNIQUE_ID)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.dropna(subset=["y"]).sort_values("ds").reset_index(drop=True)
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
