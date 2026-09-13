# Descarga USD/CLP (ticker CLP=X) via yfinance, guarda copia cruda en datos/bases/,
# transforma a formato largo unique_id/ds/y que esperan statsforecast/neuralforecast,
# y separa el tramo final (test) que no se toca hasta la evaluacion final.

import yfinance as yf

TICKER = "CLP=X"
DATOS_BASES = "../datos/bases"


def descargar_serie():
    raise NotImplementedError


def transformar_a_formato_largo(df):
    raise NotImplementedError


def split_train_test(df, fraccion_test=0.2):
    raise NotImplementedError


if __name__ == "__main__":
    pass
