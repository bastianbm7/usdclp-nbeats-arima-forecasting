# Modelos Nixtla/neuralforecast: NBEATS y NHITS, mismo esquema de backtesting
# walk-forward que el baseline (mismas ventanas, mismo horizonte) para comparar
# apples-to-apples. loss=MQLoss() da los cuantiles para los intervalos de prediccion.

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.losses.pytorch import MQLoss

HORIZONTE = None  # igual al de 02_baseline_arima_naive.py
N_WINDOWS = 5


def correr_backtesting(df):
    raise NotImplementedError


if __name__ == "__main__":
    pass
