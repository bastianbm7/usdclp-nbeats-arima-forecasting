# Baseline clasico: AutoARIMA + Naive via statsforecast, con backtesting walk-forward
# (cross_validation, n_windows>1 - nunca un solo split train/test).

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, Naive

HORIZONTE = None  # varios pasos, no 1 solo (ver README - mitigacion de autocorrelacion)
N_WINDOWS = 5


def correr_backtesting(df):
    raise NotImplementedError


if __name__ == "__main__":
    pass
