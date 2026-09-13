# Modelos Nixtla/neuralforecast: NBEATS y NHITS, mismo esquema de backtesting
# walk-forward que el baseline (mismas ventanas, mismo horizonte) para comparar
# apples-to-apples. loss=MQLoss() da los cuantiles para los intervalos de prediccion.

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import NBEATS, NHITS

DATOS_LARGO = "../datos/bases/usdclp_long.csv"
RESULTADOS_DIR = "../datos/resultados"

FREQ = "B"  # igual que 02_baseline_arima_naive.py
HORIZONTE = 14  # igual que el baseline
N_WINDOWS = 5
NIVELES = [80, 90]
INPUT_SIZE = 4 * HORIZONTE
MAX_STEPS = 300  # reducido de 1000 (default) para que corra en minutos en CPU


def correr_backtesting(df):
    models = [
        NBEATS(h=HORIZONTE, input_size=INPUT_SIZE, loss=MQLoss(), max_steps=MAX_STEPS),
        NHITS(h=HORIZONTE, input_size=INPUT_SIZE, loss=MQLoss(), max_steps=MAX_STEPS),
    ]
    nf = NeuralForecast(models=models, freq=FREQ)
    cv_df = nf.cross_validation(df=df, n_windows=N_WINDOWS, step_size=HORIZONTE, level=NIVELES)
    return cv_df


if __name__ == "__main__":
    df = pd.read_csv(DATOS_LARGO, parse_dates=["ds"])
    cv_df = correr_backtesting(df)
    cv_df.to_csv(f"{RESULTADOS_DIR}/cv_nbeats_nhits.csv", index=False)
    print(cv_df.head())
    print(f"Ventanas: {cv_df['cutoff'].nunique()}, filas totales: {len(cv_df)}")
