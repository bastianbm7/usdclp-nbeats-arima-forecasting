# Baseline clasico: AutoARIMA + Naive via statsforecast, con backtesting walk-forward
# (cross_validation, n_windows>1 - nunca un solo split train/test).

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, Naive

DATOS_LARGO = "../datos/bases/usdclp_long.csv"
RESULTADOS_DIR = "../datos/resultados"

FREQ = "B"  # dias habiles
HORIZONTE = 14
N_WINDOWS = 5
NIVELES = [80, 90]


def correr_backtesting(df):
    sf = StatsForecast(models=[AutoARIMA(season_length=1), Naive()], freq=FREQ, n_jobs=-1)
    cv_df = sf.cross_validation(
        df=df, h=HORIZONTE, n_windows=N_WINDOWS, step_size=HORIZONTE, level=NIVELES
    )
    return cv_df


if __name__ == "__main__":
    df = pd.read_csv(DATOS_LARGO, parse_dates=["ds"])
    cv_df = correr_backtesting(df)
    cv_df.to_csv(f"{RESULTADOS_DIR}/cv_baseline.csv", index=False)
    print(cv_df.head())
    print(f"Ventanas: {cv_df['cutoff'].nunique()}, filas totales: {len(cv_df)}")
