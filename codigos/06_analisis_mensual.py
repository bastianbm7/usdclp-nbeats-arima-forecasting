# Mismo analisis que 02-05 (Naive/AutoARIMA vs. N-BEATS/N-HiTS, backtesting
# walk-forward) pero resampleando USD/CLP a frecuencia mensual (cierre de mes),
# para ver si la historia cambia a una escala de tiempo distinta a la diaria.

import matplotlib.pyplot as plt
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import NBEATS, NHITS
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, Naive
from utilsforecast.losses import mae, mape, rmse

DATOS_LARGO = "../datos/bases/usdclp_long.csv"
RESULTADOS_DIR = "../datos/resultados"

FREQ = "ME"
HORIZONTE = 6
N_WINDOWS = 5
NIVELES = [80, 90]
INPUT_SIZE = 4 * HORIZONTE
MAX_STEPS = 1500
SCALER_TYPE = "standard"

COLUMNA_PUNTUAL = {
    "Naive": "Naive",
    "AutoARIMA": "AutoARIMA",
    "NBEATS": "NBEATS-median",
    "NHITS": "NHITS-median",
}


def resamplear_mensual(df_diario):
    s = df_diario.set_index("ds")["y"].resample(FREQ).last()
    s = s.iloc[:-1]  # descarta el ultimo mes, incompleto (yfinance corta a mitad de mes)
    df = s.reset_index()
    df.insert(0, "unique_id", "USDCLP")
    return df.dropna().reset_index(drop=True)


def correr_baseline(df):
    sf = StatsForecast(models=[AutoARIMA(season_length=12), Naive()], freq=FREQ, n_jobs=-1)
    return sf.cross_validation(df=df, h=HORIZONTE, n_windows=N_WINDOWS, step_size=HORIZONTE, level=NIVELES)


def correr_nn(df):
    models = [
        NBEATS(h=HORIZONTE, input_size=INPUT_SIZE, loss=MQLoss(), max_steps=MAX_STEPS, scaler_type=SCALER_TYPE),
        NHITS(h=HORIZONTE, input_size=INPUT_SIZE, loss=MQLoss(), max_steps=MAX_STEPS, scaler_type=SCALER_TYPE),
    ]
    nf = NeuralForecast(models=models, freq=FREQ)
    return nf.cross_validation(df=df, n_windows=N_WINDOWS, step_size=HORIZONTE, level=NIVELES)


def calcular_metricas(cv_df):
    columnas = list(COLUMNA_PUNTUAL.values())
    tablas = {"RMSE": rmse(cv_df, columnas), "MAE": mae(cv_df, columnas), "MAPE": mape(cv_df, columnas)}
    filas = []
    for nombre_modelo, columna in COLUMNA_PUNTUAL.items():
        fila = {"modelo": nombre_modelo}
        for metrica, tabla in tablas.items():
            fila[metrica] = tabla[columna].iloc[0]
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    mejor_baseline = tabla.loc[tabla["modelo"].isin(["Naive", "AutoARIMA"]), "RMSE"].min()
    tabla["mejora_%_vs_mejor_baseline_RMSE"] = (mejor_baseline - tabla["RMSE"]) / mejor_baseline * 100
    return tabla.sort_values("RMSE").reset_index(drop=True)


def graficar(cv_df, metricas):
    ganador = metricas.iloc[0]["modelo"]
    columna_ganador = COLUMNA_PUNTUAL[ganador]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cv_df["ds"], cv_df["y"], label="USD/CLP real (mensual)", color="black", linewidth=2)
    ax.plot(cv_df["ds"], cv_df["Naive"], label="Naive", linestyle="--", color="gray")
    ax.plot(cv_df["ds"], cv_df["AutoARIMA"], label="AutoARIMA", linestyle="--", color="tab:orange")
    for modelo, columna in [("NBEATS", "NBEATS-median"), ("NHITS", "NHITS-median")]:
        if columna != columna_ganador:
            ax.plot(cv_df["ds"], cv_df[columna], label=modelo, linestyle="-.", color="tab:green", alpha=0.6)
    ax.plot(cv_df["ds"], cv_df[columna_ganador], label=f"{ganador} (mejor RMSE)", color="tab:blue", linewidth=2)
    ax.fill_between(
        cv_df["ds"], cv_df[f"{ganador}-lo-80"], cv_df[f"{ganador}-hi-80"],
        color="tab:blue", alpha=0.15, label=f"{ganador} intervalo 80%",
    )
    ax.set_title(f"USD/CLP mensual: prediccion vs. real ({N_WINDOWS} ventanas x {HORIZONTE} meses)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("CLP por USD")
    ax.legend(loc="best", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{RESULTADOS_DIR}/prediccion_vs_real_mensual.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    colores = ["tab:blue" if m == ganador else "tab:gray" for m in metricas["modelo"]]
    ax.bar(metricas["modelo"], metricas["RMSE"], color=colores)
    ax.set_ylabel(f"RMSE (backtesting, {N_WINDOWS} ventanas x {HORIZONTE} meses)")
    ax.set_title("USD/CLP mensual: RMSE por modelo")
    fig.tight_layout()
    fig.savefig(f"{RESULTADOS_DIR}/rmse_comparacion_mensual.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    diario = pd.read_csv(DATOS_LARGO, parse_dates=["ds"])
    mensual = resamplear_mensual(diario)
    print(f"Serie mensual: {len(mensual)} observaciones, {mensual['ds'].min().date()} a {mensual['ds'].max().date()}")

    cv_baseline = correr_baseline(mensual)
    cv_nn = correr_nn(mensual)
    claves = ["unique_id", "ds", "cutoff", "y"]
    cv_df = cv_baseline.merge(cv_nn, on=claves, how="inner")
    cv_df.to_csv(f"{RESULTADOS_DIR}/cv_mensual.csv", index=False)

    metricas = calcular_metricas(cv_df)
    metricas.to_csv(f"{RESULTADOS_DIR}/metricas_comparacion_mensual.csv", index=False)
    print(metricas.to_string(index=False))

    graficar(cv_df, metricas)
    print(f"Graficos guardados en {RESULTADOS_DIR}/")
