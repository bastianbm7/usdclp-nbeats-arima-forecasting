# Mismo tipo de comparacion que 02/06 pero a escala anual (cierre de diciembre).
#
# A proposito NO se incluyen N-BEATS/N-HiTS aca: la serie anual completa tiene
# ~16 observaciones (2010-2025, se descarta 2026 por ser un anio incompleto).
# Entrenar una red neuronal con 16 puntos no tiene sustento estadistico real
# -- cualquier "mejora" seria ruido de una unica corrida, no una senal
# reproducible. Es mas honesto compararlo solo contra metodos clasicos
# (Naive, Naive con drift, AutoARIMA) que estan disenados para funcionar con
# poca data, y dejar esto documentado en vez de forzar un resultado vistoso.

import matplotlib.pyplot as plt
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, Naive, RandomWalkWithDrift
from utilsforecast.losses import mae, mape, rmse

DATOS_LARGO = "../datos/bases/usdclp_long.csv"
RESULTADOS_DIR = "../datos/resultados"

FREQ = "YE"
HORIZONTE = 1
N_WINDOWS = 3
NIVELES = [80, 90]

COLUMNA_PUNTUAL = {"Naive": "Naive", "RWD": "RWD", "AutoARIMA": "AutoARIMA"}


def resamplear_anual(df_diario):
    s = df_diario.set_index("ds")["y"].resample(FREQ).last()
    s = s.iloc[:-1]  # descarta 2026, anio incompleto (yfinance corta en septiembre)
    df = s.reset_index()
    df.insert(0, "unique_id", "USDCLP")
    return df.dropna().reset_index(drop=True)


def correr_backtesting(df):
    sf = StatsForecast(
        models=[AutoARIMA(season_length=1), Naive(), RandomWalkWithDrift(alias="RWD")],
        freq=FREQ, n_jobs=-1,
    )
    return sf.cross_validation(df=df, h=HORIZONTE, n_windows=N_WINDOWS, step_size=HORIZONTE, level=NIVELES)


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
    tabla["mejora_%_vs_Naive"] = (tabla.loc[tabla["modelo"] == "Naive", "RMSE"].iloc[0] - tabla["RMSE"]) / tabla.loc[tabla["modelo"] == "Naive", "RMSE"].iloc[0] * 100
    return tabla.sort_values("RMSE").reset_index(drop=True)


def graficar(df_completo, cv_df, metricas):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_completo["ds"], df_completo["y"], label="USD/CLP real (anual)", color="black", linewidth=2, marker="o")
    for modelo, columna in COLUMNA_PUNTUAL.items():
        ax.plot(cv_df["ds"], cv_df[columna], label=modelo, linestyle="--", marker="x")
    ax.set_title(f"USD/CLP anual: prediccion vs. real (backtesting, {N_WINDOWS} anios, solo modelos clasicos)")
    ax.set_xlabel("Anio")
    ax.set_ylabel("CLP por USD")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{RESULTADOS_DIR}/prediccion_vs_real_anual.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ganador = metricas.iloc[0]["modelo"]
    colores = ["tab:blue" if m == ganador else "tab:gray" for m in metricas["modelo"]]
    ax.bar(metricas["modelo"], metricas["RMSE"], color=colores)
    ax.set_ylabel(f"RMSE (backtesting, {N_WINDOWS} anios)")
    ax.set_title("USD/CLP anual: RMSE por modelo (n=16 obs. - solo ilustrativo)")
    fig.tight_layout()
    fig.savefig(f"{RESULTADOS_DIR}/rmse_comparacion_anual.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    diario = pd.read_csv(DATOS_LARGO, parse_dates=["ds"])
    anual = resamplear_anual(diario)
    print(f"Serie anual: {len(anual)} observaciones, {anual['ds'].min().date()} a {anual['ds'].max().date()}")

    cv_df = correr_backtesting(anual)
    cv_df.to_csv(f"{RESULTADOS_DIR}/cv_anual.csv", index=False)

    metricas = calcular_metricas(cv_df)
    metricas.to_csv(f"{RESULTADOS_DIR}/metricas_comparacion_anual.csv", index=False)
    print(metricas.to_string(index=False))

    graficar(anual, cv_df, metricas)
    print(f"Graficos guardados en {RESULTADOS_DIR}/")
