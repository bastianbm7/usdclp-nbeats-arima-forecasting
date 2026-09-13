# Grafico de prediccion vs. real en el tramo de test, con banda de intervalo de
# confianza (80%) superpuesta, y baseline clasico superpuesto para comparacion
# visual directa. Entregable obligatorio en datos/resultados/.

import matplotlib.pyplot as plt
import pandas as pd

RESULTADOS_DIR = "../datos/resultados"


def cargar_cv_combinado():
    baseline = pd.read_csv(f"{RESULTADOS_DIR}/cv_baseline.csv", parse_dates=["ds"])
    nixtla_nn = pd.read_csv(f"{RESULTADOS_DIR}/cv_nbeats_nhits.csv", parse_dates=["ds"])
    claves = ["unique_id", "ds", "cutoff", "y"]
    df = baseline.merge(nixtla_nn, on=claves, how="inner").sort_values("ds")
    return df


MODELO_A_COLUMNA = {"Naive": "Naive", "AutoARIMA": "AutoARIMA", "NBEATS": "NBEATS-median", "NHITS": "NHITS-median"}


def graficar_prediccion_vs_real(cv_df):
    metricas = pd.read_csv(f"{RESULTADOS_DIR}/metricas_comparacion.csv")
    ganador = metricas.sort_values("RMSE").iloc[0]["modelo"]
    columna_ganador = MODELO_A_COLUMNA[ganador]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cv_df["ds"], cv_df["y"], label="USD/CLP real", color="black", linewidth=2)
    ax.plot(cv_df["ds"], cv_df["Naive"], label="Naive", linestyle="--", color="gray")
    ax.plot(cv_df["ds"], cv_df["AutoARIMA"], label="AutoARIMA", linestyle="--", color="tab:orange")
    otro_modelo_nn = "NHITS-median" if columna_ganador == "NBEATS-median" else "NBEATS-median"
    otro_alias = "N-HiTS" if columna_ganador == "NBEATS-median" else "N-BEATS"
    ax.plot(cv_df["ds"], cv_df[otro_modelo_nn], label=otro_alias, linestyle="-.", color="tab:green", alpha=0.6)
    ax.plot(cv_df["ds"], cv_df[columna_ganador], label=f"{ganador} (mejor RMSE)", color="tab:blue", linewidth=2)
    ax.fill_between(
        cv_df["ds"], cv_df[f"{ganador}-lo-80"], cv_df[f"{ganador}-hi-80"],
        color="tab:blue", alpha=0.15, label=f"{ganador} intervalo 80%",
    )
    ax.set_title("USD/CLP: prediccion vs. real (backtesting walk-forward, 5 ventanas x 14 dias)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("CLP por USD")
    ax.legend(loc="best", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{RESULTADOS_DIR}/prediccion_vs_real.png", dpi=150)
    plt.close(fig)


def graficar_barras_rmse():
    metricas = pd.read_csv(f"{RESULTADOS_DIR}/metricas_comparacion.csv")
    ganador = metricas.sort_values("RMSE").iloc[0]["modelo"]
    fig, ax = plt.subplots(figsize=(7, 5))
    colores = ["tab:blue" if m == ganador else "tab:gray" for m in metricas["modelo"]]
    ax.bar(metricas["modelo"], metricas["RMSE"], color=colores)
    ax.set_ylabel("RMSE (backtesting, 5 ventanas x 14 dias)")
    ax.set_title("USD/CLP: RMSE por modelo")
    fig.tight_layout()
    fig.savefig(f"{RESULTADOS_DIR}/rmse_comparacion.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    cv_df = cargar_cv_combinado()
    graficar_prediccion_vs_real(cv_df)
    graficar_barras_rmse()
    print(f"Graficos guardados en {RESULTADOS_DIR}/")
