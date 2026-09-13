# RMSE/MAE/MAPE por modelo (Naive, AutoARIMA, NBEATS, NHITS) sobre las mismas
# ventanas de backtesting. No es opcional: es la leccion de la auditoria de
# weather-time-series-forecasting, que no calculo error cuantitativo en test.

import pandas as pd
from utilsforecast.losses import mae, mape, rmse

RESULTADOS_DIR = "../datos/resultados"

COLUMNA_PUNTUAL = {
    "Naive": "Naive",
    "AutoARIMA": "AutoARIMA",
    "NBEATS": "NBEATS-median",
    "NHITS": "NHITS-median",
}


def cargar_y_combinar():
    baseline = pd.read_csv(f"{RESULTADOS_DIR}/cv_baseline.csv")
    nixtla_nn = pd.read_csv(f"{RESULTADOS_DIR}/cv_nbeats_nhits.csv")
    claves = ["unique_id", "ds", "cutoff", "y"]
    return baseline.merge(nixtla_nn, on=claves, how="inner")


def calcular_metricas(cv_df):
    columnas = list(COLUMNA_PUNTUAL.values())
    tablas = {"RMSE": rmse(cv_df, columnas), "MAE": mae(cv_df, columnas), "MAPE": mape(cv_df, columnas)}
    filas = []
    for nombre_modelo, columna in COLUMNA_PUNTUAL.items():
        fila = {"modelo": nombre_modelo}
        for metrica, tabla in tablas.items():
            fila[metrica] = tabla[columna].iloc[0]
        filas.append(fila)
    return pd.DataFrame(filas)


def armar_tabla_comparativa(metricas):
    mejor_baseline_rmse = metricas.loc[metricas["modelo"].isin(["Naive", "AutoARIMA"]), "RMSE"].min()
    metricas["mejora_%_vs_mejor_baseline_RMSE"] = (
        (mejor_baseline_rmse - metricas["RMSE"]) / mejor_baseline_rmse * 100
    )
    return metricas.sort_values("RMSE").reset_index(drop=True)


if __name__ == "__main__":
    cv_df = cargar_y_combinar()
    metricas = calcular_metricas(cv_df)
    tabla = armar_tabla_comparativa(metricas)
    tabla.to_csv(f"{RESULTADOS_DIR}/metricas_comparacion.csv", index=False)
    print(tabla.to_string(index=False))
