# Compara modelos de volatilidad (no de precio) antes de elegir cual alimenta al
# agente de RL (Issue #1 / Fase 1). Mismo espiritu que 02/03: no asumir que el
# modelo mas sofisticado (GARCH/EGARCH) le gana al baseline (naive/media movil)
# sin medirlo primero, walk-forward, out-of-sample.
#
# Volatilidad realizada semanal = sqrt(suma de retornos log diarios al cuadrado
# dentro de esa semana) - se computa sobre datos diarios (no sobre la serie ya
# resampleada a semanal) porque con 1 solo dato por semana no hay forma de
# estimar volatilidad intra-semanal.
#
# GARCH/EGARCH ajustan en ~0.08s sobre las ~4.300 obs diarias (medido) - por eso
# el walk-forward de este script reentrena en CADA ventana de test, a diferencia
# de N-HiTS (~10s/ventana) donde eso seria carisimo.

import numpy as np
import pandas as pd
from arch import arch_model

DATOS_LARGO = "../datos/bases/usdclp_long.csv"
RESULTADOS_DIR = "../datos/resultados"

N_WINDOWS_TEST = 100  # ~2 anos de semanas de test, cada una con su propio refit
VENTANA_MEDIA_MOVIL = 8


def volatilidad_realizada_semanal(diario):
    diario = diario.sort_values("ds").reset_index(drop=True)
    diario["retorno_log"] = np.log(diario["y"] / diario["y"].shift(1))
    semanal = diario.set_index("ds")["retorno_log"].resample("W").apply(lambda r: np.sqrt((r**2).sum()))
    return semanal.dropna()


def forecast_naive(rv_hist):
    return rv_hist.iloc[-1]


def forecast_media_movil(rv_hist, ventana=VENTANA_MEDIA_MOVIL):
    return rv_hist.iloc[-ventana:].mean()


def forecast_garch(retornos_diarios, tipo_vol):
    # EGARCH no tiene forecast analitico para horizon>1 (asimetria del modelo) -
    # simulation funciona para ambos, se usa siempre para no bifurcar el codigo.
    am = arch_model(retornos_diarios * 100, vol=tipo_vol, p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    pronostico = res.forecast(horizon=5, method="simulation", simulations=200, reindex=False)
    varianza_5d = pronostico.variance.values[-1].sum()
    return np.sqrt(varianza_5d) / 100


def correr_comparacion(diario, rv_semanal, n_windows):
    diario = diario.sort_values("ds").reset_index(drop=True)
    diario["retorno_log"] = np.log(diario["y"] / diario["y"].shift(1))
    retornos_diarios = diario.set_index("ds")["retorno_log"].dropna()

    fechas_test = rv_semanal.index[-n_windows:]
    filas = []
    for fecha_objetivo in fechas_test:
        rv_hist = rv_semanal[rv_semanal.index < fecha_objetivo]
        ret_hist = retornos_diarios[retornos_diarios.index < fecha_objetivo]
        if len(rv_hist) < VENTANA_MEDIA_MOVIL or len(ret_hist) < 500:
            continue

        real = rv_semanal.loc[fecha_objetivo]
        filas.append({
            "fecha": fecha_objetivo,
            "real": real,
            "Naive": forecast_naive(rv_hist),
            "MediaMovil": forecast_media_movil(rv_hist),
            "GARCH": forecast_garch(ret_hist, "GARCH"),
            "EGARCH": forecast_garch(ret_hist, "EGARCH"),
        })
    return pd.DataFrame(filas)


def calcular_metricas(resultados):
    modelos = ["Naive", "MediaMovil", "GARCH", "EGARCH"]
    filas = []
    for modelo in modelos:
        error = resultados[modelo] - resultados["real"]
        filas.append({
            "modelo": modelo,
            "RMSE": np.sqrt((error**2).mean()),
            "MAE": error.abs().mean(),
        })
    tabla = pd.DataFrame(filas).sort_values("RMSE").reset_index(drop=True)
    mejor_rmse = tabla["RMSE"].iloc[0]
    tabla["mejora_vs_naive_%"] = 100 * (tabla.loc[tabla["modelo"] == "Naive", "RMSE"].iloc[0] - tabla["RMSE"]) / tabla.loc[tabla["modelo"] == "Naive", "RMSE"].iloc[0]
    return tabla


if __name__ == "__main__":
    diario = pd.read_csv(DATOS_LARGO, parse_dates=["ds"])
    rv_semanal = volatilidad_realizada_semanal(diario)
    print(f"Volatilidad realizada semanal: {len(rv_semanal)} semanas, {rv_semanal.index.min().date()} a {rv_semanal.index.max().date()}")

    resultados = correr_comparacion(diario, rv_semanal, N_WINDOWS_TEST)
    resultados.to_csv(f"{RESULTADOS_DIR}/comparacion_volatilidad_cv.csv", index=False)

    metricas = calcular_metricas(resultados)
    metricas.to_csv(f"{RESULTADOS_DIR}/comparacion_volatilidad_metricas.csv", index=False)
    print(f"\n{len(resultados)} ventanas de test, refit en cada una:\n")
    print(metricas.to_string(index=False))
