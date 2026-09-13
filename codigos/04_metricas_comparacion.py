# RMSE/MAE/MAPE por modelo (Naive, AutoARIMA, NBEATS, NHITS) sobre las mismas
# ventanas de backtesting. No es opcional: es la leccion de la auditoria de
# weather-time-series-forecasting, que no calculo error cuantitativo en test.

RESULTADOS_DIR = "../datos/resultados"


def calcular_metricas(cv_df):
    raise NotImplementedError


def armar_tabla_comparativa(metricas_por_modelo):
    raise NotImplementedError


if __name__ == "__main__":
    pass
