# Seguimiento a 9.13: valida si copper_ret_1d (|r|=-0.256 con el retorno del
# dia siguiente, ver 21_features_nuevas_frecuencias.py) es una señal REALMENTE
# aprovechable - correlacion no es lo mismo que rentable. Mismo metodo que
# 18_kelly_validacion.py (Kelly condicional via regresion OLS, sin entrenar
# ningun agente), pero a frecuencia DIARIA.
#
# Simplificacion honesta frente a 14_backtest_walkforward_gestion_riesgo.py:
# el trailing stop / take-profit de ese backtest chequea precios INTRADIA
# dentro de la semana - a frecuencia diaria no hay datos mas finos que el
# cierre diario, asi que no hay forma de replicar esa misma mecanica de
# gestion de riesgo (no es que se decida omitirla, es que no hay con que
# construirla). Se simula en cambio con la misma logica que
# simular_buy_and_hold_simple() de 14: mantener la posicion de cierre a
# cierre, con costo de slippage al cambiar de posicion.
#
# sigma^2 para Kelly: volatilidad realizada rolling de 20 dias (no GARCH) -
# construir un forecast GARCH walk-forward diario con miles de ventanas seria
# caro y no es el punto de este chequeo (validar si HAY señal explotable, no
# optimizar el sizing).

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

features_mod = importlib.import_module("19_features_nuevas_validacion")
generar_mod = importlib.import_module("10_generar_dataset_rl")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")

RESULTADOS_DIR = "../datos/resultados"
DATOS_LARGO = "../datos/bases/usdclp_long.csv"

CAPITAL_INICIAL = 100.0
SLIPPAGE_PCT = 0.0005  # mismo valor que el resto del proyecto
LIMITE_APALANCAMIENTO = 1.0
VENTANA_VOL = 20
N_WINDOWS_WF = 5
N_TEST_POR_VENTANA = 60  # ~3 meses de trading por ventana, 300 dias (~14 meses) de test total

FEATURES_DIARIAS = ["retorno_1d", "macd_rel", "rsi_norm", "copper_ret_1d", "copper_mom_5d"]


def preparar_dataset_diario(macro):
    df = pd.read_csv(DATOS_LARGO, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)
    df["retorno_1d"] = np.log(df["y"] / df["y"].shift(1))
    df["macd_rel"] = generar_mod.calcular_macd(df["y"]) / df["y"]
    df["rsi_norm"] = generar_mod.calcular_rsi(df["y"]) / 100
    df["vol_realizada"] = df["retorno_1d"].rolling(VENTANA_VOL).std()
    df = pd.merge_asof(df, macro[["ds", "copper"]], on="ds", direction="backward")
    df["copper_ret_1d"] = np.log(df["copper"] / df["copper"].shift(1))
    df["copper_mom_5d"] = (df["copper"] - df["copper"].shift(5)) / df["copper"].shift(5)
    df["y_next"] = df["y"].shift(-1)
    return df.dropna().reset_index(drop=True)


def ajustar_regresion(df_train, features):
    X = np.column_stack([np.ones(len(df_train)), df_train[features].to_numpy(dtype=float)])
    y_ret = ((df_train["y_next"] - df_train["y"]) / df_train["y"]).to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, y_ret, rcond=None)
    return coef


def posiciones_kelly(df_train, df_test, features):
    coef = ajustar_regresion(df_train, features)
    X = np.column_stack([np.ones(len(df_test)), df_test[features].to_numpy(dtype=float)])
    mu_pred = X @ coef
    sigma2 = df_test["vol_realizada"].to_numpy(dtype=float) ** 2
    f = np.divide(mu_pred, sigma2, out=np.zeros_like(mu_pred), where=sigma2 > 0)
    return np.clip(f, -LIMITE_APALANCAMIENTO, LIMITE_APALANCAMIENTO)


def posiciones_umbral_cobre(df_test, umbral=0.0):
    # Señal directa (sin regresion): copper_ret_1d correlaciona NEGATIVO con
    # el retorno futuro de USD/CLP -> cobre sube hoy => se espera que USD/CLP
    # baje mañana (largo en CLP, es decir corto en USD/CLP) y viceversa.
    return np.where(df_test["copper_ret_1d"] > umbral, -1.0, np.where(df_test["copper_ret_1d"] < -umbral, 1.0, 0.0))


def simular(df_test, posiciones, nombre, capital_inicial):
    capital = capital_inicial
    posicion_previa = 0.0
    filas = []
    for i, fila in df_test.reset_index(drop=True).iterrows():
        posicion = posiciones[i]
        retorno = (fila["y_next"] - fila["y"]) / fila["y"]
        pnl = capital * posicion * retorno
        costo_slippage = SLIPPAGE_PCT * abs(posicion) * capital if posicion != posicion_previa else 0.0
        pnl -= costo_slippage
        capital += pnl
        posicion_previa = posicion
        filas.append({"ds": fila["ds"], "estrategia": nombre, "posicion": posicion, "pnl": pnl, "capital": capital})
    return pd.DataFrame(filas)


def calcular_metricas(resultado, nombre):
    capital_previo = resultado["capital"].shift(1).fillna(CAPITAL_INICIAL)
    r = resultado["pnl"] / capital_previo
    capital = resultado["capital"]
    retorno_total_pct = 100 * (capital.iloc[-1] / CAPITAL_INICIAL - 1)
    sharpe = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else np.nan
    drawdown = capital / capital.cummax() - 1
    operaciones = (resultado["posicion"] != 0).sum()
    win_rate = 100 * (resultado.loc[resultado["posicion"] != 0, "pnl"] > 0).mean() if operaciones > 0 else np.nan
    return {"estrategia": nombre, "capital_final": capital.iloc[-1], "retorno_total_%": retorno_total_pct,
            "sharpe_anualizado": sharpe, "max_drawdown_%": 100 * drawdown.min(), "win_rate_%": win_rate, "operaciones": operaciones}


def correr_walkforward(df, features, nombre):
    capital = CAPITAL_INICIAL
    partes = []
    for df_train, df_test in wf_mod.ventanas_walkforward(df, N_WINDOWS_WF, N_TEST_POR_VENTANA):
        posiciones = posiciones_kelly(df_train, df_test, features)
        r = simular(df_test, posiciones, nombre, capital)
        capital = r["capital"].iloc[-1]
        partes.append(r)
    return pd.concat(partes, ignore_index=True)


def graficar(resultados, path_salida):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colores = {"Buy-and-hold": "black", "Umbral cobre": "darkorange", "Kelly diario (sin cobre)": "mediumpurple",
               "Kelly diario (con cobre)": "crimson"}
    for nombre, r in resultados.items():
        ax.plot(r["ds"], r["capital"], label=nombre, color=colores[nombre], linewidth=1.6)
    ax.axhline(CAPITAL_INICIAL, color="gray", linestyle=":", linewidth=1, label="Capital inicial ($100)")
    ax.set_ylabel("Capital ($)")
    ax.set_title(f"Kelly a frecuencia diaria con/sin cobre - walk-forward ({N_WINDOWS_WF * N_TEST_POR_VENTANA} dias)")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    macro = features_mod.cargar_macro()
    df = preparar_dataset_diario(macro)
    print(f"Dataset diario: {len(df)} obs, {df['ds'].min().date()} a {df['ds'].max().date()}")
    print(f"Walk-forward: {N_WINDOWS_WF} ventanas x {N_TEST_POR_VENTANA} dias ({N_WINDOWS_WF*N_TEST_POR_VENTANA} dias de test)\n")

    resultados, filas_metricas = {}, []

    r_bh = simular(df.iloc[-(N_WINDOWS_WF * N_TEST_POR_VENTANA):], [1.0] * (N_WINDOWS_WF * N_TEST_POR_VENTANA), "Buy-and-hold", CAPITAL_INICIAL)
    resultados["Buy-and-hold"] = r_bh
    filas_metricas.append(calcular_metricas(r_bh, "Buy-and-hold"))

    capital = CAPITAL_INICIAL
    partes_umbral = []
    for df_train, df_test in wf_mod.ventanas_walkforward(df, N_WINDOWS_WF, N_TEST_POR_VENTANA):
        r = simular(df_test, posiciones_umbral_cobre(df_test), "Umbral cobre", capital)
        capital = r["capital"].iloc[-1]
        partes_umbral.append(r)
    resultados["Umbral cobre"] = pd.concat(partes_umbral, ignore_index=True)
    filas_metricas.append(calcular_metricas(resultados["Umbral cobre"], "Umbral cobre"))

    resultados["Kelly diario (sin cobre)"] = correr_walkforward(df, ["retorno_1d", "macd_rel", "rsi_norm"], "Kelly diario (sin cobre)")
    filas_metricas.append(calcular_metricas(resultados["Kelly diario (sin cobre)"], "Kelly diario (sin cobre)"))

    resultados["Kelly diario (con cobre)"] = correr_walkforward(df, FEATURES_DIARIAS, "Kelly diario (con cobre)")
    filas_metricas.append(calcular_metricas(resultados["Kelly diario (con cobre)"], "Kelly diario (con cobre)"))

    tabla = pd.DataFrame(filas_metricas).sort_values("sharpe_anualizado", ascending=False).reset_index(drop=True)
    tabla.to_csv(f"{RESULTADOS_DIR}/kelly_diario_cobre_metricas.csv", index=False)
    for nombre, r in resultados.items():
        r.to_csv(f"{RESULTADOS_DIR}/kelly_diario_{nombre.lower().replace(' ', '_').replace('(', '').replace(')', '')}_operaciones.csv", index=False)

    graficar(resultados, f"{RESULTADOS_DIR}/kelly_diario_cobre_curva_capital.png")

    print(tabla.to_string(index=False))
