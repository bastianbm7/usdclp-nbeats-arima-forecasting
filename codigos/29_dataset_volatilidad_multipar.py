# Issue #6: genera, para cada par nuevo (USD/MXN, USD/BRL, USD/COP), el mismo
# dataset semanal walk-forward que 10_generar_dataset_rl.py usa para CLP
# (forecast N-HiTS + volatilidad GARCH + MACD/RSI/min-max), y repite la
# comparacion de modelos de volatilidad de 09_comparacion_modelos_volatilidad.py
# para cada uno - NO se asume que GARCH(1,1) gane igual en todos, se verifica.
#
# Pares elegidos (candidatos sugeridos en el propio Issue #6): USD/MXN, USD/BRL,
# USD/COP - similitud de dinamica cambiaria LatAm con USD/CLP. Mismos tickers y
# misma convencion de cotizacion (moneda local por 1 USD, sin invertir) que ya
# uso 23_dataset_multi_par_diario.py para estos 3 pares - consistente con como
# CLP=X ya se trata en todo el proyecto.
#
# Reusa funciones de 01 (limpiar_ticks_erroneos), 09 (comparacion de modelos de
# volatilidad) y 10 (indicadores tecnicos + forecast NHITS walk-forward) via
# importlib, en vez de duplicar codigo - mismo patron que 14/17/18/20/22/23.
# NO toca 11_entorno_trading_rl.py (el Issue #5 hermano lo esta modificando en
# paralelo para frecuencia diaria) ni los scripts 01/09/10 (que siguen siendo
# la fuente de verdad para el pipeline semanal de CLP) - solo los importa.
#
# Nota de costo: generar_forecast_nhits reentrena NHITS en cada una de las
# N_WINDOWS ventanas (~10s/ventana, medido en 10_generar_dataset_rl.py) - con
# N_WINDOWS=350 (mismo valor que CLP, para dataset de tamano comparable) son
# ~58 min por par, ~3h para los 3 pares nuevos. Es el costo dominante de este
# script; la comparacion de volatilidad (GARCH/EGARCH ajustan en ~0.08s) es
# practicamente gratis en comparacion.

import importlib

import pandas as pd
import yfinance as yf

obtener_datos_mod = importlib.import_module("01_obtener_datos")
vol_mod = importlib.import_module("09_comparacion_modelos_volatilidad")
generar_mod = importlib.import_module("10_generar_dataset_rl")

BASES_DIR = "../datos/bases"
RESULTADOS_DIR = "../datos/resultados"
FECHA_INICIO = "2010-01-01"

# ticker yfinance -> unique_id interno (mismos tickers/convencion que PARES en
# 23_dataset_multi_par_diario.py: cotizados como "moneda local por 1 USD", sin
# invertir - igual que CLP=X en todo el resto del proyecto)
PARES_NUEVOS = {
    "MXN=X": "USDMXN",
    "BRL=X": "USDBRL",
    "COP=X": "USDCOP",
}

N_WINDOWS = generar_mod.N_WINDOWS  # 350, mismo que el dataset semanal de CLP
N_WINDOWS_VOL_TEST = vol_mod.N_WINDOWS_TEST  # 100


def descargar_diario(ticker, unique_id, start=FECHA_INICIO):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel("Ticker")
    df = df[["Close"]].reset_index()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    df = df.dropna(subset=["y"]).sort_values("ds").reset_index(drop=True)
    df = obtener_datos_mod.limpiar_ticks_erroneos(df)  # mismo bug de ticks corruptos de un dia ya documentado para CLP=X (y en 23 para COP=X/PEN=X/ZAR=X/CHF=X)
    df.insert(0, "unique_id", unique_id)
    return df


def comparar_volatilidad(diario, unique_id):
    rv_semanal = vol_mod.volatilidad_realizada_semanal(diario)
    resultados = vol_mod.correr_comparacion(diario, rv_semanal, N_WINDOWS_VOL_TEST)
    metricas = vol_mod.calcular_metricas(resultados)
    metricas.insert(0, "par", unique_id)
    return metricas


def generar_dataset_par(diario, unique_id, n_windows=N_WINDOWS):
    semanal = generar_mod.resamplear_semanal(diario)
    semanal["unique_id"] = unique_id  # resamplear_semanal hardcodea "USDCLP" (pensado solo para CLP) - se corrige aca sin tocar 10
    print(f"  Serie semanal: {len(semanal)} obs, {semanal['ds'].min().date()} a {semanal['ds'].max().date()}")

    con_indicadores = generar_mod.calcular_indicadores_tecnicos(semanal)

    n_windows_usable = min(n_windows, len(semanal) - generar_mod.INPUT_SIZE - 2)
    print(f"  Forecast N-HiTS walk-forward, {n_windows_usable} ventanas (~10s/ventana, ~{n_windows_usable * 10 / 60:.0f} min estimados)...")
    forecast_nhits = generar_mod.generar_forecast_nhits(semanal, n_windows_usable)

    fechas_decision = forecast_nhits["ds"]
    vol = generar_mod.generar_volatilidad_por_semana(diario, fechas_decision)

    dataset = forecast_nhits.merge(con_indicadores, on="ds", how="left").merge(vol, on="ds", how="left")
    dataset = dataset.sort_values("ds").reset_index(drop=True)
    dataset["y_next"] = dataset["y"].shift(-1)  # precio real de la semana siguiente, para el reward del entorno (igual a 10)
    dataset = dataset.dropna().reset_index(drop=True)

    columnas = ["unique_id", "ds", "y", "y_next", "nhits_h1", "nhits_h2", "vol_garch", "macd", "rsi", "precio_min_ventana", "precio_max_ventana"]
    return dataset[columnas]


if __name__ == "__main__":
    tabla_vol_todas = []
    for ticker, unique_id in PARES_NUEVOS.items():
        print(f"\n=== {unique_id} ({ticker}) ===")
        diario = descargar_diario(ticker, unique_id)
        diario.to_csv(f"{BASES_DIR}/{unique_id.lower()}_long.csv", index=False)
        print(f"Diario: {len(diario)} obs, {diario['ds'].min().date()} a {diario['ds'].max().date()} -> {BASES_DIR}/{unique_id.lower()}_long.csv")

        print(f"\n[1/2] Comparacion de modelos de volatilidad ({N_WINDOWS_VOL_TEST} ventanas de test, refit en cada una)...")
        metricas_vol = comparar_volatilidad(diario, unique_id)
        print(metricas_vol.to_string(index=False))
        tabla_vol_todas.append(metricas_vol)

        print(f"\n[2/2] Dataset RL semanal (forecast NHITS walk-forward + volatilidad GARCH + indicadores tecnicos)...")
        dataset = generar_dataset_par(diario, unique_id)
        path_dataset = f"{RESULTADOS_DIR}/dataset_entrenamiento_rl_{unique_id.lower()}.csv"
        dataset.to_csv(path_dataset, index=False)
        print(f"  listo: {len(dataset)} semanas, {dataset['ds'].min().date()} a {dataset['ds'].max().date()} -> {path_dataset}")

    tabla_final = pd.concat(tabla_vol_todas, ignore_index=True)
    tabla_final.to_csv(f"{RESULTADOS_DIR}/comparacion_volatilidad_multipar_metricas.csv", index=False)
    print("\n\n=== Comparacion de modelos de volatilidad, los 3 pares nuevos (mismo esquema que 09 para CLP) ===")
    print(tabla_final.to_string(index=False))
