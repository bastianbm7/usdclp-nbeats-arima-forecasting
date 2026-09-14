# Hoja de ruta del radar-baseline (2026-09-13), Tier 1 (diagnostico) + Tier 2:
# valida candidatas concretas de features nuevas encontradas en la investigacion
# (ver artifact "Radar USD/CLP") ANTES de gastar ~15-17 min de entrenamiento PPO
# por config en 20_agente_rl_ronda2.py - mismo criterio que 15_analisis_features.py
# (correlacion con el retorno futuro) mas el backtest de Kelly condicional de
# 18_kelly_validacion.py (reusado, no reimplementado) para ver si, ademas de
# correlacionar, producen una estrategia que de verdad gane plata.
#
# Candidatas (todas ancladas en papers con codigo real, ver seccion "Los 12
# candidatos" y "Ronda 2" del radar):
#   - rate_diff: TPM Chile (proxy: tasa interbancaria 3m, FRED IR3TIB01CLM156N)
#     menos Fed Funds Rate (FRED DFF) - ancla: Filippou, Rapach, Taylor & Zhou
#     (2020), carry trade / paridad de tasas.
#   - copper_ret_1s / copper_mom_4s: retorno y momentum del cobre (yfinance
#     HG=F) - ancla: Chen & Rogoff (2003), "Commodity Currencies", especifico
#     para CLP.
#   - mom_4s / mom_12s: momentum de USD/CLP normalizado por vol_garch - ancla:
#     Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum".
#
# Caveat honesto: la tasa de Chile via FRED (serie OECD MEI, mensual) tiene
# rezago de publicacion de 2-4 meses - los ultimos meses del dataset quedan con
# el ultimo valor conocido (forward-fill), no el dato real de esa semana. Ver
# hallazgo de la "ronda 1" del radar sobre rezago de publicacion en variables
# macro trimestrales/mensuales.

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

entorno_mod = importlib.import_module("11_entorno_trading_rl")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")
kelly_mod = importlib.import_module("18_kelly_validacion")

RESULTADOS_DIR = "../datos/resultados"
BASES_DIR = "../datos/bases"
MACRO_CACHE = f"{BASES_DIR}/macro_tasas_cobre.csv"

FRED_TASA_CHILE = "IR3TIB01CLM156N"  # tasa interbancaria 3m Chile (OECD MEI via FRED) - proxy de TPM, ver caveat de rezago arriba
FRED_TASA_US = "DFF"  # Fed Funds Rate efectiva, diaria


def descargar_macro():
    url_chile = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_TASA_CHILE}"
    url_us = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_TASA_US}"
    tasa_chile = pd.read_csv(url_chile).rename(columns={"observation_date": "ds", FRED_TASA_CHILE: "tasa_chile"})
    tasa_us = pd.read_csv(url_us).rename(columns={"observation_date": "ds", FRED_TASA_US: "tasa_us"})
    tasa_chile["ds"], tasa_us["ds"] = pd.to_datetime(tasa_chile["ds"]), pd.to_datetime(tasa_us["ds"])

    cobre = yf.download("HG=F", start="2009-01-01", progress=False)[["Close"]]
    cobre.columns = ["copper"]
    cobre = cobre.reset_index().rename(columns={"Date": "ds"})
    cobre["ds"] = pd.to_datetime(cobre["ds"]).dt.tz_localize(None)

    macro = cobre.merge(tasa_chile, on="ds", how="outer").merge(tasa_us, on="ds", how="outer").sort_values("ds")
    macro[["copper", "tasa_chile", "tasa_us"]] = macro[["copper", "tasa_chile", "tasa_us"]].ffill()
    macro = macro.dropna().reset_index(drop=True)
    macro.to_csv(MACRO_CACHE, index=False)
    return macro


def cargar_macro():
    try:
        macro = pd.read_csv(MACRO_CACHE, parse_dates=["ds"])
        print(f"Macro: usando cache {MACRO_CACHE} ({len(macro)} filas, hasta {macro['ds'].max().date()})")
        return macro
    except FileNotFoundError:
        print("Macro: sin cache, descargando (FRED + yfinance)...")
        return descargar_macro()


def agregar_features_nuevas(df, macro):
    df = df.sort_values("ds").reset_index(drop=True)
    df = pd.merge_asof(df, macro[["ds", "copper", "tasa_chile", "tasa_us"]], on="ds", direction="backward")

    df["rate_diff"] = df["tasa_chile"] - df["tasa_us"]
    df["copper_ret_1s"] = np.log(df["copper"] / df["copper"].shift(1))
    df["copper_mom_4s"] = (df["copper"] - df["copper"].shift(4)) / df["copper"].shift(4)

    df["mom_4s"] = ((df["y"] - df["y"].shift(4)) / df["y"].shift(4)) / df["vol_garch"]
    df["mom_12s"] = ((df["y"] - df["y"].shift(12)) / df["y"].shift(12)) / df["vol_garch"]

    return df.dropna().reset_index(drop=True)


FEATURES_NUEVAS = ["rate_diff", "copper_ret_1s", "copper_mom_4s", "mom_4s", "mom_12s"]


def calcular_correlaciones(df):
    retorno_futuro = (df["y_next"] - df["y"]) / df["y"]
    filas = []
    for feature in FEATURES_NUEVAS:
        corr = df[feature].corr(retorno_futuro)
        acierto = (np.sign(df[feature]) == np.sign(retorno_futuro)).mean()
        filas.append({"feature": feature, "correlacion_con_retorno_futuro": corr, "acierto_direccion_%": 100 * acierto})
    return pd.DataFrame(filas).sort_values("correlacion_con_retorno_futuro", key=abs, ascending=False).reset_index(drop=True)


def graficar_correlaciones(tabla_nuevas, path_salida):
    tabla_actuales = pd.read_csv(f"{RESULTADOS_DIR}/analisis_features_correlacion.csv")
    combinado = pd.concat([
        tabla_actuales[["feature", "correlacion_con_retorno_futuro"]].assign(grupo="7 actuales (seccion 9.5)"),
        tabla_nuevas[["feature", "correlacion_con_retorno_futuro"]].assign(grupo="candidatas nuevas (radar-baseline)"),
    ]).sort_values("correlacion_con_retorno_futuro", key=abs)

    fig, ax = plt.subplots(figsize=(9, 7))
    colores = ["darkorange" if g.startswith("candidatas") else "steelblue" for g in combinado["grupo"]]
    ax.barh(combinado["feature"], combinado["correlacion_con_retorno_futuro"], color=colores)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.11, color="gray", linewidth=0.6, linestyle="--")
    ax.axvline(-0.11, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Correlacion con el retorno semanal siguiente (y_next vs y)")
    ax.set_title("Features actuales (azul) vs. candidatas del radar-baseline (naranja)\nlineas grises = |r|=0.11, el techo que ninguna de las 7 actuales supera")
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    macro = cargar_macro()
    diario = pd.read_csv(wf_mod.DATOS_LARGO, parse_dates=["ds"])
    df_base = entorno_mod.cargar_dataset()
    df = agregar_features_nuevas(df_base, macro)
    print(f"\nDataset con features nuevas: {len(df)} semanas ({len(df_base) - len(df)} perdidas por momentum de 12 semanas + merge)\n")

    tabla_corr = calcular_correlaciones(df)
    tabla_corr.to_csv(f"{RESULTADOS_DIR}/analisis_features_nuevas_correlacion.csv", index=False)
    graficar_correlaciones(tabla_corr, f"{RESULTADOS_DIR}/analisis_features_nuevas_correlacion.png")
    print("=== Correlacion con el retorno futuro (candidatas nuevas) ===")
    print(tabla_corr.to_string(index=False))

    # El incondicional no usa features (solo mu/sigma^2 del retorno historico),
    # asi que se descarta ese resultado aca - ya esta en la tabla via 18 (la
    # version con df ampliado da practicamente el mismo numero, ver notas).
    features_ampliadas = entorno_mod.FEATURES_ESTADO + FEATURES_NUEVAS
    _, resultado_cond, metricas, _, tabla_f_cond = kelly_mod.correr_walkforward_kelly(
        df, diario, features_ampliadas, "_incondicional_descartado", "Kelly condicional (7 + 5 nuevas)")
    metrica_cond = metricas[1]

    resultado_cond.to_csv(f"{RESULTADOS_DIR}/kelly_condicional_con_macro_operaciones.csv", index=False)
    tabla_f_cond.to_csv(f"{RESULTADOS_DIR}/kelly_fraccion_condicional_con_macro.csv", index=False)

    referencia = pd.read_csv(f"{RESULTADOS_DIR}/kelly_validacion_metricas.csv")
    referencia = referencia[referencia["estrategia"] != metrica_cond["estrategia"]]
    tabla = pd.concat([pd.DataFrame([metrica_cond]), referencia], ignore_index=True)
    tabla = tabla.sort_values("sharpe_anualizado", ascending=False).reset_index(drop=True)
    tabla.to_csv(f"{RESULTADOS_DIR}/kelly_validacion_metricas.csv", index=False)

    print(f"\n=== Kelly condicional con las 5 features nuevas sumadas, vs. todo lo demas ===\n")
    print(tabla.to_string(index=False))
