# INVALIDADO / SUPERADO (2026-09-23) - el screening de 90 pruebas (el 'rezago +1' era el contemporaneo): ver script 61.
# Motivo: artefacto de timestamp (la barra diaria Yahoo FX con fecha D es el precio
# de ~20:00 NY de D-1, el cierre de un futuro de commodity con fecha D es su
# settlement de ~13:00-14:30 ET de D; el merge_asof(direction='backward') por
# fecha usaba informacion posterior al precio de entrada) + costos cobrados una
# sola vez al cambiar de posicion en vez de ida+vuelta en cada operacion. Ver
# alineacion_temporal.py, costos_y_estadistica.py y la errata en 9.35 del paper.
# Se conserva sin cambios de logica como registro de los numeros originales; no
# reproduce exactamente sus CSV si se vuelve a correr despues de la correccion de
# costos en 27/32/36.
#
# Issue #13, Fase 1: screening de correlacion NOK/ZAR/BRL x los 5 commodities
# que sobrevivieron Fase 0 (WTI, oro, platino, soja, hierro - 51_verificar_liquidez...py),
# con escaneo de rezagos -2 a +3 (mismo rango que valido el hallazgo de
# copper_ret_1d en 9.14 del paper: el efecto real esta concentrado en el
# rezago +1, no en el 0, lo que descarta contaminacion de timestamp).
#
# 3 monedas x 5 commodities x 6 rezagos = 90 pruebas de correlacion. Con ese
# volumen, el riesgo de falsos positivos por multiple testing es real - se
# aplica correccion Benjamini-Hochberg FDR (statsmodels.stats.multitest)
# sobre las 90 pruebas ANTES de reportar cualquier hallazgo como real, y se
# documentan TODOS los resultados (positivos y negativos), mismo estandar de
# honestidad que 9.16 (12 indicadores tecnicos que no superaron el umbral).
#
# Convencion de rezago (identica a la tabla de 9.14): correlacion entre
# commodity_ret_1d en t y el retorno de la moneda en t+rezago. rezago=+1 es la
# direccion "predictiva" (la que uso 9.13/9.14 para cobre); rezago=0 es
# contemporaneo (solapamiento de cierres); rezago<0 chequea si es la MONEDA la
# que lidera al commodity (heuristica anti-contaminacion de timestamp, no solo
# para rezago=0).
#
# NOK/ZAR/BRL: BRL y ZAR ya estan en datos/bases/panel_fx_diario.csv (9.14),
# no se redescargan. NOK se arma desde datos/bases/USDNOK_X_long.csv (Fase 0),
# invertido a "USD por NOK" con la misma convencion que el resto del panel.

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASES_DIR = "../datos/bases"
RESULTADOS_DIR = "../datos/resultados"
PANEL_CACHE = f"{BASES_DIR}/panel_fx_diario.csv"

UMBRAL_CORR = 0.11  # mismo umbral de referencia que el resto del proyecto (seccion 9.5)
ALPHA_FDR = 0.05
LAGS = list(range(-2, 4))  # -2..+3, igual que 9.14

MONEDAS = ["NOK", "ZAR", "BRL"]
COMMODITIES = {
    "CL_F": "WTI (petroleo)",
    "GC_F": "Oro",
    "PL_F": "Platino",
    "ZS_F": "Soja",
    "TIO_F": "Hierro",
}


def cargar_moneda(codigo):
    if codigo == "NOK":
        precios = pd.read_csv(f"{BASES_DIR}/USDNOK_X_long.csv", parse_dates=["ds"])
        precios = precios.sort_values("ds").reset_index(drop=True)
        precios["y"] = 1 / precios["y"]  # USDNOK=X cotiza NOK por USD -> invertir a USD por NOK
        precios["retorno_1d"] = np.log(precios["y"] / precios["y"].shift(1))
        return precios[["ds", "y", "retorno_1d"]].dropna().reset_index(drop=True)
    # BRL / ZAR ya estan en el panel de 9.14, invertidos y con retorno_1d listo
    ticker = {"BRL": "BRL=X", "ZAR": "ZAR=X"}[codigo]
    panel = pd.read_csv(PANEL_CACHE, parse_dates=["ds"])
    serie = panel[panel["par"] == ticker].sort_values("ds").reset_index(drop=True)
    return serie[["ds", "y", "retorno_1d"]].reset_index(drop=True)


def cargar_commodity(codigo):
    precios = pd.read_csv(f"{BASES_DIR}/{codigo}_long.csv", parse_dates=["ds"])
    return precios[["ds", "y"]].rename(columns={"y": "precio_commodity"}).sort_values("ds").reset_index(drop=True)


def construir_par(moneda_df, commodity_df):
    # merge_asof direction=backward: mismo criterio que 22/23_dataset_multi_par_diario.py
    # (fusiona el ultimo precio de commodity conocido a la fecha de la moneda,
    # sin look-ahead), y el retorno del commodity se calcula DESPUES del merge,
    # sobre el calendario de la moneda (misma secuencia que el resto del proyecto).
    df = pd.merge_asof(moneda_df, commodity_df, on="ds", direction="backward")
    df["commodity_ret_1d"] = np.log(df["precio_commodity"] / df["precio_commodity"].shift(1))
    return df.dropna().reset_index(drop=True)


def escaneo_de_rezagos(df, moneda, commodity_nombre):
    filas = []
    for lag in LAGS:
        retorno_futuro = df["retorno_1d"].shift(-lag)
        valido = retorno_futuro.notna() & df["commodity_ret_1d"].notna()
        x, y = df.loc[valido, "commodity_ret_1d"], retorno_futuro[valido]
        r, p = stats.pearsonr(x, y)
        filas.append({
            "moneda": moneda, "commodity": commodity_nombre, "rezago": lag,
            "correlacion": r, "p_valor": p, "n_obs": int(valido.sum()),
        })
    return filas


def graficar_heatmap(tabla, path_salida):
    pivot_r = tabla.pivot_table(index=["moneda", "commodity"], columns="rezago", values="correlacion")
    sig = tabla.pivot_table(index=["moneda", "commodity"], columns="rezago", values="sobrevive_fdr")

    fig, ax = plt.subplots(figsize=(9, 12))
    im = ax.imshow(pivot_r.values, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_xticks(range(len(pivot_r.columns)))
    ax.set_xticklabels(pivot_r.columns)
    ax.set_yticks(range(len(pivot_r.index)))
    ax.set_yticklabels([f"{m} x {c}" for m, c in pivot_r.index])
    for i in range(pivot_r.shape[0]):
        for j in range(pivot_r.shape[1]):
            valor = pivot_r.values[i, j]
            marca = "*" if sig.values[i, j] else ""
            ax.text(j, i, f"{valor:.2f}{marca}", ha="center", va="center", fontsize=7)
    ax.set_xlabel("Rezago (commodity_ret_1d en t vs. retorno moneda en t+rezago)")
    ax.set_title("Fase 1 (Issue #13): screening NOK/ZAR/BRL x commodities\n* = sobrevive correccion FDR (Benjamini-Hochberg, alpha=0.05)")
    fig.colorbar(im, ax=ax, label="correlacion de Pearson")
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    monedas_data = {m: cargar_moneda(m) for m in MONEDAS}
    commodities_data = {c: cargar_commodity(c) for c in COMMODITIES}

    filas = []
    for moneda, moneda_df in monedas_data.items():
        for codigo_commodity, nombre_commodity in COMMODITIES.items():
            par = construir_par(moneda_df, commodities_data[codigo_commodity])
            filas.extend(escaneo_de_rezagos(par, moneda, nombre_commodity))

    tabla = pd.DataFrame(filas)
    print(f"Total de pruebas: {len(tabla)} (3 monedas x 5 commodities x {len(LAGS)} rezagos)")

    # Correccion Benjamini-Hochberg FDR sobre las 90 pruebas juntas, antes de
    # mirar ninguna individualmente.
    rechaza, p_corregido, _, _ = multipletests(tabla["p_valor"], alpha=ALPHA_FDR, method="fdr_bh")
    tabla["p_valor_fdr"] = p_corregido
    tabla["sobrevive_fdr"] = rechaza
    tabla["supera_umbral_r"] = tabla["correlacion"].abs() >= UMBRAL_CORR
    tabla["hallazgo_real"] = tabla["sobrevive_fdr"] & tabla["supera_umbral_r"]

    tabla = tabla.sort_values("p_valor").reset_index(drop=True)
    tabla.to_csv(f"{RESULTADOS_DIR}/fase1_screening_correlacion_completo.csv", index=False)

    print(f"\n=== TODAS las 90 pruebas (positivas y negativas), ordenadas por p-valor ===")
    print(tabla.to_string(index=False))

    graficar_heatmap(tabla, f"{RESULTADOS_DIR}/fase1_screening_heatmap.png")

    supervivientes = tabla[tabla["hallazgo_real"]]
    print(f"\n=== Sobreviven FDR Y superan |r|>={UMBRAL_CORR}: {len(supervivientes)} de {len(tabla)} ===")
    if len(supervivientes):
        print(supervivientes.to_string(index=False))
    else:
        print("Ninguna combinacion sobrevive ambos filtros.")

    solo_fdr = tabla[tabla["sobrevive_fdr"] & ~tabla["supera_umbral_r"]]
    print(f"\n=== Sobreviven FDR pero NO superan |r|>={UMBRAL_CORR} (estadisticamente distinto de 0, pero chico): {len(solo_fdr)} ===")
    if len(solo_fdr):
        print(solo_fdr.to_string(index=False))

    supervivientes.to_csv(f"{RESULTADOS_DIR}/fase1_supervivientes_fdr.csv", index=False)
