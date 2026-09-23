# INVALIDADO / SUPERADO (2026-09-23) - las columnas de cobre del panel extendido: ver script 60.
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
# Issue #13, Fase 3 (paso 1): extiende el panel de 13 pares de 9.14
# (datos/bases/panel_fx_diario.csv) agregando NOK, que sobrevivio Fase 0
# (51_verificar_liquidez_commodities_nok.py) pero no estaba en el panel
# original (construido antes del Issue #13). Reusa EXACTAMENTE la misma
# receta de features que 23_dataset_multi_par_diario.py aplico a los otros 13
# pares (retorno_1d, macd_rel, rsi_norm, vol_realizada, copper_ret_1d,
# copper_mom_5d, y_next) para que NOK sea comparable fila a fila con el resto
# del panel.
#
# Se guarda en un archivo NUEVO (panel_fx_diario_extendido.csv), no se
# sobreescribe panel_fx_diario.csv - las Fases 1 y 2 de este mismo Issue ya
# leen ese archivo para BRL/ZAR y no deben verse afectadas por este paso.
#
# NOK se clasifica es_commodity=True: Noruega es una economia exportadora neta
# de petroleo (mismo canal que valida a NOK x WTI con r=0.275 en la Fase 1 de
# este Issue, seccion 9.29 del paper) - coherente con el criterio de
# 23_dataset_multi_par_diario.py para CAD (tambien commodity-petroleo).

import importlib

import numpy as np
import pandas as pd

features_mod = importlib.import_module("19_features_nuevas_validacion")
generar_mod = importlib.import_module("10_generar_dataset_rl")
kelly_diario_mod = importlib.import_module("22_kelly_diario_cobre")
obtener_datos_mod = importlib.import_module("01_obtener_datos")

BASES_DIR = "../datos/bases"
PANEL_ORIGINAL = f"{BASES_DIR}/panel_fx_diario.csv"
PANEL_EXTENDIDO = f"{BASES_DIR}/panel_fx_diario_extendido.csv"

NOK_TICKER, NOK_NOMBRE = "USDNOK=X", "USD/NOK"


def construir_fila_nok(macro):
    precios = pd.read_csv(f"{BASES_DIR}/USDNOK_X_long.csv", parse_dates=["ds"])
    precios = precios.sort_values("ds").reset_index(drop=True)
    precios["y"] = 1 / precios["y"]  # USDNOK=X cotiza NOK por USD -> invertir a USD por NOK, misma convencion que el panel

    df = precios[["ds", "y"]].copy()
    df["retorno_1d"] = np.log(df["y"] / df["y"].shift(1))
    df["macd_rel"] = generar_mod.calcular_macd(df["y"]) / df["y"]
    df["rsi_norm"] = generar_mod.calcular_rsi(df["y"]) / 100
    df["vol_realizada"] = df["retorno_1d"].rolling(kelly_diario_mod.VENTANA_VOL).std()
    df = pd.merge_asof(df, macro[["ds", "copper"]], on="ds", direction="backward")
    df["copper_ret_1d"] = np.log(df["copper"] / df["copper"].shift(1))
    df["copper_mom_5d"] = (df["copper"] - df["copper"].shift(5)) / df["copper"].shift(5)
    df["y_next"] = df["y"].shift(-1)
    df["par"], df["nombre_par"], df["es_commodity"] = NOK_TICKER, NOK_NOMBRE, True

    return df.dropna().reset_index(drop=True)


if __name__ == "__main__":
    macro = features_mod.cargar_macro()
    panel_original = pd.read_csv(PANEL_ORIGINAL, parse_dates=["ds"])
    fila_nok = construir_fila_nok(macro)

    columnas = panel_original.columns.tolist()
    panel_extendido = pd.concat([panel_original, fila_nok[columnas]], ignore_index=True)
    panel_extendido.to_csv(PANEL_EXTENDIDO, index=False)

    print(f"Panel original: {len(panel_original)} filas, {panel_original['par'].nunique()} pares")
    print(f"NOK agregado: {len(fila_nok)} filas, {fila_nok['ds'].min().date()} a {fila_nok['ds'].max().date()}")
    print(f"Panel extendido: {len(panel_extendido)} filas, {panel_extendido['par'].nunique()} pares -> {PANEL_EXTENDIDO}")
    print(sorted(panel_extendido["par"].unique()))
