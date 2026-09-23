# Issue #13, Fase 0: antes de invertir en screening de correlacion o
# walk-forward con commodities nuevos, verificar que la data de yfinance sea
# de verdad liquida (un dato por dia habil, no un future poco transado que
# publica un precio cada tanto). Mismo criterio de "barato antes de caro" que
# el resto del proyecto - la investigacion previa de este mismo repo ya
# descarto litio (LTH=F) por tener ~1 dato/mes en vez de ~21 (los dias
# habiles de un mes tipico), y no tiene sentido correr screening/backtest
# sobre una serie que no es realmente diaria.
#
# Commodities nuevos a evaluar (via yfinance):
#   - WTI (CL=F), oro (GC=F), platino (PL=F), soja (ZS=F), hierro (TIO=F)
#   - NOK (USDNOK=X) - la moneda nueva del panel FX (junto a BRL/ZAR que ya
#     estan en datos/bases/panel_fx_diario.csv, no se redescargan)
#
# Ademas: BRL=X y ZAR=X ya estan en el panel existente (23_dataset_multi_par_diario.py
# ya les aplico limpiar_ticks_erroneos al construirlo) - este script re-verifica
# que no les haya quedado ningun tick corrupto sin limpiar, sin volver a
# descargarlos.
#
# Umbral de liquidez: un future/par FX liquido cotiza ~21 dias habiles/mes.
# Se descarta si el promedio de puntos/mes cae por debajo de 10 (menos de la
# mitad de lo esperado) - un piso deliberadamente muy por encima del ~1/mes
# que goleo al litio, para no repetir ese error por el lado optimista (aceptar
# algo casi tan iliquido como el litio solo porque no es EXACTAMENTE 1/mes).

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

obtener_datos_mod = importlib.import_module("01_obtener_datos")


def limpiar_ticks_erroneos_commodity(df, umbral=0.3):
    # Reusa la logica de 01_obtener_datos.limpiar_ticks_erroneos (caida+rebote
    # > umbral contra ambos vecinos = tick corrupto de un solo dia), PERO con
    # una guarda extra: y > 0. Se necesita porque, al correr esto sobre CL=F
    # (WTI), la version original de la funcion marco el 2020-04-20 como "tick
    # corrupto" e iba a interpolarlo - ese dato (-37.63) NO es un error de
    # yfinance, es el colapso real a precio negativo del WTI durante la
    # pandemia (verificado contra el crudo de yfinance directamente, ese valor
    # es real y esta documentado publicamente). El bug de ticks corruptos que
    # esta funcion corrige siempre produce un valor chico pero POSITIVO (ej.
    # 5.46 en vez de 544) - nunca negativo - asi que exigir y > 0 preserva
    # exactamente el mismo comportamiento para los pares FX (nunca negativos)
    # y evita borrar un evento de mercado real y unico en la historia del
    # petroleo. Hallazgo documentado en el paper (Fase 0, Issue #13).
    y = df["y"]
    ratio_prev = y / y.shift(1)
    ratio_next = y / y.shift(-1)
    es_tick_malo = (ratio_prev < umbral) & (ratio_next < umbral) & (y > 0)
    if es_tick_malo.any():
        fechas_malas = df.loc[es_tick_malo, "ds"].dt.date.tolist()
        print(f"Ticks corruptos detectados y corregidos (interpolados): {fechas_malas}")
        df.loc[es_tick_malo, "y"] = None
        df["y"] = df["y"].interpolate()
    return df

BASES_DIR = "../datos/bases"
RESULTADOS_DIR = "../datos/resultados"
PANEL_CACHE = f"{BASES_DIR}/panel_fx_diario.csv"

FECHA_INICIO_LIQUIDEZ = "2025-09-01"  # ~12 meses de historia reciente, alcanza para el chequeo de liquidez
FECHA_INICIO_HISTORIA = "2010-01-01"  # historia completa, para las fases 1-3 si el ticker sobrevive

UMBRAL_PUNTOS_MES = 10  # ver nota de umbral arriba

CANDIDATOS = {
    "CL=F": "WTI (petroleo)",
    "GC=F": "Oro",
    "PL=F": "Platino",
    "ZS=F": "Soja",
    "TIO=F": "Hierro",
    "USDNOK=X": "USD/NOK",
}


def descargar_y_limpiar(ticker, start):
    precios = yf.download(ticker, start=start, progress=False)
    if precios.empty:
        return pd.DataFrame(columns=["ds", "y"])
    precios = precios[["Close"]]
    precios.columns = ["y"]  # nombre esperado por limpiar_ticks_erroneos
    precios = precios.reset_index().rename(columns={"Date": "ds"})
    precios["ds"] = pd.to_datetime(precios["ds"]).dt.tz_localize(None)
    precios = limpiar_ticks_erroneos_commodity(precios)
    return precios.sort_values("ds").reset_index(drop=True)


def puntos_por_mes(df):
    conteo = df.set_index("ds").resample("ME").size()
    return conteo


def evaluar_liquidez(ticker, nombre, df_reciente):
    conteo_mensual = puntos_por_mes(df_reciente)
    # se descartan el primer y ultimo mes (parciales por el corte de fecha,
    # no reflejan la liquidez real del instrumento)
    conteo_meses_completos = conteo_mensual.iloc[1:-1] if len(conteo_mensual) > 2 else conteo_mensual
    promedio = conteo_meses_completos.mean() if len(conteo_meses_completos) else 0.0
    minimo = conteo_meses_completos.min() if len(conteo_meses_completos) else 0
    sobrevive = promedio >= UMBRAL_PUNTOS_MES
    return {
        "ticker": ticker,
        "nombre": nombre,
        "n_obs_12m": len(df_reciente),
        "meses_evaluados": len(conteo_meses_completos),
        "puntos_por_mes_promedio": round(promedio, 1),
        "puntos_por_mes_minimo": int(minimo) if len(conteo_meses_completos) else 0,
        "sobrevive_fase0": sobrevive,
    }


def reverificar_ticks_panel(ticker):
    # BRL=X y ZAR=X ya estan en el panel (23_dataset_multi_par_diario.py les
    # aplico limpiar_ticks_erroneos al construirlo) - esto solo re-chequea que
    # no haya quedado ningun salto+rebote sospechoso sin limpiar, sobre la
    # serie 'y' ya invertida (USD por unidad de moneda) que quedo en el panel.
    panel = pd.read_csv(PANEL_CACHE, parse_dates=["ds"])
    serie = panel[panel["par"] == ticker].sort_values("ds").reset_index(drop=True)
    y = serie["y"]
    ratio_prev = y / y.shift(1)
    ratio_next = y / y.shift(-1)
    umbral = 0.3
    sospechosos = (ratio_prev < umbral) & (ratio_next < umbral) | (ratio_prev > 1 / umbral) & (ratio_next > 1 / umbral)
    n_sospechosos = int(sospechosos.sum())
    return {"ticker": ticker, "n_obs": len(serie), "ticks_sospechosos_residuales": n_sospechosos}


def graficar_liquidez(tabla, path_salida):
    fig, ax = plt.subplots(figsize=(9, 5))
    colores = ["seagreen" if s else "firebrick" for s in tabla["sobrevive_fase0"]]
    ax.barh(tabla["nombre"], tabla["puntos_por_mes_promedio"], color=colores)
    ax.axvline(UMBRAL_PUNTOS_MES, color="gray", linewidth=0.8, linestyle="--", label=f"umbral = {UMBRAL_PUNTOS_MES} pts/mes")
    ax.axvline(21, color="black", linewidth=0.6, linestyle=":", label="~21 dias habiles/mes (liquido ideal)")
    ax.set_xlabel("Puntos promedio por mes (ultimos ~12 meses, meses completos)")
    ax.set_title("Fase 0 (Issue #13): liquidez real de commodities/NOK candidatos")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    filas = []
    historias_completas = {}
    for ticker, nombre in CANDIDATOS.items():
        reciente = descargar_y_limpiar(ticker, FECHA_INICIO_LIQUIDEZ)
        fila = evaluar_liquidez(ticker, nombre, reciente)
        filas.append(fila)
        print(f"{nombre} ({ticker}): {fila['n_obs_12m']} obs en ~12m, "
              f"{fila['puntos_por_mes_promedio']} pts/mes promedio "
              f"(min {fila['puntos_por_mes_minimo']}) -> "
              f"{'SOBREVIVE' if fila['sobrevive_fase0'] else 'DESCARTADO'} fase 0")

        if fila["sobrevive_fase0"]:
            # solo se descarga la historia completa (2010-) para los que
            # pasan el filtro de liquidez - no tiene sentido bajar 15 anios de
            # un ticker que ya se va a descartar
            completa = descargar_y_limpiar(ticker, FECHA_INICIO_HISTORIA)
            historias_completas[ticker] = completa
            completa.to_csv(f"{BASES_DIR}/{ticker.replace('=', '_')}_long.csv", index=False)
            print(f"  -> historia completa guardada: {len(completa)} obs, "
                  f"{completa['ds'].min().date()} a {completa['ds'].max().date()}")

    tabla = pd.DataFrame(filas).sort_values("puntos_por_mes_promedio", ascending=False).reset_index(drop=True)
    tabla.to_csv(f"{RESULTADOS_DIR}/fase0_liquidez_commodities_nok.csv", index=False)
    graficar_liquidez(tabla, f"{RESULTADOS_DIR}/fase0_liquidez_commodities_nok.png")

    print("\n=== Resumen Fase 0 ===")
    print(tabla.to_string(index=False))

    print("\n=== Re-verificacion de ticks corruptos residuales en BRL=X y ZAR=X (ya en el panel) ===")
    filas_reverif = [reverificar_ticks_panel(t) for t in ["BRL=X", "ZAR=X"]]
    tabla_reverif = pd.DataFrame(filas_reverif)
    tabla_reverif.to_csv(f"{RESULTADOS_DIR}/fase0_reverificacion_ticks_brl_zar.csv", index=False)
    print(tabla_reverif.to_string(index=False))

    sobrevivientes = tabla[tabla["sobrevive_fase0"]]["ticker"].tolist()
    print(f"\nSobreviven Fase 0 ({len(sobrevivientes)}/{len(CANDIDATOS)}): {sobrevivientes}")
