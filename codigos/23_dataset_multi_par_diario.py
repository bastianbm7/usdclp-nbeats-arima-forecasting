# INVALIDADO / SUPERADO (2026-09-23) - panel de 13 pares, correlacion por par, backtest por par y pooled vs solo-CLP: ver script 60 (panel_fx_diario_alineado.csv).
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
# Pedido explicito de Bastian tras validar el hallazgo de 9.13 (copper_ret_1d
# a frecuencia diaria): aumentar la cantidad de datos incorporando varias
# series de forex (no solo USD/CLP) - minimo 10 pares. Sirve para 2 cosas:
#
# 1. Generar el panel de datos en si (datos/bases/panel_fx_diario.csv),
#    reusable para lo que siga (Tier 3/4 de la hoja de ruta del radar-baseline).
# 2. Probar directo la hipotesis de X-Trend (Wood et al. 2023, ver Artifact
#    "Radar USD/CLP"): ¿generaliza el efecto cobre-CLP a otras monedas
#    commodity? ¿un modelo entrenado con el panel completo (mas datos, mas
#    variedad) predice MEJOR el retorno de USD/CLP en el mismo periodo de test
#    que un modelo entrenado solo con la historia de USD/CLP (22_kelly_diario_cobre.py)?
#
# 13 pares (>= 10 pedidos): 6 emergentes/commodity (CLP, MXN, BRL, COP, PEN,
# ZAR - todas expuestas a cobre/petroleo/oro en distinto grado) + 7 G10
# (JPY, CAD, CHF, EUR, GBP, AUD, NZD) como grupo de comparacion no-commodity.
#
# Todas las series se normalizan a "USD por 1 unidad de moneda extranjera"
# (invertir las cotizadas como "moneda por 1 USD") para que un retorno
# positivo signifique lo mismo en todo el panel: la moneda extranjera se
# aprecia frente al USD. Sin esto, pooling cruzado no tiene sentido.

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

features_mod = importlib.import_module("19_features_nuevas_validacion")
generar_mod = importlib.import_module("10_generar_dataset_rl")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")
kelly_diario_mod = importlib.import_module("22_kelly_diario_cobre")
obtener_datos_mod = importlib.import_module("01_obtener_datos")

RESULTADOS_DIR = "../datos/resultados"
BASES_DIR = "../datos/bases"
PANEL_CACHE = f"{BASES_DIR}/panel_fx_diario.csv"
FECHA_INICIO = "2010-01-01"

# ticker -> (nombre para mostrar, es_cotizada_por_usd, es_commodity)
# Commodity = economia cuya exportacion dominante es una materia prima
# (Chen & Rogoff 2003): cobre (CLP, PEN), petroleo (CAD, MXN, COP), hierro/
# carbon (AUD), lacteos/agro (NZD), oro/platino (ZAR), canasta amplia de
# commodities (BRL). JPY/CHF (refugio) y EUR/GBP (economias diversificadas,
# centros financieros) quedan como grupo de comparacion no-commodity.
PARES = {
    "CLP=X": ("USD/CLP", True, True),
    "MXN=X": ("USD/MXN", True, True),
    "BRL=X": ("USD/BRL", True, True),
    "COP=X": ("USD/COP", True, True),
    "PEN=X": ("USD/PEN", True, True),
    "ZAR=X": ("USD/ZAR", True, True),
    "CAD=X": ("USD/CAD", True, True),
    "AUDUSD=X": ("AUD/USD", False, True),
    "NZDUSD=X": ("NZD/USD", False, True),
    "JPY=X": ("USD/JPY", True, False),
    "CHF=X": ("USD/CHF", True, False),
    "EURUSD=X": ("EUR/USD", False, False),
    "GBPUSD=X": ("GBP/USD", False, False),
}

N_WINDOWS_WF = kelly_diario_mod.N_WINDOWS_WF
N_TEST_POR_VENTANA = kelly_diario_mod.N_TEST_POR_VENTANA
FEATURES_DIARIAS = kelly_diario_mod.FEATURES_DIARIAS


def descargar_panel(macro):
    partes = []
    for ticker, (nombre, es_cotizada_por_usd, es_commodity) in PARES.items():
        precios = yf.download(ticker, start=FECHA_INICIO, progress=False)[["Close"]]
        precios.columns = ["y"]  # nombre esperado por limpiar_ticks_erroneos
        precios = precios.reset_index().rename(columns={"Date": "ds"})
        precios["ds"] = pd.to_datetime(precios["ds"]).dt.tz_localize(None)
        # Mismo bug de yfinance ya documentado en 01_obtener_datos.py para
        # CLP=X (ticks corruptos de un solo dia, ej. y=5.46 en vez de ~544) -
        # aparece tambien en COP=X (22 dias), PEN=X, ZAR=X y CHF=X al bajar
        # estos 13 pares frescos - se limpia con la MISMA funcion, no una
        # nueva, antes de invertir la cotizacion.
        precios = obtener_datos_mod.limpiar_ticks_erroneos(precios)
        precios["y"] = 1 / precios["y"] if es_cotizada_por_usd else precios["y"]

        df = precios[["ds", "y"]].sort_values("ds").reset_index(drop=True)
        df["retorno_1d"] = np.log(df["y"] / df["y"].shift(1))
        df["macd_rel"] = generar_mod.calcular_macd(df["y"]) / df["y"]
        df["rsi_norm"] = generar_mod.calcular_rsi(df["y"]) / 100
        df["vol_realizada"] = df["retorno_1d"].rolling(kelly_diario_mod.VENTANA_VOL).std()
        df = pd.merge_asof(df, macro[["ds", "copper"]], on="ds", direction="backward")
        df["copper_ret_1d"] = np.log(df["copper"] / df["copper"].shift(1))
        df["copper_mom_5d"] = (df["copper"] - df["copper"].shift(5)) / df["copper"].shift(5)
        df["y_next"] = df["y"].shift(-1)
        df["par"], df["nombre_par"], df["es_commodity"] = ticker, nombre, es_commodity
        partes.append(df.dropna().reset_index(drop=True))

    panel = pd.concat(partes, ignore_index=True)
    panel.to_csv(PANEL_CACHE, index=False)
    return panel


def cargar_panel(macro):
    try:
        panel = pd.read_csv(PANEL_CACHE, parse_dates=["ds"])
        print(f"Panel: usando cache {PANEL_CACHE} ({len(panel)} filas, {panel['par'].nunique()} pares)")
        return panel
    except FileNotFoundError:
        print("Panel: sin cache, descargando 13 pares via yfinance...")
        return descargar_panel(macro)


def correlacion_por_par(panel):
    filas = []
    for ticker, grupo in panel.groupby("par"):
        retorno_futuro = (grupo["y_next"] - grupo["y"]) / grupo["y"]
        corr = grupo["copper_ret_1d"].corr(retorno_futuro)
        filas.append({"par": PARES[ticker][0], "es_commodity": PARES[ticker][2],
                       "correlacion_copper_ret_1d": corr, "n_obs": len(grupo)})
    return pd.DataFrame(filas).sort_values("correlacion_copper_ret_1d").reset_index(drop=True)


def graficar_correlacion_por_par(tabla, path_salida):
    fig, ax = plt.subplots(figsize=(9, 6))
    colores = ["darkorange" if c else "steelblue" for c in tabla["es_commodity"]]
    ax.barh(tabla["par"], tabla["correlacion_copper_ret_1d"], color=colores)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.11, color="gray", linewidth=0.6, linestyle="--")
    ax.axvline(-0.11, color="gray", linewidth=0.6, linestyle="--")
    for etiqueta, color in [("Commodity/emergente", "darkorange"), ("G10 no-commodity", "steelblue")]:
        ax.barh([], [], color=color, label=etiqueta)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("Correlacion de copper_ret_1d con el retorno futuro del par")
    ax.set_title("¿El efecto cobre-USD/CLP generaliza a otras monedas? (13 pares, mismo chequeo de 9.13)")
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


def _posiciones_desde_coef(df_test, coef):
    X = np.column_stack([np.ones(len(df_test)), df_test[FEATURES_DIARIAS].to_numpy(dtype=float)])
    mu_pred = X @ coef
    sigma2 = df_test["vol_realizada"].to_numpy(dtype=float) ** 2
    f = np.divide(mu_pred, sigma2, out=np.zeros_like(mu_pred), where=sigma2 > 0)
    return np.clip(f, -kelly_diario_mod.LIMITE_APALANCAMIENTO, kelly_diario_mod.LIMITE_APALANCAMIENTO)


def entrenar_pooled_vs_single(panel):
    # Comparacion apples-to-apples con 22_kelly_diario_cobre.py: MISMO
    # esquema de walk-forward (5 ventanas, reentrena en cada corte) para las
    # dos variantes - lo unico que cambia es de donde sale la data de
    # entrenamiento en cada ventana (solo historia de CLP vs. panel completo
    # de 13 pares hasta esa misma fecha de corte, sin look-ahead). Es la
    # prueba directa de la hipotesis de X-Trend: ¿mas datos de otras monedas
    # generaliza mejor para CLP que su propia historia sola?
    clp = panel[panel["par"] == "CLP=X"].sort_values("ds").reset_index(drop=True)

    capital_single = capital_pooled = kelly_diario_mod.CAPITAL_INICIAL
    partes_single, partes_pooled = [], []
    for clp_train, clp_test in wf_mod.ventanas_walkforward(clp, N_WINDOWS_WF, N_TEST_POR_VENTANA):
        fecha_corte = clp_test["ds"].iloc[0]
        panel_train_pooled = panel[panel["ds"] < fecha_corte]

        coef_single = kelly_diario_mod.ajustar_regresion(clp_train, FEATURES_DIARIAS)
        coef_pooled = kelly_diario_mod.ajustar_regresion(panel_train_pooled, FEATURES_DIARIAS)

        r_single = kelly_diario_mod.simular(clp_test, _posiciones_desde_coef(clp_test, coef_single), "Kelly diario (solo CLP)", capital_single)
        capital_single = r_single["capital"].iloc[-1]
        partes_single.append(r_single)

        r_pooled = kelly_diario_mod.simular(clp_test, _posiciones_desde_coef(clp_test, coef_pooled), "Kelly diario (panel pooled, 13 pares)", capital_pooled)
        capital_pooled = r_pooled["capital"].iloc[-1]
        partes_pooled.append(r_pooled)

    print(f"Filas de entrenamiento en la ultima ventana -> solo CLP: {len(clp_train)} | "
          f"panel pooled (13 pares): {len(panel_train_pooled)}")

    return {"Kelly diario (solo CLP)": pd.concat(partes_single, ignore_index=True),
            "Kelly diario (panel pooled, 13 pares)": pd.concat(partes_pooled, ignore_index=True)}


def graficar_pooled_vs_single(resultados, path_salida):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colores = {"Kelly diario (solo CLP)": "crimson", "Kelly diario (panel pooled, 13 pares)": "steelblue"}
    for nombre, r in resultados.items():
        ax.plot(r["ds"], r["capital"], label=nombre, color=colores[nombre], linewidth=1.8)
    ax.axhline(kelly_diario_mod.CAPITAL_INICIAL, color="gray", linestyle=":", linewidth=1, label="Capital inicial ($100)")
    ax.set_ylabel("Capital ($)")
    ax.set_title("¿Ayuda entrenar con el panel de 13 monedas en vez de solo la historia de CLP?")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    macro = features_mod.cargar_macro()
    panel = cargar_panel(macro)
    print(f"\nPanel completo: {len(panel)} obs, {panel['par'].nunique()} pares, "
          f"{panel['ds'].min().date()} a {panel['ds'].max().date()}\n")

    tabla_corr = correlacion_por_par(panel)
    tabla_corr.to_csv(f"{RESULTADOS_DIR}/panel_fx_correlacion_cobre_por_par.csv", index=False)
    graficar_correlacion_por_par(tabla_corr, f"{RESULTADOS_DIR}/panel_fx_correlacion_cobre_por_par.png")
    print("=== Correlacion de copper_ret_1d con el retorno futuro, por par ===")
    print(tabla_corr.to_string(index=False))

    resultados = entrenar_pooled_vs_single(panel)
    filas_metricas = [kelly_diario_mod.calcular_metricas(r, nombre) for nombre, r in resultados.items()]
    referencia = pd.read_csv(f"{RESULTADOS_DIR}/kelly_diario_cobre_metricas.csv")
    tabla = pd.concat([pd.DataFrame(filas_metricas), referencia], ignore_index=True).sort_values("sharpe_anualizado", ascending=False).reset_index(drop=True)
    tabla.to_csv(f"{RESULTADOS_DIR}/panel_fx_pooled_vs_single_metricas.csv", index=False)
    for nombre, r in resultados.items():
        slug = nombre.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        r.to_csv(f"{RESULTADOS_DIR}/panel_fx_{slug}_operaciones.csv", index=False)

    graficar_pooled_vs_single(resultados, f"{RESULTADOS_DIR}/panel_fx_pooled_vs_single_curva_capital.png")

    print(f"\n=== Pooled (13 pares) vs. solo CLP, evaluado sobre el mismo test de USD/CLP ===\n")
    print(tabla.to_string(index=False))
