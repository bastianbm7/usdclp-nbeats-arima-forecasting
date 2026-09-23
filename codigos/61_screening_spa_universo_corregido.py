# CORRECCION (2026-09-23) - rehace 9.31 (Issue #13 Fase 1: screening NOK/ZAR/BRL
# x 5 commodities + bootstrap SPA). Reemplaza para efectos del paper a 52/53.
#
# Tres cambios, todos correcciones de metodo (no configuraciones nuevas):
#   1. Alineacion: el retorno del commodity de la fila FX D es el conocido
#      ANTES del precio FX de D (alineacion_temporal). En este marco el rezago
#      0 es el contemporaneo (NO operable) y los rezagos +1..+3 son operables
#      (se entra en un precio FX posterior al settlement que genero la senal).
#      En 52, el "rezago +1" que concentraba el efecto (NOK x WTI 0.275, etc.)
#      era el contemporaneo.
#   2. SPA sobre el UNIVERSO completo, no sobre los sobrevivientes: 53 aplico
#      SPA solo a las 10 estrategias que ya habian sobrevivido un screening de
#      90 pruebas en la MISMA muestra - eso no corrige el data-snooping de la
#      busqueda, lo hereda. Aca el universo es todo lo que se exploro como
#      estrategia operable: 15 pares moneda x commodity x 3 rezagos operables
#      (+1, +2, +3) x 2 signos = 90 estrategias (incluir ambos signos evita
#      que el signo elegido con toda la muestra entre gratis).
#   3. Costos: SPA se corre bruto y neto de spread ida+vuelta por moneda.

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch.bootstrap import SPA
from scipy import stats
from statsmodels.stats.multitest import multipletests

alin = importlib.import_module("alineacion_temporal")
ce = importlib.import_module("costos_y_estadistica")

BASES_DIR = "../datos/bases"
RESULTADOS_DIR = "../datos/resultados"
PANEL_ALINEADO = f"{BASES_DIR}/panel_fx_diario_alineado.csv"  # generado por 60

UMBRAL_CORR = 0.11
ALPHA_FDR = 0.05
LAGS = list(range(-2, 4))
LAGS_OPERABLES = [1, 2, 3]
MONEDAS = {"NOK": "USDNOK=X", "ZAR": "ZAR=X", "BRL": "BRL=X"}
COMMODITIES = {"CL_F": "WTI (petroleo)", "GC_F": "Oro", "PL_F": "Platino", "ZS_F": "Soja", "TIO_F": "Hierro"}
SEED, REPS = 42, 5000


def cargar_moneda(codigo, panel=None):
    panel = pd.read_csv(PANEL_ALINEADO, parse_dates=["ds"]) if panel is None else panel
    g = panel[panel["par"] == MONEDAS[codigo]].sort_values("ds").reset_index(drop=True)
    return g[["ds", "y", "retorno_1d"]]


def cargar_commodity(codigo):
    x = pd.read_csv(f"{BASES_DIR}/{codigo}_long.csv", parse_dates=["ds"]).rename(columns={"y": "precio"})
    return x[x["precio"] > 0].sort_values("ds").reset_index(drop=True)


def construir_par(moneda_df, commodity_df, codigo_commodity, corregido=True):
    if corregido:
        df = alin.agregar_commodity_conocido(moneda_df, commodity_df, codigo_commodity, "precio_commodity")
    else:  # 52 original: merge_asof fecha <=
        df = pd.merge_asof(moneda_df.sort_values("ds"), commodity_df.rename(columns={"precio": "precio_commodity"}), on="ds", direction="backward")
    df["commodity_ret_1d"] = np.log(df["precio_commodity"] / df["precio_commodity"].shift(1))
    df = df.dropna(subset=["commodity_ret_1d", "retorno_1d"]).reset_index(drop=True)
    df["ret_simple"] = np.expm1(df["retorno_1d"])
    return df


def escaneo(df, moneda, commodity):
    filas = []
    for lag in LAGS:
        fut = df["retorno_1d"].shift(-lag)
        v = fut.notna() & df["commodity_ret_1d"].notna()
        r, p = stats.pearsonr(df.loc[v, "commodity_ret_1d"], fut[v])
        filas.append({"moneda": moneda, "commodity": commodity, "rezago": lag, "operable": lag >= 1,
                      "correlacion": r, "p_valor": p, "n_obs": int(v.sum())})
    return filas


def retornos_estrategia(df, lag, signo, spread):
    pos = signo * np.sign(df["commodity_ret_1d"].to_numpy())
    ret = df["ret_simple"].shift(-lag).to_numpy()
    r = ce.retornos_posicion_fija(pos, np.nan_to_num(ret), spread)
    r[np.isnan(ret)] = np.nan
    return pd.Series(r, index=df["ds"])


def correr_spa(universo, etiqueta):
    modelos = pd.DataFrame(universo).dropna()
    bench = np.zeros(len(modelos))
    spa = SPA(-bench, -modelos, reps=REPS, seed=SEED)
    spa.compute()
    mejores = modelos.columns[spa.better_models(pvalue=0.05, pvalue_type="consistent")]
    sh = modelos.mean() / modelos.std() * np.sqrt(252)
    return {"universo": etiqueta, "n_estrategias": modelos.shape[1], "T": len(modelos),
            "p_lower": spa.pvalues["lower"], "p_consistent": spa.pvalues["consistent"], "p_upper": spa.pvalues["upper"],
            "n_mejores_que_no_operar_(consistent,5%)": len(mejores), "mejores": ";".join(mejores),
            "sharpe_max_del_universo": sh.max(), "estrategia_sharpe_max": sh.idxmax(),
            "sharpe_max_esperado_bajo_nulo": ce.sharpe_maximo_esperado_nulo(modelos.shape[1], len(modelos))}


def graficar(tabla, path):
    pv = tabla.pivot_table(index=["moneda", "commodity"], columns="rezago", values="correlacion")
    sig = tabla.pivot_table(index=["moneda", "commodity"], columns="rezago", values="sobrevive_fdr")
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(pv.values, cmap="RdBu_r", vmin=-0.35, vmax=0.35, aspect="auto")
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels([f"{c}\n{'contemp.' if c == 0 else ('operable' if c >= 1 else 'FX lidera')}" for c in pv.columns], fontsize=8)
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels([f"{m} x {c}" for m, c in pv.index])
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i, j]:.2f}{'*' if sig.values[i, j] else ''}", ha="center", va="center", fontsize=7)
    ax.set_title("Screening CORREGIDO (Issue #13 Fase 1): rezago 0 = contemporaneo, +1..+3 = operables\n* = sobrevive FDR (BH, 5%) sobre las 90 pruebas")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    panel = pd.read_csv(PANEL_ALINEADO, parse_dates=["ds"])
    monedas = {m: cargar_moneda(m, panel) for m in MONEDAS}
    comms = {c: cargar_commodity(c) for c in COMMODITIES}

    filas = []
    pares = {}
    for m, mdf in monedas.items():
        for c, nombre in COMMODITIES.items():
            df = construir_par(mdf, comms[c], c)
            pares[(m, c)] = df
            filas.extend(escaneo(df, m, nombre))
    tabla = pd.DataFrame(filas)
    rech, pfdr, _, _ = multipletests(tabla["p_valor"], alpha=ALPHA_FDR, method="fdr_bh")
    tabla["p_valor_fdr"], tabla["sobrevive_fdr"] = pfdr, rech
    tabla["supera_umbral_r"] = tabla["correlacion"].abs() >= UMBRAL_CORR
    tabla["hallazgo_operable"] = tabla["sobrevive_fdr"] & tabla["supera_umbral_r"] & tabla["operable"]
    tabla.sort_values("p_valor").to_csv(f"{RESULTADOS_DIR}/correccion_fase1_screening_completo.csv", index=False)
    graficar(tabla, f"{RESULTADOS_DIR}/correccion_fase1_screening_heatmap.png")

    print("=== Resumen por rezago (90 pruebas, alineacion corregida) ===")
    res = tabla.groupby("rezago").agg(corr_media=("correlacion", "mean"), corr_max_abs=("correlacion", lambda s: s.abs().max()),
                                      sobreviven_fdr=("sobrevive_fdr", "sum"), sobreviven_fdr_y_umbral=("supera_umbral_r", lambda s: int((s & tabla.loc[s.index, "sobrevive_fdr"]).sum())))
    print(res.round(3).to_string())
    print("\n=== Sobreviven FDR (cualquier rezago) ===")
    print(tabla[tabla["sobrevive_fdr"]].sort_values(["rezago", "p_valor"]).round(4).to_string(index=False))
    print(f"\nHallazgos OPERABLES (FDR + |r|>={UMBRAL_CORR} + rezago>=1): {int(tabla['hallazgo_operable'].sum())}")

    # --- SPA sobre el universo completo de 90 estrategias operables ---
    filas_spa = []
    for etiqueta, usar_costos in [("bruto (spread 0)", False), ("neto (spread ida+vuelta por moneda)", True)]:
        universo = {}
        for (m, c), df in pares.items():
            spread = ce.SPREAD_IDA_VUELTA[m] if usar_costos else 0.0
            for lag in LAGS_OPERABLES:
                for signo in (1, -1):
                    universo[f"{m}x{c}_rez{lag}_{'+' if signo > 0 else '-'}"] = retornos_estrategia(df, lag, signo, spread)
        filas_spa.append(correr_spa(universo, etiqueta))
    spa = pd.DataFrame(filas_spa)
    spa.to_csv(f"{RESULTADOS_DIR}/correccion_fase1_spa_universo.csv", index=False)
    print("\n=== SPA de Hansen sobre el universo completo (90 estrategias operables) ===")
    print(spa.drop(columns=["mejores"]).round(4).to_string(index=False))
    print("Mejores que no operar:", {r["universo"]: r["mejores"] for r in filas_spa})

    # --- los 10 'sobrevivientes' originales de 53, evaluados en el marco operable ---
    orig = pd.read_csv(f"{RESULTADOS_DIR}/fase1_supervivientes_fdr.csv")
    orig = orig[orig["rezago"] == 1]
    cod = {v: k for k, v in COMMODITIES.items()}
    filas10 = []
    for _, f in orig.iterrows():
        df = pares[(f["moneda"], cod[f["commodity"]])]
        for costo, spread in [("bruto", 0.0), ("neto", ce.SPREAD_IDA_VUELTA[f["moneda"]])]:
            r = retornos_estrategia(df, 1, np.sign(f["correlacion"]), spread).dropna().to_numpy()
            filas10.append({"moneda": f["moneda"], "commodity": f["commodity"], "corr_original_reportada": f["correlacion"],
                            "costo": costo, **ce.resumen_sharpe(r)})
    t10 = pd.DataFrame(filas10)
    t10.to_csv(f"{RESULTADOS_DIR}/correccion_fase1_sobrevivientes_originales_operables.csv", index=False)
    print("\n=== Los 10 sobrevivientes de 53, con la regla ejecutable (rezago +1 corregido, signo original) ===")
    print(t10.round(3).to_string(index=False))
