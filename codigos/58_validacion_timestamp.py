# CORRECCION (2026-09-23) - Errata, evidencia 1: ¿a que hora del reloj
# corresponde cada barra diaria, y donde queda la correlacion commodity->FX
# cuando las dos series se comparan en el MISMO reloj?
#
# Consolida (y deja reproducible, sin red) la evidencia de los scripts de la
# auditoria copiados en codigos/validacion_timestamp/ (t2c, t3, t3b, t4). Datos
# en datos/bases/validacion_timestamp/ (barras horarias Yahoo de 730 dias,
# FRED H.10, dolar observado del BCCh via mindicador.cl).
#
# Cuatro tablas (todas en datos/resultados/errata_timestamp_*.csv):
#   A. Hora UTC de las barras horarias que mejor reproduce el retorno de la
#      barra DIARIA de Yahoo (AUDUSD/NOK -> 00:00 UTC de D; HG=F -> ~17 UTC,
#      el settlement de ~13:00 ET). CLP=X horario es de mala calidad
#      (precios constantes por horas, par poco liquido) y no ajusta bien.
#   B. USD/CLP vs cobre: escaneo de rezagos con (1) la alineacion ORIGINAL del
#      proyecto, (2) la corregida de alineacion_temporal.py, (3) el dolar
#      observado del BCCh reetiquetado al dia de transaccion, (4) barras
#      horarias a la misma hora de reloj.
#   C. Validacion cruzada con una fuente independiente: FRED H.10 (tipo de
#      cambio al mediodia de Nueva York) para AUD/CAD/MXN (cobre), NOK (WTI),
#      ZAR (platino, oro) y BRL (soja) - correlacion contemporanea vs.
#      operable y Sharpe de la regla signo-del-commodity (signo fijado por
#      teoria, no estimado), bruto y neto de costos, con IC95.
#   D. Misma tabla C pero con Yahoo diario (corregido), para comparar fuentes.
#
# Convencion: todas las monedas en "USD por unidad de moneda extranjera"
# (commodity sube -> moneda commodity se aprecia -> retorno POSITIVO), igual
# que el panel de 23. Por eso el signo teorico de la regla es +1 en todos los
# casos (para USD/CLP crudo seria -1).

import importlib
import json

import numpy as np
import pandas as pd

alin = importlib.import_module("alineacion_temporal")
ce = importlib.import_module("costos_y_estadistica")
features_mod = importlib.import_module("19_features_nuevas_validacion")

BASES_DIR = "../datos/bases"
VAL_DIR = f"{BASES_DIR}/validacion_timestamp"
RESULTADOS_DIR = "../datos/resultados"


def cargar_horario(ticker):
    x = pd.read_csv(f"{VAL_DIR}/hourly_{ticker}.csv", index_col=0)
    x.index = pd.to_datetime(x.index, utc=True) + pd.Timedelta(hours=1)  # indice = FIN de la barra
    return x["Close"]


def cargar_diario_yahoo_corto(ticker):
    d = pd.read_csv(f"{VAL_DIR}/daily_{ticker}.csv", index_col=0)
    d.index = pd.to_datetime(pd.Series(d.index).str[:10].values)
    return d["Close"].iloc[:-1]  # la ultima barra puede estar incompleta


# ---------------------------------------------------------------------------
# A. hora real de la barra diaria
# ---------------------------------------------------------------------------
def tabla_hora_barra_diaria():
    filas = []
    for t in ["CLP=X", "AUDUSD=X", "NOK=X", "HG=F"]:
        d = cargar_diario_yahoo_corto(t)
        h = cargar_horario(t)
        d = d[d.index >= h.index.min().tz_localize(None) + pd.Timedelta(days=3)]
        dr = np.log(d).diff()
        mejor = None
        for off in range(-30, 31):
            T = pd.DatetimeIndex(d.index).tz_localize("UTC") + pd.Timedelta(hours=off)
            p = h.reindex(h.index.union(T)).ffill().reindex(T)
            s = pd.Series(np.log(p.values), index=d.index).diff()
            c = dr.corr(s)
            if mejor is None or c > mejor[1]:
                mejor = (off, c)
        filas.append({"ticker": t, "hora_utc_que_mejor_reproduce_la_barra_diaria": mejor[0],
                      "corr_retornos": mejor[1], "n_dias": len(d)})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# B. USD/CLP vs cobre en distintas alineaciones
# ---------------------------------------------------------------------------
def tabla_clp(macro):
    filas = []
    clp = pd.read_csv(f"{BASES_DIR}/usdclp_long.csv", parse_dates=["ds"])[["ds", "y"]]
    clp["retorno_1d"] = np.log(clp["y"] / clp["y"].shift(1))

    orig = alin.agregar_features_cobre_original(clp, macro)
    esc = alin.escaneo_rezagos_filas_fx(orig, "copper_ret_1d")
    filas.append({"alineacion": "Original del proyecto (merge fecha<=, 9.14)", "fuente_fx": "Yahoo CLP=X diario", **{f"rezago_{k}": v for k, v in esc.items()}})

    corr = alin.agregar_features_cobre(clp, macro)
    esc = alin.escaneo_rezagos_filas_fx(corr, "copper_ret_1d")
    filas.append({"alineacion": "Corregida (settlement < timestamp FX; rezago 0=contemporaneo, +1=operable)", "fuente_fx": "Yahoo CLP=X diario", **{f"rezago_{k}": v for k, v in esc.items()}})

    # dolar observado: valor publicado en t = promedio de transacciones del dia habil anterior
    m = json.load(open(f"{VAL_DIR}/mindicador_dolar.json"))
    ob = pd.Series({pd.Timestamp(e["fecha"][:10]): e["valor"] for e in m}).sort_index()
    ob = ob[~ob.index.duplicated()]
    ob_rel = pd.Series(ob.values[1:], index=ob.index[:-1])
    cu = alin.serie_commodity_dias_habiles(macro).set_index("ds")["precio"]
    for lab, ser in [("Dolar observado BCCh, fecha de publicacion", ob), ("Dolar observado BCCh, reetiquetado al dia de transaccion", ob_rel)]:
        df = pd.concat([cu.rename("c"), ser.rename("f")], axis=1).dropna()
        r = np.log(df).diff().dropna()
        esc = {k: r["c"].corr(r["f"].shift(-k)) for k in range(-2, 4)}
        filas.append({"alineacion": lab + " (promedio intradia: no operable)", "fuente_fx": "BCCh via mindicador.cl", **{f"rezago_{k}": v for k, v in esc.items()}})

    # barras horarias a la misma hora de reloj (730 dias)
    hg = cargar_horario("HG=F")
    fxh = cargar_horario("CLP=X")
    fechas = pd.bdate_range(max(fxh.index.min(), hg.index.min()).tz_localize(None).normalize() + pd.Timedelta(days=2),
                            cargar_diario_yahoo_corto("CLP=X").index.max())
    for H in [18, 20]:
        c = np.log(_asof(hg, fechas, H))
        f = np.log(_asof(fxh, fechas, H))
        r = pd.concat([c.rename("c"), f.rename("f")], axis=1).dropna().diff().dropna()
        esc = {k: r["c"].corr(r["f"].shift(-k)) for k in range(-2, 4)}
        filas.append({"alineacion": f"Barras horarias, ambas a las {H}:00 UTC (730 dias)", "fuente_fx": "Yahoo CLP=X horario", **{f"rezago_{k}": v for k, v in esc.items()}})
    return pd.DataFrame(filas)


def _asof(s, fechas, hora, maxgap=3):
    T = pd.DatetimeIndex(fechas).tz_localize("UTC") + pd.Timedelta(hours=hora)
    idx = s.index.searchsorted(T, side="right") - 1
    v = np.full(len(T), np.nan)
    ok = idx >= 0
    tt = s.index[np.clip(idx, 0, None)]
    bueno = ok & ((T - tt) <= pd.Timedelta(hours=maxgap))
    v[bueno] = s.values[idx[bueno]]
    return pd.Series(v, index=fechas)


# ---------------------------------------------------------------------------
# C/D. FRED H.10 vs Yahoo diario corregido
# ---------------------------------------------------------------------------
FRED = {  # moneda: (serie FRED, cotiza_por_usd, ticker/archivo Yahoo en el repo)
    "AUD": ("DEXUSAL", False, "AUDUSD=X"),
    "CAD": ("DEXCAUS", True, "CAD=X"),
    "MXN": ("DEXMXUS", True, "MXN=X"),
    "NOK": ("DEXNOUS", True, "USDNOK=X"),
    "ZAR": ("DEXSFUS", True, "ZAR=X"),
    "BRL": ("DEXBZUS", True, "BRL=X"),
}
PARES_VALIDACION = [("AUD", "HG_F"), ("CAD", "HG_F"), ("MXN", "HG_F"), ("NOK", "CL_F"), ("ZAR", "PL_F"), ("ZAR", "GC_F"), ("BRL", "ZS_F")]


def cargar_fred(moneda):
    serie, por_usd, _ = FRED[moneda]
    x = pd.read_csv(f"{VAL_DIR}/fred_{serie}.csv")
    x.columns = ["ds", "y"]
    x["ds"] = pd.to_datetime(x["ds"])
    x["y"] = pd.to_numeric(x["y"], errors="coerce")
    x = x.dropna()
    x = x[x["ds"] >= "2010-01-01"]
    if por_usd:
        x["y"] = 1 / x["y"]
    return x.reset_index(drop=True)


def cargar_yahoo_repo(moneda):
    ticker = FRED[moneda][2]
    if ticker == "USDNOK=X":
        x = pd.read_csv(f"{BASES_DIR}/USDNOK_X_long.csv", parse_dates=["ds"])
        x["y"] = 1 / x["y"]
        return x[["ds", "y"]]
    panel = pd.read_csv(f"{BASES_DIR}/panel_fx_diario.csv", parse_dates=["ds"])
    return panel[panel["par"] == ticker][["ds", "y"]].reset_index(drop=True)


def cargar_commodity(codigo, macro):
    if codigo == "HG_F":
        return alin.serie_commodity_dias_habiles(macro)
    x = pd.read_csv(f"{BASES_DIR}/{codigo}_long.csv", parse_dates=["ds"]).rename(columns={"y": "precio"})
    return x[x["precio"] > 0].reset_index(drop=True)  # WTI negativo del 2020-04-20: log indefinido, se excluye ese dia


def fila_validacion(moneda, commodity, fx, cm, ts_fx, fuente):
    tab = alin.tabla_senal_operacion(fx, cm, commodity, ts_fx=ts_fx)
    esc = alin.escaneo_rezagos_tabla(tab)
    sub = tab.dropna(subset=["cm_ret", "fx_ret_operable"])
    ret_op = np.expm1(sub["fx_ret_operable"].to_numpy())
    pos = np.sign(sub["cm_ret"].to_numpy())  # signo teorico +1 (convencion USD por unidad)
    spread = ce.SPREAD_IDA_VUELTA[moneda]
    r_bruto = ce.retornos_posicion_fija(pos, ret_op, 0.0)
    r_neto = ce.retornos_posicion_fija(pos, ret_op, spread)
    rb, rn = ce.resumen_sharpe(r_bruto), ce.resumen_sharpe(r_neto)
    return {
        "fuente_fx": fuente, "moneda": moneda, "commodity": commodity, "n": len(sub),
        "corr_contemporanea": sub["cm_ret"].corr(sub["fx_ret_contemp"]),
        "corr_operable": sub["cm_ret"].corr(sub["fx_ret_operable"]),
        **{f"rezago_{k}": v for k, v in esc.items()},
        "horas_mediana_settlement_a_entrada": sub["horas_hasta_entrada"].median(),
        "sharpe_bruto": rb["sharpe"], "ic95_bruto": f"[{rb['ic95_bajo']:.2f}, {rb['ic95_alto']:.2f}]",
        "spread_supuesto_%": 100 * spread,
        "sharpe_neto": rn["sharpe"], "ic95_neto": f"[{rn['ic95_bajo']:.2f}, {rn['ic95_alto']:.2f}]",
    }


if __name__ == "__main__":
    macro = features_mod.cargar_macro()

    ta = tabla_hora_barra_diaria()
    ta.to_csv(f"{RESULTADOS_DIR}/errata_timestamp_hora_barra_diaria.csv", index=False)
    print("=== A. Hora UTC que mejor reproduce la barra diaria de Yahoo ===")
    print(ta.round(3).to_string(index=False))

    tb = tabla_clp(macro)
    tb.to_csv(f"{RESULTADOS_DIR}/errata_timestamp_clp_rezagos.csv", index=False)
    print("\n=== B. USD/CLP (crudo, CLP por USD) vs cobre: escaneo de rezagos por alineacion ===")
    print(tb.round(3).to_string(index=False))

    filas = []
    for moneda, commodity in PARES_VALIDACION:
        cm = cargar_commodity(commodity, macro)
        filas.append(fila_validacion(moneda, commodity, cargar_fred(moneda), cm, alin.ts_fred_h10, "FRED H.10 (mediodia NY)"))
        filas.append(fila_validacion(moneda, commodity, cargar_yahoo_repo(moneda), cm, alin.ts_fx_yahoo, "Yahoo diario (corregido)"))
    tc = pd.DataFrame(filas)
    tc.to_csv(f"{RESULTADOS_DIR}/errata_timestamp_fred_vs_yahoo.csv", index=False)
    print("\n=== C/D. Commodity -> FX: FRED H.10 vs Yahoo, contemporaneo vs operable (signo teorico, sin estimar) ===")
    cols = ["fuente_fx", "moneda", "commodity", "n", "corr_contemporanea", "corr_operable", "horas_mediana_settlement_a_entrada",
            "sharpe_bruto", "ic95_bruto", "spread_supuesto_%", "sharpe_neto", "ic95_neto"]
    print(tc[cols].round(3).to_string(index=False))
