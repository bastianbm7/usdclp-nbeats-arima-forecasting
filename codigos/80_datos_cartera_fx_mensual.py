# Issue #15 - paso 1: panel MENSUAL de spot + tasas cortas para una cartera
# de primas de riesgo FX (carry + momentum de serie de tiempo), 14 monedas
# contra USD, incluida CLP.
#
# Correr desde codigos/:  python 80_datos_cartera_fx_mensual.py [--refrescar]
#
# FUENTES Y TIMESTAMPS (la leccion de la errata 9.35: cada precio en su hora real)
#   - Spot de 13 monedas: FRED H.10 (Federal Reserve, "noon buying rates in New
#     York"), timestamp = 12:00 ET de la fecha de la fila
#     (alineacion_temporal.ts_fred_h10). Convencion del proyecto: USD por
#     unidad de moneda extranjera (sube = la moneda extranjera se aprecia).
#   - CLP no esta en H.10: Yahoo CLP=X, timestamp = 00:00 UTC de la fecha de
#     la fila (~20:00 NY del dia habil anterior; alineacion_temporal.
#     ts_fx_yahoo). Se eliminan precios repetidos (marcar_precios_repetidos) y
#     puntos corruptos: Yahoo tiene dias con el precio dividido por 100
#     (5.46 en 2014-04, 5.00 en 2016-12). Regla: se descarta la barra si se
#     desvia >5% del dolar observado del BCCh (mindicador.cl) de la fecha mas
#     cercana (+-3 dias). El dolar observado se usa SOLO como filtro y como
#     chequeo de cordura (correlacion de retornos mensuales), no como precio:
#     es un promedio de transacciones del dia anterior, no un precio operable.
#   - Tasas cortas: OECD MEI en FRED, interbancaria 3 meses (IR3TIB01xxM156N)
#     para 13 paises + EE.UU.; para Brasil no hay 3M en FRED y se usa la tasa
#     de call money (IRSTCI01BRM156N, ~Selic). Son PROMEDIOS MENSUALES en %
#     anual.
#
# REBALANCEO (fin de mes): d_t = ultima fecha con datos H.10 del mes, ejecucion
#   al precio H.10 de d_t (mediodia NY) o, para CLP, a la primera barra Yahoo
#   POSTERIOR a ese mediodia (~20:00 NY de d_t). Las SENALES usan solo precios
#   con timestamp ESTRICTAMENTE anterior al mediodia NY de d_t
#   (alineacion_temporal.valor_conocido): H.10 del dia habil anterior; para
#   CLP la barra Yahoo de 00:00 UTC de d_t (~20:00 NY de d_t-1).
#
# REZAGO DE PUBLICACION DE LAS TASAS (supuesto, documentado en el paper):
#   - Senal (ranking de carry): la tasa del mes M-2 para un rebalanceo a fin
#     del mes M (REZAGO_SENAL_MESES = 2). El 2026-09-23 el ultimo dato OECD en
#     FRED era el de 2026-08 para EE.UU. y ~2026-07/08 para la mayoria; zona
#     euro y Reino Unido estaban atrasados hasta 2026-01. Con 2 meses de
#     rezago el dato casi siempre existia en la fecha de la decision. Donde
#     falta (series atrasadas) se arrastra el ultimo valor hasta 6 meses:
#     conservador (dato viejo), nunca futuro.
#   - Devengo (retorno de carry que efectivamente paga la posicion de t a
#     t+1): la tasa del mes M. En la realidad ese retorno lo fija el precio
#     forward en d_t (conocido por el mercado en d_t); el promedio mensual del
#     mes M es el mejor proxy disponible y no contiene informacion posterior
#     a d_t. Nunca se usa para decidir.
#   - Interbancaria 3M como proxy de la prima forward a 1 mes: ignora
#     desviaciones de la paridad cubierta (CIP), que en KRW/BRL/CLP (mercados
#     NDF) y post-2008 en JPY/CHF/EUR no son despreciables. Limitacion.
#
# Salidas:
#   datos/bases/cartera_fx_fuentes/          CSV crudos (reproducibilidad sin red)
#   datos/bases/cartera_fx_diario.csv        precios diarios limpios, formato largo, con ts real
#   datos/bases/cartera_fx_mensual.csv       panel mensual (una fila por rebalanceo x moneda)
#   datos/bases/cartera_fx_riesgo_mercado.csv  S&P 500 y VIX a fin de mes (solo para correlaciones)
#   datos/resultados/issue15_validacion_datos.csv

import argparse
import importlib
import io
import json
import os
import subprocess
import warnings

import numpy as np
import pandas as pd

alin = importlib.import_module("alineacion_temporal")
warnings.filterwarnings("ignore", category=DeprecationWarning)

BASES_DIR = "../datos/bases"
RESULTADOS_DIR = "../datos/resultados"
FUENTES_DIR = f"{BASES_DIR}/cartera_fx_fuentes"
REZAGO_SENAL_MESES = 2
MAX_ARRASTRE_TASA_MESES = 6
MAX_DESVIO_OBSERVADO = 0.05
INICIO = pd.Timestamp("1997-01-01")

# codigo -> (serie spot H.10, True si la serie viene en unidades por USD, serie de tasa)
MONEDAS = {
    "EUR": ("DEXUSEU", False, "IR3TIB01EZM156N"),
    "GBP": ("DEXUSUK", False, "IR3TIB01GBM156N"),
    "JPY": ("DEXJPUS", True, "IR3TIB01JPM156N"),
    "CHF": ("DEXSZUS", True, "IR3TIB01CHM156N"),
    "CAD": ("DEXCAUS", True, "IR3TIB01CAM156N"),
    "AUD": ("DEXUSAL", False, "IR3TIB01AUM156N"),
    "NZD": ("DEXUSNZ", False, "IR3TIB01NZM156N"),
    "NOK": ("DEXNOUS", True, "IR3TIB01NOM156N"),
    "SEK": ("DEXSDUS", True, "IR3TIB01SEM156N"),
    "MXN": ("DEXMXUS", True, "IR3TIB01MXM156N"),
    "ZAR": ("DEXSFUS", True, "IR3TIB01ZAM156N"),
    "KRW": ("DEXKOUS", True, "IR3TIB01KRM156N"),
    "BRL": ("DEXBZUS", True, "IRSTCI01BRM156N"),
    "CLP": (None, None, "IR3TIB01CLM156N"),  # spot desde Yahoo CLP=X
}
TASA_US = "IR3TIB01USM156N"


# ---------------------------------------------------------------------------
# Descargas (con cache en disco)
# ---------------------------------------------------------------------------
def _curl(url):
    # FRED rechaza a urllib/requests sin navegador; curl funciona.
    # (con un User-Agent de navegador, FRED tambien corta la conexion)
    out = subprocess.run(["curl", "-s", "-m", "120", url], capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0 or not out.stdout:
        raise RuntimeError(f"descarga fallida: {url}")
    return out.stdout


def fred(serie, refrescar=False):
    ruta = f"{FUENTES_DIR}/fred_{serie}.csv"
    if refrescar or not os.path.exists(ruta):
        txt = _curl(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie}")
        df = pd.read_csv(io.StringIO(txt))
        df.to_csv(ruta, index=False)
    df = pd.read_csv(ruta)
    df.columns = ["ds", "valor"]
    df["ds"] = pd.to_datetime(df["ds"])
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df.dropna().reset_index(drop=True)


def yahoo_clp(refrescar=False):
    ruta = f"{FUENTES_DIR}/yahoo_CLP=X.csv"
    if refrescar or not os.path.exists(ruta):
        import yfinance as yf
        h = yf.Ticker("CLP=X").history(period="max", interval="1d", auto_adjust=False)
        h = h.reset_index()[["Date", "Close"]]
        h["Date"] = pd.to_datetime(h["Date"].dt.date)  # fecha de la barra (convencion Yahoo)
        h.columns = ["ds", "close"]
        h.to_csv(ruta, index=False)
    return pd.read_csv(ruta, parse_dates=["ds"])


def yahoo_sp500(refrescar=False):
    ruta = f"{FUENTES_DIR}/yahoo_GSPC.csv"
    if refrescar or not os.path.exists(ruta):
        import yfinance as yf
        h = yf.Ticker("^GSPC").history(period="max", interval="1d", auto_adjust=False).reset_index()
        h = h[["Date", "Close"]]
        h["Date"] = pd.to_datetime(h["Date"].dt.date)
        h.columns = ["ds", "close"]
        h.to_csv(ruta, index=False)
    return pd.read_csv(ruta, parse_dates=["ds"])


def dolar_observado(refrescar=False):
    ruta = f"{FUENTES_DIR}/mindicador_dolar_observado.csv"
    if refrescar or not os.path.exists(ruta):
        filas = []
        for anio in range(2003, pd.Timestamp.today().year + 1):
            js = json.loads(_curl(f"https://mindicador.cl/api/dolar/{anio}"))
            filas += [(s["fecha"][:10], s["valor"]) for s in js["serie"]]
        df = pd.DataFrame(filas, columns=["ds", "observado"]).drop_duplicates("ds").sort_values("ds")
        df.to_csv(ruta, index=False)
    return pd.read_csv(ruta, parse_dates=["ds"])


# ---------------------------------------------------------------------------
# Precios diarios limpios
# ---------------------------------------------------------------------------
def precios_diarios(refrescar=False):
    partes = []
    for cod, (serie, invertir, _) in MONEDAS.items():
        if serie is None:
            continue
        df = fred(serie, refrescar)
        df = df[df["ds"] >= INICIO]
        precio = 1.0 / df["valor"] if invertir else df["valor"]
        partes.append(pd.DataFrame({"moneda": cod, "ds": df["ds"], "ts_utc": alin.ts_fred_h10(df["ds"]),
                                    "precio": precio.to_numpy(), "fuente": f"FRED {serie} (H.10, 12:00 ET)"}))
    # CLP desde Yahoo, con limpieza
    y = yahoo_clp(refrescar).dropna().sort_values("ds").reset_index(drop=True)
    n0 = len(y)
    y = y[~alin.marcar_precios_repetidos(y["close"])].reset_index(drop=True)
    n_rep = n0 - len(y)
    obs = dolar_observado(refrescar)
    m = pd.merge_asof(y, obs, on="ds", direction="nearest", tolerance=pd.Timedelta(days=3))
    desvio = np.abs(np.log(m["close"] / m["observado"]))
    malo = desvio > MAX_DESVIO_OBSERVADO  # NaN (sin observado cercano) -> se conserva
    y = m.loc[~malo, ["ds", "close"]].reset_index(drop=True)
    diag_clp = {"clp_barras_yahoo": n0, "clp_repetidas_eliminadas": n_rep,
                "clp_descartadas_desvio_observado": int(malo.sum()),
                "clp_fechas_descartadas": ";".join(m.loc[malo, "ds"].dt.strftime("%Y-%m-%d"))}
    partes.append(pd.DataFrame({"moneda": "CLP", "ds": y["ds"], "ts_utc": alin.ts_fx_yahoo(y["ds"]),
                                "precio": 1.0 / y["close"].to_numpy(), "fuente": "Yahoo CLP=X (00:00 UTC)"}))
    diario = pd.concat(partes, ignore_index=True).dropna(subset=["ts_utc", "precio"])
    return diario.sort_values(["moneda", "ts_utc"]).reset_index(drop=True), obs, diag_clp


# ---------------------------------------------------------------------------
# Tasas mensuales con rezago de publicacion
# ---------------------------------------------------------------------------
def tasa_para_mes(serie_mensual, meses_objetivo, rezago):
    """Para cada mes objetivo M (Timestamp primer dia del mes), el valor del
    mes M-rezago; si no existe, el ultimo anterior con antiguedad <=
    MAX_ARRASTRE_TASA_MESES. Devuelve (valor, mes_del_valor)."""
    s = serie_mensual.set_index("ds")["valor"].sort_index()
    objetivo = pd.DatetimeIndex(meses_objetivo) - pd.DateOffset(months=rezago)
    izq = pd.DataFrame({"ds": objetivo, "_o": np.arange(len(objetivo))}).sort_values("ds")
    der = pd.DataFrame({"ds": s.index, "valor": s.to_numpy(), "mes_valor": s.index})
    mm = pd.merge_asof(izq, der, on="ds", direction="backward").sort_values("_o")
    edad = (mm["ds"].dt.year - mm["mes_valor"].dt.year) * 12 + (mm["ds"].dt.month - mm["mes_valor"].dt.month)
    mm.loc[edad > MAX_ARRASTRE_TASA_MESES, ["valor", "mes_valor"]] = np.nan
    return mm["valor"].to_numpy(), mm["mes_valor"].to_numpy()


# ---------------------------------------------------------------------------
# Panel mensual
# ---------------------------------------------------------------------------
def panel_mensual(diario, refrescar=False):
    h10 = diario[diario["fuente"].str.startswith("FRED")]
    fechas = pd.Series(sorted(h10["ds"].unique()))
    rebal = fechas.groupby(fechas.dt.to_period("M")).max().reset_index(drop=True)
    # el ultimo mes solo si esta completo (su ultima fecha H.10 es fin de mes habil)
    ultimo = rebal.iloc[-1]
    if ultimo + pd.offsets.BMonthEnd(0) != ultimo:
        rebal = rebal.iloc[:-1]
    T = alin.ts_fred_h10(rebal)
    meses = rebal.dt.to_period("M").dt.to_timestamp()

    us = fred(TASA_US, refrescar)
    us_senal, us_mes_senal = tasa_para_mes(us, meses, REZAGO_SENAL_MESES)
    us_dev, us_mes_dev = tasa_para_mes(us, meses, 0)

    filas = []
    for cod, (_, _, serie_tasa) in MONEDAS.items():
        d = diario[diario["moneda"] == cod].sort_values("ts_utc")
        ts_src = pd.DatetimeIndex(d["ts_utc"])
        # senal: ultimo precio con ts ESTRICTAMENTE anterior al mediodia NY de d_t
        p_sen, ts_sen = alin.valor_conocido(T, ts_src, d["precio"])
        if cod == "CLP":
            # ejecucion: primera barra Yahoo estrictamente POSTERIOR al mediodia NY de d_t
            idx = np.searchsorted(ts_src.asi8, T.asi8, side="right")
            ok = idx < len(d)
            ie = np.where(ok, idx, 0)
            p_ej = np.where(ok, d["precio"].to_numpy()[ie], np.nan)
            ts_ej = np.where(ok, ts_src.to_numpy()[ie], np.datetime64("NaT"))
        else:
            # ejecucion: el propio precio H.10 de d_t (ultimo con ts <= T)
            p_ej, ts_ej = alin.valor_conocido(T + pd.Timedelta(minutes=1), ts_src, d["precio"])
        tasa = fred(serie_tasa, refrescar)
        t_sen, mes_sen = tasa_para_mes(tasa, meses, REZAGO_SENAL_MESES)
        t_dev, mes_dev = tasa_para_mes(tasa, meses, 0)
        filas.append(pd.DataFrame({
            "ds": rebal.to_numpy(), "moneda": cod, "ts_rebalanceo": T,
            "precio_ejec": p_ej, "ts_ejec": pd.to_datetime(ts_ej, utc=True),
            "precio_senal": p_sen, "ts_senal": pd.to_datetime(ts_sen, utc=True),
            "tasa_senal": t_sen, "mes_tasa_senal": mes_sen, "tasa_devengo": t_dev, "mes_tasa_devengo": mes_dev,
            "tasa_us_senal": us_senal, "tasa_us_devengo": us_dev,
            "fuente_spot": d["fuente"].iloc[0], "serie_tasa": serie_tasa}))
    panel = pd.concat(filas, ignore_index=True)

    # precios viejos: ejecucion o senal a mas de 5 dias del rebalanceo -> NaN
    hrs_ej = (panel["ts_ejec"] - panel["ts_rebalanceo"]).dt.total_seconds() / 3600
    hrs_sen = (panel["ts_rebalanceo"] - panel["ts_senal"]).dt.total_seconds() / 3600
    panel.loc[(hrs_ej.abs() > 24 * 5), "precio_ejec"] = np.nan
    panel.loc[(hrs_sen > 24 * 7), "precio_senal"] = np.nan
    panel["horas_ejec_menos_rebalanceo"] = hrs_ej
    panel["horas_rebalanceo_menos_senal"] = hrs_sen
    return panel.sort_values(["ds", "moneda"]).reset_index(drop=True)


def riesgo_mercado(rebal, refrescar=False):
    sp = yahoo_sp500(refrescar)
    vix = fred("VIXCLS", refrescar).rename(columns={"valor": "vix"})
    r = pd.DataFrame({"ds": pd.to_datetime(rebal)}).sort_values("ds")
    r = pd.merge_asof(r, sp.rename(columns={"close": "sp500"}), on="ds", direction="backward")
    r = pd.merge_asof(r, vix, on="ds", direction="backward")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refrescar", action="store_true", help="vuelve a descargar todas las fuentes")
    args = ap.parse_args()
    os.makedirs(FUENTES_DIR, exist_ok=True)

    diario, obs, diag_clp = precios_diarios(args.refrescar)
    diario.to_csv(f"{BASES_DIR}/cartera_fx_diario.csv", index=False)
    panel = panel_mensual(diario, args.refrescar)
    panel.to_csv(f"{BASES_DIR}/cartera_fx_mensual.csv", index=False)
    rm = riesgo_mercado(panel["ds"].unique(), args.refrescar)
    rm.to_csv(f"{BASES_DIR}/cartera_fx_riesgo_mercado.csv", index=False)

    # ---------------- chequeos de cordura ----------------
    print(f"Rebalanceos: {panel['ds'].min().date()} .. {panel['ds'].max().date()} ({panel['ds'].nunique()} meses)")
    assert not (panel["ts_senal"] >= panel["ts_rebalanceo"]).any(), "senal no anterior al rebalanceo"
    assert not (panel["ts_ejec"] < panel["ts_rebalanceo"]).any(), "ejecucion anterior al rebalanceo"
    ok_ej = panel["precio_ejec"].notna() & panel["precio_senal"].notna()
    assert (panel.loc[ok_ej, "ts_senal"] < panel.loc[ok_ej, "ts_ejec"]).all()
    mes_reb = panel["ds"].dt.to_period("M").dt.to_timestamp()
    rez = (mes_reb.dt.year - pd.to_datetime(panel["mes_tasa_senal"]).dt.year) * 12 + \
          (mes_reb.dt.month - pd.to_datetime(panel["mes_tasa_senal"]).dt.month)
    assert (rez.dropna() >= REZAGO_SENAL_MESES).all(), "tasa de senal con rezago menor al supuesto"

    val = []
    for cod, g in panel.groupby("moneda"):
        g = g.sort_values("ds")
        ok = g.dropna(subset=["precio_ejec", "precio_senal", "tasa_senal", "tasa_devengo", "tasa_us_senal"])
        r = np.log(g["precio_ejec"]).diff()
        val.append({"moneda": cod, "fuente_spot": g["fuente_spot"].iloc[0], "serie_tasa": g["serie_tasa"].iloc[0],
                    "primer_mes_completo": ok["ds"].min().date() if len(ok) else None,
                    "ultimo_mes": ok["ds"].max().date() if len(ok) else None, "meses_completos": len(ok),
                    "ultimo_mes_tasa_publicado": pd.to_datetime(g["mes_tasa_devengo"]).max().date(),
                    "max_abs_ret_mensual_%": 100 * r.abs().max(),
                    "fecha_max_abs_ret": g.loc[r.abs().idxmax(), "ds"].date() if r.notna().any() else None,
                    "vol_anual_%": 100 * r.std() * np.sqrt(12),
                    "diferencial_medio_vs_us_pp": (g["tasa_devengo"] - g["tasa_us_devengo"]).mean(),
                    "horas_ejec_menos_rebalanceo_mediana": g["horas_ejec_menos_rebalanceo"].median(),
                    "horas_rebalanceo_menos_senal_mediana": g["horas_rebalanceo_menos_senal"].median()})
    val = pd.DataFrame(val)

    # CLP Yahoo vs dolar observado BCCh: correlacion de retornos mensuales
    clp = panel[panel["moneda"] == "CLP"][["ds", "precio_ejec"]].dropna()
    o = pd.merge_asof(clp.sort_values("ds"), obs.sort_values("ds"), on="ds", direction="forward",
                      tolerance=pd.Timedelta(days=5))  # observado de D+1 = transacciones de D
    o = o.dropna()
    corr = np.corrcoef(np.diff(np.log(o["precio_ejec"])), np.diff(np.log(1 / o["observado"])))[0, 1]
    dif = np.abs(np.log(o["precio_ejec"] * o["observado"]))
    diag_clp.update({"corr_ret_mensual_yahoo_vs_observado": corr, "n_meses_comparados": len(o),
                     "desvio_mediano_nivel_%": 100 * dif.median(), "desvio_max_nivel_%": 100 * dif.max()})
    for k, v in diag_clp.items():
        val[k] = None
        val.loc[val["moneda"] == "CLP", k] = v
    val.to_csv(f"{RESULTADOS_DIR}/issue15_validacion_datos.csv", index=False)
    pd.set_option("display.width", 250)
    print(val.drop(columns=[c for c in val.columns if c.startswith("clp_")]).to_string(index=False))
    print("Diagnostico CLP:", json.dumps({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                          for k, v in diag_clp.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
