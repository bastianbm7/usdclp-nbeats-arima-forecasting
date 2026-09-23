# Issue #15: motor de backtest de la cartera mensual de primas de riesgo FX
# (carry, momentum de serie de tiempo y su combinacion), compartido por
# 81 (seleccion solo con datos < 2025-01-01) y 82 (apertura unica del hold-out).
#
# Convenciones (ver 80_datos_cartera_fx_mensual.py para las fuentes):
#   - Precios en USD por unidad de moneda extranjera; w_i > 0 = largo en la
#     moneda i contra USD (un forward a 1 mes con notional = w_i x capital).
#   - Retorno en exceso del activo i de t a t+1 (forward cubierto por CIP):
#         X_i = (S_{t+1}/S_t) * (1 + i_i*tau) / (1 + i_us*tau) - 1
#     con S = precio de ejecucion, i = tasa de "devengo" del mes t (proxy de la
#     prima forward fijada en t) y tau = dias/365. Es retorno EN EXCESO de la
#     caja en USD: "no hacer nada" = 0.
#   - Decision en t: SOLO con precios de senal (timestamp < rebalanceo), tasas
#     con rezago de publicacion (tasa_senal) y retornos diarios con fecha < d_t.
#   - Pesos del mes: w_t. Deriva hasta t+1: w_i*(S_{t+1}/S_t)/(1+r_p) (el
#     notional en moneda extranjera queda fijo). Turnover = sum|w_t - w_deriva|.
#   - Costo principal: spread IDA Y VUELTA completo sobre el turnover (el doble
#     de la convencion de medio spread por operacion; cubre a grandes rasgos
#     el roll del forward). Costo estricto: ademas, una ida y vuelta completa
#     sobre toda la posicion cada mes (cerrar y reabrir todo; es el "1.8%/ano
#     del notional en CLP" del Issue).
import importlib

import numpy as np
import pandas as pd

alin = importlib.import_module("alineacion_temporal")
ce = importlib.import_module("costos_y_estadistica")
proto = importlib.import_module("protocolo_evaluacion")

BASES_DIR = "../datos/bases"
PANEL = f"{BASES_DIR}/cartera_fx_mensual.csv"
DIARIO = f"{BASES_DIR}/cartera_fx_diario.csv"
RIESGO = f"{BASES_DIR}/cartera_fx_riesgo_mercado.csv"

# Primer retorno evaluado (posiciones armadas a fin de 1999-12): fijado a
# priori - el euro existe desde 1999-01 y TSMOM de 12 meses necesita un ano.
INICIO_EVAL = pd.Timestamp("2000-01-01")
VENTANA_VOL_DIAS = 126      # vol estimada con ~6 meses de retornos diarios pasados
MIN_OBS_VOL = 60
G_MAX = 4.0                 # tope de exposicion bruta sum|w| (notional/capital)
MIN_MONEDAS = 8             # minimo de monedas elegibles para armar la cartera

# Spreads ida+vuelta: los de costos_y_estadistica; SEK y KRW no estan definidos
# ahi y se suponen conservadores (SEK como NOK sin su prima de liquidez
# nocturna, KRW como mercado NDF emergente).
SPREADS = dict(ce.SPREAD_IDA_VUELTA)
SPREADS.update({"SEK": 0.0005, "KRW": 0.0010})


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar():
    p = pd.read_csv(PANEL, parse_dates=["ds"])
    piv = {c: p.pivot(index="ds", columns="moneda", values=c).sort_index()
           for c in ["precio_ejec", "precio_senal", "tasa_senal", "tasa_devengo"]}
    us = p.groupby("ds")[["tasa_us_senal", "tasa_us_devengo"]].first().sort_index()
    monedas = list(piv["precio_ejec"].columns)
    ds = piv["precio_ejec"].index

    PE, PS = piv["precio_ejec"], piv["precio_senal"]
    tau = pd.Series(np.r_[np.diff(ds.values).astype("timedelta64[D]").astype(float), np.nan] / 365.0, index=ds)
    fact = (1 + piv["tasa_devengo"].div(100).mul(tau, axis=0)).div(1 + us["tasa_us_devengo"].div(100) * tau, axis=0)
    spot_sig = PE.shift(-1) / PE          # S_{t+1}/S_t, alineado a la fecha de DECISION t
    X = spot_sig * fact - 1               # retorno en exceso realizado en (t, t+1], fila t
    X_spot = spot_sig - 1

    # retornos diarios para la vol: cada precio en su fecha de reloj real
    d = pd.read_csv(DIARIO, parse_dates=["ds"])
    es_y = d["fuente"].str.startswith("Yahoo")
    d.loc[es_y, "ds"] = alin.fecha_reloj_fx_yahoo(d.loc[es_y, "ds"])  # barra Yahoo D = ~20:00 NY de D-1
    dpx = d.pivot_table(index="ds", columns="moneda", values="precio", aggfunc="last").sort_index()
    rd = np.log(dpx).diff()  # retorno entre observaciones consecutivas de cada moneda (NaN en huecos)
    rd = rd.reindex(columns=monedas)
    return {"ds": ds, "monedas": monedas, "PE": PE, "PS": PS, "IS": piv["tasa_senal"], "ID": piv["tasa_devengo"],
            "US_S": us["tasa_us_senal"], "X": X, "X_spot": X_spot, "fact": fact, "rd": rd}


def sin_monedas(D, excluir):
    """Copia de D con esas monedas fuera del universo (para sensibilidades)."""
    D2 = dict(D)
    for k in ["PE", "PS"]:
        D2[k] = D[k].copy()
        D2[k][excluir] = np.nan
    return D2


def elegibles(D, t):
    """Monedas operables en la decision t, con informacion disponible en t
    (precio de senal y de ejecucion, tasa de senal y de devengo)."""
    fila = pd.concat([D["PE"].iloc[t], D["PS"].iloc[t], D["IS"].iloc[t], D["ID"].iloc[t]], axis=1)
    return fila.dropna().index.tolist()


def vol_cartera(D, t, w):
    """Vol anual de la cartera w con retornos diarios de fecha < d_t."""
    dt = D["ds"][t]
    rd = D["rd"]
    ventana = rd[rd.index < dt].iloc[-VENTANA_VOL_DIAS:]
    if len(ventana) < MIN_OBS_VOL:
        return np.nan
    rp = ventana.reindex(columns=w.index).fillna(0.0).to_numpy() @ w.to_numpy()
    return rp.std(ddof=1) * np.sqrt(252)


def vol_activos(D, t):
    dt = D["ds"][t]
    ventana = D["rd"][D["rd"].index < dt].iloc[-VENTANA_VOL_DIAS:]
    v = ventana.std(ddof=1) * np.sqrt(252)
    v[ventana.count() < MIN_OBS_VOL] = np.nan
    return v


# ---------------------------------------------------------------------------
# Senales (pesos crudos, antes de vol targeting)
# ---------------------------------------------------------------------------
def pesos_carry(D, t, n_pata, rezago_tasas=2):
    ok = elegibles(D, t)
    if len(ok) < max(MIN_MONEDAS, 2 * n_pata + 1):
        return None
    tasa = D["IS"].iloc[t][ok] if rezago_tasas == 2 else D["ID"].iloc[t][ok]  # rezago 0 = sensibilidad
    orden = tasa.sort_values(kind="mergesort")
    w = pd.Series(0.0, index=ok)
    w[orden.index[-n_pata:]] = 1.0 / n_pata
    w[orden.index[:n_pata]] = -1.0 / n_pata
    return w


def senal_momentum(D, t, k, ok):
    """Retorno pasado en exceso de k meses conocido en t: spot desde el precio
    de ejecucion de t-k hasta el precio de SENAL de t (anterior al rebalanceo),
    mas el carry devengado en esos meses (factores ya realizados)."""
    if t - k < 0:
        return pd.Series(np.nan, index=ok)
    spot = np.log(D["PS"].iloc[t][ok] / D["PE"].iloc[t - k][ok])
    carry = np.log(D["fact"].iloc[t - k:t][ok]).sum(min_count=k)
    return spot + carry


def pesos_tsmom(D, t, lookback):
    ok = elegibles(D, t)
    if len(ok) < MIN_MONEDAS:
        return None
    ks = [1, 3, 6, 12] if lookback == "ens" else [lookback]
    senal = sum(np.sign(senal_momentum(D, t, k, ok)) for k in ks) / len(ks)
    sig = vol_activos(D, t).reindex(ok)
    w = (senal / sig).dropna()
    if (w != 0).sum() == 0:
        return None
    return w / w.abs().sum()


def pesos_ew(D, t):
    ok = elegibles(D, t)
    if len(ok) < MIN_MONEDAS:
        return None
    return pd.Series(1.0 / len(ok), index=ok)


def aplicar_vol_target(D, t, w, vol_obj):
    if w is None:
        return None, np.nan, False
    v = vol_cartera(D, t, w)
    if not np.isfinite(v) or v <= 0:
        return None, v, False
    w = w * (vol_obj / v)
    bruto = w.abs().sum()
    saturado = bruto > G_MAX
    if saturado:
        w = w * (G_MAX / bruto)
    return w, v, saturado


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def config_nombre(cfg):
    if cfg["tipo"] == "carry":
        s = f"carry_n{cfg['n_pata']}"
        if cfg.get("rezago_tasas", 2) != 2:
            s += f"_rezago{cfg['rezago_tasas']}"
    elif cfg["tipo"] == "tsmom":
        s = f"tsmom_k{cfg['lookback']}"
    elif cfg["tipo"] == "combo":
        s = f"combo[{config_nombre(cfg['carry'])}+{config_nombre(cfg['tsmom'])}]"
    else:
        s = cfg["tipo"]
    if cfg.get("vol_obj") is not None:
        s += f"_vt{int(round(100 * cfg['vol_obj']))}"
    return s


def pesos_config(D, t, cfg):
    """Pesos finales (con vol targeting) de la configuracion en la decision t."""
    tipo = cfg["tipo"]
    if tipo == "ew":
        return pesos_ew(D, t), np.nan, False
    if tipo == "carry":
        return aplicar_vol_target(D, t, pesos_carry(D, t, cfg["n_pata"], cfg.get("rezago_tasas", 2)), cfg["vol_obj"])
    if tipo == "tsmom":
        return aplicar_vol_target(D, t, pesos_tsmom(D, t, cfg["lookback"]), cfg["vol_obj"])
    if tipo == "combo":
        wc, _, _ = pesos_config(D, t, cfg["carry"])
        wm, _, _ = pesos_config(D, t, cfg["tsmom"])
        if wc is None or wm is None:
            return None, np.nan, False
        w = (0.5 * wc).add(0.5 * wm, fill_value=0.0)  # 50/50 de las dos carteras ya con vol objetivo
        return aplicar_vol_target(D, t, w, cfg["vol_obj"])
    raise ValueError(tipo)


def backtest(D, cfg):
    """Una fila por mes de RETORNO (fecha = fin del periodo, t+1)."""
    ds, mon = D["ds"], D["monedas"]
    s = pd.Series(SPREADS).reindex(mon).to_numpy()
    w_prev_deriva = pd.Series(0.0, index=mon)
    filas, pesos = [], []
    for t in range(len(ds) - 1):
        w, v_ex_ante, sat = pesos_config(D, t, cfg)
        w = (w if w is not None else pd.Series(dtype=float)).reindex(mon).fillna(0.0)
        x = D["X"].iloc[t].reindex(mon)
        xs = D["X_spot"].iloc[t].reindex(mon)
        faltante = (w != 0) & x.isna()   # posicion abierta sin precio de salida (raro)
        x, xs = x.fillna(0.0), xs.fillna(0.0)
        dw = (w - w_prev_deriva).abs()
        costo_rot = float((s * dw.to_numpy()).sum())
        costo_estr = costo_rot + float((s * w.abs().to_numpy()).sum())
        r_bruto = float((w * x).sum())
        filas.append({"ds": ds[t + 1], "ds_decision": ds[t], "r_bruto": r_bruto,
                      "r_neto": r_bruto - costo_rot, "r_neto_estricto": r_bruto - costo_estr,
                      "r_spot": float((w * xs).sum()), "r_carry": r_bruto - float((w * xs).sum()),
                      "turnover": float(dw.sum()), "bruto_exposicion": float(w.abs().sum()),
                      "neto_exposicion": float(w.sum()), "vol_ex_ante": v_ex_ante, "saturado": bool(sat),
                      "n_monedas": int((w != 0).sum()), "posiciones_sin_precio": int(faltante.sum())})
        pesos.append(w.rename(ds[t]))
        w_prev_deriva = w * (1 + xs) / (1 + r_bruto) if (1 + r_bruto) > 0 else w * 0
    return pd.DataFrame(filas), pd.DataFrame(pesos)


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def max_drawdown(r):
    cap = np.cumprod(1 + np.asarray(r, dtype=float))
    return float((cap / np.maximum.accumulate(cap) - 1).min())


def metricas(bt, nombre, riesgo=None, seed=0):
    out = {"estrategia": nombre, "desde": bt["ds"].min().date(), "hasta": bt["ds"].max().date(), "meses": len(bt)}
    for col, et in [("r_bruto", "bruto"), ("r_neto", "neto"), ("r_neto_estricto", "neto_estricto")]:
        r = bt[col].to_numpy()
        lo, hi = ce.ic_sharpe_bootstrap(r, periodos=12, n_boot=5000, bloque=6, seed=seed)
        out[f"sharpe_{et}"] = ce.sharpe(r, periodos=12)
        out[f"ic95_bajo_{et}"], out[f"ic95_alto_{et}"] = lo, hi
        out[f"ret_anual_%_{et}"] = 100 * np.nanmean(r) * 12
    out["vol_anual_%"] = 100 * bt["r_neto"].std() * np.sqrt(12)
    out["max_dd_%_neto"] = 100 * max_drawdown(bt["r_neto"])
    out["turnover_anual"] = bt["turnover"].mean() * 12
    out["exposicion_bruta_media"] = bt["bruto_exposicion"].mean()
    out["exposicion_neta_media"] = bt["neto_exposicion"].mean()
    out["%_meses_saturado"] = 100 * bt["saturado"].mean()
    out["carry_anual_%"] = 100 * bt["r_carry"].mean() * 12
    out["spot_anual_%"] = 100 * bt["r_spot"].mean() * 12
    peor = bt.loc[bt["r_neto"].idxmin()]
    out["peor_mes_%"] = 100 * peor["r_neto"]
    out["fecha_peor_mes"] = peor["ds"].date()
    for et, (a, b) in {"2008_ago_dic": ("2008-08-01", "2008-12-31"), "2020_feb_mar": ("2020-02-01", "2020-03-31")}.items():
        m = bt[(bt["ds"] >= a) & (bt["ds"] <= b)]
        out[f"ret_%_{et}"] = 100 * (np.prod(1 + m["r_neto"]) - 1) if len(m) else np.nan
    if riesgo is not None:
        m = bt.merge(riesgo, on="ds", how="left")
        out["corr_sp500"] = m["r_neto"].corr(m["ret_sp500"])
        out["corr_dvix"] = m["r_neto"].corr(m["dvix"])
    return out


def cargar_riesgo():
    r = pd.read_csv(RIESGO, parse_dates=["ds"]).sort_values("ds")
    r["ret_sp500"] = r["sp500"].pct_change()
    r["dvix"] = r["vix"].diff()
    return r[["ds", "ret_sp500", "dvix"]]


def recortar(bt, periodo):
    """periodo='pre': retornos que terminan antes del hold-out (y desde INICIO_EVAL);
    'holdout': retornos que terminan en o despues de HOLDOUT_INICIO."""
    bt = bt[bt["ds"] >= INICIO_EVAL]
    if periodo == "pre":
        return proto.antes_del_holdout(bt, "ds").reset_index(drop=True)
    return bt[bt["ds"] >= proto.HOLDOUT_INICIO].reset_index(drop=True)
