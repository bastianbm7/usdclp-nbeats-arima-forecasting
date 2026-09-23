# CORRECCION (2026-09-23): modulo central de alineacion temporal. Lo usan TODOS
# los scripts corregidos (58+) que cruzan una serie FX diaria con un commodity
# diario. Existe porque dos auditorias independientes encontraron (y la
# evidencia de 58_validacion_timestamp.py lo confirma) que la "prediccion a un
# dia" del cobre sobre USD/CLP de 9.13-9.34 era en realidad CONTEMPORANEA:
#
# EL MECANISMO (dicho sin atenuar):
#   - Yahoo Finance etiqueta la barra diaria de un par FX (CLP=X, AUDUSD=X,
#     NOK=X, todo el panel de 23/55) con fecha D, pero el precio es el de
#     ~00:00 UTC del dia D, es decir ~20:00 de Nueva York del dia D-1 (19:00
#     en invierno). La fila "D" del FX es, en el reloj real, la tarde de D-1.
#   - El "cierre" diario de un futuro de commodity (HG=F cobre, CL=F WTI,
#     GC=F oro, PL=F platino, ZS=F soja, TIO=F hierro) etiquetado D es el
#     settlement de ese mismo dia D (~13:00-14:30 ET).
#   - Entonces el retorno FX etiquetado D+1 (precio de ~20:00 NY de D sobre
#     precio de ~20:00 NY de D-1) cubre 17 de las 24 horas de la ventana del
#     retorno de commodity etiquetado D (settlement D-1 -> settlement D). Lo
#     que el proyecto llamo "rezago +1 predictivo" (cobre en t -> FX en t+1)
#     es mayormente la reaccion SIMULTANEA del FX al mismo shock. Y el
#     backtest entraba al precio de la fila t (= ~20:00 NY de t-1), ANTES de
#     que existiera el settlement de cobre de t que usaba como senal.
#
# LA CORRECCION (una sola regla, aplicada en todos lados):
#   Cada observacion se ubica en su timestamp REAL (UTC). Un dato de commodity
#   solo puede usarse en una decision FX si su settlement ocurrio
#   ESTRICTAMENTE ANTES del timestamp del precio FX al que se entra. Para
#   barras Yahoo FX (00:00 UTC de D) y settlements de EE.UU. (~13-15h ET),
#   eso equivale a: el commodity usable en la fila FX D es el del ultimo
#   settlement con fecha < D (no <= D, como hacia merge_asof(direction=
#   "backward") en 21/22/23/25/26/45-47/52-57). Para precios FRED H.10
#   (mediodia de Nueva York) la regla resulta ser la misma (settlement de D
#   ocurre despues del mediodia de D).
#
#   Equivalentemente, en terminos de "reetiquetar": la fila Yahoo FX D se
#   reetiqueta a la fecha habil anterior (su reloj real, ~20:00 NY de D-1) y
#   el commodity conserva su fecha. Despues de reetiquetar, el retorno FX con
#   la misma fecha que el retorno del commodity es el contemporaneo (rezago
#   0) y la unica prueba OPERABLE es: senal = retorno del commodity hasta el
#   settlement de t, entrada en el primer precio FX posterior (~20:00 NY de
#   t = fila Yahoo original t+1), salida en el siguiente precio FX (fila
#   original t+2). Chequeo de cordura que confirma que la alineacion es la
#   correcta: tras reetiquetar, la correlacion fuerte debe aparecer en el
#   rezago 0 (contemporanea) y NO en el +1 - ver 58/59.
#
# Por que no se regeneran los datasets NHITS (26/35/39/45-47): el forecast
# NHITS, GARCH, MACD/RSI y el min/max se calculan SOLO con la propia serie FX
# hasta el precio de la fila D - todo eso ya era conocido a ~20:00 NY de D-1,
# no depende del commodity. Lo unico mal alineado eran las columnas de
# commodity (copper_ret_1d/copper_mom_5d), que se recalculan aca con la regla
# estricta. Ver 63_realinear_datasets_rl_diarios.py.

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

ZONA_NY = "America/New_York"

# Hora (ET) aproximada del settlement diario oficial de cada futuro. Solo
# importa que TODAS caen antes de las ~19:00-20:00 ET (hora real de la barra
# diaria Yahoo FX) y despues del mediodia ET (hora de FRED H.10) - la regla
# estricta de fechas vale para cualquier valor dentro de ese rango.
HORA_SETTLEMENT_ET = {
    "HG_F": (13, 0),    # cobre COMEX
    "GC_F": (13, 30),   # oro COMEX
    "PL_F": (13, 5),    # platino NYMEX
    "CL_F": (14, 30),   # WTI NYMEX
    "ZS_F": (14, 15),   # soja CBOT (13:15 CT)
    "TIO_F": (14, 0),   # mineral de hierro (CME) - aproximado
}
HORA_FRED_H10_ET = (12, 0)  # "noon buying rates in New York" (Federal Reserve H.10)


# ---------------------------------------------------------------------------
# Timestamps reales
# ---------------------------------------------------------------------------
def ts_fx_yahoo(ds):
    """Barra diaria Yahoo FX con fecha D -> precio de ~00:00 UTC de D."""
    return pd.DatetimeIndex(pd.to_datetime(ds)).tz_localize("UTC")


def ts_settlement(ds, commodity="HG_F"):
    hh, mm = HORA_SETTLEMENT_ET[commodity]
    local = pd.DatetimeIndex(pd.to_datetime(ds)) + pd.Timedelta(hours=hh, minutes=mm)
    return local.tz_localize(ZONA_NY, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")


def ts_fred_h10(ds):
    hh, mm = HORA_FRED_H10_ET
    local = pd.DatetimeIndex(pd.to_datetime(ds)) + pd.Timedelta(hours=hh, minutes=mm)
    return local.tz_localize(ZONA_NY, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")


def fecha_reloj_fx_yahoo(ds):
    """Reetiquetado: la fila Yahoo FX D representa ~20:00 NY del dia habil anterior."""
    return pd.DatetimeIndex(pd.to_datetime(ds)) - BDay(1)


# ---------------------------------------------------------------------------
# Limpieza de la serie FX
# ---------------------------------------------------------------------------
def marcar_precios_repetidos(y):
    """True donde el precio es EXACTAMENTE igual al de la fila anterior.

    Yahoo rellena feriados (25-dic, 1-ene, etc.) y huecos de liquidez de pares
    poco liquidos (CLP=X, PEN=X, COP=X) repitiendo el ultimo precio. Operar a
    un precio repetido no es realista (no hubo cotizacion nueva)."""
    y = pd.Series(np.asarray(y, dtype=float))
    return (y.diff() == 0).to_numpy()


def serie_commodity_dias_habiles(macro, col="copper"):
    """El cache macro_tasas_cobre.csv viene con forward-fill a dias CALENDARIO.
    Para el marco "por settlement" se descartan sabados/domingos (no hay
    settlement); los feriados de EE.UU. quedan como retorno 0 (pocos)."""
    s = macro[["ds", col]].dropna().copy()
    s["ds"] = pd.to_datetime(s["ds"])
    s = s[s["ds"].dt.dayofweek < 5].rename(columns={col: "precio"})
    return s.sort_values("ds").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Merge "lo que se sabia a esa hora" (regla estricta)
# ---------------------------------------------------------------------------
def valor_conocido(ts_objetivo, ts_fuente, valores):
    """Para cada ts_objetivo, el ultimo valor cuya ts_fuente es ESTRICTAMENTE
    anterior. Devuelve (valores, ts_del_valor_usado)."""
    izq = pd.DataFrame({"ts": ts_objetivo, "_orden": np.arange(len(ts_objetivo))}).sort_values("ts")
    der = pd.DataFrame({"ts": ts_fuente, "valor": np.asarray(valores, dtype=float), "ts_fuente": ts_fuente}).sort_values("ts")
    der = der.dropna(subset=["ts"])
    m = pd.merge_asof(izq, der, on="ts", direction="backward", allow_exact_matches=False)
    m = m.sort_values("_orden")
    return m["valor"].to_numpy(), m["ts_fuente"].to_numpy()


def agregar_commodity_conocido(df_fx, serie_cm, commodity="HG_F", col_salida="copper", ts_fx=ts_fx_yahoo):
    """df_fx: DataFrame con 'ds' (fecha de la barra FX, convencion de la fuente).
    serie_cm: DataFrame con 'ds' (fecha de settlement) y 'precio'.
    Agrega col_salida = precio del commodity conocido ANTES del precio FX de la fila."""
    df = df_fx.copy()
    vals, ts_usado = valor_conocido(ts_fx(df["ds"]), ts_settlement(serie_cm["ds"], commodity), serie_cm["precio"])
    df[col_salida] = vals
    df[f"_ts_{col_salida}"] = ts_usado
    return df


def agregar_features_cobre(df_fx, macro, ts_fx=ts_fx_yahoo):
    """Reemplazo CORREGIDO del bloque que se repetia en 21/22/23/26/45-47/55:
        merge_asof(df, macro[["ds","copper"]], on="ds", direction="backward")
        copper_ret_1d = log(copper/copper.shift(1)); copper_mom_5d = ...
    Mismas formulas (sobre el calendario de la serie FX, igual que antes), pero
    el cobre de cada fila es el del ultimo settlement ANTERIOR al precio FX."""
    df = df_fx.sort_values("ds").reset_index(drop=True)
    df = agregar_commodity_conocido(df, serie_commodity_dias_habiles(macro), "HG_F", "copper", ts_fx)
    df["copper_ret_1d"] = np.log(df["copper"] / df["copper"].shift(1))
    df["copper_mom_5d"] = (df["copper"] - df["copper"].shift(5)) / df["copper"].shift(5)
    return df.drop(columns=["_ts_copper"])


def agregar_features_cobre_original(df_fx, macro):
    """La version ORIGINAL (con el artefacto), solo para reproducir las tablas
    de errata lado a lado. No usar para nada nuevo."""
    df = df_fx.sort_values("ds").reset_index(drop=True)
    df = pd.merge_asof(df, macro[["ds", "copper"]], on="ds", direction="backward")
    df["copper_ret_1d"] = np.log(df["copper"] / df["copper"].shift(1))
    df["copper_mom_5d"] = (df["copper"] - df["copper"].shift(5)) / df["copper"].shift(5)
    return df


# ---------------------------------------------------------------------------
# Marco "por settlement": senal -> operacion ejecutable
# ---------------------------------------------------------------------------
def tabla_senal_operacion(fx, cm, commodity="HG_F", ts_fx=ts_fx_yahoo, excluir_repetidos=True,
                          latencia=pd.Timedelta(0)):
    """Una fila por settlement t del commodity.

    fx: DataFrame ['ds','y'] (precio FX; convencion de la fuente para 'ds').
    cm: DataFrame ['ds','precio'] (settlements).

    Columnas:
      cm_ret         log(P_t / P_{t-1}) del commodity (settlement a settlement)
      fx_ret_contemp log-retorno FX entre las observaciones FX mas cercanas en
                     el tiempo a settlement(t-1) y settlement(t) - la ventana
                     FX que mas se solapa con la del commodity (chequeo de
                     cordura: aca debe estar la correlacion fuerte)
      fx_ret_operable log-retorno FX desde el PRIMER precio FX con timestamp
                     > settlement(t) + latencia (entrada) hasta el siguiente
                     precio FX (salida) - lo unico que se puede operar
      horas_hasta_entrada, ds_entrada, ds_salida
    """
    fx = fx[["ds", "y"]].dropna().sort_values("ds").reset_index(drop=True)
    if excluir_repetidos:
        fx = fx[~marcar_precios_repetidos(fx["y"])].reset_index(drop=True)
    cm = cm[["ds", "precio"]].dropna().sort_values("ds").reset_index(drop=True)

    t_fx = ts_fx(fx["ds"]).asi8
    t_cm = ts_settlement(cm["ds"], commodity).asi8
    logy = np.log(fx["y"].to_numpy(dtype=float))

    # indice de la primera obs FX estrictamente posterior al settlement (+latencia)
    idx_desp = np.searchsorted(t_fx, t_cm + latencia.value, side="right")
    # obs FX mas cercana al settlement (antes o despues)
    idx_antes = np.clip(np.searchsorted(t_fx, t_cm, side="right") - 1, 0, len(t_fx) - 1)
    idx_desp_0 = np.clip(np.searchsorted(t_fx, t_cm, side="right"), 0, len(t_fx) - 1)
    dist_antes = np.abs(t_cm - t_fx[idx_antes])
    dist_desp = np.abs(t_fx[idx_desp_0] - t_cm)
    idx_cercano = np.where(dist_desp < dist_antes, idx_desp_0, idx_antes)

    out = pd.DataFrame({"ds_senal": cm["ds"].to_numpy()})
    out["cm_ret"] = np.log(cm["precio"] / cm["precio"].shift(1)).to_numpy()
    valido_c = np.r_[False, (idx_cercano[1:] > idx_cercano[:-1])]
    ic_prev = np.r_[0, idx_cercano[:-1]]
    out["fx_ret_contemp"] = np.where(valido_c, logy[idx_cercano] - logy[ic_prev], np.nan)

    ok = idx_desp + 1 < len(t_fx)
    ie = np.where(ok, idx_desp, 0)
    ix = np.where(ok, idx_desp + 1, 0)
    out["fx_ret_operable"] = np.where(ok, logy[ix] - logy[ie], np.nan)
    out["ds_entrada"] = np.where(ok, fx["ds"].to_numpy()[ie], np.datetime64("NaT"))
    out["ds_salida"] = np.where(ok, fx["ds"].to_numpy()[ix], np.datetime64("NaT"))
    out["horas_hasta_entrada"] = np.where(ok, (t_fx[ie] - t_cm) / 3.6e12, np.nan)
    # si dos settlements consecutivos caen antes de la misma entrada (feriado
    # FX), solo el ultimo es operable a ese precio: se descarta el anterior
    dup = pd.Series(out["ds_entrada"]).duplicated(keep="last").to_numpy()
    out.loc[dup, "fx_ret_operable"] = np.nan
    return out


def escaneo_rezagos_tabla(tabla, lags=range(-2, 4)):
    """corr(cm_ret(t), fx_ret_contemp(t+k)) en el marco por settlement."""
    res = {}
    for k in lags:
        res[k] = tabla["cm_ret"].corr(tabla["fx_ret_contemp"].shift(-k))
    return res


def escaneo_rezagos_filas_fx(df, col_senal, col_ret="retorno_1d", lags=range(-2, 4)):
    """Escaneo "a la manera del proyecto": corr(senal en la fila t, retorno FX
    en la fila t+k), usando las filas de la serie FX. Con la senal ORIGINAL
    (merge <=) reproduce la tabla de 9.14; con la CORREGIDA (merge <), el
    rezago 0 pasa a ser el contemporaneo y el +1 el operable."""
    return {k: df[col_senal].corr(df[col_ret].shift(-k)) for k in lags}
