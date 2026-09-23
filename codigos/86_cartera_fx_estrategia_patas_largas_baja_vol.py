# Issue #15, estrategia pedida por Bastian a partir de lo observado en 85:
#   - Carry: el trailing stop solo sirvio en la pata LARGA (monedas de tasa
#     alta), y esa pata aporta la mayor parte del retorno.
#   - Momentum 6m: casi todo el retorno viene de la mitad MENOS volatil.
# Se arma una estrategia con ambas ideas. Todo se fija antes de correr:
#   - k del trailing = 1.5 (el de 85, no se vuelve a optimizar).
#   - "Solo patas largas" admite dos lecturas, se reportan ambas:
#       (a) carry completo (2 largas + 2 cortas) con trailing solo en las largas;
#       (b) carry SOLO con las 2 largas (sin vender las de tasa baja), con trailing.
#   - Momentum 6m solo en la mitad menos volatil (vol ex-ante bajo la mediana de
#     las monedas activas en cada rebalanceo), renormalizado.
#   - Cada componente con vol objetivo 5%, combinacion 50/50 y vol objetivo 5%.
#   - En la combinacion, el trailing se aplica a las monedas de la pata larga
#     del carry de ese mes (cierra la posicion total de esa moneda).
# ADVERTENCIA en el propio diseño: ambas ideas salieron de mirar 2000-2024
# (85); evaluarlas en ese mismo periodo sobreestima su resultado. Solo datos
# < 2025-01-01; el hold-out no se abre aqui.

import importlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cf = importlib.import_module("cartera_fx")
proto = importlib.import_module("protocolo_evaluacion")
ce = importlib.import_module("costos_y_estadistica")
m83 = importlib.import_module("83_cartera_fx_stop_take_profit")
m84 = importlib.import_module("84_cartera_fx_stops_por_posicion_trailing")

RESULTADOS_DIR = "../datos/resultados"
SCRIPT = "86_cartera_fx_estrategia_patas_largas_baja_vol.py"
K_TRAILING = 1.5
VOL = 0.05
_pesos_config_original = cf.pesos_config


def pesos_carry_largo(D, t, n_pata=2):
    w = cf.pesos_carry(D, t, n_pata)
    if w is None:
        return None
    w = w[w > 0]
    return w / w.abs().sum() if len(w) else None


def pesos_tsmom_baja_vol(D, t, lookback=6):
    w = cf.pesos_tsmom(D, t, lookback)
    if w is None:
        return None
    activos = w[w != 0].index
    v = cf.vol_activos(D, t).reindex(activos).dropna().sort_values()
    baja = v.index[: len(v) // 2]
    w = w.reindex(baja).fillna(0.0)
    return w / w.abs().sum() if w.abs().sum() > 0 else None


def pesos_config(D, t, cfg):
    """Extiende cartera_fx.pesos_config con los tipos nuevos (el resto igual)."""
    tipo = cfg["tipo"]
    if tipo == "carry_largo":
        return cf.aplicar_vol_target(D, t, pesos_carry_largo(D, t, cfg["n_pata"]), cfg["vol_obj"])
    if tipo == "tsmom_baja_vol":
        return cf.aplicar_vol_target(D, t, pesos_tsmom_baja_vol(D, t, cfg["lookback"]), cfg["vol_obj"])
    if tipo == "combo":
        wc, _, _ = pesos_config(D, t, cfg["carry"])
        wm, _, _ = pesos_config(D, t, cfg["tsmom"])
        if wc is None or wm is None:
            return None, np.nan, False
        return cf.aplicar_vol_target(D, t, (0.5 * wc).add(0.5 * wm, fill_value=0.0), cfg["vol_obj"])
    return _pesos_config_original(D, t, cfg)


cf.pesos_config = pesos_config  # 84 y cf.backtest llaman a cf.pesos_config


def filtro_pata_larga_carry(D, t, w):
    wc = cf.pesos_carry(D, t, 2)
    return [] if wc is None else [m for m in wc.index if wc[m] > 0]


CARRY = {"tipo": "carry", "n_pata": 2, "vol_obj": VOL}
CARRY_LARGO = {"tipo": "carry_largo", "n_pata": 2, "vol_obj": VOL}
TSMOM = {"tipo": "tsmom", "lookback": 6, "vol_obj": VOL}
TSMOM_BAJA = {"tipo": "tsmom_baja_vol", "lookback": 6, "vol_obj": VOL}

# (nombre, cfg, usa trailing en la pata larga del carry, es prueba nueva)
VARIANTES = [
    ("Original: carry + momentum (81)", {"tipo": "combo", "carry": CARRY, "tsmom": TSMOM, "vol_obj": VOL}, False, False),
    ("Original + trailing pata larga", {"tipo": "combo", "carry": CARRY, "tsmom": TSMOM, "vol_obj": VOL}, True, True),
    ("Carry completo (81)", CARRY, False, False),
    ("Carry completo + trailing pata larga (85)", CARRY, True, False),
    ("Carry solo pata larga", CARRY_LARGO, False, True),
    ("Carry solo pata larga + trailing", CARRY_LARGO, True, True),
    ("Momentum 6m (81)", TSMOM, False, False),
    ("Momentum 6m solo mitad menos volatil", TSMOM_BAJA, False, True),
    ("NUEVA (a): carry completo c/ trailing pata larga + momentum baja vol",
     {"tipo": "combo", "carry": CARRY, "tsmom": TSMOM_BAJA, "vol_obj": VOL}, True, True),
    ("NUEVA (b): carry solo pata larga c/ trailing + momentum baja vol",
     {"tipo": "combo", "carry": CARRY_LARGO, "tsmom": TSMOM_BAJA, "vol_obj": VOL}, True, True),
]


def main():
    D = cf.cargar()
    P = m83.grilla_diaria(D)
    TRI = m84.indice_retorno_total(D, P)
    riesgo = cf.cargar_riesgo()
    filas, curvas = [], {}
    for nombre, cfg, trailing, nueva in VARIANTES:
        if trailing:
            bt = m84.backtest(D, P, TRI, cfg, "trailing_moneda", K_TRAILING, filtro=filtro_pata_larga_carry)
        else:
            bt = cf.backtest(D, cfg)[0]
            bt["n_cierres"] = 0
        bt = cf.recortar(bt, "pre")
        m = cf.metricas(bt, nombre, riesgo)
        m["cierres_por_anio"] = 12 * bt["n_cierres"].mean()
        m["prueba_nueva"] = nueva
        sub = {}
        for a, b in [("2000", "2008"), ("2009", "2016"), ("2017", "2024")]:
            r = bt[(bt["ds"].dt.year >= int(a)) & (bt["ds"].dt.year <= int(b))]["r_neto"]
            sub[f"sharpe_{a}_{b}"] = ce.sharpe(r, 12)
        m.update(sub)
        filas.append(m)
        curvas[nombre] = bt.set_index("ds")["r_neto"]
    tabla = pd.DataFrame(filas)
    tabla.to_csv(f"{RESULTADOS_DIR}/issue15_estrategia_patas_largas_baja_vol_pre_holdout.csv", index=False)

    proto.registrar_pruebas(
        [{"prueba": r.estrategia, "frecuencia": "mensual (trailing diario)", "metrica": "sharpe_neto_pre_holdout",
          "valor": r.sharpe_neto, "n": r.meses, "usa_holdout": False, "nota": "estrategia patas largas + momentum baja vol"}
         for r in tabla.itertuples() if r.prueba_nueva], issue="#15", script=SCRIPT)
    n_total = proto.n_pruebas_registradas("#15")
    pd.set_option("display.width", 260)
    cols = ["estrategia", "sharpe_neto", "ic95_bajo_neto", "ic95_alto_neto", "ret_anual_%_neto", "vol_anual_%",
            "max_dd_%_neto", "peor_mes_%", "ret_%_2008_ago_dic", "ret_%_2020_feb_mar", "cierres_por_anio",
            "sharpe_2000_2008", "sharpe_2009_2016", "sharpe_2017_2024", "corr_sp500"]
    print(tabla[cols].round(2).to_string(index=False))
    print(f"\nPruebas registradas en el Issue #15: {n_total}. Sharpe maximo esperado por azar: "
          f"{ce.sharpe_maximo_esperado_nulo(n_total, 300, periodos=12):.2f}")

    fig, ax = plt.subplots(figsize=(13, 5.8), facecolor="#fcfcfb")
    estilo = [("Original: carry + momentum (81)", "#2a78d6", 2.0),
              ("NUEVA (a): carry completo c/ trailing pata larga + momentum baja vol", "#eb6834", 1.5),
              ("NUEVA (b): carry solo pata larga c/ trailing + momentum baja vol", "#1baf7a", 1.5)]
    for nombre, col, lw in estilo:
        r = curvas[nombre]
        ax.plot(r.index, 100 * np.cumprod(1 + r), color=col, lw=lw, label=nombre)
    ax.set_facecolor("#fcfcfb")
    ax.set_title("$100 invertidos en 2000, neto de costos (vol objetivo 5%)", loc="left", fontsize=12)
    ax.grid(axis="y", color="#e1e0d9"); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for a, b in [("2008-08-01", "2008-12-31"), ("2020-02-01", "2020-03-31")]:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#f0efe9", zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.01, 0.01, "Las variantes nuevas se armaron con ideas sacadas de este mismo periodo (85): su resultado aca esta sobreestimado. "
             "Sombreado: 2008-08..12 y 2020-02..03.", fontsize=8, color="#898781")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{RESULTADOS_DIR}/issue15_estrategia_patas_largas_baja_vol.png", dpi=120, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
