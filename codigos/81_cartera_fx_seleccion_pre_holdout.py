# Issue #15 - paso 2: carry, momentum de serie de tiempo (TSMOM) y su
# combinacion 50/50, SELECCION SOLO CON DATOS ANTERIORES AL HOLD-OUT
# (retornos que terminan antes de 2025-01-01, desde 2000-01).
#
# Correr desde codigos/ (despues de 80):  python 81_cartera_fx_seleccion_pre_holdout.py
#
# Grilla (TODAS las variantes quedan en datos/resultados/registro_pruebas.csv):
#   carry: n monedas por pata in {2,3,4} x vol objetivo in {5%,10%}         -> 6
#   TSMOM: lookback in {1,3,6,12 meses, "ens" = promedio de los 4 signos}
#          x vol objetivo in {5%,10%}                                     -> 10
#   combinacion 50/50 del mejor carry + mejor TSMOM x vol objetivo           -> 2
#   sensibilidad: mejor carry con tasas SIN rezago de publicacion           -> 1
#   benchmark: equiponderada larga en las monedas (sin vol targeting)        -> 1
# Criterio de seleccion (fijado antes de correr): Sharpe NETO (costo principal)
# en el periodo pre-hold-out. La candidata final que se lleva al hold-out es
# la combinacion (la estrategia que pide el Issue), con los parametros que
# gane cada componente.
# Fijado a priori (no se barre): ventana de vol 126 dias, tope de exposicion
# bruta 4x, rebalanceo mensual, pesos iguales por pata en carry y 1/vol por
# moneda en TSMOM.

import importlib
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
cfx = importlib.import_module("cartera_fx")
ce = importlib.import_module("costos_y_estadistica")
proto = importlib.import_module("protocolo_evaluacion")

RES = "../datos/resultados"
SCRIPT = "81_cartera_fx_seleccion_pre_holdout.py"
VOLS = [0.05, 0.10]
COLORES = {"carry": "#2a78d6", "tsmom": "#eb6834", "combo": "#1baf7a", "ew": "#eda100"}


def correr(D, riesgo, cfg, extra=None):
    bt, w = cfx.backtest(D, cfg)
    pre = cfx.recortar(bt, "pre")
    m = cfx.metricas(pre, cfx.config_nombre(cfg), riesgo)
    m.update(extra or {})
    return m, bt, w


def main():
    D = cfx.cargar()
    riesgo = cfx.cargar_riesgo()
    filas, bts, pesos, cfgs = [], {}, {}, {}

    def evaluar(cfg, familia, rol, D_=None, sufijo=""):
        nombre = cfx.config_nombre(cfg) + sufijo
        m, bt, w = correr(D if D_ is None else D_, riesgo, cfg, {"familia": familia, "rol": rol})
        m["estrategia"] = nombre
        filas.append(m)
        bts[nombre], pesos[nombre], cfgs[nombre] = bt, w, cfg
        print(f"  {nombre:55s} bruto {m['sharpe_bruto']:+.2f}  neto {m['sharpe_neto']:+.2f} "
              f"[{m['ic95_bajo_neto']:+.2f}, {m['ic95_alto_neto']:+.2f}]  DD {m['max_dd_%_neto']:.1f}%  "
              f"sat {m['%_meses_saturado']:.0f}%", flush=True)
        return m

    print("Carry:")
    for n in [2, 3, 4]:
        for v in VOLS:
            evaluar({"tipo": "carry", "n_pata": n, "vol_obj": v}, "carry", "candidata")
    print("TSMOM:")
    for k in [1, 3, 6, 12, "ens"]:
        for v in VOLS:
            evaluar({"tipo": "tsmom", "lookback": k, "vol_obj": v}, "tsmom", "candidata")

    tab = pd.DataFrame(filas)
    mejor_carry = tab[tab["familia"] == "carry"].sort_values("sharpe_neto").iloc[-1]["estrategia"]
    mejor_tsmom = tab[tab["familia"] == "tsmom"].sort_values("sharpe_neto").iloc[-1]["estrategia"]
    print(f"Mejor carry: {mejor_carry}   mejor TSMOM: {mejor_tsmom}")

    print("Combinacion 50/50:")
    for v in VOLS:
        cc = dict(cfgs[mejor_carry], vol_obj=v)
        cm = dict(cfgs[mejor_tsmom], vol_obj=v)
        evaluar({"tipo": "combo", "carry": cc, "tsmom": cm, "vol_obj": v}, "combo", "candidata")
    print("Sensibilidad y benchmark:")
    evaluar(dict(cfgs[mejor_carry], rezago_tasas=0), "carry", "sensibilidad (tasas sin rezago de publicacion)")
    evaluar({"tipo": "ew", "vol_obj": None}, "ew", "benchmark")

    tab = pd.DataFrame(filas)
    combos = tab[(tab["familia"] == "combo") & (tab["rol"] == "candidata")].sort_values("sharpe_neto")
    elegida = combos.iloc[-1]["estrategia"]
    # sensibilidad (no seleccionable): la elegida sin BRL, la moneda que mas aporta y
    # cuyo carry esta mas sobreestimado por usar la tasa local (call money) en vez
    # de la implicita en el NDF offshore (desvio de CIP / "cupom cambial")
    evaluar(cfgs[elegida], "combo", "sensibilidad (sin BRL)", cfx.sin_monedas(D, ["BRL"]), "_sinBRL")
    tab = pd.DataFrame(filas)
    tab["elegida_para_holdout"] = tab["estrategia"] == elegida

    # ---------------- pruebas multiples ----------------
    n_pruebas = len(tab)
    T = int(tab["meses"].max())
    sr_nulo = ce.sharpe_maximo_esperado_nulo(n_pruebas, T, periodos=12)
    tab["sharpe_max_esperado_H0"] = sr_nulo
    tab["n_pruebas"] = n_pruebas
    tab.to_csv(f"{RES}/issue15_pre_holdout_configuraciones.csv", index=False)
    print(f"\n{n_pruebas} pruebas, T={T} meses -> Sharpe maximo esperado bajo H0 = {sr_nulo:.2f}")

    altos = tab[tab["sharpe_bruto"] > 1.5]
    if len(altos):
        print("ATENCION: Sharpe > 1.5, revisar look-ahead/datos antes de reportar:", altos["estrategia"].tolist())

    proto.registrar_pruebas([{
        "prueba": r["estrategia"], "frecuencia": "mensual", "metrica": "sharpe_neto_pre_holdout",
        "valor": round(r["sharpe_neto"], 4), "n": r["meses"], "usa_holdout": False,
        "nota": (f"{r['rol']}; bruto {r['sharpe_bruto']:.2f}; IC95 neto [{r['ic95_bajo_neto']:.2f},"
                 f"{r['ic95_alto_neto']:.2f}]; {r['desde']}..{r['hasta']}"
                 + ("; ELEGIDA para hold-out" if r["elegida_para_holdout"] else ""))}
        for _, r in tab.iterrows()], issue="#15", script=SCRIPT)

    # configuracion elegida (la lee 82)
    with open(f"{RES}/issue15_config_elegida.json", "w") as f:
        json.dump({"estrategia": elegida, "config": cfgs[elegida], "n_pruebas_pre_holdout": n_pruebas,
                   "sharpe_max_esperado_H0": sr_nulo, "criterio": "max Sharpe neto pre-hold-out",
                   "fecha_limite_datos_seleccion": str(proto.HOLDOUT_INICIO.date())}, f, indent=1)

    # ---------------- tabla resumen ----------------
    principales = [mejor_carry, mejor_tsmom, elegida, "ew"]
    cols = ["estrategia", "meses", "sharpe_bruto", "ic95_bajo_bruto", "ic95_alto_bruto", "sharpe_neto",
            "ic95_bajo_neto", "ic95_alto_neto", "sharpe_neto_estricto", "ret_anual_%_neto", "vol_anual_%",
            "max_dd_%_neto", "turnover_anual", "exposicion_bruta_media", "carry_anual_%", "spot_anual_%",
            "peor_mes_%", "fecha_peor_mes", "ret_%_2008_ago_dic", "ret_%_2020_feb_mar", "corr_sp500", "corr_dvix"]
    res = tab.set_index("estrategia").loc[principales].reset_index()[cols]
    res.to_csv(f"{RES}/issue15_pre_holdout_resumen.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).T.to_string())

    # retornos mensuales de las principales
    ret = pd.concat({k: cfx.recortar(bts[k], "pre").set_index("ds")[["r_bruto", "r_neto", "r_neto_estricto"]]
                     for k in principales}, axis=1)
    ret.to_csv(f"{RES}/issue15_retornos_mensuales_pre_holdout.csv")

    # estabilidad por subperiodo (descriptivo; no es una prueba nueva)
    sub = []
    for k in principales:
        r = cfx.recortar(bts[k], "pre")
        for a, b in [("2000", "2008"), ("2009", "2016"), ("2017", "2024")]:
            m = r[(r["ds"].dt.year >= int(a)) & (r["ds"].dt.year <= int(b))]["r_neto"]
            lo, hi = ce.ic_sharpe_bootstrap(m, periodos=12, bloque=6)
            sub.append({"estrategia": k, "periodo": f"{a}-{b}", "meses": len(m),
                        "sharpe_neto": ce.sharpe(m, 12), "ic95_bajo": lo, "ic95_alto": hi})
    sub = pd.DataFrame(sub)
    sub.to_csv(f"{RES}/issue15_subperiodos_pre_holdout.csv", index=False)
    print("\nSubperiodos (neto):\n", sub.pivot(index="estrategia", columns="periodo", values="sharpe_neto").round(2))

    # ---------------- contribucion por moneda (elegida) ----------------
    bt, w = bts[elegida], pesos[elegida]
    w = w[(w.index >= cfx.INICIO_EVAL - pd.offsets.MonthEnd(1)) & (w.index < proto.HOLDOUT_INICIO - pd.Timedelta(days=1))]
    X = D["X"].reindex(index=w.index, columns=w.columns).fillna(0.0)
    contrib = (w * X)
    tabla_mon = pd.DataFrame({"peso_medio": w.mean(), "peso_abs_medio": w.abs().mean(),
                              "%_meses_largo": 100 * (w > 0).mean(), "%_meses_corto": 100 * (w < 0).mean(),
                              "contrib_anual_bruta_%": 100 * contrib.mean() * 12,
                              "spread_ida_vuelta_%": 100 * pd.Series(cfx.SPREADS).reindex(w.columns)})
    tabla_mon = tabla_mon.sort_values("contrib_anual_bruta_%", ascending=False)
    tabla_mon.to_csv(f"{RES}/issue15_contribucion_por_moneda_pre_holdout.csv")
    print("\nContribucion por moneda (elegida, pre-hold-out):\n", tabla_mon.round(3).to_string())

    # ---------------- graficos ----------------
    etiquetas = {mejor_carry: ("Carry", COLORES["carry"]), mejor_tsmom: ("TSMOM", COLORES["tsmom"]),
                 elegida: ("Combinación 50/50", COLORES["combo"]), "ew": ("Equiponderada larga", COLORES["ew"])}
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for k, (lab, col) in etiquetas.items():
        r = cfx.recortar(bts[k], "pre").set_index("ds")["r_neto"]
        cap = (1 + r).cumprod()
        ax[0].plot(cap.index, cap.values, color=col, lw=1.8, label=lab)
        ax[0].annotate(lab, (cap.index[-1], cap.values[-1]), xytext=(4, 0), textcoords="offset points",
                       fontsize=8.5, color="#52514e", va="center")
        ax[1].plot(cap.index, 100 * (cap / cap.cummax() - 1), color=col, lw=1.4)
    for a in ax:
        a.grid(color="#e4e3df", lw=0.6)
        a.spines[["top", "right"]].set_visible(False)
        a.axvspan(pd.Timestamp("2008-08-01"), pd.Timestamp("2008-12-31"), color="#f0efec", zorder=0)
        a.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-31"), color="#f0efec", zorder=0)
    ax[0].axhline(1, color="#9a9994", lw=0.8)
    ax[0].set_yscale("log")
    ax[0].yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.8, 1, 1.5, 2, 3, 4]))
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_ylabel("Capital (neto de costos, log)")
    ax[0].set_title("Issue #15 — cartera FX mensual, 2000-01 a 2024-12 (antes del hold-out)", loc="left", fontsize=11)
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")
    ax[1].set_ylabel("Drawdown (%)")
    fig.text(0.01, 0.005, "Sombreado: 2008-08..12 y 2020-02..03. Carry/TSMOM/combinación con vol objetivo; "
             "equiponderada sin apalancar (exposición bruta 1).", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 0.95, 1))
    fig.savefig(f"{RES}/issue15_capital_drawdown_pre_holdout.png", dpi=140)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    orden = tab[tab["rol"] == "candidata"].sort_values("sharpe_neto")
    y = np.arange(len(orden))
    col = [COLORES[f] for f in orden["familia"]]
    ax.errorbar(orden["sharpe_neto"], y, xerr=[orden["sharpe_neto"] - orden["ic95_bajo_neto"],
                orden["ic95_alto_neto"] - orden["sharpe_neto"]], fmt="none", ecolor="#b5b4ae", lw=1.2)
    ax.scatter(orden["sharpe_neto"], y, c=col, s=40, zorder=3)
    ax.scatter(orden["sharpe_bruto"], y, facecolors="none", edgecolors=col, s=40, zorder=3)
    ax.axvline(0, color="#9a9994", lw=0.8)
    ax.axvline(sr_nulo, color="#52514e", lw=0.8, ls="--")
    ax.annotate(f"máx. esperado bajo H0 ({n_pruebas} pruebas) = {sr_nulo:.2f}", (sr_nulo, len(orden) - 0.5),
                xytext=(4, 0), textcoords="offset points", fontsize=8, color="#52514e")
    ax.set_yticks(y, orden["estrategia"], fontsize=7.5)
    ax.set_xlabel("Sharpe anual (relleno = neto con IC95 bootstrap de bloques; hueco = bruto)")
    ax.grid(axis="x", color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Issue #15 — todas las variantes candidatas, 2000-2024", loc="left", fontsize=11)
    fig.subplots_adjust(left=0.27, right=0.97, bottom=0.12, top=0.92)
    fig.savefig(f"{RES}/issue15_sharpe_variantes_pre_holdout.png", dpi=140)
    print("Elegida para el hold-out:", elegida)


if __name__ == "__main__":
    main()
