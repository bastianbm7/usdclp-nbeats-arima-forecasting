# Issue #15: APERTURA UNICA del hold-out (>= 2025-01-01) para la estrategia (a)
# de 86 - carry completo con trailing 1.5 sigma en la pata larga + momentum 6m
# solo en la mitad menos volatil - pedida por Bastian. Configuracion congelada
# tal como esta en 86 (commit f6f53f4); nada se ajusta aca.
#
# Contexto honesto para leer el resultado: el hold-out ya se habia abierto en
# 82 para la estrategia original (Sharpe neto 0.47), asi que esta apertura no
# es "a ciegas" del todo (se sabia que el periodo fue bueno para el carry y
# malo para el momentum). La original se recalcula al lado solo como
# referencia (mismo numero de 82, no es una apertura nueva).
# Ademas se reporta 2015 en adelante, dividido en antes / despues del hold-out.
# El script se niega a correr si la apertura ya esta en el registro.

import importlib

import numpy as np
import pandas as pd

m86 = importlib.import_module("86_cartera_fx_estrategia_patas_largas_baja_vol")  # registra los tipos nuevos en cf
cf = importlib.import_module("cartera_fx")
proto = importlib.import_module("protocolo_evaluacion")
ce = importlib.import_module("costos_y_estadistica")
m83 = importlib.import_module("83_cartera_fx_stop_take_profit")
m84 = importlib.import_module("84_cartera_fx_stops_por_posicion_trailing")

RESULTADOS_DIR = "../datos/resultados"
SCRIPT = "87_cartera_fx_holdout_estrategia_a.py"
NOMBRE_A = "NUEVA (a): carry completo c/ trailing pata larga + momentum baja vol"
NOMBRE_O = "Original: carry + momentum (81)"


def metricas_periodo(r, nombre, periodo):
    r = r.dropna()
    lo, hi = ce.ic_sharpe_bootstrap(r.to_numpy(), periodos=12, n_boot=5000, bloque=6)
    cap = np.cumprod(1 + r.to_numpy())
    return {"estrategia": nombre, "periodo": periodo, "meses": len(r),
            "sharpe_neto": ce.sharpe(r, 12), "ic95_bajo": lo, "ic95_alto": hi,
            "ret_anual_%": 100 * (cap[-1] ** (12 / len(r)) - 1), "ret_total_%": 100 * (cap[-1] - 1),
            "vol_anual_%": 100 * r.std() * np.sqrt(12), "max_dd_%": 100 * cf.max_drawdown(r),
            "peor_mes_%": 100 * r.min(), "meses_positivos_%": 100 * (r > 0).mean()}


def main():
    reg = pd.read_csv(proto.RUTA_REGISTRO)
    if ((reg["script"] == SCRIPT) & (reg["usa_holdout"] == True)).any():
        raise SystemExit("El hold-out de la estrategia (a) ya se abrio; no se vuelve a abrir.")

    D = cf.cargar()
    P = m83.grilla_diaria(D)
    TRI = m84.indice_retorno_total(D, P)
    cfgs = {n: (c, t) for n, c, t, _ in m86.VARIANTES}
    series = {}
    for nombre in [NOMBRE_O, NOMBRE_A]:
        cfg, trailing = cfgs[nombre]
        bt = (m84.backtest(D, P, TRI, cfg, "trailing_moneda", m86.K_TRAILING, filtro=m86.filtro_pata_larga_carry)
              if trailing else cf.backtest(D, cfg)[0])
        series[nombre] = bt.set_index("ds")["r_neto"]
    df = pd.DataFrame(series)
    df = df[df.index >= "2015-01-01"]

    filas = []
    for nombre, etiqueta in [(NOMBRE_O, "Original"), (NOMBRE_A, "Estrategia (a)")]:
        r = df[nombre]
        filas.append(metricas_periodo(r[r.index < proto.HOLDOUT_INICIO], etiqueta, "2015-2024 (usado para elegir)"))
        filas.append(metricas_periodo(r[r.index >= proto.HOLDOUT_INICIO], etiqueta, "2025-01 a 2026-08 (hold-out)"))
        filas.append(metricas_periodo(r, etiqueta, "2015-2026 completo"))
    tabla = pd.DataFrame(filas)
    tabla.to_csv(f"{RESULTADOS_DIR}/issue15_estrategia_a_desde_2015_y_holdout.csv", index=False)

    anual = (1 + df).groupby(df.index.year).prod() - 1
    anual.columns = ["Original", "Estrategia (a)"]
    anual = 100 * anual
    anual.to_csv(f"{RESULTADOS_DIR}/issue15_estrategia_a_retorno_por_anio.csv")
    df.to_csv(f"{RESULTADOS_DIR}/issue15_estrategia_a_retornos_mensuales_desde_2015.csv")

    pd.set_option("display.width", 220)
    print(tabla.round(2).to_string(index=False))
    print("\nRetorno neto por anio calendario (%) - 2026 hasta agosto:")
    print(anual.round(2).to_string())

    ho = tabla[(tabla["estrategia"] == "Estrategia (a)") & (tabla["periodo"].str.contains("hold-out"))].iloc[0]
    proto.registrar_pruebas([{"prueba": f"HOLD-OUT {NOMBRE_A}", "frecuencia": "mensual (trailing diario)",
                              "metrica": "sharpe_neto_holdout", "valor": ho["sharpe_neto"], "n": ho["meses"],
                              "usa_holdout": True, "nota": "apertura unica; el hold-out ya se habia abierto en 82 para la original"}],
                            issue="#15", script=SCRIPT)


if __name__ == "__main__":
    main()
