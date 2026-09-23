# Issue #13, Fase 1 (continuacion): refuerza con bootstrap SPA de Hansen
# (arch.bootstrap.SPA) los pares moneda-commodity que sobrevivieron la
# correccion FDR Y el umbral |r|=0.11 en 52_screening_correlacion_commodities_fx.py.
# Correlacion no es lo mismo que "una estrategia le gana de verdad a no hacer
# nada" (mismo principio que 22_kelly_diario_cobre.py, que valida el hallazgo
# de correlacion con un backtest antes de festejarlo) - SPA es la version con
# respaldo estadistico formal de esa misma pregunta: ¿la mejor estrategia de
# la familia le gana al benchmark de forma que sobreviva un bootstrap, o es
# el tipo de resultado que el propio data-snooping de haber probado 90
# combinaciones podria producir por azar?
#
# Solo se testean los supervivientes con rezago=+1 (10 de los 11 de Fase 1):
# es la unica direccion directamente "operable" como regla diaria (senal de
# hoy -> posicion para el retorno de manana), igual que el resto del proyecto
# (copper_ret_1d en 9.13/9.14). El superviviente de rezago=+2 (NOK-WTI) queda
# documentado como hallazgo secundario, no se lleva a la construccion de una
# estrategia (ver seccion del paper).
#
# Estrategia de juguete para el test (deliberadamente simple, sin Kelly ni
# gestion de riesgo - el punto es aislar si la DIRECCION del commodity tiene
# poder predictivo real, no optimizar cuanto se ganaria): posicion = signo(r)
# * signo(commodity_ret_1d), retorno realizado = posicion * retorno de la
# moneda al dia siguiente. Benchmark = no operar (retorno 0 todos los dias).

import importlib

import numpy as np
import pandas as pd
from arch.bootstrap import SPA

screening_mod = importlib.import_module("52_screening_correlacion_commodities_fx")

RESULTADOS_DIR = "../datos/resultados"
SEED = 42
REPS = 5000


def construir_retornos_estrategia(moneda_df, commodity_df, signo_r):
    par = screening_mod.construir_par(moneda_df, commodity_df)
    senal = np.sign(par["commodity_ret_1d"]) * np.sign(signo_r)
    retorno_futuro = par["retorno_1d"].shift(-1)
    valido = retorno_futuro.notna()
    retorno_estrategia = (senal * retorno_futuro)[valido].to_numpy()
    return retorno_estrategia


if __name__ == "__main__":
    tabla_screening = pd.read_csv(f"{RESULTADOS_DIR}/fase1_screening_correlacion_completo.csv")
    supervivientes = tabla_screening[tabla_screening["hallazgo_real"]].copy()
    supervivientes_lag1 = supervivientes[supervivientes["rezago"] == 1].reset_index(drop=True)
    print(f"Supervivientes de Fase 1 con rezago=+1 (operables): {len(supervivientes_lag1)}")
    print(supervivientes_lag1[["moneda", "commodity", "correlacion", "p_valor_fdr"]].to_string(index=False))

    monedas_data = {m: screening_mod.cargar_moneda(m) for m in screening_mod.MONEDAS}
    codigo_por_nombre = {v: k for k, v in screening_mod.COMMODITIES.items()}
    commodities_data = {c: screening_mod.cargar_commodity(c) for c in screening_mod.COMMODITIES}

    retornos_por_estrategia = {}
    n_min = None
    for _, fila in supervivientes_lag1.iterrows():
        codigo_commodity = codigo_por_nombre[fila["commodity"]]
        r = construir_retornos_estrategia(
            monedas_data[fila["moneda"]], commodities_data[codigo_commodity], fila["correlacion"])
        nombre = f"{fila['moneda']}_x_{fila['commodity']}"
        retornos_por_estrategia[nombre] = r
        n_min = len(r) if n_min is None else min(n_min, len(r))

    # SPA necesita un T comun para benchmark y modelos - se recorta todo a la
    # ventana comun mas reciente (misma longitud n_min) para no perder
    # observaciones de mas de lo necesario.
    modelos = pd.DataFrame({nombre: r[-n_min:] for nombre, r in retornos_por_estrategia.items()})
    benchmark_retorno = np.zeros(n_min)  # "no operar" - retorno 0 todos los dias

    perdida_modelos = -modelos  # SPA opera sobre PERDIDAS (Hansen 2005) - minimizar perdida = maximizar retorno
    perdida_benchmark = -benchmark_retorno

    print(f"\nCorriendo SPA de Hansen (bootstrap estacionario, {REPS} repeticiones, T={n_min} dias comunes)...")
    spa = SPA(perdida_benchmark, perdida_modelos, reps=REPS, seed=SEED)
    spa.compute()

    # spa.pvalues es UN solo resultado conjunto (lower/consistent/upper): la
    # hipotesis nula de que NINGUNA de las 10 estrategias le gana al
    # benchmark (no operar), corrigiendo por haber probado 10 a la vez (el
    # mismo espiritu que la correccion FDR, pero especifico para comparar
    # multiples estrategias de trading - Hansen 2005 / White 2000).
    p_conjunto = spa.pvalues
    print("\n=== P-valor conjunto de SPA (H0: ninguna estrategia le gana al benchmark) ===")
    print(p_conjunto.to_string())

    # Que estrategias especificas se identifican como mejores que el
    # benchmark de forma individual, con cada uno de los 3 criterios de
    # Hansen (lower = mas exigente, upper = menos exigente).
    mejores_por_criterio = {}
    for tipo in ["lower", "consistent", "upper"]:
        mejores_por_criterio[tipo] = set(modelos.columns[spa.better_models(pvalue=0.05, pvalue_type=tipo)])

    tabla_spa = pd.DataFrame({
        "estrategia": modelos.columns,
        "retorno_promedio_diario": modelos.mean().values,
        "sharpe_anualizado": (modelos.mean() / modelos.std() * np.sqrt(252)).values,
    })
    for tipo in ["lower", "consistent", "upper"]:
        tabla_spa[f"mejor_que_benchmark_{tipo}"] = tabla_spa["estrategia"].isin(mejores_por_criterio[tipo])
    tabla_spa = tabla_spa.sort_values("sharpe_anualizado", ascending=False).reset_index(drop=True)

    tabla_spa.to_csv(f"{RESULTADOS_DIR}/fase1_bootstrap_spa.csv", index=False)
    p_conjunto.to_csv(f"{RESULTADOS_DIR}/fase1_bootstrap_spa_pvalor_conjunto.csv")
    print("\n=== Por estrategia: Sharpe y si se identifica individualmente mejor que 'no operar' ===")
    print(tabla_spa.to_string(index=False))
    print(f"\nSobreviven SPA (criterio 'consistent', el estandar de Hansen): "
          f"{tabla_spa['mejor_que_benchmark_consistent'].sum()} de {len(tabla_spa)}")
