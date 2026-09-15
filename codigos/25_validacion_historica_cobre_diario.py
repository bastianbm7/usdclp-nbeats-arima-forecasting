# Paso 1 del Issue #5 (reconstruccion RL diaria), pedido explicitamente por el
# Issue antes de comprometerse a la reconstruccion completa: el backtest de
# 22_kelly_diario_cobre.py (+86.2% / Sharpe 4.57, corregido a +1.9% con sizing
# realista en 9.14 del paper) solo se probo en el periodo mas reciente
# (2025-07 a 2026-09, ~14 meses) - el unico tramo donde el cobre estuvo en un
# regimen alcista sostenido. Este script repite EXACTAMENTE el mismo metodo
# (Kelly condicional via OLS + umbral simple, mismo walk-forward de 5 ventanas
# x 60 dias) pero anclado a cortes de fecha historicos mas viejos, para ver si
# el Sharpe ~4.5 sobrevive fuera del tramo reciente o es especifico de el.
#
# Como reusar el walk-forward para "el pasado": wf_mod.ventanas_walkforward()
# siempre toma las ULTIMAS N_WINDOWS_WF*N_TEST_POR_VENTANA filas del dataframe
# que se le pasa como periodo de test (con todo lo anterior como train). No
# hace falta reescribir esa logica - alcanza con truncar el dataset diario a
# una fecha de corte historica ANTES de llamar a la misma funcion, y el "final"
# de ese dataframe truncado pasa a ser el periodo de test de esa epoca.
#
# Todas las funciones (preparar_dataset_diario, posiciones_kelly,
# posiciones_umbral_cobre, simular, calcular_metricas) se importan de 22, no
# se reimplementan - si el metodo cambia en 22, este script queda consistente
# automaticamente.

import importlib

import matplotlib.pyplot as plt
import pandas as pd

kelly_diario_mod = importlib.import_module("22_kelly_diario_cobre")
features_mod = importlib.import_module("19_features_nuevas_validacion")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")

RESULTADOS_DIR = "../datos/resultados"
CAPITAL_INICIAL = kelly_diario_mod.CAPITAL_INICIAL
N_WINDOWS_WF = kelly_diario_mod.N_WINDOWS_WF
N_TEST_POR_VENTANA = kelly_diario_mod.N_TEST_POR_VENTANA
FEATURES_DIARIAS = kelly_diario_mod.FEATURES_DIARIAS

# Cortes de fecha: cada uno trunca el dataset diario ANTES de ese limite, asi
# que el periodo de test (las ultimas 300 filas de la porcion truncada) cae
# antes de esa fecha. "reciente" no trunca nada - es el mismo resultado que
# 22_kelly_diario_cobre.py, usado como referencia para comparar directamente.
CORTES = {
    "reciente_2025_2026 (ref. 22)": None,
    "2020_2021": "2021-12-31",
    "2017_2018": "2018-12-31",
    "2015_2016": "2016-12-31",
}


def correr_periodo(df_completo, nombre_periodo, corte):
    df = df_completo if corte is None else df_completo[df_completo["ds"] <= pd.Timestamp(corte)].reset_index(drop=True)
    n_necesario = N_WINDOWS_WF * N_TEST_POR_VENTANA
    if len(df) < n_necesario * 2:  # exige al menos tanta historia de train como de test
        print(f"  [omitido] {nombre_periodo}: solo {len(df)} filas disponibles antes del corte, insuficiente")
        return None, None

    filas_metricas = []
    resultados = {}

    ventanas_test = list(wf_mod.ventanas_walkforward(df, N_WINDOWS_WF, N_TEST_POR_VENTANA))
    fecha_ini_test = ventanas_test[0][1]["ds"].iloc[0]
    fecha_fin_test = ventanas_test[-1][1]["ds"].iloc[-1]

    df_test_completo = pd.concat([t for _, t in ventanas_test], ignore_index=True)
    r_bh = kelly_diario_mod.simular(df_test_completo, [1.0] * len(df_test_completo), "Buy-and-hold", CAPITAL_INICIAL)
    resultados["Buy-and-hold"] = r_bh
    filas_metricas.append(kelly_diario_mod.calcular_metricas(r_bh, "Buy-and-hold"))

    capital = CAPITAL_INICIAL
    partes_umbral = []
    for df_train, df_test in ventanas_test:
        r = kelly_diario_mod.simular(df_test, kelly_diario_mod.posiciones_umbral_cobre(df_test), "Umbral cobre", capital)
        capital = r["capital"].iloc[-1]
        partes_umbral.append(r)
    resultados["Umbral cobre"] = pd.concat(partes_umbral, ignore_index=True)
    filas_metricas.append(kelly_diario_mod.calcular_metricas(resultados["Umbral cobre"], "Umbral cobre"))

    resultados["Kelly diario (con cobre)"] = kelly_diario_mod.correr_walkforward(df, FEATURES_DIARIAS, "Kelly diario (con cobre)")
    filas_metricas.append(kelly_diario_mod.calcular_metricas(resultados["Kelly diario (con cobre)"], "Kelly diario (con cobre)"))

    tabla = pd.DataFrame(filas_metricas)
    tabla.insert(0, "periodo", nombre_periodo)
    tabla.insert(1, "test_desde", fecha_ini_test.date())
    tabla.insert(2, "test_hasta", fecha_fin_test.date())
    return tabla, resultados


def graficar_sharpe_por_periodo(tabla_todos, path_salida):
    pivot = tabla_todos.pivot(index="periodo", columns="estrategia", values="sharpe_anualizado")
    pivot = pivot.reindex(index=list(CORTES.keys()))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = range(len(pivot.index))
    ancho = 0.25
    colores = {"Buy-and-hold": "black", "Umbral cobre": "darkorange", "Kelly diario (con cobre)": "crimson"}
    for i, estrategia in enumerate(pivot.columns):
        ax.bar([xi + i * ancho for xi in x], pivot[estrategia], width=ancho, label=estrategia, color=colores.get(estrategia))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks([xi + ancho for xi in x])
    ax.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax.set_ylabel("Sharpe anualizado")
    ax.set_title("Sharpe de la senal de cobre por periodo historico\n(mismo metodo de 22_kelly_diario_cobre.py, distintos cortes de fecha)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    macro = features_mod.cargar_macro()
    df_completo = kelly_diario_mod.preparar_dataset_diario(macro)
    print(f"Dataset diario completo: {len(df_completo)} obs, {df_completo['ds'].min().date()} a {df_completo['ds'].max().date()}\n")

    tablas = []
    for nombre_periodo, corte in CORTES.items():
        print(f"--- {nombre_periodo} (corte={corte}) ---")
        tabla, _ = correr_periodo(df_completo, nombre_periodo, corte)
        if tabla is not None:
            print(tabla.to_string(index=False))
            print()
            tablas.append(tabla)

    tabla_todos = pd.concat(tablas, ignore_index=True)
    tabla_todos.to_csv(f"{RESULTADOS_DIR}/validacion_historica_cobre_diario_metricas.csv", index=False)
    graficar_sharpe_por_periodo(tabla_todos, f"{RESULTADOS_DIR}/validacion_historica_cobre_diario_sharpe_por_periodo.png")

    print("=== Resumen: Sharpe de 'Kelly diario (con cobre)' por periodo ===")
    resumen = tabla_todos[tabla_todos["estrategia"] == "Kelly diario (con cobre)"][["periodo", "test_desde", "test_hasta", "retorno_total_%", "sharpe_anualizado", "operaciones"]]
    print(resumen.to_string(index=False))
