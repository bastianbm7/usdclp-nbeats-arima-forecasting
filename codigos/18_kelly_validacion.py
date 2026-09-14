# Hoja de ruta del radar-baseline (2026-09-13), Tier 0: valida el hallazgo de
# la seccion 9.5 (ninguna de las 7 variables del estado supera |r|=0.11 con el
# retorno futuro) con un metodo COMPLETAMENTE distinto al RL - el criterio de
# Kelly (f* = mu/sigma^2, la fraccion de capital que maximiza el crecimiento
# geometrico esperado). Si f* ronda 0%, corrobora "no operar" con matematica
# cerrada en vez de depender de que el PPO converja (o no) a esa politica -
# resuelve la duda que dejo el Issue #2 ("¿es un artefacto del entrenamiento?").
#
# Dos variantes, ambas simuladas con la MISMA gestion de riesgo real que el
# resto del proyecto (wf_mod.simular_con_gestion_riesgo), para que las metricas
# sean directamente comparables con PPO/Umbral/Buy-and-hold:
#
# 1. Kelly incondicional: mu y sigma^2 del retorno semanal historico (ventana
#    de train), sin usar ninguna feature - la apuesta que haria alguien que
#    solo conoce la distribucion no condicional del retorno de USD/CLP.
# 2. Kelly condicional: mu_t predicho por una regresion lineal (OLS) sobre las
#    features del estado (fit en train, sin look-ahead), sigma^2_t = la
#    volatilidad GARCH ya pronosticada para esa semana (causal, no una
#    reestimacion). Responde: "incluso combinando las 7 features de forma
#    optima (lineal), ¿el edge implicado es mayor a ruido?"
#
# Reusable desde 19_macro_features_validacion.py con una lista de features
# distinta (7 originales + macro) para comparar si agregar esas variables
# cambia la fraccion de Kelly.

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

entorno_mod = importlib.import_module("11_entorno_trading_rl")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")

RESULTADOS_DIR = "../datos/resultados"
CAPITAL_INICIAL = wf_mod.CAPITAL_INICIAL
LIMITE_APALANCAMIENTO = 1.0  # mismo rango que Box(-1,1) de la accion continua del PPO (Issue #2) - comparable


def ajustar_regresion(df_train, features):
    X = np.column_stack([np.ones(len(df_train)), df_train[features].to_numpy(dtype=float)])
    y_ret = ((df_train["y_next"] - df_train["y"]) / df_train["y"]).to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X, y_ret, rcond=None)
    return coef


def predecir_retorno(df, features, coef):
    X = np.column_stack([np.ones(len(df)), df[features].to_numpy(dtype=float)])
    return X @ coef


def kelly_incondicional(df_train):
    mu, sigma2 = df_train["retorno_1s"].mean(), df_train["retorno_1s"].var()
    return mu / sigma2 if sigma2 > 0 else 0.0


def posiciones_kelly_incondicional(df_train, df_test):
    f = kelly_incondicional(df_train)
    return [float(np.clip(f, -LIMITE_APALANCAMIENTO, LIMITE_APALANCAMIENTO))] * len(df_test), f


def posiciones_kelly_condicional(df_train, df_test, features):
    coef = ajustar_regresion(df_train, features)
    mu_pred = predecir_retorno(df_test, features, coef)
    sigma2 = df_test["vol_garch"].to_numpy(dtype=float) ** 2
    f = np.divide(mu_pred, sigma2, out=np.zeros_like(mu_pred), where=sigma2 > 0)
    return np.clip(f, -LIMITE_APALANCAMIENTO, LIMITE_APALANCAMIENTO).tolist(), f.tolist()


def correr_walkforward_kelly(df_completo, diario, features, nombre_incond, nombre_cond):
    capital_incond = capital_cond = CAPITAL_INICIAL
    partes_incond, partes_cond, fs_incond, fs_cond = [], [], [], []
    for w, (df_train, df_test) in enumerate(wf_mod.ventanas_walkforward(df_completo)):
        pos_incond, f_i = posiciones_kelly_incondicional(df_train, df_test)
        r_incond = wf_mod.simular_con_gestion_riesgo(df_test, diario, pos_incond, nombre_incond, capital_inicial=capital_incond)
        capital_incond = r_incond["capital"].iloc[-1]
        partes_incond.append(r_incond)
        fs_incond.append({"ventana": w + 1, "f_kelly": f_i})

        pos_cond, f_c = posiciones_kelly_condicional(df_train, df_test, features)
        r_cond = wf_mod.simular_con_gestion_riesgo(df_test, diario, pos_cond, nombre_cond, capital_inicial=capital_cond)
        capital_cond = r_cond["capital"].iloc[-1]
        partes_cond.append(r_cond)
        fs_cond.append({"ventana": w + 1, "f_kelly_medio": float(np.mean(f_c)), "f_kelly_max_abs": float(np.max(np.abs(f_c)))})

    resultado_incond, resultado_cond = pd.concat(partes_incond, ignore_index=True), pd.concat(partes_cond, ignore_index=True)
    metricas = [wf_mod.calcular_metricas(resultado_incond, nombre_incond), wf_mod.calcular_metricas(resultado_cond, nombre_cond)]
    return resultado_incond, resultado_cond, metricas, pd.DataFrame(fs_incond), pd.DataFrame(fs_cond)


def graficar_fraccion_kelly(tabla_incond, tabla_cond, path_salida):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = tabla_incond["ventana"].to_numpy()
    ancho = 0.35
    ax.bar(x - ancho / 2, tabla_incond["f_kelly"], width=ancho, label="Kelly incondicional (sin features)", color="steelblue")
    ax.bar(x + ancho / 2, tabla_cond["f_kelly_medio"], width=ancho, label="Kelly condicional (7 features, promedio semanal)", color="darkorange")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color="gray", linewidth=0.6, linestyle="--")
    ax.axhline(-1, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xticks(x)
    ax.set_xlabel("Ventana de walk-forward")
    ax.set_ylabel("Fraccion optima de Kelly (f*)")
    ax.set_title("Kelly criterion: fraccion de capital a apostar, sin entrenar ningun agente")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    diario = pd.read_csv(wf_mod.DATOS_LARGO, parse_dates=["ds"])
    df_completo = entorno_mod.cargar_dataset()
    print(f"Dataset: {len(df_completo)} semanas. Walk-forward: {wf_mod.N_WINDOWS_WF} ventanas x {wf_mod.N_TEST_POR_VENTANA} semanas "
          f"(mismo esquema de 14/17, comparable a la seccion 9.4 del paper)\n")

    resultado_incond, resultado_cond, metricas, tabla_f_incond, tabla_f_cond = correr_walkforward_kelly(
        df_completo, diario, entorno_mod.FEATURES_ESTADO, "Kelly incondicional", "Kelly condicional (7 features)")

    resultado_incond.to_csv(f"{RESULTADOS_DIR}/kelly_incondicional_operaciones.csv", index=False)
    resultado_cond.to_csv(f"{RESULTADOS_DIR}/kelly_condicional_operaciones.csv", index=False)
    tabla_f_incond.to_csv(f"{RESULTADOS_DIR}/kelly_fraccion_incondicional.csv", index=False)
    tabla_f_cond.to_csv(f"{RESULTADOS_DIR}/kelly_fraccion_condicional.csv", index=False)

    referencia = pd.read_csv(f"{RESULTADOS_DIR}/walkforward_gestion_riesgo_metricas.csv")
    tabla = pd.concat([pd.DataFrame(metricas), referencia], ignore_index=True).sort_values("sharpe_anualizado", ascending=False).reset_index(drop=True)
    tabla.to_csv(f"{RESULTADOS_DIR}/kelly_validacion_metricas.csv", index=False)

    graficar_fraccion_kelly(tabla_f_incond, tabla_f_cond, f"{RESULTADOS_DIR}/kelly_fraccion_por_ventana.png")

    print("=== Fraccion de Kelly por ventana (incondicional) ===")
    print(tabla_f_incond.to_string(index=False))
    print("\n=== Fraccion de Kelly por ventana (condicional, 7 features) ===")
    print(tabla_f_cond.to_string(index=False))
    print(f"\n=== Comparacion contra el resto de estrategias ({wf_mod.N_WINDOWS_WF * wf_mod.N_TEST_POR_VENTANA} semanas de test) ===\n")
    print(tabla.to_string(index=False))
