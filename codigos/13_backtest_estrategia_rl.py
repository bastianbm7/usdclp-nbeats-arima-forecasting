# Backtest con metricas financieras (Fase 3, Issue #1) - no solo retorno
# acumulado como el chequeo rapido de 12_entrenar_agente_rl.py. Compara 3
# estrategias sobre el mismo holdout (test, nunca visto en entrenamiento):
# agente PPO, buy-and-hold, y una regla simple de umbral (la Opcion A que se
# descarto a favor de RL) - el control barato que permite decir honestamente
# si el RL aporta algo por sobre algo mucho mas simple.

import importlib

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

entorno_mod = importlib.import_module("11_entorno_trading_rl")
entrenar_mod = importlib.import_module("12_entrenar_agente_rl")

RESULTADOS_DIR = "../datos/resultados"
MODELO_PATH = f"{RESULTADOS_DIR}/modelo_ppo_trading.zip"

UMBRAL_UMBRAL_SIMPLE = 0.003  # 0.3% de desviacion del forecast h1 vs precio actual para operar


def correr_agente_ppo(df, modelo):
    env = entorno_mod.USDCLPTradingEnv(df=df)
    obs, _ = env.reset()
    filas = []
    terminado = False
    while not terminado:
        accion, _ = modelo.predict(obs, deterministic=True)
        obs, reward, terminado, _, info = env.step(accion)
        filas.append({"ds": df["ds"].iloc[env._paso - 1], "retorno_neto": reward, "posicion": info["posicion"], "valor_portafolio": info["valor_portafolio"]})
    return pd.DataFrame(filas)


def correr_buy_and_hold(df):
    retorno_semana = (df["y_next"] - df["y"]) / df["y"]
    valor = (1 + retorno_semana).cumprod()
    return pd.DataFrame({"ds": df["ds"], "retorno_neto": retorno_semana, "posicion": 1.0, "valor_portafolio": valor})


def correr_umbral_simple(df, umbral=UMBRAL_UMBRAL_SIMPLE, slippage_pct=entorno_mod.SLIPPAGE_PCT):
    forecast_rel = (df["nhits_h1"] - df["y"]) / df["y"]
    posicion = np.where(forecast_rel > umbral, 1.0, np.where(forecast_rel < -umbral, -1.0, 0.0))
    posicion_previa = np.roll(posicion, 1)
    posicion_previa[0] = 0.0
    retorno_semana = (df["y_next"] - df["y"]) / df["y"]
    costo = slippage_pct * np.abs(posicion - posicion_previa)
    retorno_neto = posicion * retorno_semana - costo
    valor = (1 + retorno_neto).cumprod()
    return pd.DataFrame({"ds": df["ds"], "retorno_neto": retorno_neto, "posicion": posicion, "valor_portafolio": valor})


def calcular_metricas(resultado, nombre):
    r = resultado["retorno_neto"]
    valor_final = resultado["valor_portafolio"].iloc[-1]
    retorno_total = valor_final - 1
    sharpe = (r.mean() / r.std()) * np.sqrt(52) if r.std() > 0 else np.nan  # anualizado, base semanal
    equity = resultado["valor_portafolio"]
    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()
    operaciones = (resultado["posicion"].diff().fillna(resultado["posicion"].iloc[0]) != 0).sum()
    semanas_ganadoras = (r > 0).sum()
    win_rate = semanas_ganadoras / len(r) if len(r) > 0 else np.nan
    return {
        "estrategia": nombre, "retorno_total_%": 100 * retorno_total, "sharpe_anualizado": sharpe,
        "max_drawdown_%": 100 * max_drawdown, "win_rate_%": 100 * win_rate, "operaciones": operaciones,
    }


if __name__ == "__main__":
    df_completo = entorno_mod.cargar_dataset()
    _, df_test = entrenar_mod.split_train_test(df_completo)
    print(f"Backtest sobre holdout: {len(df_test)} semanas ({df_test['ds'].min().date()} a {df_test['ds'].max().date()})")

    modelo = PPO.load(MODELO_PATH)

    resultados = {
        "PPO (RL)": correr_agente_ppo(df_test, modelo),
        "Buy-and-hold": correr_buy_and_hold(df_test),
        "Umbral simple (Opcion A)": correr_umbral_simple(df_test),
    }

    tabla = pd.DataFrame([calcular_metricas(r, nombre) for nombre, r in resultados.items()])
    tabla = tabla.sort_values("sharpe_anualizado", ascending=False).reset_index(drop=True)
    tabla.to_csv(f"{RESULTADOS_DIR}/backtest_estrategia_rl_metricas.csv", index=False)

    for nombre, r in resultados.items():
        r.to_csv(f"{RESULTADOS_DIR}/backtest_estrategia_rl_{nombre.split()[0].lower()}.csv", index=False)

    print("\n=== Backtest en holdout genuino (nunca visto en entrenamiento) ===\n")
    print(tabla.to_string(index=False))
