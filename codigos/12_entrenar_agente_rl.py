# Entrena el agente PPO sobre el entorno de 11_entorno_trading_rl.py. Split
# train/test simple por corte de fecha (no walk-forward completo como el resto
# del proyecto - ver limitacion conocida en el paper): las ultimas
# N_SEMANAS_TEST quedan afuera del entrenamiento, como holdout genuino para
# 13_backtest_estrategia_rl.py.
#
# TOTAL_TIMESTEPS >> largo del episodio de train a proposito: el agente
# recorre la misma secuencia historica muchas veces (multiples episodios), no
# hace falta un dataset con miles de semanas unicas - asi se entrena RL de un
# solo activo (ver FinRL, la referencia metodologica de este proyecto).
#
# Import via importlib porque "11_entorno_trading_rl" no es un identificador
# valido para `import` directo (empieza con digito) - mismo motivo que en
# 10_generar_dataset_rl.py cuando se probo el pipeline a mano.

import importlib

from stable_baselines3 import PPO

entorno_mod = importlib.import_module("11_entorno_trading_rl")

RESULTADOS_DIR = "../datos/resultados"
MODELO_PATH = f"{RESULTADOS_DIR}/modelo_ppo_trading.zip"

N_SEMANAS_TEST = 52
TOTAL_TIMESTEPS = 100_000
SEED = 42


def split_train_test(df, n_test=N_SEMANAS_TEST):
    return df.iloc[:-n_test].reset_index(drop=True), df.iloc[-n_test:].reset_index(drop=True)


def evaluar_politica(env, modelo, determinista=True):
    obs, _ = env.reset()
    terminado = False
    while not terminado:
        accion, _ = modelo.predict(obs, deterministic=determinista)
        obs, reward, terminado, _, info = env.step(accion)
    return env.valor_portafolio


def evaluar_buy_and_hold(df):
    return df["y_next"].iloc[-1] / df["y"].iloc[0]


if __name__ == "__main__":
    df_completo = entorno_mod.cargar_dataset()
    df_train, df_test = split_train_test(df_completo)
    print(f"Train: {len(df_train)} semanas ({df_train['ds'].min().date()} a {df_train['ds'].max().date()})")
    print(f"Test (holdout, no visto en entrenamiento): {len(df_test)} semanas ({df_test['ds'].min().date()} a {df_test['ds'].max().date()})")

    env_train = entorno_mod.USDCLPTradingEnv(df=df_train)
    modelo = PPO("MlpPolicy", env_train, seed=SEED, verbose=0)
    modelo.learn(total_timesteps=TOTAL_TIMESTEPS)
    modelo.save(MODELO_PATH)
    print(f"\nModelo guardado en {MODELO_PATH}")

    valor_train = evaluar_politica(entorno_mod.USDCLPTradingEnv(df=df_train), modelo)
    valor_test = evaluar_politica(entorno_mod.USDCLPTradingEnv(df=df_test), modelo)
    bh_train = evaluar_buy_and_hold(df_train)
    bh_test = evaluar_buy_and_hold(df_test)

    print("\n=== Chequeo rapido (metricas financieras completas en 13_backtest_estrategia_rl.py) ===")
    print(f"Train - agente PPO: {valor_train:.4f} | buy-and-hold: {bh_train:.4f}")
    print(f"Test  - agente PPO: {valor_test:.4f} | buy-and-hold: {bh_test:.4f}")
