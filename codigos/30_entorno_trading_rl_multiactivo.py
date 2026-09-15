# Issue #6: entorno Gym multi-activo para entrenar una politica PPO compartida
# sobre varios pares de FX (USD/CLP + USD/MXN + USD/BRL + USD/COP), en vez de
# solo USD/CLP - responde si "no hay edge explotable" es un hallazgo especifico
# de CLP o generico de tener pocas semanas de un unico activo.
#
# DISENO ELEGIDO (documentado en detalle en la seccion 9.18 del paper) -
# composicion, no reimplementacion: cada par tiene su propia instancia de
# USDCLPTradingEnv (11_entorno_trading_rl.py, SIN TOCAR - el Issue #5 hermano
# lo esta modificando en paralelo para frecuencia diaria, asi que este archivo
# solo lo IMPORTA), con el mismo motor de riesgo real (apalancamiento via
# risk sizing, trailing stop, take-profit, slippage) que ya entrena/evalua al
# agente solo-CLP - no una segunda implementacion de esa economia que podria
# divergir. Este wrapper decide, en cada reset(), que sub-entorno (que par)
# esta "activo" para ese episodio completo, y concatena al estado un ONE-HOT
# del par activo.
#
# Por que one-hot y no pooling ciego: la seccion 9.14 del paper encontro que
# entrenar un modelo LINEAL (OLS) con las filas de 13 pares pooled sin poder
# condicionar por moneda rindio PEOR para CLP que entrenar solo con su propia
# historia - la sensibilidad al cobre varia demasiado entre monedas (0.02 a
# 0.37) para que un promedio ciego tenga sentido. Una red PPO que recibe el
# one-hot del par SI puede aprender una politica condicional (comportarse
# distinto segun que moneda esta operando) en vez de promediar a todas por
# igual - es el mecanismo mas simple y barato disponible en este stack
# (gymnasium + stable-baselines3, sin arquitectura nueva) que se acerca al
# espiritu de "aprender a ponderar que monedas importan" que sugiere el
# Issue #6, sin llegar a una arquitectura cross-attention completa (X-Trend,
# fuera de alcance de este Issue).
#
# "Politica compartida" en el sentido de que UNA sola red se entrena sobre
# episodios de todos los pares (mismos pesos), no una red separada por
# activo - la conditioning via one-hot es lo que le permite especializarse
# sin dejar de compartir la mayoria de los pesos entre pares.

import importlib

import numpy as np
import gymnasium as gym
from gymnasium import spaces

entorno_mod = importlib.import_module("11_entorno_trading_rl")


class MultiFXTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dfs_por_par, diario_paths_por_par, modo_muestreo="aleatorio", par_fijo=None,
                 slippage_pct=entorno_mod.SLIPPAGE_PCT, capital_inicial=entorno_mod.CAPITAL_INICIAL,
                 riesgo_max_pct=entorno_mod.RIESGO_MAX_PCT, k_stop_loss=entorno_mod.K_STOP_LOSS,
                 accion_continua=False, modo_recompensa="cruda", seed=None):
        super().__init__()
        # dfs_por_par: {par: df ya cargado con entorno_mod.cargar_dataset(path_del_par),
        # ya recortado train/test si corresponde}. diario_paths_por_par: {par: ruta al
        # CSV diario de ESE par} - USDCLPTradingEnv necesita la ruta (no un df en
        # memoria, no se le agrego ese parametro a 11 para no tocarlo) para precomputar
        # las salidas TP/SL con precios reales dia a dia.
        #
        # modo_muestreo="aleatorio" (default, entrenamiento): cada reset() elige un par
        # uniforme al azar - el episodio completo (hasta terminated=True) se juega
        # sobre ESE par, igual que un episodio normal de USDCLPTradingEnv. par_fijo
        # (evaluacion): fuerza siempre el mismo par en cada reset() - necesario para
        # caminar deterministicamente el test set de un par especifico (ej. CLP) sin
        # cambiar la dimension del one-hot respecto al entrenamiento (dfs_por_par debe
        # seguir incluyendo TODOS los pares de entrenamiento, aunque solo se camine uno).
        if not dfs_por_par:
            raise ValueError("dfs_por_par no puede estar vacio")
        self.pares = list(dfs_por_par.keys())
        self.n_pares = len(self.pares)
        self.par_index = {par: i for i, par in enumerate(self.pares)}
        self.modo_muestreo = modo_muestreo
        self.par_fijo = par_fijo
        self.accion_continua = accion_continua

        self.sub_envs = {
            par: entorno_mod.USDCLPTradingEnv(
                diario_path=diario_paths_por_par[par], df=dfs_por_par[par],
                slippage_pct=slippage_pct, capital_inicial=capital_inicial,
                riesgo_max_pct=riesgo_max_pct, k_stop_loss=k_stop_loss,
                accion_continua=accion_continua, modo_recompensa=modo_recompensa,
            )
            for par in self.pares
        }
        ref_env = self.sub_envs[self.pares[0]]
        self.action_space = ref_env.action_space
        n_features_base = ref_env.observation_space.shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features_base + self.n_pares,), dtype=np.float32)

        self._par_activo = None
        self._rng = np.random.default_rng(seed)
        self._idx_ciclico = 0

    def _onehot(self, par):
        v = np.zeros(self.n_pares, dtype=np.float32)
        v[self.par_index[par]] = 1.0
        return v

    def _elegir_par(self, options):
        if options and "par" in options:
            return options["par"]
        if self.par_fijo is not None:
            return self.par_fijo
        if self.modo_muestreo == "ciclico":
            par = self.pares[self._idx_ciclico % self.n_pares]
            self._idx_ciclico += 1
            return par
        return self.pares[self._rng.integers(self.n_pares)]  # "aleatorio"

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._par_activo = self._elegir_par(options)
        base_obs, info = self.sub_envs[self._par_activo].reset(seed=seed)
        obs = np.concatenate([base_obs, self._onehot(self._par_activo)]).astype(np.float32)
        return obs, {**info, "par": self._par_activo}

    def step(self, action):
        base_obs, reward, terminated, truncated, info = self.sub_envs[self._par_activo].step(action)
        obs = np.concatenate([base_obs, self._onehot(self._par_activo)]).astype(np.float32)
        return obs, reward, terminated, truncated, {**info, "par": self._par_activo}

    @property
    def capital(self):
        return self.sub_envs[self._par_activo].capital


if __name__ == "__main__":
    # Smoke test minimo: reusa el propio dataset de CLP bajo 2 nombres de par
    # distintos (no requiere que los datasets de MXN/BRL/COP de 29 ya existan)
    # solo para validar la mecanica del wrapper (dimension del one-hot,
    # limites de episodio, delegacion correcta a cada sub-entorno).
    df_clp = entorno_mod.cargar_dataset()
    mitad = len(df_clp) // 2
    dfs = {"USDCLP_A": df_clp.iloc[:mitad].reset_index(drop=True), "USDCLP_B": df_clp.iloc[mitad:].reset_index(drop=True)}
    diarios = {"USDCLP_A": entorno_mod.DATOS_DIARIOS_PATH, "USDCLP_B": entorno_mod.DATOS_DIARIOS_PATH}

    env = MultiFXTradingEnv(dfs, diarios, modo_muestreo="ciclico", seed=42)
    print(f"observation_space: {env.observation_space.shape}, action_space: {env.action_space}")
    for _ in range(2):
        obs, info = env.reset()
        print(f"Episodio sobre par={info['par']}, obs.shape={obs.shape}")
        terminado = False
        pasos = 0
        while not terminado:
            accion = env.action_space.sample()
            obs, reward, terminado, _, info = env.step(accion)
            pasos += 1
        print(f"  {pasos} pasos, capital final ${env.capital:.2f}")
    print("Smoke test OK")
