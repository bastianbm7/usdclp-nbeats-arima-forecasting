# Version "barata" de la idea de Bastian: el modelo NO gana ninguna salida
# nueva (sigue siendo Discrete(3)/Box(1), exactamente igual que
# 27_entorno_trading_rl_diario.py y 32_entorno_trading_rl_diario_multidia.py)
# - lo que cambia es la formula FIJA que hoy calcula take_profit=nhits_h1
# siempre. Acá la formula mira la trayectoria h1->h2->h3 de NHITS: si el
# movimiento sigue en la MISMA direccion y la distancia respecto al precio de
# entrada sigue CRECIENDO paso a paso (el modelo "cree" que el movimiento
# continua), apunta al horizonte mas lejano consistente (h2 o h3); si se
# estanca despues de h1, se queda en h1 - mismo comportamiento que el agente
# de 9.17-9.20 en ese caso.
#
# dias_holding queda FIJO en 3 (no parametrizable como en 32): el horizonte
# mas lejano que esta formula puede elegir es h3, asi que la ventana de
# resolucion tiene que cubrir esos 3 dias para que ese objetivo tenga margen
# real de activarse (si el TP se fija en h3 pero la posicion se resuelve en 1
# dia, el objetivo nunca podria alcanzarse incluso si el forecast fuera
# perfecto).
#
# Reusa por composicion, no reescribe: construir_decisiones_multidia() y
# ejecutar_operacion_multidia() de 32 se usan tal cual (ya son genericas
# sobre cualquier take_profit, no solo nhits_h1) - lo unico nuevo es COMO se
# elige ese take_profit antes de llamarlas.

import importlib

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

entorno_diario_mod = importlib.import_module("27_entorno_trading_rl_diario")
multidia_mod = importlib.import_module("32_entorno_trading_rl_diario_multidia")

DATASET_PATH_H3 = "../datos/resultados/dataset_entrenamiento_rl_diario_h3.csv"
DIAS_HOLDING = 3  # fijo - ver nota de diseno arriba

ACCION_CORTO, ACCION_PLANO, ACCION_LARGO = multidia_mod.ACCION_CORTO, multidia_mod.ACCION_PLANO, multidia_mod.ACCION_LARGO
POSICION_POR_ACCION = multidia_mod.POSICION_POR_ACCION
SLIPPAGE_PCT = multidia_mod.SLIPPAGE_PCT
CAPITAL_INICIAL = multidia_mod.CAPITAL_INICIAL
RIESGO_MAX_PCT = multidia_mod.RIESGO_MAX_PCT
K_STOP_LOSS = multidia_mod.K_STOP_LOSS
FEATURES_ESTADO = multidia_mod.FEATURES_ESTADO  # SIN cambios - mismas 9 features de siempre, nhits_h3 no se agrega al estado (ver nota de diseno: esto aisla el experimento a "elegir mejor el TP", no a "el agente ve mas informacion", que es una pregunta distinta)


def cargar_dataset_h3(path=DATASET_PATH_H3):
    df = entorno_diario_mod.cargar_dataset(path)  # reusa retorno_1d/nhits_h1_rel/nhits_h2_rel/macd_rel/rsi_norm/posicion_en_rango tal cual
    return df  # nhits_h3 (nivel de precio) viaja intacto en la columna original - no hace falta relativizarla para el estado porque no se usa ahi, solo en la formula de TP


def elegir_take_profit_adaptativo(y, nhits_h1, nhits_h2, nhits_h3):
    # Devuelve (take_profit, horizonte_elegido in {1,2,3}). "Consistente" =
    # misma direccion que el primer paso Y la distancia respecto a la
    # entrada sigue creciendo (no alcanza con no invertir signo: un
    # forecast que sube 0.3% a h1 y sigue subiendo pero solo 0.31% a h2 no
    # esta "extendiendo la tendencia" de forma que justifique aguantar mas).
    d1, d2, d3 = nhits_h1 - y, nhits_h2 - y, nhits_h3 - y
    if d1 == 0:
        return nhits_h1, 1
    signo = np.sign(d1)
    consistente_h2 = (np.sign(d2) == signo) and (abs(d2) > abs(d1))
    consistente_h3 = consistente_h2 and (np.sign(d3) == signo) and (abs(d3) > abs(d2))
    if consistente_h3:
        return nhits_h3, 3
    if consistente_h2:
        return nhits_h2, 2
    return nhits_h1, 1


def precomputar_salidas_tp_sl_adaptativo(decisiones, k_stop_loss=K_STOP_LOSS):
    filas = {"largo": [], "corto": []}
    horizontes = []
    for _, fila in decisiones.iterrows():
        entrada, vol = fila["y"], fila["vol_garch"]
        take_profit, horizonte = elegir_take_profit_adaptativo(entrada, fila["nhits_h1"], fila["nhits_h2"], fila["nhits_h3"])
        horizontes.append(horizonte)
        precios_periodo = fila["_precios_periodo"]
        fechas_periodo = fila["_fechas_periodo"]
        precio_cierre_periodo = precios_periodo[-1]

        sl_largo = entrada * (1 - k_stop_loss * vol)
        precio_largo, razon_largo, dia_largo = multidia_mod.ejecutar_operacion_multidia(
            entrada, take_profit, sl_largo, k_stop_loss, vol, "largo", precios_periodo, precio_cierre_periodo)
        filas["largo"].append({"stop_loss": sl_largo, "precio_salida": precio_largo, "razon_cierre": razon_largo,
                                "dias_hasta_salida": dia_largo, "fecha_salida": fechas_periodo[dia_largo - 1]})

        sl_corto = entrada * (1 + k_stop_loss * vol)
        precio_corto, razon_corto, dia_corto = multidia_mod.ejecutar_operacion_multidia(
            entrada, take_profit, sl_corto, k_stop_loss, vol, "corto", precios_periodo, precio_cierre_periodo)
        filas["corto"].append({"stop_loss": sl_corto, "precio_salida": precio_corto, "razon_cierre": razon_corto,
                                "dias_hasta_salida": dia_corto, "fecha_salida": fechas_periodo[dia_corto - 1]})

    decisiones = decisiones.drop(columns=["_precios_periodo", "_fechas_periodo"]).copy()
    decisiones["horizonte_tp_elegido"] = horizontes
    for direccion in ["largo", "corto"]:
        for campo in ["stop_loss", "precio_salida", "razon_cierre", "dias_hasta_salida", "fecha_salida"]:
            decisiones[f"{campo}_{direccion}"] = [f[campo] for f in filas[direccion]]
    return decisiones


class USDCLPTradingEnvDiarioTPAdaptativo(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset_path=DATASET_PATH_H3, slippage_pct=SLIPPAGE_PCT, capital_inicial=CAPITAL_INICIAL,
                 riesgo_max_pct=RIESGO_MAX_PCT, k_stop_loss=K_STOP_LOSS, df=None, accion_continua=False):
        super().__init__()
        base = df.reset_index(drop=True) if df is not None else cargar_dataset_h3(dataset_path)
        decisiones = multidia_mod.construir_decisiones_multidia(base, DIAS_HOLDING)  # reusado tal cual de 32
        self.df = precomputar_salidas_tp_sl_adaptativo(decisiones, k_stop_loss)

        self.slippage_pct = slippage_pct
        self.capital_inicial = capital_inicial
        self.riesgo_max_pct = riesgo_max_pct
        self.k_stop_loss = k_stop_loss
        self.accion_continua = accion_continua
        self.features_estado = FEATURES_ESTADO

        if accion_continua:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features_estado),), dtype=np.float32)

        self._paso = 0
        self._posicion_previa = 0.0
        self.capital = capital_inicial
        self.valor_portafolio = 1.0

    def _obs(self):
        return self.df.loc[self._paso, self.features_estado].to_numpy(dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._paso = 0
        self._posicion_previa = 0.0
        self.capital = self.capital_inicial
        self.valor_portafolio = 1.0
        return self._obs(), {}

    def step(self, action):
        # Identico a USDCLPTradingEnvDiarioMultidia.step() (32) - la unica
        # diferencia esta en como se eligio take_profit arriba.
        if self.accion_continua:
            posicion = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        else:
            posicion = POSICION_POR_ACCION[int(action)]
        fila = self.df.loc[self._paso]
        capital_previo = self.capital

        if posicion == 0:
            notional, pnl, razon = 0.0, 0.0, "plano"
        else:
            direccion = "largo" if posicion > 0 else "corto"
            signo = 1.0 if direccion == "largo" else -1.0
            distancia_riesgo = self.k_stop_loss * fila["vol_garch"]
            notional = abs(posicion) * (self.riesgo_max_pct * capital_previo) / distancia_riesgo if distancia_riesgo > 0 else 0.0
            precio_salida, razon = fila[f"precio_salida_{direccion}"], fila[f"razon_cierre_{direccion}"]
            retorno_pct = signo * (precio_salida - fila["y"]) / fila["y"]
            costo_slippage = self.slippage_pct * notional if posicion != self._posicion_previa else 0.0
            pnl = notional * retorno_pct - costo_slippage

        self.capital += pnl
        self.valor_portafolio = self.capital / self.capital_inicial
        self._posicion_previa = posicion
        self._paso += 1

        terminated = self._paso >= len(self.df)
        obs = self._obs() if not terminated else np.zeros(len(FEATURES_ESTADO), dtype=np.float32)
        reward = pnl / capital_previo
        info = {"posicion": posicion, "pnl": pnl, "notional": notional, "razon_cierre": razon, "capital": self.capital}
        return obs, float(reward), terminated, False, info


if __name__ == "__main__":
    # Smoke test: sin el dataset con h3 (35_generar_dataset_rl_diario_h3.py
    # corriendo en background) esto todavia no se puede correr de punta a
    # punta - se deja preparado para correr apenas el dataset este listo.
    import os
    if not os.path.exists(DATASET_PATH_H3):
        print(f"Esperando {DATASET_PATH_H3} (35_generar_dataset_rl_diario_h3.py) - smoke test pendiente.")
    else:
        env = USDCLPTradingEnvDiarioTPAdaptativo()
        print(f"Dataset cargado: {len(env.df)} decisiones (holding fijo de {DIAS_HOLDING} dias)")
        print("Distribucion de horizonte de TP elegido:")
        print(env.df["horizonte_tp_elegido"].value_counts().sort_index())
        obs, _ = env.reset()
        terminado = False
        while not terminado:
            accion = env.action_space.sample()
            obs, reward, terminado, _, info = env.step(accion)
        print(f"\nEpisodio de prueba (accion aleatoria) - capital final: ${env.capital:.2f}")
