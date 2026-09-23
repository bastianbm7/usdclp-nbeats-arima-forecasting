# Issue #13, Fase 4 (Nivel 1): porta el MECANISMO de Variable Selection
# Network (VSN) + atencion del Momentum Transformer (Wood, Giegerich, Roberts
# & Zohren 2021, arXiv:2112.08534; repo ancla kieranjwood/trading-momentum-transformer,
# clonado aparte SOLO para leer mom_trans/momentum_transformer.py y
# mom_trans/deep_momentum_network.py como referencia de arquitectura - no se
# usa su pipeline de datos Pinnacle/Quandl, no esta commiteado en este repo).
#
# DECISION DE DISEÑO 1 (documentada, no trivial): el repo ancla es
# TensorFlow 1.x-style Keras (keras.backend ops, tf.keras.Model funcional) -
# no es codigo que se pueda "importar y usar" en un proyecto PyTorch
# (neuralforecast, stable-baselines3). Y neuralforecast.models.TFT (ya usado
# en la Fase 3, 56_tft_panel_walkforward.py) YA es una reimplementacion fiel
# de la MISMA arquitectura VSN+atencion en PyTorch - portar esa parte desde
# cero seria reescribir dos veces lo mismo. La pieza REALMENTE nueva del
# Momentum Transformer frente a un TFT generico (y frente a la Fase 3 de este
# mismo Issue) no es la arquitectura de atencion en si, es el MECANISMO DE
# ENTRENAMIENTO: en vez de pronosticar un valor (retorno) y despues convertir
# ese forecast en una posicion via Kelly (lo que hace la Fase 3), el Momentum
# Transformer entrena la red para emitir la POSICION directamente (salida
# tanh en [-1,1]) optimizando el Sharpe ratio de la cartera como funcion de
# perdida (diferenciable, ver SharpeLoss en deep_momentum_network.py:32-48) -
# sin objetivo intermedio de prediccion de precio/retorno.
#
# Por eso este script porta: (a) el bloque de Variable Selection Network
# (gated residual network + softmax sobre las features, exactamente la
# formula de gated_residual_network()/lstm_combine_and_mask() del archivo
# ancla, reimplementada en PyTorch) + (b) un bloque de auto-atencion causal
# simple (la pieza "Transformer" del nombre) + (c) la SharpeLoss identica
# (misma formula: -mean(pos*y)/std(pos*y)*sqrt(252)) - y NO reimplementa el
# resto del andamiaje TFT (encoder/decoder LSTM completo, quantile forecasts,
# static enrichment de 4 contextos separados) porque esa parte no es
# especifica del Momentum Transformer, ya esta cubierta por Fase 3.
#
# DECISION DE DISEÑO 2: volatility targeting. El paper ancla escala la
# posicion cruda [-1,1] por VOL_TARGET/vol_anualizada (VOL_TARGET=15%, ver
# classical_strategies.py:18 del repo ancla). Este proyecto ya tiene su
# propio mecanismo de sizing por volatilidad (vol_realizada, usado en
# 22/23/56) - se reusa ESE en vez de introducir una constante nueva, y se
# clipea a +-LIMITE_APALANCAMIENTO=1 (mismo limite que el resto del proyecto,
# para no repetir el problema de apalancamiento sin acotar de 9.14).
#
# DECISION DE DISEÑO 3: entrenamiento pooled con ID de moneda. Igual que
# Fase 3, se entrena sobre el panel de 14 monedas (panel_fx_diario_extendido.csv)
# para poder comparar los 3 mecanismos de pooling en el mismo eje: OLS pooled
# ingenuo (9.14, +67.9%), TFT forecast+Kelly pooled (Fase 3), y Momentum
# Transformer pooled con Sharpe-loss directa (esta fase) - los 3 contra el
# mismo baseline solo-CLP (+83.0%, 9.15). El ID de moneda entra como
# embedding aprendido (nn.Embedding), analogo al embedding de unique_id que
# TFT usa internamente - no un one-hot a mano.

import importlib
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")
kelly_diario_mod = importlib.import_module("22_kelly_diario_cobre")

BASES_DIR = "../datos/bases"
RESULTADOS_DIR = "../datos/resultados"
PANEL_EXTENDIDO = f"{BASES_DIR}/panel_fx_diario_extendido.csv"

FEATURES = ["macd_rel", "rsi_norm", "copper_ret_1d", "copper_mom_5d", "vol_realizada"]
N_FEATURES = len(FEATURES)
LOOKBACK = 20  # mismo input_size que TFT en Fase 3, comparable
EMBED_DIM_MONEDA = 8
HIDDEN_SIZE = 32
N_WINDOWS_WF = kelly_diario_mod.N_WINDOWS_WF
N_TEST_POR_VENTANA = kelly_diario_mod.N_TEST_POR_VENTANA
CAPITAL_INICIAL = kelly_diario_mod.CAPITAL_INICIAL
LIMITE_APALANCAMIENTO = kelly_diario_mod.LIMITE_APALANCAMIENTO

BATCH_SIZE = 256
STEPS_POR_VENTANA = 1500  # mismo presupuesto que TFT en Fase 3, comparable
LEARNING_RATE = 1e-3
SEED = 42

DEVICE = torch.device("cpu")


class VariableSelectionNetwork(nn.Module):
    # Port directo de gated_residual_network() + lstm_combine_and_mask() del
    # repo ancla (mom_trans/momentum_transformer.py:128-187 y 468-529): una
    # GRN (Linear->ELU->Linear->GLU gate->residual+LayerNorm) produce pesos
    # softmax sobre las N_FEATURES variables en cada paso de tiempo, y cada
    # variable pasa ademas por su PROPIA GRN antes de la suma ponderada - el
    # mecanismo real de "seleccion de variables" del paper, no una atencion
    # generica.
    def __init__(self, n_features, hidden_size, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.flatten_grn = self._make_grn(n_features * hidden_size, hidden_size, n_features, dropout)
        self.feature_grns = nn.ModuleList([
            self._make_grn(hidden_size, hidden_size, hidden_size, dropout) for _ in range(n_features)
        ])
        self.input_proj = nn.ModuleList([nn.Linear(1, hidden_size) for _ in range(n_features)])

    @staticmethod
    def _make_grn(in_size, hidden_size, out_size, dropout):
        # Sigue la formula exacta de gated_residual_network() del ancla:
        # fc1 (in->hidden) -> ELU -> fc2 (hidden->hidden) -> GLU (dos
        # proyecciones hidden->out, una lineal y una sigmoid, multiplicadas)
        # -> residual (skip in->out) + LayerNorm. fc2 se queda en hidden_size
        # (no en out_size) porque el GLU es el que proyecta a out_size.
        return nn.ModuleDict({
            "skip": nn.Linear(in_size, out_size) if in_size != out_size else nn.Identity(),
            "fc1": nn.Linear(in_size, hidden_size),
            "fc2": nn.Linear(hidden_size, hidden_size),
            "glu_linear": nn.Linear(hidden_size, out_size),
            "glu_gate": nn.Linear(hidden_size, out_size),
            "dropout": nn.Dropout(dropout),
            "norm": nn.LayerNorm(out_size),
        })

    def _grn_forward(self, grn, x):
        skip = grn["skip"](x)
        hidden = torch.nn.functional.elu(grn["fc1"](x))
        hidden = grn["fc2"](hidden)
        activado = grn["glu_linear"](hidden)
        gate = torch.sigmoid(grn["glu_gate"](hidden))
        gated = grn["dropout"](activado * gate)
        return grn["norm"](skip + gated)

    def forward(self, x):
        # x: (batch, time, n_features)
        projected = [self.input_proj[i](x[..., i : i + 1]) for i in range(self.n_features)]  # cada uno (b,t,h)
        flat = torch.cat(projected, dim=-1)  # (b, t, n_features*h)
        weights = torch.softmax(self._grn_forward(self.flatten_grn, flat), dim=-1)  # (b, t, n_features)

        transformed = torch.stack(
            [self._grn_forward(self.feature_grns[i], projected[i]) for i in range(self.n_features)], dim=-1
        )  # (b, t, h, n_features)
        selected = (transformed * weights.unsqueeze(-2)).sum(-1)  # (b, t, h)
        return selected, weights


class CausalSelfAttention(nn.Module):
    # Version simplificada (1 cabeza, sin la reconstruccion "interpretable"
    # completa de InterpretableMultiHeadAttention) del bloque de atencion del
    # repo ancla (momentum_transformer.py:202-315) - mismo mecanismo scaled
    # dot-product con mascara causal (un dia no puede atender a dias futuros).
    def __init__(self, hidden_size):
        super().__init__()
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = hidden_size ** 0.5

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return self.out(torch.bmm(attn, v))


class MomentumTransformerLite(nn.Module):
    def __init__(self, n_monedas, n_features=N_FEATURES, hidden_size=HIDDEN_SIZE, embed_dim=EMBED_DIM_MONEDA):
        super().__init__()
        self.moneda_embed = nn.Embedding(n_monedas, embed_dim)
        self.embed_proj = nn.Linear(embed_dim, hidden_size)
        self.vsn = VariableSelectionNetwork(n_features, hidden_size)
        self.attn = CausalSelfAttention(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output_head = nn.Linear(hidden_size, 1)

    def forward(self, x, moneda_id):
        # x: (batch, time, n_features), moneda_id: (batch,)
        selected, vsn_weights = self.vsn(x)
        contexto_moneda = self.embed_proj(self.moneda_embed(moneda_id)).unsqueeze(1)  # (b,1,h)
        selected = selected + contexto_moneda  # enriquecimiento estatico (version simplificada del original)

        atendido = self.attn(selected)
        combinado = self.norm(selected + atendido)  # residual + norm, como add_and_norm() del ancla

        lstm_out, _ = self.lstm(combinado)
        ultimo = lstm_out[:, -1, :]  # estado del ultimo dia de la ventana -> decide la posicion de mañana
        posicion = torch.tanh(self.output_head(ultimo)).squeeze(-1)
        return posicion, vsn_weights


def sharpe_loss(posiciones, retornos_objetivo, eps=1e-9):
    # Formula identica a SharpeLoss del repo ancla (deep_momentum_network.py:37-48):
    # -Sharpe anualizado de los retornos capturados (posicion x retorno real).
    capturado = posiciones * retornos_objetivo
    media, var = capturado.mean(), capturado.var(unbiased=False)
    return -(media / torch.sqrt(var + eps)) * (252 ** 0.5)


def construir_dataset_ventanas(panel_ventana, monedas_a_id):
    # Para cada moneda y cada dia t con suficiente historia, arma una
    # ventana de LOOKBACK dias de features (hasta t) y el retorno REAL de
    # t+1 como objetivo (la SharpeLoss decide la posicion con info hasta t,
    # la aplica al retorno de t+1 - identico timing a Fase 3/Kelly).
    X, moneda_ids, y_obj, fechas, monedas = [], [], [], [], []
    for par, grupo in panel_ventana.groupby("par"):
        grupo = grupo.sort_values("ds").reset_index(drop=True)
        feats = grupo[FEATURES].to_numpy(dtype=np.float32)
        vol = grupo["vol_realizada"].to_numpy(dtype=np.float32)
        retorno_futuro = grupo["retorno_1d"].shift(-1).to_numpy(dtype=np.float32)
        for t in range(LOOKBACK - 1, len(grupo) - 1):
            if np.isnan(retorno_futuro[t]) or np.isnan(vol[t]) or vol[t] <= 0:
                continue
            X.append(feats[t - LOOKBACK + 1 : t + 1])
            moneda_ids.append(monedas_a_id[par])
            y_obj.append(retorno_futuro[t])
            fechas.append(grupo["ds"].iloc[t])
            monedas.append(par)
    return (np.stack(X), np.array(moneda_ids), np.array(y_obj, dtype=np.float32),
            np.array(fechas), np.array(monedas))


def normalizar_features(X_train, X_eval):
    media, std = X_train.mean(axis=(0, 1), keepdims=True), X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (X_train - media) / std, (X_eval - media) / std


def entrenar_ventana(panel_train, panel_train_y_test, monedas_a_id, n_monedas):
    torch.manual_seed(SEED)
    X_train, ids_train, y_train, _, _ = construir_dataset_ventanas(panel_train, monedas_a_id)
    X_all, ids_all, y_all, fechas_all, monedas_all = construir_dataset_ventanas(panel_train_y_test, monedas_a_id)
    X_train_norm, X_all_norm = normalizar_features(X_train, X_all)

    modelo = MomentumTransformerLite(n_monedas).to(DEVICE)
    opt = torch.optim.Adam(modelo.parameters(), lr=LEARNING_RATE)

    X_t = torch.tensor(X_train_norm, device=DEVICE)
    ids_t = torch.tensor(ids_train, dtype=torch.long, device=DEVICE)
    y_t = torch.tensor(y_train, device=DEVICE)
    n = len(X_t)

    modelo.train()
    for step in range(STEPS_POR_VENTANA):
        idx = torch.randint(0, n, (min(BATCH_SIZE, n),))
        pos, _ = modelo(X_t[idx], ids_t[idx])
        loss = sharpe_loss(pos, y_t[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    step {step+1}/{STEPS_POR_VENTANA}, loss (Sharpe negativo) = {loss.item():.4f}")

    modelo.eval()
    with torch.no_grad():
        pos_all, _ = modelo(torch.tensor(X_all_norm, device=DEVICE), torch.tensor(ids_all, dtype=torch.long, device=DEVICE))
    return pd.DataFrame({"ds": fechas_all, "par": monedas_all, "posicion_cruda": pos_all.numpy(), "retorno_1d_futuro": y_all})


if __name__ == "__main__":
    panel = pd.read_csv(PANEL_EXTENDIDO, parse_dates=["ds"])
    monedas = sorted(panel["par"].unique())
    monedas_a_id = {m: i for i, m in enumerate(monedas)}
    clp = panel[panel["par"] == "CLP=X"].sort_values("ds").reset_index(drop=True)

    print(f"Panel: {len(panel)} filas, {len(monedas)} monedas. Momentum Transformer Lite: "
          f"LOOKBACK={LOOKBACK}, hidden={HIDDEN_SIZE}, steps/ventana={STEPS_POR_VENTANA}, batch={BATCH_SIZE}")

    capital = CAPITAL_INICIAL
    partes_resultado = []
    t0_total = time.time()

    for i, (clp_train, clp_test) in enumerate(wf_mod.ventanas_walkforward(clp, N_WINDOWS_WF, N_TEST_POR_VENTANA)):
        fecha_inicio_test = clp_test["ds"].iloc[0]
        fecha_fin_test = clp_test["ds"].iloc[-1]
        panel_train = panel[panel["ds"] < fecha_inicio_test]
        panel_train_y_test = panel[panel["ds"] <= fecha_fin_test]

        print(f"\n--- Ventana {i+1}/{N_WINDOWS_WF}: test {fecha_inicio_test.date()} a {fecha_fin_test.date()} ---")
        t0 = time.time()
        salida = entrenar_ventana(panel_train, panel_train_y_test, monedas_a_id, len(monedas))
        print(f"  Entrenado y evaluado en {time.time()-t0:.0f}s")

        salida_clp = salida[(salida["par"] == "CLP=X") & (salida["ds"] >= fecha_inicio_test) & (salida["ds"] <= fecha_fin_test)]
        df_test = clp_test.merge(salida_clp[["ds", "posicion_cruda"]], on="ds", how="inner")

        # Volatility targeting con vol_realizada del proyecto (decision de diseño 2, ver cabecera) - misma
        # formula que el resto del proyecto (f = escala/sigma), clip a +-1 (mismo limite que 22/23/56).
        vol_anualizada = df_test["vol_realizada"].to_numpy() * np.sqrt(252)
        target_vol = 0.15  # mismo VOL_TARGET que el paper ancla (classical_strategies.py:18)
        posiciones = np.clip(df_test["posicion_cruda"].to_numpy() * target_vol / np.maximum(vol_anualizada, 1e-6),
                              -LIMITE_APALANCAMIENTO, LIMITE_APALANCAMIENTO)

        r = kelly_diario_mod.simular(df_test, posiciones, "Momentum Transformer Lite (panel 14 monedas, Sharpe-loss directa)", capital)
        capital = r["capital"].iloc[-1]
        partes_resultado.append(r)
        print(f"  Capital al cierre de la ventana {i+1}: ${capital:.2f}")

    print(f"\nTiempo total: {(time.time()-t0_total)/60:.1f} min")

    resultado = pd.concat(partes_resultado, ignore_index=True)
    resultado.to_csv(f"{RESULTADOS_DIR}/fase4_momentum_transformer_operaciones.csv", index=False)

    metrica = kelly_diario_mod.calcular_metricas(resultado, "Momentum Transformer Lite (panel 14 monedas, Sharpe-loss directa)")
    referencia = pd.read_csv(f"{RESULTADOS_DIR}/panel_fx_pooled_vs_single_metricas.csv")
    tft_panel = pd.read_csv(f"{RESULTADOS_DIR}/fase3_tft_vs_baselines_metricas.csv")
    tft_fila = tft_panel[tft_panel["estrategia"].str.contains("TFT panel")]

    tabla_final = pd.concat([pd.DataFrame([metrica]), tft_fila, referencia], ignore_index=True).drop_duplicates(
        subset="estrategia").sort_values("sharpe_anualizado", ascending=False).reset_index(drop=True)
    tabla_final.to_csv(f"{RESULTADOS_DIR}/fase4_momentum_transformer_vs_baselines_metricas.csv", index=False)

    print("\n=== Momentum Transformer Lite vs. TFT panel (Fase 3) vs. baselines de 9.14 ===")
    print(tabla_final.to_string(index=False))

    baseline_solo_clp = referencia.loc[referencia["estrategia"].str.contains("solo CLP"), "retorno_total_%"].iloc[0]
    print(f"\nCriterio de exito (Issue #13): retorno Momentum Transformer ({metrica['retorno_total_%']:.1f}%) "
          f">= solo-CLP ({baseline_solo_clp:.1f}%)? "
          f"{'SI' if metrica['retorno_total_%'] >= baseline_solo_clp else 'NO'}")
