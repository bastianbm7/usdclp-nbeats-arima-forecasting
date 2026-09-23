"""Registro de estrategias disponibles (se elige con ESTRATEGIA=... en bot/.env)."""
from __future__ import annotations

from estrategias.base import Estrategia
from estrategias.cobre_aud import CobreAud

REGISTRO = {
    "cobre_aud": CobreAud,
}


def crear_estrategia(cfg) -> Estrategia:
    if cfg.estrategia not in REGISTRO:
        raise ValueError(f"Estrategia desconocida: {cfg.estrategia}. Opciones: {list(REGISTRO)}")
    clase = REGISTRO[cfg.estrategia]
    if cfg.estrategia == "cobre_aud":
        return clase(instrumento=cfg.instrumento, instrumento_senal=cfg.instrumento_senal)
    return clase()
