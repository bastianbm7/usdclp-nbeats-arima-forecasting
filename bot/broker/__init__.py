"""Brokers: OANDA práctica (órdenes reales en la demo) y simulado (ledger interno)."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def crear_fuente_datos(cfg):
    """Devuelve la fuente de velas/precios según la config ('oanda' o 'yahoo')."""
    if cfg.fuente_datos == "oanda":
        from broker.oanda import OandaClient
        return OandaClient(cfg.oanda_token, cfg.oanda_account_id, cfg.oanda_host)
    from broker.yahoo import FuenteYahoo
    return FuenteYahoo(spread_pips=cfg.spread_fallback_pips)


def crear_broker(cfg, ledger):
    if cfg.modo == "oanda_practice":
        from broker.oanda import OandaBroker, OandaClient
        return OandaBroker(OandaClient(cfg.oanda_token, cfg.oanda_account_id, cfg.oanda_host))
    from broker.simulado import SimuladoBroker
    # El capital inicial manda el que quedó guardado en la base (si cambias CAPITAL_INICIAL
    # en .env con una base ya creada, se ignora para no romper la contabilidad).
    capital = ledger.capital_inicial if ledger.capital_inicial is not None else cfg.capital_inicial
    if ledger.capital_inicial is not None and abs(capital - cfg.capital_inicial) > 1e-9:
        log.warning("CAPITAL_INICIAL=%.2f en .env difiere del de la base (%.2f); se usa el de "
                    "la base.", cfg.capital_inicial, capital)
    return SimuladoBroker(crear_fuente_datos(cfg), ledger, capital)
