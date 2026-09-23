from datetime import datetime, timezone

import pytest

from broker.simulado import (costo_medio_spread, fill_apertura, fill_cierre, pnl_no_realizado,
                             pnl_trade, precio_ejecucion)
from modelos import Cotizacion
from motor import calcular_unidades

T = datetime(2026, 9, 22, 21, 30, tzinfo=timezone.utc)


def cot(bid, ask):
    return Cotizacion("AUD_USD", bid, ask, T)


def test_precio_ejecucion_bid_ask():
    c = cot(0.65000, 0.65014)
    assert precio_ejecucion(1000, c) == 0.65014   # compra al ASK
    assert precio_ejecucion(-1000, c) == 0.65000  # venta al BID


def test_costo_spread():
    c = cot(0.65000, 0.65014)
    assert costo_medio_spread(10_000, c) == pytest.approx(0.7)
    assert costo_medio_spread(-10_000, c) == pytest.approx(0.7)


def test_roundtrip_sin_movimiento_pierde_exactamente_el_spread():
    c = cot(0.65000, 0.65014)
    ap = fill_apertura("AUD_USD", 10_000, c)
    ci = fill_cierre("AUD_USD", 10_000, ap.precio, c)
    assert ci.pnl == pytest.approx(-10_000 * 0.00014)
    assert ci.pnl == pytest.approx(-(ap.costo_spread + ci.costo_spread))


def test_pnl_largo_y_corto():
    # Largo: entra al ask 0.65010, sale al bid 0.66000
    assert pnl_trade(10_000, 0.65010, 0.66000) == pytest.approx(99.0)
    # Corto: entra al bid 0.65000, sale al ask 0.64010
    assert pnl_trade(-10_000, 0.65000, 0.64010) == pytest.approx(99.0)
    ci = fill_cierre("AUD_USD", -10_000, 0.65000, cot(0.64000, 0.64010))
    assert ci.unidades == 10_000 and ci.precio == 0.64010
    assert ci.pnl == pytest.approx(99.0)


def test_no_realizado_a_precio_de_cierre():
    c = cot(0.66000, 0.66010)
    assert pnl_no_realizado(10_000, 0.65010, c) == pytest.approx(99.0)   # largo al bid
    assert pnl_no_realizado(-10_000, 0.67000, c) == pytest.approx(99.0)  # corto al ask


def test_sizing_fijo_y_tope_apalancamiento():
    assert calcular_unidades(10_000, 0.65, 1.0, 2.0) == 15_384
    assert calcular_unidades(10_000, 0.65, 5.0, 2.0) == 30_769  # tope 2x
    assert calcular_unidades(0, 0.65, 1.0, 2.0) == 0
