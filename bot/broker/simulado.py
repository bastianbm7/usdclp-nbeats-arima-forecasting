"""Broker simulado: misma interfaz que OANDA, pero la cuenta vive en el ledger SQLite.

- Precios: bid/ask reales de OANDA si hay token (sin mandar órdenes), o el fallback
  de Yahoo con spread sintético (etiquetado `fallback_yahoo`).
- Fills: compra al ASK, vende al BID -> el spread se paga de verdad en el P&L.
- Balance = capital inicial + P&L realizado de los trades cerrados del ledger.
- Equity = balance + P&L no realizado valorizado a precio de cierre (BID para largos,
  ASK para cortos), igual que el NAV de OANDA.
- Financiamiento (swap overnight): NO se modela en simulado (queda en 0). En
  oanda_practice se registra el valor real que informa OANDA.
"""
from __future__ import annotations

from broker.base import Broker, Posicion
from modelos import Cotizacion, Fill, Vela


class SimuladoBroker(Broker):
    nombre = "simulado"

    def __init__(self, fuente, ledger, capital_inicial: float):
        """`fuente`: objeto con velas_diarias(instr, n) y precio(instr) (OandaClient o FuenteYahoo)."""
        self.fuente = fuente
        self.ledger = ledger
        self.capital_inicial = float(capital_inicial)

    @property
    def nombre_fuente(self) -> str:
        return getattr(self.fuente, "nombre", "oanda")

    # --- datos
    def velas_diarias(self, instrumento: str, n: int = 10) -> list[Vela]:
        return self.fuente.velas_diarias(instrumento, n)

    def precio(self, instrumento: str) -> Cotizacion:
        return self.fuente.precio(instrumento)

    # --- cuenta
    def balance(self) -> float:
        return self.capital_inicial + self.ledger.pnl_realizado_total()

    def posicion_abierta(self, instrumento: str) -> Posicion | None:
        t = self.ledger.trade_abierto(instrumento)
        if t is None:
            return None
        return Posicion(instrumento, int(t["unidades"]), float(t["entrada_precio"]))

    def equity(self, cotizacion: Cotizacion | None = None) -> float:
        eq = self.balance()
        for t in self.ledger.trades_abiertos():
            cot = cotizacion if (cotizacion and cotizacion.instrumento == t["instrumento"]) \
                else self.precio(t["instrumento"])
            eq += pnl_no_realizado(int(t["unidades"]), float(t["entrada_precio"]), cot)
        return eq

    # --- órdenes
    def abrir_mercado(self, instrumento: str, unidades: int, etiqueta: str = "") -> Fill:
        if unidades == 0:
            raise ValueError("unidades = 0")
        cot = self.precio(instrumento)
        return fill_apertura(instrumento, int(unidades), cot)

    def cerrar_posicion(self, instrumento: str, etiqueta: str = "") -> Fill | None:
        pos = self.posicion_abierta(instrumento)
        if pos is None:
            return None
        cot = self.precio(instrumento)
        return fill_cierre(instrumento, pos.unidades, pos.precio_promedio, cot)


# ----------------------------------------------------------------- matemática pura
def precio_ejecucion(unidades: int, cot: Cotizacion) -> float:
    """Compra (unidades > 0) al ASK; venta (unidades < 0) al BID."""
    return cot.ask if unidades > 0 else cot.bid


def costo_medio_spread(unidades: int, cot: Cotizacion) -> float:
    """Costo vs. mid de ejecutar `unidades` (siempre >= 0). Moneda cotizada (USD en XXX_USD)."""
    return abs(unidades) * (cot.ask - cot.bid) / 2.0


def pnl_trade(unidades: int, precio_entrada: float, precio_salida: float) -> float:
    """P&L en moneda cotizada: unidades con signo x (salida - entrada)."""
    return unidades * (precio_salida - precio_entrada)


def pnl_no_realizado(unidades: int, precio_entrada: float, cot: Cotizacion) -> float:
    """Valorizado al precio al que se cerraría hoy (BID si largo, ASK si corto)."""
    salida = cot.bid if unidades > 0 else cot.ask
    return pnl_trade(unidades, precio_entrada, salida)


def fill_apertura(instrumento: str, unidades: int, cot: Cotizacion) -> Fill:
    return Fill(
        instrumento=instrumento, unidades=unidades, precio=precio_ejecucion(unidades, cot),
        bid=cot.bid, ask=cot.ask, tiempo=cot.tiempo,
        costo_spread=costo_medio_spread(unidades, cot), fuente=cot.fuente,
    )


def fill_cierre(instrumento: str, unidades_abiertas: int, precio_entrada: float,
                cot: Cotizacion) -> Fill:
    unidades_cierre = -unidades_abiertas
    precio = precio_ejecucion(unidades_cierre, cot)
    return Fill(
        instrumento=instrumento, unidades=unidades_cierre, precio=precio,
        bid=cot.bid, ask=cot.ask, tiempo=cot.tiempo,
        costo_spread=costo_medio_spread(unidades_cierre, cot),
        pnl=pnl_trade(unidades_abiertas, precio_entrada, precio), fuente=cot.fuente,
    )
