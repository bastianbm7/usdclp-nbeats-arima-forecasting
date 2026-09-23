"""Ciclo diario del bot, independiente del broker (se testea con brokers falsos).

Pasos de `ejecutar_ciclo`:
1. Pedir velas diarias y calcular la señal sobre la última vela COMPLETA.
2. Si todavía no pasaron MINUTOS_ESPERA_CIERRE desde el cierre -> 'esperando'.
3. Si esa vela ya fue procesada (señal 'ejecutada' o 'sin_cambio') -> 'ya_procesada'.
   (Idempotencia 1: correr el bot dos veces el mismo día no duplica operaciones.)
4. Si el mercado está cerrado (p.ej. viernes 17:00 NY) la señal queda 'pendiente' y se
   ejecuta en la primera corrida con mercado abierto (domingo 17:00 NY en adelante).
5. Llevar la posición a la objetivo: cerrar la actual (largo al BID / corto al ASK) y
   abrir la nueva (largo al ASK / corto al BID). Si la posición ya es la objetivo, no se
   opera. (Idempotencia 2: la operación es "dejar la posición = objetivo", así que
   repetirla es inofensiva aunque algo falle a mitad de camino.)
6. Guardar la foto de equity del día.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from broker.simulado import precio_ejecucion
from modelos import Senal

log = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")
ESTADOS_FINALES = ("ejecutada", "sin_cambio")


@dataclass
class ResultadoCiclo:
    estado: str
    mensaje: str
    senal: Senal | None = None
    trades_abiertos: list[int] = field(default_factory=list)
    trades_cerrados: list[int] = field(default_factory=list)


def fecha_trading(vela_cierre_utc: datetime) -> str:
    """Fecha del día de trading = fecha en Nueva York del cierre de la vela (17:00 NY)."""
    return vela_cierre_utc.astimezone(NY).date().isoformat()


def calcular_unidades(equity: float, precio: float, fraccion: float,
                      apalancamiento_max: float) -> int:
    """Sizing fijo: notional = min(fracción, apalancamiento_max) x equity (sin Kelly).
    Para pares XXX_USD en cuenta USD: notional USD = unidades x precio."""
    if equity <= 0 or precio <= 0:
        return 0
    f = min(fraccion, apalancamiento_max)
    return int(math.floor(equity * f / precio))


def _foto_equity(ledger, broker, instrumento: str, fecha: str, cot) -> float:
    eq = broker.equity(cot)
    bal = broker.balance()
    pos = broker.posicion_abierta(instrumento)
    ledger.registrar_equity(fecha, eq, bal, eq - bal, pos.unidades if pos else 0, cot.fuente)
    log.info("Equity %s: %.2f (balance %.2f, no realizado %.2f)", fecha, eq, bal, eq - bal)
    return eq


def ejecutar_ciclo(cfg, broker, estrategia, ledger, ahora: datetime | None = None,
                   dry_run: bool = False) -> ResultadoCiclo:
    ahora = ahora or datetime.now(timezone.utc)
    instr = estrategia.instrumento

    velas = {i: broker.velas_diarias(i, estrategia.n_velas) for i in estrategia.instrumentos_datos}
    senal = estrategia.calcular_senal(velas)
    log.info("Señal %s/%s: dir=%+d ret=%s vela=%s motivo=%s fuente=%s",
             estrategia.nombre, estrategia.version, senal.direccion,
             f"{senal.retorno_senal:+.5%}" if senal.retorno_senal is not None else "n/d",
             senal.vela_tiempo.isoformat() if senal.vela_tiempo else "-", senal.motivo,
             senal.fuente)

    if senal.vela_tiempo is None or senal.vela_cierre is None:
        return ResultadoCiclo("sin_datos", senal.motivo or "sin velas completas", senal)

    listo_desde = senal.vela_cierre + timedelta(minutes=cfg.minutos_espera_cierre)
    if ahora < listo_desde:
        return ResultadoCiclo("esperando", f"se opera desde {listo_desde.isoformat()}", senal)

    previa = ledger.obtener_senal(estrategia.nombre, instr, senal.vela_tiempo)
    if previa and previa["estado"] in ESTADOS_FINALES:
        return ResultadoCiclo("ya_procesada",
                              f"vela {previa['vela_tiempo']} ya procesada ({previa['estado']})",
                              senal)
    if dry_run:
        return ResultadoCiclo("dry_run", "dry-run: no se registran señales ni se opera", senal)

    fecha = fecha_trading(senal.vela_cierre)
    senal_id = ledger.registrar_senal(senal, fecha)
    n_viejas = ledger.reemplazar_pendientes_anteriores(estrategia.nombre, instr, senal_id)
    if n_viejas:
        log.warning("%d señal(es) antigua(s) sin ejecutar quedaron 'reemplazada'", n_viejas)

    cot = broker.precio(instr)
    if not cot.tradeable:
        _foto_equity(ledger, broker, instr, fecha, cot)
        return ResultadoCiclo("mercado_cerrado",
                              "mercado cerrado: la señal queda pendiente para la próxima corrida",
                              senal)

    pos = broker.posicion_abierta(instr)
    actual = pos.direccion if pos else 0
    objetivo = senal.direccion
    res = ResultadoCiclo("sin_cambio", "", senal)
    etiqueta = f"{estrategia.nombre}-{estrategia.version} vela {fecha}"

    if actual == objetivo and not (cfg.reabrir_si_misma_senal and objetivo != 0):
        ledger.marcar_senal(senal_id, "sin_cambio")
        res.mensaje = f"posición ya es la objetivo ({objetivo:+d}); no se opera"
    else:
        if actual != 0:
            fill = broker.cerrar_posicion(instr, etiqueta)
            if fill is not None:
                tid = ledger.cerrar_trade(fill, senal_id, estrategia.nombre, cfg.modo)
                res.trades_cerrados.append(tid)
                log.info("CIERRE trade %d: %+d %s @ %.5f (bid %.5f / ask %.5f) spread %.2f "
                         "financ. %.2f P&L %.2f", tid, fill.unidades, instr, fill.precio,
                         fill.bid, fill.ask, fill.costo_spread, fill.financiamiento, fill.pnl)
        if objetivo != 0:
            cot = broker.precio(instr)
            if not cot.tradeable:
                raise RuntimeError("El mercado cerró entre el cierre y la apertura; se reintentará")
            eq = broker.equity(cot)
            unidades = calcular_unidades(eq, precio_ejecucion(objetivo, cot),
                                         cfg.fraccion_notional, cfg.apalancamiento_max)
            if unidades <= 0:
                raise RuntimeError(f"Sizing inválido (equity={eq:.2f}): 0 unidades")
            fill = broker.abrir_mercado(instr, objetivo * unidades, etiqueta)
            tid = ledger.abrir_trade(estrategia.nombre, cfg.modo, fill, senal_id, eq)
            res.trades_abiertos.append(tid)
            log.info("APERTURA trade %d: %+d %s @ %.5f (bid %.5f / ask %.5f) spread %.2f "
                     "equity %.2f", tid, fill.unidades, instr, fill.precio, fill.bid, fill.ask,
                     fill.costo_spread, eq)
        ledger.marcar_senal(senal_id, "ejecutada")
        res.estado = "ejecutada"
        res.mensaje = f"posición {actual:+d} -> {objetivo:+d}"

    _foto_equity(ledger, broker, instr, fecha, broker.precio(instr))
    return res
