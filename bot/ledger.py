"""Ledger SQLite del bot (bot/data/bot.db por defecto).

Tablas:
- senales         una fila por (estrategia, instrumento, vela). UNIQUE -> idempotencia.
- trades          una fila por posición: entrada, salida, spread pagado, financiamiento, P&L.
- equity_diaria   una foto de equity por día de trading (fecha = cierre 17:00 NY).
- ejecuciones_log una fila por corrida de run_diario.py (ok / sin acción / error).
- meta            modo y capital inicial con que se creó la base (no se mezclan modos).

Todas las horas se guardan en ISO 8601 UTC.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from modelos import Fill, Senal

ESQUEMA = """
CREATE TABLE IF NOT EXISTS meta (
    clave TEXT PRIMARY KEY,
    valor TEXT
);
CREATE TABLE IF NOT EXISTS senales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    estrategia TEXT NOT NULL,
    instrumento TEXT NOT NULL,
    vela_tiempo TEXT NOT NULL,
    vela_cierre TEXT,
    fecha TEXT,
    direccion INTEGER NOT NULL,
    retorno_senal REAL,
    motivo TEXT,
    fuente TEXT,
    detalle_json TEXT,
    estado TEXT NOT NULL DEFAULT 'pendiente',
    creado_utc TEXT NOT NULL,
    actualizado_utc TEXT,
    UNIQUE (estrategia, instrumento, vela_tiempo)
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    estrategia TEXT NOT NULL,
    instrumento TEXT NOT NULL,
    modo TEXT NOT NULL,
    fuente TEXT,
    direccion INTEGER NOT NULL,
    unidades INTEGER NOT NULL,
    senal_id_entrada INTEGER,
    senal_id_salida INTEGER,
    equity_entrada REAL,
    entrada_tiempo TEXT NOT NULL,
    entrada_precio REAL NOT NULL,
    entrada_bid REAL,
    entrada_ask REAL,
    entrada_costo_spread REAL,
    entrada_broker_id TEXT,
    salida_tiempo TEXT,
    salida_precio REAL,
    salida_bid REAL,
    salida_ask REAL,
    salida_costo_spread REAL,
    salida_broker_id TEXT,
    financiamiento REAL DEFAULT 0,
    pnl REAL,
    retorno_pct REAL,
    estado TEXT NOT NULL DEFAULT 'abierta',
    notas TEXT
);
CREATE TABLE IF NOT EXISTS equity_diaria (
    fecha TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    balance REAL,
    pnl_no_realizado REAL,
    unidades_posicion INTEGER,
    fuente TEXT,
    actualizado_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ejecuciones_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    inicio_utc TEXT NOT NULL,
    fin_utc TEXT,
    modo TEXT,
    estado TEXT,
    mensaje TEXT,
    vela_tiempo TEXT
);
CREATE INDEX IF NOT EXISTS ix_trades_estado ON trades (instrumento, estado);
"""


class ErrorLedger(RuntimeError):
    pass


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, ruta: Path | str, modo: str | None = None,
                 capital_inicial: float | None = None):
        self.ruta = Path(ruta)
        if str(ruta) != ":memory:":
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(ruta), timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        with self.conn:
            self.conn.executescript(ESQUEMA)
        if modo is not None:
            self.fijar_meta(modo, capital_inicial)

    def cerrar(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ meta
    def meta(self, clave: str, defecto: str | None = None) -> str | None:
        fila = self.conn.execute("SELECT valor FROM meta WHERE clave = ?", (clave,)).fetchone()
        return fila["valor"] if fila else defecto

    def fijar_meta(self, modo: str, capital_inicial: float | None) -> None:
        modo_db = self.meta("modo")
        if modo_db is not None and modo_db != modo:
            raise ErrorLedger(
                f"La base {self.ruta} fue creada en modo '{modo_db}' y ahora se pide '{modo}'. "
                "Usa otro DB_PATH para no mezclar resultados de modos distintos."
            )
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO meta VALUES ('modo', ?)", (modo,))
            self.conn.execute("INSERT OR IGNORE INTO meta VALUES ('creado_utc', ?)", (_ahora(),))
            if capital_inicial is not None:
                self.conn.execute("INSERT OR IGNORE INTO meta VALUES ('capital_inicial', ?)",
                                  (str(float(capital_inicial)),))

    @property
    def capital_inicial(self) -> float | None:
        v = self.meta("capital_inicial")
        return float(v) if v is not None else None

    # ----------------------------------------------------------- ejecuciones
    def iniciar_ejecucion(self, modo: str) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO ejecuciones_log (inicio_utc, modo, estado) VALUES (?, ?, 'corriendo')",
                (_ahora(), modo))
        return int(cur.lastrowid)

    def finalizar_ejecucion(self, id_: int, estado: str, mensaje: str = "",
                            vela_tiempo: datetime | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE ejecuciones_log SET fin_utc = ?, estado = ?, mensaje = ?, vela_tiempo = ? "
                "WHERE id = ?", (_ahora(), estado, mensaje[:2000], _iso(vela_tiempo), id_))

    # --------------------------------------------------------------- señales
    def obtener_senal(self, estrategia: str, instrumento: str,
                      vela_tiempo: datetime) -> dict | None:
        fila = self.conn.execute(
            "SELECT * FROM senales WHERE estrategia = ? AND instrumento = ? AND vela_tiempo = ?",
            (estrategia, instrumento, _iso(vela_tiempo))).fetchone()
        return dict(fila) if fila else None

    def registrar_senal(self, s: Senal, fecha: str) -> int:
        """Inserta la señal si no existe (idempotente). Devuelve su id."""
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO senales (estrategia, instrumento, vela_tiempo, vela_cierre, "
                "fecha, direccion, retorno_senal, motivo, fuente, detalle_json, estado, creado_utc) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pendiente', ?)",
                (s.estrategia, s.instrumento, _iso(s.vela_tiempo), _iso(s.vela_cierre), fecha,
                 s.direccion, s.retorno_senal, s.motivo, s.fuente,
                 json.dumps(s.detalle, ensure_ascii=False, default=str), _ahora()))
        return int(self.obtener_senal(s.estrategia, s.instrumento, s.vela_tiempo)["id"])

    def marcar_senal(self, id_: int, estado: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE senales SET estado = ?, actualizado_utc = ? WHERE id = ?",
                              (estado, _ahora(), id_))

    def reemplazar_pendientes_anteriores(self, estrategia: str, instrumento: str,
                                         id_actual: int) -> int:
        """Señales viejas que nunca se ejecutaron (p.ej. PC apagado) quedan 'reemplazada'."""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE senales SET estado = 'reemplazada', actualizado_utc = ? "
                "WHERE estrategia = ? AND instrumento = ? AND estado = 'pendiente' AND id <> ?",
                (_ahora(), estrategia, instrumento, id_actual))
        return cur.rowcount

    # ---------------------------------------------------------------- trades
    def trade_abierto(self, instrumento: str) -> dict | None:
        fila = self.conn.execute(
            "SELECT * FROM trades WHERE instrumento = ? AND estado = 'abierta' "
            "ORDER BY id DESC LIMIT 1", (instrumento,)).fetchone()
        return dict(fila) if fila else None

    def trades_abiertos(self) -> list[dict]:
        return [dict(f) for f in self.conn.execute(
            "SELECT * FROM trades WHERE estado = 'abierta' ORDER BY id")]

    def abrir_trade(self, estrategia: str, modo: str, fill: Fill, senal_id: int | None,
                    equity_entrada: float) -> int:
        if self.trade_abierto(fill.instrumento) is not None:
            raise ErrorLedger(f"Ya hay un trade abierto en {fill.instrumento}; no se duplica.")
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO trades (estrategia, instrumento, modo, fuente, direccion, unidades, "
                "senal_id_entrada, equity_entrada, entrada_tiempo, entrada_precio, entrada_bid, "
                "entrada_ask, entrada_costo_spread, entrada_broker_id, estado) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'abierta')",
                (estrategia, fill.instrumento, modo, fill.fuente,
                 1 if fill.unidades > 0 else -1, fill.unidades, senal_id, equity_entrada,
                 _iso(fill.tiempo), fill.precio, fill.bid, fill.ask, fill.costo_spread,
                 fill.broker_id))
        return int(cur.lastrowid)

    def cerrar_trade(self, fill: Fill, senal_id: int | None, estrategia: str = "",
                     modo: str = "") -> int:
        """Registra el cierre sobre el trade abierto. Si no hay (posición abierta a mano
        en la demo), crea una fila 'externa' para no perder el P&L."""
        abierto = self.trade_abierto(fill.instrumento)
        with self.conn:
            if abierto is None:
                cur = self.conn.execute(
                    "INSERT INTO trades (estrategia, instrumento, modo, fuente, direccion, unidades, "
                    "entrada_tiempo, entrada_precio, salida_tiempo, salida_precio, salida_bid, "
                    "salida_ask, salida_costo_spread, salida_broker_id, senal_id_salida, "
                    "financiamiento, pnl, estado, notas) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'cerrada', ?)",
                    (estrategia, fill.instrumento, modo, fill.fuente,
                     -1 if fill.unidades > 0 else 1, -fill.unidades, _iso(fill.tiempo), 0.0,
                     _iso(fill.tiempo), fill.precio, fill.bid, fill.ask, fill.costo_spread,
                     fill.broker_id, senal_id, fill.financiamiento, fill.pnl,
                     "posición externa (no abierta por el bot)"))
                return int(cur.lastrowid)
            eq = abierto["equity_entrada"] or 0.0
            ret = (fill.pnl + fill.financiamiento) / eq if eq else None
            self.conn.execute(
                "UPDATE trades SET salida_tiempo = ?, salida_precio = ?, salida_bid = ?, "
                "salida_ask = ?, salida_costo_spread = ?, salida_broker_id = ?, "
                "senal_id_salida = ?, financiamiento = ?, pnl = ?, retorno_pct = ?, "
                "estado = 'cerrada' WHERE id = ?",
                (_iso(fill.tiempo), fill.precio, fill.bid, fill.ask, fill.costo_spread,
                 fill.broker_id, senal_id, fill.financiamiento, fill.pnl, ret, abierto["id"]))
            return int(abierto["id"])

    def pnl_realizado_total(self) -> float:
        fila = self.conn.execute(
            "SELECT COALESCE(SUM(COALESCE(pnl, 0) + COALESCE(financiamiento, 0)), 0) AS t "
            "FROM trades WHERE estado = 'cerrada'").fetchone()
        return float(fila["t"])

    # ---------------------------------------------------------------- equity
    def registrar_equity(self, fecha: str, equity: float, balance: float | None = None,
                         pnl_no_realizado: float | None = None, unidades: int = 0,
                         fuente: str = "") -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO equity_diaria (fecha, equity, balance, pnl_no_realizado, "
                "unidades_posicion, fuente, actualizado_utc) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(fecha) DO UPDATE SET equity = excluded.equity, "
                "balance = excluded.balance, pnl_no_realizado = excluded.pnl_no_realizado, "
                "unidades_posicion = excluded.unidades_posicion, fuente = excluded.fuente, "
                "actualizado_utc = excluded.actualizado_utc",
                (fecha, equity, balance, pnl_no_realizado, unidades, fuente, _ahora()))

    # ------------------------------------------------------- lectura (pandas)
    def df(self, tabla: str):
        import pandas as pd
        if tabla not in ("trades", "equity_diaria", "senales", "ejecuciones_log"):
            raise ValueError(tabla)
        orden = "fecha" if tabla == "equity_diaria" else "id"
        return pd.read_sql_query(f"SELECT * FROM {tabla} ORDER BY {orden}", self.conn)
