"""Punto de entrada del bot. Pensado para correr cada hora (o una vez al día).

    python run_diario.py              # ciclo normal (idempotente: si la vela ya se
                                      # procesó, no hace nada y sale con código 0)
    python run_diario.py --dry-run    # calcula la señal y la muestra, sin registrar ni operar
    python run_diario.py --verificar  # chequea config, conexión OANDA e instrumentos

Códigos de salida: 0 = ok (incluye "nada que hacer"), 1 = error, 2 = config insegura/inválida.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from config import Config, ErrorConfig, ErrorSeguridad, cargar_config  # noqa: E402
from registro import configurar_logging  # noqa: E402

log = logging.getLogger("run_diario")


class Candado:
    """Evita dos corridas simultáneas (lock file). Un candado de más de 2 h se considera huérfano."""

    def __init__(self, ruta: Path, max_edad_s: int = 7200):
        self.ruta = ruta
        self.max_edad_s = max_edad_s
        self._fd = None

    def __enter__(self):
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        if self.ruta.exists() and time.time() - self.ruta.stat().st_mtime > self.max_edad_s:
            log.warning("Candado huérfano encontrado (%s); se elimina", self.ruta)
            self.ruta.unlink(missing_ok=True)
        try:
            self._fd = os.open(str(self.ruta), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self._fd, str(os.getpid()).encode())
        except FileExistsError:
            raise RuntimeError(f"Otra ejecución está en curso (candado {self.ruta})") from None
        return self

    def __exit__(self, *exc):
        if self._fd is not None:
            os.close(self._fd)
            self.ruta.unlink(missing_ok=True)
        return False


def verificar(cfg: Config) -> int:
    log.info("Config: %r", cfg)
    log.info("Fuente de datos: %s", cfg.fuente_datos)
    if not cfg.tiene_token:
        log.warning("Sin OANDA_TOKEN: solo se puede usar MODO=simulado con fallback de Yahoo.")
        return 0
    from broker.oanda import OandaClient
    cli = OandaClient(cfg.oanda_token, cfg.oanda_account_id, cfg.oanda_host)
    velas = cli.velas(cfg.instrumento_senal, n=3)
    log.info("OK velas %s: %d (última %s)", cfg.instrumento_senal, len(velas),
             velas[-1].tiempo.isoformat() if velas else "-")
    if cfg.oanda_account_id:
        cuenta = cli.resumen_cuenta()
        log.info("OK cuenta demo: moneda=%s NAV=%s balance=%s", cuenta.get("currency"),
                 cuenta.get("NAV"), cuenta.get("balance"))
        if cuenta.get("currency") != "USD":
            log.warning("La cuenta no está en USD: el P&L se reporta en %s.", cuenta.get("currency"))
        disp = {i["name"] for i in cli.instrumentos([cfg.instrumento, cfg.instrumento_senal])}
        for ins in (cfg.instrumento, cfg.instrumento_senal):
            log.info("Instrumento %s: %s", ins, "disponible" if ins in disp else "NO DISPONIBLE")
        if not {cfg.instrumento, cfg.instrumento_senal} <= disp:
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bot diario de paper trading (DEMO)")
    ap.add_argument("--dry-run", action="store_true", help="solo calcula y muestra la señal")
    ap.add_argument("--verificar", action="store_true", help="chequea config y conexión")
    ap.add_argument("--env", default=None, help="ruta alternativa al archivo .env")
    args = ap.parse_args(argv)

    try:
        cfg = cargar_config(args.env)
    except (ErrorSeguridad, ErrorConfig) as e:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
        log.error("CONFIG RECHAZADA: %s", e)
        return 2

    archivo_log = configurar_logging(cfg.log_dir, secretos=[cfg.oanda_token])
    log.info("=== run_diario | modo=%s | estrategia=%s | log=%s ===",
             cfg.modo, cfg.estrategia, archivo_log.name)

    if args.verificar:
        try:
            return verificar(cfg)
        except Exception as e:  # noqa: BLE001
            log.exception("Verificación fallida: %s", e)
            return 1

    from broker import crear_broker
    from estrategias import crear_estrategia
    from ledger import Ledger
    from motor import ejecutar_ciclo

    ledger = None
    id_ejec = None
    try:
        with Candado(cfg.db_path.with_suffix(".lock")):
            ledger = Ledger(cfg.db_path, modo=cfg.modo)  # falla si la base es de otro modo
            id_ejec = ledger.iniciar_ejecucion(cfg.modo)
            broker = crear_broker(cfg, ledger)
            if ledger.capital_inicial is None:
                # simulado: CAPITAL_INICIAL de la config; demo OANDA: NAV real al primer día.
                capital = cfg.capital_inicial if cfg.modo == "simulado" else broker.equity()
                ledger.fijar_meta(cfg.modo, capital)
                log.info("Capital inicial registrado: %.2f", capital)
            estrategia = crear_estrategia(cfg)
            res = ejecutar_ciclo(cfg, broker, estrategia, ledger, dry_run=args.dry_run)
            ledger.finalizar_ejecucion(id_ejec, res.estado, res.mensaje,
                                       res.senal.vela_tiempo if res.senal else None)
            log.info("Resultado: %s | %s", res.estado, res.mensaje)
            return 0
    except Exception as e:  # noqa: BLE001
        log.exception("ERROR en la ejecución: %s", e)
        if ledger is not None and id_ejec is not None:
            try:
                ledger.finalizar_ejecucion(id_ejec, "error", f"{type(e).__name__}: {e}")
            except Exception:  # noqa: BLE001
                pass
        return 1
    finally:
        if ledger is not None:
            ledger.cerrar()


if __name__ == "__main__":
    sys.exit(main())
