"""Cliente mínimo de la API REST v20 de OANDA (solo cuenta de PRÁCTICA) + broker.

- `OandaClient`: wrapper delgado sobre `requests` (velas, precios, cuenta, órdenes).
- `OandaBroker`: implementa la interfaz `Broker` usando la cuenta demo real.

Seguridad:
- El host se valida con `config.validar_host` en el constructor: solo
  `api-fxpractice.oanda.com`. Con el host real lanza `ErrorSeguridad`.
- Los GET se reintentan ante errores de red / 5xx; las órdenes (POST/PUT) NUNCA se
  reintentan automáticamente, para no duplicar una orden si la respuesta se perdió.
- El token solo viaja en el header Authorization; nunca se loguea.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import requests

from broker.base import Broker, Posicion
from config import validar_host
from modelos import Cotizacion, Fill, Vela

log = logging.getLogger(__name__)


class OandaError(RuntimeError):
    pass


def parsear_tiempo(valor: str) -> datetime:
    """Parsea RFC3339 de OANDA (nanosegundos incluidos) a datetime UTC."""
    v = valor.strip().replace("Z", "+00:00")
    if "." in v:
        base, resto = v.split(".", 1)
        # separar fracción de la zona horaria
        idx = next((i for i, ch in enumerate(resto) if ch in "+-"), len(resto))
        frac, tz = resto[:idx], resto[idx:]
        v = f"{base}.{frac[:6].ljust(6, '0')}{tz}"
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class OandaClient:
    nombre = "oanda"

    def __init__(self, token: str, account_id: str = "", host: str = "api-fxpractice.oanda.com",
                 timeout: float = 20.0, session: requests.Session | None = None,
                 reintentos: int = 3):
        self.host = validar_host(host)  # ErrorSeguridad si no es práctica
        if not token:
            raise OandaError("Falta OANDA_TOKEN")
        self.base = f"https://{self.host}"
        self.account_id = account_id
        self.timeout = timeout
        self.reintentos = reintentos
        self._s = session or requests.Session()
        self._s.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339",
        })

    def __repr__(self) -> str:
        return f"OandaClient(host={self.host!r}, account_id={self.account_id!r})"

    # ------------------------------------------------------------------ HTTP
    def _request(self, metodo: str, ruta: str, *, params=None, json=None) -> dict:
        url = self.base + ruta
        intentos = self.reintentos if metodo == "GET" else 1
        ultimo_error: Exception | None = None
        for i in range(intentos):
            try:
                r = self._s.request(metodo, url, params=params, json=json, timeout=self.timeout)
            except requests.RequestException as e:
                ultimo_error = e
                log.warning("Error de red en %s %s (intento %d/%d): %s",
                            metodo, ruta, i + 1, intentos, type(e).__name__)
                if i + 1 < intentos:
                    time.sleep(2 ** i)
                continue
            if r.status_code >= 500 and i + 1 < intentos:
                log.warning("OANDA %s en %s %s, reintentando", r.status_code, metodo, ruta)
                time.sleep(2 ** i)
                continue
            if r.status_code >= 400:
                try:
                    cuerpo = r.json()
                    detalle = cuerpo.get("errorMessage") or cuerpo.get("rejectReason") or str(cuerpo)[:300]
                except ValueError:
                    detalle = r.text[:300]
                raise OandaError(f"OANDA {r.status_code} en {metodo} {ruta}: {detalle}")
            return r.json() if r.content else {}
        raise OandaError(f"Sin respuesta de OANDA en {metodo} {ruta}: {ultimo_error}")

    def _cuenta(self) -> str:
        if not self.account_id:
            raise OandaError("Falta OANDA_ACCOUNT_ID")
        return f"/v3/accounts/{self.account_id}"

    # ----------------------------------------------------------------- datos
    def velas(self, instrumento: str, n: int = 10, granularidad: str = "D") -> list[Vela]:
        """Velas con alineación diaria 17:00 Nueva York y precios MBA (mid/bid/ask)."""
        params = {
            "granularity": granularidad,
            "count": n,
            "price": "MBA",
            "dailyAlignment": 17,
            "alignmentTimezone": "America/New_York",
        }
        data = self._request("GET", f"/v3/instruments/{instrumento}/candles", params=params)
        velas = []
        for c in data.get("candles", []):
            t = parsear_tiempo(c["time"])
            mid = c.get("mid") or {}
            if not mid:
                continue
            velas.append(Vela(
                instrumento=instrumento,
                tiempo=t,
                cierre=t + (timedelta(days=1) if granularidad == "D" else timedelta(0)),
                completa=bool(c.get("complete")),
                o=float(mid["o"]), h=float(mid["h"]), l=float(mid["l"]), c=float(mid["c"]),
                bid_c=float(c["bid"]["c"]) if c.get("bid") else None,
                ask_c=float(c["ask"]["c"]) if c.get("ask") else None,
                volumen=int(c.get("volume", 0)),
                fuente="oanda",
            ))
        return velas

    def velas_diarias(self, instrumento: str, n: int = 10) -> list[Vela]:
        return self.velas(instrumento, n=n, granularidad="D")

    def precio(self, instrumento: str) -> Cotizacion:
        """Bid/ask actual. Usa /pricing si hay cuenta; si no, la última vela S5 BA."""
        if self.account_id:
            data = self._request("GET", f"{self._cuenta()}/pricing",
                                 params={"instruments": instrumento})
            p = data["prices"][0]
            return Cotizacion(
                instrumento=instrumento,
                bid=float(p["bids"][0]["price"]),
                ask=float(p["asks"][0]["price"]),
                tiempo=parsear_tiempo(p["time"]),
                tradeable=bool(p.get("tradeable", True)),
                fuente="oanda",
            )
        data = self._request("GET", f"/v3/instruments/{instrumento}/candles",
                             params={"granularity": "S5", "count": 1, "price": "BA"})
        c = data["candles"][-1]
        t = parsear_tiempo(c["time"])
        # Sin cuenta no hay flag 'tradeable': se asume abierto si la vela es reciente.
        abierto = (datetime.now(timezone.utc) - t) < timedelta(minutes=5)
        return Cotizacion(instrumento, float(c["bid"]["c"]), float(c["ask"]["c"]), t,
                          tradeable=abierto, fuente="oanda")

    # ---------------------------------------------------------------- cuenta
    def resumen_cuenta(self) -> dict:
        return self._request("GET", f"{self._cuenta()}/summary")["account"]

    def instrumentos(self, nombres: list[str] | None = None) -> list[dict]:
        params = {"instruments": ",".join(nombres)} if nombres else None
        return self._request("GET", f"{self._cuenta()}/instruments", params=params)["instruments"]

    def posicion(self, instrumento: str) -> dict:
        try:
            return self._request("GET", f"{self._cuenta()}/positions/{instrumento}")["position"]
        except OandaError as e:
            if "404" in str(e):
                return {"long": {"units": "0"}, "short": {"units": "0"}}
            raise

    def orden_mercado(self, instrumento: str, unidades: int, etiqueta: str = "") -> dict:
        orden = {
            "type": "MARKET",
            "instrument": instrumento,
            "units": str(int(unidades)),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
        if etiqueta:
            orden["clientExtensions"] = {"tag": "paperbot", "comment": etiqueta[:120]}
        return self._request("POST", f"{self._cuenta()}/orders", json={"order": orden})

    def cerrar_posicion(self, instrumento: str, lado: str) -> dict:
        cuerpo = {"longUnits": "ALL"} if lado == "long" else {"shortUnits": "ALL"}
        return self._request("PUT", f"{self._cuenta()}/positions/{instrumento}/close", json=cuerpo)


def _fill_desde_transaccion(tx: dict, instrumento: str) -> Fill:
    full = tx.get("fullPrice") or {}
    bid = float(full["bids"][0]["price"]) if full.get("bids") else float(tx["price"])
    ask = float(full["asks"][0]["price"]) if full.get("asks") else float(tx["price"])
    return Fill(
        instrumento=instrumento,
        unidades=int(float(tx["units"])),
        precio=float(tx["price"]),
        bid=bid,
        ask=ask,
        tiempo=parsear_tiempo(tx["time"]),
        costo_spread=abs(float(tx.get("halfSpreadCost", 0.0))),
        pnl=float(tx.get("pl", 0.0)),
        financiamiento=float(tx.get("financing", 0.0)),
        broker_id=str(tx.get("id", "")),
        fuente="oanda",
    )


class OandaBroker(Broker):
    """Broker contra la cuenta DEMO de OANDA. La posición vive en OANDA."""
    nombre = "oanda_practice"

    def __init__(self, client: OandaClient):
        if not client.account_id:
            raise OandaError("OandaBroker requiere OANDA_ACCOUNT_ID")
        self.client = client

    def velas_diarias(self, instrumento: str, n: int = 10) -> list[Vela]:
        return self.client.velas(instrumento, n=n)

    def precio(self, instrumento: str) -> Cotizacion:
        return self.client.precio(instrumento)

    def equity(self, cotizacion: Cotizacion | None = None) -> float:
        return float(self.client.resumen_cuenta()["NAV"])

    def balance(self) -> float:
        return float(self.client.resumen_cuenta()["balance"])

    def moneda_cuenta(self) -> str:
        return str(self.client.resumen_cuenta().get("currency", "USD"))

    def posicion_abierta(self, instrumento: str) -> Posicion | None:
        p = self.client.posicion(instrumento)
        largo = int(float(p.get("long", {}).get("units", 0)))
        corto = int(float(p.get("short", {}).get("units", 0)))
        neto = largo + corto
        if neto == 0:
            return None
        lado = p["long"] if neto > 0 else p["short"]
        precio_prom = float(lado.get("averagePrice", 0.0) or 0.0)
        return Posicion(instrumento, neto, precio_prom)

    def abrir_mercado(self, instrumento: str, unidades: int, etiqueta: str = "") -> Fill:
        resp = self.client.orden_mercado(instrumento, unidades, etiqueta)
        tx = resp.get("orderFillTransaction")
        if not tx:
            motivo = (resp.get("orderCancelTransaction") or {}).get("reason", str(resp)[:300])
            raise OandaError(f"Orden no ejecutada por OANDA: {motivo}")
        return _fill_desde_transaccion(tx, instrumento)

    def cerrar_posicion(self, instrumento: str, etiqueta: str = "") -> Fill | None:
        pos = self.posicion_abierta(instrumento)
        if pos is None:
            return None
        lado = "long" if pos.unidades > 0 else "short"
        resp = self.client.cerrar_posicion(instrumento, lado)
        tx = resp.get("longOrderFillTransaction") or resp.get("shortOrderFillTransaction")
        if not tx:
            raise OandaError(f"Cierre no ejecutado por OANDA: {str(resp)[:300]}")
        return _fill_desde_transaccion(tx, instrumento)
