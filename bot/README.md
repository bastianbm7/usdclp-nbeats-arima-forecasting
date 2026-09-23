# bot/ — paper trading diario (cobre → AUD/USD) + dashboard

Bot mínimo y robusto que opera **una vez por día de trading** en una cuenta **DEMO** de OANDA
(API REST v20, host de práctica), o en modo **simulado** con un ledger interno si todavía no
existe la cuenta. Incluye un dashboard muy simple (Streamlit) con las operaciones y los
retornos diarios / mensuales / anuales, y un exportador a HTML estático.

> **Descargo honesto.** Esto es un *forward test* en vivo de una señal que, según la
> investigación corregida de este repo, **se espera que tenga edge ~0 después de costos**.
> La señal original "retorno del cobre en t → moneda commodity en t+1" resultó ser un
> **artefacto de timestamps de Yahoo** (cada ticker cierra su barra diaria a una hora
> distinta). Acá se prueba la versión corregida, con ambas velas cerrando exactamente a las
> 17:00 de Nueva York. El objetivo es **medir**, no ganar plata; y dejar infraestructura
> reutilizable para probar otras reglas. Solo cuenta demo. No es asesoría financiera.

## Regla v1 (congelada)

| | |
|---|---|
| Instrumento operado | `AUD_USD` |
| Instrumento de señal | `XCU_USD` (CFD de cobre en OANDA) |
| Velas | diarias, `dailyAlignment=17`, `alignmentTimezone=America/New_York`, `price=MBA` |
| Señal | signo del retorno del cobre en la última vela **completa** (cierre/cierre previo − 1, mid) |
| Posición | largo AUD_USD si el cobre subió, corto si bajó, plano si = 0 o faltan datos / velas desalineadas |
| Ejecución | ~17:20 NY (cierre + `MINUTOS_ESPERA_CIERRE`, por defecto 15 min, para evitar el rollover) |
| Precios | apertura al **ASK** (largo) / **BID** (corto); cierre al BID / ASK → el spread se paga de verdad |
| Sizing | fijo: notional = `FRACCION_NOTIONAL` × equity (defecto 1.0×), tope `APALANCAMIENTO_MAX`; sin Kelly |
| Misma señal que ayer | se mantiene la posición (no se paga spread de nuevo); `REABRIR_SI_MISMA_SENAL=true` para cerrar y reabrir cada día |

Se registra todo: vela usada, hora de la señal, precios de entrada/salida con su bid/ask,
spread pagado, financiamiento (swap real en la demo de OANDA; en simulado no se modela y queda
en 0) y P&L.

**Fines de semana:** el viernes a las 17:00 NY el mercado FX cierra. La señal de esa vela
queda `pendiente` y se ejecuta en la primera corrida con mercado abierto (domingo ~17:20 NY).

## Estructura

```
bot/
├── config.py              # .env + chequeos de seguridad (solo host de práctica)
├── registro.py            # logging a bot/logs/ con el token redactado
├── modelos.py             # Vela, Cotizacion, Fill, Senal
├── ledger.py              # SQLite: senales, trades, equity_diaria, ejecuciones_log
├── motor.py               # ciclo diario independiente del broker (idempotente)
├── metricas.py            # retornos diarios/mensuales/anuales, Sharpe, drawdown
├── run_diario.py          # punto de entrada (exit 0 ok, 1 error, 2 config insegura)
├── broker/
│   ├── base.py            # interfaz Broker
│   ├── oanda.py           # cliente REST v20 (práctica) + OandaBroker
│   ├── simulado.py        # ledger interno, fills a bid/ask
│   └── yahoo.py           # fallback gratis (timestamps ambiguos → 'fallback_yahoo')
├── estrategias/
│   ├── base.py            # interfaz Estrategia
│   └── cobre_aud.py       # regla v1
├── dashboard/
│   ├── app.py             # Streamlit
│   ├── exportar_html.py   # HTML estático de un solo archivo
│   └── datos_dashboard.py
├── scripts/
│   ├── instalar_tarea_windows.ps1
│   └── sembrar_demo.py    # base de DEMOSTRACIÓN con historia sintética
├── tests/                 # pytest, sin red
├── .env.example
└── requirements.txt
```

## 1. Instalación (Windows, PowerShell)

```powershell
cd <repo>\bot
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`bot/.env`, `bot/.venv/`, `bot/data/` y `bot/logs/` están en `.gitignore`: el token y el
ledger nunca van al repo.

## 2. Crear la cuenta demo de OANDA y el token (lo haces tú)

El bot **no** crea cuentas ni maneja contraseñas. Pasos (la interfaz de OANDA puede cambiar):

1. Entra a <https://www.oanda.com> y elige **abrir una cuenta demo / práctica** (fxTrade
   Practice). Para residentes en Chile normalmente corresponde la entidad global de OANDA.
2. Al crearla, elige **USD como moneda de la cuenta** (el P&L del bot se reporta en USD).
3. Ya dentro de la cuenta demo, busca **"Manage API Access" / "API"** (en el hub de la cuenta,
   sección de herramientas o perfil) y **genera un token personal**. Cópialo una sola vez.
4. Copia el **Account ID** de la subcuenta demo (formato `101-001-XXXXXXX-001`).
5. Pega ambos en `bot/.env`:
   ```
   OANDA_TOKEN=...tu token...
   OANDA_ACCOUNT_ID=101-001-XXXXXXX-001
   OANDA_HOST=api-fxpractice.oanda.com
   ```
6. Verifica conexión y que tu cuenta tenga **ambos** instrumentos (los CFD como `XCU_USD`
   no están disponibles en todas las jurisdicciones de OANDA):
   ```powershell
   .\.venv\Scripts\python.exe run_diario.py --verificar
   ```

**Seguridad dura:** el único host permitido es `api-fxpractice.oanda.com`. Si `OANDA_HOST`
apunta a la cuenta real (`api-fxtrade.oanda.com`) o a cualquier otro host, el bot se niega a
correr (código de salida 2). El token nunca se imprime ni se escribe en los logs.

## 3. Correr en modo simulado (sin cuenta todavía)

Con `MODO=simulado` el bot no manda órdenes a nadie: lleva una cuenta interna en
`bot/data/bot.db` con `CAPITAL_INICIAL` (10.000 USD por defecto).

- **Con token** (aunque sigas en simulado): usa velas y bid/ask **reales** de OANDA.
- **Sin token**: cae al fallback de Yahoo Finance (`AUDUSD=X`, `HG=F`) con spread sintético
  (`SPREAD_FALLBACK_PIPS`). El log lo advierte y todo queda etiquetado `fallback_yahoo`:
  **esos resultados no sirven para evaluar la señal** (es exactamente la ambigüedad de
  timestamps que produjo el artefacto). Sirve para probar la infraestructura.

```powershell
.\.venv\Scripts\python.exe run_diario.py --dry-run   # muestra la señal, no registra nada
.\.venv\Scripts\python.exe run_diario.py             # ciclo normal
.\.venv\Scripts\python.exe run_diario.py             # 2ª vez: 'ya_procesada', no duplica
```

Cuando tengas la cuenta, cambia a `MODO=oanda_practice`. La base guarda el modo con que se
creó y **no mezcla modos**: usa otro `DB_PATH` (p.ej. `data/bot_oanda.db`) o borra la base
simulada. En `oanda_practice` el capital inicial es el NAV real de la demo en la primera corrida.

### Idempotencia

- Cada vela genera una fila única en `senales`; si ya está `ejecutada`/`sin_cambio`, el bot no
  hace nada.
- La acción es "dejar la posición igual a la objetivo", así que repetirla es inofensiva aunque
  una corrida anterior haya fallado a medias.
- Un archivo candado evita dos corridas simultáneas.
- Las órdenes (POST/PUT a OANDA) **nunca** se reintentan automáticamente.

## 4. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Cubren: lógica de la señal, idempotencia, matemática de P&L/spread/sizing, agregación
mensual/anual/Sharpe/drawdown, rechazo del host real, redacción del token, cliente OANDA con
HTTP simulado (sin red), broker simulado y fallback de Yahoo.

## 5. Dashboard

```powershell
.\.venv\Scripts\streamlit.exe run dashboard\app.py
# otra base, p.ej. la demo:
.\.venv\Scripts\python.exe scripts\sembrar_demo.py
.\.venv\Scripts\streamlit.exe run dashboard\app.py -- --db data\demo.db
```

Muestra KPIs (capital actual, retorno total, último día, mes, año, Sharpe, máx. drawdown,
n° de operaciones), curva de equity, tablas de retornos diarios / mensuales / anuales y la
tabla de operaciones con filtros. En la barra lateral se elige cualquier `.db` de `bot/data/`.
Solo lee el ledger; nunca opera. `sembrar_demo.py` crea una base **sintética** separada
(`data/demo.db`, marcada como demo) y se niega a escribir sobre la base real.

HTML estático (un solo archivo, sin CDN, modo claro/oscuro automático):

```powershell
.\.venv\Scripts\python.exe dashboard\exportar_html.py                 # -> data\bot.html
.\.venv\Scripts\python.exe dashboard\exportar_html.py --db data\demo.db --salida data\demo.html
```

## 6. Tarea programada en Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\instalar_tarea_windows.ps1
# quitarla:
powershell -ExecutionPolicy Bypass -File .\scripts\instalar_tarea_windows.ps1 -Desinstalar
```

La tarea corre **cada hora al minuto :20** (no una vez al día). Motivo: Chile y Nueva York
cambian de horario de verano en fechas distintas, así que una hora fija en Chile a veces caería
antes del cierre de las 17:00 NY. Corriendo cada hora, el bot decide solo si hay una vela nueva
(y si ya pasaron los 15 min de espera); las corridas sin nada nuevo terminan en segundos. Como la
diferencia Chile–NY es siempre de horas enteras, :20 en Chile = :20 en NY → se opera ~17:20 NY.

Corre con tu usuario y solo con sesión iniciada; si el PC estaba apagado/suspendido, se ejecuta
al volver y el bot procesa la última vela disponible (las señales viejas no ejecutadas quedan
`reemplazada`). Logs en `bot/logs/bot_AAAA-MM.log`; historial de corridas en la tabla
`ejecuciones_log` (visible en el dashboard).

## 7. Migrar a la nube

El núcleo es Python puro (requests, pandas, sqlite3), sin dependencias de Windows.

1. Una VM Linux pequeña (1 vCPU / 1 GB basta; p.ej. e2-micro, t4g.nano, un droplet básico).
2. Copiar la carpeta `bot/` (sin `.venv/`), crear el venv e instalar `requirements.txt`.
   Copiar `bot/.env` por un canal seguro (`scp`), con `chmod 600 .env`. Si quieres conservar la
   historia, copia también `bot/data/bot.db`.
3. Programar cada hora. Con **cron** (`crontab -e`):
   ```
   20 * * * * cd /opt/bot && .venv/bin/python run_diario.py >> logs/cron.log 2>&1
   ```
   O con un **timer de systemd**:
   ```ini
   # /etc/systemd/system/paperbot.service
   [Service]
   Type=oneshot
   WorkingDirectory=/opt/bot
   ExecStart=/opt/bot/.venv/bin/python run_diario.py
   User=bot

   # /etc/systemd/system/paperbot.timer
   [Timer]
   OnCalendar=*-*-* *:20:00
   Persistent=true
   [Install]
   WantedBy=timers.target
   ```
   `systemctl enable --now paperbot.timer`
4. Página:
   - **Streamlit en la misma VM**: `streamlit run dashboard/app.py --server.address 127.0.0.1`
     detrás de un proxy con autenticación (Caddy/nginx + basic auth) o un túnel; no lo expongas
     abierto a internet.
   - **HTML estático en GitHub Pages**: agregar `dashboard/exportar_html.py --salida site/index.html`
     al final de la corrida (o en otro cron) y publicar `site/` en un repo/rama de Pages. Ojo: eso
     hace públicas tus operaciones demo.

## Agregar otra estrategia

Crear una clase en `estrategias/` que herede de `Estrategia` (define `instrumento`,
`instrumentos_datos`, `calcular_senal`) y registrarla en `estrategias/__init__.py`; se elige con
`ESTRATEGIA=...` en `.env`. Recomendado: una base (`DB_PATH`) por estrategia.
