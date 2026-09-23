"""Dashboard mínimo del bot (Streamlit). Solo LEE el ledger; nunca opera.

    cd bot
    .venv\\Scripts\\streamlit run dashboard\\app.py
    # otra base (p.ej. la demo):  ... run dashboard\\app.py -- --db data\\demo.db
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos_dashboard as dd  # noqa: E402

AZUL = "#2a78d6"
ROJO = "#e34948"

st.set_page_config(page_title="Paper bot · cobre → AUD", layout="wide")


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    a, _ = ap.parse_known_args(sys.argv[1:])
    return a


@st.cache_data(ttl=60)
def _cargar(ruta: str, _mtime: float):
    return dd.cargar(ruta)


def _layout_fig(fig: go.Figure, alto: int = 320) -> go.Figure:
    fig.update_layout(height=alto, margin=dict(l=8, r=8, t=8, b=8), showlegend=False,
                      hovermode="x unified")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1)
    return fig


# ------------------------------------------------------------------ fuente de datos
db_def = str(Path(_args().db or dd.db_por_defecto()).resolve())
bases = sorted({str(p.resolve()) for p in (dd.BOT_DIR / "data").glob("*.db")} | {db_def})
with st.sidebar:
    st.header("Base de datos")
    ruta = st.selectbox("Ledger SQLite", bases, index=bases.index(db_def),
                        format_func=lambda p: Path(p).name)
    st.caption(ruta)
    if st.button("Recargar"):
        st.cache_data.clear()

st.title("Paper trading · cobre → AUD/USD")
st.caption("Forward test honesto de la regla v1 (signo del retorno diario del cobre → posición "
           "en AUD/USD). La investigación corregida espera edge ~0 después de costos: esto mide, "
           "no promete ganancias. Cuenta DEMO / simulada.")

if not Path(ruta).exists():
    st.info(f"Todavía no existe `{Path(ruta).name}`. Corre `python run_diario.py` primero "
            "(o `python scripts/sembrar_demo.py` para ver una demo).")
    st.stop()

d = _cargar(ruta, Path(ruta).stat().st_mtime)
for aviso in d.avisos:
    st.warning(aviso)

# ---------------------------------------------------------------------------- KPIs
k = d.kpis
c = st.columns(4)
c[0].metric("Capital actual (USD)", dd.fmt_num(k["capital_actual"]),
            help=f"Capital inicial {dd.fmt_num(d.capital_inicial)} · modo {d.modo}")
c[1].metric("Retorno total", dd.fmt_pct(k["retorno_total"]))
c[2].metric(f"Retorno último día ({k['fecha_ultima'] or '—'})", dd.fmt_pct(k["retorno_hoy"]))
c[3].metric("N° operaciones", f"{k['n_operaciones']}",
            help=f"{k['n_cerradas']} cerradas · {k['n_abiertas']} abierta(s)")
c = st.columns(4)
c[0].metric("Retorno mes", dd.fmt_pct(k["retorno_mes"]))
c[1].metric("Retorno año", dd.fmt_pct(k["retorno_anio"]))
c[2].metric("Sharpe (anual., rf=0)", "—" if k["sharpe"] is None else f"{k['sharpe']:.2f}")
c[3].metric("Máx. drawdown", dd.fmt_pct(k["max_drawdown"]))
st.caption(f"Spread pagado total: {dd.fmt_num(k['costo_spread_total'])} USD · "
           f"financiamiento: {dd.fmt_num(k['financiamiento_total'])} USD · "
           f"tasa de acierto: {dd.fmt_tasa(k['tasa_acierto'])}"
           f" · {k['n_dias']} días registrados")

# ------------------------------------------------------------------- curva equity
st.subheader("Curva de equity")
if len(d.diaria):
    fechas = pd.to_datetime(d.diaria["fecha"])
    pocos = len(fechas) < 30
    fig = go.Figure(go.Scatter(x=fechas, y=d.diaria["equity"],
                               mode="lines+markers" if pocos else "lines",
                               line=dict(color=AZUL, width=2), marker=dict(size=8),
                               name="Equity",
                               hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f} USD<extra></extra>"))
    fig.add_hline(y=d.capital_inicial, line_width=1, line_dash="dot", opacity=0.5)
    if pocos:
        fig.update_xaxes(tickformat="%Y-%m-%d", dtick=86_400_000)
    st.plotly_chart(_layout_fig(fig), width="stretch", theme="streamlit")
else:
    st.info("Sin fotos de equity todavía.")

# --------------------------------------------------------------- tablas de retornos
st.subheader("Retornos")
t_d, t_m, t_a = st.tabs(["Diarios", "Mensuales", "Anuales"])
fmt_ret = {"equity": "{:,.2f}", "equity_final": "{:,.2f}", "retorno": "{:+.2%}"}
with t_d:
    st.dataframe(d.diaria.iloc[::-1].style.format(fmt_ret), width="stretch", hide_index=True)
with t_m:
    if len(d.mensual):
        fig = go.Figure(go.Bar(x=d.mensual["periodo"], y=d.mensual["retorno"] * 100,
                               marker_color=[AZUL if r >= 0 else ROJO for r in d.mensual["retorno"]],
                               hovertemplate="%{x}: %{y:+.2f}%<extra></extra>"))
        fig.update_yaxes(ticksuffix="%")
        st.plotly_chart(_layout_fig(fig, 240), width="stretch", theme="streamlit")
    st.dataframe(d.mensual.iloc[::-1].style.format(fmt_ret), width="stretch", hide_index=True)
with t_a:
    st.dataframe(d.anual.iloc[::-1].style.format(fmt_ret), width="stretch", hide_index=True)

# --------------------------------------------------------------------------- trades
st.subheader("Operaciones")
tt = dd.tabla_trades(d.trades)
if len(tt):
    f1, f2, f3 = st.columns(3)
    lados = f1.multiselect("Lado", ["largo", "corto"], default=["largo", "corto"])
    estados = f2.multiselect("Estado", sorted(tt["estado"].unique()),
                             default=sorted(tt["estado"].unique()))
    ent = pd.to_datetime(tt["entrada_tiempo"])
    rango = f3.date_input("Entrada entre", (ent.min().date(), ent.max().date()))
    m = tt["lado"].isin(lados) & tt["estado"].isin(estados)
    if isinstance(rango, (list, tuple)) and len(rango) == 2:
        m &= (ent.dt.date >= rango[0]) & (ent.dt.date <= rango[1])
    st.dataframe(
        tt[m].style.format({"entrada_precio": "{:.5f}", "salida_precio": "{:.5f}",
                            "costo_spread": "{:,.2f}", "financiamiento": "{:,.2f}",
                            "pnl_neto": "{:+,.2f}", "retorno_pct": "{:+.3%}"}, na_rep="—"),
        width="stretch", hide_index=True)
    st.caption(f"{int(m.sum())} de {len(tt)} operaciones")
else:
    st.info("Sin operaciones todavía.")

with st.expander("Señales y últimas ejecuciones del bot"):
    st.dataframe(d.senales.drop(columns=["detalle_json"]).iloc[::-1].head(60),
                 width="stretch", hide_index=True)
    st.dataframe(d.ejecuciones.iloc[::-1].head(30), width="stretch", hide_index=True)
