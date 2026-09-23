"""Genera un HTML estático de un solo archivo con la misma info del dashboard.

Sin dependencias externas en el HTML (ni CDN): CSS + SVG + un poco de JS inline, así
se puede subir tal cual a GitHub Pages o a cualquier hosting estático.

    python dashboard/exportar_html.py                       # usa DB_PATH de .env
    python dashboard/exportar_html.py --db data/demo.db --salida data/demo.html
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos_dashboard as dd  # noqa: E402

CSS = """
:root{color-scheme:light;--bg:#f7f7f5;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
--serie:#2a78d6;--neg:#e34948;--aviso-bg:#fff4db;--aviso-ink:#6b4a00}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--bg:#121211;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--axis:#383835;--border:rgba(255,255,255,.10);--serie:#3987e5;--neg:#e66767;
--aviso-bg:#3a2e10;--aviso-ink:#f5d78a}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#121211;--surface:#1a1a19;--ink:#fff;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--serie:#3987e5;--neg:#e66767;--aviso-bg:#3a2e10;--aviso-ink:#f5d78a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px}
.sub{color:var(--ink2);margin:0 0 16px;max-width:75ch}
.aviso{background:var(--aviso-bg);color:var(--aviso-ink);border-radius:8px;padding:8px 12px;margin:8px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.kpi .l{color:var(--ink2);font-size:12px}.kpi .v{font-size:22px;font-weight:600;
font-variant-numeric:tabular-nums;margin-top:2px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px;position:relative}
svg{display:block;width:100%;height:auto}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--border);
border-radius:6px;padding:4px 8px;font-size:12px;color:var(--ink);display:none;white-space:nowrap;
box-shadow:0 2px 8px rgba(0,0,0,.12)}
.tabs{display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap}
.tabs button,select{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--border);
border-radius:6px;padding:4px 10px;cursor:pointer}
.tabs button[aria-selected="true"]{border-color:var(--serie);font-weight:600}
.tw{overflow-x:auto;max-height:420px;overflow-y:auto;border:1px solid var(--border);border-radius:8px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:5px 10px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
th{position:sticky;top:0;background:var(--surface);color:var(--ink2);font-weight:600}
td:first-child,th:first-child{text-align:left}
.filtros{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:8px;color:var(--ink2)}
footer{color:var(--muted);font-size:12px;margin-top:32px}
"""

JS = """
(function(){
const d=JSON.parse(document.getElementById('datos-equity').textContent);
const svg=document.getElementById('eq'),tip=document.getElementById('tip');
if(!d.length||!svg)return;
const W=1000,H=300,L=64,R=12,T=12,B=28;
const ys=d.map(p=>p[1]),cap=Number(svg.dataset.cap);
let mn=Math.min(...ys,cap),mx=Math.max(...ys,cap);if(mx===mn){mx+=1;mn-=1}
const pad=(mx-mn)*.08;mn-=pad;mx+=pad;
const x=i=>L+(d.length===1?0.5:i/(d.length-1))*(W-L-R),y=v=>T+(1-(v-mn)/(mx-mn))*(H-T-B);
const ns='http://www.w3.org/2000/svg',el=(n,a)=>{const e=document.createElementNS(ns,n);
for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e};
for(let k=0;k<=4;k++){const v=mn+(mx-mn)*k/4;el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),stroke:'var(--grid)','stroke-width':1});
const t=el('text',{x:L-8,y:y(v)+4,'text-anchor':'end','font-size':11,fill:'var(--muted)'});
const dec=(mx-mn)<50?2:0;t.textContent=v.toLocaleString('es-CL',{minimumFractionDigits:dec,maximumFractionDigits:dec})}
el('line',{x1:L,x2:W-R,y1:y(cap),y2:y(cap),stroke:'var(--muted)','stroke-dasharray':'4 4'});
[0,Math.floor((d.length-1)/2),d.length-1].forEach(i=>{const t=el('text',{x:x(i),y:H-8,
'text-anchor':i===0?'start':(i===d.length-1?'end':'middle'),'font-size':11,fill:'var(--muted)'});t.textContent=d[i][0]});
el('path',{d:d.map((p,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(p[1]).toFixed(1)).join(''),
fill:'none',stroke:'var(--serie)','stroke-width':2,'stroke-linejoin':'round'});
if(d.length<30)d.forEach((p,i)=>el('circle',{cx:x(i),cy:y(p[1]),r:4,fill:'var(--serie)',stroke:'var(--surface)','stroke-width':2}));
const cx=el('line',{y1:T,y2:H-B,stroke:'var(--axis)','stroke-width':1,visibility:'hidden'});
const dot=el('circle',{r:4,fill:'var(--serie)',stroke:'var(--surface)','stroke-width':2,visibility:'hidden'});
svg.addEventListener('mousemove',ev=>{const r=svg.getBoundingClientRect();
const px=(ev.clientX-r.left)/r.width*W;let i=Math.round((px-L)/(W-L-R)*(d.length-1));
i=Math.max(0,Math.min(d.length-1,i));const p=d[i];
cx.setAttribute('x1',x(i));cx.setAttribute('x2',x(i));cx.setAttribute('visibility','visible');
dot.setAttribute('cx',x(i));dot.setAttribute('cy',y(p[1]));dot.setAttribute('visibility','visible');
tip.style.display='block';tip.innerHTML='<b>'+p[0]+'</b><br>'+p[1].toLocaleString('es-CL',{minimumFractionDigits:2,maximumFractionDigits:2})+' USD · '+(p[2]>=0?'+':'')+(p[2]*100).toFixed(2)+'%';
const left=x(i)/W*r.width;tip.style.left=Math.min(left+12,r.width-tip.offsetWidth-4)+'px';tip.style.top='16px'});
svg.addEventListener('mouseleave',()=>{tip.style.display='none';cx.setAttribute('visibility','hidden');dot.setAttribute('visibility','hidden')});
})();
document.querySelectorAll('.tabs button').forEach(b=>b.addEventListener('click',()=>{
document.querySelectorAll('.tabs button').forEach(o=>o.setAttribute('aria-selected',o===b));
document.querySelectorAll('.panel').forEach(p=>p.hidden=p.id!==b.dataset.p)}));
function filtrar(){const l=document.getElementById('f-lado').value,e=document.getElementById('f-estado').value;
let n=0;document.querySelectorAll('#trades tbody tr').forEach(tr=>{const ok=(!l||tr.dataset.lado===l)&&(!e||tr.dataset.estado===e);
tr.hidden=!ok;if(ok)n++});document.getElementById('f-n').textContent=n}
['f-lado','f-estado'].forEach(i=>{const s=document.getElementById(i);if(s)s.addEventListener('change',filtrar)});
"""


def _tabla(df: pd.DataFrame, fmts: dict, id_: str = "", attrs_fila=None) -> str:
    if len(df) == 0:
        return "<p class='sub'>Sin datos todavía.</p>"
    cab = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    filas = []
    for _, r in df.iterrows():
        celdas = []
        for c in df.columns:
            v = r[c]
            if c in fmts:
                txt = fmts[c](v)
            else:
                txt = "—" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
            celdas.append(f"<td>{html.escape(txt)}</td>")
        extra = attrs_fila(r) if attrs_fila else ""
        filas.append(f"<tr{extra}>{''.join(celdas)}</tr>")
    idattr = f" id='{id_}'" if id_ else ""
    return (f"<div class='tw'><table{idattr}><thead><tr>{cab}</tr></thead>"
            f"<tbody>{''.join(filas)}</tbody></table></div>")


def _barras_mensuales(mensual: pd.DataFrame) -> str:
    if len(mensual) == 0:
        return ""
    W, H, L, B, T = 1000, 200, 48, 24, 10
    vals = (mensual["retorno"] * 100).tolist()
    m = max(max(abs(v) for v in vals), 0.01)
    ancho = (W - L - 8) / len(vals)
    y0 = T + (H - T - B) / 2
    esc = (H - T - B) / 2 / m
    partes = [f"<line x1='{L}' x2='{W - 8}' y1='{y0}' y2='{y0}' stroke='var(--axis)'/>",
              f"<text x='{L - 6}' y='{T + 10}' text-anchor='end' font-size='11' fill='var(--muted)'>+{m:.2f}%</text>",
              f"<text x='{L - 6}' y='{H - B}' text-anchor='end' font-size='11' fill='var(--muted)'>-{m:.2f}%</text>"]
    for i, (p, v) in enumerate(zip(mensual["periodo"], vals)):
        h = abs(v) * esc
        x = L + (i + 0.5) * ancho - min(max(ancho * 0.7, 1), 60) / 2
        w = min(max(ancho * 0.7, 1), 60)
        y = y0 - h if v >= 0 else y0
        color = "var(--serie)" if v >= 0 else "var(--neg)"
        partes.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{max(h, 0.5):.1f}' "
                      f"rx='2' fill='{color}'><title>{p}: {v:+.2f}%</title></rect>")
    paso = max(1, len(vals) // 8)
    for i in range(0, len(vals), paso):
        partes.append(f"<text x='{L + (i + 0.5) * ancho:.1f}' y='{H - 6}' text-anchor='middle' "
                      f"font-size='11' fill='var(--muted)'>{mensual['periodo'].iloc[i]}</text>")
    return (f"<div class='card'><svg viewBox='0 0 {W} {H}' role='img' "
            f"aria-label='Retornos mensuales'>{''.join(partes)}</svg></div>")


def generar_html(d: dd.DatosDashboard) -> str:
    k = d.kpis
    pct, num = dd.fmt_pct, dd.fmt_num
    tiles = [
        ("Capital actual (USD)", num(k["capital_actual"])),
        ("Retorno total", pct(k["retorno_total"])),
        (f"Retorno último día ({k['fecha_ultima'] or '—'})", pct(k["retorno_hoy"])),
        ("Retorno mes", pct(k["retorno_mes"])),
        ("Retorno año", pct(k["retorno_anio"])),
        ("Sharpe (anual., rf=0)", "—" if k["sharpe"] is None else f"{k['sharpe']:.2f}"),
        ("Máx. drawdown", pct(k["max_drawdown"])),
        ("N° operaciones", f"{k['n_operaciones']}"),
    ]
    kpis_html = "".join(f"<div class='kpi'><div class='l'>{html.escape(l)}</div>"
                        f"<div class='v'>{html.escape(v)}</div></div>" for l, v in tiles)
    avisos = "".join(f"<div class='aviso'>{html.escape(a)}</div>" for a in d.avisos)
    serie = [[f, round(float(e), 2), float(r)] for f, e, r in
             zip(d.diaria["fecha"], d.diaria["equity"], d.diaria["retorno"])]

    f_ret = {"equity": num, "equity_final": num, "retorno": pct}
    tt = dd.tabla_trades(d.trades)
    f_tr = {"entrada_precio": lambda v: num(v, 5), "salida_precio": lambda v: num(v, 5),
            "costo_spread": num, "financiamiento": num, "pnl_neto": num,
            "retorno_pct": lambda v: pct(v, 3), "unidades": lambda v: f"{int(v):,}"}
    trades_html = _tabla(tt, f_tr, "trades",
                         lambda r: f" data-lado='{r['lado']}' data-estado='{r['estado']}'")
    estados = sorted(tt["estado"].unique()) if len(tt) else []
    generado = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paper bot cobre-AUD</title><style>{CSS}</style></head>
<body><main>
<h1>Paper trading · cobre → AUD/USD</h1>
<p class="sub">Forward test honesto de la regla v1 (signo del retorno diario del cobre → posición en
AUD/USD). La investigación corregida espera edge ~0 después de costos: esto mide, no promete
ganancias. Cuenta DEMO / simulada · modo <b>{html.escape(d.modo)}</b> · capital inicial
{num(d.capital_inicial)} USD.</p>
{avisos}
<div class="kpis">{kpis_html}</div>
<p class="sub" style="margin-top:8px">Spread pagado total: {num(k['costo_spread_total'])} USD ·
financiamiento: {num(k['financiamiento_total'])} USD · {k['n_dias']} días registrados.</p>

<h2>Curva de equity</h2>
<div class="card"><svg id="eq" viewBox="0 0 1000 300" data-cap="{d.capital_inicial}" role="img"
aria-label="Curva de equity"></svg><div class="tip" id="tip"></div></div>
<script type="application/json" id="datos-equity">{json.dumps(serie)}</script>

<h2>Retornos</h2>
<div class="tabs" role="tablist">
<button data-p="p-d" aria-selected="false">Diarios</button>
<button data-p="p-m" aria-selected="true">Mensuales</button>
<button data-p="p-a" aria-selected="false">Anuales</button></div>
<div class="panel" id="p-d" hidden>{_tabla(d.diaria.iloc[::-1], f_ret)}</div>
<div class="panel" id="p-m">{_barras_mensuales(d.mensual)}<div style="height:8px"></div>
{_tabla(d.mensual.iloc[::-1], f_ret)}</div>
<div class="panel" id="p-a" hidden>{_tabla(d.anual.iloc[::-1], f_ret)}</div>

<h2>Operaciones</h2>
<div class="filtros">
<label>Lado <select id="f-lado"><option value="">todos</option><option>largo</option>
<option>corto</option></select></label>
<label>Estado <select id="f-estado"><option value="">todos</option>
{''.join(f'<option>{html.escape(e)}</option>' for e in estados)}</select></label>
<span><span id="f-n">{len(tt)}</span> operaciones</span></div>
{trades_html}
<footer>Generado {generado} desde {html.escape(d.ruta.name)}. No es asesoría financiera.</footer>
</main><script>{JS}</script></body></html>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Exporta el dashboard a un HTML estático")
    ap.add_argument("--db", default=None, help="ruta al ledger SQLite (por defecto DB_PATH)")
    ap.add_argument("--salida", default=None, help="archivo HTML de salida")
    a = ap.parse_args(argv)
    ruta = Path(a.db) if a.db else dd.db_por_defecto()
    salida = Path(a.salida) if a.salida else ruta.with_suffix(".html")
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(generar_html(dd.cargar(ruta)), encoding="utf-8")
    print(f"HTML generado: {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
