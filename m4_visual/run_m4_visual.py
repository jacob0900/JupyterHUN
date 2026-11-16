"""M4 · EDA espacio-temporal, parte visual (HUN-203) — Persona 3.

Uso:  python m4_visual/run_m4_visual.py
Salida: outputs/m4_visual/{figuras,tablas,html}/ y outputs/m4_visual/M4_visual_informe.md
Los HTML interactivos (Sankey, heatmaps) están pensados para el dashboard de Persona 1 (M8).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import seaborn as sns  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from statsmodels.tsa.seasonal import MSTL, STL  # noqa: E402

from eda_comun import (  # noqa: E402
    COLORES, DIAS, M4, NOMBRE, OUT, bh, carpeta, con, dif_medianas_ic, fmt_p, guardar,
    kruskal_eps2, magnitud_eps2, md_tabla, parametros,
)

FIG, TAB = carpeta("m4_visual")
HTML = OUT / "m4_visual" / "html"
HTML.mkdir(exist_ok=True)
C = con()
P = parametros()
SERV4 = ["urgencias_ingreso", "urgencias_observacion", "hospitalizacion", "ambulatorio"]
SERV2 = ["hospitalizacion", "urgencias_observacion"]
informe: list[str] = []


def sec(t):
    print(f"· {t}")
    informe.append(f"\n## {t}\n")


def add(*x):
    informe.extend(x)


def fig_md(n, alt):
    add(f"\n![{alt}](figuras/{n})\n")


def html(fig: go.Figure, nombre: str) -> str:
    fig.write_html(HTML / nombre, include_plotlyjs="cdn", full_html=True)
    return nombre


# =====================================================================================
# 1. Heatmaps día × hora
# =====================================================================================
sec("1. Densidad de admisiones por día de la semana × hora")
hl = pd.read_parquet(M4 / "heatmap_llegadas.parquet")
fig, axes = plt.subplots(2, 2, figsize=(15, 8), gridspec_kw={"hspace": 0.45})
pico = []
for ax, s in zip(axes.ravel(), SERV4):
    m = hl[hl.servicio == s].pivot(index="dia_semana", columns="hora_del_dia", values="llegadas_prom").sort_index()
    sns.heatmap(m, ax=ax, cmap="YlOrRd", cbar_kws={"label": "llegadas / h"}, linewidths=0.2)
    ax.set_yticklabels(DIAS, rotation=0)
    ax.set_title(NOMBRE[s])
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("")
    i, j = np.unravel_index(np.nanargmax(m.values), m.shape)
    pico.append({"servicio": NOMBRE[s], "pico": f"{DIAS[i]} {m.columns[j]}:00", "llegadas_pico_h": m.values[i, j],
                 "llegadas_valle_h": np.nanmin(m.values), "razón pico/valle": m.values[i, j] / max(np.nanmin(m.values), 0.01),
                 "llegadas_prom_h": np.nanmean(m.values)})
fig.suptitle("Llegadas promedio por hora (periodo estable)", fontweight="bold")
fig_md(guardar(fig, FIG / "01_heatmap_llegadas.png"), "heatmap llegadas")
pico = pd.DataFrame(pico)
pico.to_csv(TAB / "picos_llegadas.csv", index=False)
add(md_tabla(pico, floatfmt=".1f"))

pf = make_subplots(rows=2, cols=2, subplot_titles=[NOMBRE[s] for s in SERV4], horizontal_spacing=0.08, vertical_spacing=0.12)
for k, s in enumerate(SERV4):
    m = hl[hl.servicio == s].pivot(index="dia_semana", columns="hora_del_dia", values="llegadas_prom").sort_index()
    pf.add_trace(go.Heatmap(z=m.values, x=list(m.columns), y=DIAS, colorscale="YlOrRd", showscale=False,
                            hovertemplate="%{y} %{x}:00<br>%{z:.1f} llegadas/h<extra>" + NOMBRE[s] + "</extra>"),
                 row=k // 2 + 1, col=k % 2 + 1)
pf.update_yaxes(autorange="reversed")
pf.update_layout(title="Llegadas promedio por día × hora — HUN (periodo estable)", height=650)
html(pf, "heatmap_llegadas.html")

ho = pd.read_parquet(M4 / "heatmap_ocupacion.parquet")
fig, axes = plt.subplots(2, 2, figsize=(15, 8), gridspec_kw={"hspace": 0.45})
for r, s in enumerate(SERV2):
    d = ho[ho.servicio == s]
    m1 = d.pivot(index="dia_semana", columns="hora_del_dia", values="ocupacion_prom").sort_index()
    m2 = d.pivot(index="dia_semana", columns="hora_del_dia", values="frac_horas_saturadas").sort_index()
    sns.heatmap(m1, ax=axes[r, 0], cmap="RdYlGn_r", cbar_kws={"label": "ocupación media"}, linewidths=0.2)
    sns.heatmap(m2, ax=axes[r, 1], cmap="Reds", vmin=0, vmax=1, cbar_kws={"label": "fracción saturada"}, linewidths=0.2)
    umbral = 0.95 if s == "hospitalizacion" else 0.85
    axes[r, 0].set_title(f"{NOMBRE[s]} — ocupación media")
    axes[r, 1].set_title(f"{NOMBRE[s]} — fracción de horas con ocupación ≥ {umbral}")
    for ax in axes[r]:
        ax.set_yticklabels(DIAS, rotation=0)
        ax.set_xlabel("Hora del día")
        ax.set_ylabel("")
fig.suptitle("Mapas de calor de saturación (periodo estable, capacidad = proxy P95 móvil)", fontweight="bold")
fig_md(guardar(fig, FIG / "02_heatmap_saturacion.png"), "heatmap saturación")
pf = make_subplots(rows=2, cols=2, horizontal_spacing=0.1, vertical_spacing=0.12,
                   subplot_titles=[f"{NOMBRE[s]} — {t}" for s in SERV2 for t in ("ocupación media", "fracción saturada")])
for r, s in enumerate(SERV2):
    d = ho[ho.servicio == s]
    for c, (col, cs) in enumerate([("ocupacion_prom", "RdYlGn_r"), ("frac_horas_saturadas", "Reds")]):
        m = d.pivot(index="dia_semana", columns="hora_del_dia", values=col).sort_index()
        pf.add_trace(go.Heatmap(z=m.values, x=list(m.columns), y=DIAS, colorscale=cs, showscale=False,
                                hovertemplate="%{y} %{x}:00<br>%{z:.3f}<extra></extra>"), row=r + 1, col=c + 1)
pf.update_yaxes(autorange="reversed")
pf.update_layout(title="Saturación por día × hora — HUN", height=650)
html(pf, "heatmap_saturacion.html")
sat_top = (ho.sort_values("frac_horas_saturadas", ascending=False).groupby("servicio").head(5)
           .assign(dia=lambda d: d.dia_semana.map(dict(enumerate(DIAS, 1))))
           [["servicio", "dia", "hora_del_dia", "ocupacion_prom", "frac_horas_saturadas"]])
sat_top.to_csv(TAB / "franjas_mas_saturadas.csv", index=False)
add(
    "\n**Franjas con más horas saturadas** (top 5 por servicio):\n", md_tabla(sat_top, floatfmt=".3f"),
    "\n**Lectura.**\n",
    "- **Urgencias (ingreso y observación)**: las llegadas son muy diurnas, con pico a media mañana "
    "(≈ 10–11 h) de lunes a viernes, valle a las 3–5 h y fines de semana más planos. La saturación de "
    "observación llega con 4–6 h de retraso: se concentra **por la tarde, de lunes a viernes, sobre "
    "todo martes y miércoles entre las 14 y las 17 h**.\n",
    "- **Ambulatorio**: solo en horario hábil (7–17 h, lunes a viernes), con un hueco al mediodía (12 h). "
    "Prácticamente no hay actividad el fin de semana.\n",
    "- **Hospitalización**: las llegadas se concentran en **dos franjas exactas, 16–17 h y 20–22 h**, "
    "todos los días. Un patrón tan rígido parece la hora de **registro administrativo** (lotes por cambio "
    "de turno) más que la hora clínica de ingreso. Persona 2 debe tenerlo en cuenta: la `hora_del_dia` de "
    "ingreso a hospitalización refleja el proceso de registro. La saturación se da en la **madrugada y "
    "la mañana**, antes de las altas del mediodía, y es máxima el **lunes**: durante el fin de semana hay "
    "menos altas y las camas se acumulan.",
)

# =====================================================================================
# 2. Sankey y matriz de transición
# =====================================================================================
sec("2. Flujo de pacientes entre servicios (Sankey y matriz de transición)")
sk = pd.read_parquet(M4 / "sankey_aristas.parquet")
MIN_N = 100
skf = sk[sk.n >= MIN_N].copy()
nodos = sorted(set(skf.origen) | set(skf.destino), key=lambda x: (int(x.split(":")[0]), x))
idx = {n: i for i, n in enumerate(nodos)}
etq_serv = {"(llegada)": "Llegada", "(egreso)": "Egreso", **NOMBRE}
col_serv = {"(llegada)": "#7f7f7f", "(egreso)": "#404040", **COLORES}
total = sk[sk.etapa == 0].n.sum()
entra = {n: 0 for n in nodos}
for o, d, n in zip(sk.origen, sk.destino, sk.n):
    if d in entra:
        entra[d] += n
labels = []
for n in nodos:
    e, s = n.split(":", 1)
    v = total if s == "(llegada)" else entra[n]
    labels.append(f"{etq_serv[s]} · paso {e}<br>{v:,} ({100*v/total:.1f} %)" if s != "(llegada)" else f"Llegada<br>{total:,} visitas")
etapas_max = max(int(n.split(":")[0]) for n in nodos)
xs = [0.001 + 0.998 * int(n.split(":")[0]) / etapas_max for n in nodos]


def rgba(hexc, a):
    h = hexc.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


sfig = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(label=labels, x=xs, pad=18, thickness=16, color=[col_serv[n.split(":", 1)[1]] for n in nodos],
              hovertemplate="%{label}<extra></extra>"),
    link=dict(source=[idx[o] for o in skf.origen], target=[idx[d] for d in skf.destino], value=skf.n,
              color=[rgba(col_serv[d.split(":", 1)[1]], 0.35) for d in skf.destino],
              hovertemplate="%{source.label} → %{target.label}<br>%{value:,} visitas<extra></extra>"),
))
sfig.update_layout(title=f"Recorrido de las visitas por servicio — HUN ({total:,} visitas; aristas con n ≥ {MIN_N})",
                   font=dict(size=11), height=600, width=1200)
html(sfig, "sankey_flujo.html")


def sankey_estatico(aristas: pd.DataFrame, ruta: Path):
    """Sankey estático en matplotlib (kaleido se cuelga en Windows). Nodos = 'etapa:servicio'."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    etapa = lambda n: int(n.split(":")[0])  # noqa: E731
    serv = lambda n: n.split(":", 1)[1]  # noqa: E731
    flujo = {}
    for o, d, n in zip(aristas.origen, aristas.destino, aristas.n):
        flujo[o] = max(flujo.get(o, 0), 0)
        flujo[d] = flujo.get(d, 0)
    entrada = aristas.groupby("destino").n.sum()
    salida = aristas.groupby("origen").n.sum()
    tam = {n: max(entrada.get(n, 0), salida.get(n, 0)) for n in flujo}
    total = max(sum(v for n, v in tam.items() if etapa(n) == e) for e in {etapa(n) for n in tam})
    gap = 0.03 * total
    orden_s = ["(llegada)", "ambulatorio", "urgencias_ingreso", "urgencias_observacion", "hospitalizacion", "(egreso)"]
    pos = {}
    for e in sorted({etapa(n) for n in tam}):
        y = 0.0
        for n in sorted([n for n in tam if etapa(n) == e], key=lambda n: orden_s.index(serv(n))):
            pos[n] = [y, y]  # [y0 de nodo, cursor salida]
            y += tam[n] + gap
    cur_in = {n: pos[n][0] for n in pos}
    fig, ax = plt.subplots(figsize=(15, 7.5))
    w = 0.12
    for o, d, n in sorted(zip(aristas.origen, aristas.destino, aristas.n), key=lambda t: (etapa(t[0]), orden_s.index(serv(t[0])), orden_s.index(serv(t[1])))):
        x0, x1 = etapa(o) + w / 2, etapa(d) - w / 2
        y0a = pos[o][1]; pos[o][1] += n
        y1a = cur_in[d]; cur_in[d] += n
        xm = (x0 + x1) / 2
        verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1a + n), (xm, y1a + n), (xm, y0a + n), (x0, y0a + n), (x0, y0a)]
        codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), fc=col_serv[serv(d)], alpha=0.35, lw=0))
    for n, (y0, _) in pos.items():
        e = etapa(n)
        ax.add_patch(plt.Rectangle((e - w / 2, y0), w, tam[n], fc=col_serv[serv(n)], ec="none"))
        if tam[n] / total > 0.004:
            ax.text(e + w / 2 + 0.03, y0 + tam[n] / 2, f"{etq_serv[serv(n)]}\n{tam[n]:,} ({100*tam[n]/total:.1f} %)",
                    va="center", fontsize=7.5)
    emax = max(etapa(n) for n in pos)
    ax.set_xlim(-0.2, emax + 0.9)
    ax.set_ylim(max(v[0] + tam[k] for k, v in pos.items()) + gap, -gap)
    ax.set_xticks(range(emax + 1))
    ax.set_xticklabels([f"Paso {i}" for i in range(emax + 1)])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Recorrido de las visitas por servicio ({total:,} visitas; aristas con n ≥ {MIN_N})", fontweight="bold")
    return guardar(fig, ruta)


fig_md(sankey_estatico(skf, FIG / "03_sankey.png"), "Sankey")
add(f"Sankey interactivo: [`html/sankey_flujo.html`](html/sankey_flujo.html). Se omiten las aristas con menos de {MIN_N} visitas "
    f"({(sk.n < MIN_N).sum()} de {len(sk)}, {100*sk.loc[sk.n < MIN_N,'n'].sum()/sk.n.sum():.2f} % del flujo).\n")

mt = pd.read_parquet(M4 / "matriz_transicion.parquet")
orden_f = ["(llegada)", "ambulatorio", "urgencias_ingreso", "urgencias_observacion", "hospitalizacion"]
orden_c = ["ambulatorio", "urgencias_ingreso", "urgencias_observacion", "hospitalizacion", "(egreso)"]
M = mt.pivot(index="desde", columns="hacia", values="prob").reindex(index=orden_f, columns=orden_c).fillna(0)
N = mt.pivot(index="desde", columns="hacia", values="n").reindex(index=orden_f, columns=orden_c).fillna(0)
fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.1, 1]})
ann = np.vectorize(lambda p, n: f"{p:.1%}\n(n={int(n):,})" if n > 0 else "")(M.values, N.values)
sns.heatmap(M, ax=axes[0], cmap="Blues", annot=ann, fmt="", annot_kws={"size": 7.5}, vmin=0, vmax=1,
            cbar_kws={"label": "P(siguiente | actual)"}, linewidths=0.5)
axes[0].set_xticklabels([etq_serv[c] for c in orden_c], rotation=20, ha="right")
axes[0].set_yticklabels([etq_serv[c] for c in orden_f], rotation=0)
axes[0].set_xlabel("Siguiente")
axes[0].set_ylabel("Actual")
axes[0].set_title("Matriz de transición entre servicios")
rf = pd.read_parquet(M4 / "rutas_frecuentes.parquet").head(10).iloc[::-1]
axes[1].barh(rf.ruta.str.replace(" → (egreso)", "").str.replace("urgencias_", "urg_"), rf.frac * 100, color="#4a7fb5")
for y, (f, n) in enumerate(zip(rf.frac, rf.n_visitas)):
    axes[1].text(f * 100 + 0.5, y, f"{f:.1%} ({n:,})", va="center", fontsize=7.5)
axes[1].set_xlabel("% de visitas")
axes[1].set_title("10 recorridos completos más frecuentes (→ egreso)")
axes[1].tick_params(axis="y", labelsize=7.5)
axes[1].set_xlim(0, 80)
fig.suptitle("Transiciones entre servicios (visit_detail encadenado por visita)", fontweight="bold")
fig_md(guardar(fig, FIG / "04_matriz_transicion_rutas.png"), "matriz transición")
M.to_csv(TAB / "matriz_transicion_prob.csv")
hosp_desde = mt[(mt.hacia == "hospitalizacion") & (mt.desde != "hospitalizacion")].groupby("desde").n.sum()
add(
    "**Lectura.**\n",
    "- El 65 % de las visitas son una consulta ambulatoria aislada, y el 17 % sigue el circuito típico de "
    "urgencias (ingreso → observación → egreso).\n",
    f"- De los ingresos a hospitalización (sin contar registros encadenados del mismo servicio), el {100*hosp_desde.get('urgencias_observacion',0)/hosp_desde.sum():.0f} % "
    "viene de observación de urgencias. Hospitalización es casi exclusivamente un destino de urgencias.\n",
    "- Desde observación, el 54 % egresa, el 40 % pasa a otro registro de observación (cadenas de "
    "registros del mismo servicio) y el 6 % se hospitaliza. Ese 6 % es la tasa de conversión que "
    "alimenta la presión sobre camas.\n",
    "- Las transiciones raras (hospitalización → urgencias, observación → ambulatorio) tienen esperas "
    "de cientos de horas: son reingresos dentro de la misma visita, no flujo operativo.",
)

# =====================================================================================
# 3. Calendario colombiano
# =====================================================================================
sec("3. Efecto del calendario colombiano (festivos, puentes, día después de festivo)")
ld = pd.read_parquet(M4 / "llegadas_dia.parquet")
ld["fecha"] = pd.to_datetime(ld.fecha)
# OJO: en `calendario_dia` de Persona 1, `dia_despues_festivo` marca el primer día hábil tras
# CUALQUIER día no hábil (incluye ~130 lunes normales). Aquí se separa el lunes normal del día
# siguiente a un festivo real.
cd_ = pd.read_parquet(M4 / "calendario_dia.parquet").sort_values("fecha")
cd_["fecha"] = pd.to_datetime(cd_.fecha)
cd_["tras_festivo_real"] = cd_.festivo.shift(1, fill_value=False) & ~cd_.festivo & ~cd_.fin_de_semana
n_flag_p1 = int(cd_.loc[cd_.fecha.between(ld.fecha.min(), ld.fecha.max()), "dia_despues_festivo"].sum())
n_flag_real = int(cd_.loc[cd_.fecha.between(ld.fecha.min(), ld.fecha.max()), "tras_festivo_real"].sum())
ld = ld.merge(cd_[["fecha", "tras_festivo_real"]], on="fecha", how="left")
ld["tipo_dia"] = np.select(
    [ld.festivo, ld.tras_festivo_real, ld.fin_de_semana, ld.dia_semana == 1],
    ["festivo", "día después festivo", "fin de semana", "lunes"], "laboral (mar–vie)")
# Referencia sin tendencia: mediana de los días laborales "normales" en ±14 días
ref = []
for s, d in ld.groupby("servicio"):
    d = d.sort_values("fecha").set_index("fecha")
    lab = d.llegadas.where(d.tipo_dia == "laboral (mar–vie)")
    r = lab.rolling("29D", center=True, min_periods=5).median()
    ref.append(pd.Series(r.values, index=d.index).rename(s))
ld = ld.merge(pd.concat(ref, axis=1).stack().rename("ref").reset_index().rename(columns={"level_1": "servicio"}),
              on=["fecha", "servicio"], how="left")
ld["indice"] = ld.llegadas / ld.ref
filas, tests = [], []
for s, d in ld.groupby("servicio"):
    g = d.groupby("tipo_dia").agg(dias=("llegadas", "size"), llegadas_media=("llegadas", "mean"),
                                  indice_mediano=("indice", "median"))
    g["servicio"] = s
    filas.append(g.reset_index())
    _, p, e2 = kruskal_eps2([x.indice.dropna().values for _, x in d.groupby("tipo_dia")])
    lab = d.loc[d.tipo_dia == "laboral (mar–vie)", "indice"].dropna().values
    lun = d.loc[d.tipo_dia == "lunes", "indice"].dropna().values
    dd = d.loc[d.tipo_dia == "día después festivo", "indice"].dropna().values
    pu = d.loc[d.puente & d.festivo, "indice"].dropna().values
    dm, lo, hi = dif_medianas_ic(dd, lab, max_n=5000)
    dl, lol, hil = dif_medianas_ic(lun, lab, max_n=5000)
    tests.append({"servicio": NOMBRE[s], "KW ε² (tipo de día, índice sin tendencia)": e2, "magnitud": magnitud_eps2(e2), "p": p,
                  "Δ día después festivo − laboral": f"{dm:+.3f} [{lo:+.3f}, {hi:+.3f}]",
                  "Δ lunes − laboral": f"{dl:+.3f} [{lol:+.3f}, {hil:+.3f}]",
                  "índice mediano festivo de puente": np.median(pu) if len(pu) else np.nan})
cal = pd.concat(filas)[["servicio", "tipo_dia", "dias", "llegadas_media", "indice_mediano"]]
cal.to_csv(TAB / "efecto_calendario.csv", index=False)
tests = pd.DataFrame(tests)
tests["p_BH"] = bh(tests.p)
tests["p_BH"] = tests.p_BH.map(fmt_p)
tests.drop(columns="p").to_csv(TAB / "efecto_calendario_tests.csv", index=False)
add(
    "Para separar el efecto del día de la tendencia de largo plazo, cada día se expresa como un **índice** "
    "= llegadas del día / mediana de los días laborales normales (martes–viernes) en ±14 días (1.0 = un "
    "día laboral típico de esas fechas).\n",
    f"\n> ⚠️ **Nota para Persona 1**: en `calendario_dia`, `dia_despues_festivo` está marcado en {n_flag_p1} "
    f"días del periodo, pero solo {n_flag_real} siguen a un festivo real. La columna marca el primer día "
    "hábil tras **cualquier** día no hábil, así que incluye los lunes normales tras un fin de semana. "
    "Por eso el \"rebote\" de `outputs/m4_resumen.md` (1 067/día) mezcla el efecto lunes con el efecto "
    "festivo. Aquí se separan: `lunes` (tras un fin de semana normal) y `día después festivo` (el día "
    "hábil siguiente a un festivo real). Sugiero renombrar la columna o añadir `tras_festivo`.\n",
    md_tabla(cal, floatfmt=".3f"),
    "\n", md_tabla(tests.drop(columns="p"), floatfmt=".3f"),
)
orden_t = ["laboral (mar–vie)", "lunes", "día después festivo", "fin de semana", "festivo"]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
dplot = ld[ld.servicio != "ambulatorio"].assign(s=lambda d: d.servicio.map(NOMBRE))
sns.boxplot(data=dplot, x="s", y="indice", hue="tipo_dia", hue_order=orden_t, ax=axes[0], showfliers=False,
            palette=["#4a7fb5", "#2e75b6", "#e6a03c", "#9dc3e6", "#c0504d"])
axes[0].axhline(1, color="k", lw=0.7, ls=":")
axes[0].set_xlabel("")
axes[0].set_ylabel("Índice (1 = laboral típico)")
axes[0].set_title("Llegadas diarias por tipo de día (sin tendencia)")
axes[0].legend(fontsize=8, title="")
dw = ld[ld.servicio != "ambulatorio"].copy()
dw = dw[~dw.festivo]
wk = dw.groupby(["servicio", "dia_semana"]).llegadas.mean().unstack(0)
wk = wk / wk.mean()
for s in wk.columns:
    axes[1].plot(DIAS, wk[s], "-o", color=COLORES[s], label=NOMBRE[s])
axes[1].axhline(1, color="k", lw=0.7, ls=":")
axes[1].set_title("Patrón semanal (sin festivos; 1 = media semanal)")
axes[1].legend(fontsize=8)
fig.suptitle("Efecto del calendario colombiano en las llegadas (periodo estable)", fontweight="bold")
fig_md(guardar(fig, FIG / "05_efecto_calendario.png"), "calendario")

# Festivos individuales: índice en urgencias/observación
fest = ld[(ld.servicio == "urgencias_observacion") & ld.festivo].groupby("nombre_festivo").indice.agg(["median", "size"]).sort_values("median")
fest.to_csv(TAB / "indice_por_festivo_urgencias.csv")
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(fest.index, fest["median"], color=np.where(fest["median"] < 0.8, "#c0504d", "#e6a03c"))
ax.axvline(1, color="k", ls=":", lw=0.8)
ax.set_xlabel("Índice mediano de llegadas (1 = laboral típico)")
ax.set_title("Urgencias / observación: llegadas en cada festivo")
ax.tick_params(axis="y", labelsize=8)
fig_md(guardar(fig, FIG / "06_festivos_urgencias.png"), "festivos")
c_ = cal.pivot(index="servicio", columns="tipo_dia", values="indice_mediano")
u_, h_ = c_.loc["urgencias_observacion"], c_.loc["hospitalizacion"]
add(
    "\n**Lectura.**\n",
    f"- **Festivo ≈ domingo** en urgencias/observación: índice {u_['festivo']:.2f} frente a "
    f"{u_['fin de semana']:.2f} en fin de semana.\n",
    f"- **El rebote existe, pero es sobre todo un efecto lunes**: el primer día hábil tras el descanso "
    f"concentra la demanda acumulada. En urgencias/observación, lunes normal = {u_['lunes']:.2f} y día después de "
    f"festivo = {u_['día después festivo']:.2f} (con la Ley Emiliani suele ser martes). Ingreso a urgencias se comporta igual. "
    "Los dos efectos son de magnitud parecida: la variable útil es \"primer día hábil tras descanso\".\n",
    "- El tipo de día tiene un efecto **grande** sobre las llegadas a urgencias y ambulatorio y **mediano** "
    "sobre hospitalización (ε² en la tabla). Para el modelo de servicio son variables obligatorias: "
    "`festivo`, `fin_de_semana`, día de la semana y `primer_dia_habil`.\n",
    f"- **Hospitalización baja más que urgencias en festivos** (índice {h_['festivo']:.2f} frente a {u_['festivo']:.2f}) y "
    f"no rebota al día siguiente ({h_['día después festivo']:.2f}). Esto es coherente con que se aplacen los "
    "ingresos programados y, quizá, con menos traslados desde urgencias en festivos.\n",
    "- Los festivos con menos llegadas a urgencias son **Viernes Santo, Año Nuevo y Jueves Santo** "
    "(índice ≈ 0.6–0.7). La mayoría de los festivos trasladados a lunes (Emiliani) quedan en 0.75–0.85.",
)

# =====================================================================================
# 4. Descomposición STL / MSTL
# =====================================================================================
sec("4. Tendencia y estacionalidad (descomposición STL / MSTL)")
od = pd.read_parquet(M4 / "ocupacion_dia.parquet")
od["fecha"] = pd.to_datetime(od.fecha)
od = od[od.en_periodo_estable]
series = {
    "Llegadas diarias · urgencias/observación": ld[ld.servicio == "urgencias_observacion"].set_index("fecha").llegadas.asfreq("D"),
    "Llegadas diarias · hospitalización": ld[ld.servicio == "hospitalizacion"].set_index("fecha").llegadas.asfreq("D"),
    "Censo medio diario · hospitalización": od[od.servicio == "hospitalizacion"].set_index("fecha").censo_prom.asfreq("D"),
}
fig, axes = plt.subplots(4, 3, figsize=(17, 10), sharex="col")
resumen = []
for c, (nombre, y) in enumerate(series.items()):
    y = y.interpolate()
    res = MSTL(y, periods=(7, 365), stl_kwargs={"robust": True}).fit()
    comp = [("Observado", res.observed), ("Tendencia", res.trend), ("Estacional semanal", res.seasonal["seasonal_7"]),
            ("Estacional anual", res.seasonal["seasonal_365"])]
    for r, (t, v) in enumerate(comp):
        axes[r, c].plot(v.index, v.values, lw=0.8 if r != 1 else 1.6, color="#1f4e79" if r != 1 else "#c0504d")
        axes[r, c].set_ylabel(t if c == 0 else "")
    axes[0, c].set_title(nombre, fontsize=10)
    tr = res.trend
    var_res = np.var(res.resid)
    resumen.append({
        "serie": nombre,
        "tendencia inicio": tr.iloc[:30].mean(), "tendencia fin": tr.iloc[-30:].mean(),
        "cambio %": 100 * (tr.iloc[-30:].mean() / tr.iloc[:30].mean() - 1),
        "amplitud semanal (máx−mín)": np.ptp(res.seasonal["seasonal_7"].iloc[-28:]),
        "amplitud anual (máx−mín)": np.ptp(res.seasonal["seasonal_365"]),
        "fuerza estacional semanal": max(0, 1 - var_res / np.var(res.resid + res.seasonal["seasonal_7"])),
        "fuerza estacional anual": max(0, 1 - var_res / np.var(res.resid + res.seasonal["seasonal_365"])),
        "fuerza de tendencia": max(0, 1 - var_res / np.var(res.resid + tr)),
    })
for ax in axes[-1]:
    ax.tick_params(axis="x", rotation=30)
fig.suptitle("Descomposición MSTL (periodos 7 y 365 días, robusta) — periodo estable", fontweight="bold")
fig_md(guardar(fig, FIG / "07_mstl.png"), "MSTL")
resumen = pd.DataFrame(resumen)
resumen.to_csv(TAB / "mstl_resumen.csv", index=False)
add(
    "Se usa **MSTL** (STL con varias estacionalidades: semanal y anual) con ajuste robusto, para que "
    "festivos y picos no deformen la tendencia. La *fuerza* (Hyndman: 1 − Var(resto)/Var(resto + "
    "componente)) va de 0 (sin efecto) a 1 (domina la serie).\n",
    md_tabla(resumen, floatfmt=".2f"),
)
# Estacionalidad mensual (perfil por año) urgencias
mm = ld[ld.servicio.isin(["urgencias_observacion", "hospitalizacion"])].copy()
mm["anio"] = mm.fecha.dt.year
mm["mes"] = mm.fecha.dt.month
prof = mm.groupby(["servicio", "anio", "mes"]).indice.median().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, s in zip(axes, SERV2[::-1]):
    for a, d in prof[prof.servicio == s].groupby("anio"):
        ax.plot(d.mes, d.indice, "-o", ms=3, label=str(a))
    ax.axhline(1, color="k", ls=":", lw=0.7)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_title(NOMBRE[s])
axes[0].set_ylabel("Índice mediano de llegadas")
axes[0].legend(fontsize=8)
fig.suptitle("Perfil mensual por año (índice sin tendencia local)", fontweight="bold")
fig_md(guardar(fig, FIG / "08_perfil_mensual.png"), "mensual")
r = resumen.set_index("serie")
add(
    "\n**Lectura.**\n",
    f"- **Censo hospitalario**: la tendencia sube un {r.iloc[2]['cambio %']:.0f} % entre el inicio y el final del "
    "periodo estable. El censo de un año no es comparable con el de otro sin desestacionalizar, y la "
    "capacidad debe ser móvil (como ya hace el proxy de Persona 1).\n",
    f"- **Llegadas a urgencias/observación**: la tendencia **baja** ({r.iloc[0]['cambio %']:.0f} %). Esto "
    "coincide con el cambio de registro visto en M2/M3 (menos subprocesos, más largos), así que no "
    "necesariamente hay menos pacientes. El ciclo semanal es la componente estacional más fuerte.\n",
    "- **Estacionalidad anual**: el perfil mensual (índice sin tendencia local) **no se repite de forma "
    "consistente entre años**. Lo único estable son las caídas de **Semana Santa** (marzo en 2024, abril en "
    "2023 y 2025) y, en hospitalización, de julio–agosto y diciembre (vacaciones, menos cirugía "
    "programada). No se ve un pico respiratorio claro. La componente anual de MSTL absorbe parte del "
    "cambio de registro, así que su \"fuerza\" está sobreestimada. Con ~3 años, conviene modelar la "
    "estacionalidad con `mes`/Semana Santa o términos de Fourier, no con un rezago de 365 días.",
)

# =====================================================================================
# 5. Turnos
# =====================================================================================
sec("5. Turno operativo (mañana 7–13, tarde 13–19, noche 19–7)")
tur = C.sql(
    """
    SELECT o.servicio, h.turno, h.fin_de_semana OR h.festivo no_habil,
           avg(o.llegadas) llegadas_h, avg(o.ocupacion) ocupacion, avg((o.ocupacion >= o.umbral_ocupacion)::int) frac_sat,
           count(DISTINCT h.fecha) dias
    FROM 'data/m1/ocupacion_servicio_hora.parquet' o JOIN 'data/m4/calendario_hora.parquet' h USING (hora)
    WHERE o.en_periodo_estable GROUP BY ALL ORDER BY ALL
    """
).df()
tur["turno"] = tur.turno.replace({"manana": "mañana"})
tur.to_csv(TAB / "turnos.csv", index=False)
# Horas saturadas esperadas por turno (de 6 u 12 h) → insumo del presupuesto de alertas
tur["horas_turno"] = np.where(tur.turno == "noche", 12, 6)
tur["horas_saturadas_por_turno"] = tur.frac_sat * tur.horas_turno
add(md_tabla(tur.drop(columns="dias"), floatfmt=".3f"))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, s in zip(axes, SERV2):
    d = tur[tur.servicio == s].pivot(index="turno", columns="no_habil", values="frac_sat").reindex(["mañana", "tarde", "noche"])
    d.columns = ["hábil", "fin de semana / festivo"]
    d.plot.bar(ax=ax, rot=0, color=["#4a7fb5", "#e6a03c"])
    ax.set_title(NOMBRE[s])
    ax.set_ylabel("Fracción de horas saturadas")
    ax.set_ylim(0, 1)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", fontsize=8)
fig.suptitle("Saturación por turno y tipo de día", fontweight="bold")
fig_md(guardar(fig, FIG / "09_turnos.png"), "turnos")
add(
    "\n**Lectura.** La saturación se concentra en un turno distinto en cada servicio: la **mañana** "
    "en hospitalización y la **tarde** en urgencias/observación (en días hábiles). Para el dashboard "
    "(M8) y el presupuesto de fatiga (máx. 2 falsas alarmas por turno, M7), el umbral de decisión "
    "debería fijarse **por servicio y turno**, porque la tasa base de saturación cambia mucho entre turnos.",
)

# =====================================================================================
# 6. Calendario de calor diario (urgencias)
# =====================================================================================
sec("6. Vista calendario de las llegadas diarias")
u = ld[ld.servicio == "urgencias_observacion"].copy()
u["anio"] = u.fecha.dt.isocalendar().year
u["semana"] = u.fecha.dt.isocalendar().week
anios = sorted(u.anio.unique())
fig, axes = plt.subplots(len(anios), 1, figsize=(15, 1.6 * len(anios)), squeeze=False)
vmin, vmax = u.indice.quantile([0.02, 0.98])
for ax, a in zip(axes[:, 0], anios):
    d = u[u.anio == a].pivot(index="dia_semana", columns="semana", values="indice").reindex(index=range(1, 8), columns=range(1, 54))
    sns.heatmap(d, ax=ax, cmap="RdBu_r", center=1, vmin=vmin, vmax=vmax, cbar=False, linewidths=0.3, linecolor="white")
    fx = u[(u.anio == a) & u.festivo]
    ax.scatter(fx.semana - 0.5, fx.dia_semana - 0.5, marker="x", color="k", s=12, lw=0.8)
    ax.set_yticks(np.arange(7) + 0.5)
    ax.set_yticklabels(DIAS, rotation=0, fontsize=7)
    ax.set_ylabel(str(a), rotation=0, labelpad=18, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=6)
fig.suptitle("Urgencias/observación: índice diario de llegadas (azul = bajo, rojo = alto; × = festivo)", fontweight="bold")
fig_md(guardar(fig, FIG / "10_calendario_urgencias.png"), "calendario urgencias")

# =====================================================================================
cab = [
    "# M4 · EDA espacio-temporal y dinámica de ocupación — parte visual (HUN-203)",
    "",
    "Persona 3 · generado por `m4_visual/run_m4_visual.py` sobre las tablas `data/m4/*` de Persona 1 · "
    "periodo estable. Los HTML interactivos (`html/`) son para el dashboard (M8). Para verlos hace "
    "falta conexión, porque cargan plotly.js desde CDN.",
    "",
    "**Resumen ejecutivo**",
    "",
    "- Las llegadas siguen un ciclo diario y semanal muy fuerte (pico entre semana a las 10–11 h). La saturación de urgencias llega por la tarde y la de hospitalización por la mañana.",
    "- El 65 % de las visitas son ambulatorias aisladas. Hospitalización se alimenta casi solo de observación de urgencias (≈ 6 % de conversión).",
    "- Festivo ≈ domingo. El \"rebote\" tras festivo es sobre todo un efecto de primer día hábil (los lunes normales rebotan igual). La columna `dia_despues_festivo` de Persona 1 incluye los lunes normales y hay que corregirla.",
    "- Hospitalización satura de madrugada y por la mañana, con el máximo los lunes (pocas altas en fin de semana).",
    "- El censo hospitalario tiene una tendencia creciente. Las llegadas a urgencias muestran una tendencia a la baja que coincide con un cambio de registro.",
    "- Los umbrales de alerta deberían fijarse por servicio y turno.",
]
(OUT / "m4_visual" / "M4_visual_informe.md").write_text("\n".join(cab + informe) + "\n", encoding="utf-8")
print("OK → outputs/m4_visual/M4_visual_informe.md")
