"""M2 · EDA univariado, calidad y censura (HUN-201) — Persona 3.

Uso:  python m2_eda/run_m2.py
Salida: outputs/m2/{figuras,tablas}/ y outputs/m2/M2_informe.md (solo agregados).

Periodo: `en_periodo_estable = true` (descriptivo sobre todo el periodo estable,
entrenamiento + prueba). La auditoría de truncamiento usa todos los episodios.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402
from lifelines.statistics import multivariate_logrank_test  # noqa: E402
from lifelines.utils import median_survival_times  # noqa: E402

from eda_comun import (  # noqa: E402
    COLORES, GRUPOS_EDAD, M1, NOMBRE, OUT, bh, carpeta, con, cramers_v, fmt_p, grupo_edad_sql,
    guardar, kruskal_eps2, magnitud_eps2, magnitud_v, md_tabla, parametros,
)

FIG, TAB = carpeta("m2")
C = con()
P = parametros()
CORTE = pd.Timestamp(P.fecha_corte_datos)
SERV = ["hospitalizacion", "urgencias_observacion"]
XMAX = {"hospitalizacion": 600, "urgencias_observacion": 48}
informe: list[str] = []


def sec(titulo: str):
    print(f"· {titulo}")
    informe.append(f"\n## {titulo}\n")


def add(*lineas: str):
    informe.extend(lineas)


def fig_md(nombre: str, alt: str):
    add(f"\n![{alt}](figuras/{nombre})\n")


# =====================================================================================
# 1. Calidad de datos
# =====================================================================================
sec("1. Calidad de los datos (auditoría de inconsistencias)")
cal = pd.read_parquet(M1 / "reporte_calidad.parquet")
cal.to_csv(TAB / "calidad.csv", index=False)
add(
    "Chequeos cuantificados por Persona 1 en el M1 (`reporte_calidad`), incluidos los que el plan "
    "pedía explícitamente (`visit_end < visit_start`, `year_of_birth` anómalo).\n",
    md_tabla(cal.rename(columns={"pct": "%"}).astype({"n_afectados": "Int64", "n_total": "Int64"}), floatfmt=".2f"),
)
cc = cal.dropna(subset=["pct"]).copy()
cc["etiqueta"] = cc.tabla + " · " + cc.chequeo.str.slice(0, 60)
cc = cc.sort_values("pct")
fig, ax = plt.subplots(figsize=(9, 6))
color = np.where(cc.pct >= 20, "#c0504d", np.where(cc.pct >= 1, "#e6a03c", "#5b9b64"))
ax.barh(cc.etiqueta, cc.pct, color=color)
for y, v in enumerate(cc.pct):
    ax.text(v + 0.8, y, f"{v:.2f} %", va="center", fontsize=8)
ax.set_xlabel("% de registros afectados")
ax.set_title("Auditoría de calidad por tabla (rojo ≥ 20 %, naranja ≥ 1 %)")
ax.set_xlim(0, 112)
fig_md(guardar(fig, FIG / "01_calidad.png"), "calidad")
add(
    "**Lectura.** Las inconsistencias clásicas son marginales (`visit_detail` con fin < inicio = 0; "
    "`year_of_birth < 1900` = 87 personas, 0.01 %). Los problemas graves son estructurales: "
    "la estancia en `visit_occurrence` es siempre 0 (se usa `visit_detail`), ~29 % de los "
    "subprocesos y ~19–43 % de diagnósticos/mediciones están vinculados a la visita de otra persona "
    "(se descartan con `vinculo_coherente`), y 38–54 % de conceptos son 0 (sin vocabulario).",
)

# =====================================================================================
# 2. Demografía
# =====================================================================================
sec("2. Caracterización demográfica")
per = C.sql(
    """SELECT CASE gender_concept_id WHEN 8507 THEN 'M' WHEN 8532 THEN 'F' ELSE 'desconocido' END sexo,
              year_of_birth FROM 'data/raw/person.parquet'"""
).df()
n_per = len(per)
anom_yob = int(((per.year_of_birth < 1900) | (per.year_of_birth > 2026)).sum())
sexo_per = per.sexo.value_counts(normalize=True).mul(100).round(2)
add(
    f"**Tabla `person`** ({n_per:,} personas): F {sexo_per.get('F', 0):.1f} %, "
    f"M {sexo_per.get('M', 0):.1f} %, desconocido {sexo_per.get('desconocido', 0):.2f} %. "
    f"Año de nacimiento fuera de [1900, 2026]: {anom_yob} personas.\n"
)

ep = C.sql(
    f"""SELECT servicio, person_id, edad, sexo, origen_urgencias, los_h, target_los_largo,
               es_entrenamiento, inicio, fin, {grupo_edad_sql()} grupo_edad
        FROM 'data/m1/episodio.parquet' WHERE en_periodo_estable"""
).df()

filas = []
for s, d in ep.groupby("servicio"):
    filas.append(
        {
            "servicio": NOMBRE[s],
            "episodios": len(d),
            "personas": d.person_id.nunique(),
            "episodios/persona": len(d) / d.person_id.nunique(),
            "edad mediana [RIC]": f"{d.edad.median():.0f} [{d.edad.quantile(.25):.0f}–{d.edad.quantile(.75):.0f}]",
            "% mujeres": 100 * (d.sexo == "F").mean(),
            "% < 15 años": 100 * (d.edad < 15).mean(),
            "% ≥ 65 años": 100 * (d.edad >= 65).mean(),
            "% vienen de urgencias": 100 * d.origen_urgencias.mean(),
        }
    )
demo = pd.DataFrame(filas)
demo.to_csv(TAB / "demografia_servicio.csv", index=False)
add("**Episodios del periodo estable** (2023-01-31 → 2026-04-01):\n", md_tabla(demo, floatfmt=".1f"))

# Pirámide de edad (personas únicas, primer episodio por servicio)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
bins = np.arange(0, 110, 5)
for ax, s in zip(axes, SERV):
    d = ep[ep.servicio == s].sort_values("inicio").drop_duplicates("person_id")
    hm = np.histogram(d.loc[d.sexo == "M", "edad"], bins)[0] / len(d) * 100
    hf = np.histogram(d.loc[d.sexo == "F", "edad"], bins)[0] / len(d) * 100
    ax.barh(bins[:-1], -hm, height=4.6, color="#4a7fb5", label="Hombres")
    ax.barh(bins[:-1], hf, height=4.6, color="#d07a8a", label="Mujeres")
    lim = max(hm.max(), hf.max()) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([f"{abs(t):.0f}" for t in ax.get_xticks()])
    ax.set_title(f"{NOMBRE[s]} (n = {len(d):,} personas)")
    ax.set_xlabel("% de personas")
    ax.axvline(0, color="k", lw=0.6)
axes[0].set_ylabel("Edad (años)")
axes[0].legend(loc="upper left")
fig.suptitle("Pirámide de edad por servicio (periodo estable, primer episodio por persona)", fontweight="bold")
fig_md(guardar(fig, FIG / "02_piramide_edad.png"), "pirámide")
add(
    "**Lectura.**\n",
    "- Ambos servicios tienen un **pico en 0–4 años** y un **pico de mujeres de 15–35 años**. En "
    "hospitalización ese pico femenino es mucho más marcado que en hombres, lo que sugiere un componente "
    "**obstétrico** importante (partos/cesáreas, y recién nacidos en 0–4). No hay códigos de servicio "
    "para confirmarlo; si Persona 2 lo necesita, se puede aproximar con los diagnósticos.\n",
    "- Hospitalización tiene más adultos mayores (21 % ≥ 65 frente a 13 % en urgencias) y el 97 % de "
    "sus episodios viene de urgencias.\n",
    "- Sexo desconocido es despreciable (< 0.01 %).",
)

# =====================================================================================
# 3. Estancia (LOS) univariada
# =====================================================================================
sec("3. Distribución univariada de la estancia (LOS)")
umb = pd.read_parquet(M1 / "umbral_los.parquet")
umbral = dict(zip(umb.servicio, umb.umbral_los_h))
qs = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
filas = []
for s, d in ep.groupby("servicio"):
    x = d.los_h
    fila = {"servicio": NOMBRE[s], "n": len(x), "media_h": x.mean()}
    fila.update({f"p{int(q*100)}_h": x.quantile(q) for q in qs})
    fila.update({"% LOS = 0": 100 * (x == 0).mean(), "asimetría": x.skew(), "umbral P75 (train)": umbral[s]})
    filas.append(fila)
los_tab = pd.DataFrame(filas)
los_tab.to_csv(TAB / "los_cuantiles.csv", index=False)
add(md_tabla(los_tab, floatfmt=".2f"))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, s in zip(axes, SERV):
    x = ep.loc[ep.servicio == s, "los_h"]
    pos = x[x > 0]
    ax.hist(np.log10(pos), bins=80, color=COLORES[s], alpha=0.85)
    ax.axvline(np.log10(umbral[s]), color="k", ls="--", lw=1)
    ax.text(np.log10(umbral[s]), ax.get_ylim()[1] * 0.92, f" P75 = {umbral[s]:.1f} h", fontsize=8)
    ax.axvline(np.log10(x.median() if x.median() > 0 else 0.01), color="grey", ls=":", lw=1)
    ticks = [-2, -1, 0, 1, 2, 3, 4]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["36 s", "6 min", "1 h", "10 h", "100 h", "1000 h", "10⁴ h"])
    ax.set_title(f"{NOMBRE[s]} — LOS > 0 (excluye {100*(x==0).mean():.1f} % con LOS = 0)")
    ax.set_xlabel("Estancia (escala log)")
    ax.set_ylabel("Episodios")
fig.suptitle("Distribución de la estancia por servicio (periodo estable)", fontweight="bold")
fig_md(guardar(fig, FIG / "03_los_hist.png"), "LOS")
h = los_tab.set_index("servicio")
add(
    f"**Lectura.** Ambas distribuciones son muy asimétricas a la derecha (asimetría "
    f"{h.loc[NOMBRE['hospitalizacion'], 'asimetría']:.0f} y {h.loc[NOMBRE['urgencias_observacion'], 'asimetría']:.0f}): "
    "la media sobreestima la estancia típica y no debe usarse como resumen. En urgencias/observación "
    f"el {h.loc[NOMBRE['urgencias_observacion'], '% LOS = 0']:.0f} % de los episodios dura 0 h (registro puntual) y la "
    "distribución es bimodal (≈ minutos vs. ≈ horas); en hospitalización la moda está en 2–4 días "
    "con cola larga de estancias de semanas.",
)

# =====================================================================================
# 4. Censura y truncamiento
# =====================================================================================
sec("4. Censura y truncamiento al final de los datos")
todo = C.sql(
    """SELECT servicio, date_trunc('month', inicio) mes, los_h, fin, inicio, en_periodo_estable,
              en_riesgo_truncamiento FROM 'data/m1/episodio.parquet'"""
).df()
todo["abierto_en_corte"] = todo.fin >= CORTE - pd.Timedelta(hours=1)
resumen_cens = (
    todo.groupby("servicio")
    .agg(
        episodios=("los_h", "size"),
        en_riesgo_truncamiento=("en_riesgo_truncamiento", "sum"),
        abiertos_en_corte=("abierto_en_corte", "sum"),
    )
    .reset_index()
)
est = todo[todo.en_periodo_estable]
resumen_cens["abiertos_en_corte_periodo_estable"] = est.groupby("servicio").abierto_en_corte.sum().values
resumen_cens["truncamiento_periodo_estable"] = est.groupby("servicio").en_riesgo_truncamiento.sum().values
# margen: última admisión del periodo estable vs corte
margen_h = (CORTE - pd.Timestamp(P.fin_periodo_estable)).total_seconds() / 3600
resumen_cens["margen_h_hasta_corte"] = margen_h
resumen_cens["% LOS > margen (prob. de truncar)"] = [
    100 * (est.loc[est.servicio == s, "los_h"] > margen_h).mean() for s in resumen_cens.servicio
]
resumen_cens.to_csv(TAB / "censura_resumen.csv", index=False)
add(
    "No hay visitas abiertas (`visit_end` nunca es NULL), así que la censura clásica del plan no "
    "existe. Hay dos fenómenos reales:\n",
    "1. **Estancias abiertas en la extracción** (censura por la derecha): episodios cuyo `fin` coincide "
    f"con la fecha de corte ({CORTE:%Y-%m-%d %H:%M}) ±1 h. Seguían en curso; su LOS es un mínimo.\n",
    "2. **Truncamiento**: los ingresos de los últimos 30 días (`en_riesgo_truncamiento`) solo pueden "
    "aparecer si ya terminaron, así que están sesgados hacia estancias cortas.\n",
    md_tabla(resumen_cens, floatfmt=".3f"),
    f"\nEl periodo estable termina {margen_h/24:.0f} días antes del corte: un ingreso del último día del "
    "periodo solo quedaría truncado si su estancia supera ese margen, cosa que ocurre en "
    f"{resumen_cens.iloc[0]['% LOS > margen (prob. de truncar)']:.3f} % (hosp.) y "
    f"{resumen_cens.iloc[1]['% LOS > margen (prob. de truncar)']:.3f} % (urg.) de los episodios. "
    "**Conclusión: dentro del periodo estable el sesgo por censura/truncamiento es despreciable.**",
)

# Evolución mensual de la mediana y P75 de LOS por mes de ingreso
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
cero_mes = todo[todo.servicio == "urgencias_observacion"].groupby("mes").los_h.apply(lambda x: 100 * (x == 0).mean())
ax = axes[2]
ax.plot(cero_mes.index, cero_mes.values, "-o", ms=3, color=COLORES["urgencias_observacion"])
ax.axvspan(pd.Timestamp(P.inicio_periodo_estable), pd.Timestamp(P.fin_periodo_estable), color="green", alpha=0.06)
ax.set_title("Urgencias / observación — % de episodios con LOS = 0")
ax.set_ylabel("%")
ax.tick_params(axis="x", rotation=30)
mes_tab = []
for ax, s in zip(axes, SERV):
    m = todo[todo.servicio == s].groupby("mes").los_h.agg(
        p50="median", p75=lambda x: x.quantile(0.75), n="size"
    )
    mes_tab.append(m.assign(servicio=s).reset_index())
    ax.plot(m.index, m.p50, "-o", ms=3, color=COLORES[s], label="Mediana")
    ax.plot(m.index, m.p75, "-s", ms=3, color=COLORES[s], alpha=0.5, label="P75")
    ax.axvspan(pd.Timestamp(P.inicio_periodo_estable), pd.Timestamp(P.fin_periodo_estable), color="green", alpha=0.06, label="Periodo estable")
    ax.axvspan(CORTE - pd.Timedelta(days=30), CORTE, color="red", alpha=0.12, label="Riesgo de truncamiento")
    ax.axhline(umbral[s], color="k", ls="--", lw=0.8)
    ax.set_title(NOMBRE[s])
    ax.set_ylabel("Horas")
    ax.tick_params(axis="x", rotation=30)
axes[0].legend(fontsize=8)
fig.suptitle("Estancia por mes de ingreso (todos los episodios) — línea punteada = umbral P75", fontweight="bold")
fig_md(guardar(fig, FIG / "04_los_por_mes.png"), "LOS por mes")
mes_tab = pd.concat(mes_tab)
mes_tab.to_csv(TAB / "los_por_mes.csv", index=False)
anual = (
    todo[todo.en_periodo_estable].assign(anio=lambda d: d.inicio.dt.year)
    .groupby(["servicio", "anio"]).los_h.agg(mediana="median", p75=lambda x: x.quantile(0.75),
                                             pct_cero=lambda x: 100 * (x == 0).mean(), n="size").reset_index()
)
anual["% > umbral P75 de train"] = [
    100 * (todo[(todo.servicio == s) & todo.en_periodo_estable & (todo.inicio.dt.year == a)].los_h > umbral[s]).mean()
    for s, a in zip(anual.servicio, anual.anio)
]
anual.to_csv(TAB / "los_por_anio.csv", index=False)
add(
    "\n**Deriva de la estancia dentro del periodo estable** (por año de ingreso):\n",
    md_tabla(anual, floatfmt=".2f"),
    "\n**Lectura de la figura.**\n",
    "- **Truncamiento visible al final**: en julio de 2026 la mediana y el P75 de hospitalización caen "
    "bruscamente (solo aparecen las estancias que ya terminaron). Ese mes está fuera del periodo estable.\n",
    "- **La estancia crece con el tiempo**: la mediana de hospitalización pasa de ~68 h (2023) a ~75 h "
    "(2025–26) y el P75 mensual queda casi siempre **por encima del umbral fijo** de entrenamiento "
    f"({umbral['hospitalizacion']:.1f} h) desde 2024-07. Por eso la tasa de positivos en prueba (0.28) es "
    "mayor que en entrenamiento (0.25). Es deriva real del target, y Persona 2 debe tenerla en cuenta "
    "al calibrar (M7).\n",
    "- **Cambio de registro en urgencias/observación**: el % de episodios con LOS = 0 cambia por "
    "tramos (cae entre 2024-08 y 2025-04 y otra vez desde 2025-12), y con él la mediana salta de 0 a "
    "1–2 h. Parece un cambio en cómo se registra, no en la atención. Hay que reportarlo a Persona 1 y "
    "considerar una variable de \"régimen\" o validar por periodos.",
)


def km(dur, ev, label=None):
    k = KaplanMeierFitter()
    k.fit(np.round(np.asarray(dur, float), 2), np.asarray(ev, int), label=label)
    return k


def mediana_ic(k: KaplanMeierFitter) -> tuple[float, float, float]:
    ci = median_survival_times(k.confidence_interval_)
    return float(k.median_survival_time_), float(ci.iloc[0, 0]), float(ci.iloc[0, 1])


# Sensibilidad: 3 tratamientos de los episodios recientes (todos los datos)
filas = []
for s in SERV:
    d = todo[todo.servicio == s]
    escenarios = {
        "A. Ingenuo (todos como evento)": (d.los_h, np.ones(len(d))),
        "B. Abiertos en corte = censurados": (d.los_h, (~d.abierto_en_corte).astype(int)),
        "C. Excluir riesgo de truncamiento": (d.loc[~d.en_riesgo_truncamiento, "los_h"], (~d.loc[~d.en_riesgo_truncamiento, "abierto_en_corte"]).astype(int)),
        "D. Solo periodo estable (recomendado)": (d.loc[d.en_periodo_estable, "los_h"], (~d.loc[d.en_periodo_estable, "abierto_en_corte"]).astype(int)),
    }
    for nombre, (dur, ev) in escenarios.items():
        k = km(dur, ev)
        med, lo, hi = mediana_ic(k)
        filas.append(
            {
                "servicio": NOMBRE[s],
                "escenario": nombre,
                "n": len(dur),
                "censurados": int((np.asarray(ev) == 0).sum()),
                "mediana_KM_h": med,
                "IC95": f"[{lo:.2f}, {hi:.2f}]",
                "S(umbral P75)": float(k.survival_function_at_times(umbral[s]).iloc[0]),
                "media_simple_h": float(np.mean(dur)),
            }
        )
sens = pd.DataFrame(filas)
sens.to_csv(TAB / "km_sensibilidad_censura.csv", index=False)
add(
    "\n**Sensibilidad de Kaplan-Meier al tratamiento de la censura** (S(umbral) = fracción que supera el P75):\n",
    md_tabla(sens, floatfmt=".3f"),
    "\nLas medianas KM y la fracción por encima del umbral apenas cambian entre escenarios: la "
    "censura afecta a < 0.5 % de los episodios. En el resto del análisis se usa el escenario D "
    "(periodo estable, abiertos en corte como censurados).",
)

# =====================================================================================
# 5. Kaplan-Meier estratificado
# =====================================================================================
sec("5. Curvas de Kaplan-Meier estratificadas (sustituto del triaje)")
add(
    "El plan pedía estratificar por nivel de triaje (1–5), pero **el triaje no existe** en los datos. "
    "Se estratifica por servicio, origen en urgencias (solo hospitalización: en urgencias/observación "
    "es siempre falso), grupo de edad y sexo. Evento = alta del servicio. Se reporta el log-rank "
    "(p ajustado por BH) **y** el tamaño de efecto ε² de Kruskal-Wallis sobre LOS, porque con n "
    "grande todo log-rank es significativo.\n"
)
ep["evento"] = (ep.fin < CORTE - pd.Timedelta(hours=1)).astype(int)
ep["origen"] = np.where(ep.origen_urgencias, "Desde urgencias", "Ingreso directo")
estratos = {
    "hospitalizacion": [("origen", ["Desde urgencias", "Ingreso directo"]), ("grupo_edad", GRUPOS_EDAD), ("sexo", ["F", "M"])],
    "urgencias_observacion": [("grupo_edad", GRUPOS_EDAD), ("sexo", ["F", "M"])],
}
paleta = plt.get_cmap("viridis")

# 5a. por servicio (dos escalas)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, s in zip(axes, SERV):
    d = ep[ep.servicio == s]
    k = km(d.los_h, d.evento, NOMBRE[s])
    k.plot_survival_function(ax=ax, color=COLORES[s], ci_show=True)
    ax.axvline(umbral[s], color="k", ls="--", lw=0.8)
    ax.axhline(0.25, color="grey", ls=":", lw=0.8)
    ax.set_xlim(0, XMAX[s])
    ax.set_xlabel("Horas desde el ingreso al servicio")
    ax.set_ylabel("P(seguir en el servicio)")
    med, lo, hi = mediana_ic(k)
    ax.set_title(f"{NOMBRE[s]} — mediana {med:.1f} h [{lo:.1f}–{hi:.1f}]")
fig.suptitle("Kaplan-Meier de la estancia por servicio (línea vertical = umbral P75)", fontweight="bold")
fig_md(guardar(fig, FIG / "05_km_servicio.png"), "KM servicio")

filas = []
for s in SERV:
    d = ep[ep.servicio == s]
    n_est = len(estratos[s])
    fig, axes = plt.subplots(1, n_est, figsize=(5 * n_est, 4), sharey=True)
    for ax, (var, niveles) in zip(np.atleast_1d(axes), estratos[s]):
        dd = d[d[var].isin(niveles)]
        for i, nv in enumerate(niveles):
            g = dd[dd[var] == nv]
            if len(g) < 30:
                continue
            k = km(g.los_h, g.evento, f"{nv} (n={len(g):,})")
            k.plot_survival_function(ax=ax, ci_show=False, color=paleta(i / max(1, len(niveles) - 1) * 0.9))
            med, lo, hi = mediana_ic(k)
            filas.append(
                {"servicio": NOMBRE[s], "variable": var, "nivel": nv, "n": len(g), "mediana_h": med,
                 "IC95_inf": lo, "IC95_sup": hi, "P(LOS > umbral)": float(k.survival_function_at_times(umbral[s]).iloc[0])}
            )
        lr = multivariate_logrank_test(np.round(dd.los_h, 2), dd[var], dd.evento)
        _, _, e2 = kruskal_eps2([dd.loc[dd[var] == nv, "los_h"].values for nv in niveles])
        filas.append({"servicio": NOMBRE[s], "variable": var, "nivel": "__test__", "n": len(dd),
                      "logrank_chi2": lr.test_statistic, "p": lr.p_value, "eps2": e2})
        ax.axvline(umbral[s], color="k", ls="--", lw=0.8)
        ax.set_xlim(0, XMAX[s])
        ax.set_title(f"por {var}")
        ax.set_xlabel("Horas")
        ax.legend(fontsize=7.5)
    np.atleast_1d(axes)[0].set_ylabel("P(seguir en el servicio)")
    fig.suptitle(f"Kaplan-Meier estratificado — {NOMBRE[s]}", fontweight="bold")
    fig_md(guardar(fig, FIG / f"06_km_estratos_{s}.png"), f"KM {s}")

km_tab = pd.DataFrame(filas)
tests = km_tab[km_tab.nivel == "__test__"].copy()
tests["p_BH"] = bh(tests.p)
tests["magnitud ε²"] = tests.eps2.map(magnitud_eps2)
tests["p_BH"] = tests.p_BH.map(fmt_p)
niveles_tab = km_tab[km_tab.nivel != "__test__"].dropna(axis=1, how="all")
niveles_tab.to_csv(TAB / "km_medianas_estrato.csv", index=False)
tests.drop(columns=["nivel", "p"]).dropna(axis=1, how="all").to_csv(TAB / "km_logrank.csv", index=False)
add(
    "**Medianas KM por estrato** (horas, IC 95 %):\n",
    md_tabla(niveles_tab, floatfmt=".2f"),
    "\n**Log-rank + tamaño de efecto:**\n",
    md_tabla(tests[["servicio", "variable", "n", "logrank_chi2", "p_BH", "eps2", "magnitud ε²"]], floatfmt=".4f"),
)
add(
    "\n**Lectura.** Todos los log-rank son significativos (n enorme), pero el tamaño de efecto dice "
    "cuáles importan: la **edad** es el estratificador con más efecto en ambos servicios (en "
    "hospitalización, los mayores de 65 tienen curvas claramente más lentas); sexo y origen tienen "
    "efecto despreciable o pequeño. Ningún estrato por sí solo separa bien quién supera el P75: el "
    "modelo de paciente necesita variables clínicas e historial (M5).",
)

# =====================================================================================
# 6. Balance de clases
# =====================================================================================
sec("6. Balance de clases de los dos targets")
snap = C.sql(
    """SELECT servicio, horas_desde_ingreso, es_entrenamiento, y_target_los_largo,
              n_mediciones_24h, n_mediciones_historia, n_dx_historia, n_dx_distintos_historia, n_dx_30d,
              n_visitas_365d, n_hospitalizaciones_previas, horas_desde_ultima_alta_hosp,
              censo_servicio_t_menos_1h, ocupacion_servicio_t_menos_1h, llegadas_servicio_t_menos_1h, edad
       FROM 'data/m1/snapshot.parquet' WHERE en_periodo_estable"""
).df()
snap["particion"] = np.where(snap.es_entrenamiento, "entrenamiento", "prueba")
bal_snap = (
    snap.groupby(["servicio", "horas_desde_ingreso", "particion"])
    .y_target_los_largo.agg(n="size", positivos="sum", tasa="mean").reset_index()
)
ep["particion"] = np.where(ep.es_entrenamiento, "entrenamiento", "prueba")
bal_ep = ep.groupby(["servicio", "particion"]).target_los_largo.agg(n="size", positivos="sum", tasa="mean").reset_index()
osh = C.sql(
    """SELECT servicio, es_entrenamiento, count(*) n, sum(y_target_saturacion) positivos,
              avg(y_target_saturacion) tasa, avg(ocupacion) ocupacion_media
       FROM 'data/m1/ocupacion_servicio_hora.parquet' WHERE en_periodo_estable GROUP BY ALL ORDER BY ALL"""
).df()
osh["particion"] = np.where(osh.es_entrenamiento, "entrenamiento", "prueba")
bal_ep.to_csv(TAB / "balance_episodio.csv", index=False)
bal_snap.to_csv(TAB / "balance_snapshot.csv", index=False)
osh.drop(columns="es_entrenamiento").to_csv(TAB / "balance_saturacion.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
piv = bal_ep.pivot(index="servicio", columns="particion", values="tasa").rename(index=NOMBRE)
piv.plot.bar(ax=axes[0], color=["#4a7fb5", "#e6a03c"], rot=0)
axes[0].set_title("Paciente · episodio (LOS > P75)")
pv = bal_snap.assign(etq=lambda d: d.servicio.map({"hospitalizacion": "Hosp", "urgencias_observacion": "Urg"}) + " τ=" + d.horas_desde_ingreso.astype(str) + "h")
pv.pivot(index="etq", columns="particion", values="tasa").plot.bar(ax=axes[1], color=["#4a7fb5", "#e6a03c"], rot=30)
axes[1].set_title("Paciente · snapshot (por instante τ)")
osh.assign(s=osh.servicio.map(NOMBRE)).pivot(index="s", columns="particion", values="tasa").plot.bar(ax=axes[2], color=["#4a7fb5", "#e6a03c"], rot=0)
axes[2].set_title("Servicio · saturación en t+4 h")
for ax in axes:
    ax.set_ylabel("Tasa de positivos")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7)
fig.suptitle("Balance de clases por partición temporal (periodo estable)", fontweight="bold")
fig_md(guardar(fig, FIG / "07_balance_clases.png"), "balance")
add(
    "**Episodio:**\n", md_tabla(bal_ep, floatfmt=".3f"),
    "\n**Snapshot (unidad del modelo de paciente):**\n", md_tabla(bal_snap, floatfmt=".3f"),
    "\n**Saturación por servicio y hora:**\n", md_tabla(osh.drop(columns="es_entrenamiento"), floatfmt=".3f"),
    "\n**Lectura.**\n",
    "- A nivel episodio el target está en ~26 % por construcción (P75). A nivel **snapshot** la tasa "
    "**sube con τ en urgencias/observación** (≈ 0.52 → 0.59 → 0.76): los snapshots se crean solo si el "
    "paciente sigue en el servicio, así que en τ = 0 ya se excluyen los episodios de LOS 0 y en τ = 6 h "
    "solo quedan los que llevan 6 h. Persona 2 debe estratificar métricas por τ y no comparar AUC-PR "
    "entre τ sin mirar la prevalencia base.\n",
    "- En hospitalización la tasa es estable entre τ (≈ 0.26) porque casi nadie sale antes de 6 h.\n",
    "- La prevalencia de prueba es ~3 puntos mayor que la de entrenamiento en todos los casos del "
    "modelo de paciente (0.25 → 0.28), coherente con la estancia creciente (sección 4). En saturación "
    "no hay diferencia, porque la capacidad proxy es móvil.\n",
    "- El desbalance es moderado (25–40 % positivos en casi todo): no exige sobremuestreo; basta con "
    "`class_weight` o umbral por costo (M7).",
)

# =====================================================================================
# 7. Nulos informativos (MNAR)
# =====================================================================================
sec("7. Faltantes y nulos informativos (MNAR)")
cols = [c for c in snap.columns if c not in ("servicio", "horas_desde_ingreso", "es_entrenamiento", "y_target_los_largo", "particion")]
falt = snap.groupby("servicio")[cols].apply(lambda d: d.isna().mean() * 100).T.round(2)
falt.columns = [NOMBRE[c] for c in falt.columns]
falt.to_csv(TAB / "faltantes_snapshot.csv")
add(
    "**% de NULL por variable del snapshot:**\n",
    md_tabla(falt.reset_index().rename(columns={"index": "variable"}), floatfmt=".2f"),
    "\nLos NULL reales son pocos y **estructurales**: `horas_desde_ultima_alta_hosp` es NULL cuando la "
    "persona nunca estuvo hospitalizada (no es un dato perdido). El resto de variables no tiene NULL "
    "porque los conteos se construyeron con 0 por defecto. El faltante clínicamente relevante no aparece como "
    "NULL sino como **cero**: `n_mediciones_24h = 0` (no se pidió ningún laboratorio).\n",
)
indic = {
    "sin_mediciones_24h": snap.n_mediciones_24h.fillna(0) == 0,
    "sin_mediciones_historia": snap.n_mediciones_historia.fillna(0) == 0,
    "sin_dx_historia": snap.n_dx_historia.fillna(0) == 0,
    "nunca_hospitalizado": snap.horas_desde_ultima_alta_hosp.isna(),
}
filas = []
for (s, tau), d in snap.groupby(["servicio", "horas_desde_ingreso"]):
    for nombre, m in indic.items():
        mm = m.loc[d.index]
        y = d.y_target_los_largo
        tabla = pd.crosstab(mm, y).reindex(index=[False, True], columns=[0, 1], fill_value=0).values
        chi2, p, v, gl = cramers_v(tabla)
        filas.append(
            {"servicio": NOMBRE[s], "τ_h": tau, "indicador": nombre, "% con indicador": 100 * mm.mean(),
             "tasa_target_si": y[mm].mean(), "tasa_target_no": y[~mm].mean(), "chi2": chi2, "p": p, "V_cramer": v}
        )
mnar = pd.DataFrame(filas)
mnar["p_BH"] = bh(mnar.p)
mnar["magnitud"] = mnar.V_cramer.map(magnitud_v)
mnar["p_BH"] = mnar.p_BH.map(fmt_p)
mnar.drop(columns="p").to_csv(TAB / "mnar_indicadores.csv", index=False)
add(
    "**Indicadores de ausencia vs target (Chi² + V de Cramér, BH):**\n",
    md_tabla(mnar.drop(columns=["p", "chi2"]), floatfmt=".3f"),
)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, s in zip(axes, SERV):
    d = mnar[(mnar.servicio == NOMBRE[s]) & (mnar.indicador == "sin_mediciones_24h")]
    x = np.arange(len(d))
    ax.bar(x - 0.2, d.tasa_target_si, 0.4, color="#c0504d", label="Sin laboratorios en 24 h")
    ax.bar(x + 0.2, d.tasa_target_no, 0.4, color="#4a7fb5", label="Con laboratorios")
    for xi, (a, b, pc) in enumerate(zip(d.tasa_target_si, d.tasa_target_no, d["% con indicador"])):
        ax.text(xi, max(a, b) + 0.03, f"{pc:.0f} % sin lab.", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"τ = {t} h" for t in d["τ_h"]])
    ax.set_title(NOMBRE[s])
    ax.set_ylim(0, 1)
axes[0].set_ylabel("P(LOS > P75)")
axes[0].legend()
fig.suptitle("Tasa de estancia larga con y sin laboratorios en las 24 h previas", fontweight="bold")
fig_md(guardar(fig, FIG / "08_mnar_mediciones.png"), "MNAR")
add(
    "\n**Lectura.**\n",
    "- La ausencia **no es aleatoria** (todas las diferencias son significativas tras BH y tienen "
    "dirección consistente entre τ), pero su **efecto marginal es despreciable** (V de Cramér < 0.1). "
    "Por sí sola no predice la estancia larga.\n",
    "- Dirección: no tener laboratorios o no tener historia se asocia con **menos** estancia larga, "
    "salvo `sin_dx_historia` en hospitalización. Que el paciente haya estado hospitalizado antes "
    "(`nunca_hospitalizado = 0`) es el indicador con más efecto (hosp.: 0.33 frente a 0.24).\n",
    "- En urgencias, en τ = 0 el 97 % no tiene laboratorios, y en τ = 6 h baja al 34 %: los "
    "laboratorios se registran durante la estancia. La información MNAR crece con τ.\n",
    "\n**Recomendación para M5.** Incluir los indicadores binarios (`sin_mediciones_24h`, "
    "`nunca_hospitalizado`, `sin_dx_historia`). Son baratos y pueden interactuar con otras variables "
    "en modelos de árboles. **No imputar** `horas_desde_ultima_alta_hosp` con la mediana: usar el "
    "indicador más un valor centinela dentro del `Pipeline`.",
)

# =====================================================================================
# 8. Resumen univariado de variables del snapshot
# =====================================================================================
sec("8. Resumen univariado de las variables de entrada (snapshot τ = 0)")
s0 = snap[snap.horas_desde_ingreso == 0]
filas = []
for s, d in s0.groupby("servicio"):
    for c in cols:
        x = d[c].dropna()
        filas.append(
            {"servicio": NOMBRE[s], "variable": c, "mediana": x.median(), "p25": x.quantile(.25), "p75": x.quantile(.75),
             "p99": x.quantile(.99), "máx": x.max(), "% cero": 100 * (x == 0).mean(), "asimetría": x.skew()}
        )
uni = pd.DataFrame(filas)
uni.to_csv(TAB / "univariado_snapshot_tau0.csv", index=False)
add(md_tabla(uni, floatfmt=".2f"))
num_plot = ["edad", "n_dx_historia", "n_dx_30d", "n_mediciones_historia", "n_visitas_365d", "n_hospitalizaciones_previas", "ocupacion_servicio_t_menos_1h", "llegadas_servicio_t_menos_1h"]
fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
for ax, c in zip(axes.ravel(), num_plot):
    for s in SERV:
        x = s0.loc[s0.servicio == s, c].dropna()
        logx = x.skew() > 3
        v = np.log1p(x) if logx else x
        ax.hist(v, bins=50, density=True, alpha=0.55, color=COLORES[s], label=NOMBRE[s])
    ax.set_title(c + (" (log1p)" if logx else ""), fontsize=9)
axes[0, 0].legend(fontsize=7)
fig.suptitle("Distribuciones de las variables de entrada (τ = 0, periodo estable)", fontweight="bold")
fig_md(guardar(fig, FIG / "09_univariado_variables.png"), "univariado")
add(
    "\n**Lectura.** Las variables de conteo (diagnósticos, mediciones, visitas) son muy asimétricas y "
    "con muchos ceros: conviene `log1p` o modelos de árboles. La ocupación del servicio en t−1 está "
    "concentrada cerca de 1 en hospitalización (el servicio opera casi siempre lleno respecto al proxy).",
)

# =====================================================================================
cab = [
    "# M2 · EDA univariado, calidad y censura (HUN-201)",
    "",
    "Persona 3 · generado por `m2_eda/run_m2.py` · datos: `data/m1/*`, `data/raw/person` · "
    "periodo estable 2023-01-31 → 2026-04-01 salvo donde se indica. Solo agregados.",
    "",
    "**Resumen ejecutivo**",
    "",
    "- Calidad: inconsistencias clásicas despreciables; problemas graves y ya mitigados en M1 (vínculo roto, LOS de visita = 0).",
    "- La estancia es muy asimétrica: se resume con mediana/percentiles y curvas KM, no con la media.",
    "- **No hay triaje**: se estratifica por servicio, origen, edad y sexo. La edad es el estratificador con más efecto.",
    "- La censura real (estancias abiertas en la extracción) y el truncamiento afectan < 0.5 % y no tocan el periodo estable.",
    "- La prevalencia del target de paciente en snapshots de urgencias depende de τ (0.52 → 0.76): evaluar por τ.",
    "- **Deriva**: la estancia hospitalaria crece (mediana ~68 h en 2023 → ~75 h en 2025–26; % sobre el umbral 23 % → 28 %), así que la prevalencia en prueba es 0.28 frente a 0.25 en entrenamiento.",
    "- **Cambio de registro en urgencias**: el % de LOS = 0 cambia por tramos dentro del periodo estable (hay que reportarlo a Persona 1).",
    "- Hospitalización tiene un componente obstétrico/neonatal visible en la pirámide de edad.",
    "- La falta de laboratorios no es aleatoria pero su efecto marginal es despreciable (V < 0.1): usar indicadores, no imputar.",
]
(OUT / "m2" / "M2_informe.md").write_text("\n".join(cab + informe) + "\n", encoding="utf-8")
print("OK → outputs/m2/M2_informe.md")
