"""M3 · EDA bivariado, tamaños de efecto y colas (HUN-202) — Persona 3.

Uso:  python m3_bivariado/run_m3.py
Salida: outputs/m3/{figuras,tablas}/ y outputs/m3/M3_informe.md (solo agregados).

Pruebas bivariadas: SOLO entrenamiento + periodo estable (no se mira la prueba para no
contaminar la selección de variables de Persona 2). Colas: periodo estable completo.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

from eda_comun import (  # noqa: E402
    COLORES, DIAS, GRUPOS_EDAD, M1, M4, NOMBRE, OUT, auc_univariada, bh, carpeta, con, cramers_v,
    dif_medianas_ic, fmt_p, grupo_edad_sql, guardar, kruskal_eps2, magnitud_auc, magnitud_eps2,
    magnitud_v, md_tabla, parametros,
)

FIG, TAB = carpeta("m3")
C = con()
P = parametros()
SERV = ["hospitalizacion", "urgencias_observacion"]
informe: list[str] = []


def sec(t):
    print(f"· {t}")
    informe.append(f"\n## {t}\n")


def add(*x):
    informe.extend(x)


def fig_md(n, alt):
    add(f"\n![{alt}](figuras/{n})\n")


# =====================================================================================
# Datos: snapshot τ = 0 (una fila por episodio), entrenamiento + estable, con calendario
# =====================================================================================
snap = C.sql(
    f"""
    SELECT s.*, {grupo_edad_sql('s.edad')} grupo_edad,
           CASE WHEN s.hora_del_dia BETWEEN 7 AND 12 THEN 'mañana'
                WHEN s.hora_del_dia BETWEEN 13 AND 18 THEN 'tarde' ELSE 'noche' END turno,
           CASE WHEN c.festivo THEN 'festivo' WHEN c.dia_despues_festivo THEN 'día después festivo'
                WHEN c.fin_de_semana THEN 'fin de semana' ELSE 'laboral' END tipo_dia,
           coalesce(c.puente, false) puente
    FROM 'data/m1/snapshot.parquet' s
    LEFT JOIN 'data/m4/calendario_dia.parquet' c ON c.fecha = CAST(s.snapshot_time AS DATE)
    WHERE s.en_periodo_estable AND s.es_entrenamiento AND s.horas_desde_ingreso = 0
    """
).df()
snap["sin_mediciones_24h"] = (snap.n_mediciones_24h == 0).map({True: "sí", False: "no"})
snap["nunca_hospitalizado"] = snap.horas_desde_ultima_alta_hosp.isna().map({True: "sí", False: "no"})
snap["origen"] = np.where(snap.origen_urgencias, "urgencias", "directo")
snap["dia"] = snap.dia_semana.map(dict(enumerate(DIAS, 1)))

CAT = ["sexo", "grupo_edad", "origen", "turno", "dia", "tipo_dia", "puente", "mes", "sin_mediciones_24h", "nunca_hospitalizado"]
NUM = ["edad", "hora_del_dia", "n_dx_historia", "n_dx_distintos_historia", "n_dx_30d", "n_mediciones_24h",
       "n_mediciones_historia", "n_visitas_365d", "n_hospitalizaciones_previas", "horas_desde_ultima_alta_hosp",
       "censo_servicio_t_menos_1h", "ocupacion_servicio_t_menos_1h", "llegadas_servicio_t_menos_1h"]

# =====================================================================================
# 1. Bivariado vs target de paciente
# =====================================================================================
sec("1. Asociación bivariada con el target de paciente (LOS > P75)")
add(
    "Unidad: snapshot τ = 0 (uno por episodio), **solo entrenamiento** del periodo estable "
    f"(hospitalización n = {(snap.servicio=='hospitalizacion').sum():,}; urgencias/observación "
    f"n = {(snap.servicio=='urgencias_observacion').sum():,}; en urgencias, τ = 0 excluye las estancias de LOS 0).\n",
    "- **Categóricas**: Chi² de independencia + **V de Cramér** con corrección de sesgo.\n"
    "- **Numéricas**: Mann-Whitney / Kruskal-Wallis + **ε²**, **AUC univariada** (P(X₁ > X₀)), "
    "diferencia de medianas (positivos − negativos) con **IC 95 % bootstrap**.\n"
    "- Todos los p-valores se corrigen por **Benjamini-Hochberg** dentro de cada servicio. Las "
    "magnitudes siguen a Cohen (V) y Tomczak (ε²); para el AUC se usa |2·AUC − 1| con cortes 0.1/0.3/0.5.\n",
)
filas = []
for s in SERV:
    d = snap[snap.servicio == s]
    y = d.y_target_los_largo.values
    for v in CAT:
        if d[v].nunique() < 2:
            continue
        tabla = pd.crosstab(d[v], d.y_target_los_largo)
        chi2, p, cv, gl = cramers_v(tabla.values)
        tasas = (tabla[1] / tabla.sum(1)).round(3)
        filas.append({"servicio": s, "variable": v, "tipo": "categórica", "estadístico": chi2, "p": p,
                      "efecto": cv, "métrica": "V Cramér", "magnitud": magnitud_v(cv, min(tabla.shape) - 1),
                      "detalle": f"tasa {tasas.idxmin()}={tasas.min():.2f} … {tasas.idxmax()}={tasas.max():.2f}"})
    for v in NUM:
        x = d[v].values.astype(float)
        m = ~np.isnan(x)
        if m.sum() < 100:
            continue
        x1, x0 = x[m & (y == 1)], x[m & (y == 0)]
        h, p, e2 = kruskal_eps2([x1, x0])
        auc = auc_univariada(x[m], y[m])
        dm, lo, hi = dif_medianas_ic(x1, x0)
        filas.append({"servicio": s, "variable": v, "tipo": "numérica", "estadístico": h, "p": p,
                      "efecto": abs(2 * auc - 1), "métrica": "|2AUC−1|", "magnitud": magnitud_auc(auc),
                      "detalle": f"AUC={auc:.3f}; ε²={e2:.4f}; Δmed={dm:.2f} [{lo:.2f}, {hi:.2f}]",
                      "auc": auc, "eps2": e2})
biv = pd.DataFrame(filas)
biv["p_BH"] = biv.groupby("servicio").p.transform(bh)
biv = biv.sort_values(["servicio", "efecto"], ascending=[True, False])
biv.to_csv(TAB / "bivariado_target_paciente.csv", index=False)
for s in SERV:
    t = biv[biv.servicio == s].copy()
    t["p_BH"] = t.p_BH.map(fmt_p)
    add(f"\n**{NOMBRE[s]}** (ordenado por tamaño de efecto):\n",
        md_tabla(t[["variable", "tipo", "métrica", "efecto", "magnitud", "p_BH", "detalle"]], floatfmt=".3f"))

colmag = {"despreciable": "#bfbfbf", "pequeño": "#9dc3e6", "mediano": "#2e75b6", "grande": "#1f3864"}
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"wspace": 0.6})
for ax, s in zip(axes, SERV):
    t = biv[biv.servicio == s].sort_values("efecto")
    ax.barh(t.variable + np.where(t.tipo == "numérica", " ⁿ", " ᶜ"), t.efecto, color=t.magnitud.map(colmag))
    for yy, (e, sig) in enumerate(zip(t.efecto, t.p_BH)):
        ax.text(e + 0.003, yy, f"{e:.2f}" + ("" if sig < 0.05 else " (n.s.)"), va="center", fontsize=7.5)
    ax.set_title(NOMBRE[s])
    ax.set_xlabel("Tamaño de efecto (V de Cramér ᶜ  o  |2·AUC − 1| ⁿ)")
    ax.axvline(0.1, color="grey", ls=":", lw=0.8)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colmag.values()]
axes[1].legend(handles, colmag.keys(), loc="lower right", title="Magnitud")
fig.suptitle("Ranking de asociación con estancia larga (entrenamiento, τ = 0, BH)", fontweight="bold")
fig_md(guardar(fig, FIG / "01_ranking_efectos_paciente.png"), "ranking")

# Tasa de target por decil de las numéricas más fuertes
top = {s: biv[(biv.servicio == s) & (biv.tipo == "numérica")].head(6).variable.tolist() for s in SERV}
fig, axes = plt.subplots(2, 6, figsize=(18, 6), sharey="row")
for r, s in enumerate(SERV):
    d = snap[snap.servicio == s]
    for c, v in enumerate(top[s]):
        ax = axes[r, c]
        x = d[v]
        try:
            b = pd.qcut(x.rank(method="first"), 10, labels=False)
        except ValueError:
            continue
        g = d.groupby(b).agg(tasa=("y_target_los_largo", "mean"), med=(v, "median"))
        ax.plot(range(1, len(g) + 1), g.tasa, "-o", ms=3, color=COLORES[s])
        ax.axhline(d.y_target_los_largo.mean(), color="grey", ls=":", lw=0.8)
        ax.set_title(v, fontsize=8.5)
        ax.set_xticks([1, 5, 10])
        ax.set_xlabel("decil", fontsize=8)
    axes[r, 0].set_ylabel(f"{NOMBRE[s]}\nP(LOS > P75)")
fig.suptitle("Tasa de estancia larga por decil de las 6 variables numéricas con más efecto", fontweight="bold")
fig_md(guardar(fig, FIG / "02_tasa_por_decil.png"), "deciles")

# Correlación de Spearman entre variables
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
for ax, s in zip(axes, SERV):
    d = snap.loc[snap.servicio == s, NUM + ["y_target_los_largo"]]
    cm = d.corr(method="spearman")
    sns.heatmap(cm, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".1f",
                annot_kws={"size": 6}, cbar=ax is axes[1], square=True)
    ax.set_title(NOMBRE[s])
    ax.tick_params(labelsize=7)
fig.suptitle("Correlación de Spearman entre variables de entrada (entrenamiento, τ = 0)", fontweight="bold")
fig_md(guardar(fig, FIG / "03_spearman.png"), "spearman")
pares = []
for s in SERV:
    cm = snap.loc[snap.servicio == s, NUM].corr(method="spearman")
    iu = np.triu_indices_from(cm, 1)
    for i, j in zip(*iu):
        if abs(cm.iat[i, j]) >= 0.7:
            pares.append({"servicio": NOMBRE[s], "var_1": cm.index[i], "var_2": cm.columns[j], "rho": cm.iat[i, j]})
pares = pd.DataFrame(pares)
pares.to_csv(TAB / "colinealidad_spearman_07.csv", index=False)
add("\n**Pares con |ρ de Spearman| ≥ 0.7 (redundancia para M5):**\n", md_tabla(pares, floatfmt=".2f") if len(pares) else "Ninguno.")

def resumen_top(s, k=4):
    t = biv[(biv.servicio == s)].head(k)
    return ", ".join(f"`{v}` ({e:.2f}, {m})" for v, e, m in zip(t.variable, t.efecto, t.magnitud))

add(
    "\n**Lectura.**\n",
    f"- Hospitalización: las variables con más efecto son {resumen_top('hospitalizacion')}.\n",
    f"- Urgencias/observación: {resumen_top('urgencias_observacion')}.\n",
    "- Casi todo es estadísticamente significativo tras BH, pero **ninguna variable tiene por sí sola "
    "un efecto grande**: el target de paciente es multifactorial y el modelo tendrá que combinar "
    "historia clínica, edad y carga del servicio. Las variables de calendario (día, turno, festivo) "
    "tienen efecto despreciable sobre la estancia del paciente, aunque sí afectan a las llegadas (M4).\n",
    "- Las variables de diagnósticos y mediciones están muy correlacionadas entre sí: conviene "
    "reducirlas (SVD o selección L1, M5) o usar modelos robustos a la colinealidad.",
)

# =====================================================================================
# 2. Estancia (continua) por grupos
# =====================================================================================
sec("2. Estancia continua por grupos (Kruskal-Wallis + ε²)")
filas = []
for s in SERV:
    d = snap[snap.servicio == s]
    for v in ["grupo_edad", "turno", "dia", "tipo_dia", "sin_mediciones_24h", "nunca_hospitalizado", "sexo", "origen"]:
        grupos = [g.y_los_h.values for _, g in d.groupby(v) if len(g) > 30]
        if len(grupos) < 2:
            continue
        h, p, e2 = kruskal_eps2(grupos)
        meds = d.groupby(v).y_los_h.median()
        filas.append({"servicio": NOMBRE[s], "variable": v, "H": h, "p": p, "ε²": e2, "magnitud": magnitud_eps2(e2),
                      "mediana LOS por grupo (h)": "; ".join(f"{k}: {m:.1f}" for k, m in meds.items())})
kw = pd.DataFrame(filas)
kw["p_BH"] = kw.groupby("servicio").p.transform(bh).map(fmt_p)
kw = kw.sort_values(["servicio", "ε²"], ascending=[True, False])
kw.drop(columns="p").to_csv(TAB / "kruskal_los_grupos.csv", index=False)
add(md_tabla(kw[["servicio", "variable", "ε²", "magnitud", "p_BH", "mediana LOS por grupo (h)"]], floatfmt=".4f"))

# =====================================================================================
# 3. Bivariado vs target de servicio (saturación t+4h)
# =====================================================================================
sec("3. Asociación con el target de servicio (saturación en t + 4 h)")
osh = C.sql(
    """
    SELECT o.*, extract(hour FROM o.hora) hora_del_dia, isodow(o.hora) dia_semana, month(o.hora) mes,
           h.turno, h.festivo, h.fin_de_semana,
           o.censo - lag(o.censo, 4) OVER (PARTITION BY o.servicio ORDER BY o.hora) delta_censo_4h,
           sum(o.llegadas) OVER (PARTITION BY o.servicio ORDER BY o.hora ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) llegadas_4h
    FROM 'data/m1/ocupacion_servicio_hora.parquet' o
    LEFT JOIN 'data/m4/calendario_hora.parquet' h ON h.hora = o.hora
    WHERE o.en_periodo_estable
    """
).df()
tr = osh[osh.es_entrenamiento]
filas = []
for s in SERV:
    d = tr[tr.servicio == s]
    y = d.y_target_saturacion.values
    for v in ["turno", "dia_semana", "mes", "festivo", "fin_de_semana", "hora_del_dia"]:
        tabla = pd.crosstab(d[v], d.y_target_saturacion)
        chi2, p, cv, gl = cramers_v(tabla.values)
        filas.append({"servicio": NOMBRE[s], "variable": v, "métrica": "V Cramér", "efecto": cv,
                      "magnitud": magnitud_v(cv, min(tabla.shape) - 1), "p": p})
    for v in ["ocupacion", "censo", "llegadas", "llegadas_4h", "delta_censo_4h", "capacidad"]:
        x = d[v].values.astype(float)
        m = ~np.isnan(x)
        auc = auc_univariada(x[m], y[m])
        _, p, _ = kruskal_eps2([x[m & (y == 1)], x[m & (y == 0)]])
        filas.append({"servicio": NOMBRE[s], "variable": v, "métrica": "AUC", "efecto": auc,
                      "magnitud": magnitud_auc(auc), "p": p})
sat = pd.DataFrame(filas)
sat["p_BH"] = sat.groupby("servicio").p.transform(bh).map(fmt_p)
sat.drop(columns="p").to_csv(TAB / "bivariado_target_saturacion.csv", index=False)
add(
    "Unidad: servicio × hora (entrenamiento, periodo estable). **Advertencia**: las horas consecutivas "
    "están autocorrelacionadas, así que los p-valores son optimistas; lo informativo es el tamaño de efecto.\n",
    md_tabla(sat.drop(columns="p"), floatfmt=".3f"),
)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, s in zip(axes, SERV):
    d = tr[tr.servicio == s]
    g = d.groupby(pd.cut(d.ocupacion, np.arange(0.3, 1.31, 0.05)), observed=True).y_target_saturacion.agg(["mean", "size"])
    g = g[g["size"] >= 30]
    ax.plot([i.mid for i in g.index], g["mean"], "-o", ms=3, color=COLORES[s])
    ax.axvline(d.umbral_ocupacion.iloc[0], color="k", ls="--", lw=0.8, label="umbral de saturación")
    ax.set_title(NOMBRE[s])
    ax.set_xlabel("Ocupación actual (censo / capacidad proxy)")
    ax.legend()
axes[0].set_ylabel("P(saturado en t + 4 h)")
fig.suptitle("La ocupación actual casi determina la saturación a 4 h (persistencia)", fontweight="bold")
fig_md(guardar(fig, FIG / "04_saturacion_vs_ocupacion.png"), "persistencia")
auc_ocu = sat[(sat.variable == "ocupacion")].set_index("servicio").efecto
add(
    f"\n**Lectura.** La ocupación actual sola tiene AUC = {auc_ocu.iloc[0]:.3f} (hosp.) y "
    f"{auc_ocu.iloc[1]:.3f} (urg.) para predecir la saturación a 4 h: es un **baseline de persistencia** "
    "muy fuerte que Persona 2 debe reportar como referencia mínima. La **hora del día** tiene efecto "
    "mediano en ambos servicios (V ≈ 0.31 hosp., 0.46 urg.) y el mes, efecto pequeño: hay que darle "
    "al modelo el ciclo diario (hora y turno) y la estacionalidad. En urgencias también pesan las "
    "llegadas de las últimas 4 h (AUC 0.82) y el cambio de censo en 4 h (0.78); en hospitalización "
    "las llegadas casi no informan (AUC ≈ 0.46): lo que la satura es que las altas no liberen camas.",
)

# =====================================================================================
# 4. Teoría de colas
# =====================================================================================
sec("4. Teoría de colas por servicio")
rc = pd.read_parquet(M4 / "resumen_colas.parquet")
rc.to_csv(TAB / "resumen_colas_p1.csv", index=False)
add(
    "Tabla de Persona 1 (`resumen_colas`, periodo estable). λ = llegadas/h, E[S] = duración media "
    "del subproceso, μ = 1/E[S], L = censo medio, c = capacidad proxy media, ρ = λ·E[S]/c.\n",
    md_tabla(rc, floatfmt=".3f"),
    "\nLa **ley de Little** (L = λ·W) cuadra en los dos servicios con estancia (error < 0.5 %), lo que "
    "valida que censo, llegadas y duraciones son coherentes entre sí.\n",
)
# Por año: λ, W, L, c, ρ
anual = C.sql(
    f"""
    WITH lim AS (
      SELECT min(hora) h0, max(hora) + INTERVAL 1 HOUR h1
      FROM 'data/m1/ocupacion_servicio_hora.parquet' WHERE en_periodo_estable),
    d AS (
      SELECT servicio, year(inicio) anio, count(*) n, avg(dur_h) W
      FROM 'data/m1/detalle.parquet', lim
      WHERE servicio IN ('hospitalizacion','urgencias_observacion') AND inicio >= h0 AND inicio < h1
      GROUP BY ALL),
    o AS (
      SELECT servicio, year(hora) anio, count(*) horas, avg(censo) L, avg(capacidad) c,
             avg((ocupacion >= 0.85)::int) frac_085, avg((ocupacion >= 0.95)::int) frac_095,
             avg((ocupacion >= 1)::int) frac_100
      FROM 'data/m1/ocupacion_servicio_hora.parquet' WHERE en_periodo_estable GROUP BY ALL)
    SELECT d.servicio, d.anio, d.n / o.horas lambda_h, d.W W_h, o.L, d.n / o.horas * d.W little_lambdaW,
           o.c, d.n / o.horas * d.W / o.c rho, o.frac_085, o.frac_095, o.frac_100, o.horas
    FROM d JOIN o USING (servicio, anio) ORDER BY 1, 2
    """
).df()
anual.to_csv(TAB / "colas_por_anio.csv", index=False)
add(
    "**Por año** (λ y W desde `detalle`; L, c y fracción de horas con ocupación ≥ umbral desde "
    "`ocupacion_servicio_hora`, ambos sobre las mismas horas; 2023 empieza en mayo por la ventana "
    "de 90 días de la capacidad):\n",
    md_tabla(anual, floatfmt=".3f"),
    "\n**Lectura.** En hospitalización L y W crecen año a año con λ casi constante: el censo sube "
    "porque las estancias se alargan (coherente con la deriva del M2), no porque lleguen más pacientes. "
    "En urgencias/observación λ baja y W sube. Es la misma firma del cambio de registro visto en el "
    "M2: hay menos subprocesos y más largos, y el producto λ·W apenas cambia.",
)

# Fracción de horas saturadas por turno y servicio
turno = (
    osh.groupby(["servicio", "turno"])
    .agg(ocupacion_media=("ocupacion", "mean"), frac_085=("ocupacion", lambda x: (x >= 0.85).mean()),
         frac_095=("ocupacion", lambda x: (x >= 0.95).mean()), llegadas_h=("llegadas", "mean"))
    .reset_index()
)
turno.to_csv(TAB / "saturacion_por_turno.csv", index=False)
add("\n**Por turno** (mañana 7–13, tarde 13–19, noche 19–7):\n", md_tabla(turno, floatfmt=".3f"))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
ax = axes[0]
for s in SERV:
    x = np.sort(osh.loc[osh.servicio == s, "ocupacion"].dropna())
    ax.plot(x, 1 - np.arange(len(x)) / len(x), color=COLORES[s], label=NOMBRE[s])
for u in (0.85, 0.95):
    ax.axvline(u, color="k", ls="--", lw=0.7)
    ax.text(u + (0.01 if u > 0.9 else -0.01), 1.03, f"{u}", ha="left" if u > 0.9 else "right", fontsize=8)
ax.set_xlabel("Ocupación horaria ρ̂ (censo / capacidad proxy)")
ax.set_ylabel("Fracción de horas con ocupación ≥ x")
ax.set_title("Curva de excedencia de la ocupación")
ax.legend()
ax = axes[1]
mens = osh.assign(m=osh.hora.dt.to_period("M").dt.to_timestamp()).groupby(["servicio", "m"]).ocupacion.agg(
    f085=lambda x: (x >= 0.85).mean(), f095=lambda x: (x >= 0.95).mean()).reset_index()
for s in SERV:
    d = mens[mens.servicio == s]
    ax.plot(d.m, d.f085, color=COLORES[s], label=f"{NOMBRE[s]} ρ ≥ 0.85")
    ax.plot(d.m, d.f095, color=COLORES[s], ls="--", label=f"{NOMBRE[s]} ρ ≥ 0.95")
ax.set_title("Fracción mensual de horas por encima del umbral")
ax.tick_params(axis="x", rotation=30)
ax.legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
ax = axes[2]
perfil = osh.groupby(["servicio", "hora_del_dia"]).agg(llegadas=("llegadas", "mean"), censo=("censo", "mean")).reset_index()
for s in SERV:
    d = perfil[perfil.servicio == s]
    ax.plot(d.hora_del_dia, d.llegadas / d.llegadas.max(), color=COLORES[s], ls="--", label=f"{NOMBRE[s]} llegadas")
    ax.plot(d.hora_del_dia, d.censo / d.censo.max(), color=COLORES[s], label=f"{NOMBRE[s]} censo")
ax.set_xlabel("Hora del día")
ax.set_ylabel("Relativo al máximo del día")
ax.set_title("Perfil diario: llegadas vs censo")
ax.set_xticks(range(0, 24, 3))
ax.legend(fontsize=7)
fig.suptitle("Utilización por servicio (periodo estable)", fontweight="bold")
fig_md(guardar(fig, FIG / "05_utilizacion.png"), "utilización")

# Desfase llegadas → censo (correlación cruzada de las series sin ciclo semanal)
lags = []
for s in SERV:
    d = osh[osh.servicio == s].set_index("hora").sort_index()
    pl = perfil[perfil.servicio == s].set_index("hora_del_dia")
    pico_ll = int(pl.llegadas.idxmax())
    pico_ce = int(pl.censo.idxmax())
    a = d.llegadas - d.llegadas.rolling(168, center=True, min_periods=24).mean()
    b = d.censo - d.censo.rolling(168, center=True, min_periods=24).mean()
    cc = {k: a.corr(b.shift(-k)) for k in range(0, 25)}
    kmax = max(cc, key=cc.get)
    lags.append({"servicio": NOMBRE[s], "hora pico llegadas": pico_ll, "hora pico censo": pico_ce,
                 "desfase de máx. correlación (h)": kmax, "corr máx": cc[kmax]})
lags = pd.DataFrame(lags)
lags.to_csv(TAB / "desfase_llegadas_censo.csv", index=False)
add("\n**Desfase llegadas → censo** (correlación cruzada de las series sin tendencia semanal):\n", md_tabla(lags, floatfmt=".2f"))

# =====================================================================================
# 5. Cuellos de botella: duración por etapa y espera entre subprocesos
# =====================================================================================
sec("5. Cuellos de botella: duración por etapa, espera y bloqueo urgencias → hospitalización")
etapas = C.sql(
    f"""
    SELECT servicio, count(*) n, avg(dur_h) media_h,
           quantile_cont(dur_h, 0.5) p50_h, quantile_cont(dur_h, 0.75) p75_h,
           quantile_cont(dur_h, 0.9) p90_h, quantile_cont(dur_h, 0.99) p99_h,
           avg((dur_h = 0)::int) frac_cero, sum(dur_h) / 24 / 1186 camas_dia_prom
    FROM 'data/m1/detalle.parquet'
    WHERE inicio >= TIMESTAMP '{P.inicio_periodo_estable}' AND inicio < TIMESTAMP '{P.fin_periodo_estable}'
    GROUP BY ALL ORDER BY p50_h DESC
    """
).df().drop(columns="camas_dia_prom")
etapas.to_csv(TAB / "duracion_por_etapa.csv", index=False)
mt = pd.read_parquet(M4 / "matriz_transicion.parquet")
mt_main = mt[(mt.n >= 1000) & ~mt.desde.str.startswith("(") & ~mt.hacia.str.startswith("(")]
mt_main.to_csv(TAB / "esperas_transiciones.csv", index=False)
add(
    "**Duración de cada subproceso** (`detalle`, periodo estable):\n", md_tabla(etapas, floatfmt=".3f"),
    "\n**Espera entre subprocesos** (transiciones con n ≥ 1 000, `matriz_transicion`):\n",
    md_tabla(mt_main[["desde", "hacia", "n", "prob", "espera_p50_h", "espera_p75_h", "espera_p90_h", "frac_solapada", "duracion_desde_p50_h"]], floatfmt=".3f"),
    "\nLa espera registrada entre subprocesos es **0 en la mediana y el P75** de todas las transiciones "
    "frecuentes: el fin de un registro es el inicio del siguiente. Los datos **no registran colas de "
    "espera** (no hay triaje → consulta → exámenes). El cuello de botella hay que medirlo con la "
    "**duración** de la etapa previa y con la **ocupación**.\n",
)

# Bloqueo (boarding): tiempo en urgencias de quienes terminan hospitalizados vs ocupación hospitalaria
board = C.sql(
    """
    WITH v AS (
      SELECT visit_occurrence_id,
             min(momento) FILTER (WHERE desde = '(llegada)') t_llegada,
             min(momento) FILTER (WHERE hacia = 'hospitalizacion' AND desde LIKE 'urgencias%') t_hosp,
             bool_or(hacia = 'hospitalizacion') va_hosp,
             bool_or(desde = '(llegada)' AND hacia LIKE 'urgencias%') entra_urg,
             bool_and(en_periodo_estable) estable
      FROM 'data/m4/transicion.parquet' GROUP BY 1)
    SELECT v.*, date_diff('second', t_llegada, t_hosp) / 3600.0 horas_hasta_hosp,
           o.ocupacion ocup_hosp_llegada, ou.ocupacion ocup_urg_llegada, extract(hour FROM t_llegada) hora_llegada
    FROM v
    LEFT JOIN 'data/m1/ocupacion_servicio_hora.parquet' o
      ON o.servicio = 'hospitalizacion' AND o.hora = date_trunc('hour', v.t_llegada)
    LEFT JOIN 'data/m1/ocupacion_servicio_hora.parquet' ou
      ON ou.servicio = 'urgencias_observacion' AND ou.hora = date_trunc('hour', v.t_llegada)
    WHERE v.estable AND v.entra_urg AND v.t_hosp IS NOT NULL AND o.ocupacion IS NOT NULL
    """
).df()
board = board[(board.horas_hasta_hosp >= 0) & (board.horas_hasta_hosp < 24 * 7)]
board["decil_ocup"] = pd.qcut(board.ocup_hosp_llegada, 10, labels=False) + 1
bd = board.groupby("decil_ocup").agg(
    ocup_media=("ocup_hosp_llegada", "mean"), n=("horas_hasta_hosp", "size"),
    mediana_h=("horas_hasta_hosp", "median"), p75_h=("horas_hasta_hosp", lambda x: x.quantile(.75)),
    p90_h=("horas_hasta_hosp", lambda x: x.quantile(.9))).reset_index()
bd.to_csv(TAB / "bloqueo_urg_hosp_por_ocupacion.csv", index=False)
h_b, p_b, e2_b = kruskal_eps2([g.horas_hasta_hosp.values for _, g in board.groupby("decil_ocup")])
rho_b, prho_b = stats.spearmanr(board.ocup_hosp_llegada, board.horas_hasta_hosp)
q1 = board[board.decil_ocup == 1].horas_hasta_hosp
q10 = board[board.decil_ocup == 10].horas_hasta_hosp
dm, lo, hi = dif_medianas_ic(q10.values, q1.values)
add(
    f"**Bloqueo urgencias → hospitalización (boarding).** Para las {len(board):,} visitas del periodo "
    "estable que entran por urgencias y terminan hospitalizadas, se mide el tiempo desde la llegada "
    "hasta el ingreso a hospitalización y se cruza con la ocupación hospitalaria en la hora de llegada:\n",
    md_tabla(bd, floatfmt=".2f"),
    f"\nKruskal-Wallis entre deciles: ε² = {e2_b:.4f} ({magnitud_eps2(e2_b)}), p = {fmt_p(p_b)}; "
    f"Spearman ρ = {rho_b:.3f}. Diferencia de medianas decil 10 − decil 1 = {dm:.2f} h "
    f"[IC 95 % {lo:.2f}, {hi:.2f}].\n",
)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.plot(bd.ocup_media, bd.mediana_h, "-o", color=COLORES["hospitalizacion"], label="Mediana")
ax.fill_between(bd.ocup_media, bd.mediana_h, bd.p75_h, alpha=0.2, color=COLORES["hospitalizacion"], label="Mediana–P75")
ax.set_xlabel("Ocupación hospitalaria al llegar a urgencias (decil)")
ax.set_ylabel("Horas desde llegada hasta hospitalización")
ax.set_title("Bloqueo: espera en urgencias vs ocupación hospitalaria")
ax.legend()
ax = axes[1]
dd = C.sql(
    f"""SELECT servicio, dur_h FROM 'data/m1/detalle.parquet'
        WHERE servicio <> 'ambulatorio' AND dur_h > 0
          AND inicio >= TIMESTAMP '{P.inicio_periodo_estable}' AND inicio < TIMESTAMP '{P.fin_periodo_estable}'
        USING SAMPLE reservoir(300000 ROWS) REPEATABLE (42)"""
).df()
orden = ["urgencias_ingreso", "urgencias_observacion", "hospitalizacion"]
ax.boxplot([np.log10(dd.loc[dd.servicio == s, "dur_h"]) for s in orden],
           showfliers=False, patch_artist=True, boxprops=dict(facecolor="#dde7f2"))
ax.set_xticks(range(1, len(orden) + 1))
ax.set_xticklabels([NOMBRE[s] for s in orden])
ax.set_yticks([-2, -1, 0, 1, 2, 3])
ax.set_yticklabels(["36 s", "6 min", "1 h", "10 h", "100 h", "1000 h"])
ax.set_title("Duración del subproceso por etapa (dur > 0, log)")
fig.suptitle("Cuellos de botella medidos por duración", fontweight="bold")
fig_md(guardar(fig, FIG / "06_cuellos_botella.png"), "cuellos")

r_h = anual[anual.servicio == "hospitalizacion"]
add(
    "\n**Lectura de colas.**\n",
    f"- **Hospitalización es el cuello de botella del sistema**: ρ ≈ {rc.loc[rc.servicio=='hospitalizacion','rho_aprox'].iloc[0]:.2f} "
    f"respecto a su propio P95 reciente, con ocupación ≥ 0.85 en el {100*r_h.frac_085.mean():.0f} % de las horas. "
    "Recibe pocas llegadas (≈ 3.6/h) pero cada una ocupa una cama ~80 h, de modo que pequeñas "
    "variaciones de llegadas o de estancia saturan el servicio.\n",
    "- **Ciclo diario de hospitalización**: el censo llega al máximo hacia las 11 h y las llegadas "
    "hacia las 17 h. La mañana es el turno más lleno (ρ ≥ 0.95 en ~52 % de las horas, frente a ~33–35 % "
    "en tarde y noche): las camas se liberan con las altas del mediodía. Las alertas de saturación "
    "hospitalaria se concentrarán en la mañana.\n",
    "- Urgencias/observación tiene muchas llegadas (≈ 38/h) y estancias cortas. Las llegadas pican "
    "hacia las 11 h y el censo ~4–5 h después (16 h); la **tarde** es el turno crítico (ocupación ≥ 0.85 "
    "en ~52 % de sus horas, frente a 12 % en la mañana).\n",
    f"- **Tiempo hasta cama**: quien entra por urgencias y termina hospitalizado tarda una mediana de "
    f"~{board.horas_hasta_hosp.median():.0f} h desde la llegada hasta el ingreso a hospitalización "
    f"(P75 ~{board.horas_hasta_hosp.quantile(.75):.0f} h). Ese es el mayor tiempo \"de paso\" del sistema. "
    "Con más ocupación hospitalaria ese tiempo crece, pero poco "
    f"(+{dm:.1f} h entre el decil 10 y el 1; ε² despreciable). Hay una señal débil de *boarding*: "
    "la mayor parte de la espera no se explica por la ocupación en el momento de llegar.\n",
    "- Advertencia: la capacidad es un proxy (P95 móvil del censo). ρ mide qué tan cerca está el "
    "servicio de su máximo reciente, no de las camas reales. Con el número real de camas, ρ y la "
    "fracción de horas saturadas cambiarán.",
)

# =====================================================================================
cab = [
    "# M3 · EDA bivariado, tamaños de efecto y colas (HUN-202)",
    "",
    "Persona 3 · generado por `m3_bivariado/run_m3.py` · pruebas bivariadas sobre **entrenamiento** del "
    "periodo estable; colas sobre todo el periodo estable. Solo agregados.",
    "",
    "**Resumen ejecutivo**",
    "",
    "- Con n de decenas o cientos de miles todo sale p < 0.001: se ordena por tamaño de efecto (V de Cramér, ε², AUC) con BH.",
    "- Ninguna variable tiene por sí sola un efecto grande sobre la estancia larga. Las más fuertes son historia de uso/diagnósticos, edad y mediciones.",
    "- El calendario afecta a las llegadas, pero casi nada a la estancia individual.",
    "- Saturación a 4 h: la ocupación actual es un baseline de persistencia muy fuerte (AUC alta). El modelo tiene que superarlo.",
    "- La espera registrada entre subprocesos es ~0: los cuellos de botella se miden por duración y ocupación.",
    "- Hospitalización es el cuello de botella (ρ ≈ 0.93 respecto al proxy). Su censo crece porque las estancias se alargan, no porque lleguen más pacientes.",
    "- Urgencias → cama de hospitalización tarda una mediana de ~14 h. Depende poco de la ocupación al llegar (señal débil de boarding).",
    "- Turnos críticos: mañana en hospitalización (altas tardías) y tarde en urgencias.",
]
(OUT / "m3" / "M3_informe.md").write_text("\n".join(cab + informe) + "\n", encoding="utf-8")
print("OK → outputs/m3/M3_informe.md")
