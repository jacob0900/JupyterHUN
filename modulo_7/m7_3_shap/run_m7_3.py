"""Subfase 7.3 sobre los datos reales: SHAP del mejor modelo (LightGBM) en una muestra de validación.

- Modelo: el Pipeline LightGBM de la 6.4 (hiperparámetros de `hiperparametros_finales.json`) reentrenado **sin
  el pliegue 1**. Se explica una muestra estratificada (servicio × τ × etiqueta) de 10 000 snapshots del pliegue 1:
  pacientes que ese modelo nunca vio (validación agrupada de la 6.1). La prueba temporal no se toca.
- Los SHAP agrupados quedan en caché (`data/m7/shap_m7_3.parquet`, nivel paciente: no compartir); con
  `--recalcular` se reentrena y se recalcula.
- Casos de estudio **dentro de cada servicio** (5 % de mayor / menor riesgo), como perfiles agregados.
- Figuras (PNG, 300 DPI) en reports/figures/; tablas agregadas en outputs/m7/tablas/; síntesis clínica en
  reports/m7_3_sintesis_medica.md.

Uso (desde la raíz del proyecto):
    python modulo_7/m7_3_shap/run_m7_3.py [--n 10000] [--recalcular]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modulo_5.m5_4_pipeline import construir_preprocesador, preparar_X  # noqa: E402
from modulo_6.m6_4_diagnostico import instanciar  # noqa: E402
from modulo_7.m7_3_shap import (  # noqa: E402
    TRADUCCION_DX,
    agrupar_shap,
    efecto_diagnosticos,
    etiqueta,
    explicacion,
    explicar,
    formatear_valor,
    graficar_barras,
    graficar_barras_por_servicio,
    graficar_beeswarm,
    graficar_waterfall,
    importancia_global,
    mapa_grupos,
    perfil,
    sintesis_medica,
    valores_color,
)

TARGET = "y_target_los_largo"
MODELO = "lightgbm"
PLIEGUE_EXPLICADO = 1
FIG, TAB, REP = ROOT / "reports" / "figures", ROOT / "outputs" / "m7" / "tablas", ROOT / "reports"
CACHE, CACHE_META = ROOT / "data" / "m7" / "shap_m7_3.parquet", ROOT / "data" / "m7" / "shap_m7_3.json"
NOMBRE_SRV = {"hospitalizacion": "Hospitalización", "urgencias_observacion": "Urgencias / observación"}
CORTO_SRV = {"hospitalizacion": "hosp", "urgencias_observacion": "urg"}
SIG = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def calcular(tr, esquema, cols, n):
    hp = json.loads((ROOT / "outputs" / "m6" / "hiperparametros_finales.json").read_text(encoding="utf-8"))["modelos"]
    t0 = time.time()
    ent, val = tr[tr.pliegue != PLIEGUE_EXPLICADO], tr[tr.pliegue == PLIEGUE_EXPLICADO].reset_index(drop=True)
    pipe = instanciar(MODELO, hp[MODELO], construir_preprocesador(esquema)).fit(ent[cols], ent[TARGET])
    auc_val = roc_auc_score(val[TARGET], pipe.predict_proba(val[cols])[:, 1])
    log(f"{MODELO} sin el pliegue {PLIEGUE_EXPLICADO}: {len(ent):,} filas · AUC en el pliegue {auc_val:.4f} "
        f"({(time.time() - t0) / 60:.1f} min)")
    estrato = val.servicio + "_" + val.horas_desde_ingreso.astype(str) + "_" + val[TARGET].astype(str)
    mu = (val.groupby(estrato, group_keys=False).sample(frac=min(1.0, n / len(val)), random_state=42)
          .reset_index(drop=True))
    pre, lgbm = pipe.named_steps["preprocesador"], pipe.named_steps["modelo"]
    Z = pre.transform(mu[cols])
    t0 = time.time()
    phi, base = explicar(lgbm, Z)
    margen = lgbm.predict(Z, raw_score=True)
    err = float(np.abs(base + phi.sum(axis=1) - margen).max())
    if err > 1e-4:
        raise ValueError(f"SHAP no aditivo (error máximo {err:.2e})")
    nombres = list(pre.get_feature_names_out())
    phi_g = agrupar_shap(phi, mapa_grupos(nombres, esquema))
    log(f"SHAP: {len(mu):,} snapshots × {len(nombres)} columnas → {phi_g.shape[1]} variables clínicas "
        f"({(time.time() - t0):.0f} s; aditividad exacta, error máx. {err:.1e})")
    phi_g.assign(snapshot_id=mu.snapshot_id.to_numpy(), _margen=margen).to_parquet(CACHE, index=False)
    CACHE_META.write_text(json.dumps({"base": base, "auc_pliegue": auc_val, "n_columnas": len(nombres),
                                      "error_aditividad": err, "fecha": datetime.now().isoformat(timespec="seconds")},
                                     indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--recalcular", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    for d in (FIG, TAB, CACHE.parent):
        d.mkdir(parents=True, exist_ok=True)

    X, esquema = preparar_X(ROOT)
    tr = X[X.en_periodo_estable.astype(bool) & X.es_entrenamiento.astype(bool)].reset_index(drop=True)
    pl = pd.read_parquet(ROOT / "data" / "m6" / "pliegues_m6_1.parquet")
    if not pl.snapshot_id.reset_index(drop=True).equals(tr.snapshot_id):
        raise ValueError("Pliegues de la 6.1 no alineados con el entrenamiento estable")
    tr["pliegue"] = pl.pliegue.to_numpy()
    cols = esquema.todas()
    if a.recalcular or not CACHE.exists():
        calcular(tr, esquema, cols, a.n)
    cache = pd.read_parquet(CACHE)
    meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
    base = meta["base"]
    mu = tr.set_index("snapshot_id").loc[cache.snapshot_id].reset_index()
    margen = cache.pop("_margen").to_numpy()
    phi_g = cache.drop(columns="snapshot_id")
    col = valores_color(mu, list(phi_g.columns), clinica=esquema.clinica)
    srv = mu.servicio.to_numpy()
    log(f"{len(mu):,} snapshots explicados · E[f(X)] = {base:.3f} (p = {SIG(base):.3f}) · AUC del modelo en el pliegue "
        f"{meta['auc_pliegue']:.4f}")

    # ---------------- importancia global y por servicio (dentro de un servicio, "servicio" es constante)
    imp = importancia_global(phi_g, col)
    imp.to_csv(TAB / "m7_3_importancia_shap.csv", index=False)
    imps_srv = {}
    for s in np.unique(srv):
        m = srv == s
        sin_srv = [c for c in phi_g.columns if c != "servicio"]
        imps_srv[NOMBRE_SRV[s]] = importancia_global(phi_g.loc[m, sin_srv].reset_index(drop=True),
                                                     col.loc[m, sin_srv].reset_index(drop=True))
    pd.concat([d.assign(servicio=s) for s, d in imps_srv.items()]).to_csv(TAB / "m7_3_importancia_shap_por_servicio.csv",
                                                                           index=False)

    # ---------------- diagnósticos concretos detrás del "perfil de diagnósticos"
    concept = pd.read_parquet(ROOT / "data" / "raw" / "concept.parquet", columns=["concept_id", "concept_name"])
    nombres_dx = {f"dx_{int(c)}": TRADUCCION_DX.get(n, n) for c, n in zip(concept.concept_id, concept.concept_name)}
    dx = efecto_diagnosticos(phi_g[esquema.clinica], mu[esquema.clinica], nombres_dx, min_n=100, top=5)
    dx_srv = {}
    for s in np.unique(srv):
        m = srv == s
        dx_srv[NOMBRE_SRV[s]] = efecto_diagnosticos(phi_g.loc[m, esquema.clinica].to_numpy(),
                                                    mu.loc[m, esquema.clinica].reset_index(drop=True), nombres_dx,
                                                    min_n=max(30, int(0.01 * m.sum())), top=5)
    pd.concat([dx.assign(servicio="todos")] + [d.assign(servicio=k) for k, d in dx_srv.items()]).to_csv(
        TAB / "m7_3_diagnosticos_shap.csv", index=False)

    # ---------------- figuras globales
    exp = explicacion(phi_g, base, col)
    graficar_beeswarm(exp, FIG / "m7_3_shap_beeswarm.png",
                      "Impacto de cada variable en el riesgo de estancia prolongada (LightGBM, validación)")
    graficar_barras(exp, FIG / "m7_3_shap_bar.png", "Importancia global: impacto medio absoluto por variable")
    graficar_barras_por_servicio(imps_srv, FIG / "m7_3_shap_bar_por_servicio.png")

    # ---------------- casos de estudio: perfiles agregados dentro de cada servicio
    for viejo in ("m7_3_shap_waterfall_alto_riesgo.png", "m7_3_shap_waterfall_bajo_riesgo.png"):
        (FIG / viejo).unlink(missing_ok=True)
    riesgo, filas = SIG(margen), []
    for s in np.unique(srv):
        ms = srv == s
        q95, q05 = np.quantile(riesgo[ms], [0.95, 0.05])
        for nombre, mk in (("alto_riesgo", ms & (riesgo >= q95)), ("bajo_riesgo", ms & (riesgo <= q05))):
            medio, tipico = perfil(phi_g, col, mk)
            medio = medio.drop("servicio")  # constante dentro del servicio: se suma al punto de partida
            base_srv = base + float(phi_g.loc[mk, "servicio"].mean())
            txt = "mayor" if nombre == "alto_riesgo" else "menor"
            graficar_waterfall(medio, tipico, base_srv, FIG / f"m7_3_shap_waterfall_{nombre}_{CORTO_SRV[s]}.png",
                               f"{NOMBRE_SRV[s]} · perfil del 5 % de {txt} riesgo ({int(mk.sum())} snapshots)")
            for v in medio.abs().sort_values(ascending=False).head(8).index:
                filas.append({"servicio": s, "caso": nombre, "n_snapshots": int(mk.sum()),
                              "riesgo_medio_predicho": riesgo[mk].mean(), "tasa_observada": mu[TARGET][mk].mean(),
                              "variable": etiqueta(v), "shap_medio": medio[v], "valor_tipico": formatear_valor(v, tipico[v])})
            log(f"{s} · {nombre}: {int(mk.sum())} snapshots · riesgo predicho {riesgo[mk].mean():.3f} · "
                f"observado {mu[TARGET][mk].mean():.3f}")
    casos = pd.DataFrame(filas)
    casos.to_csv(TAB / "m7_3_casos_estudio.csv", index=False)

    # ---------------- síntesis clínica
    texto = sintesis_medica(imp, 5, dx)
    texto_srv = {s: sintesis_medica(d, 5, dx_srv[s]) for s, d in imps_srv.items()}
    md = ["# Síntesis clínica de la explicabilidad del modelo (Subfase 7.3)", "",
          "Modelo explicado: predicción de **estancia prolongada** (que la estancia supere el percentil 75 de su "
          "servicio), evaluado en pacientes que el modelo no vio durante el entrenamiento.", "",
          f"Riesgo promedio de partida: {SIG(base):.0%}. Cada variable sube o baja el riesgo de cada paciente a partir "
          "de ese punto.", "", "## Las 5 variables más influyentes en todo el hospital", "", texto, ""]
    for s, t in texto_srv.items():
        md += [f"## Dentro de {s.lower()}", "", t, ""]
    md += ["## Diagnósticos que más mueven el riesgo dentro de cada servicio", ""]
    for k, d in dx_srv.items():
        for efecto, titulo in (("aumenta", "elevan el riesgo"), ("reduce", "reducen el riesgo")):
            md.append(f"- **{k}, {titulo}:** " + "; ".join(d[d.efecto == efecto].diagnostico))
    md += ["", "## Perfiles de mayor y menor riesgo dentro de cada servicio", ""]
    for (s, caso), d in casos.groupby(["servicio", "caso"], sort=False):
        top = d.sort_values("shap_medio", key=abs, ascending=False).head(4)
        partes = [f"{r.variable.split(' (')[0].lower()} ({r.valor_tipico})" for r in top.itertuples()]
        md.append(f"- **{NOMBRE_SRV[s]}, {'mayor' if caso == 'alto_riesgo' else 'menor'} riesgo** (riesgo estimado "
                  f"{d.riesgo_medio_predicho.iloc[0]:.0%}, observado {d.tasa_observada.iloc[0]:.0%}): "
                  + ", ".join(partes) + ".")
    md += ["", "## Cómo leerlo", "",
           "- Son asociaciones aprendidas de los registros, no relaciones causales.",
           "- El registro clínico del hospital no incluye nivel de triaje, por lo que el modelo no lo usa.",
           "- La presión del servicio y el calendario describen el contexto operativo, no el estado clínico del paciente.",
           "- Los perfiles son promedios de grupos de pacientes; la explicación individual se calcula para cada paciente "
           "en el momento de usar el modelo."]
    (REP / "m7_3_sintesis_medica.md").write_text("\n".join(md), encoding="utf-8")

    print("\nImportancia global (top 15):")
    print(imp.head(15)[["etiqueta", "dominio", "media_abs_shap", "porcentaje", "spearman_valor_shap"]]
          .round(3).to_string(index=False))
    print("\nImportancia por dominio (%):")
    print(imp.groupby("dominio").porcentaje.sum().sort_values(ascending=False).round(1).to_string())
    for s, d in imps_srv.items():
        print(f"\nTop 8 en {s} (sin la variable servicio):")
        print(d.head(8)[["etiqueta", "porcentaje", "spearman_valor_shap"]].round(3).to_string(index=False))
    for k, d in [("todos", dx)] + list(dx_srv.items()):
        print(f"\nDiagnósticos con mayor efecto · {k} (log-odds, presente − ausente):")
        print(d[["efecto", "diagnostico", "n_snapshots", "diferencia"]].round(3).to_string(index=False))
    print("\n" + "\n".join(md))
    log("Listo")


if __name__ == "__main__":
    main()
