"""Subfase 7.2 sobre los datos reales: calibración + umbral de costo asimétrico, con prueba única sellada.

Fases (por defecto, las tres en orden):
  desarrollo  OOF de la 6.4 (entrenamiento estable): calibración cruzada por pliegue (isotónica / sigmoide),
              elección del método (menor Brier), barrido θ ∈ [0,01; 0,99] con C = 5·FN + 1·FP, θ* global y por
              servicio. **Congela** los umbrales en outputs/m7/m7_2_umbrales_congelados.json.
  final       CalibratedClassifierCV(Pipeline, método elegido, cv = pliegues agrupados de la 6.1, ensemble=True)
              con todo el entrenamiento estable → models/<modelo>_calibrado.joblib.
  prueba      Evalúa UNA vez en la prueba temporal (periodo estable posterior al corte) con los umbrales congelados.
              Deja un sello (outputs/m7/m7_2_prueba_unica.json); repetirla exige --repetir-prueba y queda registrado.

Uso (desde la raíz del proyecto):
    python modulo_7/m7_2_calibracion_costo/run_m7_2.py [--fase desarrollo|final|prueba|todo]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modulo_7.m7_2_calibracion_costo import (  # noqa: E402
    C_FN,
    C_FP,
    METODOS,
    ajustar_calibrado_final,
    barrido_umbrales,
    calibrar_cruzado,
    desempeno_umbral,
    graficar_costo,
    graficar_fiabilidad,
    metricas_calibracion,
    politicas_referencia,
    umbral_optimo,
)

TARGET = "y_target_los_largo"
N_MEJORES = 3
OUT, TAB, FIG = ROOT / "outputs" / "m7", ROOT / "outputs" / "m7" / "tablas", ROOT / "outputs" / "m7" / "figuras"
CONGELADOS = OUT / "m7_2_umbrales_congelados.json"
SELLO = OUT / "m7_2_prueba_unica.json"


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def reglas_de(cfg: dict, servicio: np.ndarray) -> dict[str, np.ndarray | float]:
    """Reglas de alerta pre-especificadas para un modelo (todas fijadas en desarrollo)."""
    por_srv = np.vectorize(cfg["theta_por_servicio"].get)(servicio).astype(float)
    return {"θ = 0,5": 0.5, "θ* global": cfg["theta_global"], "θ* por servicio": por_srv,
            f"θ Bayes = {C_FP / (C_FP + C_FN):.3f}": C_FP / (C_FP + C_FN)}


def tabla_reglas(y, p, servicio, reglas: dict, conjunto: str, modelo: str) -> pd.DataFrame:
    filas = []
    for nombre, t in reglas.items():
        for srv, m in [("todos", np.ones(len(y), bool))] + [(s, servicio == s) for s in np.unique(servicio)]:
            tt = t[m] if isinstance(t, np.ndarray) else t
            d = desempeno_umbral(y[m], p[m], tt)
            d["umbral"] = float(np.unique(tt)[0]) if np.size(np.unique(tt)) == 1 else np.nan
            ref = politicas_referencia(y[m])
            filas.append({"conjunto": conjunto, "modelo": modelo, "servicio": srv, "regla": nombre, **d,
                          "costo_alertar_a_todos": ref["alertar_a_todos"], "costo_no_alertar_nunca": ref["no_alertar_nunca"]})
    df = pd.DataFrame(filas)
    base = df[df.regla == "θ = 0,5"].set_index("servicio").costo_por_100
    df["ahorro_vs_0_5"] = df.servicio.map(base) - df.costo_por_100
    df["ahorro_vs_mejor_sin_modelo"] = df[["costo_alertar_a_todos", "costo_no_alertar_nunca"]].min(axis=1) - df.costo_por_100
    return df


# ------------------------------------------------------------------------------------------------ desarrollo
def fase_desarrollo() -> None:
    oof_p = ROOT / "data" / "m6" / "oof_m6_4.parquet"
    oof = pd.read_parquet(oof_p)
    resumen = pd.read_csv(OUT / "multimetric_evaluation_summary.csv")
    mejores = resumen[resumen.n == len(oof)].modelo.head(N_MEJORES).tolist()  # KNN (20 % pac.) no compite aquí
    y, pl, srv = oof[TARGET].to_numpy(), oof.pliegue.to_numpy(), oof.servicio.to_numpy()
    log(f"Desarrollo (OOF): {len(oof):,} snapshots · prevalencia {y.mean():.4f} · modelos {mejores}")

    filas_cal, versiones_todas, congelados, barridos, tablas, sens = [], {}, {}, {}, [], []
    for m in mejores:
        s = oof[f"oof_{m}"].to_numpy(float)
        versiones = {} if m.startswith("svm") else {"sin_calibrar": s}
        for metodo in METODOS:
            versiones[metodo] = calibrar_cruzado(s, y, pl, metodo)
        for v, p in versiones.items():
            for nombre_srv, mk in [("todos", np.ones(len(y), bool))] + [(x, srv == x) for x in np.unique(srv)]:
                filas_cal.append({"modelo": m, "version": v, "servicio": nombre_srv, **metricas_calibracion(y[mk], p[mk])})
        cal = pd.DataFrame([f for f in filas_cal if f["modelo"] == m and f["servicio"] == "todos" and f["version"] in METODOS])
        metodo = cal.sort_values("brier").version.iloc[0]
        p = versiones[metodo]
        b = barrido_umbrales(y, p)
        barridos[m] = b
        theta = umbral_optimo(b)
        theta_srv = {x: umbral_optimo(barrido_umbrales(y[srv == x], p[srv == x])) for x in np.unique(srv)}
        theta10 = umbral_optimo(barrido_umbrales(y, p, c_fn=10))
        congelados[m] = {"metodo": metodo, "theta_global": theta, "theta_por_servicio": theta_srv,
                         "theta_global_cfn10_sensibilidad": theta10}
        versiones_todas[m] = versiones
        tablas.append(tabla_reglas(y, p, srv, reglas_de(congelados[m], srv), "desarrollo (OOF)", m))
        sens.append(b.assign(modelo=m))
        log(f"{m}: método {metodo} · θ* = {theta:.2f} · por servicio {theta_srv} · (c_FN = 10: θ* = {theta10:.2f})")

    cal = pd.DataFrame(filas_cal)
    cal.to_csv(TAB / "m7_2_calibracion_desarrollo.csv", index=False)
    pd.concat(tablas).to_csv(TAB / "m7_2_umbrales_desarrollo.csv", index=False)
    pd.concat(sens).to_csv(TAB / "m7_2_barrido_costo_desarrollo.csv", index=False)

    graficar_fiabilidad(y, versiones_todas, FIG / "m7_2_fiabilidad_desarrollo.png",
                        "Diagramas de fiabilidad · desarrollo (OOF, calibración cruzada por pliegue)")
    best = mejores[0]
    por_srv = {f"{best} · {x}": {v: np.where(srv == x, p, np.nan) for v, p in versiones_todas[best].items()}
               for x in np.unique(srv)}
    graficar_fiabilidad(y, por_srv, FIG / "m7_2_fiabilidad_por_servicio_desarrollo.png",
                        f"Fiabilidad por servicio · {best} · desarrollo (OOF)")
    graficar_costo(barridos, politicas_referencia(y), FIG / "m7_2_costo_umbral_desarrollo.png",
                   "Costo esperado vs umbral · desarrollo (OOF calibrado)")

    CONGELADOS.write_text(json.dumps({
        "congelado": datetime.now().isoformat(timespec="seconds"),
        "c_fn": C_FN, "c_fp": C_FP, "regla": "alerta si p_calibrada ≥ θ",
        "criterio_metodo": "menor Brier en OOF con calibración cruzada por pliegue",
        "criterio_umbral": "menor costo esperado por 100 en OOF calibrado, θ ∈ [0,01; 0,99] paso 0,01",
        "oof_sha256": sha256(oof_p), "modelos": congelados}, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Umbrales congelados en {CONGELADOS.relative_to(ROOT)}")

    t = pd.concat(tablas)
    print("\nCalibración (desarrollo, todos los servicios):")
    print(cal[cal.servicio == "todos"][["modelo", "version", "p_media", "prevalencia", "brier", "brier_skill", "ece",
                                        "auc_roc"]].round(4).to_string(index=False))
    print("\nECE por servicio (desarrollo):")
    print(cal[cal.servicio != "todos"].pivot_table(index=["modelo", "version"], columns="servicio", values="ece")
          .round(4).to_string())
    print("\nθ = 0,5 vs θ* (desarrollo):")
    print(t[["modelo", "servicio", "regla", "umbral", "sensibilidad", "especificidad", "vpp", "alertas_por_100",
             "costo_por_100", "ahorro_vs_0_5", "costo_alertar_a_todos", "ahorro_vs_mejor_sin_modelo"]]
          .round(3).to_string(index=False))


# ------------------------------------------------------------------------------------------------ final y prueba
def cargar_X():
    from modulo_5.m5_4_pipeline import preparar_X
    X, esquema = preparar_X(ROOT)
    estable = X.en_periodo_estable.astype(bool)
    tr = X[estable & X.es_entrenamiento.astype(bool)].reset_index(drop=True)
    te = X[estable & ~X.es_entrenamiento.astype(bool)].reset_index(drop=True)
    pl = pd.read_parquet(ROOT / "data" / "m6" / "pliegues_m6_1.parquet")
    if not pl.snapshot_id.reset_index(drop=True).equals(tr.snapshot_id):
        raise ValueError("Los pliegues de la 6.1 no están alineados con el entrenamiento estable")
    tr["pliegue"] = pl.pliegue.to_numpy()
    return tr, te, esquema


def fase_final() -> None:
    from modulo_5.m5_4_pipeline import construir_preprocesador
    from modulo_6.m6_4_diagnostico import instanciar
    cfg = json.loads(CONGELADOS.read_text(encoding="utf-8"))
    hp = json.loads((ROOT / "outputs" / "m6" / "hiperparametros_finales.json").read_text(encoding="utf-8"))["modelos"]
    tr, _, esquema = cargar_X()
    cols = esquema.todas()
    man_p = ROOT / "models" / "manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8"))
    man.setdefault("calibrados", {})
    for m, c in cfg["modelos"].items():
        t0 = time.time()
        cal = ajustar_calibrado_final(instanciar(m, hp[m], construir_preprocesador(esquema)), tr[cols],
                                      tr[TARGET].to_numpy(), tr.pliegue.to_numpy(), c["metodo"])
        ruta = ROOT / "models" / f"{m}_calibrado.joblib"
        joblib.dump(cal, ruta, compress=3)
        p = joblib.load(ruta).predict_proba(tr[cols].iloc[:5])[:, 1]
        assert ((p >= 0) & (p <= 1)).all()
        man["calibrados"][m] = {"archivo": f"models/{m}_calibrado.joblib", "metodo": c["metodo"],
                                "cv": "5 pliegues agrupados por paciente de la 6.1 (ensemble=True)",
                                "theta_global": c["theta_global"], "theta_por_servicio": c["theta_por_servicio"],
                                "c_fn": cfg["c_fn"], "c_fp": cfg["c_fp"], "salida": "predict_proba (calibrada)",
                                "mb": round(ruta.stat().st_size / 1e6, 1), "min_ajuste": round((time.time() - t0) / 60, 1)}
        man_p.write_text(json.dumps(man, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        log(f"{m}: CalibratedClassifierCV({c['metodo']}) ajustado en {(time.time() - t0) / 60:.1f} min → {ruta.name}")


def fase_prueba(repetir: bool) -> None:
    if SELLO.exists() and not repetir:
        raise SystemExit(f"La prueba ya se evaluó ({SELLO.relative_to(ROOT)}). Repetirla exige --repetir-prueba "
                         "y anula su carácter de evaluación única.")
    cfg = json.loads(CONGELADOS.read_text(encoding="utf-8"))
    _, te, esquema = cargar_X()
    cols = esquema.todas()
    y, srv = te[TARGET].to_numpy(), te.servicio.to_numpy()
    log(f"PRUEBA (única): {len(te):,} snapshots · {te.person_id.nunique():,} pacientes · prevalencia {y.mean():.4f}")
    filas_cal, tablas, versiones, probas = [], [], {}, {}
    for m, c in cfg["modelos"].items():
        cal = joblib.load(ROOT / "models" / f"{m}_calibrado.joblib")
        p = cal.predict_proba(te[cols])[:, 1]
        v = {c["metodo"]: p}
        if not m.startswith("svm"):
            v = {"sin_calibrar": joblib.load(ROOT / "models" / f"{m}_pipeline.joblib").predict_proba(te[cols])[:, 1], **v}
        versiones[m], probas[f"p_{m}"] = v, p
        for nombre, pp in v.items():
            for s_n, mk in [("todos", np.ones(len(y), bool))] + [(x, srv == x) for x in np.unique(srv)]:
                filas_cal.append({"modelo": m, "version": nombre, "servicio": s_n, **metricas_calibracion(y[mk], pp[mk])})
        tablas.append(tabla_reglas(y, p, srv, reglas_de(c, srv), "prueba (única)", m))
    cal = pd.DataFrame(filas_cal)
    t = pd.concat(tablas)
    cal.to_csv(TAB / "m7_2_calibracion_prueba.csv", index=False)
    t.to_csv(TAB / "m7_2_umbrales_prueba.csv", index=False)
    pd.DataFrame({"snapshot_id": te.snapshot_id, "person_id": te.person_id, TARGET: y, "servicio": srv,
                  "horas_desde_ingreso": te.horas_desde_ingreso, **probas}).to_parquet(
        ROOT / "data" / "m7" / "probabilidades_prueba_m7_2.parquet", index=False)
    graficar_fiabilidad(y, versiones, FIG / "m7_2_fiabilidad_prueba.png",
                        "Diagramas de fiabilidad · prueba temporal (evaluación única)")
    previo = json.loads(SELLO.read_text(encoding="utf-8")) if SELLO.exists() else None
    SELLO.write_text(json.dumps({
        "evaluada": datetime.now().isoformat(timespec="seconds"), "n": int(len(te)),
        "umbrales_congelados": cfg["congelado"], "congelados_sha256": sha256(CONGELADOS),
        "repeticiones_previas": ([previo] + previo.get("repeticiones_previas", [])) if previo else []},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nCalibración (prueba, todos los servicios):")
    print(cal[cal.servicio == "todos"][["modelo", "version", "p_media", "prevalencia", "brier", "brier_skill", "ece",
                                        "auc_roc", "auc_pr"]].round(4).to_string(index=False))
    print("\nθ = 0,5 vs θ* (prueba):")
    print(t[["modelo", "servicio", "regla", "umbral", "sensibilidad", "especificidad", "vpp", "alertas_por_100",
             "costo_por_100", "ahorro_vs_0_5", "costo_alertar_a_todos", "ahorro_vs_mejor_sin_modelo"]]
          .round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fase", choices=["desarrollo", "final", "prueba", "todo"], default="todo")
    ap.add_argument("--repetir-prueba", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    for d in (TAB, FIG, ROOT / "data" / "m7"):
        d.mkdir(parents=True, exist_ok=True)
    if a.fase in ("desarrollo", "todo"):
        fase_desarrollo()
    if a.fase in ("final", "todo"):
        fase_final()
    if a.fase in ("prueba", "todo"):
        fase_prueba(a.repetir_prueba)
    log("Listo")


if __name__ == "__main__":
    main()
