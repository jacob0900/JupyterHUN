"""Ejecuta la subfase 6.4 sobre los datos reales.

1. Carga `outputs/m6/best_hyperparameters.json` e instancia los 7 Pipelines (mejor modelo de cada familia).
2. Reentrena cada uno en los 5 pliegues de la 6.1 con **todo** el entrenamiento estable: AUC train vs val,
   brecha y predicciones fuera de pliegue (OOF). Advierte si la brecha supera 0,05.
3. Curvas de aprendizaje (10–100 %) de los 3 mejores y ROC comparativa (OOF) de los 7.
4. Ajuste final con todo el entrenamiento y serialización en `models/<modelo>_pipeline.joblib`.

KNN se entrena y evalúa con el 20 % de los pacientes: predecir con 433 000 vecinos de referencia
toma ~45 min por pliegue (6.2), y su .joblib guardaría todas las filas de entrenamiento.

Escribe:
    models/<modelo>_pipeline.joblib            7 Pipelines finales (preprocesador + modelo)
    models/manifest.json                       métricas, hiperparámetros, columnas de entrada y versiones
    data/m6/oof_m6_4.parquet                   predicciones OOF por snapshot (nivel paciente: para M7)
    outputs/m6/model_benchmark_summary.csv     resumen: hiperparámetros + métricas finales
    outputs/m6/tablas/m6_4_brecha_por_pliegue.csv, m6_4_curvas_aprendizaje.csv
    outputs/m6/figuras/m6_4_curvas_aprendizaje.png, m6_4_roc_comparativa.png

Uso (desde la raíz del proyecto):
    python modulo_6/m6_4_diagnostico/run_m6_4.py [--frac-knn 0.2]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modulo_5.m5_4_pipeline import construir_preprocesador, preparar_X  # noqa: E402
from modulo_6.m6_1_cv_agrupada import PlieguesPredefinidos, verify_group_isolation  # noqa: E402
from modulo_6.m6_3_optuna import FAMILIAS  # noqa: E402
from modulo_6.m6_4_diagnostico import (  # noqa: E402
    UMBRAL_BRECHA,
    CacheDiagnostico,
    advertir_brechas,
    ajustar_final,
    cargar_mejores,
    curva_aprendizaje,
    diagnosticar,
    graficar_curvas_aprendizaje,
    graficar_roc,
    instanciar,
)

TARGET = "y_target_los_largo"


def auc_estratos(y, s, estratos) -> float:
    df = pd.DataFrame({"y": y, "s": s, "e": estratos}).dropna()
    return float(np.mean([roc_auc_score(d.y, d.s) for _, d in df.groupby("e") if d.y.nunique() == 2]))


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac-knn", type=float, default=0.2)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    import sklearn, lightgbm, xgboost  # noqa: E401

    log("Cargando maestro (M5), pliegues (6.1) y mejores hiperparámetros (6.3)")
    X, esquema = preparar_X(ROOT)
    tr = X[X.en_periodo_estable.astype(bool) & X.es_entrenamiento.astype(bool)].reset_index(drop=True)
    pl = pd.read_parquet(ROOT / "data" / "m6" / "pliegues_m6_1.parquet")
    if not pl.snapshot_id.equals(tr.snapshot_id):
        raise SystemExit("Los pliegues de la 6.1 no están alineados con el maestro: volver a correr run_m6_1.py")
    tr["pliegue"] = pl.pliegue.to_numpy()
    cols = esquema.todas()
    y, g = tr[TARGET].to_numpy(), tr.person_id.to_numpy()
    estratos = (tr.servicio + "_t" + tr.horas_desde_ingreso.astype(str)).to_numpy()
    cv = PlieguesPredefinidos(tr.pliegue)
    verify_group_isolation(tr[cols], y, g, cv, n_splits=5)
    pre = construir_preprocesador(esquema)
    mejores = cargar_mejores(ROOT / "outputs" / "m6" / "best_hyperparameters.json")
    log(f"Modelos (mejor de cada familia): {list(mejores)}")

    t0 = time.time()
    cache = CacheDiagnostico.construir(pre, tr[cols], y, g, cv)
    log(f"Preprocesador ajustado por pliegue con todo el entrenamiento ({len(tr):,} filas): {time.time() - t0:.0f} s")

    cache_knn = idx_knn = None
    if "knn" in mejores:
        personas = tr.person_id.drop_duplicates().sample(frac=a.frac_knn, random_state=42)
        idx_knn = np.flatnonzero(tr.person_id.isin(personas).to_numpy())
        sub = tr.iloc[idx_knn].reset_index(drop=True)
        cache_knn = CacheDiagnostico.construir(pre, sub[cols], sub[TARGET].to_numpy(), sub.person_id.to_numpy(),
                                               PlieguesPredefinidos(sub.pliegue))
        log(f"KNN: submuestra de {len(sub):,} filas ({a.frac_knn:.0%} de los pacientes)")

    dirs = {"models": ROOT / "models", "fig": ROOT / "outputs" / "m6" / "figuras", "tab": ROOT / "outputs" / "m6" / "tablas"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ---------------- 2. brecha + OOF
    diag, pipes = {}, {}
    oof = pd.DataFrame({"snapshot_id": tr.snapshot_id, "person_id": g, "pliegue": tr.pliegue, TARGET: y,
                        "servicio": tr.servicio, "horas_desde_ingreso": tr.horas_desde_ingreso})
    for nombre, info in mejores.items():
        pipes[nombre] = instanciar(nombre, info["hiperparametros"], pre)
        if nombre == "knn":
            d = diagnosticar(nombre, pipes[nombre], cache_knn)
            s = np.full(len(tr), np.nan)
            s[idx_knn] = d["oof"]
            d["oof"] = s
        else:
            d = diagnosticar(nombre, pipes[nombre], cache)
        diag[nombre] = d
        oof[f"oof_{nombre}"] = d["oof"]
        log(f"{nombre:<13} train {d['train_auc']:.4f} · val {d['val_auc']:.4f} (± {d['val_auc_std']:.4f}) · "
            f"brecha {d['brecha']:+.4f} · {d['segundos_cv'] / 60:.1f} min")
    oof.to_parquet(ROOT / "data" / "m6" / "oof_m6_4.parquet", index=False)
    pd.concat([d["por_pliegue"].assign(modelo=n) for n, d in diag.items()]).to_csv(
        dirs["tab"] / "m6_4_brecha_por_pliegue.csv", index=False)

    resumen = pd.DataFrame([{"modelo": n, "train_auc": d["train_auc"], "val_auc": d["val_auc"],
                             "val_auc_std": d["val_auc_std"], "brecha": d["brecha"]} for n, d in diag.items()])
    print()
    sobre = advertir_brechas(resumen, UMBRAL_BRECHA)
    print()

    # ---------------- 3. curvas de aprendizaje (3 mejores) y ROC (todos)
    top3 = resumen.sort_values("val_auc", ascending=False).modelo.head(3).tolist()
    log(f"Curvas de aprendizaje para {top3}")
    curvas = []
    for nombre in top3:
        c = cache_knn if nombre == "knn" else cache
        curvas.append(curva_aprendizaje(nombre, pipes[nombre], c, diagnostico=diag[nombre]))
        log(f"  {nombre} listo")
    curvas = pd.concat(curvas)
    curvas.to_csv(dirs["tab"] / "m6_4_curvas_aprendizaje.csv", index=False)
    graficar_curvas_aprendizaje(curvas, dirs["fig"] / "m6_4_curvas_aprendizaje.png")
    roc = graficar_roc(y, {n: d["oof"] for n, d in diag.items()}, dirs["fig"] / "m6_4_roc_comparativa.png")

    # ---------------- 4. ajuste final y empaquetado
    log("Ajuste final con todo el entrenamiento")
    t0 = time.time()
    pre_full = clone(pre).fit(tr[cols], y)
    Z_full = pre_full.transform(tr[cols]).astype(np.float32)
    log(f"  preprocesador final: {time.time() - t0:.0f} s")
    manifest = {"creado": datetime.now().isoformat(timespec="seconds"),
                "tarea": "paciente: y_target_los_largo (LOS > P75 del servicio)",
                "entrenado_con": "es_entrenamiento & en_periodo_estable", "columnas_entrada": cols,
                "esquema_ramas": esquema.__dict__, "salida": {}, "versiones": {
                    "python": platform.python_version(), "scikit-learn": sklearn.__version__,
                    "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__}, "modelos": {}}
    filas = []
    for nombre, info in mejores.items():
        t0 = time.time()
        if nombre == "knn":
            sub = tr.iloc[idx_knn]
            pre_k = clone(pre).fit(sub[cols], sub[TARGET])
            final = ajustar_final(nombre, pipes[nombre], pre_k, pre_k.transform(sub[cols]).astype(np.float32),
                                  sub[TARGET].to_numpy())
            n_final = len(sub)
        else:
            final = ajustar_final(nombre, pipes[nombre], pre_full, Z_full, y)
            n_final = len(tr)
        seg = time.time() - t0
        ruta = dirs["models"] / f"{nombre}_pipeline.joblib"
        joblib.dump(final, ruta, compress=3)
        # Verificación: el Pipeline cargado predice sobre las columnas crudas
        cargado = joblib.load(ruta)
        _ = cargado.predict(tr[cols].iloc[:5])
        d, s = diag[nombre], diag[nombre]["oof"]
        ok = ~np.isnan(s)
        fila = {
            "modelo": nombre, "familia": next(f for f, ms in FAMILIAS.items() if nombre in ms),
            "auc_cv_busqueda_6_3": info["mejor_auc_cv"],
            "train_auc": d["train_auc"], "val_auc": d["val_auc"], "val_auc_std": d["val_auc_std"],
            "brecha_train_val": d["brecha"], "sobreajuste_gt_005": d["brecha"] > UMBRAL_BRECHA,
            "auc_oof": roc_auc_score(y[ok], s[ok]), "auc_oof_servicio_tau": auc_estratos(y[ok], s[ok], estratos[ok]),
            "n_train_final": n_final, "min_cv": round(d["segundos_cv"] / 60, 2), "min_ajuste_final": round(seg / 60, 2),
            "archivo": str(ruta.relative_to(ROOT)).replace("\\", "/"), "mb": round(ruta.stat().st_size / 1e6, 1),
            "hiperparametros": json.dumps(info["hiperparametros"], default=str),
        }
        filas.append(fila)
        manifest["modelos"][nombre] = {k: (v if not isinstance(v, (np.floating, np.bool_)) else v.item())
                                       for k, v in fila.items() if k != "hiperparametros"}
        manifest["modelos"][nombre]["hiperparametros"] = info["hiperparametros"]
        manifest["salida"][nombre] = "decision_function" if nombre.startswith("svm") else "predict_proba"
        log(f"  {nombre:<13} → {fila['archivo']} ({fila['mb']} MB, {seg / 60:.1f} min)")
        bench = pd.DataFrame(filas).sort_values("val_auc", ascending=False)
        bench.to_csv(ROOT / "outputs" / "m6" / "model_benchmark_summary.csv", index=False)
    if "knn" in mejores:
        manifest["advertencia_knn"] = ("knn_pipeline.joblib contiene las filas (transformadas) de entrenamiento de "
                                       f"{len(idx_knn):,} snapshots: es dato a nivel paciente, no compartir.")
    (dirs["models"] / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
                                                  encoding="utf-8")

    print("\nResumen (model_benchmark_summary.csv):")
    print(bench[["modelo", "train_auc", "val_auc", "brecha_train_val", "sobreajuste_gt_005", "auc_oof",
                 "auc_oof_servicio_tau", "mb"]].round(4).to_string(index=False))
    if sobre:
        print(f"\n⚠️  Modelos con brecha > {UMBRAL_BRECHA}: {', '.join(sobre)}")
    log("Listo")


if __name__ == "__main__":
    main()
