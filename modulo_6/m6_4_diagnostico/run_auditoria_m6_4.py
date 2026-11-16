"""Auditoría de la brecha de sobreajuste de la 6.4 sobre los datos reales (se corre después de run_m6_4.py).

1. KNN: AUC de entrenamiento por resustitución vs leave-one-out (excluyendo el propio punto).
2. Random Forest: ruta de regularización en la submuestra del 20 % de pacientes (misma de la 6.3/6.4),
   elección de la mejor validación con brecha ≤ 0,05, confirmación con todo el entrenamiento y
   reempaquetado de `models/random_forest_pipeline.joblib`.
3. Actualiza `model_benchmark_summary.csv`, `models/manifest.json`, el OOF, la curva de aprendizaje,
   la ROC, y escribe `outputs/m6/hiperparametros_finales.json` (los de la 6.3 salvo el RF regularizado).

Uso (desde la raíz del proyecto):
    python modulo_6/m6_4_diagnostico/run_auditoria_m6_4.py
"""
from __future__ import annotations

import json
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
from modulo_6.m6_1_cv_agrupada import PlieguesPredefinidos  # noqa: E402
from modulo_6.m6_4_diagnostico import (  # noqa: E402
    UMBRAL_BRECHA,
    CacheDiagnostico,
    ajustar_final,
    cargar_mejores,
    curva_aprendizaje,
    diagnosticar,
    graficar_curvas_aprendizaje,
    graficar_roc,
    instanciar,
)
from modulo_6.m6_4_diagnostico.auditoria_brecha import auditar_knn, elegir_con_brecha, ruta_regularizacion  # noqa: E402

TARGET = "y_target_los_largo"
# Ruta de regularización del RF: de la configuración de la 6.3 (max_depth 20, hojas de 24) hacia árboles más simples.
RUTA_RF = [
    {"max_depth": 20, "min_samples_leaf": 24},
    {"max_depth": 16, "min_samples_leaf": 50},
    {"max_depth": 14, "min_samples_leaf": 100},
    {"max_depth": 12, "min_samples_leaf": 100},
    {"max_depth": 12, "min_samples_leaf": 200},
    {"max_depth": 10, "min_samples_leaf": 200},
    {"max_depth": 10, "min_samples_leaf": 400},
    {"max_depth": 8, "min_samples_leaf": 400},
]


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def auc_estratos(y, s, e):
    df = pd.DataFrame({"y": y, "s": s, "e": e}).dropna()
    return float(np.mean([roc_auc_score(d.y, d.s) for _, d in df.groupby("e") if d.y.nunique() == 2]))


def main() -> None:
    warnings.filterwarnings("ignore")
    log("Cargando datos")
    X, esquema = preparar_X(ROOT)
    tr = X[X.en_periodo_estable.astype(bool) & X.es_entrenamiento.astype(bool)].reset_index(drop=True)
    tr["pliegue"] = pd.read_parquet(ROOT / "data" / "m6" / "pliegues_m6_1.parquet").pliegue.to_numpy()
    cols = esquema.todas()
    y, g = tr[TARGET].to_numpy(), tr.person_id.to_numpy()
    estratos = (tr.servicio + "_t" + tr.horas_desde_ingreso.astype(str)).to_numpy()
    pre = construir_preprocesador(esquema)
    mejores = cargar_mejores(ROOT / "outputs" / "m6" / "best_hyperparameters.json")
    tab, fig = ROOT / "outputs" / "m6" / "tablas", ROOT / "outputs" / "m6" / "figuras"

    # Misma submuestra del 20 % de pacientes que run_m6_4 (KNN) y que la búsqueda de la 6.3.
    personas = tr.person_id.drop_duplicates().sample(frac=0.2, random_state=42)
    sub = tr[tr.person_id.isin(personas)].reset_index(drop=True)
    cache_sub = CacheDiagnostico.construir(pre, sub[cols], sub[TARGET].to_numpy(), sub.person_id.to_numpy(),
                                           PlieguesPredefinidos(sub.pliegue))
    log(f"Submuestra 20 %: {len(sub):,} filas")

    # ---------------- 1. KNN
    t0 = time.time()
    previa = pd.read_csv(tab / "m6_4_brecha_por_pliegue.csv")
    previa = previa[previa.modelo == "knn"]  # misma submuestra y pliegues que run_m6_4 → se reutiliza
    knn = auditar_knn(instanciar("knn", mejores["knn"]["hiperparametros"], pre), cache_sub,
                      resustitucion_previa=previa if len(previa) == 5 else None)
    knn.to_csv(tab / "m6_4_auditoria_knn.csv", index=False)
    log(f"KNN ({(time.time() - t0) / 60:.1f} min): train resustitución {knn.train_auc_resustitucion.mean():.4f} · "
        f"train LOO {knn.train_auc_loo.mean():.4f} · val {knn.val_auc.mean():.4f} · "
        f"brecha LOO {knn.brecha_loo.mean():+.4f} (resustitución {knn.brecha_resustitucion.mean():+.4f})")

    # ---------------- 2. RF: ruta de regularización (200 árboles para explorar; el nº de árboles no aumenta la brecha)
    t0 = time.time()
    hp_rf = dict(mejores["random_forest"]["hiperparametros"])
    pipe_rf = instanciar("random_forest", hp_rf, pre)
    ruta = ruta_regularizacion("random_forest", pipe_rf, cache_sub, [{**c, "n_estimators": 200} for c in RUTA_RF])
    ruta.to_csv(tab / "m6_4_ruta_regularizacion_rf.csv", index=False)
    log(f"Ruta de regularización RF en submuestra ({(time.time() - t0) / 60:.1f} min):")
    print(ruta[["max_depth", "min_samples_leaf", "train_auc", "val_auc", "brecha"]].round(4).to_string(index=False),
          flush=True)

    log("Confirmación con todo el entrenamiento")
    cache_full = CacheDiagnostico.construir(pre, tr[cols], y, g, PlieguesPredefinidos(tr.pliegue))
    candidatas = ruta[ruta.brecha <= UMBRAL_BRECHA].sort_values("val_auc", ascending=False)
    if candidatas.empty:
        raise SystemExit("Ninguna configuración del RF cumple la brecha en la submuestra: ampliar RUTA_RF")
    elegida, d_rf = None, None
    for _, c in candidatas.iterrows():
        hp_try = {**hp_rf, "modelo__max_depth": int(c.max_depth), "modelo__min_samples_leaf": int(c.min_samples_leaf)}
        d = diagnosticar("random_forest", instanciar("random_forest", hp_try, pre), cache_full)
        log(f"  max_depth {int(c.max_depth)}, min_samples_leaf {int(c.min_samples_leaf)}: train {d['train_auc']:.4f} · "
            f"val {d['val_auc']:.4f} · brecha {d['brecha']:+.4f}")
        if d["brecha"] <= UMBRAL_BRECHA:
            elegida, d_rf, hp_rf_final = c, d, hp_try
            break
    if elegida is None:
        raise SystemExit("Con todos los datos ninguna candidata cumple la brecha")
    _ = elegir_con_brecha  # (regla documentada: mejor validación con brecha ≤ umbral)

    # ---------------- 3. Reempaquetado y actualización de artefactos
    log("Ajuste final del RF regularizado y reempaquetado")
    pre_full = clone(pre).fit(tr[cols], y)
    final = ajustar_final("random_forest", instanciar("random_forest", hp_rf_final, pre), pre_full,
                          pre_full.transform(tr[cols]).astype(np.float32), y)
    ruta_rf = ROOT / "models" / "random_forest_pipeline.joblib"
    joblib.dump(final, ruta_rf, compress=3)
    joblib.load(ruta_rf).predict(tr[cols].iloc[:5])

    oof = pd.read_parquet(ROOT / "data" / "m6" / "oof_m6_4.parquet")
    oof["oof_random_forest"] = d_rf["oof"]
    oof.to_parquet(ROOT / "data" / "m6" / "oof_m6_4.parquet", index=False)

    bench = pd.read_csv(ROOT / "outputs" / "m6" / "model_benchmark_summary.csv")
    bench["metodo_train_auc"] = "resustitución"
    i = bench.index[bench.modelo == "random_forest"][0]
    s = d_rf["oof"]
    bench.loc[i, ["train_auc", "val_auc", "val_auc_std", "brecha_train_val"]] = [d_rf["train_auc"], d_rf["val_auc"],
                                                                                 d_rf["val_auc_std"], d_rf["brecha"]]
    bench.loc[i, ["auc_oof", "auc_oof_servicio_tau"]] = [roc_auc_score(y, s), auc_estratos(y, s, estratos)]
    bench.loc[i, "hiperparametros"] = json.dumps(hp_rf_final, default=str)
    bench.loc[i, "mb"] = round(ruta_rf.stat().st_size / 1e6, 1)
    bench.loc[i, "metodo_train_auc"] = "resustitución (RF regularizado en la auditoría)"
    j = bench.index[bench.modelo == "knn"][0]
    bench.loc[j, "train_auc"] = knn.train_auc_loo.mean()
    bench.loc[j, "brecha_train_val"] = knn.brecha_loo.mean()
    bench.loc[j, "metodo_train_auc"] = "leave-one-out (excluye el propio punto)"
    bench["sobreajuste_gt_005"] = bench.brecha_train_val > UMBRAL_BRECHA
    bench = bench.sort_values("val_auc", ascending=False)
    bench.to_csv(ROOT / "outputs" / "m6" / "model_benchmark_summary.csv", index=False)

    man_p = ROOT / "models" / "manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8"))
    for _, r in bench.iterrows():
        if r.modelo in man["modelos"]:
            for k in ("train_auc", "val_auc", "val_auc_std", "brecha_train_val", "sobreajuste_gt_005", "auc_oof",
                      "auc_oof_servicio_tau", "mb", "metodo_train_auc"):
                v = r[k]
                man["modelos"][r.modelo][k] = v.item() if hasattr(v, "item") else v
    man["modelos"]["random_forest"]["hiperparametros"] = hp_rf_final
    man["auditoria_brecha"] = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "knn": "AUC de entrenamiento medido leave-one-out; el de resustitución (≈1) es un artefacto de weights='distance'",
        "random_forest": f"regularizado: max_depth {int(elegida.max_depth)}, min_samples_leaf {int(elegida.min_samples_leaf)}",
    }
    man_p.write_text(json.dumps(man, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    finales = {m: info["hiperparametros"] for m, info in mejores.items()}
    finales["random_forest"] = hp_rf_final
    (ROOT / "outputs" / "m6" / "hiperparametros_finales.json").write_text(json.dumps(
        {"nota": "Hiperparámetros de la 6.3, salvo random_forest regularizado en la auditoría de la 6.4 (brecha ≤ 0,05)",
         "modelos": finales}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # Curva de aprendizaje y ROC con el RF regularizado
    curvas = pd.read_csv(tab / "m6_4_curvas_aprendizaje.csv")
    pipe_rf_final = instanciar("random_forest", hp_rf_final, pre)
    curvas = pd.concat([curvas[curvas.modelo != "random_forest"],
                        curva_aprendizaje("random_forest", pipe_rf_final, cache_full, diagnostico=d_rf)])
    orden = bench.modelo.tolist()
    curvas["_o"] = curvas.modelo.map({m: k for k, m in enumerate(orden)})
    curvas = curvas.sort_values(["_o", "pliegue", "fraccion"]).drop(columns="_o")
    curvas.to_csv(tab / "m6_4_curvas_aprendizaje.csv", index=False)
    graficar_curvas_aprendizaje(curvas, fig / "m6_4_curvas_aprendizaje.png")
    graficar_roc(y, {c.replace("oof_", ""): oof[c].to_numpy() for c in oof.columns if c.startswith("oof_")},
                 fig / "m6_4_roc_comparativa.png")

    print("\nResumen auditado (model_benchmark_summary.csv):")
    print(bench[["modelo", "train_auc", "val_auc", "brecha_train_val", "sobreajuste_gt_005", "auc_oof",
                 "auc_oof_servicio_tau", "metodo_train_auc"]].round(4).to_string(index=False))
    log("Listo")


if __name__ == "__main__":
    main()
