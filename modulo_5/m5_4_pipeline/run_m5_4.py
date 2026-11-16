"""Ejecuta la subfase 5.4 sobre los datos reales.

1. Arma X (M1 + 5.1 + 5.2 + 5.3) y parte en entrenamiento / prueba (solo periodo estable).
2. Validación cruzada `StratifiedGroupKFold(5)` por `person_id` dentro de entrenamiento:
   el Pipeline se clona y se ajusta en cada fold (todo el preprocesamiento incluido).
3. Verificación anti-fuga sobre datos reales en el primer fold.
4. Ajuste final con todo el entrenamiento y evaluación fuera de muestra en prueba.

Escribe:
    data/m5/pipeline_m5_4_plantilla.joblib   Pipeline SIN ajustar (lo que usa M6 para su CV / Optuna)
    data/m5/pipeline_m5_4_ajustado.joblib    Pipeline ajustado con todo el entrenamiento (baseline)
    outputs/m5/esquema_m5_4.json             columnas de cada rama + parámetros
    outputs/m5/tablas/m5_4_cv.csv            AUC-ROC y AUC-PR por fold
    outputs/m5/tablas/m5_4_prueba.csv        métricas en prueba por servicio × τ (con IC bootstrap)
    outputs/m5/tablas/m5_4_verificacion_fuga.csv
    outputs/m5/tablas/m5_4_variables_salida.csv  nombres de las columnas que salen del preprocesador

Uso (desde la raíz del proyecto):
    python modulo_5/m5_4_pipeline/run_m5_4.py            # completo (~varios minutos)
    python modulo_5/m5_4_pipeline/run_m5_4.py --rapido   # 15 % de las personas, para probar
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modulo_5.m5_4_pipeline import construir_pipeline, preparar_X, verificar_pipeline_sin_fuga  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs" / "m5"
TARGET = "y_target_los_largo"
SEMILLA = 42


def ic_bootstrap(y, p, metrica, n=500, semilla=SEMILLA):
    rng = np.random.default_rng(semilla)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(metrica(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="usa el 15 %% de las personas (prueba de humo)")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    warnings.filterwarnings("ignore", message="solo hay")

    t0 = time.time()
    print("· Armando X (M1 + 5.1 + 5.2 + 5.3)")
    X, esquema = preparar_X(ROOT)
    X = X[X.en_periodo_estable.astype(bool)].reset_index(drop=True)
    if a.rapido:
        personas = X.person_id.drop_duplicates().sample(frac=0.15, random_state=SEMILLA)
        X = X[X.person_id.isin(personas)].reset_index(drop=True)
    cols = esquema.todas()
    tr = X.es_entrenamiento.astype(bool).to_numpy()
    X_tr, y_tr, g_tr = X.loc[tr, cols], X.loc[tr, TARGET].to_numpy(), X.loc[tr, "person_id"].to_numpy()
    X_te, y_te = X.loc[~tr, cols], X.loc[~tr, TARGET].to_numpy()
    print(f"  entrenamiento {len(X_tr):,} · prueba {len(X_te):,} · variables de entrada {len(cols)} "
          f"(num {len(esquema.num)}, num_log {len(esquema.num_log)}, cat {len(esquema.cat)}, clínica 1)")

    plantilla = construir_pipeline(esquema)
    cv = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=SEMILLA)

    print(f"· Validación cruzada StratifiedGroupKFold({a.folds}) por person_id")
    filas_cv, verif = [], None
    for k, (i_tr, i_va) in enumerate(cv.split(X_tr, y_tr, groups=g_tr), 1):
        assert not set(g_tr[i_tr]) & set(g_tr[i_va]), "un paciente quedó en entrenamiento y validación"
        if k == 1:
            print("  · verificación anti-fuga en el fold 1 (datos reales)")
            verif = verificar_pipeline_sin_fuga(plantilla, X_tr.iloc[i_tr], y_tr[i_tr], X_tr.iloc[i_va], esquema)
            print("   ", {c: v for c, v in verif.items()})
            if not verif["ok"]:
                raise SystemExit("✗ Se detectó fuga en el Pipeline: no continuar")
        p = clone(plantilla).fit(X_tr.iloc[i_tr], y_tr[i_tr])
        prob = p.predict_proba(X_tr.iloc[i_va])[:, 1]
        filas_cv.append({"fold": k, "n_val": len(i_va), "prevalencia": y_tr[i_va].mean(),
                         "auc_roc": roc_auc_score(y_tr[i_va], prob), "auc_pr": average_precision_score(y_tr[i_va], prob)})
        print(f"  fold {k}: AUC-ROC {filas_cv[-1]['auc_roc']:.4f} · AUC-PR {filas_cv[-1]['auc_pr']:.4f} "
              f"({time.time() - t0:.0f} s)")
    cvdf = pd.DataFrame(filas_cv)

    print("· Ajuste final con todo el entrenamiento y evaluación en prueba")
    final = clone(plantilla).fit(X_tr, y_tr)
    prob_te = final.predict_proba(X_te)[:, 1]
    te = X.loc[~tr, ["servicio", "horas_desde_ingreso"]].assign(y=y_te, p=prob_te)
    filas = []
    grupos = [("TODOS", "todos", te)] + [(s, str(h), d) for (s, h), d in te.groupby(["servicio", "horas_desde_ingreso"])]
    for s, h, d in grupos:
        lo, hi = ic_bootstrap(d.y, d.p, roc_auc_score)
        filas.append({"servicio": s, "tau_h": h, "n": len(d), "prevalencia": d.y.mean(),
                      "auc_roc": roc_auc_score(d.y, d.p), "auc_roc_ic95": f"[{lo:.3f}, {hi:.3f}]",
                      "auc_pr": average_precision_score(d.y, d.p)})
    prueba = pd.DataFrame(filas)

    tab = OUT / "tablas"
    tab.mkdir(parents=True, exist_ok=True)
    sufijo = "_rapido" if a.rapido else ""
    cvdf.to_csv(tab / f"m5_4_cv{sufijo}.csv", index=False)
    prueba.to_csv(tab / f"m5_4_prueba{sufijo}.csv", index=False)
    pd.DataFrame([verif]).to_csv(tab / f"m5_4_verificacion_fuga{sufijo}.csv", index=False)
    nombres = final.named_steps["preprocesador"].get_feature_names_out()
    pd.Series(nombres, name="variable").to_csv(tab / f"m5_4_variables_salida{sufijo}.csv", index=False)
    if not a.rapido:
        joblib.dump(plantilla, DATA / "m5" / "pipeline_m5_4_plantilla.joblib")
        joblib.dump({"pipeline": final, "esquema": esquema.__dict__, "entrenado_con": "es_entrenamiento & en_periodo_estable"},
                    DATA / "m5" / "pipeline_m5_4_ajustado.joblib")
        (OUT / "esquema_m5_4.json").write_text(json.dumps({
            **esquema.__dict__,
            "n_variables_salida": int(len(nombres)),
            "estimador_prueba": repr(final.named_steps["modelo"]),
            "svd": {"n_components": 50, "min_df": 5, "ponderacion": "tfidf"},
            "cv": f"StratifiedGroupKFold({a.folds}, shuffle=True, random_state={SEMILLA}) por person_id",
            "verificacion_fuga_ok": bool(verif["ok"]),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  variables tras el preprocesador: {len(nombres)}")
    print("  CV (media ± DE): AUC-ROC {:.4f} ± {:.4f} · AUC-PR {:.4f} ± {:.4f}".format(
        cvdf.auc_roc.mean(), cvdf.auc_roc.std(), cvdf.auc_pr.mean(), cvdf.auc_pr.std()))
    print(prueba.to_string(index=False))
    print(f"  tiempo total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
