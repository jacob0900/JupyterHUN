"""Ejecuta la subfase 6.3 sobre los datos reales: Optuna (TPE + MedianPruner) para toda la batería.

Diseño del cómputo (medido antes de correr, ver modulo_6/README.md):
- Con el pliegue completo (433 000 filas de entrenamiento) un ajuste pesado tarda varios minutos;
  50 ensayos × 5 pliegues × 10 modelos serían días. Por eso la **búsqueda** se hace sobre una
  submuestra de **pacientes completos** (`--frac`, 20 % por defecto ≈ 108 000 snapshots), con los
  mismos pliegues de la 6.1 (una submuestra de pacientes de pliegues aislados sigue aislada).
- KNN predice en O(n_train × n_val): usa `--frac-knn` (5 % por defecto).
- El preprocesador del M5 se ajusta una vez por pliegue (`CachePliegues`), con resultado idéntico al Pipeline.
- `--timeout-modelo` limita el tiempo de cada estudio; el estudio se guarda en SQLite y se reanuda
  si se vuelve a correr (solo hace los ensayos que falten).

Escribe:
    data/m6/optuna_m6_3.db                     estudios de Optuna (reanudables; solo parámetros y AUC)
    outputs/m6/best_hyperparameters.json       mejores hiperparámetros y AUC por modelo
    outputs/m6/tablas/m6_3_resumen_modelos.csv tabla comparativa (10 modelos)
    outputs/m6/tablas/m6_3_resumen_familias.csv tabla consolidada (7 familias del prompt)
    outputs/m6/tablas/m6_3_historial_ensayos.csv todos los ensayos (estado, AUC, parámetros)

Uso (desde la raíz del proyecto):
    python modulo_6/m6_3_optuna/run_m6_3.py [--n-trials 50] [--frac 0.2] [--frac-knn 0.05]
                                            [--timeout-modelo 1800] [--modelos lightgbm xgboost ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modulo_5.m5_4_pipeline import construir_preprocesador, preparar_X  # noqa: E402
from modulo_6.m6_1_cv_agrupada import PlieguesPredefinidos, verify_group_isolation  # noqa: E402
from modulo_6.m6_3_optuna import (  # noqa: E402
    CachePliegues,
    guardar_resultados,
    optimizar_modelo,
    resumen_estudio,
    tabla_por_familia,
)

TARGET = "y_target_los_largo"
# De más rápido a más lento (prueba de humo de la 6.2): así los resultados parciales llegan antes.
ORDEN = ["naive_bayes", "logreg_ridge", "cart", "logreg_lasso", "svm_lineal", "random_forest", "lightgbm",
         "xgboost", "svm_rbf", "knn"]


def submuestra(tr: pd.DataFrame, frac: float, semilla: int = 42) -> pd.DataFrame:
    if frac >= 1:
        return tr.reset_index(drop=True)
    personas = tr.person_id.drop_duplicates().sample(frac=frac, random_state=semilla)
    return tr[tr.person_id.isin(personas)].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--frac-knn", type=float, default=0.05)
    ap.add_argument("--timeout-modelo", type=float, default=1800)
    ap.add_argument("--modelos", nargs="*", default=ORDEN)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("· Cargando maestro (M5) y pliegues (6.1)", flush=True)
    X, esquema = preparar_X(ROOT)
    tr = X[X.en_periodo_estable.astype(bool) & X.es_entrenamiento.astype(bool)].reset_index(drop=True)
    pl = pd.read_parquet(ROOT / "data" / "m6" / "pliegues_m6_1.parquet")
    if not pl.snapshot_id.equals(tr.snapshot_id):
        raise SystemExit("Los pliegues de la 6.1 no están alineados con el maestro: volver a correr run_m6_1.py")
    tr["pliegue"] = pl.pliegue.to_numpy()
    cols = esquema.todas()
    pre = construir_preprocesador(esquema)

    caches = {}

    def datos(frac):
        if frac not in caches:
            sub = submuestra(tr, frac)
            Xs, ys, gs = sub[cols], sub[TARGET].to_numpy(), sub.person_id.to_numpy()
            cv = PlieguesPredefinidos(sub.pliegue)
            verify_group_isolation(Xs, ys, gs, cv, n_splits=5)
            estr = (sub.servicio + "_t" + sub.horas_desde_ingreso.astype(str)).to_numpy()
            t0 = time.time()
            cache = CachePliegues(pre, Xs, ys, cv, groups=gs, estratos=estr)
            print(f"  submuestra {frac:.0%}: {len(sub):,} snapshots · {sub.person_id.nunique():,} pacientes · "
                  f"prevalencia {ys.mean():.3f} · preprocesador por pliegue {time.time() - t0:.0f} s", flush=True)
            caches[frac] = (Xs, ys, gs, cv, cache, len(sub), int(sub.person_id.nunique()))
        return caches[frac]

    (ROOT / "data" / "m6").mkdir(parents=True, exist_ok=True)
    tab = ROOT / "outputs" / "m6" / "tablas"
    tab.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(ROOT / 'data' / 'm6' / 'optuna_m6_3.db').as_posix()}"

    filas, historial = [], []
    for nombre in [m for m in ORDEN if m in a.modelos]:
        frac = a.frac_knn if nombre == "knn" else a.frac
        Xs, ys, gs, cv, cache, n_filas, n_pac = datos(frac)
        print(f"· {nombre}: hasta {a.n_trials} ensayos · timeout {a.timeout_modelo / 60:.0f} min", flush=True)
        t0 = time.time()
        estudio = optimizar_modelo(nombre, Xs, ys, gs, cv, n_trials=a.n_trials, timeout=a.timeout_modelo,
                                   cache=cache, preprocesador=pre, storage=storage, verificar_aislamiento=False)
        seg = time.time() - t0
        fila = resumen_estudio(nombre, estudio, seg)
        fila.update({"frac_pacientes": frac, "n_snapshots_busqueda": n_filas, "n_pacientes_busqueda": n_pac})
        filas.append(fila)
        for t in estudio.trials:
            historial.append({"modelo": nombre, "ensayo": t.number, "estado": t.state.name, "auc_cv": t.value,
                              "auc_std": t.user_attrs.get("auc_std"),
                              "auc_medio_servicio_tau": t.user_attrs.get("auc_medio_servicio_tau"),
                              "segundos": (t.duration.total_seconds() if t.duration else None),
                              "params": json.dumps(t.params, default=str)})
        if fila.get("ensayos_completos"):
            print(f"  → mejor AUC {fila['mejor_auc_cv']:.4f} (± {fila['auc_std_pliegues']:.4f}) · "
                  f"servicio×τ {fila['auc_medio_servicio_tau']:.4f} · completos {fila['ensayos_completos']} · "
                  f"podados {fila['ensayos_podados']} · {seg / 60:.1f} min", flush=True)
        else:
            print(f"  → sin ensayos completos en {seg / 60:.1f} min", flush=True)

        # Guardado incremental: si algo se interrumpe, quedan los modelos ya terminados.
        tabla = guardar_resultados(filas, ROOT / "outputs" / "m6" / "best_hyperparameters.json",
                                   tab / "m6_3_resumen_modelos.csv",
                                   meta={"n_trials_objetivo": a.n_trials, "frac_pacientes": a.frac,
                                         "frac_pacientes_knn": a.frac_knn, "timeout_modelo_s": a.timeout_modelo,
                                         "sampler": "TPESampler(seed=42, n_startup_trials=10)",
                                         "pruner": "MedianPruner(n_startup_trials=5, n_warmup_steps=1)",
                                         "cv": "pliegues de data/m6/pliegues_m6_1.parquet (StratifiedGroupKFold(5) por person_id)"})
        pd.DataFrame(historial).to_csv(tab / "m6_3_historial_ensayos.csv", index=False)

    fam = tabla_por_familia(tabla)
    fam.to_csv(tab / "m6_3_resumen_familias.csv", index=False)
    print("\nTabla comparativa (10 modelos):")
    print(tabla[["modelo", "mejor_auc_cv", "auc_std_pliegues", "auc_medio_servicio_tau", "ensayos_completos",
                 "ensayos_podados", "segundos"]].to_string(index=False))
    print("\nConsolidado por familia (7 del prompt):")
    print(fam.to_string(index=False))


if __name__ == "__main__":
    main()
