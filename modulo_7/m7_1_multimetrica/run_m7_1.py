"""Subfase 7.1 sobre los datos reales: evaluación multimétrica de las predicciones OOF de la 6.4.

Entrada: `data/m6/oof_m6_4.parquet` (541 260 snapshots de entrenamiento estable; KNN solo en el 20 % de
los pacientes). La prueba temporal no se usa aquí.

Salidas (solo agregados):
    outputs/m7/multimetric_evaluation_summary.csv          tabla consolidada, ordenada por AUC-PR
    outputs/m7/tablas/m7_1_subconjunto_comun.csv            los 7 modelos sobre las mismas filas (las de KNN)
    outputs/m7/tablas/m7_1_por_servicio_tau.csv             métricas por servicio × τ
    outputs/m7/tablas/m7_1_bootstrap_replicas.csv           réplicas pareadas (AUC-ROC, AUC-PR) por modelo
    outputs/m7/figuras/m7_1_roc_pr.png                      curvas ROC y Precision-Recall
    data/m7/probabilidades_oof_m7_1.parquet                 OOF en escala de probabilidad (SVM con Platt cruzado)

Uso (desde la raíz del proyecto):
    python modulo_7/m7_1_multimetrica/run_m7_1.py [--n-bootstrap 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modulo_7.m7_1_multimetrica import (  # noqa: E402
    N_BOOTSTRAP,
    evaluar_modelos,
    graficar_roc_pr,
    metricas_por_estrato,
    platt_cruzado,
)

TARGET = "y_target_los_largo"


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    a = ap.parse_args()

    out, tab, fig = ROOT / "outputs" / "m7", ROOT / "outputs" / "m7" / "tablas", ROOT / "outputs" / "m7" / "figuras"
    for d in (tab, fig, ROOT / "data" / "m7"):
        d.mkdir(parents=True, exist_ok=True)

    oof = pd.read_parquet(ROOT / "data" / "m6" / "oof_m6_4.parquet")
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    y, g = oof[TARGET].to_numpy(), oof.person_id.to_numpy()
    log(f"OOF: {len(oof):,} snapshots · {oof.person_id.nunique():,} pacientes · prevalencia {y.mean():.4f}")

    probas, notas = {}, {}
    for c in [c for c in oof.columns if c.startswith("oof_")]:
        m = c.removeprefix("oof_")
        s = oof[c].to_numpy(float)
        salida = manifest["modelos"].get(m, {}).get("salida", "predict_proba")
        if salida != "predict_proba" or s.min() < 0 or s.max() > 1:
            ok = ~np.isnan(s)
            p = np.full(len(s), np.nan)
            p[ok] = platt_cruzado(s[ok], y[ok], oof.pliegue.to_numpy()[ok])
            log(f"{m}: decision_function → Platt cruzado por pliegue (AUC bruto {roc_auc_score(y[ok], s[ok]):.4f} · "
                f"tras Platt {roc_auc_score(y[ok], p[ok]):.4f})")
            probas[m], notas[m] = p, "decision_function + Platt cruzado por pliegue"
        else:
            probas[m], notas[m] = s, "predict_proba"

    t0 = time.time()
    tabla, rep = evaluar_modelos(y, probas, g, n_iter=a.n_bootstrap, semilla=42)
    tabla["salida"] = tabla.modelo.map(notas)
    tabla["filas_evaluadas"] = np.where(tabla.n < len(oof), "20 % de los pacientes (KNN)", "todas")
    tabla.to_csv(out / "multimetric_evaluation_summary.csv", index=False)
    rep.to_csv(tab / "m7_1_bootstrap_replicas.csv", index=False)
    log(f"Tabla consolidada + bootstrap de {a.n_bootstrap} réplicas por paciente ({(time.time() - t0) / 60:.1f} min)")

    comun = ~np.isnan(probas["knn"]) if "knn" in probas else np.ones(len(oof), bool)
    sub, _ = evaluar_modelos(y[comun], {m: p[comun] for m, p in probas.items()}, g[comun],
                             n_iter=a.n_bootstrap, semilla=42)
    sub.to_csv(tab / "m7_1_subconjunto_comun.csv", index=False)
    log(f"Subconjunto común: {comun.sum():,} filas")

    estratos = (oof.servicio + " · τ" + oof.horas_desde_ingreso.astype(str) + "h").to_numpy()
    por = metricas_por_estrato(y, probas, estratos)
    por.to_csv(tab / "m7_1_por_servicio_tau.csv", index=False)

    graficar_roc_pr(y, probas, fig / "m7_1_roc_pr.png", {"knn": " (20 % pac.)"})
    pd.DataFrame({"snapshot_id": oof.snapshot_id, "person_id": g, "pliegue": oof.pliegue, TARGET: y,
                  "servicio": oof.servicio, "horas_desde_ingreso": oof.horas_desde_ingreso,
                  **{f"p_{m}": p for m, p in probas.items()}}).to_parquet(
        ROOT / "data" / "m7" / "probabilidades_oof_m7_1.parquet", index=False)

    r = lambda x: x.round(4).to_string(index=False)  # noqa: E731
    print("\nmultimetric_evaluation_summary.csv:")
    print(r(tabla[["modelo", "n", "auc_roc", "auc_roc_ic95_inf", "auc_roc_ic95_sup", "auc_pr", "auc_pr_ic95_inf",
                   "auc_pr_ic95_sup", "brier", "ece", "f1", "sensibilidad", "especificidad"]]))
    print("\nSubconjunto común (filas de KNN):")
    print(r(sub[["modelo", "n", "auc_roc", "auc_pr", "auc_pr_ic95_inf", "auc_pr_ic95_sup", "brier", "ece"]]))
    print("\nAUC-PR por servicio × τ (prevalencia = valor sin información):")
    piv = por.pivot(index="modelo", columns="estrato", values="auc_pr")
    piv.loc["(prevalencia)"] = por.groupby("estrato").prevalencia.first()
    print(piv.round(3).to_string())
    log("Listo")


if __name__ == "__main__":
    main()
