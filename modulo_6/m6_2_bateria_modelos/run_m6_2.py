"""Prueba de humo de la batería (6.2) sobre los datos reales.

No es la selección de modelos (eso es la 6.3, con búsqueda de hiperparámetros y los 5
pliegues). Aquí cada modelo, con sus valores por defecto, se entrena en una submuestra
de pacientes del pliegue 1 de la 6.1 y se evalúa en una submuestra de su validación,
para comprobar que toda la batería corre con el preprocesador real y medir tiempos.
Con esos tiempos se dimensiona la búsqueda de la 6.3.

Escribe:
    outputs/m6/tablas/m6_2_humo_modelos.csv   tiempos y AUC por modelo (global y media por servicio × τ)

Uso (desde la raíz del proyecto):
    python modulo_6/m6_2_bateria_modelos/run_m6_2.py [--n-train 40000] [--n-val 20000]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modulo_5.m5_4_pipeline import construir_preprocesador, preparar_X  # noqa: E402
from modulo_6.m6_2_bateria_modelos import COSTO, SALIDA_PUNTAJE, get_pipeline_and_space, listar_modelos  # noqa: E402

TARGET = "y_target_los_largo"


def submuestra_por_paciente(df: pd.DataFrame, n_filas: int, semilla: int) -> pd.DataFrame:
    """Toma pacientes completos (todas sus filas) hasta llegar a ~n_filas."""
    personas = df.person_id.drop_duplicates().sample(frac=1, random_state=semilla)
    acumulado = df.groupby("person_id").size().reindex(personas).cumsum()
    return df[df.person_id.isin(acumulado[acumulado <= n_filas].index)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=40_000)
    ap.add_argument("--n-val", type=int, default=20_000)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")

    print("· Cargando maestro (M5) y pliegues (6.1)")
    X, esquema = preparar_X(ROOT)
    tr = X[X.en_periodo_estable.astype(bool) & X.es_entrenamiento.astype(bool)].reset_index(drop=True)
    pl = pd.read_parquet(ROOT / "data" / "m6" / "pliegues_m6_1.parquet")
    if not pl.snapshot_id.equals(tr.snapshot_id):
        raise SystemExit("Los pliegues de la 6.1 no están alineados con el maestro: volver a correr run_m6_1.py")
    tr["pliegue"] = pl.pliegue.to_numpy()
    entrena = submuestra_por_paciente(tr[tr.pliegue != 1], a.n_train, 1)
    valida = submuestra_por_paciente(tr[tr.pliegue == 1], a.n_val, 2)
    assert not set(entrena.person_id) & set(valida.person_id)
    print(f"  submuestra: entrenamiento {len(entrena):,} filas ({entrena.person_id.nunique():,} pacientes) · "
          f"validación {len(valida):,} filas · prevalencia {valida[TARGET].mean():.3f}")

    pre = construir_preprocesador(esquema)
    cols = esquema.todas()
    filas = []
    for nombre in listar_modelos():
        pipe, _ = get_pipeline_and_space(nombre, pre)
        t0 = time.time()
        pipe.fit(entrena[cols], entrena[TARGET])
        t_fit = time.time() - t0
        t0 = time.time()
        s = pipe.decision_function(valida[cols]) if SALIDA_PUNTAJE[nombre] == "decision_function" \
            else pipe.predict_proba(valida[cols])[:, 1]
        t_pred = time.time() - t0
        estratos = valida.assign(s=s).groupby(["servicio", "horas_desde_ingreso"])
        auc_estratos = [roc_auc_score(d[TARGET], d.s) for _, d in estratos if d[TARGET].nunique() == 2]
        filas.append({"modelo": nombre, "costo_esperado": COSTO[nombre], "seg_fit": round(t_fit, 1),
                      "seg_predict": round(t_pred, 1), "auc_global": roc_auc_score(valida[TARGET], s),
                      "auc_media_servicio_tau": float(np.mean(auc_estratos))})
        print(f"  {nombre:<14} fit {t_fit:7.1f} s · predict {t_pred:6.1f} s · AUC global {filas[-1]['auc_global']:.3f}"
              f" · AUC medio por servicio×τ {filas[-1]['auc_media_servicio_tau']:.3f}")

    out = pd.DataFrame(filas)
    out["n_train"], out["n_val"] = len(entrena), len(valida)
    tab = ROOT / "outputs" / "m6" / "tablas"
    tab.mkdir(parents=True, exist_ok=True)
    out.to_csv(tab / "m6_2_humo_modelos.csv", index=False)
    print("\n✓ La batería completa corre con el preprocesador del M5. Tabla: outputs/m6/tablas/m6_2_humo_modelos.csv")


if __name__ == "__main__":
    main()
