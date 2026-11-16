"""Ejecuta la subfase 6.1 sobre los datos reales.

1. Carga el maestro procesado en el Módulo 5 (`preparar_X` de la 5.4: M1 + 5.1 + 5.2 + 5.3).
2. Se queda con entrenamiento del periodo estable (la prueba temporal no entra a la CV).
3. Configura `StratifiedGroupKFold(n_splits=5)` con grupos = person_id y verifica el
   aislamiento de pacientes en cada pliegue (`verify_group_isolation`).
4. Imprime y guarda el reporte por pliegue, y guarda la asignación de pliegues para que
   todo M6 use exactamente los mismos.

Escribe:
    data/m6/pliegues_m6_1.parquet                snapshot_id, person_id, pliegue (nivel paciente: no se comparte)
    outputs/m6/tablas/m6_1_reporte_pliegues.csv  muestras, pacientes y % positivos por pliegue
    outputs/m6/tablas/m6_1_prevalencia_servicio_tau.csv  % positivos por pliegue × servicio × τ

Uso (desde la raíz del proyecto):
    python modulo_6/m6_1_cv_agrupada/run_m6_1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modulo_5.m5_4_pipeline import preparar_X  # noqa: E402
from modulo_6.m6_1_cv_agrupada import (  # noqa: E402
    N_SPLITS,
    PlieguesPredefinidos,
    asignar_pliegues,
    crear_cv,
    imprimir_reporte,
    verify_group_isolation,
)

TARGET = "y_target_los_largo"
GRUPO = "person_id"


def main() -> None:
    print("· Cargando el maestro del Módulo 5")
    X, esquema = preparar_X(ROOT)
    estable = X.en_periodo_estable.astype(bool)
    tr = X[estable & X.es_entrenamiento.astype(bool)].reset_index(drop=True)
    print(f"  maestro {len(X):,} snapshots · entrenamiento estable {len(tr):,} · "
          f"prueba apartada {int((estable & ~X.es_entrenamiento.astype(bool)).sum()):,}")

    X_tr, y_tr, g_tr = tr[esquema.todas()], tr[TARGET], tr[GRUPO]
    cv = crear_cv()
    print(f"· {cv!r} · grupos = {GRUPO} ({g_tr.nunique():,} pacientes)")
    rep = verify_group_isolation(X_tr, y_tr, g_tr, cv, n_splits=N_SPLITS)
    imprimir_reporte(rep)

    pliegue = asignar_pliegues(g_tr, y_tr, cv)
    # Doble control: los pliegues guardados reproducen la CV y también están aislados.
    verify_group_isolation(X_tr, y_tr, g_tr, PlieguesPredefinidos(pliegue), n_splits=N_SPLITS)

    por_estrato = (
        tr.assign(pliegue=pliegue)
        .groupby(["pliegue", "servicio", "horas_desde_ingreso"])[TARGET]
        .agg(n="size", prop_pos="mean").reset_index()
    )
    rango = por_estrato.groupby(["servicio", "horas_desde_ingreso"]).prop_pos.agg(["min", "max"])
    print("\nProporción positiva por servicio × τ entre pliegues (mín – máx):")
    for (s, h), r in rango.iterrows():
        print(f"  {s:<22} τ={h} h: {100 * r['min']:.2f} % – {100 * r['max']:.2f} %")

    (ROOT / "data" / "m6").mkdir(parents=True, exist_ok=True)
    tab = ROOT / "outputs" / "m6" / "tablas"
    tab.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"snapshot_id": tr.snapshot_id, GRUPO: g_tr, "pliegue": pliegue}).to_parquet(
        ROOT / "data" / "m6" / "pliegues_m6_1.parquet", index=False)
    rep.to_csv(tab / "m6_1_reporte_pliegues.csv", index=False)
    por_estrato.to_csv(tab / "m6_1_prevalencia_servicio_tau.csv", index=False)
    print("\n✓ Aislamiento verificado. Pliegues guardados en data/m6/pliegues_m6_1.parquet")


if __name__ == "__main__":
    main()
