"""Ejecuta la subfase 5.1 sobre los datos reales.

Lee:
    data/m1/snapshot.parquet      dataset maestro (snapshots en τ, de M1)
    data/raw/measurement.parquet  tabla OMOP measurement (opcional: si no está, solo
                                  se crean las banderas estructurales)
    data/raw/concept.parquet      nombres de los conceptos (opcional)

Escribe:
    data/m5/snapshot_m5_1.parquet               maestro + banderas (nivel paciente: no se comparte)
    outputs/m5/tablas/labs_top_k.csv            laboratorios elegidos (agregado)
    outputs/m5/tablas/esquema_columnas.csv      clasificación num / cat / lab / excluida
    outputs/m5/tablas/prevalencia_banderas.csv  % de snapshots con cada bandera, por servicio × τ × partición
    outputs/m5/esquema_m5_1.json                listas num_cols / cat_cols / lab_cols para las subfases siguientes

Uso:
    python modulo_5/m5_1_banderas_mnar/run_m5_1.py [--top-k 20] [--ventana-horas 24|none]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modulo_5.m5_1_banderas_mnar import ConfigMNAR, construir_subfase_5_1  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "outputs" / "m5"
COLS_MED = ["person_id", "measurement_concept_id", "measurement_source_value", "measurement_source_value_extended",
            "measurement_datetime", "measurement_date"]


def _leer_columnas(ruta: Path, deseadas: list[str]) -> pd.DataFrame:
    """Lee solo las columnas que existen (measurement puede ser muy grande)."""
    import pyarrow.parquet as pq  # motor de parquet de pandas

    disponibles = set(pq.ParquetFile(ruta).schema_arrow.names)
    return pd.read_parquet(ruta, columns=[c for c in deseadas if c in disponibles])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--ventana-horas", default="24", help="horas hacia atrás desde τ, o 'none' para toda la historia")
    ap.add_argument("--col-concepto", default="measurement_concept_id",
                    help="columna que identifica el examen (p. ej. measurement_source_value_extended)")
    a = ap.parse_args()

    ventana = None if str(a.ventana_horas).lower() == "none" else float(a.ventana_horas)
    cfg = ConfigMNAR(top_k=a.top_k, ventana_horas=ventana)

    print("· Leyendo snapshot de M1")
    master = pd.read_parquet(DATA / "m1" / "snapshot.parquet")

    med, nombres = None, None
    ruta_med = DATA / "raw" / "measurement.parquet"
    if ruta_med.exists():
        print("· Leyendo measurement")
        # Solo las columnas necesarias: los textos de source_value pesan mucho (16 M de filas).
        med = _leer_columnas(ruta_med, ["person_id", a.col_concepto, "measurement_datetime", "measurement_date"])
        ruta_concept = DATA / "raw" / "concept.parquet"
        if a.col_concepto == "measurement_concept_id" and ruta_concept.exists():
            conc = _leer_columnas(ruta_concept, ["concept_id", "concept_name"])
            nombres = dict(zip(conc.concept_id, conc.concept_name))
    else:
        print(f"  ⚠ No existe {ruta_med.relative_to(ROOT)}: se omiten las banderas has_measurement_X.\n"
              "    Pedir a Persona 1 la extracción de measurement (person_id, measurement_concept_id,\n"
              "    measurement_datetime, measurement_source_value_extended) y volver a correr.")

    print(f"· Construyendo banderas (top_k={cfg.top_k}, ventana={cfg.ventana_horas} h, regla t < τ, cruce por person_id)")
    out, esquema, top = construir_subfase_5_1(master, med, cfg, col_concepto=a.col_concepto, nombres=nombres)

    (DATA / "m5").mkdir(parents=True, exist_ok=True)
    tab = OUT / "tablas"
    tab.mkdir(parents=True, exist_ok=True)

    out.to_parquet(DATA / "m5" / "snapshot_m5_1.parquet", index=False)
    top.to_csv(tab / "labs_top_k.csv", index=False)
    esquema.resumen().to_csv(tab / "esquema_columnas.csv", index=False)

    nuevas = [c for c in out.columns if c not in master.columns]
    estable = out[out[cfg.col_estable].astype(bool)] if cfg.col_estable in out.columns else out
    particion = np.where(estable[cfg.col_entrenamiento].astype(bool), "entrenamiento", "prueba")
    claves = [c for c in ("servicio", "horas_desde_ingreso") if c in out.columns]
    prev = (
        estable.assign(particion=particion)
        .groupby(claves + ["particion"])[nuevas].mean().mul(100).round(2)
        .reset_index()
        .melt(id_vars=claves + ["particion"], var_name="bandera", value_name="pct_snapshots")
    )
    prev.to_csv(tab / "prevalencia_banderas.csv", index=False)

    (OUT / "esquema_m5_1.json").write_text(
        json.dumps(
            {"num_cols": esquema.num_cols, "cat_cols": esquema.cat_cols, "lab_cols": esquema.lab_cols,
             "excluidas": esquema.excluidas, "banderas_nuevas": nuevas,
             "config": {"top_k": cfg.top_k, "ventana_horas": cfg.ventana_horas, "col_concepto": a.col_concepto,
                        "measurement_disponible": med is not None}},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    print(f"  snapshots: {len(out):,} · banderas nuevas: {len(nuevas)} ({', '.join(nuevas)})")
    print(f"  num_cols={len(esquema.num_cols)} · cat_cols={len(esquema.cat_cols)} · lab_cols={len(esquema.lab_cols)}")
    if not top.empty:
        print(top[["rango", "concepto", "nombre", "pct_personas", "bandera"]].to_string(index=False))


if __name__ == "__main__":
    main()
