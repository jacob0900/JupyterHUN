"""Ejecuta la subfase 5.3 sobre los datos reales.

Lee:
    data/m1/ocupacion_servicio_hora.parquet  serie servicio × hora (tarea de servicio)
    data/m1/snapshot.parquet                 snapshots de paciente (solo llaves, servicio y snapshot_time)
    data/m4/calendario_dia.parquet           festivos colombianos (Persona 1)

Escribe:
    data/m5/servicio_hora_m5_3.parquet   serie de servicio + variables 5.3 (agregado por hora, sin pacientes)
    data/m5/snapshot_m5_3.parquet        snapshot_id + variables temporales + srv_* (nivel paciente: no se comparte)
    outputs/m5/esquema_m5_3.json         listas de columnas nuevas para M6
    outputs/m5/tablas/m5_3_resumen.csv           media / % NaN por variable y servicio (entrenamiento estable)
    outputs/m5/tablas/m5_3_auc_saturacion.csv    AUC univariada de cada variable vs y_target_saturacion (entrenamiento)
    outputs/m5/tablas/m5_3_verificacion_fuga.csv resultado del test de no-fuga en varios cortes reales

Uso (desde la raíz del proyecto):
    python modulo_5/m5_3_temporal_ocupacion/run_m5_3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modulo_5.m5_3_temporal_ocupacion import (  # noqa: E402
    ConfigTemporal,
    build_temporal_and_occupancy_features,
    unir_ocupacion_a_snapshots,
    verificar_sin_fuga,
)

DATA = ROOT / "data"
OUT = ROOT / "outputs" / "m5"


def auc(x: np.ndarray, y: np.ndarray) -> float:
    m = ~np.isnan(x)
    x, y = x[m], y[m]
    r = pd.Series(x).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    cfg = ConfigTemporal()
    cal = pd.read_parquet(DATA / "m4" / "calendario_dia.parquet")

    print("· Serie de servicio (ocupacion_servicio_hora)")
    serie = pd.read_parquet(DATA / "m1" / "ocupacion_servicio_hora.parquet")
    feats = build_temporal_and_occupancy_features(serie, "hora", cal, cfg)
    nuevas = [c for c in feats.columns if c not in serie.columns]
    temporales = [c for c in nuevas if c in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "turno")
                  or c.startswith(("es_", "tras_", "primer_"))]
    ocupacion = [c for c in nuevas if c not in temporales]

    print("· Verificación anti-fuga sobre datos reales (varios cortes)")
    cortes = [pd.Timestamp(x) for x in ("2023-08-15 07:00", "2024-06-03 14:00", "2025-07-13 19:00", "2026-01-02 03:00")]
    filas_v = []
    for c in cortes:
        r = verificar_sin_fuga(serie, c, "hora", cal, cfg)
        filas_v.append({"corte": c, "ok": r["ok"], "filas_comparadas": r["filas_comparadas"],
                        "columnas_con_diferencias": ";".join(r["columnas_con_diferencias"])})
        print(f"  corte {c}: {'OK' if r['ok'] else 'FUGA'} ({r['filas_comparadas']:,} filas)")
    verif = pd.DataFrame(filas_v)
    if not verif.ok.all():
        raise SystemExit("✗ Se detectó fuga de información: revisar antes de usar las variables")
    # Comprobación directa: ocupacion_t1(t) == ocupacion(t − 1) en todas las horas.
    g = feats.sort_values(["servicio", "hora"])
    previa = g.groupby("servicio").ocupacion.shift(1)
    assert np.allclose(g.ocupacion_t1, previa, equal_nan=True, atol=1e-12), "ocupacion_t1 ≠ ocupación de la hora anterior"

    print("· Snapshots de paciente (temporales + srv_* de su servicio)")
    snap = pd.read_parquet(DATA / "m1" / "snapshot.parquet",
                           columns=["snapshot_id", "servicio", "snapshot_time", "es_entrenamiento", "en_periodo_estable"])
    snap_t = build_temporal_and_occupancy_features(snap[["snapshot_id", "servicio", "snapshot_time"]], "snapshot_time", cal, cfg)
    srv = unir_ocupacion_a_snapshots(snap, feats)
    snap_out = pd.concat([snap_t.drop(columns=["servicio", "snapshot_time"]), srv], axis=1)

    (DATA / "m5").mkdir(parents=True, exist_ok=True)
    tab = OUT / "tablas"
    tab.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(DATA / "m5" / "servicio_hora_m5_3.parquet", index=False)
    snap_out.to_parquet(DATA / "m5" / "snapshot_m5_3.parquet", index=False)
    verif.to_csv(tab / "m5_3_verificacion_fuga.csv", index=False)

    # Resumen agregado (entrenamiento estable)
    tr = feats[feats.es_entrenamiento & feats.en_periodo_estable]
    num = [c for c in nuevas if c != "turno"]
    resumen = (tr.groupby("servicio")[num].agg(["mean", lambda s: 100 * s.isna().mean()])
               .T.rename(index={"<lambda_0>": "pct_nan"}).round(4))
    resumen.index.names = ["variable", "estadistico"]
    resumen.reset_index().to_csv(tab / "m5_3_resumen.csv", index=False)

    filas = []
    for s, d in tr.groupby("servicio"):
        y = d.y_target_saturacion.to_numpy()
        for c in num:
            filas.append({"servicio": s, "variable": c, "auc": auc(d[c].to_numpy(float), y)})
        filas.append({"servicio": s, "variable": "REFERENCIA ocupacion(t) (prohibida en 5.3)", "auc": auc(d.ocupacion.to_numpy(float), y)})
    aucs = pd.DataFrame(filas)
    aucs["fuerza"] = (aucs.auc - 0.5).abs()
    aucs = aucs.sort_values(["servicio", "fuerza"], ascending=[True, False]).drop(columns="fuerza")
    aucs.to_csv(tab / "m5_3_auc_saturacion.csv", index=False)

    nan_snap = srv.isna().mean().mul(100).round(3)
    (OUT / "esquema_m5_3.json").write_text(json.dumps({
        "temporales": temporales,
        "ocupacion_t1_servicio": ocupacion,
        "ocupacion_t1_snapshot": list(srv.columns),
        "categoricas": ["turno"],
        "config": {"horas_pico": cfg.horas_pico, "ventanas_h": cfg.ventanas_h,
                   "ventanas_llegadas_h": cfg.ventanas_llegadas_h, "lag_estacional_h": cfg.lag_estacional_h,
                   "umbral_por_servicio": cfg.umbral_por_servicio},
        "verificacion_fuga_ok": bool(verif.ok.all()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  serie: {len(feats):,} filas · {len(temporales)} temporales + {len(ocupacion)} de ocupación t−1")
    print(f"  snapshots: {len(snap_out):,} · % NaN en srv_ocupacion_t1: {nan_snap['srv_ocupacion_t1']:.3f} %")
    print("  AUC vs y_target_saturacion (entrenamiento estable), top 6 por servicio:")
    print(aucs.groupby("servicio").head(6).to_string(index=False))


if __name__ == "__main__":
    main()
