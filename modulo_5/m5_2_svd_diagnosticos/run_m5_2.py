"""Ejecuta la subfase 5.2 (SVD sobre diagnósticos) con los datos reales.

Lee:
    data/m1/snapshot.parquet                 maestro (una fila por snapshot en τ)
    data/raw/condition_occurrence.parquet    diagnósticos OMOP (obligatorio)
    data/raw/procedure_occurrence.parquet    procedimientos (opcional; hoy sin fecha -> se descartan)
    data/raw/concept.parquet                 nombres de conceptos (opcional, para las cargas)

Escribe:
    data/m5/matriz_dx_m5_2.npz        matriz dispersa snapshots × códigos (para ajustar la SVD por fold en M6)
    data/m5/vocabulario_dx_m5_2.csv   columnas de esa matriz
    data/m5/svd_dx_m5_2.parquet       snapshot_id + svd_dx_00..49 (SVD ajustada con TODO el entrenamiento)
    data/m5/reductor_svd_m5_2.joblib  reductor ajustado (para el test final y la API de M8)
    outputs/m5/tablas/svd_varianza.csv         varianza explicada y acumulada
    outputs/m5/tablas/svd_cargas_top10.csv     códigos con más peso por componente (agregado, sin pacientes)

En la validación cruzada de M6 la SVD debe ir **dentro** del Pipeline (se reajusta en
cada fold); `svd_dx_m5_2.parquet` solo vale para el ajuste final entrenamiento → prueba.

Uso:
    python modulo_5/m5_2_svd_diagnosticos/run_m5_2.py [--n-components 50] [--min-df 5]
    python modulo_5/m5_2_svd_diagnosticos/run_m5_2.py --demo     # datos sintéticos
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modulo_5.m5_2_svd_diagnosticos import (  # noqa: E402
    ReductorSVDClinico,
    construir_matriz_dispersa,
    preparar_eventos,
)

DATA = ROOT / "data"
RAW = DATA / "raw"
OUT = ROOT / "outputs" / "m5"

FUENTES = {
    # archivo: (columna de concepto, datetime, fecha de respaldo, prefijo)
    "condition_occurrence": ("condition_concept_id", "condition_start_datetime", "condition_start_date", "dx_"),
    "procedure_occurrence": ("procedure_concept_id", "procedure_datetime", "procedure_date", "px_"),
}


def _leer_columnas(ruta: Path, deseadas: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    disponibles = set(pq.ParquetFile(ruta).schema_arrow.names)
    return pd.read_parquet(ruta, columns=[c for c in deseadas if c in disponibles])


def _primera_aparicion_duckdb(ruta: Path, c_con: str, c_dt: str | None, c_fecha: str | None) -> tuple[int, pd.DataFrame]:
    """Reduce la tabla a la primera aparición por (persona, código) sin cargarla entera en pandas.

    Aplica en SQL las mismas reglas que `preparar_eventos`: hora 00:00:00 o solo fecha ->
    final del día; concepto 0 y filas sin fecha fuera. `condition_occurrence` tiene ~38 M
    de filas y este PC no tiene RAM para leerlas en pandas.
    """
    import duckdb
    import pyarrow.parquet as pq

    cols = set(pq.ParquetFile(ruta).schema_arrow.names)
    fin_dia = "CAST({}::DATE AS TIMESTAMP) + INTERVAL 1 DAY - INTERVAL 1 MICROSECOND"
    partes = []
    if c_dt in cols:
        partes.append(f"CASE WHEN {c_dt} = date_trunc('day', {c_dt}) THEN {fin_dia.format(c_dt)} ELSE {c_dt} END")
    if c_fecha in cols:
        partes.append(fin_dia.format(c_fecha))
    p = str(ruta).replace("'", "''")
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{p}')").fetchone()[0]
    if not partes:
        return n, pd.DataFrame(columns=["person_id", c_con, "t_efectiva"])
    t = partes[0] if len(partes) == 1 else f"coalesce({', '.join(partes)})"
    df = con.execute(
        f"""SELECT person_id, {c_con}, min({t}) AS t_efectiva
            FROM read_parquet('{p}')
            WHERE person_id IS NOT NULL AND {c_con} IS NOT NULL AND {c_con} <> 0
            GROUP BY 1, 2 HAVING min({t}) IS NOT NULL"""
    ).df()
    con.close()
    return n, df


def _demo(n_components: int, min_df: int) -> None:
    """Cohorte sintética: 6 000 personas, 25 perfiles de comorbilidad, ~800 códigos."""
    rng = np.random.default_rng(42)
    n = 6000
    perfil = rng.integers(0, 25, n)
    filas = []
    for p in range(n):
        cods = np.concatenate([100 * (perfil[p] + 1) + rng.choice(20, rng.integers(2, 8), replace=False),
                               rng.integers(5000, 5300, rng.integers(0, 3))])
        filas += [(p, int(c), pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 300)))) for c in cods]
    dx = pd.DataFrame(filas, columns=["person_id", "condition_concept_id", "condition_start_datetime"])
    snap = pd.DataFrame({"person_id": np.arange(n), "snapshot_time": pd.Timestamp("2025-01-01"), "es_entrenamiento": np.arange(n) < 4800})
    X, vocab = construir_matriz_dispersa(snap, preparar_eventos(dx, "condition_concept_id", "condition_start_datetime"))
    tr = snap.es_entrenamiento.to_numpy()
    print(f"· DEMO sintética: matriz {X.shape[0]:,} × {X.shape[1]:,} · {X.nnz:,} no-ceros "
          f"({100 * X.nnz / (X.shape[0] * X.shape[1]):.2f} % densidad)")
    red = ReductorSVDClinico(n_components=n_components, min_df=min_df).fit(X[tr])
    red.imprimir_varianza()
    print(f"  transform(prueba) -> {red.transform(X[~tr]).shape}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-components", type=int, default=50)
    ap.add_argument("--min-df", type=int, default=5, help="mínimo de snapshots de entrenamiento con el código")
    ap.add_argument("--sin-tfidf", action="store_true", help="usar la matriz binaria sin ponderar")
    ap.add_argument("--demo", action="store_true", help="correr con datos sintéticos")
    a = ap.parse_args()

    if a.demo:
        _demo(a.n_components, a.min_df)
        return

    if not (RAW / "condition_occurrence.parquet").exists():
        print("⚠ No existe data/raw/condition_occurrence.parquet.\n"
              "  Pedir a Persona 1 la extracción (person_id, condition_concept_id, condition_start_datetime)\n"
              "  y volver a correr. Mientras tanto: python modulo_5/m5_2_svd_diagnosticos/run_m5_2.py --demo")
        sys.exit(1)

    print("· Leyendo snapshot de M1")
    master = pd.read_parquet(DATA / "m1" / "snapshot.parquet",
                             columns=["snapshot_id", "person_id", "snapshot_time", "es_entrenamiento", "en_periodo_estable"])

    eventos = []
    for archivo, (c_con, c_dt, c_fecha, pref) in FUENTES.items():
        ruta = RAW / f"{archivo}.parquet"
        if not ruta.exists():
            continue
        n_filas, tabla = _primera_aparicion_duckdb(ruta, c_con, c_dt, c_fecha)
        ev = preparar_eventos(tabla, c_con, "t_efectiva", None, prefijo=pref, personas=master.person_id,
                              medianoche_como_fecha=False)  # la regla ya se aplicó en DuckDB
        print(f"· {archivo}: {n_filas:,} filas -> {len(ev):,} pares persona-código con fecha")
        if ev.empty:
            print(f"  ⚠ {archivo} no tiene fechas utilizables: queda fuera (no se puede ubicar antes de τ)")
        eventos.append(ev)
        del tabla
    ev = pd.concat(eventos, ignore_index=True)

    print("· Construyendo matriz dispersa (regla t < τ, cruce por person_id)")
    X, vocab = construir_matriz_dispersa(master, ev)
    print(f"  {X.shape[0]:,} snapshots × {X.shape[1]:,} códigos · {X.nnz:,} no-ceros "
          f"({100 * X.nnz / (X.shape[0] * X.shape[1]):.3f} % densidad)")

    tr = (master.es_entrenamiento & master.en_periodo_estable).to_numpy()
    print(f"· Ajustando TruncatedSVD solo con entrenamiento estable ({tr.sum():,} snapshots)")
    red = ReductorSVDClinico(n_components=a.n_components, min_df=a.min_df,
                             ponderacion=None if a.sin_tfidf else "tfidf").fit(X[tr])
    red.imprimir_varianza()
    Z = red.transform(X)

    (DATA / "m5").mkdir(parents=True, exist_ok=True)
    tab = OUT / "tablas"
    tab.mkdir(parents=True, exist_ok=True)
    sp.save_npz(DATA / "m5" / "matriz_dx_m5_2.npz", X)
    pd.Series(vocab, name="codigo").to_csv(DATA / "m5" / "vocabulario_dx_m5_2.csv", index=False)
    out = pd.DataFrame(Z.astype(np.float32), columns=red.get_feature_names_out())
    out.insert(0, "snapshot_id", master.snapshot_id.to_numpy())
    out.to_parquet(DATA / "m5" / "svd_dx_m5_2.parquet", index=False)
    joblib.dump({"reductor": red, "vocabulario": vocab}, DATA / "m5" / "reductor_svd_m5_2.joblib")

    red.tabla_varianza().to_csv(tab / "svd_varianza.csv", index=False)
    nombres = None
    if (RAW / "concept.parquet").exists():
        conc = pd.read_parquet(RAW / "concept.parquet", columns=["concept_id", "concept_name"])
        nombres = {f"{p}{i}": n for i, n in zip(conc.concept_id, conc.concept_name) for p in ("dx_", "px_")}
    red.cargas(vocab, nombres, top=10).to_csv(tab / "svd_cargas_top10.csv", index=False)
    print(f"· Listo: {out.shape[1] - 1} componentes para {len(out):,} snapshots")


if __name__ == "__main__":
    main()
