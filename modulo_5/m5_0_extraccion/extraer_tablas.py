"""Extrae de `uninorte_omop` las tablas que necesita el Módulo 5 y las guarda en data/raw.

Solo lectura: la sesión se abre con `readonly=True` y solo se ejecutan `SELECT`
(CLAUDE.md, regla 1). Las filas van directo del servidor al parquet en bloques: no se
imprime ningún dato de pacientes, solo conteos y nombres de columnas.

Tablas:
    measurement           -> data/raw/measurement.parquet           (subfase 5.1)
    condition_occurrence  -> data/raw/condition_occurrence.parquet  (subfase 5.2)

No se extrae `visit_occurrence_id`: el vínculo evento → visita está roto en el HUN.

Requisitos: VPN FortiClient conectada y un archivo `.env` en la raíz del proyecto
(copiar `.env.ejemplo` y completar usuario y contraseña).

Uso:
    python modulo_5/m5_0_extraccion/extraer_tablas.py probar     # conexión, columnas y conteos
    python modulo_5/m5_0_extraccion/extraer_tablas.py extraer    # descarga las dos tablas
    python modulo_5/m5_0_extraccion/extraer_tablas.py extraer --tablas measurement
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

# Columnas deseadas por tabla (se piden solo las que existan en el servidor).
TABLAS: dict[str, list[str]] = {
    "measurement": [
        "person_id", "measurement_concept_id", "measurement_datetime", "measurement_date",
        "measurement_source_value", "measurement_source_value_extended", "value_as_number",
    ],
    "condition_occurrence": [
        "person_id", "condition_concept_id", "condition_start_datetime", "condition_start_date",
        "condition_type_concept_id",
    ],
}

TIPOS = {  # information_schema.data_type -> tipo arrow
    "smallint": pa.int64(), "integer": pa.int64(), "bigint": pa.int64(),
    "numeric": pa.float64(), "real": pa.float64(), "double precision": pa.float64(),
    "timestamp without time zone": pa.timestamp("us"), "timestamp with time zone": pa.timestamp("us", tz="UTC"),
    "date": pa.date32(), "boolean": pa.bool_(),
}


def _leer_env() -> dict[str, str]:
    ruta = ROOT / ".env"
    if not ruta.exists():
        sys.exit("✗ Falta el archivo .env en la raíz del proyecto. Copia .env.ejemplo como .env y completa usuario y contraseña.")
    env = {}
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            k, v = linea.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    env = {**env, **{k: v for k, v in os.environ.items() if k.startswith("HUN_DB_")}}
    faltan = [k for k in ("HUN_DB_HOST", "HUN_DB_PORT", "HUN_DB_NAME", "HUN_DB_USER", "HUN_DB_PASSWORD") if not env.get(k)]
    if faltan:
        sys.exit(f"✗ En .env faltan: {', '.join(faltan)}")
    return env


def conectar():
    import psycopg2

    env = _leer_env()
    try:
        con = psycopg2.connect(
            host=env["HUN_DB_HOST"], port=int(env["HUN_DB_PORT"]), dbname=env["HUN_DB_NAME"],
            user=env["HUN_DB_USER"], password=env["HUN_DB_PASSWORD"], connect_timeout=15,
            application_name="hun_m5_extraccion",
        )
    except psycopg2.OperationalError as e:
        sys.exit(f"✗ No se pudo conectar ({str(e).splitlines()[0]}). ¿Está la VPN conectada?")
    con.set_session(readonly=True, autocommit=False)
    return con, env.get("HUN_DB_SCHEMA", "uninorte_omop")


def columnas_servidor(con, esquema: str, tabla: str) -> dict[str, str]:
    with con.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (esquema, tabla),
        )
        return dict(cur.fetchall())


def probar(tablas: list[str]) -> None:
    con, esquema = conectar()
    print(f"✓ Conectado (solo lectura) · esquema {esquema}")
    with con.cursor() as cur:
        for t in tablas:
            cols = columnas_servidor(con, esquema, t)
            if not cols:
                print(f"  ✗ {t}: no existe en {esquema}")
                continue
            # estimación rápida del planificador (no recorre la tabla)
            cur.execute("SELECT reltuples::bigint FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                        "WHERE n.nspname = %s AND c.relname = %s", (esquema, t))
            n = cur.fetchone()[0]
            usar = [c for c in TABLAS[t] if c in cols]
            print(f"  · {t}: ~{n:,} filas · columnas en servidor: {', '.join(cols)}")
            print(f"    se extraerán: {', '.join(usar)}")
    con.close()


def extraer(tablas: list[str], bloque: int) -> None:
    con, esquema = conectar()
    RAW.mkdir(parents=True, exist_ok=True)
    for t in tablas:
        cols_srv = columnas_servidor(con, esquema, t)
        usar = [c for c in TABLAS[t] if c in cols_srv]
        if "person_id" not in usar:
            print(f"  ✗ {t}: no existe o no tiene person_id; se omite")
            continue
        schema = pa.schema([(c, TIPOS.get(cols_srv[c], pa.string())) for c in usar])
        destino = RAW / f"{t}.parquet"
        tmp = destino.with_suffix(".parquet.parcial")
        # numeric llega como Decimal de Python: se convierte a double en el servidor.
        expr = [f'"{c}"::double precision' if cols_srv[c] == "numeric" else f'"{c}"' for c in usar]
        sql = f'SELECT {", ".join(expr)} FROM "{esquema}"."{t}"'

        print(f"· {t}: extrayendo {len(usar)} columnas")
        t0, n = time.time(), 0
        with con.cursor(name=f"cur_{t}") as cur:  # cursor del lado del servidor: no carga todo en RAM
            cur.itersize = bloque
            cur.execute(sql)
            with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
                while True:
                    filas = cur.fetchmany(bloque)
                    if not filas:
                        break
                    columnas = list(zip(*filas))
                    w.write_table(pa.table([pa.array(col, type=f.type) for col, f in zip(columnas, schema)], schema=schema))
                    n += len(filas)
                    print(f"    {n:>12,} filas · {time.time() - t0:6.0f} s", end="\r")
        con.commit()
        tmp.replace(destino)
        print(f"    ✓ {n:,} filas en {time.time() - t0:.0f} s -> {destino.relative_to(ROOT)} "
              f"({destino.stat().st_size / 2**20:.1f} MB)")
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["probar", "extraer"])
    ap.add_argument("--tablas", nargs="+", default=list(TABLAS), choices=list(TABLAS))
    ap.add_argument("--bloque", type=int, default=200_000, help="filas por bloque")
    a = ap.parse_args()
    probar(a.tablas) if a.accion == "probar" else extraer(a.tablas, a.bloque)


if __name__ == "__main__":
    main()
