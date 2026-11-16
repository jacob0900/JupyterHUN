"""Utilidades compartidas de Persona 3 (M2, M3, M4 visual).

Rutas, conexión DuckDB, estilo de figuras, tamaños de efecto y helpers de informe.
Todo lo que se escribe en outputs/ son AGREGADOS: nunca filas de pacientes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
M1 = DATA / "m1"
M4 = DATA / "m4"
RAW = DATA / "raw"
OUT = ROOT / "outputs"

SEED = 42

# ---------------------------------------------------------------- estilo
COLORES = {
    "hospitalizacion": "#1f4e79",
    "urgencias_observacion": "#c0504d",
    "urgencias_ingreso": "#e6a03c",
    "ambulatorio": "#5b9b64",
}
NOMBRE = {
    "hospitalizacion": "Hospitalización",
    "urgencias_observacion": "Urgencias / observación",
    "urgencias_ingreso": "Ingreso a urgencias",
    "ambulatorio": "Ambulatorio",
}
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]  # dia_semana 1..7 (ISO)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "font.size": 9.5,
        "legend.frameon": False,
    }
)


def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute(f"SET file_search_path='{ROOT.as_posix()}'")
    return c


def q(sql: str, c: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    return (c or con()).sql(sql).df()


def parametros() -> pd.Series:
    return pd.read_parquet(M1 / "parametros.parquet").iloc[0]


def carpeta(modulo: str) -> tuple[Path, Path]:
    """Devuelve (carpeta de figuras, carpeta de tablas) del módulo."""
    fig = OUT / modulo / "figuras"
    tab = OUT / modulo / "tablas"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    return fig, tab


def guardar(fig, ruta: Path) -> str:
    fig.savefig(ruta)
    plt.close(fig)
    return ruta.name


# ---------------------------------------------------------------- tamaños de efecto
def cramers_v(tabla: np.ndarray) -> tuple[float, float, float, int]:
    """Chi² de independencia + V de Cramér con corrección de sesgo (Bergsma 2013).

    Devuelve (chi2, p, V, gl)."""
    tabla = np.asarray(tabla, dtype=float)
    tabla = tabla[tabla.sum(1) > 0][:, tabla.sum(0) > 0]
    chi2, p, gl, _ = stats.chi2_contingency(tabla, correction=False)
    n = tabla.sum()
    r, k = tabla.shape
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    v = np.sqrt(phi2c / max(1e-12, min(kc - 1, rc - 1)))
    return chi2, p, v, gl


def kruskal_eps2(grupos: list[np.ndarray]) -> tuple[float, float, float]:
    """Kruskal-Wallis + épsilon² = H / (n − 1). Devuelve (H, p, ε²)."""
    grupos = [np.asarray(g) for g in grupos if len(g) > 0]
    n = sum(len(g) for g in grupos)
    h, p = stats.kruskal(*grupos)
    return h, p, h / (n - 1)


def auc_univariada(x: np.ndarray, y: np.ndarray) -> float:
    """AUC de una variable numérica para separar y=1 de y=0 (= P(X1 > X0)).

    Equivale a (1 + correlación rango-biserial) / 2."""
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    m = ~np.isnan(x)
    x, y = x[m], y[m]
    r = stats.rankdata(x)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def dif_medianas_ic(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, max_n: int = 20000) -> tuple[float, float, float]:
    """Diferencia de medianas (a − b) con IC 95 % bootstrap percentil.

    Submuestrea a `max_n` por grupo para que sea rápido (el IC sigue siendo
    conservador: con n completo sería más estrecho)."""
    rng = np.random.default_rng(SEED)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    est = np.median(a) - np.median(b)
    if len(a) > max_n:
        a = rng.choice(a, max_n, replace=False)
    if len(b) > max_n:
        b = rng.choice(b, max_n, replace=False)
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    boots = np.median(a[ia], axis=1) - np.median(b[ib], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, lo, hi


def bh(pvals) -> np.ndarray:
    """p-valores ajustados por Benjamini-Hochberg (FDR)."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    m = ~np.isnan(p)
    if m.any():
        out[m] = multipletests(p[m], method="fdr_bh")[1]
    return out


def magnitud_v(v: float, gl: int = 1) -> str:
    """Cohen (1988) ajustado por grados de libertad mínimos (df* = min(r,k) − 1)."""
    gl = max(1, min(gl, 5))
    cortes = {1: (0.10, 0.30, 0.50), 2: (0.07, 0.21, 0.35), 3: (0.06, 0.17, 0.29), 4: (0.05, 0.15, 0.25), 5: (0.04, 0.13, 0.22)}[gl]
    return _mag(v, cortes)


def magnitud_eps2(e: float) -> str:
    """Tomczak & Tomczak (2014): 0.01 pequeño, 0.08 mediano, 0.26 grande."""
    return _mag(e, (0.01, 0.08, 0.26))


def magnitud_auc(a: float) -> str:
    """|AUC − 0.5| en escala de rango-biserial: 0.1/0.3/0.5 → AUC 0.55/0.65/0.75."""
    return _mag(abs(2 * a - 1), (0.10, 0.30, 0.50))


def _mag(x: float, cortes) -> str:
    if np.isnan(x):
        return "—"
    if x < cortes[0]:
        return "despreciable"
    if x < cortes[1]:
        return "pequeño"
    if x < cortes[2]:
        return "mediano"
    return "grande"


def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "—"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------- informe
def md_tabla(df: pd.DataFrame, floatfmt: str = ".3f", index: bool = False) -> str:
    return df.to_markdown(index=index, floatfmt=floatfmt)


def grupo_edad_sql(col: str = "edad") -> str:
    return f"""CASE WHEN {col} < 15 THEN '0-14'
                    WHEN {col} < 45 THEN '15-44'
                    WHEN {col} < 65 THEN '45-64'
                    WHEN {col} < 80 THEN '65-79'
                    ELSE '80+' END"""


GRUPOS_EDAD = ["0-14", "15-44", "45-64", "65-79", "80+"]
