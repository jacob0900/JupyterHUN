"""Subfase 7.4 · Tres pruebas formales de superioridad (α = 0,05).

1. **DeLong pareado** (ΔAUC-ROC, ganador vs cada baseline).
   - Clásico (DeLong 1988, implementación rápida de Sun y Xu 2014), que supone observaciones independientes.
   - **Por conglomerados** (Obuchowski 1997): los valores de colocación se suman por paciente, porque cada
     paciente aporta ~5,6 snapshots correlacionados. Es la prueba principal; con conglomerados de tamaño 1
     coincide exactamente con el clásico.
   - IC 95 % **BCa** de ΔAUC con bootstrap pareado por paciente (réplicas de la 7.1) y aceleración por
     jackknife de grupos de pacientes. Holm sobre las comparaciones.
2. **Friedman + Nemenyi** sobre la matriz pliegues × modelos, con rangos promedio, Iman-Davenport, W de
   Kendall y diagrama de diferencia crítica. Los p de Nemenyi salen del rango estudentizado.
3. **Wilcoxon de rangos con signo**, top 1 vs top 2, sobre las particiones, con Holm entre métricas y la
   correlación biserial de rangos como tamaño de efecto.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from modulo_7.m7_1_multimetrica import CurvaPreordenada, replicas_bootstrap

ALFA = 0.05
DELTA_RELEVANTE = 0.01  # margen pre-especificado de relevancia práctica para ΔAUC


# ---------------------------------------------------------------------------- DeLong
def _midrank(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, method="average")


def colocaciones(y, s) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Valores de colocación de DeLong: V10 (un valor por positivo) y V01 (por negativo), con sus índices."""
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    ip, ineg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    m, n = len(ip), len(ineg)
    tz = _midrank(np.r_[s[ip], s[ineg]])
    tx, ty = _midrank(s[ip]), _midrank(s[ineg])
    v10 = (tz[:m] - tx) / n            # fracción de negativos por debajo de cada positivo
    v01 = 1 - (tz[m:] - ty) / m        # fracción de positivos por encima de cada negativo
    return v10, v01, ip, ineg


def _var_conglomerados(d10: np.ndarray, d01: np.ndarray, tiene_pos: np.ndarray, tiene_neg: np.ndarray,
                       M: int, N: int) -> float:
    """Varianza de Obuchowski a partir de las desviaciones por conglomerado (sumas centradas)."""
    i10, i01, i = tiene_pos.sum(), tiene_neg.sum(), len(d10)
    return float(i10 / ((i10 - 1) * M ** 2) * (d10 ** 2).sum()
                 + i01 / ((i01 - 1) * N ** 2) * (d01 ** 2).sum()
                 + 2 * i / ((i - 1) * M * N) * (d10 * d01).sum())


def delong(y, s1, s2=None, grupos=None) -> dict:
    """AUC (o ΔAUC = AUC1 − AUC2 si se pasa s2), su error estándar, z y p bilateral.

    Sin `grupos`: DeLong clásico. Con `grupos`: varianza por conglomerados de Obuchowski.
    """
    y = np.asarray(y).astype(int)
    v10a, v01a, ip, ineg = colocaciones(y, s1)
    if s2 is not None:
        v10b, v01b, _, _ = colocaciones(y, s2)
    else:
        v10b, v01b = np.zeros_like(v10a), np.zeros_like(v01a)
    a1, a2 = v10a.mean(), (v10b.mean() if s2 is not None else 0.0)
    d = a1 - a2
    M, N = len(ip), len(ineg)
    if grupos is None:
        dv10, dv01 = v10a - v10b, v01a - v01b
        var = dv10.var(ddof=1) / M + dv01.var(ddof=1) / N
    else:
        cod, inv = np.unique(np.asarray(grupos), return_inverse=True)
        k = len(cod)
        s10 = np.bincount(inv[ip], weights=v10a - v10b, minlength=k)
        s01 = np.bincount(inv[ineg], weights=v01a - v01b, minlength=k)
        m_k = np.bincount(inv[ip], minlength=k)
        n_k = np.bincount(inv[ineg], minlength=k)
        var = _var_conglomerados(s10 - m_k * d, s01 - n_k * d, m_k > 0, n_k > 0, M, N)
    se = float(np.sqrt(var))
    z = d / se if se > 0 else np.inf
    log10_p = float((np.log(2) + stats.norm.logsf(abs(z))) / np.log(10)) if np.isfinite(z) else -np.inf
    out = {"auc_1": a1, "estimado": d, "se": se, "z": z, "p": float(2 * stats.norm.sf(abs(z))), "log10_p": log10_p,
           "ic95_inf": d - 1.959964 * se, "ic95_sup": d + 1.959964 * se}
    if s2 is not None:
        out["auc_2"] = a2
    return out


# ---------------------------------------------------------------------------- bootstrap BCa
def bootstrap_delta(y, s1, s2, grupos, n_iter: int = 1000, semilla: int = 42) -> np.ndarray:
    """Réplicas de ΔAUC con remuestreo de pacientes estratificado (mismas réplicas que la 7.1)."""
    c1, c2 = CurvaPreordenada.desde(y, s1), CurvaPreordenada.desde(y, s2)
    return np.array([c1.auc_roc(w) - c2.auc_roc(w) for w in replicas_bootstrap(y, grupos, n_iter, semilla)])


def jackknife_grupos(y, s1, s2, grupos, n_grupos: int = 100, semilla: int = 42) -> np.ndarray:
    """ΔAUC dejando fuera, por turnos, uno de `n_grupos` bloques aleatorios de pacientes."""
    c1, c2 = CurvaPreordenada.desde(y, s1), CurvaPreordenada.desde(y, s2)
    cod, inv = np.unique(np.asarray(grupos), return_inverse=True)
    bloque = np.random.default_rng(semilla).integers(0, n_grupos, len(cod))[inv]
    out = []
    for b in range(n_grupos):
        w = (bloque != b).astype(float)
        out.append(c1.auc_roc(w) - c2.auc_roc(w))
    return np.array(out)


def ic_bca(estimado: float, boot: np.ndarray, jack: np.ndarray, alfa: float = ALFA) -> tuple[float, float]:
    """Intervalo BCa (Efron 1987): corrección de sesgo z0 y aceleración a por jackknife."""
    prop = (np.sum(boot < estimado) + 0.5 * np.sum(boot == estimado)) / len(boot)
    z0 = stats.norm.ppf(np.clip(prop, 1e-6, 1 - 1e-6))
    dj = jack.mean() - jack
    a = (dj ** 3).sum() / (6 * ((dj ** 2).sum()) ** 1.5) if (dj ** 2).sum() > 0 else 0.0
    zs = stats.norm.ppf([alfa / 2, 1 - alfa / 2])
    q = stats.norm.cdf(z0 + (z0 + zs) / (1 - a * (z0 + zs)))
    lo, hi = np.quantile(boot, q)
    return float(lo), float(hi)


def p_bootstrap(boot: np.ndarray) -> float:
    """p bilateral por inversión del bootstrap: 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)), con piso 1/(B+1)."""
    b = len(boot)
    return float(min(1.0, max(2 * min((boot <= 0).sum(), (boot >= 0).sum()) / b, 1 / (b + 1))))


def comparar_ganador(y, puntajes: dict[str, np.ndarray], ganador: str, grupos, n_iter: int = 1000,
                     semilla: int = 42, n_jack: int = 100) -> pd.DataFrame:
    """Prueba 1 completa: ganador vs cada baseline (filas comunes no nulas), con Holm."""
    filas = []
    for b, sb in puntajes.items():
        if b == ganador:
            continue
        ok = ~np.isnan(puntajes[ganador]) & ~np.isnan(sb)
        yy, s1, s2, g = np.asarray(y)[ok], puntajes[ganador][ok], sb[ok], np.asarray(grupos)[ok]
        cl, co = delong(yy, s1, s2), delong(yy, s1, s2, grupos=g)
        boot = bootstrap_delta(yy, s1, s2, g, n_iter, semilla)
        jack = jackknife_grupos(yy, s1, s2, g, n_jack, semilla)
        lo, hi = ic_bca(co["estimado"], boot, jack)
        filas.append({"ganador": ganador, "baseline": b, "n": int(ok.sum()), "auc_ganador": co["auc_1"],
                      "auc_baseline": co["auc_2"], "delta_auc": co["estimado"],
                      "p_delong_clasico": cl["p"], "se_clasico": cl["se"],
                      "p_delong_conglomerados": co["p"], "se_conglomerados": co["se"], "z_conglomerados": co["z"],
                      "log10_p_conglomerados": co["log10_p"],
                      "inflacion_se": co["se"] / cl["se"],
                      "ic95_bca_inf": lo, "ic95_bca_sup": hi, "p_bootstrap": p_bootstrap(boot)})
    df = pd.DataFrame(filas)
    df["p_holm"] = multipletests(df.p_delong_conglomerados, method="holm")[1]
    df["superior"] = (df.p_holm < ALFA) & (df.delta_auc > 0)
    df["relevante"] = df.superior & (df.ic95_bca_inf >= DELTA_RELEVANTE)
    return df


def matriz_p_delong(y, puntajes: dict[str, np.ndarray], grupos, orden: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """p de DeLong por conglomerados para todos los pares (filas comunes), crudos y ajustados por Holm."""
    pares, ps = [], []
    for i, a in enumerate(orden):
        for b in orden[i + 1:]:
            ok = ~np.isnan(puntajes[a]) & ~np.isnan(puntajes[b])
            r = delong(np.asarray(y)[ok], puntajes[a][ok], puntajes[b][ok], grupos=np.asarray(grupos)[ok])
            pares.append((a, b))
            ps.append(r["p"])
    adj = multipletests(ps, method="holm")[1]
    crudo = pd.DataFrame(np.nan, index=orden, columns=orden)
    holm = crudo.copy()
    for (a, b), p, q in zip(pares, ps, adj):
        crudo.loc[a, b] = crudo.loc[b, a] = p
        holm.loc[a, b] = holm.loc[b, a] = q
    return crudo, holm


# ---------------------------------------------------------------------------- Friedman + Nemenyi
def friedman_nemenyi(matriz: pd.DataFrame, mayor_es_mejor: bool = True, alfa: float = ALFA) -> dict:
    """Matriz bloques (pliegues) × modelos. Rango 1 = mejor."""
    x = matriz.to_numpy(float)
    nb, k = x.shape
    rangos = np.apply_along_axis(lambda r: stats.rankdata(-r if mayor_es_mejor else r), 1, x)
    r_medio = pd.Series(rangos.mean(axis=0), index=matriz.columns).sort_values()
    chi2, p = stats.friedmanchisquare(*[x[:, j] for j in range(k)])
    ff = (nb - 1) * chi2 / (nb * (k - 1) - chi2) if nb * (k - 1) != chi2 else np.inf  # Iman-Davenport
    p_ff = float(stats.f.sf(ff, k - 1, (k - 1) * (nb - 1))) if np.isfinite(ff) else 0.0
    se = np.sqrt(k * (k + 1) / (6 * nb))
    q_alfa = stats.studentized_range.ppf(1 - alfa, k, np.inf) / np.sqrt(2)
    rm = r_medio.reindex(matriz.columns).to_numpy()
    q = np.abs(rm[:, None] - rm[None, :]) / se
    p_nem = stats.studentized_range.sf(q * np.sqrt(2), k, np.inf)
    np.fill_diagonal(p_nem, np.nan)
    return {"chi2": float(chi2), "p": float(p), "iman_davenport_F": float(ff), "p_iman_davenport": p_ff,
            "w_kendall": float(chi2 / (nb * (k - 1))), "rangos_medios": r_medio, "cd": float(q_alfa * se),
            "q_alfa": float(q_alfa), "n_bloques": nb, "k": k,
            "p_nemenyi": pd.DataFrame(p_nem, index=matriz.columns, columns=matriz.columns)}


# ---------------------------------------------------------------------------- Wilcoxon
def wilcoxon_pareado(a: np.ndarray, b: np.ndarray, alternativa: str = "two-sided") -> dict:
    """Wilcoxon exacto (si n ≤ 50 y sin empates) con la correlación biserial de rangos como efecto."""
    d = np.asarray(a, float) - np.asarray(b, float)
    res = stats.wilcoxon(d, alternative=alternativa, method="auto", zero_method="wilcox")
    r = stats.rankdata(np.abs(d[d != 0]))
    rbc = (r[d[d != 0] > 0].sum() - r[d[d != 0] < 0].sum()) / r.sum() if len(r) else 0.0
    n = int((d != 0).sum())
    p_min = 2 / 2 ** n if alternativa == "two-sided" else 1 / 2 ** n
    return {"n_pares": n, "mediana_diferencia": float(np.median(d)), "estadistico_W": float(res.statistic),
            "p": float(res.pvalue), "biserial_rangos": float(rbc), "p_minimo_alcanzable": min(1.0, p_min),
            "a_mejor_en": int((d > 0).sum())}


def wilcoxon_metricas(bloques: pd.DataFrame, top1: str, top2: str, metricas: dict[str, bool]) -> pd.DataFrame:
    """Wilcoxon top1 vs top2 por métrica (columna `metrica`, `modelo`, `bloque`, `valor`), Holm entre métricas.

    `metricas`: nombre → True si mayor es mejor. Para las de "menor es mejor" se compara (top2 − top1)
    para que una diferencia positiva siempre favorezca al top 1.
    """
    filas = []
    for met, mayor in metricas.items():
        t = bloques[bloques.metrica == met].pivot(index="bloque", columns="modelo", values="valor")
        a, b = (t[top1], t[top2]) if mayor else (t[top2], t[top1])
        filas.append({"metrica": met, "top1": top1, "top2": top2, "media_top1": t[top1].mean(),
                      "media_top2": t[top2].mean(), **wilcoxon_pareado(a.to_numpy(), b.to_numpy())})
    df = pd.DataFrame(filas)
    df["p_holm"] = multipletests(df.p, method="holm")[1]
    df["top1_superior"] = (df.p_holm < ALFA) & (df.mediana_diferencia > 0)
    return df


def veredicto(p: float, efecto_positivo: bool = True) -> str:
    if p < ALFA and efecto_positivo:
        return "SUPERIORIDAD ESTADÍSTICA (p < 0,05)"
    if p < ALFA:
        return "DIFERENCIA SIGNIFICATIVA A FAVOR DEL OTRO MODELO (p < 0,05)"
    return ("EQUIVALENCIA ESTADÍSTICA según la rúbrica (p ≥ 0,05): no se demuestra diferencia "
            "(formalmente no es una prueba de equivalencia)")


# ---------------------------------------------------------------------------- gráficos
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def graficar_cd(rangos: pd.Series, cd: float, ruta, titulo: str) -> None:
    """Diagrama de diferencia crítica (Demšar 2006). Rango 1 (mejor) a la izquierda; las barras azules unen
    modelos cuyos rangos medios difieren menos que CD (no distinguibles por Nemenyi)."""
    plt = _plt()
    r = rangos.sort_values()
    k = len(r)
    mitad = (k + 1) // 2
    alto = 0.42 * max(mitad, k - mitad) + 1.6
    fig, a = plt.subplots(figsize=(10, alto))
    a.set_xlim(-1.6, k + 2.6)
    y_eje = 0.0
    a.hlines(y_eje, 1, k, color="k", lw=1.2)
    for t in range(1, k + 1):
        a.vlines(t, y_eje, y_eje + 0.08, color="k", lw=1)
        a.text(t, y_eje + 0.13, str(t), ha="center", va="bottom", fontsize=9)
    a.hlines(y_eje + 0.55, 1, 1 + cd, color="#d62728", lw=2.5)
    a.vlines([1, 1 + cd], y_eje + 0.5, y_eje + 0.6, color="#d62728", lw=1.5)
    a.text(1 + cd / 2, y_eje + 0.62, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9, color="#d62728")
    # grupos no distinguibles (cliques maximales)
    vals = r.to_numpy()
    grupos = []
    for i in range(k):
        j = max(jj for jj in range(i, k) if vals[jj] - vals[i] < cd)
        if j > i and not any(gi <= i and j <= gj for gi, gj in grupos):
            grupos.append((i, j))
    y_g = y_eje - 0.18
    for n, (gi, gj) in enumerate(grupos):
        y = y_g - 0.14 * n
        a.hlines(y, vals[gi] - 0.05, vals[gj] + 0.05, color="#1f77b4", lw=5, alpha=0.75)
    y0 = y_g - 0.14 * max(len(grupos), 1) - 0.15
    for i, (m, v) in enumerate(r.items()):
        izq = i < mitad
        fila = i if izq else k - 1 - i
        y = y0 - 0.32 * fila
        x_fin = 0.7 if izq else k + 0.3
        a.plot([v, v], [y_eje, y], color="k", lw=0.8)
        a.plot([v, x_fin], [y, y], color="k", lw=0.8)
        a.text(x_fin - 0.08 if izq else x_fin + 0.08, y, f"{m} ({v:.2f})", ha="right" if izq else "left",
               va="center", fontsize=9)
    a.set_ylim(y0 - 0.32 * max(mitad, k - mitad) + 0.1, y_eje + 0.9)
    a.axis("off")
    a.set_title(titulo, fontweight="bold", fontsize=10)
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)


def graficar_heatmaps(mats: dict[str, pd.DataFrame], ruta, titulo: str,
                      anotaciones: dict[str, pd.DataFrame] | None = None) -> None:
    """Mapas de calor de p-valores (escala log10, marca de p < 0,05). `anotaciones`: texto extra por celda
    (p. ej. ΔAUC fila − columna)."""
    anotaciones = anotaciones or {}
    plt = _plt()
    from matplotlib.colors import LogNorm
    fig, ejes = plt.subplots(1, len(mats), figsize=(7.2 * len(mats), 6.2))
    for a, (nombre, m) in zip(np.atleast_1d(ejes), mats.items()):
        v = m.to_numpy(float)
        im = a.imshow(np.clip(v, 1e-300, 1), cmap="viridis_r", norm=LogNorm(vmin=1e-10, vmax=1))
        a.set_xticks(range(len(m)), m.columns, rotation=45, ha="right", fontsize=9)
        a.set_yticks(range(len(m)), m.index, fontsize=9)
        for i in range(len(m)):
            for j in range(len(m)):
                if np.isfinite(v[i, j]):
                    txt = ("p<1e-10" if v[i, j] < 1e-10 else f"p={v[i, j]:.3g}") + ("*" if v[i, j] < ALFA else "")
                    if nombre in anotaciones:
                        txt = f"{anotaciones[nombre].iloc[i, j]}\n{txt}"
                    a.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                           color="black" if v[i, j] < 1e-3 else "white")
        a.set_title(nombre, fontsize=10)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, label="p-valor (escala log)")
    fig.suptitle(titulo + "  (* p < 0,05)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)
