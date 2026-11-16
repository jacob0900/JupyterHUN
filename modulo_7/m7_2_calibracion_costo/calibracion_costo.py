"""Subfase 7.2 · Calibración (isotónica / sigmoide) y umbral de costo asimétrico (c_FN = 5 · c_FP).

Dos niveles, para no reutilizar datos:

1. **Desarrollo (OOF del entrenamiento estable).** Calibración *cruzada por pliegue*: el calibrador del pliegue
   k se ajusta con las predicciones OOF de los otros cuatro pliegues y se aplica al pliegue k. Así cada
   probabilidad calibrada viene de un modelo y un calibrador que no vieron a ese paciente. Esto replica lo
   que hace `CalibratedClassifierCV(cv=pliegues agrupados)`, sin reentrenar los modelos. Sobre estas
   probabilidades se elige el método (menor Brier) y el umbral θ* (menor costo).
2. **Modelo desplegable.** `CalibratedClassifierCV(Pipeline, method, cv=pliegues de la 6.1, ensemble=True)`
   ajustado con todo el entrenamiento estable. Es el que se evalúa **una sola vez** en la prueba temporal,
   con θ* ya congelado.

Costo: C(θ) = c_FN · FN(θ) + c_FP · FP(θ), con alerta si p ≥ θ. Con probabilidades bien calibradas el
umbral de Bayes es c_FP / (c_FP + c_FN) = 1/6 ≈ 0,167. Las políticas de referencia "alertar a todos" y
"no alertar nunca" no necesitan modelo; el valor del modelo es lo que ahorra frente a la mejor de ellas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from modulo_7.m7_1_multimetrica import ece_igual_frecuencia

C_FN, C_FP = 5.0, 1.0
UMBRALES = np.round(np.arange(0.01, 1.0, 0.01), 2)  # θ ∈ [0,01; 0,99]
METODOS = ("isotonic", "sigmoid")


# ---------------------------------------------------------------------------- calibración
def _calibrador(metodo: str):
    if metodo == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    if metodo == "sigmoid":  # Platt: regresión logística de 1 variable sobre el puntaje
        return LogisticRegression(C=1e6, max_iter=1000)
    raise ValueError(f"Método de calibración desconocido: {metodo}")


def _ajustar_aplicar(cal, s_fit, y_fit, s_apl):
    if isinstance(cal, IsotonicRegression):
        return cal.fit(s_fit, y_fit).predict(s_apl)
    return cal.fit(s_fit.reshape(-1, 1), y_fit).predict_proba(s_apl.reshape(-1, 1))[:, 1]


def calibrar_cruzado(s, y, pliegues, metodo: str) -> np.ndarray:
    """Probabilidades calibradas fuera de pliegue: el pliegue k se calibra con los otros pliegues."""
    s, y, pliegues = np.asarray(s, float), np.asarray(y).astype(int), np.asarray(pliegues)
    out = np.full(len(s), np.nan)
    for k in np.unique(pliegues):
        en = pliegues == k
        out[en] = _ajustar_aplicar(_calibrador(metodo), s[~en], y[~en], s[en])
    return out


def ajustar_calibrado_final(pipe, X, y, pliegues, metodo: str) -> CalibratedClassifierCV:
    """Modelo desplegable: 5 Pipelines (uno por pliegue agrupado), cada uno calibrado en su pliegue de validación."""
    pliegues = np.asarray(pliegues)
    cv = [(np.flatnonzero(pliegues != k), np.flatnonzero(pliegues == k)) for k in np.unique(pliegues)]
    return CalibratedClassifierCV(pipe, method=metodo, cv=cv, ensemble=True).fit(X, np.asarray(y))


def curva_fiabilidad(y, p, n_bins: int = 15) -> pd.DataFrame:
    """Diagrama de fiabilidad con intervalos de igual frecuencia: p medio vs fracción observada de positivos."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    orden = np.argsort(p, kind="mergesort")
    filas = [{"p_media": p[b].mean(), "fraccion_positivos": y[b].mean(), "n": len(b)}
             for b in np.array_split(orden, n_bins) if len(b)]
    return pd.DataFrame(filas)


def metricas_calibracion(y, p) -> dict:
    y = np.asarray(y).astype(int)
    prev = y.mean()
    brier = brier_score_loss(y, p)
    return {"n": len(y), "prevalencia": prev, "p_media": float(np.mean(p)), "auc_roc": roc_auc_score(y, p),
            "auc_pr": average_precision_score(y, p), "brier": brier, "brier_skill": 1 - brier / (prev * (1 - prev)),
            "ece": ece_igual_frecuencia(y, p)}


# ---------------------------------------------------------------------------- costo asimétrico
def barrido_umbrales(y, p, umbrales=UMBRALES, c_fn: float = C_FN, c_fp: float = C_FP) -> pd.DataFrame:
    """FN, FP y costo esperado por cada 100 evaluaciones para cada θ (alerta si p ≥ θ)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    ps, ns = np.sort(p[y == 1]), np.sort(p[y == 0])
    umbrales = np.asarray(umbrales, float)
    fn = np.searchsorted(ps, umbrales, side="left")          # positivos con p < θ
    fp = len(ns) - np.searchsorted(ns, umbrales, side="left")  # negativos con p ≥ θ
    n = len(y)
    return pd.DataFrame({"umbral": umbrales, "fn": fn, "fp": fp, "tp": len(ps) - fn, "tn": len(ns) - fp,
                         "costo_por_100": (c_fn * fn + c_fp * fp) / n * 100})


def umbral_optimo(barrido: pd.DataFrame) -> float:
    """θ* que minimiza el costo (en empate, el primero: menos alertas no es un criterio aquí)."""
    return float(barrido.umbral.iloc[int(np.argmin(barrido.costo_por_100.to_numpy()))])


def desempeno_umbral(y, p, umbral: float, c_fn: float = C_FN, c_fp: float = C_FP) -> dict:
    """Matriz de confusión y costo por cada 100 evaluaciones para una regla de alerta p ≥ θ."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    alerta = p >= umbral
    tp, fp = int((alerta & (y == 1)).sum()), int((alerta & (y == 0)).sum())
    fn, tn = int((~alerta & (y == 1)).sum()), int((~alerta & (y == 0)).sum())
    n = len(y)
    return {
        "umbral": umbral, "n": n,
        "sensibilidad": tp / (tp + fn) if tp + fn else np.nan,
        "especificidad": tn / (tn + fp) if tn + fp else np.nan,
        "vpp": tp / (tp + fp) if tp + fp else np.nan,
        "vpn": tn / (tn + fn) if tn + fn else np.nan,
        "alertas_por_100": (tp + fp) / n * 100,
        "fn_por_100": fn / n * 100, "fp_por_100": fp / n * 100,
        "costo_por_100": (c_fn * fn + c_fp * fp) / n * 100,
    }


def politicas_referencia(y, c_fn: float = C_FN, c_fp: float = C_FP) -> dict:
    """Costo por 100 de las políticas sin modelo."""
    prev = float(np.mean(y))
    return {"alertar_a_todos": c_fp * (1 - prev) * 100, "no_alertar_nunca": c_fn * prev * 100}


def comparar_umbrales(y, p, reglas: dict[str, float], c_fn: float = C_FN, c_fp: float = C_FP) -> pd.DataFrame:
    """Tabla θ = 0,5 vs θ* (y otras reglas), con el ahorro frente a θ = 0,5 y a la mejor política sin modelo."""
    ref = politicas_referencia(y, c_fn, c_fp)
    mejor_ref = min(ref.values())
    filas = [{"regla": r, **desempeno_umbral(y, p, t, c_fn, c_fp)} for r, t in reglas.items()]
    df = pd.DataFrame(filas)
    base = df.loc[df.umbral == 0.5, "costo_por_100"]
    if len(base):
        df["ahorro_vs_0_5"] = base.iloc[0] - df.costo_por_100
    df["ahorro_vs_mejor_sin_modelo"] = mejor_ref - df.costo_por_100
    return df


# ---------------------------------------------------------------------------- gráficos
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


ESTILO = {"sin_calibrar": ("#7f7f7f", "o", "Sin calibrar"), "isotonic": ("#1f77b4", "s", "Isotónica"),
          "sigmoid": ("#d62728", "^", "Sigmoide (Platt)")}


def graficar_fiabilidad(y, probas: dict[str, dict[str, np.ndarray]], ruta, titulo: str, n_bins: int = 15) -> None:
    """Un panel por modelo; en cada uno, una curva por versión (sin calibrar / isotónica / sigmoide)."""
    plt = _plt()
    n = len(probas)
    fig, ejes = plt.subplots(2, n, figsize=(5.2 * n, 6.6), gridspec_kw={"height_ratios": [3.2, 1]}, squeeze=False)
    y = np.asarray(y)
    for j, (modelo, versiones) in enumerate(probas.items()):
        a, h = ejes[0, j], ejes[1, j]
        a.plot([0, 1], [0, 1], "k:", lw=0.8)
        for v, p in versiones.items():
            ok = ~np.isnan(p)
            c = curva_fiabilidad(y[ok], p[ok], n_bins)
            color, marca, nombre = ESTILO.get(v, ("k", "o", v))
            a.plot(c.p_media, c.fraccion_positivos, marker=marca, ms=4, lw=1.4, color=color,
                   label=f"{nombre} · ECE {ece_igual_frecuencia(y[ok], p[ok], n_bins):.3f}")
            h.hist(p[ok], bins=40, range=(0, 1), histtype="step", color=color, lw=1.2)
        a.set(xlim=(0, 1), ylim=(0, 1), title=modelo, xlabel="", ylabel="Fracción observada de positivos" if j == 0 else "")
        a.legend(fontsize=8, loc="upper left")
        a.grid(alpha=0.3)
        h.set(xlim=(0, 1), xlabel="Probabilidad predicha", ylabel="Snapshots" if j == 0 else "")
        h.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.suptitle(titulo, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def graficar_costo(barridos: dict[str, pd.DataFrame], referencias: dict, ruta, titulo: str,
                   c_fn: float = C_FN, c_fp: float = C_FP) -> None:
    """Costo por cada 100 evaluaciones vs θ, con θ*, el umbral de Bayes y las políticas sin modelo."""
    plt = _plt()
    fig, a = plt.subplots(figsize=(8, 5))
    for i, (m, b) in enumerate(barridos.items()):
        t = umbral_optimo(b)
        linea, = a.plot(b.umbral, b.costo_por_100, lw=1.6, label=f"{m} · θ* = {t:.2f} ({b.costo_por_100.min():.1f})")
        a.plot([t], [b.costo_por_100.min()], "o", color=linea.get_color())
    a.axhline(referencias["alertar_a_todos"], color="k", ls="--", lw=0.9,
              label=f"Alertar a todos ({referencias['alertar_a_todos']:.1f})")
    a.axvline(c_fp / (c_fp + c_fn), color="grey", ls=":", lw=0.9, label=f"Bayes c_FP/(c_FP+c_FN) = {c_fp / (c_fp + c_fn):.3f}")
    a.axvline(0.5, color="grey", ls="-.", lw=0.8, label="θ = 0,5")
    a.set(xlabel="Umbral de alerta θ (alerta si p ≥ θ)", ylabel=f"Costo esperado por 100 (c_FN = {c_fn:g}, c_FP = {c_fp:g})",
          title=titulo, ylim=(0, min(referencias["no_alertar_nunca"], 1.05 * max(b.costo_por_100.max() for b in barridos.values()))))
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
