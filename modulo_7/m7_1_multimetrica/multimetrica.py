"""Subfase 7.1 · Evaluación multimétrica fuera de muestra (OOF de la 6.4).

Métricas por modelo:
- discriminación: AUC-ROC y AUC-PR (average precision; su valor sin información es la prevalencia, no 0,5);
- calibración: Brier y ECE con 15 intervalos de igual frecuencia;
- umbral 0,5: F1, sensibilidad, especificidad y precisión.

Intervalos de confianza del 95 % para AUC-ROC y AUC-PR con **bootstrap estratificado por conglomerados**:
- se remuestrean **pacientes** (cada uno aporta ~5,6 snapshots correlacionados; remuestrear filas daría IC
  demasiado estrechos), estratificados por su composición de clases (todo negativo / mixto / todo positivo);
- cada réplica se aplica como pesos (nº de veces que sale el paciente) y las métricas se calculan con un orden
  precalculado, en O(n) por réplica (`CurvaPreordenada`, equivalente a scikit-learn con `sample_weight`);
- las mismas réplicas sirven para todos los modelos (bootstrap pareado, reutilizable en comparaciones de M7).

Salidas de tipo `decision_function` (SVM): se convierten a probabilidad con **Platt cruzado por pliegue**
(cada pliegue se calibra con los otros cuatro), sin usar sus propias etiquetas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

N_BINS_ECE = 15
UMBRAL = 0.5
N_BOOTSTRAP = 1000


# ---------------------------------------------------------------------------- métricas puntuales
def ece_igual_frecuencia(y, p, n_bins: int = N_BINS_ECE) -> float:
    """Expected Calibration Error con `n_bins` intervalos de igual frecuencia (cuantiles de p).

    ECE = Σ_b (n_b / N) · |media(y_b) − media(p_b)|.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    orden = np.argsort(p, kind="mergesort")
    total = 0.0
    for b in np.array_split(orden, n_bins):
        if len(b):
            total += len(b) * abs(y[b].mean() - p[b].mean())
    return total / len(y)


def metricas_umbral(y, p, umbral: float = UMBRAL) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "f1": f1_score(y, pred, zero_division=0),
        "sensibilidad": tp / (tp + fn) if tp + fn else np.nan,
        "especificidad": tn / (tn + fp) if tn + fp else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "tasa_predichos_positivos": pred.mean(),
    }


def metricas(y, p) -> dict:
    """Vector multimétrico completo (sin IC) para probabilidades `p`."""
    y = np.asarray(y).astype(int)
    prev = y.mean()
    brier = brier_score_loss(y, p)
    return {
        "n": len(y),
        "prevalencia": prev,
        "auc_roc": roc_auc_score(y, p),
        "auc_pr": average_precision_score(y, p),
        "auc_pr_sin_informacion": prev,
        "brier": brier,
        "brier_skill": 1 - brier / (prev * (1 - prev)),  # vs predecir siempre la prevalencia
        "ece": ece_igual_frecuencia(y, p),
        **metricas_umbral(y, p),
    }


def platt_cruzado(s, y, pliegues) -> np.ndarray:
    """Probabilidades a partir de puntajes no probabilísticos: Platt ajustado en los otros pliegues."""
    s, y, pliegues = np.asarray(s, float), np.asarray(y).astype(int), np.asarray(pliegues)
    out = np.full(len(s), np.nan)
    for k in np.unique(pliegues):
        en, fuera = pliegues == k, pliegues != k
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(s[fuera].reshape(-1, 1), y[fuera])
        out[en] = lr.predict_proba(s[en].reshape(-1, 1))[:, 1]
    return out


# ---------------------------------------------------------------------------- bootstrap
@dataclass
class CurvaPreordenada:
    """Puntajes ordenados una sola vez; AUC-ROC y AP ponderados en O(n) por réplica.

    Con pesos enteros w (veces que sale cada fila) reproduce `roc_auc_score` / `average_precision_score`
    con `sample_weight=w`, incluidos los empates de puntaje.
    """

    orden: np.ndarray
    inicios: np.ndarray  # índice (en el orden) donde empieza cada grupo de puntaje empatado
    y_ord: np.ndarray

    @classmethod
    def desde(cls, y, s) -> "CurvaPreordenada":
        s = np.asarray(s, float)
        orden = np.argsort(-s, kind="mergesort")
        s_ord = s[orden]
        inicios = np.r_[0, np.flatnonzero(np.diff(s_ord) != 0) + 1]
        return cls(orden, inicios, np.asarray(y, float)[orden])

    def _acumulados(self, w):
        w_ord = np.asarray(w, float)[self.orden]
        pos = np.add.reduceat(w_ord * self.y_ord, self.inicios)
        neg = np.add.reduceat(w_ord * (1 - self.y_ord), self.inicios)
        return pos, neg

    def auc_roc(self, w) -> float:
        pos, neg = self._acumulados(w)
        tp_antes = np.cumsum(pos) - pos
        return float((neg * (tp_antes + pos / 2)).sum() / (pos.sum() * neg.sum()))

    def auc_pr(self, w) -> float:
        pos, neg = self._acumulados(w)
        tp, fp = np.cumsum(pos), np.cumsum(neg)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        return float((pos * precision).sum() / pos.sum())


def estratos_paciente(y, grupos) -> tuple[np.ndarray, np.ndarray]:
    """Código de paciente por fila y estrato de cada paciente (0 todo negativo, 1 mixto, 2 todo positivo)."""
    codigos, inv = np.unique(np.asarray(grupos), return_inverse=True)
    n_pos = np.bincount(inv, weights=np.asarray(y, float), minlength=len(codigos))
    n = np.bincount(inv, minlength=len(codigos))
    estrato = np.where(n_pos == 0, 0, np.where(n_pos == n, 2, 1))
    return inv, estrato


def replicas_bootstrap(y, grupos, n_iter: int = N_BOOTSTRAP, semilla: int = 42):
    """Generador de pesos por fila: pacientes remuestreados con reemplazo dentro de su estrato."""
    inv, estrato = estratos_paciente(y, grupos)
    rng = np.random.default_rng(semilla)
    por_estrato = [np.flatnonzero(estrato == e) for e in np.unique(estrato)]
    n_pac = len(estrato)
    for _ in range(n_iter):
        cuenta = np.zeros(n_pac)
        for idx in por_estrato:
            cuenta += np.bincount(rng.choice(idx, size=len(idx), replace=True), minlength=n_pac)
        yield cuenta[inv]


def bootstrap_ic(y, puntajes: dict[str, np.ndarray], grupos, n_iter: int = N_BOOTSTRAP, semilla: int = 42,
                 alfa: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IC percentil (1 − alfa) de AUC-ROC y AUC-PR para cada modelo, con réplicas pareadas.

    Los puntajes con NaN (p. ej. KNN, evaluado en un subconjunto de pacientes) usan solo sus filas válidas,
    con los mismos pesos de la réplica. Devuelve (IC por modelo, réplicas largas modelo × iteración).
    """
    y = np.asarray(y)
    curvas, mascaras = {}, {}
    for m, s in puntajes.items():
        ok = ~np.isnan(s)
        mascaras[m] = ok
        curvas[m] = CurvaPreordenada.desde(y[ok], s[ok])
    filas = []
    for it, w in enumerate(replicas_bootstrap(y, grupos, n_iter, semilla)):
        for m, c in curvas.items():
            wm = w[mascaras[m]]
            filas.append((m, it, c.auc_roc(wm), c.auc_pr(wm)))
    rep = pd.DataFrame(filas, columns=["modelo", "iteracion", "auc_roc", "auc_pr"])
    q = rep.groupby("modelo")[["auc_roc", "auc_pr"]].quantile([alfa / 2, 1 - alfa / 2]).unstack()
    ic = pd.DataFrame({
        "auc_roc_ic95_inf": q[("auc_roc", alfa / 2)], "auc_roc_ic95_sup": q[("auc_roc", 1 - alfa / 2)],
        "auc_pr_ic95_inf": q[("auc_pr", alfa / 2)], "auc_pr_ic95_sup": q[("auc_pr", 1 - alfa / 2)],
    }).reset_index()
    return ic, rep


# ---------------------------------------------------------------------------- tablas
def evaluar_modelos(y, probas: dict[str, np.ndarray], grupos, n_iter: int = N_BOOTSTRAP,
                    semilla: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabla consolidada (ordenada por AUC-PR descendente) + réplicas bootstrap."""
    y = np.asarray(y).astype(int)
    filas = []
    for m, p in probas.items():
        ok = ~np.isnan(p)
        filas.append({"modelo": m, **metricas(y[ok], p[ok])})
    tabla = pd.DataFrame(filas)
    ic, rep = bootstrap_ic(y, probas, grupos, n_iter, semilla)
    tabla = tabla.merge(ic, on="modelo")
    cols = ["modelo", "n", "prevalencia", "auc_roc", "auc_roc_ic95_inf", "auc_roc_ic95_sup", "auc_pr",
            "auc_pr_ic95_inf", "auc_pr_ic95_sup", "auc_pr_sin_informacion", "brier", "brier_skill", "ece",
            "f1", "sensibilidad", "especificidad", "precision", "tasa_predichos_positivos"]
    return tabla[cols].sort_values("auc_pr", ascending=False).reset_index(drop=True), rep


def metricas_por_estrato(y, probas: dict[str, np.ndarray], estratos) -> pd.DataFrame:
    """Vector multimétrico (sin IC) por estrato (servicio × τ): ahí la prevalencia cambia mucho."""
    y, estratos = np.asarray(y).astype(int), np.asarray(estratos)
    filas = []
    for e in pd.unique(estratos):
        en = estratos == e
        for m, p in probas.items():
            ok = en & ~np.isnan(p)
            if y[ok].min() != y[ok].max():
                filas.append({"estrato": e, "modelo": m, **metricas(y[ok], p[ok])})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------- gráfico
def graficar_roc_pr(y, probas: dict[str, np.ndarray], ruta, etiquetas: dict[str, str] | None = None) -> None:
    """Curvas ROC y Precision-Recall de todos los modelos en una figura de dos paneles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.asarray(y).astype(int)
    etiquetas = etiquetas or {}
    orden = sorted(probas, key=lambda m: -average_precision_score(y[~np.isnan(probas[m])],
                                                                   probas[m][~np.isnan(probas[m])]))
    colores = dict(zip(orden, plt.cm.tab10.colors))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.8))
    for m in orden:
        ok = ~np.isnan(probas[m])
        yy, pp = y[ok], probas[m][ok]
        fpr, tpr, _ = roc_curve(yy, pp)
        pr, rc, _ = precision_recall_curve(yy, pp)
        nombre = m + etiquetas.get(m, "")
        a1.plot(fpr, tpr, lw=1.6, color=colores[m], label=f"{nombre} · AUC {roc_auc_score(yy, pp):.3f}")
        a2.plot(rc, pr, lw=1.6, color=colores[m], label=f"{nombre} · AP {average_precision_score(yy, pp):.3f}")
    a1.plot([0, 1], [0, 1], "k:", lw=0.8)
    a2.axhline(y.mean(), color="k", ls=":", lw=0.8)
    a2.text(0.01, y.mean() + 0.008, f"sin información = prevalencia {y.mean():.3f}", fontsize=8)
    a1.set(xlabel="Tasa de falsos positivos (1 − especificidad)", ylabel="Sensibilidad", title="Curvas ROC")
    a2.set(xlabel="Sensibilidad (recall)", ylabel="Precisión (VPP)", title="Curvas Precision-Recall", ylim=(0, 1.02))
    for a in (a1, a2):
        a.legend(fontsize=8, loc="lower right" if a is a1 else "upper right")
        a.grid(alpha=0.3)
    fig.suptitle("Evaluación fuera de pliegue (OOF, 5 pliegues agrupados por paciente)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
