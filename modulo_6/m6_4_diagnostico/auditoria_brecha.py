"""Auditoría de la brecha de sobreajuste (6.4).

1. **KNN y el AUC de entrenamiento "por resustitución".** Con `weights="distance"`, al predecir
   sobre su propio entrenamiento cada punto se encuentra a sí mismo a distancia 0 y recibe peso
   infinito: el modelo "devuelve" su propia etiqueta y el AUC de entrenamiento sale ≈ 1. Eso no
   mide memorización útil sino un artefacto del procedimiento. La buena práctica en modelos
   basados en instancias es el **AUC de entrenamiento leave-one-out**: cada punto se predice con
   sus vecinos **excluyéndose a sí mismo** (`kneighbors()` sin `X`). Esa es la cifra comparable con
   la validación.
2. **Random Forest y la regularización.** Los árboles profundos sí se ajustan de más a su
   entrenamiento (brecha real). Se explora una ruta de regularización (más profundidad limitada,
   hojas más grandes) y se elige la configuración con **mayor AUC de validación entre las que
   cumplen brecha ≤ 0,05**. La exploración usa los mismos pliegues aislados por paciente; la
   prueba temporal no participa.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from modulo_6.m6_4_diagnostico.diagnostico import CacheDiagnostico, puntaje


def proba_knn_sin_si_mismo(knn, n_vecinos: int | None = None) -> np.ndarray:
    """P(y=1) de cada punto de entrenamiento usando sus k vecinos **sin incluirse a sí mismo**.

    Reproduce la ponderación de scikit-learn (`uniform` o `distance`; con distancia 0 frente a
    otro punto duplicado, ese vecino recibe todo el peso, igual que en scikit-learn).
    """
    k = n_vecinos or knn.n_neighbors
    dist, ind = knn.kneighbors(n_neighbors=k)  # X=None → excluye al propio punto
    y_fit = np.asarray(knn._y)
    clase_pos = int(np.flatnonzero(knn.classes_ == 1)[0]) if 1 in knn.classes_ else len(knn.classes_) - 1
    es_pos = (y_fit[ind] == clase_pos).astype(float)
    if knn.weights == "distance":
        with np.errstate(divide="ignore"):
            w = 1.0 / dist
        ceros = np.isinf(w)
        filas_cero = ceros.any(axis=1)
        w[filas_cero] = ceros[filas_cero].astype(float)
    else:
        w = np.ones_like(dist)
    return (w * es_pos).sum(axis=1) / w.sum(axis=1)


def proba_knn_sin_mismo_paciente(knn, grupos, n_vecinos: int | None = None,
                                 extra_max: int = 500) -> tuple[np.ndarray, int]:
    """P(y=1) de cada punto de entrenamiento con sus k vecinos **de otros pacientes** (leave-one-patient-out).

    Un paciente aporta varias filas casi idénticas (snapshots τ = 0/2/6 h, episodios repetidos) con la
    misma etiqueta. Excluir solo la propia fila deja a sus "gemelos" como vecinos más cercanos, y el AUC
    de entrenamiento sigue inflado. Esto replica la condición de la validación, donde ningún paciente
    del pliegue de validación está en entrenamiento. Devuelve (probabilidades, nº de filas con menos de
    k vecinos válidos, que solo ocurre si un paciente tiene más de `extra_max` + 1 filas).
    """
    k = n_vecinos or knn.n_neighbors
    grupos = np.asarray(grupos)
    _, cnt = np.unique(grupos, return_counts=True)
    kk = int(min(k + min(cnt.max() - 1, extra_max), len(grupos) - 1))
    dist, ind = knn.kneighbors(n_neighbors=kk)  # X=None → excluye la propia fila
    validos = grupos[ind] != grupos[:, None]
    usar = validos & (np.cumsum(validos, axis=1) <= k)
    y_fit = np.asarray(knn._y)
    clase_pos = int(np.flatnonzero(knn.classes_ == 1)[0]) if 1 in knn.classes_ else len(knn.classes_) - 1
    es_pos = (y_fit[ind] == clase_pos).astype(float)
    if knn.weights == "distance":
        with np.errstate(divide="ignore"):
            w = np.where(usar, 1.0 / dist, 0.0)
        ceros = usar & (dist == 0)
        filas_cero = ceros.any(axis=1)
        w[filas_cero] = ceros[filas_cero].astype(float)
    else:
        w = usar.astype(float)
    return (w * es_pos).sum(axis=1) / w.sum(axis=1), int((usar.sum(axis=1) < k).sum())


def auditar_knn_por_paciente(pipe_knn, cache: CacheDiagnostico) -> pd.DataFrame:
    """Por pliegue: AUC de entrenamiento leave-one-patient-out (cifra comparable con la validación agrupada)."""
    filas = []
    for k, p in enumerate(cache.pliegues, 1):
        m = clone(pipe_knn.named_steps["modelo"]).fit(p.Z_tr, p.y_tr)
        proba, n_incompletas = proba_knn_sin_mismo_paciente(m, p.g_tr)
        filas.append({"pliegue": k, "train_auc_lopo": roc_auc_score(p.y_tr, proba),
                      "filas_con_menos_de_k_vecinos": n_incompletas})
    return pd.DataFrame(filas)


def auditar_knn(pipe_knn, cache: CacheDiagnostico, resustitucion_previa: pd.DataFrame | None = None) -> pd.DataFrame:
    """Por pliegue: AUC train por resustitución (con el propio punto), AUC train leave-one-out y AUC val.

    `resustitucion_previa` (columnas `pliegue`, `train_auc`, `val_auc` de la corrida de la 6.4 con los
    mismos pliegues) evita recalcular las dos predicciones más caras; solo se calcula el leave-one-out.
    """
    filas = []
    for k, p in enumerate(cache.pliegues, 1):
        m = clone(pipe_knn.named_steps["modelo"])
        m.fit(p.Z_tr, p.y_tr)
        if resustitucion_previa is not None:
            prev = resustitucion_previa.set_index("pliegue").loc[k]
            tr_res, va = float(prev.train_auc), float(prev.val_auc)
        else:
            tr_res = roc_auc_score(p.y_tr, m.predict_proba(p.Z_tr)[:, 1])
            va = roc_auc_score(p.y_va, m.predict_proba(p.Z_va)[:, 1])
        filas.append({
            "pliegue": k,
            "train_auc_resustitucion": tr_res,
            "train_auc_loo": roc_auc_score(p.y_tr, proba_knn_sin_si_mismo(m)),
            "val_auc": va,
        })
    df = pd.DataFrame(filas)
    df["brecha_resustitucion"] = df.train_auc_resustitucion - df.val_auc
    df["brecha_loo"] = df.train_auc_loo - df.val_auc
    return df


def ruta_regularizacion(nombre: str, pipe, cache: CacheDiagnostico, configuraciones: list[dict]) -> pd.DataFrame:
    """AUC train/val medios (5 pliegues) para cada configuración de hiperparámetros del modelo."""
    filas = []
    for i, cfg in enumerate(configuraciones):
        base = clone(pipe.named_steps["modelo"]).set_params(**cfg)
        tr, va = [], []
        for p in cache.pliegues:
            m = clone(base)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(p.Z_tr, p.y_tr)
                tr.append(roc_auc_score(p.y_tr, puntaje(m, nombre, p.Z_tr)))
                va.append(roc_auc_score(p.y_va, puntaje(m, nombre, p.Z_va)))
        filas.append({"config": i, **cfg, "train_auc": float(np.mean(tr)), "val_auc": float(np.mean(va)),
                      "val_auc_std": float(np.std(va, ddof=1)), "brecha": float(np.mean(tr) - np.mean(va))})
    return pd.DataFrame(filas)


def elegir_con_brecha(ruta: pd.DataFrame, umbral: float = 0.05) -> pd.Series:
    """Mayor AUC de validación entre las configuraciones con brecha ≤ umbral."""
    ok = ruta[ruta.brecha <= umbral]
    if ok.empty:
        raise ValueError(f"Ninguna configuración cumple brecha ≤ {umbral}: ampliar la ruta de regularización")
    return ok.sort_values("val_auc", ascending=False).iloc[0]
