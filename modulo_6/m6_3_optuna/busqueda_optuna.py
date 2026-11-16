"""Subfase 6.3 · Optimización bayesiana de hiperparámetros con Optuna (TPE + MedianPruner).

Qué hace
--------
Para cada modelo de la batería (6.2), Optuna propone combinaciones de hiperparámetros con el
muestreador bayesiano **TPE** (Tree-structured Parzen Estimator). Cada combinación se evalúa
con el AUC-ROC fuera de muestra **promedio de los 5 pliegues** de la validación cruzada
agrupada por paciente (6.1). Un **MedianPruner** corta los ensayos que, tras los primeros
pliegues, van por debajo de la mediana de los ensayos anteriores en ese mismo punto.

Cómo evita Optuna sobreajustar la validación
--------------------------------------------
1. **Promedio sobre 5 pliegues aislados por paciente.** El valor que optimiza TPE no es el AUC
   de una sola partición: es la media de 5 particiones disjuntas en las que ningún `person_id`
   está a la vez en entrenamiento y validación. Una combinación que "acierta" por azar con los
   pacientes de un pliegue no gana si falla en los otros cuatro. Así se reduce la varianza del
   criterio, que es lo que explota el sobreajuste a la validación.
2. **Sin fuga de paciente.** Como los snapshots de una persona (varios episodios × τ = 0/2/6 h)
   nunca se reparten, el modelo no puede "reconocer" al paciente en validación. El AUC estimado
   corresponde a pacientes nuevos, no a filas casi idénticas ya vistas.
3. **Preprocesamiento dentro del pliegue.** Imputación, escalado, one-hot y SVD se ajustan solo
   con el entrenamiento de cada pliegue (5.4), así que la validación no aporta estadísticas.
4. **La prueba temporal no participa.** La búsqueda usa solo `es_entrenamiento & en_periodo_estable`.
   Elegir el mejor de 50 ensayos introduce un sesgo optimista inevitable en el AUC de CV (es el
   máximo de 50 estimaciones ruidosas): por eso el AUC definitivo se mide una sola vez en la
   prueba temporal (M7), que Optuna nunca vio. Se guarda también la desviación entre pliegues
   para ver cuán estable es cada ganador.
5. **Semillas fijas** (TPE `seed=42`, modelos `random_state=42`): la búsqueda es reproducible.

Caché de pliegues (equivalente exacto al Pipeline)
--------------------------------------------------
El preprocesador del M5 no tiene hiperparámetros en la búsqueda, así que ajustarlo en cada
ensayo repetiría el mismo cálculo 50 veces. `CachePliegues` lo ajusta **una vez por pliegue,
solo con el entrenamiento de ese pliegue**, y guarda las matrices transformadas. El resultado
es idéntico a correr el `Pipeline` completo (lo comprueba la prueba
`test_cache_equivale_al_pipeline`) y conserva la garantía anti-fuga.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from modulo_6.m6_1_cv_agrupada import verify_group_isolation
from modulo_6.m6_2_bateria_modelos import PARAM_SPACES, SALIDA_PUNTAJE, get_pipeline_and_space, sugerir_parametros

SEMILLA = 42

# Familias del prompt (7) → nombres de la batería (10).
FAMILIAS = {
    "Regresión logística (Ridge / Lasso)": ["logreg_ridge", "logreg_lasso"],
    "KNN": ["knn"],
    "Naive Bayes gaussiano": ["naive_bayes"],
    "Árbol de decisión (CART)": ["cart"],
    "Random Forest": ["random_forest"],
    "Gradient boosting (XGBoost / LightGBM)": ["xgboost", "lightgbm"],
    "SVM (lineal / RBF)": ["svm_lineal", "svm_rbf"],
}


def _puntaje(modelo, model_name: str, Z):
    if SALIDA_PUNTAJE[model_name] == "decision_function":
        return modelo.decision_function(Z)
    return modelo.predict_proba(Z)[:, 1]


def _auc_por_estrato(y, s, estratos) -> float:
    """AUC medio por estrato (servicio × τ). Los estratos con una sola clase se omiten."""
    if estratos is None:
        return float("nan")
    df = pd.DataFrame({"y": y, "s": s, "e": estratos})
    aucs = [roc_auc_score(d.y, d.s) for _, d in df.groupby("e") if d.y.nunique() == 2]
    return float(np.mean(aucs)) if aucs else float("nan")


@dataclass
class _Pliegue:
    Z_tr: np.ndarray
    y_tr: np.ndarray
    Z_va: np.ndarray
    y_va: np.ndarray
    e_va: np.ndarray | None


class CachePliegues:
    """Preprocesador del M5 ajustado una vez por pliegue (solo con su entrenamiento)."""

    def __init__(self, preprocesador, X: pd.DataFrame, y, cv, groups=None, estratos=None, dtype=np.float32):
        y = np.asarray(y)
        est = None if estratos is None else np.asarray(estratos)
        self.pliegues: list[_Pliegue] = []
        for i_tr, i_va in cv.split(X, y, groups=groups):
            pre = clone(preprocesador).fit(X.iloc[i_tr], y[i_tr])
            self.pliegues.append(_Pliegue(
                pre.transform(X.iloc[i_tr]).astype(dtype), y[i_tr],
                pre.transform(X.iloc[i_va]).astype(dtype), y[i_va],
                None if est is None else est[i_va],
            ))

    def __len__(self) -> int:
        return len(self.pliegues)


def objective(trial: optuna.Trial, model_name: str, X, y, groups, cv, cache: CachePliegues | None = None,
              preprocesador=None, estratos=None) -> float:
    """Función objetivo de Optuna: AUC-ROC medio fuera de muestra en los pliegues agrupados por paciente.

    - Pide los hiperparámetros de `model_name` con `trial.suggest_*` (espacio `PARAM_SPACES` de la 6.2).
    - Entrena y evalúa pliegue a pliegue. Después de cada pliegue reporta la media acumulada al
      podador (`trial.report`). Si `MedianPruner` decide que va por debajo de la mediana, corta el
      ensayo (`optuna.TrialPruned`).
    - Devuelve la media del AUC-ROC de todos los pliegues.

    Con `cache=None` se ajusta el `Pipeline` completo en cada pliegue (camino directo). Con
    `cache` se reutiliza el preprocesador ya ajustado por pliegue, y el resultado es idéntico.
    """
    params = sugerir_parametros(trial, PARAM_SPACES[model_name])
    pipe, _ = get_pipeline_and_space(model_name, preprocesador)
    pipe.set_params(**params)
    modelo_base = pipe.named_steps["modelo"]

    aucs, aucs_estrato = [], []
    if cache is not None:
        iterador = ((p.Z_tr, p.y_tr, p.Z_va, p.y_va, p.e_va, True) for p in cache.pliegues)
    else:
        y_arr = np.asarray(y)
        estr = None if estratos is None else np.asarray(estratos)
        iterador = ((X.iloc[a], y_arr[a], X.iloc[b], y_arr[b], None if estr is None else estr[b], False)
                    for a, b in cv.split(X, y_arr, groups=groups))

    for paso, (A, ya, B, yb, eb, desde_cache) in enumerate(iterador):
        modelo = clone(modelo_base) if desde_cache else clone(pipe)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo.fit(A, ya)
            s = _puntaje(modelo, model_name, B)
        aucs.append(roc_auc_score(yb, s))
        aucs_estrato.append(_auc_por_estrato(yb, s, eb))
        trial.report(float(np.mean(aucs)), step=paso)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("auc_pliegues", [float(a) for a in aucs])
    trial.set_user_attr("auc_std", float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0)
    trial.set_user_attr("auc_medio_servicio_tau", float(np.nanmean(aucs_estrato)))
    return float(np.mean(aucs))


def crear_estudio(model_name: str, storage: str | None = None, semilla: int = SEMILLA) -> optuna.Study:
    """Estudio TPE + MedianPruner (maximizar AUC). Con `storage` se puede reanudar si se interrumpe."""
    return optuna.create_study(
        study_name=f"m6_3_{model_name}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=semilla, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage=storage,
        load_if_exists=storage is not None,
    )


def optimizar_modelo(model_name: str, X, y, groups, cv, n_trials: int = 50, timeout: float | None = None,
                     cache: CachePliegues | None = None, preprocesador=None, estratos=None,
                     storage: str | None = None, verificar_aislamiento: bool = True) -> optuna.Study:
    """Corre (o reanuda) el estudio de `model_name` hasta completar `n_trials` ensayos o agotar `timeout`."""
    if verificar_aislamiento:
        verify_group_isolation(X, y, groups, cv)  # ValueError si un paciente cruza pliegues
    estudio = crear_estudio(model_name, storage)
    hechos = len([t for t in estudio.trials if t.state in (optuna.trial.TrialState.COMPLETE,
                                                           optuna.trial.TrialState.PRUNED)])
    faltan = max(0, n_trials - hechos)
    if faltan:
        estudio.optimize(
            lambda t: objective(t, model_name, X, y, groups, cv, cache=cache, preprocesador=preprocesador,
                                estratos=estratos),
            n_trials=faltan, timeout=timeout, gc_after_trial=True,
        )
    return estudio


def resumen_estudio(model_name: str, estudio: optuna.Study, segundos: float | None = None) -> dict:
    """Fila de la tabla comparativa para un modelo."""
    estados = pd.Series([t.state.name for t in estudio.trials]).value_counts()
    completos = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completos:
        return {"modelo": model_name, "ensayos_completos": 0}
    mejor = estudio.best_trial
    return {
        "modelo": model_name,
        "familia": next(f for f, ms in FAMILIAS.items() if model_name in ms),
        "mejor_auc_cv": mejor.value,
        "auc_std_pliegues": mejor.user_attrs.get("auc_std"),
        "auc_medio_servicio_tau": mejor.user_attrs.get("auc_medio_servicio_tau"),
        "ensayos_completos": int(estados.get("COMPLETE", 0)),
        "ensayos_podados": int(estados.get("PRUNED", 0)),
        "ensayos_fallidos": int(estados.get("FAIL", 0)),
        "mejor_ensayo": mejor.number,
        "segundos": None if segundos is None else round(segundos, 1),
        "hiperparametros": mejor.params,
    }


def guardar_resultados(filas: list[dict], ruta_json: Path, ruta_csv: Path, meta: dict | None = None) -> pd.DataFrame:
    """Escribe `best_hyperparameters.json` y la tabla resumen (CSV). Devuelve la tabla."""
    filas = [f for f in filas if f.get("ensayos_completos")]
    contenido = {
        "metrica": "AUC-ROC medio fuera de muestra en StratifiedGroupKFold(5) por person_id",
        **(meta or {}),
        "modelos": {f["modelo"]: {"mejor_auc_cv": f["mejor_auc_cv"], "auc_std_pliegues": f["auc_std_pliegues"],
                                  "auc_medio_servicio_tau": f["auc_medio_servicio_tau"],
                                  "hiperparametros": f["hiperparametros"]} for f in filas},
    }
    Path(ruta_json).write_text(json.dumps(contenido, ensure_ascii=False, indent=2, default=_json_default),
                               encoding="utf-8")
    tabla = pd.DataFrame(filas).sort_values("mejor_auc_cv", ascending=False).reset_index(drop=True)
    tabla.assign(hiperparametros=tabla.hiperparametros.map(lambda d: json.dumps(d, default=_json_default))) \
        .to_csv(ruta_csv, index=False)
    return tabla


def tabla_por_familia(tabla: pd.DataFrame) -> pd.DataFrame:
    """Consolidado de las 7 familias del prompt: el mejor modelo de cada familia."""
    t = tabla.sort_values("mejor_auc_cv", ascending=False).groupby("familia", as_index=False).first()
    return t.sort_values("mejor_auc_cv", ascending=False).reset_index(drop=True)[
        ["familia", "modelo", "mejor_auc_cv", "auc_std_pliegues", "auc_medio_servicio_tau",
         "ensayos_completos", "ensayos_podados"]]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def cronometro():
    t0 = time.time()
    return lambda: time.time() - t0
