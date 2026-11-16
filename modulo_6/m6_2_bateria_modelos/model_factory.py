"""Subfase 6.2 · Batería de modelos supervisados y espacios de búsqueda (`model_factory`).

Cada algoritmo se encadena **después** del preprocesador del Módulo 5 (`ColumnTransformer`
de la 5.4) dentro de un `Pipeline`. Así la imputación, el escalado, el one-hot y la SVD de
diagnósticos se reajustan dentro de cada pliegue de la validación cruzada agrupada (6.1) y
nunca ven el pliegue de validación.

Batería
-------
| Nombre | Modelo | Nota |
|---|---|---|
| `logreg_ridge` | `LogisticRegression` L2 (`l1_ratio=0`) | lineal regularizado (Ridge) |
| `logreg_lasso` | `LogisticRegression` L1 (`l1_ratio=1`, `solver="saga"`) | lineal con selección de variables (Lasso) |
| `knn` | `KNeighborsClassifier(algorithm="ball_tree")` | proximidad euclidiana sobre variables escaladas + SVD |
| `naive_bayes` | `GaussianNB` | probabilístico, supone normalidad por clase |
| `cart` | `DecisionTreeClassifier` | árbol único |
| `random_forest` | `RandomForestClassifier` | bagging de árboles |
| `lightgbm` | `LGBMClassifier` | boosting por hojas (leaf-wise) con histogramas |
| `xgboost` | `XGBClassifier(tree_method="hist")` | boosting con histogramas |
| `svm_lineal` | `LinearSVC` | margen máximo, kernel lineal |
| `svm_rbf` | `Nystroem(kernel="rbf") → LinearSVC` | kernel RBF aproximado |

Decisiones frente al prompt
---------------------------
- **Nombres de los pasos**: el prompt sugiere `make_pipeline(preprocessor, model)`. Se usa
  `Pipeline([("preprocesador", …), ("modelo", …)])`, que es lo mismo pero con nombres fijos.
  Con `make_pipeline` los nombres dependen de la clase (`lgbmclassifier`, `svc`…) y las
  claves de `PARAM_SPACES` cambiarían con cada modelo. Con nombres fijos, el espacio siempre
  se escribe `modelo__<parámetro>` y la 5.4 ya usaba esos nombres.
- **SVM**: un `SVC` con kernel exacto escala como O(n²) en memoria y O(n²–n³) en tiempo.
  Con 433 000 filas por pliegue no termina. Por eso, como pide el plan original (PDF, M6),
  el kernel RBF se **aproxima con Nyström** y se resuelve con `LinearSVC`. Las SVM no dan
  probabilidades: para AUC se usa `decision_function` y la calibración es tarea de M7.
- **KNN** usa `algorithm="ball_tree"`, también según el plan. Aun así, predecir es costoso
  con cientos de miles de filas: la búsqueda de hiperparámetros (6.3) debe submuestrear
  (ver `COSTO`).
- **Regularización en escala logarítmica**: `C`, `alpha`/`reg_*`, `gamma`, `learning_rate`,
  `var_smoothing`, `ccp_alpha` y `min_child_weight` se muestrean con `log=True`.
- **Semilla**: `random_state=42` en todo modelo o paso aleatorio (LightGBM y XGBoost incluidos).

`PARAM_SPACES` es una especificación neutral (tipo, límites, escala). Sirve para Optuna
(`sugerir_parametros(trial, espacio)`) y para `RandomizedSearchCV`
(`a_distribuciones_scipy(espacio)`).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

SEMILLA = 42


# --------------------------------------------------------------------------------------
# Especificación de hiperparámetros
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Real:
    low: float
    high: float
    log: bool = False


@dataclass(frozen=True)
class Entero:
    low: int
    high: int
    log: bool = False


@dataclass(frozen=True)
class Categorico:
    opciones: tuple


def _lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=SEMILLA, n_jobs=-1, verbose=-1, subsample_freq=1)


def _xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(random_state=SEMILLA, n_jobs=-1, tree_method="hist", eval_metric="logloss",
                         verbosity=0)


def _version_sklearn() -> tuple[int, int]:
    import sklearn
    mayor, menor = sklearn.__version__.split(".")[:2]
    return int(mayor), int("".join(ch for ch in menor if ch.isdigit()) or 0)


def _logreg(l1: bool):
    """Ridge (L2) o Lasso (L1). En scikit-learn ≥ 1.8 `penalty` está obsoleto: se usa `l1_ratio` (0 = L2, 1 = L1)."""
    comun = dict(max_iter=2000, random_state=SEMILLA)
    if _version_sklearn() >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, solver="saga", tol=1e-3, **comun) if l1 else \
            LogisticRegression(l1_ratio=0.0, solver="lbfgs", **comun)
    return LogisticRegression(penalty="l1", solver="saga", tol=1e-3, **comun) if l1 else \
        LogisticRegression(penalty="l2", solver="lbfgs", **comun)


# Constructor del estimador base de cada modelo (sin preprocesador).
MODELOS = {
    "logreg_ridge": lambda: _logreg(l1=False),
    "logreg_lasso": lambda: _logreg(l1=True),
    "knn": lambda: KNeighborsClassifier(algorithm="ball_tree", n_jobs=-1),
    "naive_bayes": lambda: GaussianNB(),
    "cart": lambda: DecisionTreeClassifier(random_state=SEMILLA),
    "random_forest": lambda: RandomForestClassifier(random_state=SEMILLA, n_jobs=-1),
    "lightgbm": _lgbm,
    "xgboost": _xgb,
    "svm_lineal": lambda: LinearSVC(random_state=SEMILLA, dual="auto", max_iter=5000),
    "svm_rbf": lambda: Pipeline([
        ("nystroem", Nystroem(kernel="rbf", random_state=SEMILLA, n_components=300)),
        ("svm", LinearSVC(random_state=SEMILLA, dual="auto", max_iter=5000)),
    ]),
}

PESO_CLASE = Categorico((None, "balanced"))

# Espacios de búsqueda. Claves = parámetros del Pipeline ("modelo__…").
PARAM_SPACES: dict[str, dict[str, Real | Entero | Categorico]] = {
    "logreg_ridge": {
        "modelo__C": Real(1e-4, 1e2, log=True),
        "modelo__class_weight": PESO_CLASE,
    },
    "logreg_lasso": {
        "modelo__C": Real(1e-4, 1e2, log=True),
        "modelo__class_weight": PESO_CLASE,
    },
    "knn": {
        "modelo__n_neighbors": Entero(5, 201, log=True),
        "modelo__weights": Categorico(("uniform", "distance")),
        "modelo__leaf_size": Entero(20, 80),
    },
    "naive_bayes": {
        "modelo__var_smoothing": Real(1e-12, 1e-1, log=True),
    },
    "cart": {
        "modelo__max_depth": Entero(3, 30),
        "modelo__min_samples_leaf": Entero(10, 5000, log=True),
        "modelo__ccp_alpha": Real(1e-7, 1e-2, log=True),
        "modelo__criterion": Categorico(("gini", "entropy")),
        "modelo__class_weight": PESO_CLASE,
    },
    "random_forest": {
        "modelo__n_estimators": Entero(100, 600, log=True),
        "modelo__max_depth": Entero(6, 40),
        "modelo__min_samples_leaf": Entero(5, 500, log=True),
        "modelo__max_features": Categorico(("sqrt", "log2", 0.3)),
        "modelo__max_samples": Real(0.2, 1.0),
        "modelo__class_weight": Categorico((None, "balanced", "balanced_subsample")),
    },
    "lightgbm": {
        "modelo__n_estimators": Entero(100, 2000, log=True),
        "modelo__learning_rate": Real(1e-3, 0.3, log=True),
        "modelo__num_leaves": Entero(8, 512, log=True),
        "modelo__min_child_samples": Entero(10, 1000, log=True),
        "modelo__subsample": Real(0.5, 1.0),
        "modelo__colsample_bytree": Real(0.3, 1.0),
        "modelo__reg_alpha": Real(1e-8, 10.0, log=True),
        "modelo__reg_lambda": Real(1e-8, 10.0, log=True),
    },
    "xgboost": {
        "modelo__n_estimators": Entero(100, 2000, log=True),
        "modelo__learning_rate": Real(1e-3, 0.3, log=True),
        "modelo__max_depth": Entero(3, 12),
        "modelo__min_child_weight": Real(1e-1, 100.0, log=True),
        "modelo__gamma": Real(1e-8, 5.0, log=True),
        "modelo__subsample": Real(0.5, 1.0),
        "modelo__colsample_bytree": Real(0.3, 1.0),
        "modelo__reg_alpha": Real(1e-8, 10.0, log=True),
        "modelo__reg_lambda": Real(1e-8, 10.0, log=True),
    },
    "svm_lineal": {
        "modelo__C": Real(1e-4, 1e2, log=True),
        "modelo__class_weight": PESO_CLASE,
    },
    "svm_rbf": {
        "modelo__nystroem__gamma": Real(1e-4, 1.0, log=True),
        "modelo__nystroem__n_components": Entero(100, 1000, log=True),
        "modelo__svm__C": Real(1e-4, 1e2, log=True),
        "modelo__svm__class_weight": PESO_CLASE,
    },
}

# Costo relativo con ~433 000 filas por pliegue. Lo usa la 6.3 para decidir si submuestrea.
COSTO = {
    "logreg_ridge": "bajo", "logreg_lasso": "medio", "knn": "alto (predicción)", "naive_bayes": "bajo",
    "cart": "bajo", "random_forest": "medio", "lightgbm": "medio", "xgboost": "medio",
    "svm_lineal": "medio", "svm_rbf": "alto",
}

# Qué devuelve el modelo para calcular AUC (las SVM no tienen predict_proba).
SALIDA_PUNTAJE = {n: ("decision_function" if n.startswith("svm") else "predict_proba") for n in MODELOS}


# --------------------------------------------------------------------------------------
# Fábrica
# --------------------------------------------------------------------------------------
def listar_modelos() -> list[str]:
    return list(MODELOS)


def construir_modelo(model_name: str):
    if model_name not in MODELOS:
        raise KeyError(f"Modelo desconocido '{model_name}'. Disponibles: {listar_modelos()}")
    return MODELOS[model_name]()


def get_pipeline_and_space(model_name: str, preprocessor: ColumnTransformer | None = None):
    """Devuelve `(Pipeline sin entrenar, espacio de búsqueda)` para `model_name`.

    `preprocessor` es el `ColumnTransformer` del Módulo 5 (se clona: el objeto que se
    recibe nunca se ajusta ni se modifica). Si no se pasa, se construye con el esquema
    por defecto de la 5.4.
    """
    if preprocessor is None:
        from modulo_5.m5_4_pipeline import construir_preprocesador
        preprocessor = construir_preprocesador()
    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError("preprocessor debe ser el ColumnTransformer del Módulo 5 (sin el estimador)")
    if hasattr(preprocessor, "transformers_"):
        raise ValueError("El preprocesador ya está ajustado: pasa la versión sin ajustar para no filtrar estadísticas")
    pipe = Pipeline([("preprocesador", clone(preprocessor)), ("modelo", construir_modelo(model_name))])
    return pipe, copy.deepcopy(PARAM_SPACES[model_name])


class ModelFactory:
    """Envoltorio orientado a objetos: guarda el preprocesador del M5 y entrega pipelines nuevos."""

    def __init__(self, preprocessor: ColumnTransformer | None = None):
        self.preprocessor = preprocessor

    def get_pipeline_and_space(self, model_name: str):
        return get_pipeline_and_space(model_name, self.preprocessor)

    def listar(self) -> list[str]:
        return listar_modelos()


# --------------------------------------------------------------------------------------
# Traducción del espacio a Optuna / scikit-learn
# --------------------------------------------------------------------------------------
def sugerir_parametros(trial, espacio: dict) -> dict:
    """Muestrea un punto del espacio con un `optuna.Trial` (o un objeto con la misma interfaz)."""
    params = {}
    for nombre, d in espacio.items():
        if isinstance(d, Real):
            params[nombre] = trial.suggest_float(nombre, d.low, d.high, log=d.log)
        elif isinstance(d, Entero):
            params[nombre] = trial.suggest_int(nombre, d.low, d.high, log=d.log)
        elif isinstance(d, Categorico):
            params[nombre] = trial.suggest_categorical(nombre, list(d.opciones))
        else:  # pragma: no cover
            raise TypeError(f"Tipo de distribución desconocido para {nombre}: {d!r}")
    return params


def a_distribuciones_scipy(espacio: dict) -> dict:
    """Convierte el espacio a distribuciones para `RandomizedSearchCV`."""
    out = {}
    for nombre, d in espacio.items():
        if isinstance(d, Real):
            out[nombre] = stats.loguniform(d.low, d.high) if d.log else stats.uniform(d.low, d.high - d.low)
        elif isinstance(d, Entero):
            if d.log:
                out[nombre] = _EnteroLog(d.low, d.high)
            else:
                out[nombre] = stats.randint(d.low, d.high + 1)
        else:
            out[nombre] = list(d.opciones)
    return out


class _EnteroLog:
    """Entero con distribución log-uniforme (interfaz `rvs` para RandomizedSearchCV)."""

    def __init__(self, low: int, high: int):
        self.low, self.high = low, high
        self._d = stats.loguniform(low, high + 1)

    def rvs(self, size=None, random_state=None):
        x = np.clip(np.floor(self._d.rvs(size=size, random_state=random_state)), self.low, self.high)
        return x.astype(int) if size is not None else int(x)


class MuestreadorAleatorio:
    """Imita la interfaz mínima de `optuna.Trial` con muestreo aleatorio reproducible (pruebas y humo)."""

    def __init__(self, semilla: int = SEMILLA):
        self.rng = np.random.default_rng(semilla)

    def suggest_float(self, nombre, low, high, log=False):
        return float(np.exp(self.rng.uniform(np.log(low), np.log(high)))) if log else float(self.rng.uniform(low, high))

    def suggest_int(self, nombre, low, high, log=False):
        if log:
            return int(min(high, max(low, np.floor(np.exp(self.rng.uniform(np.log(low), np.log(high + 1)))))))
        return int(self.rng.integers(low, high + 1))

    def suggest_categorical(self, nombre, opciones):
        return opciones[int(self.rng.integers(0, len(opciones)))]
