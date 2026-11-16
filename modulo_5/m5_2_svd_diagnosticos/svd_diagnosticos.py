"""Subfase 5.2 · Reducción de dimensionalidad de códigos clínicos (TruncatedSVD).

Contexto clínico
----------------
`condition_occurrence` tiene miles de conceptos (≈ 3 800 de dominio Condition en
`data/raw/concept.parquet`). Un one-hot directo daría miles de columnas casi vacías:
el modelo memorizaría códigos raros y sobreajustaría. Aquí cada snapshot se describe
por **qué diagnósticos tenía la persona antes de τ** y esa matriz dispersa se comprime
con `TruncatedSVD` a ~50 componentes. Cada componente resume un patrón de comorbilidad
que tiende a aparecer junto (p. ej. HTA + diabetes + ERC, o el bloque obstétrico).

Decisiones impuestas por los datos del HUN (CLAUDE.md, docs/M1_hallazgos_datos.md)
-------------------------------------------------------------------------------
- **Una fila = un snapshot** (paciente en τ), no una `visit_occurrence_id`: el
  vínculo evento → visita está roto (19 % de los diagnósticos apunta a otra persona),
  y el snapshot es la unidad del modelo de paciente.
- **Cruce por `person_id` + `t < τ`** (regla 5 de CLAUDE.md, estricta).
- **Procedimientos**: `procedure_date` está vacío en el HUN, así que no se pueden
  ubicar en el tiempo. El código los acepta (si algún día tienen fecha entran solos),
  pero hoy todas sus filas se descartan por no tener `t`.
- Concepto 0 ("No matching concept", 37 % de los dx) se excluye: no es un diagnóstico.

Garantía anti-fuga
------------------
1. Construir la matriz es una **regla fija** (presencia de un código antes de τ), sin
   nada aprendido: se puede aplicar igual a entrenamiento y prueba.
2. Todo lo aprendido vive en `ReductorSVDClinico.fit`: qué columnas sobreviven
   (`min_df`), los pesos IDF y los ejes de la SVD. Se ajusta solo con entrenamiento
   (o dentro de cada fold si va en un `Pipeline`). Un código que solo aparece en
   prueba no tiene columna útil y se ignora.
3. No se usa `PCA` denso: la matriz nunca se densifica (n ≈ 766 k × miles).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils.validation import check_is_fitted

try:  # scikit-learn >= 1.6
    from sklearn.utils.validation import validate_data as _validate_data
except ImportError:  # scikit-learn 1.3 – 1.5
    def _validate_data(est, X, reset=True, **kw):
        return est._validate_data(X, reset=reset, **kw)


# --------------------------------------------------------------------------- eventos
def _fin_del_dia(s: pd.Series) -> pd.Series:
    return s.dt.normalize() + pd.Timedelta(days=1) - pd.Timedelta(1, "ns")


def preparar_eventos(
    tabla: pd.DataFrame,
    col_concepto: str,
    col_datetime: str | None,
    col_fecha: str | None = None,
    prefijo: str = "dx_",
    conceptos_excluidos: tuple = (0,),
    personas: pd.Series | np.ndarray | None = None,
    medianoche_como_fecha: bool = True,
) -> pd.DataFrame:
    """Reduce una tabla OMOP de eventos a la **primera aparición** de cada código por persona.

    Para la comorbilidad basta saber si la persona ya tenía el código antes de τ, y eso
    equivale a `primera_aparicion < τ`. Quedarse con la primera aparición reduce millones
    de filas (n_dx_historia llega a 36 000 en una persona) a unos pocos pares
    persona-código, sin perder información para una matriz binaria.

    Fecha efectiva (igual que en la subfase 5.1): si el evento solo tiene fecha, o una
    hora 00:00:00, se asume conocido al **final del día**. Es conservador: nunca adelanta
    un diagnóstico.

    Args:
        tabla: `condition_occurrence` o `procedure_occurrence` (u otra con `person_id`).
        col_concepto: p. ej. `condition_concept_id` / `procedure_concept_id`.
        col_datetime: columna con fecha y hora (None si no existe).
        col_fecha: columna solo con fecha, de respaldo.
        prefijo: se antepone al código (`dx_`, `px_`) para saber de qué tabla viene.
        conceptos_excluidos: códigos que no son clínicos (0 = sin mapeo OMOP).
        personas: si se da, solo se conservan esas personas (las del maestro).

    Returns:
        DataFrame `person_id, codigo, t_efectiva` con un par persona-código por fila.
    """
    if "person_id" not in tabla or col_concepto not in tabla:
        raise KeyError(f"la tabla necesita 'person_id' y '{col_concepto}'")
    t = tabla
    if personas is not None:
        t = t[t["person_id"].isin(pd.unique(np.asarray(personas)))]

    if col_datetime and col_datetime in t:
        dt = pd.to_datetime(t[col_datetime], errors="coerce").astype("datetime64[ns]")
        if medianoche_como_fecha:
            dt = dt.where(~(dt.notna() & (dt == dt.dt.normalize())), _fin_del_dia(dt))
    else:
        dt = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]")
    if col_fecha and col_fecha in t:
        dt = dt.fillna(_fin_del_dia(pd.to_datetime(t[col_fecha], errors="coerce").astype("datetime64[ns]")))

    ev = pd.DataFrame({"person_id": t["person_id"].to_numpy(), "concepto": t[col_concepto].to_numpy(), "t_efectiva": dt.to_numpy()})
    ev = ev[ev["person_id"].notna() & ev["concepto"].notna() & ev["t_efectiva"].notna()]
    if conceptos_excluidos:
        ev = ev[~ev["concepto"].isin(list(conceptos_excluidos))]
    ev = ev.groupby(["person_id", "concepto"], as_index=False, sort=False)["t_efectiva"].min()
    concepto = ev["concepto"].astype("int64") if pd.api.types.is_numeric_dtype(ev["concepto"]) else ev["concepto"]
    ev["codigo"] = prefijo + concepto.astype(str)
    return ev[["person_id", "codigo", "t_efectiva"]]


def construir_matriz_dispersa(
    master: pd.DataFrame,
    eventos: pd.DataFrame,
    vocabulario: np.ndarray | list | None = None,
    col_tiempo: str = "snapshot_time",
    bloque: int = 100_000,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Matriz binaria `snapshots × códigos`: 1 si la persona tenía el código antes de τ.

    Regla fija (no aprende nada): `X[i, j] = 1` si la persona del snapshot `i` tiene
    `primera_aparicion(código j) < τ_i`. Las filas siguen el orden de `master`.

    El vocabulario por defecto son todos los códigos presentes en `eventos`, ordenados.
    Incluir códigos que solo aparecen en prueba no es fuga: `ReductorSVDClinico.fit`
    descarta las columnas sin soporte en entrenamiento.

    Returns:
        (csr_matrix float32 de forma n_snapshots × n_códigos, vocabulario)
    """
    vocab = np.asarray(sorted(eventos["codigo"].unique()) if vocabulario is None else vocabulario, dtype=object)
    tau = pd.to_datetime(master[col_tiempo]).astype("datetime64[ns]")
    if tau.isna().any():
        raise ValueError(f"hay filas sin '{col_tiempo}': no se puede fijar τ")

    izq = pd.DataFrame({"_fila": np.arange(len(master)), "person_id": master["person_id"].to_numpy(), "_tau": tau.to_numpy()})
    # El código se convierte a su índice de columna antes del cruce: se cruzan enteros, no textos.
    ev = pd.DataFrame(
        {
            "person_id": eventos["person_id"].astype(izq["person_id"].dtype).to_numpy(),
            "t_efectiva": pd.to_datetime(eventos["t_efectiva"]).astype("datetime64[ns]").to_numpy(),
            "_col": pd.Categorical(eventos["codigo"], categories=vocab).codes.astype(np.int32),
        }
    )
    ev = ev[ev["_col"] >= 0]

    # Por bloques de filas del maestro para no materializar todo el cruce persona × códigos a la vez.
    filas, cols = [], []
    for ini in range(0, len(izq), bloque):
        pares = izq.iloc[ini:ini + bloque].merge(ev, on="person_id", how="inner")
        pares = pares[pares["t_efectiva"] < pares["_tau"]]  # estricto: t < τ
        filas.append(pares["_fila"].to_numpy())
        cols.append(pares["_col"].to_numpy())
    filas = np.concatenate(filas) if filas else np.empty(0, np.int64)
    cols = np.concatenate(cols) if cols else np.empty(0, np.int32)
    X = sp.csr_matrix((np.ones(len(filas), dtype=np.float32), (filas, cols)), shape=(len(master), len(vocab)))
    X.sum_duplicates()
    X.data[:] = 1.0  # binaria
    return X, vocab


# --------------------------------------------------------------------------- transformador
class ReductorSVDClinico(TransformerMixin, BaseEstimator):
    """Comprime una matriz dispersa de códigos clínicos con `TruncatedSVD`.

    Compatible con la API de scikit-learn: `fit` aprende solo de los datos que recibe
    (entrenamiento, o el fold de entrenamiento dentro de un `Pipeline`) y `transform`
    aplica lo aprendido sin modificarlo.

    Pasos aprendidos en `fit`:
        1. Filtro de soporte: se conservan los códigos presentes en al menos `min_df`
           filas. Un código vivo en 2 pacientes solo añade ruido y riesgo de memorizar.
        2. Ponderación TF-IDF (opcional): baja el peso de los códigos casi universales
           (que no distinguen pacientes) y sube el de los específicos. Es la versión
           clínica del análisis semántico latente (LSA).
        3. `TruncatedSVD`: proyecta a `n_components` ejes de comorbilidad. Trabaja
           directamente sobre la matriz dispersa, sin densificarla.

    Parameters:
        n_components: número de componentes (se recorta si hay menos códigos útiles).
        min_df: mínimo de filas de entrenamiento con el código para conservarlo.
        ponderacion: "tfidf" o None (usa la matriz tal cual).
        algorithm, n_iter, random_state: se pasan a `TruncatedSVD`.
        prefijo_salida: prefijo de los nombres de columna de salida.

    Attributes:
        columnas_: índices de las columnas de entrada que sobrevivieron a `min_df`.
        n_components_: componentes realmente usados.
        explained_variance_ratio_: varianza explicada por componente.
        varianza_acumulada_: suma acumulada de lo anterior.
    """

    def __init__(
        self,
        n_components: int = 50,
        min_df: int = 5,
        ponderacion: str | None = "tfidf",
        algorithm: str = "randomized",
        n_iter: int = 7,
        random_state: int | None = 42,
        prefijo_salida: str = "svd_dx",
    ):
        self.n_components = n_components
        self.min_df = min_df
        self.ponderacion = ponderacion
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.random_state = random_state
        self.prefijo_salida = prefijo_salida

    def fit(self, X, y=None):
        X = _validate_data(self, X, accept_sparse="csr", dtype=[np.float64, np.float32], reset=True)
        if self.ponderacion not in ("tfidf", None):
            raise ValueError("ponderacion debe ser 'tfidf' o None")
        # Filas con valor ≠ 0 por código. Sirve para matrices densas, `csr_matrix` y `csr_array`
        # (los `csr_array` de SciPy no tienen `.getnnz`).
        soporte = np.asarray((X != 0).sum(axis=0)).ravel()
        self.columnas_ = np.flatnonzero(soporte >= self.min_df)
        if self.columnas_.size == 0:
            raise ValueError(f"ningún código aparece en al menos min_df={self.min_df} filas de entrenamiento")
        Xs = X[:, self.columnas_]

        if self.ponderacion == "tfidf":
            self.tfidf_ = TfidfTransformer(norm="l2", smooth_idf=True).fit(Xs)
            Xs = self.tfidf_.transform(Xs)

        tope = Xs.shape[1] if self.algorithm == "randomized" else Xs.shape[1] - 1
        k = int(min(self.n_components, max(tope, 1)))
        if k < self.n_components:
            warnings.warn(f"solo hay {Xs.shape[1]} códigos útiles: se usan {k} componentes en vez de {self.n_components}")
        self.svd_ = TruncatedSVD(n_components=k, algorithm=self.algorithm, n_iter=self.n_iter,
                                 random_state=self.random_state).fit(Xs)
        self.n_components_ = k
        self.explained_variance_ratio_ = self.svd_.explained_variance_ratio_
        self.varianza_acumulada_ = np.cumsum(self.explained_variance_ratio_)
        return self

    def transform(self, X):
        check_is_fitted(self, "svd_")
        X = _validate_data(self, X, accept_sparse="csr", dtype=[np.float64, np.float32], reset=False)
        Xs = X[:, self.columnas_]
        if self.ponderacion == "tfidf":
            Xs = self.tfidf_.transform(Xs)
        return self.svd_.transform(Xs)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "svd_")
        return np.asarray([f"{self.prefijo_salida}_{i:02d}" for i in range(self.n_components_)], dtype=object)

    # ------------------------------------------------------------ tags de scikit-learn
    # El reductor está pensado para matrices dispersas: se declara para que scikit-learn
    # (y `check_estimator`) lo sepa. `__sklearn_tags__` es la API de scikit-learn ≥ 1.6;
    # `_more_tags` la de versiones anteriores.
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def _more_tags(self):
        return {"X_types": ["2darray", "sparse"]}

    # ------------------------------------------------------------ interpretación
    def tabla_varianza(self) -> pd.DataFrame:
        """Varianza explicada por componente y acumulada."""
        check_is_fitted(self, "svd_")
        return pd.DataFrame(
            {"componente": self.get_feature_names_out(), "varianza": self.explained_variance_ratio_,
             "varianza_acumulada": self.varianza_acumulada_}
        )

    def imprimir_varianza(self, cada: int = 5) -> None:
        """Imprime la varianza explicada acumulada (cada `cada` componentes y la final)."""
        tv = self.tabla_varianza()
        print(f"Varianza explicada acumulada · {self.n_components_} componentes · {self.columnas_.size} códigos útiles")
        marcas = sorted(set(range(cada - 1, len(tv), cada)) | {len(tv) - 1})
        for i in marcas:
            print(f"  {i + 1:>3} comp.  {100 * tv.varianza_acumulada.iloc[i]:6.2f} %")

    def cargas(self, vocabulario, nombres: dict | None = None, top: int = 10) -> pd.DataFrame:
        """Códigos con más peso en cada componente (para nombrar los patrones de comorbilidad).

        Args:
            vocabulario: vocabulario de `construir_matriz_dispersa` (mismo orden de columnas).
            nombres: mapa código -> nombre clínico (p. ej. desde `concept_name`).
        """
        check_is_fitted(self, "svd_")
        vocab = np.asarray(vocabulario, dtype=object)[self.columnas_]
        nombres = nombres or {}
        filas = []
        for k, comp in enumerate(self.svd_.components_):
            for r, j in enumerate(np.argsort(-np.abs(comp))[:top], start=1):
                filas.append((self.get_feature_names_out()[k], r, vocab[j], nombres.get(vocab[j], ""), comp[j]))
        return pd.DataFrame(filas, columns=["componente", "rango", "codigo", "nombre", "carga"])
