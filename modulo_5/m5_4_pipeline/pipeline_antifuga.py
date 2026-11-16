"""Subfase 5.4 · Ensamblaje estricto anti-fuga (ColumnTransformer + Pipeline).

Integra en un solo objeto de scikit-learn todo el preprocesamiento del modelo de
paciente (`y_target_los_largo`). Así la imputación, el escalado, la codificación y la
reducción SVD se ajustan (`fit`) **solo** con las filas de entrenamiento que reciba el
`Pipeline`, ya sea todo el entrenamiento o el fold de entrenamiento dentro de la
validación cruzada agrupada (`StratifiedGroupKFold` por `person_id`, M6).

Ramas del `ColumnTransformer`
-----------------------------
| Rama | Columnas | Pasos |
|---|---|---|
| `num` | continuas (edad, horas, ocupación t − 1, cíclicas…) | `SimpleImputer(median)` → `StandardScaler` |
| `num_log` | conteos muy asimétricos (dx, visitas, mediciones, llegadas) | `SimpleImputer(median)` → `log1p` → `StandardScaler` |
| `cat` | categóricas y banderas 0/1 | a texto → `SimpleImputer(constant, "Desconocido")` → `OneHotEncoder(drop="first", handle_unknown="ignore")` |
| `clinica` | `dx_codigos` (texto con los códigos que la persona tenía antes de τ) | `CountVectorizer(binary)` → `ReductorSVDClinico` (5.2: `min_df` + TF-IDF + `TruncatedSVD(n_components=50)`) |

`num` y `num_log` forman juntas la "rama numérica" del prompt. Se separan porque el
EDA (M2, sección 8) mostró conteos con asimetría de 8 a 23 y muchos ceros: sin `log1p`,
un modelo lineal queda dominado por unos pocos pacientes extremos. `log1p` es una
regla fija (no aprende nada).

La rama clínica recibe los códigos como texto (`"dx_201826 dx_4329847 …"`) y no una
matriz precalculada. Así el mismo `Pipeline` sirve en M6 (validación cruzada) y en
M8 (API, paciente nuevo): basta con mandar los códigos previos a τ.

Garantía anti-fuga
------------------
- Todo lo que se aprende está **dentro** del `Pipeline` y por lo tanto solo ve el
  `X` de `fit`: medianas (`SimpleImputer.statistics_`), medias y desviaciones
  (`StandardScaler.mean_/scale_`), categorías (`OneHotEncoder.categories_`),
  vocabulario de códigos (`CountVectorizer.vocabulary_`), soporte `min_df`, IDF y
  ejes de la SVD.
- En `predict` / `transform` cada fila se procesa con esos parámetros congelados. La
  salida de una fila no depende de las demás filas del lote (lo comprueba
  `verificar_aislamiento`).
- Una categoría o un código que solo aparece en validación o prueba se ignora: queda
  como el nivel de referencia del one-hot y fuera del vocabulario.
- Las columnas `y_*`, las llaves y las columnas de control nunca entran: el
  `ColumnTransformer` usa `remainder="drop"` y listas explícitas.

Solo depende de numpy, pandas, scipy y scikit-learn.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from modulo_5.m5_2_svd_diagnosticos import ReductorSVDClinico

COL_DX = "dx_codigos"
CATEGORIA_FALTANTE = "Desconocido"


# --------------------------------------------------------------------------------------
# Esquema de columnas
# --------------------------------------------------------------------------------------
@dataclass
class EsquemaPipeline:
    """Qué columnas entran a cada rama. Por defecto, el esquema de los datos del HUN (M1 + 5.1 + 5.3)."""

    num: list[str] = field(default_factory=lambda: [
        "horas_desde_ingreso", "edad", "horas_desde_ultima_alta_hosp",
        "censo_servicio_t_menos_1h", "ocupacion_servicio_t_menos_1h",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "srv_ocupacion_media_3h", "srv_ocupacion_media_6h", "srv_ocupacion_media_12h", "srv_ocupacion_media_24h",
        "srv_ocupacion_max_24h", "srv_ocupacion_std_24h", "srv_ocupacion_tendencia_3h",
        "srv_frac_horas_sobre_umbral_24h", "srv_ocupacion_semana_previa",
    ])
    num_log: list[str] = field(default_factory=lambda: [
        "n_dx_historia", "n_dx_distintos_historia", "n_dx_30d", "n_visitas_365d", "n_hospitalizaciones_previas",
        "n_mediciones_24h", "n_mediciones_historia", "llegadas_servicio_t_menos_1h",
        "srv_llegadas_suma_4h", "srv_llegadas_suma_24h",
    ])
    cat: list[str] = field(default_factory=lambda: [
        "servicio", "sexo", "origen_urgencias", "mes", "turno",
        "es_fin_de_semana", "es_festivo", "es_hora_pico_urgencias", "tras_festivo", "primer_dia_habil",
        "sin_mediciones_24h", "sin_mediciones_historia", "sin_dx_historia", "nunca_hospitalizado",
    ])
    clinica: str | None = COL_DX
    # Las banderas has_measurement_* (5.1) se añaden a `cat` con `con_banderas_lab`.

    def con_banderas_lab(self, columnas: list[str]) -> "EsquemaPipeline":
        nuevo = copy.deepcopy(self)
        nuevo.cat += [c for c in columnas if c.startswith("has_measurement_") and c not in nuevo.cat]
        return nuevo

    def todas(self) -> list[str]:
        return self.num + self.num_log + self.cat + ([self.clinica] if self.clinica else [])

    def validar(self, X: pd.DataFrame) -> None:
        faltan = [c for c in self.todas() if c not in X.columns]
        if faltan:
            raise ValueError(f"Faltan columnas en X: {faltan}")
        prohibidas = [c for c in self.todas() if c.startswith("y_") or c in (
            "snapshot_id", "episodio_id", "person_id", "es_entrenamiento", "en_periodo_estable",
            "en_riesgo_truncamiento", "umbral_los_h")]
        if prohibidas:
            raise ValueError(f"Columnas prohibidas como variables (etiquetas, llaves o control): {prohibidas}")
        repetidas = pd.Index(self.todas())[pd.Index(self.todas()).duplicated()].tolist()
        if repetidas:
            raise ValueError(f"Columnas en más de una rama: {repetidas}")

    def guardar(self, ruta: Path) -> None:
        ruta.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def cargar(cls, ruta: Path) -> "EsquemaPipeline":
        return cls(**json.loads(Path(ruta).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------------------
# Funciones auxiliares (a nivel de módulo para que el Pipeline se pueda serializar)
# --------------------------------------------------------------------------------------
def a_categoria(X):
    """Convierte cada columna a texto conservando los faltantes como NaN (para el imputador constante)."""
    df = pd.DataFrame(X).copy()
    for c in df.columns:
        nulo = df[c].isna()
        df[c] = df[c].astype(object).where(nulo, df[c].astype(str))
        df.loc[nulo, c] = np.nan
    return df.to_numpy(dtype=object)


def separar_codigos(texto: str) -> list[str]:
    """Analizador del CountVectorizer: los códigos vienen separados por espacios y ya normalizados."""
    return texto.split() if isinstance(texto, str) else []


def _nombres_identidad(transformer, nombres):
    return nombres


# --------------------------------------------------------------------------------------
# Construcción
# --------------------------------------------------------------------------------------
def construir_preprocesador(
    esquema: EsquemaPipeline | None = None,
    n_componentes_svd: int = 50,
    min_df_svd: int = 5,
) -> ColumnTransformer:
    """`ColumnTransformer` con las ramas num / num_log / cat / clinica, sin ajustar."""
    e = esquema or EsquemaPipeline()
    ramas = []
    if e.num:
        ramas.append(("num", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), e.num))
    if e.num_log:
        ramas.append(("num_log", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out=_nombres_identidad)),
            ("escalar", StandardScaler()),
        ]), e.num_log))
    if e.cat:
        ramas.append(("cat", Pipeline([
            ("a_texto", FunctionTransformer(a_categoria, feature_names_out=_nombres_identidad)),
            ("imputar", SimpleImputer(strategy="constant", fill_value=CATEGORIA_FALTANTE)),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]), e.cat))
    if e.clinica:
        ramas.append(("clinica", Pipeline([
            ("vectorizar", CountVectorizer(analyzer=separar_codigos, binary=True, lowercase=False,
                                           dtype=np.float32)),
            ("svd", ReductorSVDClinico(n_components=n_componentes_svd, min_df=min_df_svd, random_state=42)),
        ]), e.clinica))  # columna como texto (no lista): el CountVectorizer recibe un vector 1-D
    return ColumnTransformer(ramas, remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=True)


def construir_pipeline(
    esquema: EsquemaPipeline | None = None,
    estimador=None,
    n_componentes_svd: int = 50,
    min_df_svd: int = 5,
) -> Pipeline:
    """Pipeline completo (preprocesador + estimador), sin ajustar, listo para M6.

    El estimador de prueba por defecto es una regresión logística L2 con
    `class_weight="balanced"` (el target está en 25–75 % según servicio y τ, M2).
    En M6 se reemplaza con `pipe.set_params(modelo=...)` o pasando `estimador=`.
    """
    if estimador is None:
        estimador = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    return Pipeline([
        ("preprocesador", construir_preprocesador(esquema, n_componentes_svd, min_df_svd)),
        ("modelo", estimador),
    ])


# --------------------------------------------------------------------------------------
# Datos: arma X a partir de M1 + 5.1 + 5.2 + 5.3
# --------------------------------------------------------------------------------------
def codigos_desde_matriz(X: sp.spmatrix, vocabulario: np.ndarray) -> np.ndarray:
    """Convierte la matriz binaria de la 5.2 (snapshots × códigos) en texto por fila."""
    X = sp.csr_matrix(X)
    vocab = np.asarray(vocabulario, dtype=object)
    partes = np.split(X.indices, X.indptr[1:-1])
    return np.array([" ".join(vocab[p]) for p in partes], dtype=object)


def preparar_X(raiz: Path, columnas_extra: tuple[str, ...] = ()) -> tuple[pd.DataFrame, EsquemaPipeline]:
    """Une snapshot de M1, banderas de la 5.1, temporales/srv de la 5.3 y códigos de la 5.2.

    Todas las tablas tienen las filas en el mismo orden que `data/m1/snapshot.parquet`
    (se verifica con `snapshot_id`). Devuelve el DataFrame (con llaves, control y
    etiquetas para poder partir y evaluar) y el esquema de columnas de entrada.
    """
    d = Path(raiz) / "data"
    m1 = pd.read_parquet(d / "m1" / "snapshot.parquet")
    m51 = pd.read_parquet(d / "m5" / "snapshot_m5_1.parquet")
    m53 = pd.read_parquet(d / "m5" / "snapshot_m5_3.parquet")
    for nombre, t in (("5.1", m51), ("5.3", m53)):
        if not t.snapshot_id.reset_index(drop=True).equals(m1.snapshot_id.reset_index(drop=True)):
            raise ValueError(f"Las filas de la subfase {nombre} no están alineadas con data/m1/snapshot.parquet")
    nuevas_51 = [c for c in m51.columns if c not in m1.columns]
    nuevas_53 = [c for c in m53.columns if c not in m1.columns and c != "snapshot_id"]
    X = pd.concat([m1.reset_index(drop=True), m51[nuevas_51].reset_index(drop=True),
                   m53[nuevas_53].reset_index(drop=True)], axis=1)
    mat = sp.load_npz(d / "m5" / "matriz_dx_m5_2.npz")
    vocab = pd.read_csv(d / "m5" / "vocabulario_dx_m5_2.csv").codigo.to_numpy()
    if mat.shape[0] != len(X):
        raise ValueError("La matriz de diagnósticos de la 5.2 no tiene el mismo número de filas que el snapshot")
    X[COL_DX] = codigos_desde_matriz(mat, vocab)
    esquema = EsquemaPipeline().con_banderas_lab(nuevas_51)
    esquema.validar(X)
    return X, esquema


# --------------------------------------------------------------------------------------
# Verificaciones anti-fuga
# --------------------------------------------------------------------------------------
def huella_parametros(pipe: Pipeline) -> str:
    """Hash de todos los parámetros aprendidos (atributos terminados en `_`) del Pipeline."""
    h = hashlib.sha256()
    for nombre, paso in _pasos_recursivos(pipe):
        for attr in sorted(a for a in vars(paso) if a.endswith("_") and not a.startswith("_")):
            h.update(f"{nombre}.{attr}".encode())
            h.update(pickle.dumps(getattr(paso, attr), protocol=4))
    return h.hexdigest()


def _pasos_recursivos(obj, prefijo=""):
    if isinstance(obj, Pipeline):
        for n, p in obj.steps:
            yield from _pasos_recursivos(p, f"{prefijo}{n}.")
    elif isinstance(obj, ColumnTransformer):
        for n, p, _ in getattr(obj, "transformers_", []):
            if p not in ("drop", "passthrough"):
                yield from _pasos_recursivos(p, f"{prefijo}{n}.")
    yield prefijo.rstrip("."), obj


def verificar_estadisticas_solo_entrenamiento(pipe: Pipeline, X_train: pd.DataFrame, esquema: EsquemaPipeline) -> dict:
    """Compara lo aprendido por el Pipeline con lo que se obtiene calculando **solo con X_train**."""
    pre = pipe.named_steps["preprocesador"]
    r = {}
    if esquema.num:
        med = X_train[esquema.num].median().to_numpy()
        r["medianas_num"] = bool(np.allclose(pre.named_transformers_["num"].named_steps["imputar"].statistics_, med, equal_nan=True))
        imp = X_train[esquema.num].fillna(X_train[esquema.num].median())
        r["medias_num"] = bool(np.allclose(pre.named_transformers_["num"].named_steps["escalar"].mean_, imp.mean().to_numpy()))
        r["desv_num"] = bool(np.allclose(pre.named_transformers_["num"].named_steps["escalar"].scale_,
                                         imp.std(ddof=0).replace(0, 1).to_numpy()))
    if esquema.cat:
        cats = pre.named_transformers_["cat"].named_steps["onehot"].categories_
        ok = True
        for c, aprendidas in zip(esquema.cat, cats):
            vistas = set(X_train[c].astype(object).where(X_train[c].notna(), CATEGORIA_FALTANTE).astype(str))
            ok &= set(aprendidas.astype(str)) == vistas
        r["categorias_onehot"] = bool(ok)
    if esquema.clinica:
        vocab = set(pre.named_transformers_["clinica"].named_steps["vectorizar"].vocabulary_)
        vistos = set(c for t in X_train[esquema.clinica] for c in separar_codigos(t))
        r["vocabulario_dx"] = vocab == vistos
    return r


def verificar_aislamiento(pipe: Pipeline, X_eval: pd.DataFrame, n: int = 200, semilla: int = 0) -> bool:
    """La predicción de una fila no depende del resto del lote (no hay estadísticas del conjunto evaluado)."""
    rng = np.random.default_rng(semilla)
    idx = rng.choice(len(X_eval), size=min(n, len(X_eval)), replace=False)
    lote = pipe.predict_proba(X_eval.iloc[idx])[:, 1]
    sola = np.array([pipe.predict_proba(X_eval.iloc[[i]])[0, 1] for i in idx[:20]])
    return bool(np.allclose(lote[:20], sola, atol=1e-10))


def verificar_pipeline_sin_fuga(pipe: Pipeline, X_train, y_train, X_val, esquema: EsquemaPipeline) -> dict:
    """Prueba integral pedida en la 5.4: fit con entrenamiento, predict en validación, y comprobación de
    que ninguna estadística de validación entró al modelo.

    1. Lo aprendido coincide con las estadísticas calculadas solo con `X_train`.
    2. `predict` sobre validación no cambia ningún parámetro aprendido (huella idéntica).
    3. Alterar por completo la validación (valores extremos, categorías y códigos nuevos)
       y volver a ajustar con el mismo `X_train` da exactamente el mismo modelo.
    4. Cada fila de validación se predice igual sola que dentro del lote.
    """
    p = clone(pipe).fit(X_train, y_train)
    huella = huella_parametros(p)
    res = verificar_estadisticas_solo_entrenamiento(p, X_train, esquema)
    p.predict(X_val)
    res["predict_no_modifica_parametros"] = huella_parametros(p) == huella
    X_val_alterada = alterar(X_val, esquema)
    p2 = clone(pipe).fit(X_train, y_train)
    p2.predict(X_val_alterada)  # categorías y códigos nunca vistos → no debe fallar
    res["modelo_independiente_de_validacion"] = huella_parametros(p2) == huella
    res["filas_independientes_del_lote"] = verificar_aislamiento(p, X_val)
    res["ok"] = all(res.values())
    return res


def alterar(X: pd.DataFrame, esquema: EsquemaPipeline, semilla: int = 1) -> pd.DataFrame:
    """Versión destrozada de un conjunto de validación: extremos, nulos, categorías y códigos nuevos."""
    rng = np.random.default_rng(semilla)
    Z = X.copy()
    for c in esquema.num + esquema.num_log:
        Z[c] = rng.uniform(1e5, 1e6, len(Z))
        Z.loc[Z.sample(frac=0.2, random_state=semilla).index, c] = np.nan
    for c in esquema.cat:
        Z[c] = "CATEGORIA_NUNCA_VISTA"
    if esquema.clinica:
        Z[esquema.clinica] = "dx_999999991 dx_999999992"
    return Z
