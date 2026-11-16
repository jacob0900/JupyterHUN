"""Subfase 5.1 · Definición de variables y banderas de nulos informativos (MNAR).

Contexto clínico
----------------
En la historia clínica la ausencia de un laboratorio casi nunca es azar: un médico
pide (o no pide) un hemograma, un ionograma o una creatinina según cómo ve al
paciente. Que no exista el registro en `measurement` es en sí una señal clínica
(Missing Not At Random). Por eso en esta subfase **no se imputa nada**: esas
ausencias se convierten en banderas binarias explícitas que el modelo puede usar.

Garantía anti-fuga (regla 5 de CLAUDE.md y regla 4 del contrato M1)
-------------------------------------------------------------------
Cada fila del maestro (`data/m1/snapshot.parquet`) es un paciente en el instante de
predicción τ (`snapshot_time`). Todas las banderas siguen reglas fijas que solo miran
eventos **de la misma `person_id` con fecha efectiva `t < τ`**:

- El cruce con `measurement` es un `merge_asof` hacia atrás **estricto**: nunca se
  empareja un laboratorio posterior ni simultáneo a τ.
- **No se cruza por `visit_occurrence_id`**: en el OMOP del HUN el 43 % de las
  mediciones apunta a la visita de otra persona (docs/M1_hallazgos_datos.md, §4).
- Si un laboratorio solo tiene fecha (o una hora 00:00:00, típica de un ETL que
  rellenó la hora), se asume disponible **al final de ese día**. Es conservador:
  puede apagar una bandera de más, pero nunca la enciende antes de tiempo.
- La lista de "laboratorios más frecuentes" se elige solo con registros anteriores
  al último τ de entrenamiento, para que el periodo de prueba no decida qué banderas
  existen.

Solo depende de pandas y numpy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Llaves del maestro: se conservan intactas y nunca son variables de entrada.
ID_COLS: tuple[str, ...] = ("snapshot_id", "episodio_id", "person_id", "visit_occurrence_id")

# Columnas de control: partición, filtros y parámetros de construcción del target.
CONTROL_COLS: frozenset[str] = frozenset(
    {"es_entrenamiento", "en_periodo_estable", "en_riesgo_truncamiento", "particion", "umbral_los_h"}
)

# Códigos numéricos que en realidad son categorías (no tienen orden aritmético útil).
CATEGORICAS_CODIFICADAS: frozenset[str] = frozenset({"mes", "dia_semana"})

# Columnas derivadas de `measurement`: van a `lab_cols` y no se imputan a ciegas.
PATRON_LAB = re.compile(r"^(has_measurement_|n_mediciones|sin_mediciones|horas_desde_ultima_medicion|lab_)")

# Banderas que salen de columnas que ya trae el snapshot de M1
# (recomendación del M2: outputs/m2/M2_informe.md, sección 7).
#   nombre_bandera: (columna_origen, regla)
#   "cero": el conteo es 0 o falta -> no hubo registro antes de τ
#   "nulo": el valor es NULL       -> el evento nunca ocurrió
BANDERAS_ESTRUCTURALES: dict[str, tuple[str, str]] = {
    "sin_mediciones_24h": ("n_mediciones_24h", "cero"),
    "sin_mediciones_historia": ("n_mediciones_historia", "cero"),
    "sin_dx_historia": ("n_dx_historia", "cero"),
    "nunca_hospitalizado": ("horas_desde_ultima_alta_hosp", "nulo"),
}

COLS_TOP = ["rango", "concepto", "nombre", "n_personas", "n_registros", "pct_personas", "sinonimos", "bandera"]


@dataclass(frozen=True)
class ConfigMNAR:
    """Parámetros de la subfase 5.1.

    Attributes:
        col_tiempo: instante de predicción τ de cada fila del maestro.
        col_entrenamiento: bandera booleana de la partición de entrenamiento.
        col_estable: bandera del periodo estable (None si no se quiere filtrar).
        top_k: cuántos laboratorios frecuentes reciben bandera propia.
        ventana_horas: el laboratorio cuenta si se tomó en [τ − ventana, τ).
            None = cualquier momento anterior a τ (toda la historia).
        prefijo: prefijo del nombre de las banderas de toma de muestra.
        umbral_sinonimo: dos conceptos son el mismo examen si sus registros
            (persona, hora) se solapan con Jaccard >= este valor. None = no fusionar.
    """

    col_tiempo: str = "snapshot_time"
    col_entrenamiento: str = "es_entrenamiento"
    col_estable: str | None = "en_periodo_estable"
    top_k: int = 20
    ventana_horas: float | None = 24.0
    prefijo: str = "has_measurement_"
    umbral_sinonimo: float | None = 0.95


@dataclass
class EsquemaColumnas:
    """Clasificación formal de las columnas del maestro.

    Las tres listas de variables (`num_cols`, `cat_cols`, `lab_cols`) son disjuntas.
    `excluidas` guarda lo que no entra al modelo y por qué (llaves, targets, control,
    tiempos y columnas sin tipo reconocible).
    """

    num_cols: list[str]
    cat_cols: list[str]
    lab_cols: list[str]
    excluidas: dict[str, list[str]] = field(default_factory=dict)

    def resumen(self) -> pd.DataFrame:
        """Tabla larga `columna, grupo` para documentar el esquema."""
        filas = [(c, "num") for c in self.num_cols]
        filas += [(c, "cat") for c in self.cat_cols]
        filas += [(c, "lab") for c in self.lab_cols]
        for motivo, cols in self.excluidas.items():
            filas += [(c, f"excluida:{motivo}") for c in cols]
        return pd.DataFrame(filas, columns=["columna", "grupo"])


# --------------------------------------------------------------------------- measurement
def _fin_del_dia(s: pd.Series) -> pd.Series:
    return s.dt.normalize() + pd.Timedelta(days=1) - pd.Timedelta(1, "ns")


def preparar_measurement(
    measurement: pd.DataFrame,
    col_concepto: str = "measurement_concept_id",
    conceptos_excluidos: tuple = (0,),
    personas: pd.Series | np.ndarray | None = None,
    medianoche_como_fecha: bool = True,
) -> pd.DataFrame:
    """Normaliza `measurement` a `person_id, concepto, t_efectiva`.

    `t_efectiva` es el momento desde el cual el resultado puede considerarse
    conocido al predecir. Lógica clínica y anti-fuga:

    - Si hay `measurement_datetime` con hora real, se usa tal cual.
    - Si solo hay `measurement_date`, o la hora es exactamente 00:00:00 (los ETL
      OMOP suelen rellenar así la hora desconocida), se usa el **final del día**.
      Sin esa regla, un examen tomado a las 18 h aparecería disponible para un
      snapshot de las 08 h del mismo día: fuga de información.

    `visit_occurrence_id` se descarta a propósito: el vínculo está roto en el HUN.

    Args:
        measurement: tabla OMOP `measurement` (al menos `person_id`, la columna de
            concepto y `measurement_datetime` o `measurement_date`).
        col_concepto: columna que identifica el examen. Por defecto el concepto
            estándar; usar `measurement_source_value_extended` si el mapeo es pobre.
        conceptos_excluidos: códigos que no son un examen real (0 = "No matching
            concept" en OMOP).
        personas: si se da, se descartan registros de personas fuera del maestro.
        medianoche_como_fecha: aplica la regla del final del día a las 00:00:00.

    Returns:
        DataFrame ordenado por `t_efectiva`.
    """
    faltan = {"person_id", col_concepto} - set(measurement.columns)
    if faltan:
        raise KeyError(f"measurement no tiene las columnas {sorted(faltan)}")
    if not {"measurement_datetime", "measurement_date"} & set(measurement.columns):
        raise KeyError("measurement necesita measurement_datetime o measurement_date")

    med = measurement
    if personas is not None:
        med = med[med["person_id"].isin(pd.unique(np.asarray(personas)))]

    if "measurement_datetime" in med.columns:
        dt = pd.to_datetime(med["measurement_datetime"], errors="coerce").astype("datetime64[ns]")
    else:
        dt = pd.Series(pd.NaT, index=med.index, dtype="datetime64[ns]")
    if medianoche_como_fecha:
        dt = dt.where(~(dt.notna() & (dt == dt.dt.normalize())), _fin_del_dia(dt))
    if "measurement_date" in med.columns:
        fecha = pd.to_datetime(med["measurement_date"], errors="coerce").astype("datetime64[ns]")
        dt = dt.fillna(_fin_del_dia(fecha))

    out = pd.DataFrame(
        {"person_id": med["person_id"].to_numpy(), "concepto": med[col_concepto].to_numpy(), "t_efectiva": dt.to_numpy()}
    )
    out = out[out["concepto"].notna() & out["t_efectiva"].notna() & out["person_id"].notna()]
    if conceptos_excluidos:
        out = out[~out["concepto"].isin(list(conceptos_excluidos))]
    return out.sort_values("t_efectiva", kind="mergesort").reset_index(drop=True)


def _slug(valor) -> str:
    """Nombre de columna seguro: minúsculas, alfanumérico y `_`, máx. 48 caracteres."""
    s = re.sub(r"[^0-9a-z]+", "_", str(valor).strip().lower()).strip("_")
    return s[:48] or "sin_nombre"


def _mascara_entrenamiento(master: pd.DataFrame, cfg: ConfigMNAR) -> pd.Series:
    if cfg.col_entrenamiento not in master.columns:
        raise KeyError(f"el maestro no tiene la columna de partición '{cfg.col_entrenamiento}'")
    m = master[cfg.col_entrenamiento].fillna(False).astype(bool)
    if cfg.col_estable and cfg.col_estable in master.columns:
        m &= master[cfg.col_estable].fillna(False).astype(bool)
    return m


def _elegir_sin_sinonimos(m: pd.DataFrame, ranking: pd.DataFrame, cfg: ConfigMNAR) -> tuple[list, dict]:
    """Recorre el ranking y acepta conceptos hasta `top_k`, fusionando los sinónimos.

    Devuelve (conceptos elegidos en orden, {elegido: [sinónimos fusionados]}).
    """
    if cfg.umbral_sinonimo is None:
        return ranking["concepto"].head(cfg.top_k).tolist(), {}
    candidatos = ranking["concepto"].head(max(3 * cfg.top_k, cfg.top_k + 20)).tolist()
    sub = m[m["concepto"].isin(candidatos)]
    claves = {
        c: np.unique(pd.util.hash_pandas_object(g[["person_id", "t_efectiva"]], index=False).to_numpy())
        for c, g in sub.groupby("concepto", sort=False)
    }
    elegidos: list = []
    sinonimos: dict = {}
    for c in candidatos:
        if len(elegidos) == cfg.top_k:
            break
        a = claves[c]
        for e in elegidos:
            b = claves[e]
            inter = np.intersect1d(a, b, assume_unique=True).size
            if inter / (a.size + b.size - inter) >= cfg.umbral_sinonimo:
                sinonimos.setdefault(e, []).append(c)
                break
        else:
            elegidos.append(c)
    return elegidos, sinonimos


def labs_mas_frecuentes(
    master: pd.DataFrame,
    med: pd.DataFrame,
    cfg: ConfigMNAR = ConfigMNAR(),
    nombres: dict | None = None,
) -> pd.DataFrame:
    """Elige los `top_k` laboratorios más pedidos, usando solo el pasado de entrenamiento.

    Lógica clínica: un examen frecuente (hemograma, creatinina, electrolitos…) es parte
    de la rutina; que **falte** dice algo del paciente o del flujo. Un examen raro casi
    siempre falta y su bandera sería casi constante, así que solo los frecuentes
    reciben bandera propia.

    Frecuencia = número de **personas distintas** con al menos un registro (no número de
    registros), para que un paciente con 200 gases arteriales no infle un examen poco
    común. Desempate por número de registros. No se usa la visita: el vínculo está roto.

    Sinónimos: en el HUN el mismo examen viene codificado dos veces (un concepto LOINC y
    uno SNOMED, p. ej. "Sodium [Moles/volume] in Serum or Plasma" y "Sodium measurement")
    con los mismos registros. Dos banderas así serían copias exactas y gastarían cupos del
    top. Si los registros (persona, hora) de un concepto se solapan con los de uno ya
    elegido con Jaccard >= `umbral_sinonimo`, se fusiona con él: no ocupa cupo y su
    bandera se enciende con cualquiera de los dos códigos.

    Anti-fuga: solo cuentan registros con `t_efectiva < max(τ de entrenamiento)`.

    Returns:
        DataFrame `rango, concepto, nombre, n_personas, n_registros, pct_personas,
        sinonimos, bandera` (`sinonimos`: códigos fusionados, separados por `;`).
    """
    entren = master.loc[_mascara_entrenamiento(master, cfg)]
    if entren.empty:
        raise ValueError("no hay filas de entrenamiento para elegir los laboratorios")
    corte = pd.to_datetime(entren[cfg.col_tiempo]).max()

    m = med[med["t_efectiva"] < corte]
    if m.empty:
        return pd.DataFrame(columns=COLS_TOP)

    por_concepto = (
        m.groupby("concepto").agg(n_personas=("person_id", "nunique"), n_registros=("person_id", "size")).reset_index()
    )
    por_concepto["_orden"] = por_concepto["concepto"].astype(str)
    ranking = (
        por_concepto.sort_values(["n_personas", "n_registros", "_orden"], ascending=[False, False, True], kind="mergesort")
        .drop(columns="_orden")
        .reset_index(drop=True)
    )
    elegidos, sinonimos = _elegir_sin_sinonimos(m, ranking, cfg)
    top = ranking[ranking["concepto"].isin(elegidos)].reset_index(drop=True)
    top["sinonimos"] = [";".join(str(s) for s in sinonimos.get(c, [])) for c in top["concepto"]]
    top.insert(0, "rango", np.arange(1, len(top) + 1))
    top["pct_personas"] = 100 * top["n_personas"] / m["person_id"].nunique()
    nombres = nombres or {}
    top.insert(2, "nombre", [nombres.get(c, "") for c in top["concepto"]])

    # Nombre de la bandera: el código del concepto (estable entre extracciones y sin
    # tildes); si el examen se identifica por texto, el texto. Se garantiza unicidad.
    vistos: set[str] = set()
    banderas = []
    for c in top["concepto"]:
        b = cfg.prefijo + _slug(c)
        if b in vistos:
            b = f"{b}_{len(vistos)}"
        vistos.add(b)
        banderas.append(b)
    top["bandera"] = banderas
    return top[COLS_TOP]


# --------------------------------------------------------------------------- banderas
def banderas_estructurales(master: pd.DataFrame) -> pd.DataFrame:
    """Añade las banderas MNAR que salen de columnas que ya trae el snapshot de M1.

    Lógica clínica (ver M2, sección 7):

    - `sin_mediciones_24h = 1`: no se tomó ningún laboratorio en las 24 h previas a τ.
      En urgencias en τ = 0 le pasa al ~97 % de los pacientes; baja con τ porque los
      laboratorios se piden durante la estancia.
    - `sin_mediciones_historia = 1`: el paciente nunca tuvo laboratorios antes de τ.
    - `sin_dx_historia = 1`: no hay diagnósticos previos; paciente nuevo para el hospital.
    - `nunca_hospitalizado = 1`: `horas_desde_ultima_alta_hosp` es NULL porque nunca hubo
      alta previa. **No es un dato perdido**: no se debe imputar con la mediana.

    Los conteos de M1 se construyeron con 0 por defecto, así que en la regla "cero" un
    NULL también se lee como "no hubo registro". Solo se crean las banderas cuya columna
    de origen existe. La polaridad (1 = ausencia) es la misma del EDA de M2.
    """
    nuevas = {}
    for nombre, (origen, regla) in BANDERAS_ESTRUCTURALES.items():
        if origen not in master.columns:
            continue
        if nombre in master.columns:
            raise ValueError(f"la columna '{nombre}' ya existe en el maestro")
        x = master[origen]
        m = x.isna() if regla == "nulo" else (x.fillna(0) == 0)
        nuevas[nombre] = m.to_numpy().astype(np.int8)
    if not nuevas:
        return master.copy()
    return pd.concat([master, pd.DataFrame(nuevas, index=master.index)], axis=1)


def banderas_toma_muestra(
    master: pd.DataFrame,
    med: pd.DataFrame,
    top: pd.DataFrame,
    cfg: ConfigMNAR = ConfigMNAR(),
) -> pd.DataFrame:
    """Añade `has_measurement_X` (int8) por cada laboratorio de `top`.

    Regla fija, igual para entrenamiento y prueba:

        has_measurement_X = 1  si la misma persona tiene un registro del examen X
                               con t_efectiva en [τ − ventana_horas, τ)
                           = 0  en otro caso

    Lógica clínica: la bandera responde "¿el equipo médico ya había pedido X cuando
    tocaba predecir?". Un 0 no significa "valor normal" ni "valor que hay que adivinar":
    significa que no se pidió, y eso refleja la sospecha clínica, la gravedad percibida
    y el flujo del servicio (p. ej. pacientes de urgencias que se van antes de que
    lleguen los laboratorios).

    Anti-fuga: `pd.merge_asof(direction="backward", allow_exact_matches=False)` sobre τ
    solo empareja registros con `t_efectiva < τ`. El cruce es por `person_id`.

    El maestro no se reordena ni se modifica: se devuelve una copia con las columnas
    nuevas al final, con el mismo índice y el mismo orden de filas.
    """
    t = cfg.col_tiempo
    for c in ("person_id", t):
        if c not in master.columns:
            raise KeyError(f"el maestro no tiene la columna '{c}'")
    tau = pd.to_datetime(master[t]).astype("datetime64[ns]")
    if tau.isna().any():
        raise ValueError(f"hay {int(tau.isna().sum())} filas sin '{t}': no se puede fijar τ")
    choque = [b for b in top["bandera"] if b in master.columns]
    if choque:
        raise ValueError(f"las columnas {choque} ya existen en el maestro")

    izq = pd.DataFrame({"_pos": np.arange(len(master)), "_tau": tau.to_numpy(), "person_id": master["person_id"].to_numpy()})
    izq = izq.sort_values("_tau", kind="mergesort")
    tol = pd.Timedelta(hours=cfg.ventana_horas) if cfg.ventana_horas is not None else None

    # Los sinónimos fusionados cuentan como el concepto que los representa.
    a_representante = {c: c for c in top["concepto"]}
    if "sinonimos" in top.columns:
        for c, sins in zip(top["concepto"], top["sinonimos"].fillna("").astype(str)):
            for s in filter(None, sins.split(";")):
                a_representante[type(c)(s) if not isinstance(c, str) else s] = c
    sub = med[med["concepto"].isin(list(a_representante))].copy()
    sub["concepto"] = sub["concepto"].map(a_representante)
    grupos = dict(tuple(sub.groupby("concepto", sort=False)))

    nuevas: dict[str, np.ndarray] = {}
    for concepto, bandera in zip(top["concepto"], top["bandera"]):
        flag = np.zeros(len(master), dtype=np.int8)
        der = grupos.get(concepto)
        if der is not None:
            der = (
                der[["person_id", "t_efectiva"]]
                .astype({"person_id": izq["person_id"].dtype, "t_efectiva": "datetime64[ns]"})
                .rename(columns={"t_efectiva": "_t_lab"})
                .sort_values("_t_lab", kind="mergesort")
            )
            m = pd.merge_asof(
                izq, der, left_on="_tau", right_on="_t_lab", by="person_id",
                direction="backward", allow_exact_matches=False, tolerance=tol,
            )
            flag[m["_pos"].to_numpy()] = m["_t_lab"].notna().to_numpy()
        nuevas[bandera] = flag

    if not nuevas:
        return master.copy()
    return pd.concat([master, pd.DataFrame(nuevas, index=master.index)], axis=1)


# --------------------------------------------------------------------------- clasificación
def clasificar_columnas(df: pd.DataFrame, excluir: list[str] | tuple[str, ...] = ()) -> EsquemaColumnas:
    """Clasifica las columnas en `num_cols`, `cat_cols` y `lab_cols` (disjuntas).

    Reglas, en orden de prioridad:

    1. Excluidas: llaves (`snapshot_id`, `episodio_id`, `person_id`, `*_id`), targets
       (`y_*`, `target_*`: son futuro), control (partición, filtros, `umbral_los_h`),
       tiempos (dtype datetime) y `excluir`. Los tiempos crudos no entran: lo útil
       (hora, día, mes, fin de semana) ya está como variable aparte.
    2. `lab_cols`: lo derivado de `measurement` (`has_measurement_*`, `n_mediciones*`,
       `sin_mediciones*`, `lab_*`). Se separa porque su ausencia es informativa: se
       trata con banderas y no con imputación ciega.
    3. `cat_cols`: texto, categoría, booleanos, códigos categóricos (`mes`,
       `dia_semana`) y las banderas estructurales no-lab (`sin_dx_historia`,
       `nunca_hospitalizado`).
    4. `num_cols`: el resto de columnas numéricas.
    """
    excl: dict[str, list[str]] = {"id": [], "target": [], "control": [], "tiempo": [], "manual": [], "otro": []}
    num, cat, lab = [], [], []
    excluir = set(excluir)
    flags_cat = {k for k in BANDERAS_ESTRUCTURALES if not PATRON_LAB.match(k)}

    for c in df.columns:
        s = df[c]
        if c in excluir:
            excl["manual"].append(c)
        elif c in ID_COLS or c.endswith("_id"):
            excl["id"].append(c)
        elif c.startswith(("y_", "target_")):
            excl["target"].append(c)
        elif c in CONTROL_COLS:
            excl["control"].append(c)
        elif pd.api.types.is_datetime64_any_dtype(s) or isinstance(s.dtype, pd.PeriodDtype):
            excl["tiempo"].append(c)
        elif PATRON_LAB.match(c):
            lab.append(c)
        elif (
            c in flags_cat
            or c in CATEGORICAS_CODIFICADAS
            or pd.api.types.is_bool_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
        ):
            cat.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            num.append(c)
        else:
            excl["otro"].append(c)

    return EsquemaColumnas(num, cat, lab, {k: v for k, v in excl.items() if v})


# --------------------------------------------------------------------------- orquestador
def construir_subfase_5_1(
    master: pd.DataFrame,
    measurement: pd.DataFrame | None = None,
    cfg: ConfigMNAR = ConfigMNAR(),
    col_concepto: str = "measurement_concept_id",
    nombres: dict | None = None,
) -> tuple[pd.DataFrame, EsquemaColumnas, pd.DataFrame]:
    """Ejecuta la subfase 5.1 completa.

    1. Añade las banderas estructurales (`sin_mediciones_24h`, `nunca_hospitalizado`…).
    2. Si hay `measurement`: calcula la fecha efectiva de cada examen, elige los `top_k`
       más frecuentes con el pasado de entrenamiento y añade `has_measurement_X`.
       Si `measurement` es None se omite este paso (la tabla aún no está extraída).
    3. Clasifica las columnas en `num_cols`, `cat_cols` y `lab_cols`.

    Verifica al final que el índice, las llaves (`snapshot_id`, `episodio_id`,
    `person_id`, `visit_occurrence_id` si existe) salen intactas.

    Returns:
        (maestro enriquecido, esquema de columnas, tabla de laboratorios elegidos)
    """
    out = banderas_estructurales(master)
    top = pd.DataFrame(columns=COLS_TOP)
    if measurement is not None:
        med = preparar_measurement(measurement, col_concepto=col_concepto, personas=master["person_id"])
        top = labs_mas_frecuentes(master, med, cfg, nombres=nombres)
        out = banderas_toma_muestra(out, med, top, cfg)

    assert out.index.equals(master.index), "el índice del maestro cambió"
    for c in ID_COLS:
        if c in master.columns:
            assert out[c].equals(master[c]), f"'{c}' cambió durante la subfase 5.1"

    return out, clasificar_columnas(out), top
