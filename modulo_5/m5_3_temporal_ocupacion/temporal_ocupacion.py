"""Subfase 5.3 · Variables cíclicas temporales y saturación del servicio en t − 1.

Contexto operativo
------------------
La demanda del HUN sigue ciclos muy marcados (M4): pico de llegadas a urgencias a
media mañana entre semana, saturación de observación por la tarde y de
hospitalización de madrugada y mañana, festivo ≈ domingo y rebote el primer día
hábil tras el descanso. Esta subfase convierte el tiempo en variables útiles para
un modelo y resume la presión del servicio **solo con información pasada**.

1. **Codificación cíclica.** La hora 23 y la hora 0 están a 1 h de distancia, pero
   como enteros están a 23. Con seno/coseno quedan juntas:
       hour_sin = sin(2π·h/24), hour_cos = cos(2π·h/24)
       dow_sin  = sin(2π·d/7),  dow_cos  = cos(2π·d/7)      (d: lunes = 0 … domingo = 6)
   En snapshots de paciente la hora lleva los minutos (10:30 → 10.5) para no perder
   continuidad.
2. **Banderas de calendario.** Hora pico de urgencias (por defecto 07–10 h y
   17–20 h, configurable), fin de semana, festivo colombiano (Ley Emiliani, tomado
   del `calendario_dia` de Persona 1 o, si no está, de la librería `holidays`),
   día no hábil, `tras_festivo` (el día hábil siguiente a un festivo real) y
   `primer_dia_habil` (el día hábil siguiente a cualquier día no hábil). Estas dos
   últimas se separan porque en el EDA el rebote de demanda resultó ser sobre todo
   un efecto lunes (docs/P3_entrega_eda.md). Además, `turno` (mañana 7–13, tarde
   13–19, noche 19–7).
3. **Ocupación del servicio en t − 1.** Tasa de ocupación activa = censo / capacidad
   (capacidad = proxy P95 del censo de los 90 días previos, ya sin fuga en M1). Para
   cada servicio se arma una **rejilla horaria completa** y se aplica `.shift(1)`
   **antes** de cualquier ventana móvil: la fila de la hora t solo ve las horas
   ≤ t − 1. Se calculan el último valor, medias y máximo móviles, tendencia,
   fracción de horas sobre el umbral, el valor de la misma hora de la semana
   anterior y las llegadas recientes.

Garantía anti-fuga
------------------
- El target de servicio es `y_target_saturacion(t) = 1{ocupacion(t + 4 h) ≥ umbral}`.
  Ninguna variable de esta subfase lee columnas `y_*` ni valores de la hora t o
  posteriores: todo pasa por `shift(1)` sobre la rejilla horaria.
- Si falta una hora en la serie, la rejilla deja un hueco (NaN). `shift(1)` nunca
  toma "la fila anterior" cuando esa fila no es la hora anterior.
- `verificar_sin_fuga` lo comprueba de forma empírica: altera todos los datos desde
  un instante de corte en adelante (incluido el propio corte y las etiquetas) y
  exige que las variables de las filas ≤ corte no cambien ni un bit.
- Las ventanas móviles pueden usar horas anteriores al periodo estable: es pasado,
  no fuga. Filtrar `en_periodo_estable` **después** de construir las variables.
- Nada aquí se "aprende" de los datos: son reglas fijas (no hace falta reajustar por fold).

Solo depende de pandas y numpy (`holidays` es opcional).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TURNOS = ("mañana", "tarde", "noche")


@dataclass(frozen=True)
class ConfigTemporal:
    """Parámetros de la subfase 5.3 (todos son reglas fijas, no se ajustan con datos)."""

    # Franjas pico de urgencias [inicio, fin) en horas enteras. El prompt sugiere 07–10 y 17–20 h.
    # En el EDA (M4) el pico real de llegadas a urgencias está en 08–12 h y 18 h: se puede cambiar aquí.
    horas_pico: tuple[tuple[int, int], ...] = ((7, 10), (17, 20))
    ventanas_h: tuple[int, ...] = (3, 6, 12, 24)  # ventanas móviles de ocupación (sobre la serie ya desplazada)
    ventanas_llegadas_h: tuple[int, ...] = (4, 24)
    lag_estacional_h: int = 168  # misma hora de la semana anterior
    umbral_por_servicio: dict[str, float] = field(
        default_factory=lambda: {"hospitalizacion": 0.95, "urgencias_observacion": 0.85}
    )
    col_servicio: str = "servicio"


# --------------------------------------------------------------------------------------
# 1 · Calendario colombiano
# --------------------------------------------------------------------------------------
def calendario_colombiano(
    fechas: pd.Series | pd.DatetimeIndex,
    calendario_dia: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Tabla diaria con `fecha, es_festivo, es_fin_de_semana, es_dia_no_habil, tras_festivo, primer_dia_habil`.

    Usa la columna `festivo` de `calendario_dia` (Persona 1, `data/m4/calendario_dia.parquet`).
    Si no se pasa, usa `holidays.Colombia`. **No** usa `calendario_dia.dia_despues_festivo`:
    esa columna marca también los lunes normales (ver docs/P3_entrega_eda.md).
    Cubre desde el día anterior a la primera fecha para poder calcular `tras_festivo` el primer día.
    """
    f = pd.to_datetime(pd.Series(fechas)).dt.normalize()
    rango = pd.date_range(f.min() - pd.Timedelta(days=1), f.max(), freq="D")
    cal = pd.DataFrame({"fecha": rango})
    if calendario_dia is not None:
        c = calendario_dia[["fecha", "festivo"]].copy()
        c["fecha"] = pd.to_datetime(c["fecha"]).dt.normalize()
        cal = cal.merge(c, on="fecha", how="left")
        faltan = cal["festivo"].isna()
        if faltan.any():  # fechas fuera del calendario de Persona 1 → librería holidays
            cal.loc[faltan, "festivo"] = _festivos_holidays(cal.loc[faltan, "fecha"])
        cal["es_festivo"] = cal.pop("festivo").astype(bool)
    else:
        cal["es_festivo"] = _festivos_holidays(cal["fecha"])
    cal["es_fin_de_semana"] = cal["fecha"].dt.dayofweek >= 5
    cal["es_dia_no_habil"] = cal["es_festivo"] | cal["es_fin_de_semana"]
    habil = ~cal["es_dia_no_habil"]
    cal["tras_festivo"] = habil & cal["es_festivo"].shift(1, fill_value=False)
    cal["primer_dia_habil"] = habil & cal["es_dia_no_habil"].shift(1, fill_value=False)
    return cal


def _festivos_holidays(fechas: pd.Series) -> np.ndarray:
    try:
        import holidays
    except ImportError as e:  # pragma: no cover - depende del entorno
        raise ImportError("Sin calendario_dia hace falta `pip install holidays` para los festivos de Colombia") from e
    fechas = pd.to_datetime(fechas)
    co = holidays.Colombia(years=sorted(set(fechas.dt.year)))
    return np.array([d.date() in co for d in fechas], dtype=bool)


# --------------------------------------------------------------------------------------
# 2 · Variables temporales (cíclicas + banderas)
# --------------------------------------------------------------------------------------
def variables_temporales(
    tiempo: pd.Series,
    cfg: ConfigTemporal = ConfigTemporal(),
    calendario_dia: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Variables que dependen solo del instante de predicción (conocidas en t, sin fuga posible).

    Devuelve un DataFrame con el mismo índice que `tiempo`.
    """
    t = pd.to_datetime(tiempo)
    hora_frac = t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600
    hora = t.dt.hour
    dow = t.dt.dayofweek  # lunes = 0 … domingo = 6
    out = pd.DataFrame(index=tiempo.index)
    out["hour_sin"] = np.sin(2 * np.pi * hora_frac / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hora_frac / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["es_hora_pico_urgencias"] = np.zeros(len(t), dtype=bool)
    for ini, fin in cfg.horas_pico:
        out["es_hora_pico_urgencias"] |= (hora >= ini) & (hora < fin)
    out["turno"] = np.select([(hora >= 7) & (hora < 13), (hora >= 13) & (hora < 19)], TURNOS[:2], TURNOS[2])

    cal = calendario_colombiano(t, calendario_dia).set_index("fecha")
    dia = t.dt.normalize()
    for c in ("es_fin_de_semana", "es_festivo", "es_dia_no_habil", "tras_festivo", "primer_dia_habil"):
        out[c] = dia.map(cal[c]).astype(bool).to_numpy()
    banderas = [c for c in out.columns if c.startswith(("es_", "tras_", "primer_"))]
    out[banderas] = out[banderas].astype(np.int8)
    return out


# --------------------------------------------------------------------------------------
# 3 · Ocupación del servicio en t − 1
# --------------------------------------------------------------------------------------
def variables_ocupacion_t1(
    df: pd.DataFrame,
    col_tiempo: str = "hora",
    cfg: ConfigTemporal = ConfigTemporal(),
) -> pd.DataFrame:
    """Presión del servicio con información hasta t − 1 (mismo índice que `df`).

    Necesita `servicio`, `col_tiempo` (hora exacta) y `censo` + `capacidad` (o `ocupacion`).
    `llegadas` es opcional. Nunca lee columnas `y_*`.
    """
    s_col = cfg.col_servicio
    base = df[[s_col, col_tiempo]].copy()
    base[col_tiempo] = pd.to_datetime(base[col_tiempo])
    if base.duplicated([s_col, col_tiempo]).any():
        raise ValueError("Hay filas repetidas por (servicio, hora): la serie debe ser una fila por servicio y hora")

    # Tasa de ocupación ACTIVA del instante (se calcula aquí y se desplaza abajo antes de usarla).
    if {"censo", "capacidad"}.issubset(df.columns):
        cap = df["capacidad"].astype(float).where(df["capacidad"] > 0)
        base["_ocup"] = df["censo"].astype(float) / cap
    elif "ocupacion" in df.columns:
        base["_ocup"] = df["ocupacion"].astype(float)
    else:
        raise ValueError("Faltan columnas: se necesita censo + capacidad, u ocupacion")
    base["_censo"] = df["censo"].astype(float) if "censo" in df.columns else np.nan
    base["_lleg"] = df["llegadas"].astype(float) if "llegadas" in df.columns else np.nan

    piezas = []
    for serv, g in base.groupby(s_col, sort=False):
        g = g.set_index(col_tiempo).sort_index()
        # Rejilla horaria completa: shift(1) = exactamente 1 hora antes (nunca "la fila anterior").
        rejilla = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(rejilla)
        umbral = cfg.umbral_por_servicio.get(serv, np.nan)

        # ⚠️ ANTI-FUGA: .shift(1) ANTES de cualquier ventana. Desde aquí en adelante solo se usa
        # `o1`, `c1`, `l1` (valores de t − 1 hacia atrás). El valor de la hora t (`_ocup`, que
        # comparte instante con la etiqueta y_t) no entra en ninguna variable.
        o1 = g["_ocup"].shift(1)
        c1 = g["_censo"].shift(1)
        l1 = g["_lleg"].shift(1)

        f = pd.DataFrame(index=rejilla)
        f["ocupacion_t1"] = o1
        for w in cfg.ventanas_h:
            f[f"ocupacion_media_{w}h"] = o1.rolling(w, min_periods=w).mean()
        f["ocupacion_max_24h"] = o1.rolling(24, min_periods=24).max()
        f["ocupacion_std_24h"] = o1.rolling(24, min_periods=24).std()
        f["ocupacion_tendencia_3h"] = o1 - o1.shift(3)  # ocup(t−1) − ocup(t−4)
        f["frac_horas_sobre_umbral_24h"] = (o1 >= umbral).astype(float).where(o1.notna()).rolling(24, min_periods=24).mean()
        f[f"ocupacion_semana_previa"] = g["_ocup"].shift(cfg.lag_estacional_h)
        f["censo_t1"] = c1
        f["llegadas_t1"] = l1
        for w in cfg.ventanas_llegadas_h:
            f[f"llegadas_suma_{w}h"] = l1.rolling(w, min_periods=w).sum()
        f[s_col] = serv
        piezas.append(f.rename_axis(col_tiempo).reset_index())

    feats = pd.concat(piezas, ignore_index=True)
    out = base[[s_col, col_tiempo]].reset_index().merge(feats, on=[s_col, col_tiempo], how="left").set_index("index")
    out.index.name = df.index.name
    return out.drop(columns=[s_col, col_tiempo]).loc[df.index]


# --------------------------------------------------------------------------------------
# 4 · Función principal pedida en el prompt
# --------------------------------------------------------------------------------------
def build_temporal_and_occupancy_features(
    df: pd.DataFrame,
    col_tiempo: str | None = None,
    calendario_dia: pd.DataFrame | None = None,
    cfg: ConfigTemporal = ConfigTemporal(),
) -> pd.DataFrame:
    """Devuelve `df` + variables cíclicas, banderas de calendario y ocupación en t − 1.

    - Serie de servicio (`ocupacion_servicio_hora`, columna `hora`): las tres partes.
    - Snapshots de paciente (`snapshot_time`) u otras tablas sin censo: solo la parte
      temporal; la ocupación t − 1 se une con `unir_ocupacion_a_snapshots`.

    No modifica `df` y conserva su índice y orden.
    """
    if col_tiempo is None:
        col_tiempo = next((c for c in ("hora", "snapshot_time") if c in df.columns), None)
        if col_tiempo is None:
            raise ValueError("No encuentro la columna de tiempo: pasa col_tiempo=")
    partes = [df, variables_temporales(df[col_tiempo], cfg, calendario_dia)]
    if cfg.col_servicio in df.columns and ({"censo", "capacidad"}.issubset(df.columns) or "ocupacion" in df.columns):
        partes.append(variables_ocupacion_t1(df, col_tiempo, cfg))
    nuevas = pd.concat(partes[1:], axis=1)
    choque = nuevas.columns.intersection(df.columns)
    if len(choque):
        raise ValueError(f"Columnas que ya existen en df: {list(choque)}")
    return pd.concat([df, nuevas], axis=1)


def unir_ocupacion_a_snapshots(
    snapshots: pd.DataFrame,
    feats_servicio: pd.DataFrame,
    col_tiempo_snap: str = "snapshot_time",
    col_hora: str = "hora",
    col_servicio: str = "servicio",
) -> pd.DataFrame:
    """Une a cada snapshot la ocupación t − 1 de su servicio.

    Un snapshot en τ (p. ej. 10:37) toma la fila de la hora floor(τ) = 10:00, cuyas
    variables ya solo usan horas ≤ 09:00 < τ. Devuelve solo las columnas nuevas,
    con prefijo `srv_`, alineadas al índice de `snapshots`.
    """
    cols = [c for c in feats_servicio.columns if c.startswith(("ocupacion_", "censo_t1", "llegadas_", "frac_horas_"))]
    clave = snapshots[[col_servicio]].copy()
    clave[col_hora] = pd.to_datetime(snapshots[col_tiempo_snap]).dt.floor("h")
    der = feats_servicio[[col_servicio, col_hora] + cols].copy()
    der[col_hora] = pd.to_datetime(der[col_hora])
    out = clave.reset_index().merge(der, on=[col_servicio, col_hora], how="left").set_index("index")
    out.index.name = snapshots.index.name
    return out[cols].add_prefix("srv_").loc[snapshots.index]


# --------------------------------------------------------------------------------------
# 5 · Verificación anti-fuga
# --------------------------------------------------------------------------------------
def verificar_sin_fuga(
    df: pd.DataFrame,
    corte: pd.Timestamp,
    col_tiempo: str = "hora",
    calendario_dia: pd.DataFrame | None = None,
    cfg: ConfigTemporal = ConfigTemporal(),
    semilla: int = 0,
) -> dict:
    """Test empírico de no-fuga sobre la serie de servicio.

    1. Construye las variables con los datos originales.
    2. Reemplaza por ruido **todo** lo observado en `t ≥ corte` (censo, capacidad,
       ocupación, llegadas) y todas las etiquetas `y_*` en cualquier instante.
    3. Reconstruye. Las variables de las filas con `t ≤ corte` deben ser idénticas:
       en particular, la fila del propio corte no puede depender de su ocupación
       (que comparte instante con y_t) ni del futuro (de donde sale y_t = ocup(t + 4)).

    Devuelve {"ok", "filas_comparadas", "columnas_con_diferencias"}.
    """
    rng = np.random.default_rng(semilla)
    a = build_temporal_and_occupancy_features(df, col_tiempo, calendario_dia, cfg)
    b_in = df.copy()
    fut = pd.to_datetime(b_in[col_tiempo]) >= corte
    for c in ("censo", "capacidad", "ocupacion", "llegadas"):
        if c in b_in.columns:
            b_in.loc[fut, c] = rng.uniform(1, 1000, fut.sum()).astype(b_in[c].dtype) if fut.any() else b_in[c]
    for c in [c for c in b_in.columns if c.startswith("y_")]:
        b_in[c] = rng.permutation(b_in[c].to_numpy())
    b = build_temporal_and_occupancy_features(b_in, col_tiempo, calendario_dia, cfg)

    nuevas = [c for c in a.columns if c not in df.columns]
    pasado = (pd.to_datetime(df[col_tiempo]) <= corte).to_numpy()
    distintas = [
        c for c in nuevas
        if not a.loc[pasado, c].reset_index(drop=True).equals(b.loc[pasado, c].reset_index(drop=True))
    ]
    return {"ok": not distintas, "filas_comparadas": int(pasado.sum()), "columnas_con_diferencias": distintas}
