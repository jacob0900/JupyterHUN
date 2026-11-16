# M1 · Contrato de datos (entrega de Persona 1 a Persona 2 y Persona 3)

Archivos: `data/m1/*.parquet` (no se suben a git; se regeneran con el pipeline).
Contexto de por qué cada regla existe: [M1_hallazgos_datos.md](M1_hallazgos_datos.md).

```bash
python m1_etl/run_etl.py extraer      # servidor → data/raw (solo lectura, VPN)
python m1_etl/run_etl.py construir    # data/raw → data/m1 + outputs/m1_validaciones.md
```

## Reglas que todo el equipo debe respetar

1. **Columnas `y_*` son etiquetas o futuro.** Nunca entran como variables de un modelo.
2. **Partición temporal**: usar `es_entrenamiento` (inicio del episodio < `parametros.fin_entrenamiento`) para el test final. La CV dentro de entrenamiento va con `StratifiedGroupKFold` por **`person_id`** (paciente) o `TimeSeriesSplit` con gap ≥ 4 h (servicio).
3. **Filtrar `en_periodo_estable = true`** para entrenar y evaluar. Fuera de ese periodo la carga de datos es distinta (inicio de la base y desde abril de 2026).
4. **`visit_occurrence_id` no sirve para cruzar eventos** (re-anonimización rota). Cualquier variable nueva se construye por `person_id` + `datetime < snapshot_time`.
5. **Procedimientos no se usan**: `procedure_date` está vacío.
6. **No recalcular umbrales ni capacidades con datos de prueba.** El umbral de LOS y la capacidad ya vienen calculados sin fuga.

## Tablas

### `snapshot` — tarea de PACIENTE (una fila = un paciente en un instante τ)

Snapshots en τ = ingreso al servicio + {0, 2, 6} h, solo si el paciente sigue ahí en τ y aún no superó el umbral.

| Columna | Descripción |
|---|---|
| snapshot_id, episodio_id, person_id | llaves. **`person_id` = grupo para la CV** |
| servicio | `hospitalizacion` o `urgencias_observacion` |
| horas_desde_ingreso, snapshot_time | instante de decisión τ |
| es_entrenamiento, en_periodo_estable, en_riesgo_truncamiento | partición y filtros |
| edad, sexo, origen_urgencias | demografía; si la hospitalización vino de urgencias |
| hora_del_dia, dia_semana, mes, fin_de_semana | calendario (los festivos colombianos se agregan en M4) |
| n_dx_historia, n_dx_distintos_historia, n_dx_30d | diagnósticos de la persona con datetime < τ |
| n_mediciones_24h, n_mediciones_historia | laboratorios < τ. `n_mediciones_24h = 0` es informativo (MNAR) |
| n_visitas_365d, n_hospitalizaciones_previas, horas_desde_ultima_alta_hosp | historial de uso |
| censo_servicio_t_menos_1h, ocupacion_servicio_t_menos_1h, llegadas_servicio_t_menos_1h | carga del servicio 1 h antes de τ, **sin contar al propio paciente** |
| umbral_los_h | P75 del servicio (calculado solo con entrenamiento) |
| **y_target_los_largo** | 1 si la estancia total supera `umbral_los_h` (target de clasificación) |
| y_los_h, y_los_restante_h, y_fin_episodio | targets de regresión / futuro |

### `episodio` — estancias (una fila por episodio)

Subprocesos `visit_detail` del mismo servicio y la misma visita unidos si se solapan o los separa ≤ 1 h. Solo vínculos coherentes (persona conocida). Incluye `los_h`, `target_los_largo`, `umbral_los_h` y los mismos filtros. Sirve para el EDA univariado y las curvas Kaplan-Meier (no hay estancias abiertas: `fin` siempre existe; `en_riesgo_truncamiento` marca los ingresos de los últimos 30 días).

### `ocupacion_servicio_hora` — tarea de SERVICIO (una fila por servicio y hora)

| Columna | Descripción |
|---|---|
| servicio, hora | llave |
| censo, llegadas | pacientes presentes en `hora` y llegadas en esa hora (todos los subprocesos, incluidos los mal vinculados) |
| capacidad | P95 del censo de los 90 días anteriores (proxy móvil, sin fuga) o camas reales |
| ocupacion | censo / capacidad (conocida en `hora`) |
| umbral_ocupacion | 0.85 urgencias, 0.95 hospitalización (ver abajo) |
| **y_target_saturacion** | 1 si `ocupacion` en `hora + 4 h` ≥ umbral |
| y_ocupacion_futura | ocupación en `hora + 4 h` (regresión) |
| es_entrenamiento, en_periodo_estable | partición y filtro |

Para predecir en `hora` se pueden usar `censo`, `ocupacion` y `llegadas` de `hora` y anteriores (lags). **No** se puede usar nada con `y_`.

**Sobre el umbral de hospitalización:** sin el número real de camas, "ocupación" es censo / P95 reciente. El hospital opera en promedio al ~93 % de ese valor, así que con 0.85 el 90–95 % de las horas serían positivas (target inútil). Con 0.95 la tasa es ~38 % y estable entre entrenamiento y prueba. **Si conseguimos las camas reales**, se ponen en `config/capacidad_camas.csv` (`servicio,camas`) y ambos umbrales vuelven a 0.85 (`HUN_UMBRAL_OCUPACION_HOSP=0.85` en `.env`).

### Otras

- `llegadas_servicio_hora`: llegadas por hora a los 4 servicios (incluye `urgencias_ingreso` y `ambulatorio`). Base para los heatmaps del M4.
- `detalle`: subprocesos con `servicio`, duración y `vinculo_coherente`. Base para las transiciones del M4.
- `umbral_los`, `capacidad_servicio`, `parametros`: trazabilidad de cómo se construyó todo.
- `reporte_calidad`: n afectados / n total por chequeo de calidad.

## Parámetros (`.env`)

| Variable | Defecto | Qué controla |
|---|---|---|
| HUN_SNAPSHOTS_HORAS | 0,2,6 | instantes de decisión |
| HUN_PERCENTIL_LOS | 0.75 | definición de estancia larga |
| HUN_TRAIN_END | P80 del periodo estable | corte entrenamiento/prueba |
| HUN_FIN_PERIODO_ESTABLE | 2026-04-01 | fin del periodo confiable |
| HUN_UMBRAL_OCUPACION_URG / _HOSP | 0.85 / 0.95 | definición de saturación |
| HUN_HORIZONTE_H | 4 | horizonte del target de saturación |
| HUN_PERCENTIL_CAPACIDAD, HUN_VENTANA_CAPACIDAD_DIAS | 0.95, 90 | capacidad proxy |
| HUN_TOLERANCIA_VINCULO_H | 12 | regla de vínculo coherente |
| HUN_GAP_UNION_H | 1 | unión de subprocesos en episodios |
