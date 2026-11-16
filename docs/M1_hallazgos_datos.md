# M1 · Hallazgos de la exploración de `uhn_i.uninorte_omop`

Fecha: 2026-11-10 · Autor: Persona 1 · Consultas: `m1_etl/sql/00*.sql` (solo lectura) · Resultados: `outputs/exploracion/`

## 1. Conexión y alcance

- Base: **`uhn_i`**, esquema **`uninorte_omop`** (no existe una base llamada `uninorte_omop`).
- Usuario sin permisos de escritura y servidor de **solo lectura por decisión del equipo**: todo el ETL se hace con `SELECT` y se procesa en local.
- Periodo: **2023-01-01 → 2026-08-13**. 1 706 001 visitas, 304 457 personas con visitas (912 083 en `person`).

## 2. El OMOP del HUN está recortado

| Tabla | Columnas disponibles | Lo que falta respecto al estándar |
|---|---|---|
| visit_occurrence | id, person_id, visit_concept_id, start/end_datetime, provider_id | `*_date`, source_value, admitted_from, discharged_to, care_site |
| visit_detail | id, visit_occurrence_id, visit_detail_concept_id, start/end_datetime, provider_id | **person_id**, care_site_id, source_value |
| condition_occurrence | id, person_id, visit_occurrence_id, concept_id, start_datetime, type_concept_id | fecha fin, status |
| procedure_occurrence | id, person_id, visit_occurrence_id, concept_id, **procedure_date (sin hora, todo NULL)** | datetime |
| measurement | id, person_id, visit_occurrence_id, concept_id, datetime, value_as_number, `measurement_source_value_extended` | unidad, value_as_concept |
| person | id, gender, year_of_birth, race, ethnicity | — |
| cups (extra) | codigo, descripcion, periodo | catálogo de procedimientos colombiano |

`provider_id` es siempre 0.

## 3. La estancia NO está en `visit_occurrence`

- En **el 100 % de las visitas `visit_end_datetime = visit_start_datetime`**: LOS = 0 h en todos los tipos de visita. No hay visitas abiertas.
- Las duraciones reales están en **`visit_detail`**:

| visit_detail_concept_id | Interpretación | n | Duración p50 / p75 / p90 |
|---|---|---|---|
| 581479 Ambulatory Rehabilitation Visit | Consulta ambulatoria (horario 7–17 h) | 1.54 M | 0 h |
| 3045729 Treatment urgency | Urgencias / observación (24 h) | 1.15 M | 1.2 / 4.7 / 12.6 h |
| 9203 Emergency Room Visit | Ingreso a urgencias (registro puntual) | 0.48 M | 3 / 4 / 6 min |
| 9201 Inpatient Visit | **Hospitalización** | 108 601 | **46 / 96 / 188 h** |

- Solo hay **4 "servicios"** identificables. No hay UCI, pediatría, etc.
- El **triaje no existe** en `measurement`: no hay conceptos ni texto fuente con triage/triaje/clasificación.

## 4. Problema crítico: eventos vinculados a la visita equivocada

En una parte de los registros, `visit_occurrence_id` apunta a una visita que **no es la suya**:

- Diagnósticos: cuando `person_id` del evento ≠ `person_id` de la visita (~19 %), el **98 %** está fechado **15–19 días antes** que la visita. Cuando la persona coincide, eso casi no ocurre.
- Mediciones: el `person_id` coincide con la visita solo en ~55 %; procedimientos, ~51 %.
- `visit_detail`: en visitas con 1 subproceso está alineado el 99.5 %; en visitas con ≥ 2 subprocesos, el **75–89 %** tiene subprocesos ~16 días antes de su visita.
- Ocurre de forma estable desde 2023 y desaparece en abril de 2026.
- **No se puede reparar**: solo el 12 % de los diagnósticos mal vinculados tiene una visita de su propia persona en ±48 h, y la diferencia de IDs no es constante.

**Consecuencia:** `visit_occurrence_id` no es una llave confiable para cruzar eventos. Hay que reportarlo a los organizadores del challenge.

## 5. Decisiones de diseño del M1 (propuestas)

1. **Eventos por persona y tiempo, no por visita.** Las variables de un paciente en el instante τ usan `event.person_id` y `event.datetime ≤ τ`, ignorando `visit_occurrence_id`. Esto respeta la regla anti-fuga y no depende del vínculo roto.
2. **Unidad de estancia = subproceso de hospitalización (`visit_detail` 9201).** LOS = `end − start` del subproceso. Se unen los subprocesos 9201 contiguos o solapados de una misma visita en un solo episodio.
3. **Persona del episodio confiable solo si el vínculo es coherente:** el subproceso no empieza más de 12 h antes de su visita. Los episodios no coherentes se excluyen del modelo de paciente, pero **sí cuentan para la ocupación**.
4. **Ocupación por servicio** desde todos los `visit_detail`. No necesita persona ni vínculo, solo inicio, fin y tipo. La capacidad (camas) es desconocida: se usa un proxy (P95 del censo histórico) mientras no tengamos el número real.
5. **Diagnósticos**: un dx está disponible en `condition_start_datetime`. Concepto 0 = 37 % de los dx (vocabulario `None`).
6. **Procedimientos**: `procedure_date` está vacío, así que **no se pueden ubicar en el tiempo** y quedan fuera de las variables anti-fuga.

## 6. Censo hospitalario y periodo estable

- **Calentamiento:** en enero de 2023 el censo está subestimado (~225 frente a ~265) porque faltan las estancias iniciadas antes de 2023. Se excluyen los primeros 30 días.
- **Cambio de patrón desde abril de 2026:** el censo salta (333–340), cae (223) y vuelve a saltar. Coincide con la desaparición del mal vínculo. Se excluye desde el 2026-04-01.
- **Crecimiento sostenido:** el censo de hospitalización pasa de ~265 (2023) a ~315 (2025). Con una capacidad fija calculada en entrenamiento, el 98.6 % de las horas de prueba quedaban "saturadas". Por eso la capacidad proxy es **móvil**: P95 del censo de los 90 días anteriores, que usa solo el pasado.
- **Ocupación hospitalaria alta:** el hospital opera en promedio al ~93 % de su P95 reciente. Con umbral 0.85 el target sale positivo el 90–95 % del tiempo. Se usa 0.95 para hospitalización (~38 % positivos) y 0.85 para urgencias (~27 %) hasta tener el número real de camas.
- **Pendiente:** pedir al HUN o a los organizadores el **número de camas por servicio**. Con ese dato la definición de saturación vuelve a la del enunciado (≥ 85 %).
