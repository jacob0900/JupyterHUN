# Módulo 5 · Feature Engineering (Persona 2)

Variables para la tarea de **paciente** (`snapshot.y_target_los_largo`: ¿la estancia superará el P75 de su servicio?) a partir del maestro de M1 (`data/m1/snapshot.parquet`, una fila por paciente en τ = ingreso + {0, 2, 6} h).

Antes de tocar nada, lee `CLAUDE.md` y `docs/M1_contrato_datos.md` en la raíz del proyecto: ahí están las reglas que mandan sobre cualquier prompt o PDF del plan original.

## Estructura

| Carpeta | Subfase | Estado |
|---|---|---|
| `m5_0_extraccion/` | Extracción de `measurement` y `condition_occurrence` desde el servidor (solo lectura) | Hecha (2026-11-14) |
| `m5_1_banderas_mnar/` | 5.1 · Clasificación de variables y banderas de nulos informativos (MNAR) | Hecha, con datos reales |
| `m5_2_svd_diagnosticos/` | 5.2 · Reducción de dimensionalidad de diagnósticos con `TruncatedSVD` | Hecha, con datos reales |
| `m5_3_temporal_ocupacion/` | 5.3 · Variables cíclicas temporales y saturación del servicio en t − 1 | Hecha, con datos reales (2026-11-15) |
| `m5_4_pipeline/` | 5.4 · Ensamblaje anti-fuga en `ColumnTransformer` + `Pipeline` (entrega a M6) | Hecha, con datos reales (2026-11-15) |

Cada subfase tiene la misma forma: un módulo con las funciones (`*.py`), un `run_m5_X.py` que lo corre sobre los datos reales y un `test_*.py` con datos sintéticos.

## Cómo correr

```bash
pip install -r requirements.txt -r modulo_5/requirements-m5.txt
python -m pytest modulo_5 -q                               # 106 pruebas, ~15 s
python modulo_5/m5_1_banderas_mnar/run_m5_1.py
python modulo_5/m5_2_svd_diagnosticos/run_m5_2.py
python modulo_5/m5_2_svd_diagnosticos/run_m5_2.py --demo   # SVD con datos sintéticos
python modulo_5/m5_3_temporal_ocupacion/run_m5_3.py        # ~5 s
python modulo_5/m5_4_pipeline/run_m5_4.py                  # CV 5 folds + prueba; --rapido para probar con 15 %
```

Orden de dependencias: 5.1, 5.2 y 5.3 son independientes entre sí; la 5.4 lee las salidas de las tres.

Los `run_*.py` se corren desde la raíz del proyecto (`Hospital/`). La extracción solo hace falta si se borra `data/raw/`; necesita la VPN FortiClient y un `.env` (copiar `.env.ejemplo`):

```bash
python modulo_5/m5_0_extraccion/extraer_tablas.py probar
python modulo_5/m5_0_extraccion/extraer_tablas.py extraer   # lento por la VPN (~14 000 filas/s)
```

## Subfase 5.1 · Banderas MNAR

`m5_1_banderas_mnar/banderas_mnar.py` (solo pandas y numpy).

- **Clasifica las columnas** en tres listas disjuntas: `num_cols` (12), `cat_cols` (8) y `lab_cols` (24). Deja fuera llaves (`snapshot_id`, `episodio_id`, `person_id`), targets `y_*`, control (`es_entrenamiento`, `en_periodo_estable`, `en_riesgo_truncamiento`, `umbral_los_h`) y tiempos crudos.
- **Banderas estructurales** (recomendación de M2): `sin_mediciones_24h`, `sin_mediciones_historia`, `sin_dx_historia`, `nunca_hospitalizado`. Polaridad 1 = ausencia.
- **`has_measurement_X`** para los 20 laboratorios más frecuentes: 1 si la persona tiene un registro del examen en [τ − 24 h, τ).
- **Sinónimos**: en el HUN cada examen viene con dos conceptos (LOINC + SNOMED) sobre los mismos registros. Si dos conceptos se solapan con Jaccard ≥ 0,95 en (persona, hora), se fusionan en una sola bandera. Se fusionaron 10 pares; la lista está en la columna `sinonimos` de `labs_top_k.csv`.

Salidas: `data/m5/snapshot_m5_1.parquet` (maestro + 24 banderas), `outputs/m5/esquema_m5_1.json` (listas de columnas para las subfases siguientes) y en `outputs/m5/tablas/`: `labs_top_k.csv`, `esquema_columnas.csv`, `prevalencia_banderas.csv`.

## Subfase 5.2 · SVD de diagnósticos

`m5_2_svd_diagnosticos/svd_diagnosticos.py`.

- `construir_matriz_dispersa`: matriz binaria `csr_matrix` de **snapshots × códigos** (1 si la persona tenía el diagnóstico antes de τ). Es una regla fija: no aprende nada.
- `ReductorSVDClinico` (`BaseEstimator`, `TransformerMixin`): en `fit` aprende qué códigos conservar (`min_df=5`), los pesos TF-IDF y `TruncatedSVD(n_components=50, random_state=42)`. Pasa la batería oficial `check_estimator` de scikit-learn.
- Procedimientos fuera: `procedure_date` está vacío en el HUN. El código los aceptaría si tuvieran fecha.

Salidas en `data/m5/`: `matriz_dx_m5_2.npz` + `vocabulario_dx_m5_2.csv` (para reajustar la SVD por fold en M6), `svd_dx_m5_2.parquet` (`snapshot_id` + `svd_dx_00..49`), `reductor_svd_m5_2.joblib`. En `outputs/m5/tablas/`: `svd_varianza.csv`, `svd_cargas_top10.csv`.

## Subfase 5.3 · Variables cíclicas y saturación en t − 1

`m5_3_temporal_ocupacion/temporal_ocupacion.py` (solo pandas y numpy). Función principal: `build_temporal_and_occupancy_features(df)`.

- **Cíclicas**: `hour_sin`, `hour_cos` (2π·h/24) y `dow_sin`, `dow_cos` (2π·d/7, lunes = 0). En snapshots la hora lleva los minutos.
- **Banderas**: `es_hora_pico_urgencias` (07–10 h y 17–20 h, como pedía el prompt; configurable en `ConfigTemporal.horas_pico`), `es_fin_de_semana`, `es_festivo`, `es_dia_no_habil`, `tras_festivo`, `primer_dia_habil` y `turno` (categórica).
  - Festivos: columna `festivo` de `data/m4/calendario_dia.parquet` (Persona 1); si falta, `holidays.Colombia`.
  - **No** se usa `calendario_dia.dia_despues_festivo`, porque también marca los lunes normales (docs/P3_entrega_eda.md). Se separan `tras_festivo` (tras un festivo real) y `primer_dia_habil` (tras cualquier día no hábil).
- **Ocupación del servicio en t − 1** (tabla `ocupacion_servicio_hora`): tasa activa = censo / capacidad, en una **rejilla horaria completa** por servicio con `.shift(1)` **antes** de toda ventana. Variables: `ocupacion_t1`, `ocupacion_media_{3,6,12,24}h`, `ocupacion_max_24h`, `ocupacion_std_24h`, `ocupacion_tendencia_3h`, `frac_horas_sobre_umbral_24h`, `ocupacion_semana_previa` (t − 168 h), `censo_t1`, `llegadas_t1`, `llegadas_suma_{4,24}h`.
- **Snapshots de paciente**: `unir_ocupacion_a_snapshots` les une las variables de su servicio desde la fila floor(τ), que solo ve horas ≤ floor(τ) − 1 h < τ (prefijo `srv_`).
- **Verificación anti-fuga** (`verificar_sin_fuga`): altera todo lo observado en t ≥ corte y baraja las etiquetas `y_*`. Las variables de las filas ≤ corte deben quedar idénticas. Se corre en las pruebas (con un control positivo que confirma que detecta un `shift` olvidado) y en `run_m5_3.py`, sobre 4 cortes reales.
- No se aprende nada de los datos: no hace falta reajustar por fold. Las ventanas usan horas previas al periodo estable (es pasado, no fuga). Hay que filtrar `en_periodo_estable` **después**.

Salidas: `data/m5/servicio_hora_m5_3.parquet` (serie + 25 variables), `data/m5/snapshot_m5_3.parquet` (`snapshot_id` + temporales + `srv_*`), `outputs/m5/esquema_m5_3.json` y en `outputs/m5/tablas/`: `m5_3_resumen.csv`, `m5_3_auc_saturacion.csv`, `m5_3_verificacion_fuga.csv`.

## Subfase 5.4 · Ensamblaje anti-fuga (ColumnTransformer + Pipeline)

`m5_4_pipeline/pipeline_antifuga.py`. `construir_pipeline(esquema, estimador)` devuelve un `Pipeline` **sin ajustar**: `preprocesador` (`ColumnTransformer`) + `modelo`.

| Rama | Columnas (63 de entrada) | Pasos |
|---|---|---|
| `num` (18) | edad, horas desde ingreso / última alta, ocupación t − 1 (M1 y `srv_*` de 5.3), cíclicas | `SimpleImputer(median)` → `StandardScaler` |
| `num_log` (10) | conteos asimétricos: dx, visitas, hospitalizaciones, mediciones, llegadas | `SimpleImputer(median)` → `log1p` → `StandardScaler` |
| `cat` (34) | servicio, sexo, origen, mes, turno, banderas de calendario (5.3), banderas MNAR y `has_measurement_*` (5.1) | a texto → `SimpleImputer(constant, "Desconocido")` → `OneHotEncoder(drop="first", handle_unknown="ignore")` |
| `clinica` (1) | `dx_codigos`: los códigos que la persona tenía antes de τ, como texto | `CountVectorizer(binary)` → `ReductorSVDClinico` (5.2: `min_df=5` + TF-IDF + `TruncatedSVD(n_components=50)`) |

- **Estimador de prueba**: `LogisticRegression(class_weight="balanced")`. En M6 se cambia con `pipe.set_params(modelo=...)`; LightGBM no está instalado en este entorno.
- **Ajustes al prompt, y por qué**:
  - La rama numérica se parte en `num` y `num_log` porque los conteos tienen asimetría de 8 a 23 (M2, sección 8). `log1p` es una regla fija.
  - La rama clínica recibe los códigos como texto, no la matriz precalculada. Así el mismo objeto sirve en la CV de M6 y en la API de M8: un paciente nuevo solo manda sus códigos previos a τ.
- **No se duplican variables**: la 5.3 calculó `srv_ocupacion_t1`, `srv_censo_t1` y `srv_llegadas_t1`, pero son las mismas de M1 (Spearman 1,00). Entran solo las de M1. `hora_del_dia` y `dia_semana` se sustituyen por las cíclicas, y `fin_de_semana` por `es_fin_de_semana`.
- **Garantía anti-fuga, comprobada en `verificar_pipeline_sin_fuga`** (pruebas y fold 1 real):
  1. Medianas, medias, desviaciones, categorías y vocabulario de códigos aprendidos = los calculados solo con `X_train`.
  2. `predict` en validación no cambia ningún parámetro (hash de todos los atributos aprendidos).
  3. Destrozar la validación (extremos, nulos, categorías y códigos nuevos) no cambia el modelo y no da error.
  4. Cada fila se predice igual sola que dentro del lote.
  5. Control positivo: un preprocesador ajustado con train + validación **sí** se detecta.
- `preparar_X(raiz)` arma la tabla de entrada uniendo M1 + 5.1 + 5.3 + códigos de la 5.2, y verifica que las filas estén alineadas por `snapshot_id`.

Salidas: `data/m5/pipeline_m5_4_plantilla.joblib` (sin ajustar, **es lo que usa M6**) y `data/m5/pipeline_m5_4_ajustado.joblib` (baseline ajustado con todo el entrenamiento), `outputs/m5/esquema_m5_4.json` y en `outputs/m5/tablas/`: `m5_4_cv.csv`, `m5_4_prueba.csv`, `m5_4_verificacion_fuga.csv`, `m5_4_variables_salida.csv`.

Uso en M6:

```python
import joblib
from modulo_5.m5_4_pipeline import preparar_X
X, esquema = preparar_X(".")                       # filtrar en_periodo_estable y partir por es_entrenamiento
pipe = joblib.load("data/m5/pipeline_m5_4_plantilla.joblib")
pipe.set_params(modelo=MiModelo())                 # o modelo__C=..., preprocesador__clinica__svd__n_components=...
# cross_validate(pipe, X[esquema.todas()], y, groups=X.person_id, cv=StratifiedGroupKFold(5))
```

## Reglas anti-fuga que siguen todas las subfases

1. Solo eventos de la **misma `person_id`** con fecha efectiva **`t < τ`** (estricto). Nunca se cruza por `visit_occurrence_id`: el vínculo está roto (19 % de diagnósticos y 43 % de mediciones apuntan a otra persona).
2. Si un evento solo tiene fecha, o la hora es 00:00:00, se asume conocido al **final de ese día**.
3. Todo lo que se "aprende" (qué laboratorios tienen bandera, qué códigos entran a la SVD, IDF, ejes de la SVD) se ajusta **solo con entrenamiento** (`es_entrenamiento & en_periodo_estable`).
4. En la validación cruzada de M6, la SVD va **dentro del `Pipeline`** usando `matriz_dx_m5_2.npz`, para que se reajuste en cada fold. `svd_dx_m5_2.parquet` solo vale para el ajuste final entrenamiento → prueba.

## Cifras de control (para verificar que todo corre igual)

| Qué | Valor |
|---|---|
| Filas extraídas | `measurement` 16 575 888 · `condition_occurrence` 38 636 063 |
| Snapshots | 766 531 (541 260 de entrenamiento estable) |
| `sin_mediciones_24h` (entrenamiento) | hospitalización 41,2 / 36,3 / 37,0 % · urgencias 96,9 / 76,0 / 34,9 % en τ = 0 / 2 / 6 h |
| Laboratorio más frecuente | Hemograma: 90,3 % de las personas con algún laboratorio |
| Matriz de diagnósticos | 766 531 × 3 625 códigos, 1 217 691 no-ceros (0,044 %) |
| Diagnósticos por snapshot | Coinciden en el 100 % con `n_dx_distintos_historia` de M1 |
| Varianza explicada SVD | 21,5 % con 10 comp. · 39,2 % con 30 · **48,3 % con 50** (2 629 códigos útiles) |
| 5.3 · serie de servicio | 62 102 filas (31 051 horas × 2 servicios, sin huecos) · 11 temporales + 14 de ocupación t − 1 |
| 5.3 · AUC vs `y_target_saturacion` (entrenamiento) | `ocupacion_t1` 0,863 hosp. / 0,775 urg. (la ocupación del instante t daría 0,892 / 0,835, pero está prohibida) · urgencias: `llegadas_suma_4h` 0,775, `hour_cos` 0,217 (ciclo diario) |
| 5.3 · contraste con M1 | `srv_ocupacion_t1` vs `ocupacion_servicio_t_menos_1h`: Spearman 1,00 (el censo difiere en 1 porque M1 excluye al propio paciente) |
| 5.3 · verificación anti-fuga | OK en los 4 cortes reales (2023-08, 2024-06, 2025-07 = corte de entrenamiento, 2026-01) |
| 5.4 · tamaños | entrenamiento 541 260 · prueba 148 673 snapshots · 63 columnas de entrada → 124 tras el preprocesador |
| 5.4 · verificación anti-fuga (fold 1 real) | 8/8 OK: medianas, medias, desviaciones, categorías, vocabulario dx, `predict` sin efectos, modelo independiente de validación y filas independientes del lote |
| 5.4 · baseline (regresión logística) | CV AUC-ROC 0,715 ± 0,003 · prueba 0,701. **Ojo:** el AUC global está inflado por mezclar servicios y τ con prevalencias distintas. Dentro de cada estrato: hosp. 0,62–0,63, urg. 0,58. Es el piso que M6 debe superar, y M6 debe reportar por servicio × τ |
| 5.4 · tiempo | ~6 min la corrida completa (5 folds + verificación + ajuste final) |

## Notas de entorno

- **scikit-learn ≥ 1.6 / SciPy `csr_array`** (corregido 2026-11-15): con scikit-learn 1.9.1 fallaban dos pruebas de `check_estimator` de la 5.2. `ReductorSVDClinico` ahora declara que acepta matrices dispersas (`__sklearn_tags__` y `_more_tags` para versiones viejas) y cuenta el soporte con `(X != 0).sum(axis=0)` en vez de `.getnnz()`, que no existe en `csr_array`. Sobre la matriz real el soporte es idéntico, así que las salidas de la 5.2 no cambian. 93 pruebas pasan.

- El PC original tiene 7,7 GB de RAM. Por eso `run_m5_2.py` reduce `condition_occurrence` con DuckDB (primera aparición por persona y código) antes de pasar a pandas, y la matriz se arma por bloques.
- `data/` y `.env` están en `.gitignore`: tienen datos de pacientes y credenciales. Los informes en `outputs/` son solo agregados.
