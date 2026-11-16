# CLAUDE.md — Data Challenge HUN (Hospital Universidad del Norte)

Contexto para Claude. Responde en **español**. Equipo de 3 personas; cada una puede trabajar con Claude sobre esta misma carpeta.

## El proyecto

Predecir el flujo hospitalario del HUN con datos OMOP CDM (PostgreSQL `uhn_i`, esquema `uninorte_omop`, vía VPN). Dos tareas:
1. **Paciente**: ¿la estancia superará el P75 de su servicio? (`snapshot.y_target_los_largo`)
2. **Servicio**: ¿la ocupación llegará al umbral en t+4 h? (`ocupacion_servicio_hora.y_target_saturacion`)

| Persona | Módulos | Rol |
|---|---|---|
| Persona 1 | M1 ETL y targets, M4 SQL, M8 API/Docker/dashboard | SQL & MLOps (ya entregó M1 y M4-SQL) |
| Persona 2 | M5 features, M6 modelos y CV, M7 calibración/pruebas/SHAP | ML |
| Persona 3 | M2 EDA univariado y censura, M3 bivariado y colas, M4 visual (heatmaps, Sankey) | EDA |

Los PDFs de `referencias/` (`modulos_hospital_hun_contexto_claude.pdf`, `malla_curricular_hospital_hun.pdf`, `Evaluacion_estrategica_Arquitectura_ML_Data_Challenge_HUN.pdf`) describen el plan **original**. **Cuando el PDF contradiga este archivo, manda este archivo**: el plan se escribió antes de ver los datos reales.

## Lo que el PDF supone y NO es cierto en los datos

| El PDF dice | La realidad | Qué hacer |
|---|---|---|
| Kaplan-Meier por **nivel de triaje (1–5)** | **No existe triaje** en ninguna tabla | Estratificar por `servicio`, `origen_urgencias`, grupo de `edad`, `sexo` |
| Estancia = `visit_end − visit_start` | En el 100 % de las visitas fin = inicio (LOS 0) | La estancia está en `visit_detail` → usar `episodio.los_h` |
| Visitas abiertas (`visit_end IS NULL`) = censura | No hay ninguna visita abierta | La censura real es **truncamiento al final**: `en_riesgo_truncamiento` |
| Muchos servicios (UCI, salas, etc.) | Solo **4**: `ambulatorio`, `urgencias_ingreso`, `urgencias_observacion`, `hospitalizacion` | Analizar a ese nivel |
| `visit_occurrence_id` une eventos con su visita | **Roto** en una parte: 29 % de subprocesos, 19 % de diagnósticos, 43 % de mediciones apuntan a la visita de otra persona (~16 días después) | Nunca cruzar eventos por `visit_occurrence_id`; ya viene filtrado (`vinculo_coherente`) |
| Procedimientos como variable | `procedure_date` vacío | No usar procedimientos |
| Tiempo de espera por transición (colas) | Fin de un subproceso = inicio del siguiente (espera ≈ 0) | Medir cuello de botella por **duración y ocupación**, no por espera |
| Capacidad c (camas) para ρ = λ/(cμ) | No se conoce el número de camas | `capacidad` = proxy: P95 del censo de los 90 días previos |

Detalle completo y evidencia: `docs/M1_hallazgos_datos.md`.

## Reglas no negociables

1. **El servidor es solo lectura.** Nunca DDL/INSERT. Solo Persona 1 extrae; los demás trabajan con los parquets.
2. **Columnas `y_*` = futuro/etiqueta.** Nunca como variable de entrada ni para filtrar antes de modelar.
3. **Filtrar `en_periodo_estable = true`** (2023-01-31 → 2026-04-01) en análisis y modelos. Fuera de ese rango la carga de datos cambia (censo subestimado al inicio; saltos y caídas desde abril 2026).
4. **Partición**: `es_entrenamiento` (corte temporal en `parametros.fin_entrenamiento` = 2025-07-13). CV con `StratifiedGroupKFold` por **`person_id`** (paciente) o `TimeSeriesSplit` con gap ≥ 4 h (servicio). Nunca CV estratificada simple.
5. **Variables nuevas**: solo eventos de la misma `person_id` con `datetime < snapshot_time`. Escalado/imputación dentro de `Pipeline`.
6. **n es grande** (cientos de miles): todo p-valor sale < 0.001. Reportar siempre **tamaño de efecto** (Cramér's V, ε², diferencia de medianas con IC) y corrección **Benjamini-Hochberg**.
7. **Datos de pacientes**: no subir `data/` a git ni a servicios externos; no imprimir filas individuales en informes, solo agregados.

## Mapa de archivos

Leer con `pandas.read_parquet` o `duckdb` (`SELECT … FROM 'data/m1/episodio.parquet'`). Para `detalle` (3.3 M filas) y `transicion` (3.9 M) conviene DuckDB.

**`data/m1/`**: dataset maestro (contrato completo en `docs/M1_contrato_datos.md`)
- `episodio`: una fila por estancia. `servicio` (hospitalizacion | urgencias_observacion), `person_id`, `inicio`, `fin`, `los_h`, `edad`, `sexo` (M/F/desconocido), `origen_urgencias`, `n_subprocesos`, `umbral_los_h`, `target_los_largo`, `es_entrenamiento`, `en_periodo_estable`, `en_riesgo_truncamiento`
- `snapshot`: una fila por paciente en τ = ingreso + {0, 2, 6} h. Variables conocidas en τ + etiquetas `y_*`
- `ocupacion_servicio_hora`: servicio × hora: `censo`, `llegadas`, `capacidad`, `ocupacion`, `umbral_ocupacion`, `y_target_saturacion`, `y_ocupacion_futura`
- `detalle`: subprocesos `visit_detail` con `servicio`, `dur_h`, `vinculo_coherente`
- `llegadas_servicio_hora`, `umbral_los`, `capacidad_servicio`, `parametros`, `reporte_calidad`

**`data/m4/`**: espacio-temporal (guía de uso en `docs/M4_entrega_persona3.md`)
- `heatmap_llegadas`, `heatmap_ocupacion` (servicio × dia_semana 1–7 × hora_del_dia 0–23)
- `sankey_aristas` (`origen`, `destino`, `n`; nodos `"etapa:servicio"`), `rutas_frecuentes`, `matriz_transicion`, `transicion`
- `llegadas_dia`, `ocupacion_dia` (series diarias con festivos), `calendario_dia`, `calendario_hora` (festivos Colombia, puentes, turno mañana 7–13 / tarde 13–19 / noche 19–7)
- `resumen_colas` (λ, E[S], μ, L, c, ρ por servicio)

**`data/raw/`**: `person` (`gender_concept_id` 8507 = M, 8532 = F), `concept` (nombres de conceptos usados)

## Particularidades de cada servicio

| Servicio | Qué es | Duración |
|---|---|---|
| `ambulatorio` | Consulta externa (horario 7–17 h) | **0 h siempre** (registro puntual): no tiene estancia ni censo |
| `urgencias_ingreso` | Registro de llegada a urgencias | ~3 min (puntual) |
| `urgencias_observacion` | Estancia en urgencias / observación (24 h) | p50 1.2 h, P75 umbral 8.9 h; 38 % dura 0 |
| `hospitalizacion` | Hospitalización | p50 ~46–71 h, P75 umbral 141.6 h |

## Cifras de referencia (para verificar que lees bien los datos)

- Episodios: hospitalización 49 423 (36 723 personas), urgencias/observación 493 320; ~26 % positivos en ambos.
- Snapshots: 718 k. Hospitalización 0 h: 49 416.
- Ley de Little cuadra: censo medio hospitalización 289.5 ≈ λ·W 288.7; urgencias 224.0 ≈ 223.6.
- Ocupación media (vs proxy): hospitalización 0.93, urgencias 0.74. Umbral de saturación: 0.95 hospitalización (provisional, sin camas reales), 0.85 urgencias.
- Rutas: 65 % ambulatorio → egreso; 17 % urgencias_ingreso → observación → egreso; 1.8 % urgencias → hospitalización.
- Efecto festivo en urgencias/observación: laboral ~999/día, fin de semana ~763, festivo ~758, día después de festivo ~1 067.
- Censo hospitalario con tendencia: ~265 (2023) → ~315 (2025). Usar STL / desestacionalizar antes de comparar periodos.

## Código existente (de Persona 1)

- `m1_etl/run_etl.py` (`probar | explorar | extraer | construir`) + `m1_etl/sql/` (servidor, solo lectura) + `m1_etl/sql_local/` (DuckDB). Requiere VPN y `.env` solo para `probar/explorar/extraer`.
- `m4_espaciotemporal/run_m4.py` (`--paquete` arma el zip de entrega).
- Validaciones: `outputs/m1_validaciones.md`, `outputs/m4_resumen.md`.
- No modificar estos scripts sin avisar a Persona 1: otros módulos dependen de sus columnas. Si falta una columna o tabla, pedirla a Persona 1 o crear un script propio en una carpeta del módulo (p. ej. `m2_eda/`, `m3_bivariado/`).

## Código de Persona 3 (EDA)

- `m2_eda/run_m2.py`, `m3_bivariado/run_m3.py`, `m4_visual/run_m4_visual.py` + utilidades en `eda_comun/`. Dependencias extra: `requirements-eda.txt`.
- Informes en `outputs/m2/`, `outputs/m3/`, `outputs/m4_visual/` (los HTML del dashboard en `outputs/m4_visual/html/`). Hallazgos y pendientes para P1/P2: `docs/P3_entrega_eda.md`.
- Ojo: `calendario_dia.dia_despues_festivo` marca el primer día hábil tras cualquier día no hábil (incluye lunes normales); para "tras festivo real" usar `festivo` del día anterior.

## Módulo 5 (feature engineering)

- Código en `modulo_5/` (subfases 5.0 extracción, 5.1 banderas MNAR, 5.2 SVD de diagnósticos, 5.3 cíclicas + ocupación t − 1, 5.4 Pipeline anti-fuga para M6). Leer `modulo_5/README.md` (estado, salidas y cifras de control) antes de seguir; `modulo_5/PROMPT_SESION.md` sirve para retomar en otra sesión.
- Salidas a nivel paciente en `data/m5/`; agregados en `outputs/m5/`. Pruebas: `python -m pytest modulo_5 -q`.
- `.env` tiene credenciales del servidor: nunca mostrar su contenido ni subirlo (está en `.gitignore`).

## Módulo 6 (modelos y validación cruzada)

- Código en `modulo_6/` (6.1 CV agrupada por `person_id`, 6.2 batería de modelos `model_factory.py`, 6.3 Optuna → `outputs/m6/best_hyperparameters.json`, 6.4 diagnóstico y empaquetado → `models/*.joblib` + OOF en `data/m6/oof_m6_4.parquet`). Dependencias: `modulo_6/requirements-m6.txt`. Leer `modulo_6/README.md`.
- Todas las comparaciones de modelos deben usar los pliegues de `data/m6/pliegues_m6_1.parquet` (`PlieguesPredefinidos`), calculados solo sobre entrenamiento estable.
- Hiperparámetros definitivos: `outputs/m6/hiperparametros_finales.json` (RF regularizado en la auditoría de brecha). Ningún modelo tiene brecha train − val > 0,05; el train de KNN se mide leave-one-patient-out.

## Módulo 7 (evaluación, calibración y pruebas)

- Código en `modulo_7/` (7.1 multimétrica OOF con IC bootstrap · 7.2 calibración y umbral de costo · 7.3 SHAP · 7.4 pruebas de hipótesis). Leer `modulo_7/README.md`.
- A nivel de snapshot la clase positiva es **mayoría** (53 %); el desbalance real está en hospitalización (25 %). Reportar métricas por servicio × τ, no solo globales.
- Bootstrap siempre por **paciente** (no por fila) y pareado entre modelos: réplicas en `outputs/m7/tablas/m7_1_bootstrap_replicas.csv`. OOF en probabilidad (SVM con Platt cruzado) en `data/m7/probabilidades_oof_m7_1.parquet`.
- 7.2: calibración isotónica (gana en los 3 mejores; ECE 0,001). θ* ≈ 0,16–0,17, igual al umbral de Bayes 1/6 para c_FN = 5. Umbrales **congelados** en `outputs/m7/m7_2_umbrales_congelados.json`; la prueba temporal se evalúa **una sola vez** (sello `outputs/m7/m7_2_prueba_unica.json`; no repetir). Modelos desplegables: `models/<modelo>_calibrado.joblib`. Con 5:1 el modelo solo cambia decisiones en hospitalización (en urgencias, alertar a todos es casi óptimo). Prueba temporal (ya evaluada, no repetir): AUC LightGBM 0,729 (OOF 0,742); en hospitalización θ* ahorra 15 % frente a alertar a todos; la prevalencia sube (0,578) y el modelo subestima ~2,7 pp → recalibrar en M8.
- 7.3: SHAP (TreeExplainer, LightGBM sin el pliegue 1) agrupado por variable clínica; figuras en `reports/figures/`, síntesis en `reports/m7_3_sintesis_medica.md`. "Servicio" y τ dominan por la definición del objetivo: leer los rankings por servicio.
- 7.4: DeLong **por conglomerados de paciente** (el clásico subestima el SE ×1,5), BCa pareado, Friedman + Nemenyi, Wilcoxon (con 5 pliegues el p mínimo es 0,0625; se decide con 50 particiones de pacientes). LightGBM supera a los 6; frente a RF, ΔAUC = 0,005 (significativo pero < 0,01 de relevancia).

## Organización de la carpeta

| Carpeta | Contenido |
|---|---|
| `data/` | Parquets de Persona 1 (datos de pacientes: no compartir fuera del equipo) |
| `docs/` | Contratos, hallazgos y guías de entrega entre personas |
| `m2_eda/`, `m3_bivariado/`, `m4_visual/`, `eda_comun/` | Código de Persona 3 (deben quedar en la raíz: usan `data/` y `outputs/` relativos a ella) |
| `modulo_5/` | Feature engineering (subfases 5.0–5.4) |
| `modulo_6/` | Modelos y validación cruzada (6.1 en adelante) |
| `modulo_7/` | Evaluación, calibración y pruebas estadísticas (7.1 en adelante) |
| `models/` | Pipelines finales `.joblib` + `manifest.json` (6.4). `knn_pipeline.joblib` contiene filas de pacientes |
| `outputs/` | Resultados generados (validaciones de P1, informes de P3 y tablas de M5, M6 y M7) |
| `reports/` | Figuras SHAP a 300 DPI (`reports/figures/`) y síntesis clínica de la 7.3 |
| `referencias/` | PDFs del enunciado y del plan original |
| `entregas/recibido_de_persona1/` | Paquete original de Persona 1 (`entrega_persona3.zip`, con datos) |
| `entregas/recibido_de_persona2/` | Zip recibido con el Módulo 5 (`Hospital.zip`, con datos y `.env`) |
| `entregas/persona3/` | Zips entregados por Persona 3: `entrega_persona3_eda.zip` (sin datos) y `entrega_persona3_completa_con_datos.zip` (con datos, solo canal del equipo) + su `LEEME_persona3.md` |
