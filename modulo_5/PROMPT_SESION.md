# Prompt para continuar el Módulo 5 en otra sesión

Copia todo lo que está dentro del bloque y pégalo como primer mensaje en la nueva sesión de Claude, abierta sobre la carpeta `Hospital`.

```text
Vas a continuar el Módulo 5 (Feature Engineering) del Data Challenge del Hospital Universidad del Norte (HUN). Responde siempre en español.

## Quién soy y qué hacemos
Soy Persona 2 del equipo: me toca M5 (features), M6 (modelos y validación cruzada) y M7 (calibración, pruebas, SHAP). Trabajamos en la carpeta `Hospital` (esta). La base es OMOP CDM en PostgreSQL (`uhn_i`, esquema `uninorte_omop`), pero los datos ya están extraídos en `data/`: NO hace falta conectarse al servidor ni a la VPN.

Hay dos tareas de predicción:
1. Paciente: ¿la estancia superará el P75 de su servicio? (`data/m1/snapshot.parquet`, columna `y_target_los_largo`; una fila por paciente en τ = ingreso + 0, 2 y 6 h).
2. Servicio: ¿la ocupación llegará al umbral en t+4 h? (`data/m1/ocupacion_servicio_hora.parquet`, `y_target_saturacion`).

## Lee esto antes de hacer nada, en este orden
1. `CLAUDE.md` (raíz): reglas no negociables del proyecto. Cuando un prompt mío o el PDF del plan original contradiga este archivo, manda `CLAUDE.md`; dime qué cambiaste y por qué.
2. `docs/M1_contrato_datos.md` y `docs/M1_hallazgos_datos.md`: qué tiene cada tabla y qué está roto en los datos.
3. `docs/P3_entrega_eda.md`: hallazgos del EDA (M2–M4) con recomendaciones para M5–M7.
4. `modulo_5/README.md`: lo que ya está hecho en el Módulo 5, cifras de control y decisiones.

## Estado actual
- Subfase 5.1 (banderas MNAR) y 5.2 (TruncatedSVD sobre diagnósticos): hechas y corridas con datos reales. Las 57 pruebas de `python -m pytest modulo_5 -q` deben pasar.
- `data/raw/` ya tiene `measurement` (16,6 M filas) y `condition_occurrence` (38,6 M). No vuelvas a extraer.
- Si dudas de que todo esté bien, corre las pruebas y compara con la tabla "Cifras de control" del README del módulo.

## Reglas que debes respetar (además de CLAUDE.md)
- Anti-fuga: solo eventos de la misma `person_id` con fecha efectiva `t < τ` (estricto). Nunca cruces eventos por `visit_occurrence_id` (está roto). Si un evento solo tiene fecha u hora 00:00:00, se asume conocido al final del día.
- Todo lo que se aprenda de los datos (selección de variables, imputación, escalado, SVD, umbrales) se ajusta solo con entrenamiento (`es_entrenamiento & en_periodo_estable`), y en la CV va dentro de un `Pipeline` para que se reajuste en cada fold. CV con `StratifiedGroupKFold` por `person_id`.
- Las columnas `y_*` nunca son variables de entrada.
- Procedimientos no se usan (`procedure_date` está vacío).
- Datos de pacientes: no los subas a ningún servicio externo, no imprimas filas individuales; en `outputs/` solo van agregados. No muestres ni copies el contenido de `.env` si existe.
- Este tipo de PC puede tener poca RAM (el original tiene 7,7 GB): para tablas grandes usa DuckDB o lectura por columnas antes de pandas. Si el PC tiene más RAM, igual mantén el código eficiente.

## Forma de trabajo
- Cada subfase nueva va en su carpeta `modulo_5/m5_N_nombre/` con: módulo de funciones documentado (docstrings con la lógica clínica y la garantía anti-fuga), `run_m5_N.py` que corre sobre los datos reales desde la raíz del proyecto, y `test_*.py` con datos sintéticos.
- Después de cada subfase: corre todas las pruebas de `modulo_5`, corre el `run` con datos reales, revisa que las cifras tengan sentido clínico (y contrástalas con M1/M2 cuando se pueda) y actualiza `modulo_5/README.md` (estado, salidas y cifras de control).
- No hagas commit ni push sin que te lo pida.

Te voy a ir pasando el prompt de cada subfase (la siguiente es la 5.3). Por ahora, lee los archivos indicados, corre `python -m pytest modulo_5 -q` y resúmeme en pocas líneas qué entendiste del proyecto y del estado del Módulo 5.
```
