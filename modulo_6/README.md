# Módulo 6 · Modelado supervisado y validación cruzada

Modelos para la tarea de **paciente** (`y_target_los_largo`) sobre el maestro del Módulo 5 (`modulo_5/m5_4_pipeline/preparar_X`). Antes de empezar, lee `CLAUDE.md` y `modulo_5/README.md`.

## Estructura

| Carpeta | Subfase | Estado |
|---|---|---|
| `m6_1_cv_agrupada/` | 6.1 · Validación cruzada agrupada por paciente (`StratifiedGroupKFold`) | Hecha, con datos reales (2026-11-15) |
| `m6_2_bateria_modelos/` | 6.2 · Batería de 10 clasificadores + espacios de búsqueda (`model_factory.py`) | Hecha; prueba de humo con datos reales (2026-11-15) |
| `m6_3_optuna/` | 6.3 · Optimización bayesiana con Optuna (TPE + MedianPruner) | Hecha, con datos reales (2026-11-15) |
| `m6_4_diagnostico/` | 6.4 · Brecha de sobreajuste, curvas de aprendizaje, ROC y empaquetado `.joblib` | Hecha, con datos reales (2026-11-15) |

Cada subfase sigue la forma del M5: módulo con las funciones, `run_m6_X.py` para los datos reales (se corre desde la raíz) y `test_*.py` con datos sintéticos.

```bash
pip install -r modulo_6/requirements-m6.txt         # lightgbm, xgboost, optuna
python -m pytest modulo_6 -q                        # 78 pruebas, ~40 s
python modulo_6/m6_1_cv_agrupada/run_m6_1.py        # ~40 s
python modulo_6/m6_2_bateria_modelos/run_m6_2.py    # prueba de humo de la batería, ~1,5 min
python modulo_6/m6_3_optuna/run_m6_3.py             # búsqueda Optuna, ~2–3 h (reanudable)
python modulo_6/m6_4_diagnostico/run_m6_4.py        # brecha + curvas + models/*.joblib, ~1–2 h
python modulo_6/m6_4_diagnostico/run_auditoria_m6_4.py          # auditoría: KNN LOO + RF regularizado, ~2,5 h
python modulo_6/m6_4_diagnostico/run_auditoria_knn_paciente.py  # auditoría: KNN leave-one-patient-out, ~1 h
```

## Subfase 6.1 · CV agrupada por paciente

`m6_1_cv_agrupada/cv_agrupada.py` (solo pandas y scikit-learn).

- `crear_cv()`: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, con grupos = `person_id` y estratificación por `y_target_los_largo`. Es la misma configuración de la 5.4.
- `verify_group_isolation(X, y, groups, cv)`: recorre los 5 pliegues y lanza **`ValueError`** si algún `person_id` está en entrenamiento y validación a la vez, o si los pliegues de validación no cubren cada fila exactamente una vez. Devuelve el reporte por pliegue: muestras, pacientes únicos y proporción positiva en train y val. Se usa `ValueError` explícito porque un `assert` se desactiva con `python -O`; el `assert` queda solo como documentación.
- `asignar_pliegues()` + `PlieguesPredefinidos`: guardan y reproducen los pliegues verificados. **Todo M6 debe usar estos mismos pliegues** (`cv=PlieguesPredefinidos(pliegue)`), para que las comparaciones entre modelos (y las pruebas de hipótesis de M7) sean pareadas.
- La CV va **solo sobre entrenamiento estable** (`es_entrenamiento & en_periodo_estable`). La prueba temporal (148 673 snapshots) queda apartada para M7.
- Control positivo en las pruebas: `KFold` y `StratifiedKFold`, que ignoran al paciente, provocan el `ValueError`.

Salidas: `data/m6/pliegues_m6_1.parquet` (`snapshot_id`, `person_id`, `pliegue`; nivel paciente), y en `outputs/m6/tablas/`: `m6_1_reporte_pliegues.csv` y `m6_1_prevalencia_servicio_tau.csv`.

Uso en las siguientes subfases:

```python
import pandas as pd
from modulo_5.m5_4_pipeline import preparar_X
from modulo_6.m6_1_cv_agrupada import PlieguesPredefinidos

X, esquema = preparar_X(".")
tr = X[X.en_periodo_estable & X.es_entrenamiento].reset_index(drop=True)
pl = pd.read_parquet("data/m6/pliegues_m6_1.parquet")
assert pl.snapshot_id.equals(tr.snapshot_id)          # mismo orden de filas
cv = PlieguesPredefinidos(pl.pliegue)
```

## Subfase 6.2 · Batería de modelos y espacios de búsqueda

`m6_2_bateria_modelos/model_factory.py`. `get_pipeline_and_space(model_name, preprocessor)` devuelve un `Pipeline` **sin entrenar** (`preprocesador` = clon del `ColumnTransformer` de la 5.4, y `modelo`) más su espacio de búsqueda. `ModelFactory(preprocessor)` es la versión orientada a objetos.

| Nombre | Modelo | Hiperparámetros (escala log en negrita) |
|---|---|---|
| `logreg_ridge` | Regresión logística L2 | **C** ∈ [1e-4, 1e2], class_weight |
| `logreg_lasso` | Regresión logística L1 (saga) | **C** ∈ [1e-4, 1e2], class_weight |
| `knn` | KNN (`ball_tree`) | **n_neighbors** ∈ [5, 201], weights, leaf_size |
| `naive_bayes` | GaussianNB | **var_smoothing** ∈ [1e-12, 1e-1] |
| `cart` | Árbol de decisión | max_depth, **min_samples_leaf**, **ccp_alpha**, criterion, class_weight |
| `random_forest` | Random Forest | **n_estimators**, max_depth, **min_samples_leaf**, max_features, max_samples, class_weight |
| `lightgbm` | LightGBM | **n_estimators**, **learning_rate** ∈ [1e-3, 0,3], **num_leaves** ∈ [8, 512], **min_child_samples**, subsample, colsample, **reg_alpha**, **reg_lambda** |
| `xgboost` | XGBoost (hist) | **n_estimators**, **learning_rate**, max_depth, **min_child_weight**, **gamma**, subsample, colsample, **reg_alpha**, **reg_lambda** |
| `svm_lineal` | LinearSVC | **C** ∈ [1e-4, 1e2], class_weight |
| `svm_rbf` | Nyström(RBF) → LinearSVC | **gamma** ∈ [1e-4, 1], **n_components**, **C**, class_weight |

- `random_state=42` en todos los modelos y pasos aleatorios (se comprueba en las pruebas, incluida la SVD del preprocesador).
- Las 15 variables de regularización (C, alpha/reg_*, gamma, learning_rate, var_smoothing, ccp_alpha, min_child_weight) usan escala logarítmica.
- `PARAM_SPACES` es neutral: `sugerir_parametros(trial, espacio)` lo traduce a Optuna y `a_distribuciones_scipy(espacio)` a `RandomizedSearchCV`.
- **Ajustes al prompt**:
  - Se usa `Pipeline([("preprocesador", …), ("modelo", …)])` en vez de `make_pipeline`. Es equivalente, pero los nombres no cambian con cada modelo y las claves del espacio siempre son `modelo__…`.
  - La SVM RBF se aproxima con Nyström, porque un SVC exacto con 433 000 filas no termina. Así lo pedía también el plan original.
  - Las SVM no dan probabilidades: su AUC usa `decision_function` (`SALIDA_PUNTAJE`) y la calibración queda para M7.
  - En scikit-learn ≥ 1.8, Ridge y Lasso se piden con `l1_ratio` porque `penalty` está obsoleto (con versiones viejas se usa `penalty`).

**Prueba de humo con datos reales** (`run_m6_2.py`): valores por defecto, 40 000 filas de entrenamiento y 20 000 de validación, con pacientes completos del pliegue 1 de la 6.1. **No es la selección de modelos**, que se hará en la 6.3.

| Modelo | fit (s) | predict (s) | AUC global | AUC medio servicio × τ |
|---|---|---|---|---|
| logreg_ridge | 2,8 | 0,3 | 0,702 | 0,611 |
| logreg_lasso | 5,7 | 0,3 | 0,702 | 0,611 |
| knn | 1,3 | **50,1** | 0,637 | 0,549 |
| naive_bayes | 1,1 | 0,4 | 0,632 | 0,561 |
| cart | 3,3 | 0,3 | 0,578 | 0,533 |
| random_forest | 3,8 | 0,5 | 0,700 | 0,629 |
| lightgbm | 1,7 | 0,4 | 0,711 | 0,621 |
| xgboost | 2,2 | 0,4 | 0,699 | 0,625 |
| svm_lineal | 3,8 | 0,3 | 0,702 | 0,610 |
| svm_rbf | 6,5 | 0,4 | 0,711 | 0,624 |

**Para la 6.3:**
- KNN es inviable a escala completa: escala con n_train × n_val, unos 45 min de predicción por pliegue. Hay que buscar sus hiperparámetros sobre una submuestra de pacientes.
- CART por defecto (sin límite de profundidad) sobreajusta (0,58). Eso se espera hasta que se ajuste `max_depth` / `min_samples_leaf`.
- Reportar siempre la métrica **por servicio × τ**, además de la global: el AUC global sube ~0,08 solo por mezclar estratos con prevalencias distintas.

## Subfase 6.3 · Optimización bayesiana con Optuna

`m6_3_optuna/busqueda_optuna.py`.

- `objective(trial, model_name, X, y, groups, cv)`:
  - pide los hiperparámetros con `trial.suggest_float/int/categorical` (espacio de la 6.2);
  - entrena y evalúa pliegue a pliegue, y después de cada uno reporta el AUC acumulado a `MedianPruner` (`trial.report` / `should_prune`);
  - devuelve el **AUC-ROC medio de los 5 pliegues**.
  - Guarda además la desviación entre pliegues y el AUC medio por servicio × τ.
- `crear_estudio`: `optuna.create_study(direction="maximize", sampler=TPESampler(seed=42), pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1))`, guardado en SQLite (`data/m6/optuna_m6_3.db`) para poder **reanudarlo**.
- `guardar_resultados`: escribe `outputs/m6/best_hyperparameters.json` y la tabla comparativa. `tabla_por_familia` la consolida en las **7 familias** del prompt.
- **Cómo se evita sobreajustar la validación** (documentado en el código):
  1. La métrica es la media de 5 pliegues aislados por paciente, no una sola partición.
  2. No hay fuga de paciente entre pliegues.
  3. El preprocesamiento se ajusta dentro del pliegue.
  4. La prueba temporal nunca entra en la búsqueda. Como el mejor de 50 ensayos es optimista, el AUC definitivo se mide una sola vez en prueba (M7).

**Decisiones de cómputo (medidas antes de correr):**
- **Búsqueda sobre el 20 % de los pacientes** (~108 000 snapshots, ~19 400 pacientes), con los mismos pliegues de la 6.1. KNN usa el 5 %, porque su predicción es O(n_train × n_val).
  - Con 433 000 filas por pliegue, un ajuste pesado (LightGBM/XGBoost con 1 000 árboles, SVM RBF con 1 000 componentes) tarda minutos. 50 × 5 × 10 ajustes serían días.
  - Con el 20 %: 12–60 s por ajuste pesado.
  - Los hiperparámetros ganadores se reentrenan después con todo el entrenamiento.
- **`CachePliegues`**: el preprocesador del M5 no tiene hiperparámetros en la búsqueda, así que se ajusta una vez por pliegue (solo con su entrenamiento). La prueba `test_cache_equivale_al_pipeline` comprueba que da **exactamente** el mismo AUC que el Pipeline completo.
- `--timeout-modelo 1800`: cada estudio para a los 50 ensayos o a los 30 min. Si un modelo llega al tope, la tabla dice cuántos ensayos completó.
- **Modelos: 10, no 7.** El prompt habla de 7 modelos, pero dos de ellos son pares (Ridge/Lasso y XGBoost/LightGBM) y la SVM tiene variante lineal y RBF. Se optimizan los 10 y se consolida por familia.

Salidas:
- `outputs/m6/best_hyperparameters.json`
- `outputs/m6/m6_3_log.txt`
- en `outputs/m6/tablas/`: `m6_3_resumen_modelos.csv` (10 modelos), `m6_3_resumen_familias.csv` (7 familias) y `m6_3_historial_ensayos.csv` (todos los ensayos)

### Resultados (20 % de pacientes, 5 pliegues agrupados; KNN 5 %)

| Familia | Mejor modelo | AUC-ROC CV | DE entre pliegues | AUC medio servicio × τ | Ensayos completos / podados | Tiempo |
|---|---|---|---|---|---|---|
| Gradient boosting | **lightgbm** | **0,7368** | 0,0045 | **0,674** | 13 / 10 | 30 min (tope) |
| (gradient boosting) | xgboost | 0,7361 | 0,0041 | 0,673 | 16 / 8 | 32 min (tope) |
| Random Forest | random_forest | 0,7340 | 0,0050 | 0,670 | 14 / 3 | 34 min (tope) |
| SVM | svm_rbf | 0,7241 | 0,0032 | 0,645 | 8 / 6 | 30 min (tope) |
| (SVM) | svm_lineal | 0,7143 | 0,0044 | 0,621 | 19 / 31 | 9 min |
| Regresión logística | logreg_lasso | 0,7146 | 0,0047 | 0,623 | 21 / 29 | 7 min |
| (regresión logística) | logreg_ridge | 0,7144 | 0,0046 | 0,621 | 21 / 29 | 4 min |
| CART | cart | 0,7140 | 0,0058 | 0,628 | 21 / 29 | 8 min |
| KNN | knn | 0,6969 | 0,0078 | 0,584 | 28 / 22 | 13 min |
| Naive Bayes | naive_bayes | 0,6662 | 0,0064 | 0,571 | 26 / 24 | 0,4 min |

**Lectura:**
- Los modelos de árboles en ensamble (LightGBM, XGBoost, Random Forest) son los mejores. Sobre los lineales ganan unas 0,02 de AUC global, pero **unas 0,05 dentro de cada servicio × τ** (0,67 frente a 0,62): es en los estratos donde las no linealidades aportan de verdad.
- **Los tres primeros están a menos de 1 DE entre sí** (0,734–0,737, con DE ≈ 0,0045). Con esta búsqueda no se puede declarar un ganador único: lo dirán las pruebas formales de M7 (DeLong, Friedman + Nemenyi, Wilcoxon), sobre los mismos pliegues.
- Ganadores (completos en `best_hyperparameters.json`):
  - LightGBM: `learning_rate` 0,0054, 586 árboles, `num_leaves` 166, `min_child_samples` 204.
  - XGBoost: `learning_rate` 0,013, 858 árboles, `max_depth` 10.
  - Random Forest: 428 árboles, `max_depth` 20, `min_samples_leaf` 24, `max_features` 0,3, `class_weight="balanced"`.
- CART ajustado (0,714) pasa de 0,58 a la altura de los lineales. Naive Bayes y KNN quedan al final, como era de esperar con variables correlacionadas y una alta dimensión.
- **Limitación:** los 4 modelos pesados llegaron al tope de 30 min con 8–16 ensayos completos, no 50. Su búsqueda es más corta que la de los rápidos, y aun así quedan arriba; con más presupuesto podrían mejorar un poco. Para ampliarla basta con volver a correr `run_m6_3.py --timeout-modelo 3600`: reanuda desde la base SQLite.
- El AUC de CV del mejor de N ensayos es optimista (sesgo de selección). La cifra que vale es la de la prueba temporal en M7.


## Subfase 6.4 · Diagnóstico de sobreajuste, curvas y empaquetado

`m6_4_diagnostico/diagnostico.py` + `run_m6_4.py`.

1. **Carga** `outputs/m6/best_hyperparameters.json` e instancia los **7 Pipelines**, uno por familia: el mejor de cada una (`cargar_mejores`).
2. **Brecha de generalización.** Cada Pipeline se reentrena en los 5 pliegues de la 6.1 con **todo** el entrenamiento estable (541 260 filas; la búsqueda usó el 20 %). Se registra el AUC sobre el entrenamiento y sobre la validación de cada pliegue, y `brecha = train_auc − val_auc`.
   - Si la brecha supera **0,05** se imprime `⚠️ SOBREAJUSTE` y se emite `AdvertenciaSobreajuste`.
   - Las predicciones de validación de los 5 pliegues forman el **OOF**: cada fila se predice una vez, con un modelo que no la vio. Quedan en `data/m6/oof_m6_4.parquet` para M7.
3. **Curvas de aprendizaje** de los 3 mejores: 10 %, 25 %, 50 % y 100 % del entrenamiento de cada pliegue, submuestreado por pacientes completos. Además, **ROC comparativa** de los 7 con el OOF.
   - Se implementan sobre la caché de pliegues en vez de `sklearn.model_selection.learning_curve`, que reajustaría la SVD en cada punto.
   - El 100 % reutiliza los ajustes del paso 2.
4. **Empaquetado**: Pipeline final (preprocesador + modelo) ajustado con todo el entrenamiento en `models/<modelo>_pipeline.joblib`, más `models/manifest.json` (métricas, hiperparámetros, columnas de entrada, tipo de salida y versiones de las librerías).
   - Cada `.joblib` se recarga y se prueba sobre las columnas crudas antes de seguir.
   - La prueba temporal no se toca: su evaluación es tarea de M7.
   - **KNN** se entrena y evalúa con el 20 % de los pacientes. Predecir con 433 000 vecinos toma ~45 min por pliegue. Su `.joblib` **guarda filas de pacientes** (así funciona KNN): no compartirlo. `*.joblib` está en `.gitignore`.

Uso en M7 / M8:

```python
import joblib, json
pipe = joblib.load("models/lightgbm_pipeline.joblib")
cols = json.load(open("models/manifest.json", encoding="utf-8"))["columnas_entrada"]
p = pipe.predict_proba(X[cols])[:, 1]          # SVM: pipe.decision_function(X[cols]) (sin calibrar)
```

### Resultados (todo el entrenamiento estable, 5 pliegues agrupados; KNN 20 % de pacientes)

`outputs/m6/model_benchmark_summary.csv`, ya auditado (ver abajo):

| Modelo | AUC búsqueda 6.3 | AUC train | AUC val (± DE) | Brecha | AUC OOF servicio × τ | Cómo se mide el train |
|---|---|---|---|---|---|---|
| lightgbm | 0,7368 | 0,7668 | **0,7418** (± 0,0026) | 0,025 | 0,694 | resustitución |
| random_forest | 0,7340 | 0,7621 | 0,7364 (± 0,0027) | 0,026 | 0,686 | resustitución (regularizado en la auditoría) |
| svm_rbf | 0,7241 | 0,7315 | 0,7270 (± 0,0031) | 0,004 | 0,660 | resustitución |
| cart | 0,7140 | 0,7320 | 0,7220 (± 0,0021) | 0,010 | 0,649 | resustitución |
| logreg_lasso | 0,7146 | 0,7158 | 0,7146 (± 0,0031) | 0,001 | 0,628 | resustitución |
| knn | 0,6969 | 0,7053 | 0,7055 (± 0,0032) | −0,000 | 0,612 | leave-one-patient-out |
| naive_bayes | 0,6662 | 0,6651 | 0,6644 (± 0,0027) | 0,001 | 0,572 | resustitución |

**Ningún modelo supera la brecha de 0,05.** El AUC global está inflado porque el servicio y τ separan bien la prevalencia. La columna servicio × τ es la cifra honesta para comparar con otros trabajos.

Las curvas de aprendizaje (`figuras/m6_4_curvas_aprendizaje.png`) muestran que en los 3 mejores el train baja y la validación sube a medida que crecen los datos, hasta casi converger. La validación todavía sube un poco al 100 %: más datos ayudarían algo; más complejidad, no.

### Auditoría de la brecha (`auditoria_brecha.py`, `run_auditoria_m6_4.py`, `run_auditoria_knn_paciente.py`)

La primera corrida de la 6.4 dio dos brechas > 0,05: KNN (train 0,9998) y Random Forest (train 0,861 → brecha 0,127).

**KNN: el train ≈ 1 era un artefacto de medición, no sobreajuste.**
- *Resustitución.* Con `weights="distance"`, cada fila es su propio vecino a distancia 0 y recibe peso infinito: el modelo "devuelve" su etiqueta. Da 0,9998.
- *Leave-one-out por fila* (`kneighbors()` sin `X`). Excluye la propia fila, pero da 0,762, con brecha de 0,057. El 99 % de las filas comparte paciente con otras casi idénticas (snapshots τ = 0/2/6 h, 5,6 filas por paciente en promedio) y con la misma etiqueta. Esas "gemelas" siguen siendo los vecinos más cercanos.
- *Leave-one-patient-out* (`proba_knn_sin_mismo_paciente`). Excluye todas las filas del mismo paciente, que es la misma condición de la validación agrupada. Da **0,7053 vs val 0,7055**: brecha nula.
- Las pruebas verifican que el LOO y el LOPO coinciden exactamente con reentrenar sin la fila o sin el paciente.
- Tabla: `tablas/m6_4_auditoria_knn.csv`.

**Random Forest: el sobreajuste sí era real (árboles de profundidad 20 con hojas de 24).**
- Se recorrió una ruta de regularización en la submuestra del 20 %, con los mismos pliegues (`tablas/m6_4_ruta_regularizacion_rf.csv`). Se eligió la **mayor validación entre las configuraciones con brecha ≤ 0,05**: `max_depth=14`, `min_samples_leaf=100`.
- Confirmación con todo el entrenamiento: brecha 0,026.
- La validación **subió** de 0,7334 (submuestra, sin regularizar) a 0,7364 (todos los datos, regularizado). Regularizar no costó rendimiento.
- Se reempaquetó `models/random_forest_pipeline.joblib` y se actualizaron el OOF, el manifiesto y la curva de aprendizaje.

**Sobre el "0,6965".** Es el AUC de **validación** de KNN en la búsqueda de la 6.3 (5 % de los pacientes), no un AUC de entrenamiento. En la 6.4 la validación de KNN sube a 0,7055 porque se entrena con el 20 % de los pacientes: más vecinos de referencia.

Hiperparámetros definitivos: `outputs/m6/hiperparametros_finales.json` (los de la 6.3, salvo el RF regularizado). **M7 debe partir de ese archivo, no de `best_hyperparameters.json`.**

## Cifras de control

| Qué | Valor |
|---|---|
| Entrenamiento estable | 541 260 snapshots · 97 024 pacientes · 53,34 % positivos |
| Por pliegue | train ≈ 433 008 / val ≈ 108 252 filas · val 19 285 – 19 488 pacientes · **0 pacientes compartidos** |
| Proporción positiva en validación | 53,34 % en los 5 pliegues (desvío < 0,01 pp) |
| Por servicio × τ (rango entre pliegues) | hosp. 23,8 – 25,9 % · urg. τ0 51,0 – 51,5 %, τ2 57,9 – 58,3 %, τ6 75,1 – 75,8 % |

La estratificación se hace sobre el target global, así que en hospitalización (8 % de las filas) la prevalencia oscila ±1 pp entre pliegues. Si M6 reporta métricas por servicio, es normal ver más varianza entre pliegues ahí.
