# Módulo 7 · Evaluación, calibración y pruebas estadísticas

Evalúa los 7 modelos de la tarea de **paciente** (`y_target_los_largo`) empaquetados en la 6.4. Antes de empezar, lee `CLAUDE.md` y `modulo_6/README.md`.

## Estructura

| Carpeta | Subfase | Estado |
|---|---|---|
| `m7_1_multimetrica/` | 7.1 · Evaluación multimétrica fuera de muestra (OOF) con IC bootstrap | Hecha, con datos reales (2026-11-16) |
| `m7_2_calibracion_costo/` | 7.2 · Calibración (isotónica / sigmoide) y umbral de costo c_FN = 5·c_FP, prueba única sellada | Hecha, con datos reales (2026-11-16) |
| `m7_3_shap/` | 7.3 · Explicabilidad global y local con SHAP (TreeExplainer, agrupado por variable clínica) | Hecha, con datos reales (2026-11-16) |
| `m7_4_pruebas_hipotesis/` | 7.4 · DeLong por conglomerados + BCa, Friedman + Nemenyi, Wilcoxon + Holm | Hecha, con datos reales (2026-11-16) |

```bash
python -m pytest modulo_7 -q                          # 35 pruebas, ~3 min (SHAP y DeLong)
python modulo_7/m7_1_multimetrica/run_m7_1.py         # ~5 min (1 000 réplicas bootstrap)
python modulo_7/m7_2_calibracion_costo/run_m7_2.py --fase desarrollo   # ~20 s; congela los umbrales
python modulo_7/m7_2_calibracion_costo/run_m7_2.py --fase final        # CalibratedClassifierCV, ~1,5 h
python modulo_7/m7_2_calibracion_costo/run_m7_2.py --fase prueba       # UNA sola vez (queda sellada)
python modulo_7/m7_3_shap/run_m7_3.py                 # SHAP (~1 min con caché; --recalcular ~20 min)
python modulo_7/m7_4_pruebas_hipotesis/run_m7_4.py    # DeLong + BCa, Friedman + Nemenyi, Wilcoxon (~10 min)
```

## Subfase 7.1 · Evaluación multimétrica fuera de muestra

`m7_1_multimetrica/multimetrica.py` + `run_m7_1.py`. La entrada es `data/m6/oof_m6_4.parquet`: en cada fila, la predicción de un modelo que no vio a ese paciente (5 pliegues agrupados de la 6.1). La prueba temporal no se usa.

**Métricas por modelo**
- **Discriminación**: AUC-ROC y AUC-PR (`average_precision_score`). El AUC-PR sin información vale la **prevalencia**, no 0,5. La tabla la trae en `auc_pr_sin_informacion`.
- **Calibración**:
  - Brier, más `brier_skill = 1 − Brier / [p₀(1 − p₀)]`, que compara contra predecir siempre la prevalencia;
  - ECE con **15 intervalos de igual frecuencia** (cuantiles de la probabilidad). El prompt dice "15 deciles", pero 15 intervalos de igual frecuencia es lo que se implementa.
- **Umbral 0,5**: F1, sensibilidad, especificidad, precisión (VPP) y tasa de predichos positivos.

**IC del 95 % (AUC-ROC y AUC-PR): bootstrap estratificado por paciente, 1 000 réplicas, percentil.**
- Se remuestrean **pacientes**, no filas: cada paciente aporta ~5,6 snapshots correlacionados (τ = 0/2/6 h, episodios repetidos). Remuestrear filas trataría como independientes observaciones que no lo son y daría IC demasiado estrechos.
- Estratos: pacientes todo negativos / mixtos / todo positivos. El tamaño de cada estrato queda fijo en cada réplica.
- Cada réplica se aplica como pesos (veces que sale el paciente). `CurvaPreordenada` ordena los puntajes una vez y calcula AUC-ROC y AP ponderados en O(n). Las pruebas verifican que es idéntico a scikit-learn con `sample_weight`, incluidos los empates.
- **Réplicas pareadas**: las mismas para los 7 modelos (`tablas/m7_1_bootstrap_replicas.csv`). Sirven para comparar pares en las pruebas siguientes de M7.

**Dos ajustes respecto al prompt original**
1. **El desbalance no está en el agregado.** A nivel de snapshot la clase positiva es mayoría (53,3 %): los snapshots de urgencias a las 6 h son 75 % positivos. El desbalance real está en **hospitalización** (25 % de positivos). Por eso, además de la tabla global, hay métricas por **servicio × τ** (`tablas/m7_1_por_servicio_tau.csv`), cada una con su propia línea sin información.
2. **SVM** entrega `decision_function`, no probabilidades. Para Brier, ECE y umbral se convierte con **Platt cruzado por pliegue**: cada pliegue se calibra con los otros cuatro, nunca con sus propias etiquetas (las pruebas lo verifican). AUC-ROC antes y después: 0,7270 → 0,7270.

**KNN** solo tiene OOF para el 20 % de los pacientes (6.4). En la tabla global se evalúa sobre sus filas. `tablas/m7_1_subconjunto_comun.csv` compara los 7 modelos sobre **esas mismas filas**.

### Resultados (541 260 snapshots · 97 024 pacientes · prevalencia 0,533; KNN 108 495 filas)

`multimetric_evaluation_summary.csv` (IC 95 % bootstrap por paciente, 1 000 réplicas):

| Modelo | AUC-PR [IC 95 %] | AUC-ROC [IC 95 %] | Brier | Brier skill | ECE | F1 | Sens. | Espec. |
|---|---|---|---|---|---|---|---|---|
| lightgbm | **0,755** [0,753 – 0,758] | **0,742** [0,740 – 0,744] | 0,205 | 0,178 | 0,018 | 0,704 | 0,724 | 0,620 |
| random_forest | 0,750 [0,748 – 0,752] | 0,736 [0,735 – 0,738] | 0,207 | 0,169 | 0,030 | 0,681 | 0,659 | 0,683 |
| svm_rbf | 0,737 [0,735 – 0,740] | 0,727 [0,725 – 0,729] | 0,210 | 0,158 | 0,008 | 0,700 | 0,733 | 0,589 |
| cart | 0,732 [0,730 – 0,734] | 0,722 [0,720 – 0,724] | 0,211 | 0,152 | 0,008 | 0,689 | 0,702 | 0,617 |
| logreg_lasso | 0,724 [0,721 – 0,727] | 0,715 [0,713 – 0,717] | 0,215 | 0,137 | 0,029 | 0,680 | 0,687 | 0,620 |
| knn (20 % pac.) | 0,712 [0,705 – 0,719] | 0,705 [0,701 – 0,711] | 0,217 | 0,130 | 0,014 | 0,694 | 0,742 | 0,549 |
| naive_bayes | 0,664 [0,661 – 0,667] | 0,664 [0,662 – 0,667] | 0,307 | −0,231 | 0,272 | 0,704 | 0,834 | 0,389 |

El orden por AUC-PR coincide con el de AUC-ROC. Sobre el subconjunto común (las filas de KNN) se mantiene el mismo orden.

**Lectura**
- **LightGBM es el mejor en discriminación**, y su IC de AUC-PR apenas no se solapa con el del RF (0,7528 vs 0,7523). Que no se solapen no equivale a una prueba formal. La comparación pareada queda para las pruebas siguientes de M7, con las réplicas guardadas.
- **Calibración.**
  - SVM (con Platt) y CART son los mejor calibrados (ECE < 0,01).
  - RF y Lasso tienen ECE ≈ 0,03: se entrenaron con `class_weight="balanced"`, que desplaza las probabilidades.
  - **Naive Bayes está mal calibrado** (ECE 0,27, Brier peor que predecir la prevalencia: skill negativo). Sus probabilidades se amontonan cerca de 0 y 1.
- **El F1 no ordena bien los modelos.** Naive Bayes empata con LightGBM en F1 (0,704) porque predice positivo el 73 % de las veces (sensibilidad 0,83, especificidad 0,39). Por eso la métrica primaria es AUC-PR, junto con Brier/ECE.

**Por servicio × τ** (`tablas/m7_1_por_servicio_tau.csv`, promedio de τ0/2/6 en hospitalización):

| Modelo | AUC-PR hosp. (sin info 0,25) | AUC-ROC hosp. | Sens. @0,5 | Espec. @0,5 |
|---|---|---|---|---|
| lightgbm | 0,454 | 0,727 | 0,138 | 0,964 |
| random_forest | 0,443 | 0,721 | 0,042 | 0,992 |
| svm_rbf | 0,410 | 0,686 | 0,074 | 0,979 |
| cart | 0,388 | 0,671 | 0,127 | 0,958 |

- En urgencias, el AUC-PR va de 0,67 (τ0, sin info 0,51) a 0,85 (τ6, sin info 0,76).
- **Hallazgo para las siguientes subfases: el umbral 0,5 no sirve en hospitalización.** Con 25 % de prevalencia, el mejor modelo detecta solo el 14 % de las estancias largas. Hacen falta **umbrales por servicio**, elegidos sobre el OOF según el costo clínico de cada error, y calibración por servicio.

**Salidas**

| Archivo | Contenido |
|---|---|
| `outputs/m7/multimetric_evaluation_summary.csv` | Tabla consolidada, ordenada por AUC-PR descendente |
| `outputs/m7/tablas/m7_1_subconjunto_comun.csv` | Los 7 modelos sobre las filas de KNN |
| `outputs/m7/tablas/m7_1_por_servicio_tau.csv` | Métricas por servicio × τ |
| `outputs/m7/tablas/m7_1_bootstrap_replicas.csv` | Réplicas pareadas de AUC-ROC y AUC-PR (agregados) |
| `outputs/m7/figuras/m7_1_roc_pr.png` | Curvas ROC y Precision-Recall |
| `data/m7/probabilidades_oof_m7_1.parquet` | OOF en escala de probabilidad, con la SVM ya en probabilidad (nivel paciente: no compartir) |

## Subfase 7.2 · Calibración y umbral de costo asimétrico

`m7_2_calibracion_costo/calibracion_costo.py` + `run_m7_2.py`. Modelos: los 3 mejores de la 7.1 (lightgbm, random_forest, svm_rbf).

**Dos niveles, para no reutilizar datos**
1. **Desarrollo** (OOF del entrenamiento estable).
   - *Calibración cruzada por pliegue*: el calibrador del pliegue k se ajusta con el OOF de los otros cuatro pliegues. Así cada probabilidad calibrada viene de un modelo y un calibrador que no vieron a ese paciente. Es lo mismo que hace `CalibratedClassifierCV(cv=pliegues agrupados)`, pero sin reentrenar.
   - Aquí, y solo aquí, se eligen el **método** (menor Brier) y el **umbral θ\*** (menor costo). Ambos quedan congelados en `outputs/m7/m7_2_umbrales_congelados.json`, junto con el hash del OOF.
2. **Modelo desplegable.** `CalibratedClassifierCV(Pipeline, method, cv=pliegues agrupados de la 6.1, ensemble=True)` con todo el entrenamiento estable → `models/<modelo>_calibrado.joblib`.
   - Son 5 Pipelines, cada uno calibrado en su pliegue de validación.
   - Es el modelo que se evalúa **una sola vez** en la prueba temporal (periodo estable posterior al 2025-07-13), con los umbrales congelados.
   - La fase `prueba` deja un sello (`m7_2_prueba_unica.json`). Repetirla exige `--repetir-prueba` y queda registrado en el sello.

**Costo:** C(θ) = 5·FN(θ) + 1·FP(θ), con alerta si p ≥ θ, por cada 100 evaluaciones (snapshots), con θ ∈ [0,01; 0,99] en pasos de 0,01.
- Con probabilidades calibradas, el umbral de Bayes es c_FP / (c_FP + c_FN) = **1/6 ≈ 0,167**. Sirve de control: si θ\* empírico ≈ 0,167, la calibración es buena.
- **Referencias sin modelo**: "alertar a todos" cuesta (1 − p₀)·100 y "no alertar nunca", 5·p₀·100. El valor real del modelo es lo que ahorra **frente a la mejor de ellas**, no frente a θ = 0,5.
- Reglas pre-especificadas para la prueba: θ = 0,5, θ\* global, θ\* por servicio y θ de Bayes. Como sensibilidad, θ\* con c_FN = 10 (solo en desarrollo).

### Resultados en desarrollo (OOF, 541 260 snapshots, prevalencia 0,533)

**Calibración:** la isotónica gana en los 3 modelos (`tablas/m7_2_calibracion_desarrollo.csv`, `figuras/m7_2_fiabilidad_desarrollo.png`).

| Modelo | ECE sin calibrar | ECE isotónica | ECE sigmoide | Brier sin cal. → isotónica | ECE isotónica hosp. / urg. |
|---|---|---|---|---|---|
| lightgbm | 0,018 | **0,001** | 0,017 | 0,2046 → 0,2041 | 0,011 / 0,003 |
| random_forest | 0,030 | **0,001** | 0,016 | 0,2069 → 0,2058 | 0,014 / 0,003 |
| svm_rbf | — (decision_function) | **0,001** | 0,008 | — → 0,2094 | 0,005 / 0,002 |

- La sigmoide (Platt) es una logística *lineal* sobre el puntaje. No corrige la curvatura de LightGBM y RF, y en hospitalización empeora la calibración frente a no calibrar (0,037 vs 0,019 en LightGBM).
- La isotónica conserva el AUC (−0,0003): la calibración no cambia el orden.

**Umbral de costo** (`tablas/m7_2_umbrales_desarrollo.csv`, `figuras/m7_2_costo_umbral_desarrollo.png`):

| Modelo | θ\* | Costo/100 @ θ = 0,5 | Costo/100 @ θ\* | Alertar a todos | Ahorro vs θ = 0,5 | **Ahorro vs alertar a todos** |
|---|---|---|---|---|---|---|
| lightgbm | 0,16 | 95,5 | **43,8** | 46,7 | 51,8 | **2,9** |
| random_forest | 0,17 | 96,2 | 43,8 | 46,7 | 52,3 | 2,8 |
| svm_rbf | 0,17 | 93,5 | 44,9 | 46,7 | 48,6 | 1,8 |

- **θ\* ≈ 0,16–0,17 coincide con el umbral de Bayes (0,167)**: las probabilidades isotónicas están bien calibradas.
- **Lectura honesta.** θ\* reduce el costo a menos de la mitad frente a θ = 0,5 (sensibilidad 0,70 → 0,99). Pero, con 53 % de prevalencia y 5:1, "alertar a todos" ya cuesta 46,7, y el modelo solo ahorra 2–3 puntos más.
- **Por servicio.**
  - **Urgencias**: θ\* alerta al 100 % de los snapshots. Casi ninguno tiene una probabilidad calibrada < 0,33 (`figuras/m7_2_fiabilidad_por_servicio_desarrollo.png`), así que con 5:1 el modelo no cambia la decisión.
  - **Hospitalización** (prevalencia 25 %): con LightGBM y θ\* el costo baja de 74,9 (alertar a todos) a **59,7**, un **−20 %**. Sensibilidad 0,90, especificidad 0,37, alertas en el 70 % de los casos.
- El θ\* por servicio casi no difiere del global (hosp. 0,16–0,17). Es lo esperable si la calibración se sostiene dentro de cada servicio.
- Con c_FN = 10, θ\* baja a 0,07–0,10: más alertas todavía.
- **Para el equipo:** el valor operativo del modelo con un costo de 5:1 está en **hospitalización**. En urgencias habría que revisar la razón de costos con el hospital o fijarla por servicio: con 5:1, alertar siempre es casi óptimo.

### Resultados en la prueba temporal (evaluación única, 2026-11-16 09:56)

Datos: 148 673 snapshots de 31 165 pacientes, del periodo estable posterior al 2025-07-13. Prevalencia **0,578**, frente a 0,533 en desarrollo. Los umbrales se congelaron a las 08:20 (sello en `m7_2_prueba_unica.json`, sin repeticiones). Detalle en `tablas/m7_2_{calibracion,umbrales}_prueba.csv` y `figuras/m7_2_fiabilidad_prueba.png`.

| Modelo (isotónica) | AUC-ROC prueba (OOF) | AUC-PR | Brier | ECE prueba (desarrollo) | p media vs prevalencia |
|---|---|---|---|---|---|
| lightgbm | **0,729** (0,742) | 0,773 | 0,205 | 0,027 (0,001) | 0,551 vs 0,578 |
| random_forest | 0,725 (0,736) | 0,770 | 0,206 | 0,029 (0,001) | 0,549 vs 0,578 |
| svm_rbf | 0,711 (0,727) | 0,752 | 0,211 | 0,021 (0,001) | 0,556 vs 0,578 |

- **Degradación temporal moderada.** El AUC cae entre 0,011 y 0,016 frente al OOF, y el orden de los modelos se mantiene. La calibración pierde precisión porque la prevalencia subió 4,4 pp en el periodo de prueba: el modelo **subestima el riesgo ~2,7 pp** (ECE de 0,001 a 0,027). Aun así, la isotónica mejora a los modelos sin calibrar (RF: ECE de 0,058 a 0,029).
- **Costo por cada 100 evaluaciones (LightGBM):**

| Regla | Global | Hospitalización | Urgencias |
|---|---|---|---|
| θ = 0,5 | 95,7 | 124,4 | 88,7 |
| **θ\* = 0,16** (congelado) | **40,1** | **61,0** | 35,0 |
| Alertar a todos (sin modelo) | 42,2 | 71,8 | 35,0 |

- Lo observado en desarrollo **se replica en la prueba**:
  - θ\* ahorra 55,6 por cada 100 frente a θ = 0,5;
  - frente a alertar a todos, ahorra 2,1 en el global y **10,8 en hospitalización (−15 %)**, con sensibilidad 0,90 y especificidad 0,35;
  - en urgencias el modelo no cambia la decisión con una razón de costos de 5:1.
- El θ\* por servicio no mejora al global, y el θ de Bayes (0,167) rinde igual que θ\*.
- **Recomendaciones para M8:**
  - recalibrar periódicamente (la prevalencia sube con el tiempo);
  - usar el modelo en hospitalización;
  - acordar con el hospital la razón de costos de urgencias.



**Salidas**

| Archivo | Contenido |
|---|---|
| `outputs/m7/m7_2_umbrales_congelados.json` | Método y θ\* por modelo, fijados en desarrollo (con hash del OOF) |
| `outputs/m7/m7_2_prueba_unica.json` | Sello de la evaluación única en prueba |
| `outputs/m7/tablas/m7_2_calibracion_{desarrollo,prueba}.csv` | Brier, ECE, AUC y p media vs prevalencia, por versión y servicio |
| `outputs/m7/tablas/m7_2_umbrales_{desarrollo,prueba}.csv` | θ = 0,5 vs θ\* (y las otras reglas), por servicio, con las políticas sin modelo |
| `outputs/m7/tablas/m7_2_barrido_costo_desarrollo.csv` | Costo por 100 para cada θ |
| `outputs/m7/figuras/m7_2_fiabilidad_*.png`, `m7_2_costo_umbral_desarrollo.png` | Diagramas de fiabilidad y curva de costo |
| `models/<modelo>_calibrado.joblib` | `CalibratedClassifierCV` (isotónica, 5 pliegues agrupados) |
| `data/m7/probabilidades_prueba_m7_2.parquet` | Probabilidades calibradas en la prueba (nivel paciente: no compartir) |

## Subfase 7.3 · Explicabilidad clínica con SHAP

`m7_3_shap/explicabilidad.py` + `run_m7_3.py`. Las figuras van en `reports/figures/` (PNG a 300 DPI) y la síntesis clínica en `reports/m7_3_sintesis_medica.md`.

**Diseño**
- **Explicador:** `TreeExplainer` sobre el LightGBM (el mejor modelo de la 7.1 y la 7.4). Es exacto para árboles y trabaja en log-odds. Se verifica la aditividad, E[f(X)] + Σφ = salida del modelo: error máximo 1,2·10⁻¹³.
- **Muestra de validación honesta.** El Pipeline se reentrena **sin el pliegue 1** (AUC en el pliegue = 0,739) y se explican 10 000 snapshots del pliegue 1, estratificados por servicio × τ × etiqueta. Son pacientes que ese modelo nunca vio. La prueba temporal no se usa.
- **SHAP agrupado por variable clínica.** Las 124 columnas transformadas se suman en 61 variables originales. La suma es exacta por aditividad: las 50 componentes SVD forman el "perfil de diagnósticos"; seno y coseno, la "hora del día"; las categorías one-hot, su variable.
- **"Qué diagnósticos" en vez de "cuántos".** Para cada diagnóstico frecuente se compara el SHAP del perfil de diagnósticos con el código presente y ausente. Va dentro de cada servicio y con nombres traducidos al español.
- **Casos locales sin exponer pacientes** (regla 7). Los waterfalls muestran el SHAP medio del 5 % de mayor y de menor riesgo **dentro de cada servicio**. El SHAP medio también es aditivo, y el punto de partida ya incluye el efecto del servicio. Un caso global quedaría dominado por servicio y τ: el 5 % de mayor riesgo global es 100 % urgencias a las 6 h. La explicación individual corresponde al despliegue (M8).
- **El triaje no existe en los datos**, así que la síntesis habla de admisión, historia clínica, laboratorios, presión del servicio y calendario.

**Resultados** (`tablas/m7_3_importancia_shap*.csv`, `m7_3_diagnosticos_shap.csv`, `m7_3_casos_estudio.csv`)

| Rango | Variable | % de la influencia | Dirección |
|---|---|---|---|
| 1 | Servicio | 27 % | Hospitalización → menor riesgo. **Efecto de la definición** (umbral P75 propio de cada servicio), no clínico |
| 2 | Momento de la evaluación (τ) | 16 % | Seguir en el servicio a las 2 o 6 h → mayor riesgo (información pronóstica) |
| 3 | Perfil de diagnósticos | 10 % | Depende de *qué* diagnósticos (ver abajo) |
| 4 | Diagnósticos en los últimos 30 días | 8 % | Más → mayor riesgo |
| 5 | Edad | 6 % | Mayor → mayor riesgo |

Por dominio: admisión 51 %, historia clínica 27 %, presión del servicio 9 %, laboratorios 8 %, calendario 6 %.

- **Hospitalización** (sin la variable servicio, que ahí es constante):
  - Primeros: perfil de diagnósticos (19 %), llegadas al servicio en 24 h (18 %, efecto no lineal), τ (10 %), edad (9 %).
  - Elevan el riesgo: infección de piel o tejido subcutáneo, infección respiratoria baja, neumonía.
  - Lo reducen: falso trabajo de parto y cesáreas (estancias obstétricas cortas frente al P75).
- **Urgencias:**
  - Primeros: τ (25 %), perfil de diagnósticos (12 %), diagnósticos recientes (11 %), edad (8 %), hora del día (7 %; llegar más tarde → menor riesgo).
  - Elevan el riesgo: infección de piel, cólico renal, insuficiencia cardíaca, infección urinaria.
  - Lo reducen: amenaza de aborto, hemorragia genitourinaria.
- **Perfiles por servicio:**
  - Hospitalización: el 5 % de mayor riesgo tiene riesgo estimado 57 % (observado 59 %; 68 años de mediana); el de menor riesgo, 4 % (observado 4 %; 24 años).
  - Urgencias: 85 % (observado 91 %) vs 36 % (observado 31 %).
- **Cuidado al interpretar.** Las llegadas al servicio en 24 h pesan mucho en hospitalización y su efecto no es lineal. Pueden estar capturando la tendencia temporal del censo (265 → 315 entre 2023 y 2025), no la presión del día. Conviene revisarlo con la prueba temporal antes de usarlo como explicación clínica.

**Salidas**
- `reports/figures/m7_3_shap_beeswarm.png`, `m7_3_shap_bar.png` y `m7_3_shap_bar_por_servicio.png`
- `m7_3_shap_waterfall_{alto,bajo}_riesgo_{hosp,urg}.png`
- `reports/m7_3_sintesis_medica.md`
- `data/m7/shap_m7_3.parquet`: caché de SHAP a nivel paciente; no compartir fuera del equipo

`python modulo_7/m7_3_shap/run_m7_3.py` usa la caché (~1 min); `--recalcular` reentrena (~20 min).

## Subfase 7.4 · Tres pruebas formales de superioridad

`m7_4_pruebas_hipotesis/pruebas.py` + `run_m7_4.py`, sobre el OOF de desarrollo. La prueba temporal no se usa.

**Reglas fijadas antes de ver resultados**
- Ganador y top 2 = los dos primeros de la 7.1 por AUC-PR (lightgbm, random_forest).
- α = 0,05. Margen de relevancia práctica: ΔAUC con IC BCa ≥ 0,01.
- Con n grande todo sale significativo (regla 6 de `CLAUDE.md`). Por eso **significancia y magnitud se reportan por separado**.
- **p ≥ 0,05 no demuestra equivalencia**: haría falta un TOST con margen. El script imprime "equivalencia estadística" porque lo pide la rúbrica, pero aclara que no se demuestra diferencia.

**Prueba 1 · DeLong pareado (ganador vs 6 baselines)**
- Principal: **DeLong por conglomerados de paciente** (Obuchowski, 1997). El DeLong clásico supone filas independientes, y aquí cada paciente aporta ~5,6 snapshots. El SE por conglomerados resulta **~1,5 veces** el clásico (`inflacion_se`). Con conglomerados de tamaño 1 coincide exactamente con el clásico (verificado en las pruebas), y su error tipo I con datos agrupados queda cerca del 5 %.
- IC 95 % **BCa**: bootstrap pareado por paciente (1 000 réplicas, las mismas de la 7.1) y aceleración por jackknife de 100 grupos de pacientes.
- Holm sobre las 6 comparaciones. KNN se compara sobre sus filas.

| Baseline | AUC lightgbm | AUC baseline | ΔAUC [IC 95 % BCa] | p (Holm) | Veredicto |
|---|---|---|---|---|---|
| random_forest | 0,7418 | 0,7364 | +0,0053 [0,0049; 0,0057] | < 1e-10 | Superior, **magnitud pequeña** (< 0,01) |
| svm_rbf | 0,7418 | 0,7270 | +0,0147 [0,0139; 0,0156] | < 1e-10 | Superior y relevante |
| cart | 0,7418 | 0,7222 | +0,0196 [0,0188; 0,0204] | < 1e-10 | Superior y relevante |
| logreg_lasso | 0,7418 | 0,7146 | +0,0271 [0,0261; 0,0282] | < 1e-10 | Superior y relevante |
| knn (sus filas) | 0,7440 | 0,7054 | +0,0386 [0,0356; 0,0413] | < 1e-10 | Superior y relevante |
| naive_bayes | 0,7418 | 0,6643 | +0,0774 [0,0755; 0,0793] | < 1e-10 | Superior y relevante |

En la tabla, z y log₁₀ p (`m7_4_prueba1_delong.csv`): p = 0 en pantalla es un desbordamiento numérico.

**Prueba 2 · Friedman + Nemenyi** (AUC-ROC, 5 pliegues × 7 modelos, sobre las 108 495 filas comunes)
- LightGBM queda 1.º en los 5 pliegues, y el orden completo se repite en todos: **W de Kendall = 1**, Friedman χ² = 30,0, p = 3,9·10⁻⁵ → **diferencias significativas**. Iman-Davenport da F = ∞, por la concordancia perfecta.
- Nemenyi: CD = 4,03 rangos, porque con N = 5 la potencia es baja. LightGBM se separa de **KNN (p = 0,005) y Naive Bayes (p = 0,0002)**. Frente a Lasso, p = 0,053. Frente a RF, SVM y CART no se demuestra diferencia (`figuras/m7_4_diagrama_diferencia_critica.png`).
- Sensibilidad con 6 modelos en todas las filas: Friedman p = 1,4·10⁻⁴, W = 1.

**Prueba 3 · Wilcoxon de rangos con signo** (lightgbm vs random_forest, Holm entre 4 métricas: AUC-ROC, AUC-PR, Brier, log-loss)
- **Con 5 pliegues el p exacto mínimo bilateral es 2/2⁵ = 0,0625**: no puede llegar a 0,05, aunque LightGBM gane en los 5 pliegues y en las 4 métricas (biserial de rangos = 1). Se reporta, pero no decide.
- **Decisión: 50 particiones disjuntas de pacientes** (10 por pliegue). LightGBM gana en 50/50 bloques en AUC-ROC, Brier y log-loss, y en 49/50 en AUC-PR. p_Holm = 7·10⁻¹⁵ en las 4 métricas → **superioridad estadística**. La magnitud es la misma que en la Prueba 1 (ΔAUC ≈ 0,005).
- Las particiones comparten el modelo de su pliegue: la prueba compara las predicciones sobre conjuntos disjuntos de pacientes, no reentrenamientos independientes.

**Veredicto conjunto**
- LightGBM es **estadísticamente superior** a los 6 baselines (Prueba 1) y queda primero en todos los pliegues (Prueba 2).
- Frente al Random Forest la superioridad es significativa pero **pequeña**: ΔAUC = 0,005, por debajo del margen de relevancia de 0,01. En la práctica, RF es una alternativa casi equivalente.
- Frente a SVM, CART, Lasso, KNN y Naive Bayes la ventaja es significativa y relevante (≥ 0,015).

**Salidas**
- `outputs/m7/tablas/m7_4_prueba1_delong.csv`
- `m7_4_matriz_p_delong_{crudo,holm}.csv`
- `m7_4_prueba2_auc_por_pliegue.csv` y `m7_4_prueba2_nemenyi_p.csv`
- `m7_4_prueba3_wilcoxon.csv` y `m7_4_prueba3_metricas_por_particion.csv`
- `outputs/m7/m7_4_resultados.json`
- `figuras/m7_4_diagrama_diferencia_critica.png` y `figuras/m7_4_heatmap_pvalores.png` (DeLong con ΔAUC anotado; Nemenyi con Δ de rango medio)
