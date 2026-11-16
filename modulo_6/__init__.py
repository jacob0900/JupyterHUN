"""Módulo 6 · Modelado supervisado y validación cruzada (Data Challenge HUN, `uninorte_omop`).

- `m6_1_cv_agrupada/`: validación cruzada agrupada por paciente (StratifiedGroupKFold por person_id).
- `m6_2_bateria_modelos/`: batería de clasificadores y espacios de búsqueda (model_factory).
- `m6_3_optuna/`: optimización bayesiana de hiperparámetros con Optuna (TPE + MedianPruner).
- `m6_4_diagnostico/`: brecha de sobreajuste, curvas de aprendizaje, ROC y empaquetado en models/.
"""
