"""Subfase 7.1 · Evaluación multimétrica fuera de muestra."""
from .multimetrica import (
    N_BINS_ECE,
    N_BOOTSTRAP,
    UMBRAL,
    CurvaPreordenada,
    bootstrap_ic,
    ece_igual_frecuencia,
    estratos_paciente,
    evaluar_modelos,
    graficar_roc_pr,
    metricas,
    metricas_por_estrato,
    metricas_umbral,
    platt_cruzado,
    replicas_bootstrap,
)

__all__ = ["N_BINS_ECE", "N_BOOTSTRAP", "UMBRAL", "CurvaPreordenada", "bootstrap_ic", "ece_igual_frecuencia",
           "estratos_paciente", "evaluar_modelos", "graficar_roc_pr", "metricas", "metricas_por_estrato",
           "metricas_umbral", "platt_cruzado", "replicas_bootstrap"]
