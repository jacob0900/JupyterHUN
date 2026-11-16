"""Subfase 7.4 · Pruebas formales de superioridad (DeLong, Friedman + Nemenyi, Wilcoxon)."""
from .pruebas import (
    ALFA,
    DELTA_RELEVANTE,
    bootstrap_delta,
    colocaciones,
    comparar_ganador,
    delong,
    friedman_nemenyi,
    graficar_cd,
    graficar_heatmaps,
    ic_bca,
    jackknife_grupos,
    matriz_p_delong,
    p_bootstrap,
    veredicto,
    wilcoxon_metricas,
    wilcoxon_pareado,
)

__all__ = ["ALFA", "DELTA_RELEVANTE", "bootstrap_delta", "colocaciones", "comparar_ganador", "delong",
           "friedman_nemenyi", "graficar_cd", "graficar_heatmaps", "ic_bca", "jackknife_grupos", "matriz_p_delong",
           "p_bootstrap", "veredicto", "wilcoxon_metricas", "wilcoxon_pareado"]
