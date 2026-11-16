# Entrega de Persona 3 · M2, M3 y M4 visual

De: Persona 3 · Fecha: 2026-11-12 · Para: Persona 1 (M8, datos) y Persona 2 (M5–M7)

## Cómo regenerar

```bash
pip install -r requirements.txt -r requirements-eda.txt
python m2_eda/run_m2.py            # ~20 s
python m3_bivariado/run_m3.py      # ~30 s
python m4_visual/run_m4_visual.py  # ~30 s
```

Solo leen `data/` (los parquets de Persona 1). Escriben **agregados** en `outputs/`:

| Módulo | Informe | Figuras / tablas |
|---|---|---|
| M2 · univariado, calidad y censura (HUN-201) | [outputs/m2/M2_informe.md](../outputs/m2/M2_informe.md) | `outputs/m2/figuras` (10), `outputs/m2/tablas` (CSV) |
| M3 · bivariado y colas (HUN-202) | [outputs/m3/M3_informe.md](../outputs/m3/M3_informe.md) | `outputs/m3/figuras` (6), `outputs/m3/tablas` |
| M4 · espacio-temporal, visual (HUN-203) | [outputs/m4_visual/M4_visual_informe.md](../outputs/m4_visual/M4_visual_informe.md) | `outputs/m4_visual/figuras` (10), `tablas`, **`html/`** (Sankey y heatmaps interactivos para el dashboard) |

Funciones compartidas (tamaños de efecto, BH, estilo): `eda_comun/__init__.py`.

## Decisiones de método

- Descriptivo (M2, M4) sobre todo el **periodo estable**. Pruebas bivariadas (M3) **solo sobre entrenamiento**, para no contaminar la selección de variables.
- Sin triaje: Kaplan-Meier estratificado por servicio, origen, grupo de edad y sexo. Se reportan log-rank y ε².
- Todo p-valor va con tamaño de efecto (V de Cramér con corrección de sesgo, ε², AUC univariada, diferencia de medianas con IC bootstrap) y corrección **Benjamini-Hochberg**.

## Para Persona 2 (M5–M7)

1. **Deriva del target de paciente**: la estancia hospitalaria crece (mediana ~68 h en 2023 → ~75 h en 2025–26). Con el umbral P75 fijo, la prevalencia sube de 0.25 (entrenamiento) a 0.28 (prueba). Hay que tenerlo en cuenta al calibrar y elegir el umbral (M7).
2. **La prevalencia en snapshots de urgencias depende de τ**: 0.51 / 0.58 / 0.76 en τ = 0 / 2 / 6 h. Hay que reportar las métricas por τ.
3. **Variables con más efecto marginal** (todas pequeñas, ninguna grande): en hospitalización, `edad` y `n_mediciones_historia`; en urgencias, `n_dx_30d`, `n_dx_distintos_historia`, `n_dx_historia` y `edad`. El calendario casi no influye en la estancia individual.
4. **Colinealidad**: `n_dx_historia` / `n_dx_distintos_historia` / `n_dx_30d` (ρ 0.74–0.94) y `censo` / `ocupacion` t−1 (0.80–0.94).
5. **MNAR**: los indicadores `sin_mediciones_24h`, `nunca_hospitalizado` y `sin_dx_historia` tienen efecto pequeño pero consistente. Incluirlos y no imputar `horas_desde_ultima_alta_hosp` con la mediana.
6. **Saturación a 4 h**: la ocupación actual sola da AUC 0.89 (hosp.) y 0.84 (urg.). Es el **baseline de persistencia** que el modelo tiene que superar. Variables clave: hora del día y turno, mes, `festivo`, fin de semana, primer día hábil tras descanso, y llegadas de las últimas 4 h (en urgencias).
7. **Umbrales por turno**: la tasa base de saturación cambia mucho por turno (hosp.: mañana 53 % vs. tarde 34 %; urg.: tarde hábil 68 % vs. mañana 16 %). El presupuesto de 2 falsas alarmas por turno se debe fijar por servicio × turno.
8. La **hora de ingreso a hospitalización** cae en franjas rígidas (16–17 h y 20–22 h): parece la hora de registro administrativo, no la clínica.

## Para Persona 1 (datos y M8)

1. **`calendario_dia.dia_despues_festivo` incluye los lunes normales**: está marcado en 174 días, pero solo 46 siguen a un festivo real. Marca "primer día hábil tras cualquier día no hábil". Por eso el "rebote" de `m4_resumen.md` mezcla el efecto lunes con el efecto festivo (ambos ≈ +10 % en urgencias). Propuesta: renombrar a `primer_dia_habil` y añadir `tras_festivo`.
2. **Cambio de registro en urgencias/observación dentro del periodo estable**: el % de episodios con LOS = 0 cambia por tramos (cae entre 2024-08 y 2025-04 y desde 2025-12). λ baja y W sube año a año, con λ·W casi constante. Vale la pena revisarlo en el origen o añadir una variable de régimen.
3. **Censura real**: 176 hospitalizaciones y 290 urgencias tienen `fin` = fecha de corte (estancias abiertas en la extracción). Solo 6 caen en el periodo estable, así que el impacto es despreciable, pero conviene documentarlo.
4. **Para el dashboard**: `outputs/m4_visual/html/sankey_flujo.html`, `heatmap_llegadas.html` y `heatmap_saturacion.html` (plotly, cargan plotly.js desde CDN). Las tablas agregadas que los alimentan están en `outputs/*/tablas/`.

## Hallazgos principales (resumen)

- Hospitalización tiene un componente **obstétrico/neonatal** visible (picos en 0–4 años y en mujeres de 15–35).
- **Hospitalización es el cuello de botella** (ρ ≈ 0.93 respecto al proxy, ≥ 0.85 en el 91 % de las horas). Su censo crece porque las estancias se alargan, no porque lleguen más pacientes. Satura de madrugada y por la mañana, con el máximo los lunes.
- Urgencias satura por la tarde (martes y miércoles, 14–17 h). Las llegadas pican a las 10–11 h.
- Urgencias → cama de hospitalización: mediana ~14 h. Depende poco de la ocupación al llegar (señal débil de boarding).
- Festivo ≈ domingo (−22 % en urgencias, −35 % en hospitalización). Las mayores caídas son Viernes Santo, Año Nuevo y Jueves Santo.
- La estacionalidad anual es débil e inconsistente con ~3 años de datos. Lo estable es Semana Santa y las vacaciones.
