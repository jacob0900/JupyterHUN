# Entrega para la Persona 3 (M2, M3, M4 visual)

De: Persona 1 · Fecha: 2026-11-10
Paquete: `entrega_persona3.zip`. Descomprímelo en la raíz del proyecto y quedan las rutas `data/...` y `docs/...` de esta guía.

> ⚠️ El paquete tiene datos a nivel paciente (re-anonimizados). Compártelo solo por el canal aprobado del equipo y no lo subas a git ni a servicios públicos.

## Si trabajas con Claude

El paquete trae un **`CLAUDE.md` en la raíz**. Abre Claude Code en esa carpeta y lo carga solo: así conoce las reglas, las trampas de los datos y dónde está cada archivo. Ese archivo **corrige al PDF `modulos_hospital_hun_contexto_claude.pdf`**, que se escribió antes de ver los datos reales (supone triaje, estancia en visit_occurrence, etc.). Si le pasas el PDF a Claude, dile que `CLAUDE.md` manda cuando se contradigan.

Para arrancar, pídele algo como: *"Lee CLAUDE.md y docs/M4_entrega_persona3.md y arranca el M2: caracterización demográfica y curvas Kaplan-Meier de `episodio` por servicio y origen en urgencias, solo periodo estable"*.

## Lee esto primero (5 minutos)

1. **`docs/M1_hallazgos_datos.md`**: los datos del HUN son muy distintos a lo que suponía el plan. Lo esencial:
   - **No hay triaje.** Las curvas Kaplan-Meier "por nivel de triaje" no se pueden hacer. Propongo estratificar por **servicio**, **origen en urgencias** y **grupo de edad**.
   - **Hay solo 4 servicios:** `ambulatorio`, `urgencias_ingreso`, `urgencias_observacion` y `hospitalizacion`.
   - **La estancia está en los subprocesos (`visit_detail`)**, no en la visita.
   - **Parte de los registros están mal vinculados a su visita.** Ya están filtrados en todo lo que te entrego (`vinculo_coherente`).
2. **`docs/M1_contrato_datos.md`**: qué es cada columna. Lo más importante:
   - Las columnas `y_*` son el futuro.
   - **Filtra `en_periodo_estable = true`**: del 31 de enero de 2023 al 1 de abril de 2026. Fuera de ese rango la carga de datos se comporta distinto.

## Qué usar para cada módulo

### M2 · EDA univariado y censura (HUN-201)

| Archivo | Para qué |
|---|---|
| `data/m1/episodio.parquet` | **Una fila por estancia** (hospitalización 49 k, urgencias/observación 493 k). `los_h`, `edad`, `sexo`, `origen_urgencias`, `target_los_largo`, `umbral_los_h` |
| `data/raw/person.parquet` | Demografía de todas las personas (912 k): `gender_concept_id` (8507 M, 8532 F), `year_of_birth` |
| `data/m1/reporte_calidad.parquet` | Calidad cuantificada (n afectados / n total por chequeo), lista para tablas del informe |
| `data/m1/snapshot.parquet` | Balance de clases del target por servicio y por instante (0, 2, 6 h); `n_mediciones_24h = 0` como indicador MNAR |

Sobre la **censura**: no hay estancias abiertas, todas tienen `fin`. El sesgo real es el **truncamiento al final de los datos**: las estancias largas que empezaron cerca del corte no aparecen completas. Esos episodios vienen marcados con `en_riesgo_truncamiento` (ingresos de los últimos 30 días). Para Kaplan-Meier puedes:
- tratarlos como censurados en `parametros.fecha_corte_datos`, o
- excluirlos y reportar cuántos son (924 de hospitalización y 8 548 de urgencias).

### M3 · EDA bivariado y colas (HUN-202)

| Archivo | Para qué |
|---|---|
| `data/m4/resumen_colas.parquet` | **λ (llegadas/h), E[S], μ, censo L, capacidad c y ρ ≈ λ·E[S]/c por servicio**. La ley de Little cuadra (L ≈ λW): hospitalización 289.5 frente a 288.7, urgencias 224.0 frente a 223.6 |
| `data/m4/matriz_transicion.parquet` | Probabilidad de pasar de un servicio a otro y tiempos entre subprocesos (p50 / p75 / p90) |
| `data/m1/ocupacion_servicio_hora.parquet` | Censo, capacidad y ocupación hora a hora: base para la "fracción de horas con ρ > 0.85" |
| `data/m1/episodio.parquet`, `snapshot.parquet` | Pruebas bivariadas (Chi² / Kruskal-Wallis + Cramér's V / ε², corrección BH) |

Ten en cuenta dos cosas:
- Los **tiempos de espera entre subprocesos son ~0** casi siempre: el fin de un registro es el inicio del siguiente. Los datos no registran la espera dentro de urgencias, así que el cuello de botella se mide por **duración** y **ocupación**, no por espera.
- La **capacidad es un proxy**: el P95 del censo de los últimos 90 días, porque no tenemos el número de camas. Por eso ρ en hospitalización (~0.93) está medido contra ese proxy y no contra camas reales.

### M4 · Espacio-temporal (HUN-203, tu parte visual)

| Archivo | Visual sugerido |
|---|---|
| `data/m4/heatmap_llegadas.parquet` / `.csv` | Heatmap **día de la semana × hora** de llegadas promedio por servicio (168 celdas por servicio) |
| `data/m4/heatmap_ocupacion.parquet` / `.csv` | Heatmap de ocupación promedio y `frac_horas_saturadas` por día × hora |
| `data/m4/sankey_aristas.parquet` / `.csv` | **Sankey** listo para plotly: columnas `origen`, `destino`, `n` (nodos tipo `"1:urgencias_ingreso"`) |
| `data/m4/rutas_frecuentes.parquet` / `.csv` | Top 50 recorridos completos (p. ej. urgencias_ingreso → urgencias_observacion → hospitalizacion → egreso) |
| `data/m4/matriz_transicion.parquet` / `.csv` | Heatmap de la matriz de transición (probabilidades) |
| `data/m4/llegadas_dia.parquet` / `.csv` | Serie diaria con **festivos, puentes y día después de festivo**: efecto calendario y descomposición STL |
| `data/m4/ocupacion_dia.parquet` / `.csv` | Serie diaria de ocupación (promedio y máximo) |
| `data/m4/calendario_dia.parquet`, `calendario_hora.parquet` | Calendario colombiano 2022-12 → 2026-09: festivos (Ley Emiliani), puentes, fin de semana y **turno** (mañana 7–13, tarde 13–19, noche 19–7) |
| `data/m4/transicion.parquet` | Nivel fila (3.9 M transiciones) por si necesitas otro corte |

Hallazgos rápidos para arrancar:
- **Rutas:**
  - El 65 % de las visitas son ambulatorias directas.
  - El 17 % sigue urgencias_ingreso → observación → egreso.
  - Solo el 1.8 % llega a hospitalización desde urgencias.
- **Efecto festivo:** las llegadas a urgencias en festivo (~758/día en observación) son iguales a las de un fin de semana (~763) y un 24 % menores que en día laboral (~999). El día después de un festivo hay un **rebote**: 1 067/día, un 7 % sobre lo normal.
- **Tendencia del censo hospitalario:** sube de ~265 (2023) a ~315 (2025). Conviene mostrarlo con STL.

## Cómo regenerar (si cambian los parámetros)

Solo Persona 1 necesita la VPN. Con los parquets del paquete ya puedes trabajar sin conexión.

```bash
python m1_etl/run_etl.py construir
python m4_espaciotemporal/run_m4.py --paquete
```

Reportes de validación incluidos: `outputs/m1_validaciones.md` (10 checks anti-fuga) y `outputs/m4_resumen.md` (5 checks del M4 + resúmenes).
