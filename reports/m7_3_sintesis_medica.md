# Síntesis clínica de la explicabilidad del modelo (Subfase 7.3)

Modelo explicado: predicción de **estancia prolongada** (que la estancia supere el percentil 75 de su servicio), evaluado en pacientes que el modelo no vio durante el entrenamiento.

Riesgo promedio de partida: 53%. Cada variable sube o baja el riesgo de cada paciente a partir de ese punto.

## Las 5 variables más influyentes en todo el hospital

1. Servicio (hospitalización vs urgencias) (admisión; 27 % de la influencia total). Estar en hospitalización se asocia con un riesgo menor de que la estancia supere lo habitual para su servicio. Este efecto refleja sobre todo cómo se define la estancia prolongada: el umbral es propio de cada servicio y, en urgencias, quien sigue allí a las 2 o 6 horas ya es un paciente de estancia larga. No significa que hospitalizar acorte la estancia.
2. Momento de la evaluación (0, 2 o 6 h tras el ingreso) (admisión; 16 % de la influencia total). Evaluar al paciente más horas después del ingreso se asocia con un riesgo mayor de que la estancia supere lo habitual para su servicio. Que el paciente siga en el servicio horas después del ingreso es en sí información pronóstica: cuanto más tiempo lleva, más probable es que su estancia sea larga.
3. Perfil de diagnósticos (historia clínica) (historia clínica; 10 % de la influencia total). Importa más qué diagnósticos tiene el paciente que cuántos. Entre los más frecuentes, los que más elevan el riesgo son: infección localizada de piel o tejido subcutáneo; cólico renal; infección de vías urinarias. Los que más lo reducen: amenaza de aborto; falso trabajo de parto a término (≥ 37 semanas); falso trabajo de parto.
4. Diagnósticos en los últimos 30 días (historia clínica; 8 % de la influencia total). Tener más diagnósticos recientes se asocia con un riesgo mayor de que la estancia supere lo habitual para su servicio.
5. Edad (admisión; 6 % de la influencia total). Mayor edad se asocia con un riesgo mayor de que la estancia supere lo habitual para su servicio.

## Dentro de hospitalización

1. Perfil de diagnósticos (historia clínica) (historia clínica; 19 % de la influencia total). Importa más qué diagnósticos tiene el paciente que cuántos. Entre los más frecuentes, los que más elevan el riesgo son: infección localizada de piel o tejido subcutáneo; infección aguda de vías respiratorias bajas; dolor abdominal superior. Los que más lo reducen: falso trabajo de parto a término (≥ 37 semanas); falso trabajo de parto; parto por cesárea.
2. Llegadas al servicio en las últimas 24 h (presión del servicio; 18 % de la influencia total). Su efecto no es lineal: según el rango de valores puede aumentar o disminuir el riesgo, por lo que debe leerse junto con el resto del cuadro.
3. Momento de la evaluación (0, 2 o 6 h tras el ingreso) (admisión; 10 % de la influencia total). Evaluar al paciente más horas después del ingreso se asocia con un riesgo mayor de que la estancia supere lo habitual para su servicio. Que el paciente siga en el servicio horas después del ingreso es en sí información pronóstica: cuanto más tiempo lleva, más probable es que su estancia sea larga.
4. Edad (admisión; 9 % de la influencia total). Mayor edad se asocia con un riesgo mayor de que la estancia supere lo habitual para su servicio.
5. Visitas al hospital en el último año (historia clínica; 7 % de la influencia total). Tener más visitas en el último año se asocia con un riesgo menor de que la estancia supere lo habitual para su servicio.

## Dentro de urgencias / observación

1. Momento de la evaluación (0, 2 o 6 h tras el ingreso) (admisión; 25 % de la influencia total). Evaluar al paciente más horas después del ingreso se asocia con un riesgo mayor de que la estancia supere lo habitual para su servicio. Que el paciente siga en el servicio horas después del ingreso es en sí información pronóstica: cuanto más tiempo lleva, más probable es que su estancia sea larga.
2. Perfil de diagnósticos (historia clínica) (historia clínica; 12 % de la influencia total). Importa más qué diagnósticos tiene el paciente que cuántos. Entre los más frecuentes, los que más elevan el riesgo son: infección localizada de piel o tejido subcutáneo; cólico renal; insuficiencia cardíaca congestiva. Los que más lo reducen: amenaza de aborto; hemorragia del tracto genitourinario; dolor pélvico y perineal.
3. Diagnósticos en los últimos 30 días (historia clínica; 11 % de la influencia total). Tener más diagnósticos recientes se asocia con un riesgo mayor de que la estancia supere lo habitual para su servicio.
4. Edad (admisión; 8 % de la influencia total). Mayor edad se asocia con un riesgo mayor de que la estancia supere lo habitual para su servicio.
5. Hora del día (calendario; 7 % de la influencia total). Una hora más tardía se asocia con un riesgo menor de que la estancia supere lo habitual para su servicio.

## Diagnósticos que más mueven el riesgo dentro de cada servicio

- **Hospitalización, elevan el riesgo:** infección localizada de piel o tejido subcutáneo; infección aguda de vías respiratorias bajas; dolor abdominal superior; neumonía; hipertensión esencial
- **Hospitalización, reducen el riesgo:** falso trabajo de parto a término (≥ 37 semanas); falso trabajo de parto; parto por cesárea; cesárea de urgencia; dolor pélvico y perineal
- **Urgencias / observación, elevan el riesgo:** infección localizada de piel o tejido subcutáneo; cólico renal; insuficiencia cardíaca congestiva; infección de vías urinarias; hemorragia gastrointestinal
- **Urgencias / observación, reducen el riesgo:** amenaza de aborto; hemorragia del tracto genitourinario; dolor pélvico y perineal; falso trabajo de parto a término (≥ 37 semanas); falso trabajo de parto pretérmino (< 37 semanas)

## Perfiles de mayor y menor riesgo dentro de cada servicio

- **Hospitalización, mayor riesgo** (riesgo estimado 57%, observado 59%): perfil de diagnósticos (4 diagnósticos), diagnósticos en los últimos 30 días (259), llegadas al servicio en las últimas 24 h (88), edad (68 años).
- **Hospitalización, menor riesgo** (riesgo estimado 4%, observado 4%): perfil de diagnósticos (2 diagnósticos), llegadas al servicio en las últimas 24 h (91), edad (24 años), diagnósticos en los últimos 30 días (132).
- **Urgencias / observación, mayor riesgo** (riesgo estimado 85%, observado 91%): momento de la evaluación (6 h), perfil de diagnósticos (2 diagnósticos), diagnósticos en los últimos 30 días (36), edad (49 años).
- **Urgencias / observación, menor riesgo** (riesgo estimado 36%, observado 31%): momento de la evaluación (0 h), perfil de diagnósticos (1 diagnóstico), diagnósticos en los últimos 30 días (3.0), edad (25 años).

## Cómo leerlo

- Son asociaciones aprendidas de los registros, no relaciones causales.
- El registro clínico del hospital no incluye nivel de triaje, por lo que el modelo no lo usa.
- La presión del servicio y el calendario describen el contexto operativo, no el estado clínico del paciente.
- Los perfiles son promedios de grupos de pacientes; la explicación individual se calcula para cada paciente en el momento de usar el modelo.