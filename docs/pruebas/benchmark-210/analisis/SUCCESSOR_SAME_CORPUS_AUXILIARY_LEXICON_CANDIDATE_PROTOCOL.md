# Protocolo preregistrado: `successor_same_corpus_auxiliary_lexicon`

## Hipótesis

El benchmark objetivo tiene 210 palabras y solo 5–7 clips de entrenamiento por palabra. El mismo corpus Mendeley contiene 39 palabras no incluidas en el benchmark porque no tienen la cobertura completa S01–S09, pero sí aportan 236 clips reales con al menos cuatro muestras por clase dentro de S01–S07. La hipótesis es que una fase supervisada desde cero sobre ese vocabulario **disjunto** puede enseñar al encoder configuraciones y trayectorias manuales de LSM antes de ajustar el vocabulario objetivo, sin introducir pesos ni datos externos.

La formulación corresponde al esquema de representación desde clases base hacia clases objetivo de pocos ejemplos; no se adopta contraste, memoria, transformaciones sintéticas ni clasificación por prototipos de esa literatura [1].

## Frontera de datos cerrada

| Población | Uso permitido |
|---|---|
| 39 clases auxiliares, S01–S07, 236 clips | Preentrenamiento supervisado desde cero |
| 210 clases objetivo, S01–S07 | Ajuste supervisado desde cero del clasificador objetivo |
| 210 clases objetivo, S08 | Selección por macro-F1 exclusivamente durante ajuste |
| S08/S09 auxiliares y objetivo | Prohibidos durante preentrenamiento; S09 permanece cerrado |

## Diseño técnico fijo

La extracción auxiliar usará el mismo vector `(30,126)` relativo a muñeca, mismo modelo MediaPipe de landmarks y mismo umbral `0.10` de la caché de recuperación. Se inicializa un `TemporalTCN` aleatorio de 39 clases y se entrena 40 épocas con AdamW, `lr=0.002`, `weight_decay=0.0001`, lote 64 y CrossEntropy one-hot. Luego se crea un nuevo `TemporalTCN` aleatorio de 210 clases; se transfieren únicamente stem, bloques residuales y proyección 64→128 del auxiliar, mientras el clasificador final `head.5` (128→210) se inicializa aleatoriamente. Se ajusta 40 épocas con el régimen exacto del control.

No hay consulta de S08 que determine la fase auxiliar, no se conservan logits auxiliares, no se usan pesos externos y no se crea ninguna muestra. Todos los parámetros comienzan aleatorios dentro de esta corrida.

## Puerta de evaluación

s42 debe obtener macro-F1 S08 `≥0.098413` para habilitar s13/s21. La superioridad requiere además `+0.010` sobre cada control de recuperación emparejado. S09 se abre únicamente tras cumplir ambas condiciones en las tres semillas.

## Referencias

[1] [Yin et al., *Rethinking the Sample Relations for Few-Shot Classification*](https://arxiv.org/html/2501.13418v1). Se usa solo como antecedente del escenario de clases base/nuevas; sus pérdidas contrastivas y transformaciones no forman parte del protocolo.

[2] [Espejel et al., *Mexican sign language dataset*, Mendeley Data V1](https://data.mendeley.com/datasets/6rj76z6y3n/1).

## Resultado de recuperación

En la repetición limpia de s42, la candidata obtuvo macro-F1 S08 `0.060329`, frente a `0.075556` del control; delta `−0.015227`. Se rechaza temprano sin s13/s21 ni S09.
