# Protocolo preregistrado: `successor_partial_temporal_conv`

## Hipótesis

Los frames sin landmarks se representan como ceros, pero el TCN convencional no distingue ese sustituto de un patrón numérico real. La candidata deriva una máscara binaria temporal desde la misma entrada `(30,126)` y aplica convoluciones parciales: cada respuesta se calcula sobre posiciones válidas y se renormaliza por su soporte observado. No se imputan, interpolan ni generan landmarks.

La decisión está sustentada solo en S01–S07: 762 de 1,470 clips tienen por lo menos un frame manual ausente y S07 presenta presencia media de `0.5875`. No se usó S08/S09 para formular la hipótesis.

## Diseño cerrado

La máscara de entrada es `m_t = 1[sum_c |x_{t,c}| > 0]`. En cada convolución temporal con receptivo `K` y máscara de soporte `m`, se usa:

> `y_t = Conv(x·m)_t × expected_support_t / max(observed_support_t, 1)`

La máscara se actualiza como `1[observed_support_t>0]`. `expected_support` se calcula con una máscara llena y la misma dilatación/padding; con un clip completo la escala es exactamente uno, por lo que la candidata reproduce numéricamente el control si sus pesos se copian. Se conservan stem, dos convoluciones por bloque, GroupNorm, GELU, dropout, residual, dilataciones `(1,2,4,8)`, promedio global, cabeza, parámetros y CrossEntropy one-hot.

| Elemento | Control | Candidata |
|---|---|---|
| Entrada / API | `(30,126)` | Idéntica; máscara se deriva internamente |
| Parámetros | 158,994 | 158,994 |
| Datos, etiquetas, aumentos | Sin cambio | Sin cambio |
| Convolución temporal | Conv1d estándar | Conv1d parcial renormalizada |
| Clips completos | Control normal | Identidad numérica del control |
| Frames ausentes | Ceros tratados como datos | Excluidos localmente del soporte |
| S09 | Cerrado | Cerrado |

No se permiten filtros, pesos por presencia, pérdidas auxiliares, reconstrucción enmascarada, datos sintéticos, reindexación temporal, nuevos canales ni cambios de pooling.

## Puerta de evaluación

s42 debe alcanzar macro-F1 S08 `≥0.098413` para abrir s13/s21. La declaración de superioridad requiere además un delta `≥+0.010` contra cada control de recuperación emparejado. S09 solo se evaluará si las tres semillas cumplen ambas condiciones.

## Referencias

[1] [Liu et al., *Image Inpainting for Irregular Holes Using Partial Convolutions*](https://arxiv.org/abs/1804.07723).

[2] [Appel, *Efficient data-driven gap filling of satellite image time series using deep neural networks with partial convolutions*](https://arxiv.org/abs/2208.08781).

## Resultado de recuperación

En s42, la candidata alcanzó macro-F1 S08 `0.069788`, frente a `0.075556` del control; delta `−0.005767`. Se rechaza temprano sin s13/s21 ni S09.
