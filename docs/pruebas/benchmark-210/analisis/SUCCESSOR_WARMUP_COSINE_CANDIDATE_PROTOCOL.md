# Protocolo preregistrado: `successor_warmup_cosine`

## Hipótesis

El control sucesor usa AdamW con tasa de aprendizaje constante `0.002` durante 40 épocas. La candidata modifica exclusivamente el trayecto de optimización: un warmup lineal de cuatro épocas seguido por decaimiento coseno monótono de ciclo único. La intervención busca estabilizar las primeras actualizaciones con 5–7 ejemplos reales por clase y consolidar al final; no altera los ejemplos, sus etiquetas, la representación, el modelo ni la inferencia.

## Configuración cerrada

| Elemento | Especificación preregistrada |
|---|---|
| Tarea | `successor_warmup_cosine` |
| Entrada | `(30,126)` sin transformación |
| Modelo | `TemporalTCN` de control: 158,994 parámetros, dilataciones `(1,2,4,8)`, GroupNorm y dropout 0.20 |
| Datos | S01–S07 únicamente durante train; S08 para selección; S09 cerrado |
| Pérdida | CrossEntropy one-hot sin pesos, margen, smoothing ni auxiliar |
| Optimizador | AdamW, `weight_decay=0.0001` |
| Épocas / lote | 40 / 64 |
| Tasa de inicio | `0.0002` |
| Warmup | Lineal durante épocas 1–4 hasta `0.002` |
| Decaimiento | Coseno monótono, épocas 5–40, de `0.002` a `0.00002` |
| Reinicios / ciclos | Prohibidos |
| Paso del scheduler | Una vez, después de cada época de entrenamiento |
| Búsqueda de hiperparámetros | Prohibida; no se ajustará ningún valor después de s42 |

La fórmula del tramo coseno, para `u=(e-4)/36` y épocas `e=5,…,40`, es:

> `lr(e) = 0.00002 + 0.5 × (0.002 − 0.00002) × (1 + cos(πu))`.

No se implementa SGDR con reinicios. El uso de un coseno de ciclo único se limita a una programación de tasa reproducible, compatible con la semántica de `CosineAnnealingLR` de PyTorch [2].

## Reglas de evaluación

La selección permanece exclusivamente por mejor macro-F1 en S08. En s42, la candidata debe alcanzar al menos `0.098413` para habilitar s13. Para declarar superioridad, cada réplica debe superar el control emparejado por al menos `+0.010`: s42 ≥ `0.098413`, s13 ≥ `0.107937` y s21 ≥ `0.105238`. Si una semilla no cumple, se rechaza sin abrir S09. Si y solo si las tres cumplen, se evalúa S09 una vez y se calcula IC95% bootstrap agrupado por firmante.

## Exclusiones explícitas

No se permiten cambios de datos, currículos, filtros de calidad, sampling de clase/firmante, aumentos, datos sintéticos, pesos externos, pérdida auxiliar, cambios de arquitectura, averaging de pesos, normalización espectral ni reinicios. En consecuencia, el resultado es atribuible únicamente al calendario de tasa de aprendizaje.

## Referencias

[1] [Loshchilov y Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*](https://arxiv.org/abs/1608.03983).

[2] [PyTorch, *CosineAnnealingLR*](https://docs.pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html).

## Resultado de recuperación

La corrida de recuperación s42 produjo macro-F1 S08 `0.075556`, idéntico al control recuperado s42 y por debajo de la puerta absoluta `0.098413`. Se rechaza temprano; s13/s21 y S09 permanecen sin evaluar para esta candidata.
