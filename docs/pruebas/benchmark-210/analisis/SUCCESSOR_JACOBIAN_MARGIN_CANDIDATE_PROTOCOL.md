# Protocolo preregistrado: `successor_jacobian_margin`

## Hipótesis

La caché recuperada muestra presencia manual variable por firmante y el control exhibe una brecha marcada entre macro-F1 de entrenamiento y S08. Se propone reducir sensibilidad local de los logits a perturbaciones infinitesimales de landmarks reales, sin sintetizar clips ni modificar coordenadas. La regularización Jacobiana aumenta márgenes de clasificación y puede mejorar robustez a corrupción de entrada sin deteriorar fuertemente la generalización limpia [1].

La candidata conserva el mismo `TemporalTCN` de 158,994 parámetros, la entrada `(30,126)`, los clips, el orden temporal, las etiquetas one-hot, AdamW, tasa constante `0.002`, lote 64, 40 épocas, `weight_decay=0.0001`, S01–S07 para train y S08 exclusivamente para selección. S09 queda cerrado.

## Intervención cerrada

Para cada lote de entrenamiento se computan logits `z=fθ(x)` y un vector de proyección Rademacher fijo por semilla `v∈{-1,+1}^210/√210`. La pérdida es:

> `L = CrossEntropy(z,y) + 80 × mean((∂(vᵀz)/∂x)²)`.

La proyección única evita formar la Jacobiana completa de 210 salidas. La intensidad `80` se fijó **antes de S08**, usando un único lote S01–S07 con semilla 42: `CE=5.342930`, penalización `0.000668207`, de modo que el término Jacobiano inicial equivale aproximadamente al 1% de CrossEntropy. No se ajustará tras observar s42.

| Elemento | Control de recuperación | Candidata |
|---|---|---|
| Datos / split | Idénticos | Idénticos |
| TCN / parámetros | 158,994 | Idénticos |
| CrossEntropy one-hot | Sí | Sí, más penalización Jacobiana solo en train |
| Aumentos, filtro, curriculum | Ninguno | Ninguno |
| S08 | Selección | Selección |
| S09 | Cerrado | Cerrado |

## Controles y puerta

Las corridas de validación no computarán la penalización Jacobiana ni crearán gradientes de entrada. s42 debe alcanzar macro-F1 S08 `≥0.098413` para habilitar s13 y s21. Para declarar superioridad se exige también una mejora `≥+0.010` contra cada control de recuperación emparejado; S09 se abre solo si las tres semillas cumplen ambos requisitos.

## Referencias

[1] [Hoffman, Roberts y Yaida, *Robust Learning with Jacobian Regularization*](https://arxiv.org/abs/1908.02729).

[2] [Implementación de referencia de Jacobian Regularization en PyTorch](https://github.com/facebookresearch/jacobian_regularizer).

## Resultado de recuperación

En s42, la candidata alcanzó macro-F1 S08 `0.067698`, frente a `0.075556` del control de recuperación; delta `−0.007857`. Se rechaza temprano sin s13/s21 ni S09.
