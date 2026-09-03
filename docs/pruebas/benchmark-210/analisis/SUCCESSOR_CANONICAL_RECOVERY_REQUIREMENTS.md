# Requisitos de restauración canónica del benchmark sucesor

Este documento define qué debe verificarse antes de considerar una copia restaurada equivalente al benchmark sucesor histórico. No autoriza evaluar S09 ni reemplazar la línea de recuperación.

| Artefacto solicitado | Validación mínima | Motivo |
|---|---|---|
| `successor_mendeley_positions126_model_ready_extracted.csv` | 1,857 filas totales; 1,437 train `ok`, 210 validation `ok`, 210 test `ok`, 33 train excluidas; 210 etiquetas | Fija los clips y exclusiones originales |
| `data/cache/successor_mendeley_positions126/` | Cada fila `ok` tiene tensor finito `(30,126)` y ruta existente | Permite comparar exactamente el control original |
| Extractor y fixtures | Versión MediaPipe, archivo `.task`, umbrales, lateralidad, remuestreo, geometría y criterio de `excluded` | Explica y reproduce el contrato de características |
| `tests/` y `src/lsm/training/train_classifier.py` originales | Suite completa y configuración de tarea reproducible | Evita sustituir comportamiento de entrenamiento por el arnés de recuperación |
| `runs/` de control | Semillas 42/13/21 y `metrics.json` S08 | Permite contrastar resultados, no inferirlos |

## Pruebas de admisión

La copia se admitirá solo si cumple simultáneamente lo siguiente:

1. No contiene S09 en los pasos de selección, tuning o métricas previas.
2. El manifiesto reproduce los conteos anteriores sin editar filas manualmente.
3. Una muestra fija de tensores coincide bit a bit con fixtures o, si las diferencias de plataforma son inevitables, por tolerancia documentada.
4. El control de s42 reproduce una macro-F1 S08 cercana a `0.088413` bajo el entorno fijado; las semillas s13/s21 se reservan para confirmar la recuperación.
5. Solo después se reanudan candidatas con la puerta histórica de superioridad y S09 continúa cerrado.

> La caché de recuperación, la de umbral legado y sus resultados no satisfacen estas pruebas. Son trazas diagnósticas, no sustitutos del benchmark canónico.
