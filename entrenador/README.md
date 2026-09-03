# Entrenador PyTorch

Este directorio contiene la línea de investigación vigente basada en PyTorch. El modelo temporal base está en `core-pt/lsm/models/tcn.py`; sus entradas respetan el contrato `(batch, frames, features)` y `TemporalTCN` procesa secuencias de 30 cuadros cuando la tarea usa `bone_vector126`.

## Componentes principales

| Ruta | Función |
|---|---|
| `core-pt/lsm/models/tcn.py` | TCN base, variantes temporales y fábrica `build_model`. |
| `core-pt/lsm/training/train_classifier.py` | Lectura de manifiestos, entrenamiento, métricas y checkpoints. |
| `scripts/bone_vector126.py` | Utilidades relacionadas con la representación de 126 valores por cuadro. |
| `scripts/` | Scripts de extracción, validación, exportación y experimentos sucesores. |
| `manifests/` | Documentación auxiliar de manifiestos; las fuentes grandes se mantienen separadas. |

## Experimentos que no deben mezclarse

`successor_positions126` usa 249 etiquetas y entradas de forma `(30, 126)`. El resultado versionado en `docs/analisis/fase2-entrenamiento/metrics.json` obtuvo 0% de accuracy y 0% de macro-F1 en prueba. El experimento MSL-ABC de `docs/analisis/fase2-entrenamiento-msl-abc/ANALISIS.md` es diferente: reporta 74 clases, un checkpoint de 548,701 bytes y entrenamiento cortado durante la primera época.

Estos modelos no están conectados automáticamente con la interfaz React. La aplicación actual reenvía las solicitudes a un backend esperado en `127.0.0.1:8765`; no se debe afirmar que el entrenador se integra con la cámara hasta que ese backend exista y reproduzca exactamente la extracción de características.

## Reproducibilidad

Cada nueva corrida debe conservar la tarea, el manifiesto, la forma de entrada, el número de clases, la partición por firmante, la semilla, el dispositivo, el checkpoint y las métricas de validación y prueba. No usar una ruta bajo `_desactualizado/` como fuente del sistema actual.
