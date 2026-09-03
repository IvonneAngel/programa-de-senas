# Auditoría del estado vigente del repositorio

**Repositorio:** `IvonneAngel/programa-de-senas`  
**Commit de partida:** `e0bb5ac`  
**Fecha de revisión:** 3 de septiembre de 2026

## Resultado de la auditoría

La versión actual contiene una interfaz React/Vite con cámara, captura local, exportación de dataset y autocompletado. La interfaz reenvía las solicitudes de predicción a `127.0.0.1:8765`, pero el backend que debería atenderlas no está presente en el repositorio activo. Por ello, la aplicación puede compilar y preparar el flujo de integración, pero no se puede afirmar que la predicción funcione de extremo a extremo.

El autocompletado no es un reconocedor. `app/src/components/GhostWord.tsx` carga el diccionario local y muestra una palabra que coincide con el prefijo recibido. No usa un modelo, no calcula una confianza de reconocimiento y no confirma que la palabra sugerida sea la que el usuario quiso expresar.

El modelo de investigación vigente está implementado en PyTorch. `entrenador/core-pt/lsm/models/tcn.py` define un TCN temporal con entrada `(batch, frames, features)` y `build_model("successor_positions126")` instancia la arquitectura base con características de 126 valores por cuadro. El runtime TensorFlow/TFLite de `core/modelo/usar_modelo.py` es una línea heredada distinta y no está conectado con la app React actual.

## Experimentos separados

| Experimento | Evidencia | Estado |
|---|---|---|
| `successor_positions126` | `docs/analisis/fase2-entrenamiento/metrics.json`; 249 etiquetas, forma `[30,126]`, partición 1,675/379/393 | Prueba con accuracy 0.0 y macro-F1 0.0; resultado insuficiente. |
| MSL-ABC | `docs/analisis/fase2-entrenamiento-msl-abc/ANALISIS.md`; 279,716 imágenes, TCN de 74 clases, `best.pt` de 548,701 bytes | Entrenamiento interrumpido durante la primera época; accuracy 0.048 y macro-F1 0.038 registradas para el checkpoint parcial. Sin evaluación final ni backend. |
| Mapa de etiquetas | `dataset/manifests/label_map.json`; 249 entradas nominales | Inventario de nombres, no evidencia de que exista un modelo funcional de 249 clases. |

No se deben mezclar estas cifras ni presentarlas como un único modelo. En particular, el mapa nominal de 249 etiquetas no convierte al checkpoint MSL-ABC de 74 clases en un modelo de 249 palabras.

## Cambios aplicados

Se añadió `app/src/components/TrainButton.tsx`, que faltaba aunque `TrainingPanel.tsx` lo importaba. Se mejoró el manejo de errores de `/api/entrenar` para que la interfaz no quede bloqueada cuando el backend no está conectado. También se añadió `deleteOldestFrames` a IndexedDB y se sustituyó el borrado simulado por una operación real sobre los frames más antiguos. El conteo del dataset ahora se actualiza después de capturar frames.

Se actualizaron los README raíz, de la aplicación y del entrenador. Se corrigieron los README de las fases de entrenamiento y evaluación. Se reemplazó `docs/protocolos/protocolo_LSM_v22.docx` por una versión que distingue interfaz, autocompletado, TCN experimental, runtime heredado y backend ausente. Los pendientes P0 se reorganizaron para priorizar la integración real y la evaluación por firmante.

Se creó y validó la skill reutilizable `auditoria-redaccion-evidencia` en `/home/ubuntu/skills/auditoria-redaccion-evidencia/SKILL.md`. Su objetivo es impedir que futuras revisiones mezclen modelos, rutas, métricas o versiones y orientar una redacción natural basada en evidencia.

## Verificaciones realizadas

| Verificación | Resultado |
|---|---|
| `pnpm exec tsc --noEmit` en `app/` | Correcta. |
| `pnpm run build` en `app/` | Correcta; Vite produjo el bundle de producción. |
| `unzip -t docs/protocolos/protocolo_LSM_v22.docx` | Correcta; no se detectaron errores en el archivo Word. |
| `git diff --check` | Correcta; no se detectaron errores de whitespace. |
| `quick_validate.py auditoria-redaccion-evidencia` | Correcta. |
| `sistema/test_auto.py` | No representa el estado actual: contiene rutas Windows, busca `sistema/auto_extract.py` que no está presente y depende de `psutil`, que no aparece en `entrenador/requirements.txt`. |

## Pendientes que no se deben ocultar

Todavía falta implementar o incorporar el backend de predicción, completar y evaluar el entrenamiento MSL-ABC, validar la separación por firmante, unificar el contrato de características entre cámara y TCN, y añadir confianza, alternativas y rechazo para la salida. Esos pendientes son el siguiente tramo del proyecto; no justifican volver a describir el sistema antiguo como si fuera el vigente.
