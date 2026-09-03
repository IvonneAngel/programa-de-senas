# Programa de señas

Proyecto experimental para el reconocimiento del Lenguaje de Señas Mexicano mediante puntos clave, modelos temporales y una interfaz de apoyo para formar texto.

## Estado actual

El repositorio contiene una interfaz React/Vite en `app/`, un núcleo Python en `core/`, el código de entrenamiento PyTorch en `entrenador/`, manifiestos en `dataset/` y reportes en `docs/`. La interfaz vigente puede abrir la cámara, capturar frames y usar un autocompletado local. El backend que debería atender `/api/prediccion-frame` en `127.0.0.1:8765` no está incluido en el estado actual, por lo que la predicción de extremo a extremo sigue pendiente.

El modelo temporal vigente de investigación se encuentra en `entrenador/core-pt/lsm/models/tcn.py`. El checkpoint MSL-ABC documentado en `docs/analisis/fase2-entrenamiento-msl-abc/ANALISIS.md` es experimental y su entrenamiento fue interrumpido durante la primera época. No debe confundirse con el runtime TensorFlow/TFLite heredado de `core/modelo/usar_modelo.py` ni con los archivos bajo `_desactualizado/`.

## Estructura

| Ruta | Contenido |
|---|---|
| `app/` | Interfaz React/Vite, cámara, autocompletado y captura local en IndexedDB. |
| `core/` | Extracción de puntos, datos, runtime heredado y utilidades del proyecto. |
| `entrenador/` | Modelos TCN PyTorch, entrenamiento, scripts y manifiestos auxiliares. |
| `dataset/` | Manifiestos y datos de entrada; los artefactos grandes no están todos versionados. |
| `docs/` | Protocolos, análisis de fases, métricas y pruebas. |
| `_desactualizado/` | Código histórico que no representa la integración vigente. |

## Ejecutar y verificar la interfaz

```bash
cd app
pnpm install --ignore-scripts
pnpm run dev
pnpm exec tsc --noEmit
pnpm run build
```

La captura y exportación del dataset pueden probarse sin backend. Para recibir predicciones o entrenar desde la interfaz hace falta implementar y levantar los endpoints documentados en `app/vite.config.ts`.
