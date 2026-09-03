# Pipeline de datos y entrenamiento

El pipeline está dividido en etapas que deben registrarse por separado: descarga o incorporación autorizada de datos, extracción de características, entrenamiento, evaluación y conexión con la aplicación. Los artefactos grandes de datos procesados no forman parte necesariamente del clon, porque `.npy` está excluido por `.gitignore`.

La documentación de cada corrida debe indicar el manifiesto, el tipo de representación, el número de clases, la forma del tensor, la separación de participantes, la semilla, el dispositivo, el checkpoint y las métricas obtenidas. En particular, `bone_vector126` describe una representación de 126 valores por cuadro; no identifica por sí sola un modelo listo para uso.

El resultado versionado de `docs/analisis/fase2-entrenamiento/metrics.json` corresponde a `successor_positions126` con 249 etiquetas y no debe mezclarse con el experimento MSL-ABC de 74 clases que aparece en `docs/analisis/fase2-entrenamiento-msl-abc/ANALISIS.md`.

La interfaz React se encuentra en `app/`. Su proxy espera un backend en `127.0.0.1:8765`, pero ese servidor no está incluido en la versión actual del repositorio. Por ello, ejecutar el entrenamiento o la extracción no implica que `/api/prediccion-frame` ya funcione en la aplicación.

## Verificaciones básicas

```bash
cd app
pnpm exec tsc --noEmit
pnpm run build
```

Las verificaciones anteriores cubren la compilación del frontend. La inferencia de cámara requiere además una prueba de extremo a extremo con el backend, el modelo y la extracción de características exactos.
