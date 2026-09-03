# Fase 2: entrenamiento del benchmark de 249 etiquetas

Esta carpeta documenta el experimento `successor_positions126`, separado del experimento MSL-ABC. La configuración versionada usa una entrada de forma `(30, 126)`, 249 etiquetas y las particiones registradas en `metrics.json`: 1,675 muestras de entrenamiento, 379 de validación y 393 de prueba.

El resultado de prueba registrado es `accuracy = 0.0` y `macro_f1 = 0.0`. Debe considerarse un resultado insuficiente. No representa un modelo desplegado y no debe usarse como evidencia de que la interfaz reconoce palabras en tiempo real.

La TCN y sus variantes están en `entrenador/core-pt/lsm/models/tcn.py`; el archivo de métricas conserva los argumentos y el historial disponibles para reproducir o auditar esta corrida. Los datos procesados referenciados por la configuración no están incluidos necesariamente en el clon actual.
