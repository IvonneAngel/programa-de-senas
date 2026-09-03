# Pendientes prioritarios del estado vigente

## Integración funcional

- [ ] Implementar y versionar el backend que atienda `POST /api/prediccion-frame` en `127.0.0.1:8765`.
- [ ] Definir el contrato de entrada del backend: tamaño de imagen, extracción MediaPipe, ventana temporal y forma exacta del tensor.
- [ ] Conectar el TCN PyTorch o un modelo runtime validado con la interfaz React; no reutilizar el servidor bajo `_desactualizado/` como si fuera vigente.
- [ ] Implementar `POST /api/entrenar` o desactivar la acción hasta que exista un flujo de entrenamiento real y documentado.

## Modelo y evaluación

- [ ] Completar el entrenamiento MSL-ABC, que fue cortado durante la primera época, y registrar métricas finales de validación y prueba.
- [ ] Evaluar por firmante y conservar una separación de participantes que no se use para seleccionar candidatos.
- [ ] Mantener separados el experimento `successor_positions126` de 249 etiquetas y el experimento MSL-ABC de 74 clases.
- [ ] Validar forma, etiquetas, checkpoint y dispositivo antes de cargar un modelo en runtime.

## Datos

- [ ] Versionar o documentar de forma reproducible los manifiestos usados para el checkpoint MSL-ABC.
- [ ] Registrar participantes, licencia, duplicados y estado de cada muestra sin confundir el mapa nominal de 249 etiquetas con clases entrenadas.
- [ ] Mantener los datos procesados fuera del repositorio solo con una instrucción de reconstrucción verificable.

## Autocompletado y UX

- [ ] Añadir confianza visible, alternativas y rechazo para que la sugerencia no parezca una traducción confirmada.
- [ ] Mostrar estados `backend no conectado`, `modelo cargando`, `sin manos detectadas` y `predicción rechazada` con `aria-live`.
- [x] Corregir el import faltante de `TrainButton` y verificar TypeScript/build.
- [x] Sustituir el borrado simulado del panel por borrado real de frames antiguos en IndexedDB.

> La interfaz puede capturar y exportar datos aunque el backend de predicción todavía no esté incluido.
