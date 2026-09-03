# Interfaz web del proyecto LSM

Esta carpeta contiene la interfaz React/Vite del prototipo actual. La aplicación abre la cámara mediante `getUserMedia`, muestra el feed local, captura frames y prepara solicitudes hacia el servicio de predicción. El frontend no contiene por sí mismo el modelo ni un servidor de reconocimiento.

## Ejecución local

```bash
pnpm install --ignore-scripts
pnpm run dev
```

La interfaz queda disponible en el puerto configurado por Vite. Para que aparezcan predicciones, debe existir un backend que atienda `POST /api/prediccion-frame` en `http://127.0.0.1:8765`. El proxy está declarado en `vite.config.ts`; el servidor de ese puerto no forma parte del estado vigente de este repositorio.

## Autocompletado

`src/components/GhostWord.tsx` carga `public/diccionario_grande.txt` y `public/lsm_label_map.json`. Cuando la interfaz recibe una letra o un prefijo, muestra una sugerencia visual de palabra. Esa sugerencia es autocompletado, no reconocimiento directo, y no debe presentarse como una traducción confirmada.

## Captura y entrenamiento

El panel de entrenamiento guarda las imágenes capturadas en IndexedDB y permite exportarlas como ZIP. La acción de entrenamiento solicita `POST /api/entrenar`; si el backend no está conectado, la interfaz muestra el error y conserva la posibilidad de capturar y exportar datos. El borrado de frames antiguos se realiza dentro de IndexedDB y no vacía todo el dataset.

## Verificación

```bash
pnpm exec tsc --noEmit
pnpm run build
```

La aplicación puede compilarse con estas verificaciones. La compilación correcta no demuestra que la predicción esté conectada: esa parte requiere el backend y una prueba de extremo a extremo.
