# Programa de Señas

Proyecto independiente derivado de `lenguaje-de-senas-final`.

## Estructura
- `app/` → abre cámara: `app/aplicacion/servidor_interfaz.py` + `ejecutar_camara.py` + `interfaz-ui/` (App.tsx)
- `entrenador/` → `entrenar_abecedario.py`, `entrenar_modelo.py`, `importar_imagenes_lsm.py`
- `core/` → visión (`puntos.py`), modelo, datos
- `dataset/` → manifests + links.md (no jpgs)
- `docs/` → protocolos + pruebas con análisis

## Abrir app
```bat
app\abrir traductor.bat
# o
python app/aplicacion/servidor_interfaz.py --port 8765
# luego http://localhost:8765
```
