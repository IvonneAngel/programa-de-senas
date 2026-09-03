# Pipeline Descarga-Procesa-Entrena

1. Descarga: `dataset/raw/` (ej: sjt79hnb2f.zip)
2. Procesa: `python pipeline_descarga_procesa_entrena.py` -> `dataset/landmarks/*.npy` + borra `*.jpg`
3. Entrena: desde `dataset/landmarks` sin reprocesar
4. Prueba: `python test_accuracy.py` -> grafica en `docs/graficas/accuracy_curve.png`
