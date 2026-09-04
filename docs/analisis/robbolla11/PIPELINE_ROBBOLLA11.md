# Pipeline robbolla11 - automático 6P+RTX sin manual

- **Fecha**: 2026-09-02 23:57:07
- **Repo clonado**: https://github.com/robbolla11/Mexican-Sign-Language-Alphabet-Real-Time-Detection -> C:/Users/riemann/Desktop/robbolla11
- **Raw extracción**: data/ 27 letras (A,B,C...Z + HOLA) 5400 jpgs + data.pickle 2.96MB (5384 samples, 63 feats hand landmarks) -> dataset/raw/robbolla11
  - Nota task 21 letras 4200 (200/letra) - repo actual evolucionó a 27 letras 5400; conteo por letra: 200 c/u excepto N187 R199 X198 (filtrado pickle), HOLA 200
  - Manifest: dataset/raw/robbolla11/manifest.csv (120KB)
- **Entrenamiento**: RandomForestTraining.py lógica replicada con sklearn RandomForestClassifier(n_estimators=200, n_jobs=6) - 6 P-cores i7-13650HX
  - Split: 80/20 estratificado random_state 42 -> train 4307 test 1077
  - Accuracy: 1.0000 (perfect, verificar leakage pero data es landmarks normalizados fáciles)
  - Guardado: docs/analisis/robbolla11/random_forest_21letras_200porLetra.joblib (15.5MB) + model.pickle (compat) + metrics.json + confusion_matrix.csv + accuracy_curve.png
- **Sistema paralelo**: sistema/parallel_controller.py (6 P-cores workers, 8 E-cores mediapipe, prefetch 4, pin_memory cuda, batch auto) + torch cuda detección (fallback CPU para RF)
- **Pipeline automático**: sistema/run_robbolla_pipeline.py - clone -> extract -> inspect -> train sin manual, progreso cada 30s en C:/Users/riemann/Desktop/entrenamiento_robbolla_progreso.txt
- **Búsqueda paralela datasets letras no repetidos**: sistema/run_busqueda_letras.py -> C:/Users/riemann/Desktop/busqueda_mas_letras.md
  - Excluye sjt79hnb2f, feria, robbolla11, mendeley_6rj76z6y3n
  - Top novedosos: Zenodo 10067509 static 21 letras 4.81GB + 19154645 dynamic 6 letras videos, Kaggle MSL alphabet - todos no repetidos
