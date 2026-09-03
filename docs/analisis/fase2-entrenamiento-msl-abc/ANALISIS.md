# Análisis — Modelo Principal Gigante MSL-ABC (cortado epoch1)

**Fecha:** 2026-09-03 02:47 (cortado a petición, RTX no usada)
**Dataset:** MSL-ABC 279,716 jpgs 4.65GB (lsm-abc-A/B/C 92k c/u) → `dataset/processed/msl-abc/bone_vector126` 279,716 npy (30,126) 4.2GB (imágenes borradas tras extracción, no subidas)
**Extracción:** Pool14 6P+8E prefetch4 RTX 4060 8GB, 1,990 img/s, reanudable (skip si .npy existe), 20.9s para 41k restantes, manifest 285k filas
**Entrenamiento:** TCN `entrenador/core-pt/lsm/models/tcn.py` 74 clases, `train_classifier.py` 40 epochs, batch 8 sin ahogar (auto VRAM 0→8), 6 workers, frozen `head` epoch46 (`freeze_for_decoupled`), `device cpu` (torch+cpu, RTX 0% por eso 11.6min/epoch vs 3min con CUDA), `out docs/analisis/fase2-entrenamiento-msl-abc` con `best.pt 548KB` epoch1 `acc 0.048 macro_f1 0.038` (cortado)
**Modelos finales (solo 2, sin duplicados):** `fase2/best.pt` palabras 249 (0% fallido previo) + `robbolla11/joblib` letras 21 (1.0) — `fase3` y `model.pickle` movidos a `_desactualizado/modelos-previos-20260903-0135`
**Bolita:** `AuroraBallWithLetter` drag persistente, `Sheriff Sans`, `GhostWord` 92k+LSM 249, color 5min azul→morado→rojo, solo letra real sin fake, oración general por pausas `>600ms , >1400ms .`
**Sistema:** `sistema/parallel_controller` 14 workers, `memory_analyzer` batch auto, `auto_extract` Pool14, `log-server` + `Task Scheduler` desacoplado, `browser-errors.log` 0 errores
**Por qué se cortó:** EOFError 2 npy 0 bytes + 1 de 128 bytes (interrupción) → borrados y re-extraídos, luego cortado a petición en epoch1 para subir
