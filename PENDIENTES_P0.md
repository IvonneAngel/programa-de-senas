# PENDIENTES P0 - Antes de entrenar (4 auditorías distintas)

## Seguridad (BAJO-MEDIO)
- [ ] P0: `tf.keras.models.load_model` con safe-mode (bloquear Lambda)
- [ ] P0: `servidor_interfaz.py` limitar `Content-Length` y tamaño imagen en `/api/prediccion-frame` (DoS base64)
- [ ] P0: Validar `subprocess.Popen` en `/api/abrir-traductor` (solo 127.0.0.1)

## Performance
- [ ] P0: `tcn_legacy.py` 2466 líneas → usar split `tcn_part1..5` (74 clases) y borrar legacy como fuente
- [ ] P0: `train_classifier.py:main` 350 y `run_epoch` 280 → partir en helpers (ya hay TODO)
- [ ] P0: DataLoader añadir `prefetch_factor`, `persistent_workers`, `pin_memory`

## Datos
- [ ] P0: Completar 5 manifests (faltan 2 csv), unificar `label_map.json` 249 vs `labels.json` 145 (104 perdidos)
- [ ] P0: Implementar splits S01-S07/S08/S09 (ahora P01-P11, 0 filas ok)
- [ ] P0: Añadir `SHA256` por foto + `is_duplicate` para evitar 11-12x stem duplicado

## UX
- [ ] P0: `App.tsx` estados vacíos `blocked/idle` → mostrar mensaje + aria-live
- [ ] P0: `panel.ps1` carrera puerto 8765 → esperar a que `127.0.0.1:8765` esté listo antes de abrir browser
- [ ] P0: Ruta `Downloads\proyecto de señas archivos externos` frágil → mover a `app/assets/models` (ya hay .task ahí) y usar relativo

> Todo pendiente, no aplicado aún como pediste.

## Siguiente repo (pendiente, foco letras)
- [ ] Zenodo 18330565 (121 glosas dinámicas) https://zenodo.org/records/18330565 - PENDIENTE, no instalar ahora, foco en letras (21 clases, 0.595)
