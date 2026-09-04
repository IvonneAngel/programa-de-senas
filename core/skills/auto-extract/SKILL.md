---
name: auto-extract
description: Extraccion automatica 249 clases MSL con Pool 14 (6P+8E) + RTX 4060, 30f->bone_vector126, sin manual
---

# auto-extract

Sistema automatico: extrae ZIP, ordena 249 carpetas, landmarks MediaPipe HandLandmarker 30f->bone_vector126 con Pool 14 workers, manifests y docs/pruebas.

Uso: `python -m sistema.auto_extract` (sin --help manual)

Ponytail: todo en carpetas/subcarpetas, sin dispersion:
- raw: dataset/raw/mendeley_6rj76z6y3n/
- processed: dataset/processed/mendeley_6rj76z6y3n/bone_vector126/
- manifests: dataset/manifests/
- docs: docs/analisis/fase1-4 y docs/pruebas/mendeley_6rj76z6y3n/
