---
name: validar-estructura
description: Valida que no haya cosas dispersas y todo esté en carpetas/subcarpetas
---

# validar-estructura

Valida que no haya cosas dispersas y todo esté en carpetas/subcarpetas

## Uso

- Verifica que no haya `__pycache__`, `*.pyc`, `codigo/` viejo
- Valida que todo esté en `app/core/entrenador/dataset/docs`
- Ejecuta: `python core/skills/validar-estructura/check.py`

## Contrato

Entrada: `programa de señas/` root
Salida: 0 dispersos, 0 duplicados

