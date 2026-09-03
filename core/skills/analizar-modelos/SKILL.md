---
name: analizar-modelos
description: Analiza modelos bajos vs mejor realista con líneas simples
---

# analizar-modelos

Analiza modelos bajos vs mejor realista con líneas simples

## Uso

- Lista modelos en `docs/pruebas/*/modelo/*.keras`
- Separa bajos (F1<0.6) vs mejor realista (F1>=0.9)
- Genera `analisis.md` con 1 línea por modelo
- Ejecuta: `python core/skills/analizar-modelos/run.py`

## Contrato

Entrada: `docs/pruebas/*/modelo/` con .keras
Salida: reporte corto sin texto largo

