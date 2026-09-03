from __future__ import annotations


NO_REAL_MODEL_TEXT = "Modelo LSM no entrenado"
LOADING_MODEL_TEXT = "Cargando modelo"
LOW_CONFIDENCE_TEXT = "Modelo no seguro"
LOW_QUALITY_TEXT = "Palabra aun no validada"
WARMING_SEQUENCE_TEXT = "Leyendo sena"
SHOW_HAND_TEXT = "Muestra tu mano"


def display_phrase(text: str) -> str:
    """display phrase."""
    clean = str(text).replace("_", " ").strip()
    if not clean:
        return ""

    words = clean.split()
    formatted: list[str] = []
    for index, word in enumerate(words):
        if word.upper() == "LSM":
            formatted.append("LSM")
        elif index == 0:
            formatted.append(word[:1].upper() + word[1:].lower())
        else:
            formatted.append(word.lower())
    return " ".join(formatted)