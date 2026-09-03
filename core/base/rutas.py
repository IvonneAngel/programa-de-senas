from __future__ import annotations

import os
from pathlib import Path


def carpeta_proyecto() -> Path:
    return Path(__file__).resolve().parents[1]


def carpeta_base() -> Path:
    """carpeta base."""
    current = carpeta_proyecto()
    for path in (current, *current.parents):
        if (path / "app").exists() and (path / "core").exists():
            return path
        if (path / "dataset").exists() and (path / "entrenador").exists():
            return path
    return current


    """ruta pruebas."""
def ruta_pruebas() -> Path:
    env_path = os.environ.get("PROYECTO_SENAS_PRUEBAS")
    if env_path:
        return Path(env_path)

    downloads_path = Path.home() / "Downloads" / "proyecto de señas pruebas"
    if downloads_path.exists():
        return downloads_path

    return carpeta_base() / "pruebas"

    """ruta archivos externos."""

def ruta_archivos_externos() -> Path:
    env_path = os.environ.get("PROYECTO_SENAS_EXTERNOS")
    if env_path:
        return Path(env_path)

    downloads_path = Path.home() / "Downloads" / "proyecto de señas archivos externos"
    if downloads_path.exists():
        return downloads_path

    return ruta_pruebas()


def ruta_reportes() -> Path:
    return ruta_archivos_externos() / "reportes"


def ruta_imagenes() -> Path:
    local = carpeta_base() / "imagenes de entrenamiento"
    if local.exists():
        return local
    return ruta_archivos_externos() / "imagenes"


def ruta_fuentes() -> Path:
    externos = ruta_archivos_externos()
    nueva = externos / "fuentes externas"
    if nueva.exists():
        return nueva
    return externos / "fuentes"


def ruta_videos() -> Path:
    return ruta_archivos_externos() / "videos"


def ruta_sesiones() -> Path:
    return ruta_archivos_externos() / "sesiones"


def ruta_entrenamiento() -> Path:
    local = carpeta_base() / "datos procesados del modelo"
    if local.exists():
        return local
    return ruta_pruebas() / "entrenamiento"


def ruta_cuarentena() -> Path:
    return ruta_archivos_externos() / "cuarentena"


def ruta_logs() -> Path:
    return ruta_archivos_externos() / "logs"


def texto_relativo_a_proyecto(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(carpeta_base().resolve()))
    except ValueError:
        return str(path)