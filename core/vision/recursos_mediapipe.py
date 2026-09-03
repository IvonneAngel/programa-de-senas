from __future__ import annotations

import shutil
import sys
from importlib import invalidate_caches
from importlib.util import find_spec
from pathlib import Path
from typing import Any


def _has_mediapipe_bindings(package_dir: Path) -> bool:
    python_dir = package_dir / "python"
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return python_dir.exists() and any(tag in path.name for path in python_dir.glob("_framework_bindings*"))


def _patch_mediapipe_init(package_dir: Path) -> None:
    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        return
    text = init_file.read_text(encoding="utf-8")
    old = "import mediapipe.tasks.python as tasks\n"
    if old not in text:
        return
    new = "tasks = None  # LEGACY: Holistic ya migrado a Tasks, no anular si se usa Tasks\n"
    init_file.write_text(text.replace(old, new), encoding="utf-8")


def mediapipe_ascii_site_dir() -> Path:
    target_site = Path.home() / "Downloads" / "proyecto_senas_mediapipe_site"
    target_package = target_site / "mediapipe"
    if target_package.exists() and _has_mediapipe_bindings(target_package):
        _patch_mediapipe_init(target_package)
        return target_site

    if target_package.exists():
        shutil.rmtree(target_package)
        invalidate_caches()

    spec = find_spec("mediapipe")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError("No encontre el paquete mediapipe instalado.")

    package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    if package_root == target_package:
        raise ModuleNotFoundError("El cache ASCII de mediapipe esta incompleto y no encontre otra instalacion valida.")

    target_site.mkdir(parents=True, exist_ok=True)
    shutil.copytree(package_root, target_package, dirs_exist_ok=True)
    if not _has_mediapipe_bindings(target_package):
        raise ModuleNotFoundError("No se pudieron copiar los binarios de mediapipe al cache ASCII.")
    _patch_mediapipe_init(target_package)
    return target_site


def prepare_mediapipe_import_path() -> Path:
    site_dir = mediapipe_ascii_site_dir()
    site_text = str(site_dir)
    if site_text not in sys.path:
        sys.path.insert(0, site_text)
    return site_dir


def mediapipe_resource_dir(mp_module: Any) -> Path:
    target = Path.home() / "Downloads" / "proyecto_senas_mediapipe_recursos"
    modules_target = target / "modules"
    if modules_target.exists():
        return target

    package_root = Path(mp_module.__file__).resolve().parent
    modules_source = package_root / "modules"
    if not modules_source.exists():
        raise FileNotFoundError(f"No encontre recursos de MediaPipe en {modules_source}")

    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(modules_source, modules_target, dirs_exist_ok=True)
    return target


def configure_mediapipe_resources(mp_module: Any) -> Path:
    from mediapipe.python._framework_bindings import resource_util

    resource_dir = mediapipe_resource_dir(mp_module)
    resource_util.set_resource_dir(str(resource_dir))
    holistic = getattr(getattr(mp_module, "solutions", None), "holistic", None)
    if holistic is not None and hasattr(holistic, "_BINARYPB_FILE_PATH"):
        holistic._BINARYPB_FILE_PATH = str(
            resource_dir / "modules" / "holistic_landmark" / "holistic_landmark_cpu.binarypb"
        )
    hands = getattr(getattr(mp_module, "solutions", None), "hands", None)
    if hands is not None and hasattr(hands, "_BINARYPB_FILE_PATH"):
        hands._BINARYPB_FILE_PATH = str(
            resource_dir / "modules" / "hand_landmark" / "hand_landmark_tracking_cpu.binarypb"
        )
    return resource_dir