from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from time import strftime
from typing import Any


CRITICAL_MODULES = ["cv2", "mediapipe", "ai_edge_litert", "tensorflow", "numpy", "sklearn", "matplotlib"]


def venv_dir_path(app_dir: str | Path) -> Path:
    env_path = os.environ.get("PROYECTO_SENAS_VENV")
    if env_path:
        return Path(env_path)

    return Path.home() / "Downloads" / "proyecto de señas archivos externos" / "entorno python" / ".venv"


def venv_python_path(app_dir: str | Path) -> Path:
    root = venv_dir_path(app_dir)
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def requirements_file_path(app_dir: str | Path) -> Path:
    app_path = Path(app_dir)
    tools = app_path / "herramientas" / "requisitos.txt"
    if tools.exists():
        return tools
    nested = app_path / "sistema" / "requisitos.txt"
    if nested.exists():
        return nested
    return app_path / "requisitos.txt"


def build_import_probe_code(modules: list[str] | None = None) -> str:
    names = modules or CRITICAL_MODULES
    return (
        "import importlib.util, json\n"
        f"names = {names!r}\n"
        "print(json.dumps({name: importlib.util.find_spec(name) is not None for name in names}))\n"
    )


def _run_command(command: list[str], *, cwd: Path, timeout: int = 900) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
            "status": "pass" if completed.returncode == 0 else "fail",
        }
    except Exception as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "status": "fail",
        }


def probe_imports(python_path: str | Path, *, modules: list[str] | None = None) -> dict[str, Any]:
    python = Path(python_path)
    if not python.exists():
        return {
            "status": "missing_python",
            "python": str(python),
            "modules": {name: False for name in (modules or CRITICAL_MODULES)},
            "returncode": None,
            "stderr_tail": "",
        }

    try:
        completed = subprocess.run(
            [str(python), "-c", build_import_probe_code(modules)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:
        return {
            "status": "fail",
            "python": str(python),
            "modules": {name: False for name in (modules or CRITICAL_MODULES)},
            "returncode": None,
            "stderr_tail": str(exc),
        }
    try:
        module_status = json.loads(completed.stdout.strip() or "{}")
    except json.JSONDecodeError:
        module_status = {name: False for name in (modules or CRITICAL_MODULES)}
    return {
        "status": "pass" if completed.returncode == 0 and all(module_status.values()) else "fail",
        "python": str(python),
        "modules": module_status,
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr[-2000:],
    }


def check_environment(app_dir: str | Path) -> dict[str, Any]:
    app_path = Path(app_dir)
    python = venv_python_path(app_path)
    requirements = requirements_file_path(app_path)
    probe = probe_imports(python)
    missing_modules = [name for name, ok in probe["modules"].items() if not ok]
    ready = python.exists() and requirements.exists() and probe["status"] == "pass"

    return {
        "generated_at": strftime("%Y-%m-%d %H:%M:%S"),
        "app_dir": str(app_path),
        "venv_python": str(python),
        "venv_exists": python.exists(),
        "requirements_exists": requirements.exists(),
        "critical_modules": probe["modules"],
        "missing_modules": missing_modules,
        "ready": ready,
        "probe": probe,
        "actions": [],
        "next_actions": _next_actions(ready, python.exists(), requirements.exists(), missing_modules),
    }


def setup_environment(
    app_dir: str | Path,
    *,
    create_if_missing: bool = False,
    install_requirements: bool = False,
) -> dict[str, Any]:
    app_path = Path(app_dir)
    venv_dir = venv_dir_path(app_path)
    python = venv_python_path(app_path)
    actions: list[dict[str, Any]] = []

    if create_if_missing and not python.exists():
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        actions.append(_run_command([sys.executable, "-m", "venv", str(venv_dir)], cwd=app_path, timeout=600))

    if install_requirements and python.exists():
        requirements = requirements_file_path(app_path)
        actions.append(_run_command([str(python), "-m", "pip", "install", "--upgrade", "pip"], cwd=app_path, timeout=600))
        actions.append(_run_command([str(python), "-m", "pip", "install", "-r", str(requirements)], cwd=app_path, timeout=1800))

    report = check_environment(app_path)
    report["actions"] = actions
    report["ready"] = report["ready"] and all(action["status"] == "pass" for action in actions)
    if actions and not report["ready"]:
        report["next_actions"] = [
            "Revisar el reporte environment_status.md.",
            "Confirmar que Python pueda crear entornos virtuales.",
            "Ejecutar python preparar_entorno.py --install cuando haya internet disponible.",
        ]
    return report


def _next_actions(ready: bool, venv_exists: bool, requirements_exists: bool, missing_modules: list[str]) -> list[str]:
    if ready:
        return [
            "Entorno listo.",
            "Abrir la app con abrir traductor.bat.",
        ]
    if not requirements_exists:
        return ["Falta requisitos.txt; restaurar el archivo antes de instalar dependencias."]
    if not venv_exists:
        return ["Ejecutar python preparar_entorno.py --install para crear el entorno python e instalar requisitos.txt."]
    if missing_modules:
        return [
            f"Faltan modulos criticos: {', '.join(missing_modules)}.",
            "Ejecutar python preparar_entorno.py --repair para reparar dependencias.",
        ]
    return ["Revisar environment_status.json para ver el detalle del fallo."]


def save_environment_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "environment_status.json"
    md_path = output_path / "environment_status.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    lines = [
        "# Estado del Entorno Python",
        "",
        f"Generado: {report['generated_at']}",
        f"Listo: {'si' if report['ready'] else 'no'}",
        f"Python del entorno: {report['venv_python']}",
        f"Entorno existe: {'si' if report['venv_exists'] else 'no'}",
        f"requisitos.txt existe: {'si' if report['requirements_exists'] else 'no'}",
        "",
        "## Modulos criticos",
        "",
    ]
    for module, ok in report["critical_modules"].items():
        lines.append(f"- {'OK' if ok else 'FALTA'}: {module}")

    if report["actions"]:
        lines.extend(["", "## Acciones ejecutadas", ""])
        for action in report["actions"]:
            command = " ".join(action["command"])
            lines.append(f"- {action['status'].upper()}: {command}")

    lines.extend(["", "## Siguientes acciones", ""])
    for action in report["next_actions"]:
        lines.append(f"- {action}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path