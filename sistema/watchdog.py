"""Watchdog: cada 5 min verifica entreno + descarga CUDA y los levanta solos con --resume.

No depende de ninguna sesion de chat: vive como Scheduled Task 'lsm-watchdog'.
Si entreno murio o su log no avanza en 25 min -> relanza con --resume.
Si falta torch CUDA en .venv y no hay descarga corriendo -> relanza instalacion.
Todo con append a logs, nunca borra. Una sola instancia via lockfile.
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("C:/Users/riemann/Desktop/programa de señas")
LOCK = BASE / "sistema/.watchdog.lock"
LOG = BASE / "watchdog.log"
TRAIN_LOG = BASE / "training_msl_abc.log"
# ponytail: el watchdog cuida TODOS los entrenos (msl-abc gigante + dinamicas). Un solo dueño.
ENTRENOS = [
    {"nombre": "msl-abc", "manifest": "dataset/manifests/msl-abc_manifest.csv",
     "out": "docs/analisis/fase2-entrenamiento-msl-abc", "log": "training_msl_abc.log",
     "batch_cpu": "16", "batch_cuda": "256"},
    {"nombre": "dinamicas", "manifest": "dataset/manifests/msl-dynamic_manifest.csv",
     "out": "docs/analisis/fase2-dinamicas", "log": "training_dinamicas.log",
     "batch_cpu": "16", "batch_cuda": "32"},
]
VENV_PY = Path("C:/Users/riemann/.venv/Scripts/python.exe")
SYS_PY = Path("C:/Users/riemann/AppData/Local/Programs/Python/Python312/python.exe")
CUDA_URL = "https://download.pytorch.org/whl/cu126"
STALE_SECONDS = 25 * 60


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [watchdog] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def procs_match(exe: str, substr: str) -> list[int]:
    """Filtra dentro de PowerShell (rapido, sin traer 200 cmdlines).
    Reintenta 1 vez: el CIM frio puede tardar. Nunca devuelve [] sin 2 intentos."""
    # ponytail: el filtro va dentro de powershell; traer todo y filtrar en python era lento y fragil.
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='" + exe + "'\" | "
          "Where-Object { $_.CommandLine -like '*" + substr + "*' } | "
          "ForEach-Object { $_.ProcessId }")
    for attempt in (1, 2):
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                                 capture_output=True, text=True, timeout=180)
            found = []
            for line in out.stdout.splitlines():
                try:
                    found.append(int(line.strip()))
                except ValueError:
                    pass
            if found or attempt == 2:
                return found
        except Exception as exc:
            log(f"procs_match intento {attempt} fallo: {exc}")
            if attempt == 2:
                return []
    return []


def train_python() -> str:
    """Usa .venv solo si ya tiene torch CUDA; si no, python sistema (CPU)."""
    try:
        out = subprocess.run([str(VENV_PY), "-c", "import torch; print(torch.cuda.is_available())"],
                             capture_output=True, text=True, timeout=60)
        if "True" in out.stdout:
            return str(VENV_PY)
    except Exception:
        pass
    return str(SYS_PY)


def train_alive() -> bool:
    pids = procs_match("python.exe", "train_classifier")
    if not pids:
        return False
    try:
        age = time.time() - TRAIN_LOG.stat().st_mtime
        return age < STALE_SECONDS
    except OSError:
        return False


def wmi_launch(cmd: str, workdir: str) -> int | None:
    """Lanza desacoplado via WMI (probado: sobrevive a la sesion). Devuelve PID o None."""
    # ponytail: start /min fallaba en silencio; WMI Win32_Process.Create si funciona (PID 8784).
    esc = cmd.replace('"', '""')
    ps = (f'$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments '
          f'@{{CommandLine="{esc}"; CurrentDirectory="{workdir}"}}; $r.ProcessId')
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=120)
        for line in out.stdout.splitlines():
            try:
                pid = int(line.strip())
                if pid > 0:
                    return pid
            except ValueError:
                pass
    except Exception as exc:
        log(f"wmi_launch fallo: {exc}")
    return None


def _cmd_tiene(pid: int, tag: str) -> bool:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              f"(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\").CommandLine"],
                             capture_output=True, text=True, timeout=60)
        return tag in out.stdout
    except Exception:
        return False


def terminado_ok(ent: dict) -> bool:
    """Si el log dice done:true, no relanzar (completo por early stopping o fin)."""
    try:
        for line in open(BASE / ent["log"], encoding="utf-8", errors="ignore").read().splitlines()[-5:]:
            if '"done": true' in line:
                return True
    except OSError:
        pass
    return False


def train_alive(ent: dict) -> bool:
    tag = ent["out"].split("/")[-1]
    pids = [p for p in procs_match("python.exe", "train_classifier") if _cmd_tiene(p, tag)]
    if not pids:
        return False
    try:
        age = time.time() - (BASE / ent["log"]).stat().st_mtime
        return age < STALE_SECONDS
    except OSError:
        return False


def launch_train(ent: dict) -> None:
    py = train_python()
    device = "cuda" if Path(py) == VENV_PY else "cpu"
    batch = ent["batch_cuda"] if device == "cuda" else ent["batch_cpu"]
    # ponytail: num-workers 0 porque el spawn de torch 2.9.1 en Windows revienta los workers
    # (pickle truncado). Sin spawn no hay muerte; la GPU compensa la carga monohilo.
    inner = (f'set PYTHONPATH=entrenador/core-pt && "{py}" -u -m lsm.training.train_classifier '
             f"--manifest {ent['manifest']} --cache-root dataset/processed "
             f"--task successor_positions126 --out {ent['out']} "
             f"--device {device} --batch-size {batch} --epochs 40 --num-workers 0 --pin-memory --resume "
             f">> {ent['log']} 2>&1")
    cmd = f"cmd.exe /c {inner}"
    # defensa: no duplicar si alguien lo levanto entre el chequeo y ahora
    if procs_match("python.exe", ent["out"].split("/")[-1]):
        log(f"                                   {ent['nombre']} ya levantado por otro, no duplico")
        return
    pid = wmi_launch(cmd, str(BASE))
    log(f"relanzando entreno {ent['nombre']} {device} batch {batch} ({py}) pid={pid}")


def cuda_ok() -> bool:
    try:
        out = subprocess.run([str(VENV_PY), "-c", "import torch; print(torch.cuda.is_available())"],
                             capture_output=True, text=True, timeout=120)
        return "True" in out.stdout
    except Exception:
        return False


def launch_download() -> None:
    if procs_match("uv.exe", "download.pytorch.org"):
        log("descarga ya levantada por otro, no duplico")
        return
    cmd = (f'cmd.exe /c uv pip install --python "{VENV_PY}" torch --index-url {CUDA_URL} '
           f'>> torch_install.log 2>&1')
    pid = wmi_launch(cmd, str(BASE))
    log(f"relanzando descarga torch CUDA pid={pid}")


def main() -> None:
    try:
        LOCK.write_text(str(time.time()), encoding="utf-8")
    except Exception:
        pass
    try:
        for ent in ENTRENOS:
            if terminado_ok(ent):
                log(f"entreno {ent['nombre']} COMPLETO (done:true), no tocar")
            elif train_alive(ent):
                log(f"entreno {ent['nombre']} vivo y avanzando")
            else:
                log(f"entreno {ent['nombre']} MUERTO o estancado -> relanzar con --resume")
                launch_train(ent)
        if cuda_ok():
            log("torch CUDA ok en .venv")
        elif procs_match("uv.exe", "cu126") or procs_match("uv.exe", "download.pytorch.org") or procs_match("aria2c.exe", "torch"):
            log("descarga CUDA en curso")
        else:
            # ponytail: CUDA ya instalado; no auto-reinstalar por uv (se traba). Solo avisar.
            log("ALERTA sin torch CUDA y sin descarga: reinstalacion manual requerida")
    except Exception as exc:  # noqa: BLE001 - el watchdog nunca muere
        log(f"error interno: {exc}")
    finally:
        try:
            LOCK.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
