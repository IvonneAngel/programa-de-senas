"""Extrae los 7z de dinamicas a dataset/raw/msl-dynamic/ + inventario."""
import time
from pathlib import Path

import py7zr

BASE = Path("C:/Users/riemann/Desktop/programa de señas")
DST = BASE / "dataset/raw/msl-dynamic"
DL = Path("C:/Users/riemann/Desktop/programa de señas/../riemann/Downloads")
DL = Path("C:/Users/riemann/Downloads")
PROGRESO = Path("C:/Users/riemann/Desktop/extraccion_dynamic_progreso.txt")

FILES = ["MSL-dynamic-signs-frontal.7z", "MSL-dynamic-signs-perfil.7z"]


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    PROGRESO.write_text(f"[{time.strftime('%H:%M:%S')}] extrayendo dinamicas\n", encoding="utf-8")
    for name in FILES:
        src = DL / name
        out = DST / src.stem
        if out.exists() and any(out.iterdir()):
            msg = f"[{time.strftime('%H:%M:%S')}] {name} ya extraido, salto"
            print(msg, flush=True)
        else:
            t0 = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] extrayendo {name} ({src.stat().st_size / 1024**2:.0f} MB)...", flush=True)
            out.mkdir(parents=True, exist_ok=True)
            with py7zr.SevenZipFile(src, mode="r") as z:
                z.extractall(path=out)
            msg = f"[{time.strftime('%H:%M:%S')}] {name} listo en {time.time() - t0:.0f}s"
            print(msg, flush=True)
        PROGRESO.write_text(PROGRESO.read_text(encoding="utf-8", errors="ignore") + msg + "\n", encoding="utf-8")
    # inventario
    vids = list(DST.rglob("*.mp4"))
    print(f"[inv] videos: {len(vids)}", flush=True)
    by_letter: dict[str, int] = {}
    for v in vids:
        for part in v.parts:
            if len(part) == 1 and part.isupper() or part in ("J", "K", "Ñ", "Q", "X", "Z", "N"):
                by_letter[part] = by_letter.get(part, 0) + 1
                break
    print(f"[inv] por letra: {by_letter}", flush=True)
    for v in vids[:5]:
        print(f"  ej: {v.relative_to(DST)}", flush=True)


if __name__ == "__main__":
    main()
