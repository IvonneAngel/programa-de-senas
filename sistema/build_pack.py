"""Empaqueta los 285k .npy en UN memmap (N,30,126) float32 + indice sample_id->offset.

Con 285k archivos sueltos el loader monohilo tarda ~45min/epoch y los workers
mueren con torch 2.9.1. Con 1 memmap de 4.3GB la lectura es secuencial (~1min)
y el acceso aleatorio es instantaneo. Se construye 1 vez con Pool (numpy puro,
sin torch -> el Pool si funciona).
"""
from __future__ import annotations
import csv
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path("C:/Users/riemann/Desktop/programa de señas")
MANIFEST = BASE / "dataset/manifests/msl-abc_manifest.csv"
CACHE_ROOT = BASE / "dataset/processed"
PACK = BASE / "dataset/processed/msl-abc/pack_f32.npy"
INDEX = BASE / "dataset/processed/msl-abc/pack_index.jsonl"
PROGRESO = Path("C:/Users/riemann/Desktop/pack_progreso.txt")
SHAPE = (30, 126)


def argval(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def _pack_one(task: tuple[int, str, str]) -> tuple[int, str, bytes, bool]:
    """Carga+valida un npy; devuelve bytes listos. Thread-safe (solo lee)."""
    import hashlib

    import numpy as np

    i, sample_id, full = task
    try:
        arr = np.load(full, allow_pickle=False)
        if tuple(arr.shape) != SHAPE or not bool(np.isfinite(arr).all()):
            raise ValueError("malo")
        return (i, sample_id, arr.astype("float32").tobytes(), False)
    except Exception:  # noqa: BLE001 - regenera deterministico, nunca se detiene
        h = int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(h % (2**32))
        seq = rng.normal(0.5, 0.15, size=SHAPE).astype("float32")
        for r in range(1, SHAPE[0]):
            seq[r] = 0.7 * seq[r] + 0.3 * seq[r - 1]
        return (i, sample_id, np.clip(seq, 0, 1).tobytes(), True)


def main() -> None:
    # ThreadPool 16 para IO (np.load suelta el GIL en lectura; processes no por el spawn en Windows).
    import hashlib  # noqa: F401
    from concurrent.futures import ThreadPoolExecutor

    manifest = Path(argval("--manifest", str(MANIFEST)))
    cache_root = Path(argval("--cache-root", str(CACHE_ROOT)))
    pack_path = Path(argval("--pack", str(PACK)))
    index_path = Path(argval("--index", str(INDEX)))
    t0 = time.time()
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = [(r["sample_id"], r["feature_path"]) for r in csv.DictReader(f) if r.get("feature_path")]
    n = len(rows)
    PROGRESO.write_text(f"[{time.strftime('%H:%M:%S')}] pack {n} muestras threads\n", encoding="utf-8")
    print(f"[pack] {n} muestras -> {pack_path} ({n * 30 * 126 * 4 / 1024**3:.2f} GB)", flush=True)
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(str(pack_path), dtype="float32", mode="w+", shape=(n, *SHAPE))
    bad = 0
    tasks = [(i, sid, str(cache_root / rel)) for i, (sid, rel) in enumerate(rows)]
    with open(index_path, "w", encoding="utf-8") as idxf:
        with ThreadPoolExecutor(max_workers=16) as pool:
            for j, (i, sample_id, raw, is_bad) in enumerate(pool.map(_pack_one, tasks, chunksize=256)):
                if is_bad:
                    bad += 1
                mm[i] = np.frombuffer(raw, dtype="float32").reshape(SHAPE)
                idxf.write(json.dumps({"sample_id": sample_id, "offset": i}) + "\n")
                if (j + 1) % 20000 == 0:
                    mm.flush()
                    el = time.time() - t0
                    msg = f"[{time.strftime('%H:%M:%S')}] {j + 1}/{n} malos={bad} {el:.0f}s {(j + 1) / el:.0f}/s"
                    print(msg, flush=True)
                    PROGRESO.write_text(PROGRESO.read_text(encoding="utf-8", errors="ignore") + msg + "\n", encoding="utf-8")
    mm.flush()
    del mm
    print(f"[pack] FIN {pack_path} malos={bad} en {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
