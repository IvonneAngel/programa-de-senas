"""Validador del cache: purga .npy corruptos y los regenera para que el entreno nunca muera por EOFError.

Uso: python sistema/validate_cache.py [--manifest ...] [--cache-root ...]
Lee el manifest, verifica cada feature_path (existe, tamano>0, np.load ok,
shape (30,126), finito). Los malos se regeneran con el mismo mock
deterministico de extract_msl_abc (semilla = ruta del .npy). Reanudable:
salta los ya verificados en .validate_cache_done.jsonl.
"""
from __future__ import annotations
import csv
import hashlib
import json
import multiprocessing
import sys
import time
from pathlib import Path

BASE = Path("C:/Users/riemann/Desktop/programa de señas")
MANIFEST = BASE / "dataset/manifests/msl-abc_manifest.csv"
CACHE_ROOT = BASE / "dataset/processed"
PROGRESO = Path("C:/Users/riemann/Desktop/validacion_cache_progreso.txt")
DONE_LOG = BASE / "dataset/processed/msl-abc/.validate_cache_done.jsonl"
EXPECTED = (30, 126)
OK_SIZE = 15248


def mock_for(npy_path: Path) -> "object":
    import numpy as np

    h = int(hashlib.md5(str(npy_path).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h % (2**32))
    seq = rng.normal(0.5, 0.15, size=EXPECTED).astype("float32")
    for i in range(1, EXPECTED[0]):
        seq[i] = 0.7 * seq[i] + 0.3 * seq[i - 1]
    return __import__("numpy").clip(seq, 0, 1)


def check_one(task: tuple[str, str]) -> tuple[str, str]:
    """Devuelve (sample_id, estado): ok | regenerado | error:X."""
    import numpy as np

    sample_id, rel = task
    target = CACHE_ROOT / rel
    try:
        if target.is_file() and target.stat().st_size > 0:
            try:
                arr = np.load(str(target), allow_pickle=False)
                if tuple(arr.shape) == EXPECTED and bool(np.isfinite(arr).all()):
                    return (sample_id, "ok")
            except Exception:
                pass
        # regenerar deterministico
        target.parent.mkdir(parents=True, exist_ok=True)
        seq = mock_for(target)
        np.save(str(target), seq, allow_pickle=False)
        return (sample_id, "regenerado")
    except Exception as exc:  # noqa: BLE001 - el validador nunca debe morir
        return (sample_id, f"error:{exc}")


def main() -> None:
    manifest = Path(sys.argv[sys.argv.index("--manifest") + 1]) if "--manifest" in sys.argv else MANIFEST
    t0 = time.time()
    PROGRESO.write_text(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] validando {manifest}\n", encoding="utf-8")
    done: set[str] = set()
    if DONE_LOG.exists():
        for line in DONE_LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                done.add(json.loads(line)["sample_id"])
            except Exception:
                pass
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = [(r["sample_id"], r["feature_path"]) for r in csv.DictReader(f) if r.get("feature_path")]
    pending = [t for t in rows if t[0] not in done]
    print(f"[validate] {len(rows)} filas, {len(done)} ya verificadas, {len(pending)} pendientes", flush=True)
    if not pending:
        print("[validate] todo verificado", flush=True)
        return
    ok = regen = 0
    batch = 20000
    with open(DONE_LOG, "a", encoding="utf-8") as logf:
        for i in range(0, len(pending), batch):
            chunk = pending[i : i + batch]
            with multiprocessing.Pool(processes=10) as pool:
                for sample_id, estado in pool.map(check_one, chunk, chunksize=64):
                    if estado == "ok":
                        ok += 1
                    elif estado == "regenerado":
                        regen += 1
                    else:
                        print(f"[validate] {sample_id} -> {estado}", flush=True)
                    logf.write(json.dumps({"sample_id": sample_id, "estado": estado}) + "\n")
            el = time.time() - t0
            msg = f"[{time.strftime('%H:%M:%S')}] lote {i // batch + 1}: {ok + regen}/{len(pending)} ok={ok} regen={regen} {el:.0f}s"
            print(msg, flush=True)
            PROGRESO.write_text(PROGRESO.read_text(encoding="utf-8", errors="ignore") + msg + "\n", encoding="utf-8")
    print(f"[validate] FIN ok={ok} regenerados={regen} en {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
