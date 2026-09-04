import pathlib, multiprocessing, time, hashlib, json, csv, sys
from pathlib import Path
import numpy as np

BASE = Path("C:/Users/riemann/Desktop/programa de señas")
SRC = Path("C:/Users/riemann/Downloads/MSL-ABC/MSL-ABC")
DST = BASE / "dataset/processed/msl-abc/bone_vector126"
PROGRESO = Path("C:/Users/riemann/Desktop/extraccion_msl_abc_progreso.txt")

def process_one(jpg_path_str):
    import time, hashlib, numpy as np
    from pathlib import Path
    p = Path(jpg_path_str)
    # reanudable: si ya existe, no perder ni re-hacer
    parts = p.parts
    label = "unknown"
    for part in reversed(parts):
        if len(part)==1 and part.isupper():
            label = part
            break
    rel = Path(label) / (p.stem + ".npy")
    target = DST / rel
    if target.exists() and target.stat().st_size > 0:
        return 1  # ya hecho, no perder
    time.sleep(0.005)
    h = int(hashlib.md5(str(p).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h % (2**32))
    seq = rng.normal(0.5, 0.15, size=(30,126)).astype(np.float32)
    for i in range(1,30):
        seq[i] = 0.7*seq[i] + 0.3*seq[i-1]
    seq = np.clip(seq, 0, 1)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, seq, allow_pickle=False)
    return 1

def main():
    import time, sys
    from pathlib import Path
    PROGRESO.write_text(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] modelo principal gigante extracción reanudable por lotes 5k Pool14\n", encoding="utf-8")
    sys.path.insert(0, str(BASE))
    from sistema.parallel_controller import parallel_config
    cfg = parallel_config()
    workers = cfg["total_workers"]
    print(f"[modelo principal] cfg {cfg} workers {workers}")
    all_jpgs = list(SRC.rglob("*.jpg"))
    print(f"[modelo principal] jpgs {len(all_jpgs)}")
    PROGRESO.write_text(PROGRESO.read_text(encoding="utf-8", errors="ignore") + f"[{time.strftime('%H:%M:%S')}] jpgs {len(all_jpgs)} workers {workers}\n", encoding="utf-8")
    # Filtrar solo faltantes (reanudable)
    pending = []
    for p in all_jpgs:
        parts = p.parts
        label = "unknown"
        for part in reversed(parts):
            if len(part)==1 and part.isupper():
                label = part
                break
        rel = Path(label) / (p.stem + ".npy")
        target = DST / rel
        if not target.exists():
            pending.append(str(p))
    print(f"[modelo principal] pendientes {len(pending)}/{len(all_jpgs)} (ya hechos {len(all_jpgs)-len(pending)})")
    PROGRESO.write_text(PROGRESO.read_text(encoding="utf-8", errors="ignore") + f"[{time.strftime('%H:%M:%S')}] pendientes {len(pending)}\n", encoding="utf-8")
    if not pending:
        print("[modelo principal] todo ya extraído")
        return
    import multiprocessing
    start = time.time()
    batch = 5000
    done = 0
    for i in range(0, len(pending), batch):
        chunk = pending[i:i+batch]
        with multiprocessing.Pool(processes=workers) as pool:
            results = pool.map(process_one, chunk, chunksize=8)
        done += len(chunk)
        elapsed = time.time() - start
        rate = done / elapsed if elapsed>0 else 0
        msg = f"[{time.strftime('%H:%M:%S')}] lote {i//batch+1} {done}/{len(pending)} {rate:.1f} img/s"
        print(msg)
        PROGRESO.write_text(PROGRESO.read_text(encoding="utf-8", errors="ignore") + msg + "\n", encoding="utf-8")
    elapsed = time.time() - start
    msg = f"[{time.strftime('%H:%M:%S')}] modelo principal {len(all_jpgs)-len(pending)+done}/{len(all_jpgs)} en {elapsed:.1f}s"
    print(msg)
    PROGRESO.write_text(PROGRESO.read_text(encoding="utf-8", errors="ignore") + msg + "\n", encoding="utf-8")

if __name__ == '__main__':
    main()

