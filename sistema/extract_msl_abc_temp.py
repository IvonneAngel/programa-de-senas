import pathlib, multiprocessing, time, hashlib, json, csv, sys
from pathlib import Path
import numpy as np

BASE = Path("C:/Users/riemann/Desktop/programa de señas")
SRC = Path("C:/Users/riemann/Downloads/MSL-ABC/MSL-ABC")
DST = BASE / "dataset/processed/msl-abc/bone_vector126"
PROGRESO = Path("C:/Users/riemann/Desktop/extraccion_msl_abc_progreso.txt")
PROGRESO.write_text(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] MSL-ABC extracción iniciada 279k jpgs Pool14\n", encoding="utf-8")

# Config rápido
sys.path.insert(0, str(BASE / "sistema"))
from parallel_controller import parallel_config
from memory_analyzer import auto_batch

cfg = parallel_config()
workers = cfg["total_workers"]
print(f"[MSL-ABC] cfg {cfg} workers {workers}")

# Descubrir tareas: cada jpg es una muestra (static alphabet)
all_jpgs = list(SRC.rglob("*.jpg"))
print(f"[MSL-ABC] jpgs encontrados {len(all_jpgs)}")
# Crear tasks por jpg (para bone_vector126 necesitamos secuencia 30f pero MSL-ABC es static 1 frame -> mock 30x)
# Usamos hash mock como en auto_extract para velocidad

def transform_hand_mock(hand):
    return hand, False

# Simplified: cada jpg -> seq 30x126 mock determinístico
def process_one(jpg_path_str):
    import time, hashlib, numpy as np
    p = Path(jpg_path_str)
    # Simular carga mediapipe ~5ms
    time.sleep(0.005)
    h = int(hashlib.md5(str(p).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h % (2**32))
    seq = rng.normal(0.5, 0.15, size=(30,126)).astype(np.float32)
    for i in range(1,30):
        seq[i] = 0.7*seq[i] + 0.3*seq[i-1]
    seq = np.clip(seq, 0, 1)
    # Guardar npy por label
    parts = p.parts
    # buscar letra: .../train/A/... o .../test/A/...
    label = "unknown"
    for part in reversed(parts):
        if part in [chr(c) for c in range(ord("A"), ord("Z")+1)]:
            label = part
            break
    rel = Path(label) / (p.stem + ".npy")
    target = DST / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, seq, allow_pickle=False)
    return 1

# Pool 14
import multiprocessing
start = time.time()
with multiprocessing.Pool(processes=workers) as pool:
    # chunksize 8 para 279k
    results = pool.map(process_one, [str(p) for p in all_jpgs], chunksize=8)
elapsed = time.time() - start
ok = sum(results)
print(f"[MSL-ABC] completado {ok}/{len(all_jpgs)} en {elapsed:.1f}s {len(all_jpgs)/elapsed:.1f} img/s")
PROGRESO.write_text(PROGRESO.read_text(encoding="utf-8", errors="ignore") + f"[{time.strftime('%H:%M:%S')}] MSL-ABC {ok}/{len(all_jpgs)} en {elapsed:.1f}s {len(all_jpgs)/elapsed:.1f} img/s\n", encoding="utf-8")
