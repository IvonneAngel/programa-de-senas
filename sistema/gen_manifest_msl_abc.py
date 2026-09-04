import pathlib, hashlib, csv
from pathlib import Path
BASE = Path("C:/Users/riemann/Desktop/programa de señas")
DST = BASE / "dataset/processed/msl-abc/bone_vector126"
OUT = BASE / "dataset/manifests/msl-abc_manifest.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
import time
start = time.time()
files = list(DST.rglob("*.npy"))
print(f"Encontrados {len(files)} npy")
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["sample_id","label","label_id","word","word_id","grouping","image_count","feature_path","bone_degenerate_hands","status","split"])
    for p in files:
        label = p.parent.name
        stem = p.stem
        rel = p.relative_to(BASE / "dataset/processed").as_posix()
        h = int(hashlib.md5(stem.encode()).hexdigest()[:4], 16) % 100
        split = "train" if h < 70 else "validation" if h < 85 else "test"
        w.writerow([stem, label, label, label, label, "", 1, rel, 0, "ok", split])
print(f"Manifest {OUT} con {len(files)} filas en {time.time()-start:.1f}s")
