"""Video MP4 -> 30 frames -> MediaPipe Hands -> bone_vector126 (mismo math que auto_extract).

Pool 6 workers (cada uno su detector). Reanudable: salta .npy existentes.
Progreso en C:/Users/riemann/Desktop/extraccion_dynamic_progreso.txt
"""
from __future__ import annotations
import csv
import hashlib
import multiprocessing
import time
from pathlib import Path

import numpy as np

BASE = Path("C:/Users/riemann/Desktop/programa de señas")
SRC = BASE / "dataset/raw/msl-dynamic"
DST = BASE / "dataset/processed/msl-dynamic/bone_vector126"
MANIFEST = BASE / "dataset/manifests/msl-dynamic_manifest.csv"
PROGRESO = Path("C:/Users/riemann/Desktop/extraccion_dynamic_progreso.txt")

EPSILON = 1e-6
PARENTS = np.asarray((0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19), dtype=np.int64)
CHILDREN = np.arange(1, 21, dtype=np.int64)
MCP = np.asarray((5, 9, 13, 17), dtype=np.int64)

LETTERS = ("J", "K", "N", "Ñ", "Q", "X", "Z")


def letra_de(path: Path) -> str:
    for chunk in path.stem.split("-"):
        if chunk in LETTERS:
            return "Ñ" if chunk == "N" else chunk
    return "unknown"


def split_de(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "test" in parts:
        return "test"
    return "train"


def palm_scale(hand: np.ndarray):
    v = np.asarray(hand, dtype=np.float32)
    if v.shape != (21, 3) or not np.isfinite(v).all() or not v.any():
        return None
    return float(np.linalg.norm(v[MCP], axis=1).mean())


def transform_hand(hand: np.ndarray):
    v = np.asarray(hand, dtype=np.float32)
    sc = palm_scale(v)
    if sc is None:
        return np.zeros((21, 3), dtype=np.float32), False
    if sc <= EPSILON:
        return np.zeros((21, 3), dtype=np.float32), True
    out = np.empty((21, 3), dtype=np.float32)
    out[:20] = (v[CHILDREN] - v[PARENTS]) / sc
    out[20] = v[MCP].mean(axis=0) / sc
    return out, False


def frames_a_seq(frames: np.ndarray) -> np.ndarray:
    n = frames.shape[0]
    if n == 30:
        return frames.astype(np.float32)
    if n == 0:
        return np.zeros((30, 126), dtype=np.float32)
    if n == 1:
        return np.repeat(frames, 30, axis=0).astype(np.float32)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, 30)
    out = np.empty((30, 126), dtype=np.float32)
    for c in range(126):
        out[:, c] = np.interp(x_new, x_old, frames[:, c])
    return out


_DETECTOR = None
# Ruta sin ñ ni espacios: el C++ de MediaPipe no abre "señas".
_TASK_PATH = Path("C:/msl-models/hand_landmarker.task")
# VIDEO mode exige timestamps crecientes POR DETECTOR (no por video).
_TS = [0]


def _next_ts() -> int:
    _TS[0] += 33
    return _TS[0]


def _init_worker() -> None:
    # 1 detector por worker (crearlo por video costaba 0.5s x 1242).
    global _DETECTOR
    import mediapipe as mp

    base = mp.tasks.BaseOptions(model_asset_path=str(_TASK_PATH))
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base, running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.3)
    _DETECTOR = mp.tasks.vision.HandLandmarker.create_from_options(opts)


def process_one(video_str: str) -> dict:
    import cv2
    import mediapipe as mp

    global _DETECTOR
    if _DETECTOR is None:
        _init_worker()
    p = Path(video_str)
    label = letra_de(p)
    target = DST / label / (p.stem + ".npy")
    if target.exists() and target.stat().st_size > 0:
        return {"ok": True, "skip": True, "label": label, "frames": 0, "hands": 0}
    cap = cv2.VideoCapture(str(p))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // 60)
    raw: list[np.ndarray] = []
    hands_seen = 0
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if idx % step == 0:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            res = _DETECTOR.detect_for_video(image, _next_ts())
            vec = np.zeros(126, dtype=np.float32)
            if res.hand_landmarks:
                hands_seen += 1
                for h, lm in enumerate(res.hand_landmarks[:2]):
                    for j, pt in enumerate(lm):
                        vec[h * 63 + j * 3:(h * 63 + j * 3 + 3)] = (pt.x, pt.y, pt.z)
            raw.append(np.clip(vec, 0.01, 0.99))
        idx += 1
    cap.release()
    arr = np.stack(raw, axis=0) if raw else np.zeros((0, 126), dtype=np.float32)
    seq = frames_a_seq(arr)
    # bone transform por frame
    out = np.empty_like(seq)
    for i, frame in enumerate(seq):
        left, _ = transform_hand(frame[:63].reshape(21, 3))
        right, _ = transform_hand(frame[63:].reshape(21, 3))
        out[i, :63] = left.reshape(63)
        out[i, 63:] = right.reshape(63)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(target), out.astype(np.float32), allow_pickle=False)
    return {"ok": True, "skip": False, "label": label, "frames": len(raw), "hands": hands_seen}


def main() -> None:
    vids = sorted(SRC.rglob("*.mp4"))
    PROGRESO.write_text(f"[{time.strftime('%H:%M:%S')}] {len(vids)} videos\n", encoding="utf-8")
    print(f"[dyn] {len(vids)} videos, Pool 6", flush=True)
    t0 = time.time()
    results = []
    batch = 60
    strs = [str(v) for v in vids]
    done = 0
    with open(MANIFEST, "w", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        w.writerow(["sample_id", "label", "label_id", "word", "word_id", "grouping", "image_count", "feature_path", "bone_degenerate_hands", "status", "split"])
        for i in range(0, len(strs), batch):
            chunk = strs[i : i + batch]
            with multiprocessing.Pool(processes=6, initializer=_init_worker) as pool:
                for r, s in zip(pool.map(process_one, chunk, chunksize=2), chunk):
                    p = Path(s)
                    rel = f"msl-dynamic/bone_vector126/{r['label']}/{p.stem}.npy"
                    h = int(hashlib.md5(p.stem.encode()).hexdigest()[:4], 16) % 100
                    split = split_de(p)
                    w.writerow([p.stem, r["label"], r["label"], r["label"], r["label"], "dynamic", r["frames"], rel, 0, "ok", split])
                    results.append(r)
            done += len(chunk)
            el = time.time() - t0
            hands = sum(1 for x in results if x["hands"] > 0)
            msg = f"[{time.strftime('%H:%M:%S')}] {done}/{len(strs)} {done / el:.1f} vid/s con-mano={hands}"
            print(msg, flush=True)
            PROGRESO.write_text(PROGRESO.read_text(encoding="utf-8", errors="ignore") + msg + "\n", encoding="utf-8")
    print(f"[dyn] FIN {len(results)} en {time.time() - t0:.0f}s manifest={MANIFEST}", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
