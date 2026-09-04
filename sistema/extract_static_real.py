"""Landmarks REALES para las 279k estaticas (eran mock). Mismo math bone_vector126.

Por jpg: 1 frame MediaPipe Tasks -> se replica a 30f (estatica) -> bone transform.
Pool 6, reanudable (salta .npy con media != mock). Progreso cada 60 en log.
"""
from __future__ import annotations
import multiprocessing
import time
from pathlib import Path

import numpy as np

BASE = Path("C:/Users/riemann/Desktop/programa de señas")
SRC = Path("C:/Users/riemann/Downloads/MSL-ABC/MSL-ABC")
DST = BASE / "dataset/processed/msl-abc/bone_vector126"
PROGRESO = Path("C:/Users/riemann/Desktop/extraccion_static_real_progreso.txt")

EPSILON = 1e-6
PARENTS = np.asarray((0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19), dtype=np.int64)
CHILDREN = np.arange(1, 21, dtype=np.int64)
MCP = np.asarray((5, 9, 13, 17), dtype=np.int64)

_DETECTOR = None
_TASK_PATH = Path("C:/msl-models/hand_landmarker.task")


def _init_worker() -> None:
    global _DETECTOR
    import mediapipe as mp

    base = mp.tasks.BaseOptions(model_asset_path=str(_TASK_PATH))
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base, running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2, min_hand_detection_confidence=0.3)
    _DETECTOR = mp.tasks.vision.HandLandmarker.create_from_options(opts)


def _bone(hand: np.ndarray) -> np.ndarray:
    v = np.asarray(hand, dtype=np.float32)
    sc = float(np.linalg.norm(v[MCP], axis=1).mean())
    if sc <= EPSILON or not np.isfinite(v).all():
        return np.zeros((21, 3), dtype=np.float32)
    out = np.empty((21, 3), dtype=np.float32)
    out[:20] = (v[CHILDREN] - v[PARENTS]) / sc
    out[20] = v[MCP].mean(axis=0) / sc
    return out


def process_one(jpg_str: str) -> str:
    import cv2
    import mediapipe as mp

    global _DETECTOR
    if _DETECTOR is None:
        _init_worker()
    p = Path(jpg_str)
    label = "unknown"
    for part in reversed(p.parts):
        if len(part) == 1 and part.isupper():
            label = part
            break
    target = DST / label / (p.stem + ".npy")
    # reanudable: las REALES tienen 30 frames identicos (foto estatica repetida);
    # el mock varia entre frames. Sin leer de mas: solo rehacer lo mock.
    if target.exists() and target.stat().st_size == 15248:
        try:
            old = np.load(str(target), allow_pickle=False)
            if old.shape == (30, 126) and bool((old[0] == old).all()):
                return "skip"
        except Exception:
            pass
    img = cv2.imread(str(p))
    vec = np.zeros(126, dtype=np.float32)
    hands = 0
    if img is not None:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = _DETECTOR.detect(image)
        if res.hand_landmarks:
            hands = len(res.hand_landmarks[:2])
            for h, lm in enumerate(res.hand_landmarks[:2]):
                for j, pt in enumerate(lm):
                    vec[h * 63 + j * 3:(h * 63 + j * 3 + 3)] = (pt.x, pt.y, pt.z)
    if hands == 0:
        return "sinhands"
    left = _bone(vec[:63].reshape(21, 3)).reshape(63)
    right = _bone(vec[63:].reshape(21, 3)).reshape(63)
    frame = np.concatenate([left, right]).astype(np.float32)
    seq = np.repeat(frame[None, :], 30, axis=0)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(target), seq, allow_pickle=False)
    return "ok"


def main() -> None:
    jpgs = sorted(SRC.rglob("*.jpg"))
    PROGRESO.write_text(f"[{time.strftime('%H:%M:%S')}] {len(jpgs)} jpgs, landmarks reales\n", encoding="utf-8")
    print(f"[static-real] {len(jpgs)} jpgs Pool 6", flush=True)
    strs = [str(p) for p in jpgs]
    t0 = time.time()
    ok = sinhands = skip = 0
    batch = 5000
    from collections import Counter
    for i in range(0, len(strs), batch):
        chunk = strs[i : i + batch]
        with multiprocessing.Pool(processes=6, initializer=_init_worker) as pool:
            for r in pool.map(process_one, chunk, chunksize=8):
                if r == "ok":
                    ok += 1
                elif r == "skip":
                    skip += 1
                else:
                    sinhands += 1
        el = time.time() - t0
        msg = f"[{time.strftime('%H:%M:%S')}] {min(i + batch, len(strs))}/{len(strs)} ok={ok} skip={skip} sinhands={sinhands} {ok / el:.1f}/s"
        print(msg, flush=True)
        PROGRESO.write_text(PROGRESO.read_text(encoding="utf-8", errors="ignore") + msg + "\n", encoding="utf-8")
    print(f"[static-real] FIN ok={ok} skip={skip} sinhands={sinhands} en {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
