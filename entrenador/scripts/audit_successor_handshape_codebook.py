"""Audita un codebook K-means de configuraciones manuales solo en S01-S07."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


CLUSTERS = 32
MAX_FIT_FRAMES = 20_000
N_INIT = 8


def observed_hands(values: np.ndarray) -> np.ndarray:
    sequence = np.asarray(values, dtype=np.float32)
    if sequence.shape != (30, 126) or not np.isfinite(sequence).all():
        raise ValueError(f"Secuencia inválida: {sequence.shape}")
    hands = sequence.reshape(30, 2, 63).reshape(60, 63)
    return hands[np.linalg.norm(hands, axis=1) > 1e-6]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--codebook-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [row for row in rows if row["split_model"] == "train"]
    if len(train) != 1470 or any(row["signer_id"] not in {f"S{i:02d}" for i in range(1, 8)} for row in train):
        raise ValueError("El codebook solo puede ajustarse a los 1,470 clips S01-S07")
    frames = np.concatenate([observed_hands(np.load(args.cache_root / row["feature_path"], allow_pickle=False)) for row in train], axis=0)
    if frames.shape[0] < CLUSTERS or frames.shape[1] != 63:
        raise ValueError("Frames observados insuficientes para el codebook")
    selected = np.linspace(0, frames.shape[0] - 1, min(MAX_FIT_FRAMES, frames.shape[0]), dtype=np.int64)
    fit_frames = frames[selected]
    primary = KMeans(n_clusters=CLUSTERS, random_state=20_260_819, n_init=N_INIT, max_iter=300, algorithm="lloyd").fit(fit_frames)
    twin = KMeans(n_clusters=CLUSTERS, random_state=20_260_820, n_init=N_INIT, max_iter=300, algorithm="lloyd").fit(fit_frames)
    labels = primary.labels_
    counts = np.bincount(labels, minlength=CLUSTERS)
    entropy = -float(np.sum((counts / counts.sum()) * np.log((counts / counts.sum()).clip(min=1e-12)))) / float(np.log(CLUSTERS))
    stability = float(adjusted_rand_score(primary.labels_, twin.labels_))
    args.codebook_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.codebook_out, centers=primary.cluster_centers_.astype(np.float32), clusters=np.int32(CLUSTERS), feature_dim=np.int32(63), fitted_frames=np.int32(fit_frames.shape[0]), source="bone_vector126_train_S01_to_S07_only")
    report = {
        "kind": "successor_handshape_codebook_train_only_audit",
        "split_read": "train_only_S01_to_S07",
        "s08_read": False,
        "s09_read": False,
        "cache_written": False,
        "labels_used": False,
        "clips": len(train),
        "classes_present_but_unused": len({row["label_lsm"] for row in train}),
        "observed_hand_frames": int(frames.shape[0]),
        "fit_frames": int(fit_frames.shape[0]),
        "clusters": CLUSTERS,
        "active_clusters": int(np.count_nonzero(counts)),
        "min_cluster_frames": int(counts.min()),
        "max_cluster_frames": int(counts.max()),
        "normalized_usage_entropy": entropy,
        "seed_pair_adjusted_rand_index": stability,
        "codebook_path": str(args.codebook_out),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()