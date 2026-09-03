"""Audita cobertura local LSM sin descargar ni alterar corpus."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
OUT = ROOT / "artifacts" / "successor_mendeley_recovery" / "lsm_data_coverage_audit.json"
WORD = MANIFESTS / "successor_mendeley_positions126_recovery_extracted.csv"
AUXILIARY = MANIFESTS / "successor_mendeley_auxiliary_lexicon_extracted.csv"
STATIC = MANIFESTS / "lsm_3d_static_alphabet_ccby_v2.csv"
STATIC_CANONICAL = MANIFESTS / "lsm_3d_static_alphabet_canonical_v1.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def histogram(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def summarize_words(rows: list[dict[str, str]]) -> dict[str, object]:
    unique_ids = {row["sample_id"] for row in rows}
    label_counts = Counter(row["label_lsm"] for row in rows)
    by_split = defaultdict(Counter)
    by_signer = defaultdict(Counter)
    per_label_split = defaultdict(Counter)
    for row in rows:
        by_split[row["split_model"]][row["label_lsm"]] += 1
        by_signer[row["signer_id"]][row["label_lsm"]] += 1
        per_label_split[row["label_lsm"]][row["split_model"]] += 1
    return {
        "manifest": str(WORD),
        "clips_rows": len(rows),
        "clips_unique_sample_id": len(unique_ids),
        "word_classes": len(label_counts),
        "clips_per_word_min": min(label_counts.values()),
        "clips_per_word_max": max(label_counts.values()),
        "clips_per_word_histogram": histogram([str(value) for value in label_counts.values()]),
        "clips_by_split": {split: sum(counts.values()) for split, counts in sorted(by_split.items())},
        "clips_by_signer": {signer: sum(counts.values()) for signer, counts in sorted(by_signer.items())},
        "per_word_split_pattern_histogram": histogram(["|".join(f"{split}:{counts[split]}" for split in sorted(counts)) for counts in per_label_split.values()]),
        "leakage_guard": "Cada palabra debe tener S01-S07 train, S08 validation y S09 test; este reporte solo cuenta el manifiesto y no evalúa S09.",
    }


def summarize_auxiliary(rows: list[dict[str, str]], word_labels: set[str]) -> dict[str, object]:
    label_field = "label_lsm" if "label_lsm" in rows[0] else "label"
    labels = {row[label_field] for row in rows}
    return {
        "manifest": str(AUXILIARY),
        "clips_rows": len(rows),
        "clips_unique_sample_id": len({row["sample_id"] for row in rows}),
        "classes": len(labels),
        "label_overlap_with_210_word_benchmark": sorted(labels & word_labels),
        "status": "Ya auditado como léxico auxiliar separado; no se contará como más ejemplos de las 210 palabras.",
    }


def summarize_static(rows: list[dict[str, str]], canonical_rows: list[dict[str, str]]) -> dict[str, object]:
    hashes = {row["sha256"] for row in rows}
    canonical_hashes = {row["source_v1_sha256"] for row in canonical_rows}
    labels = Counter(row["label_lsm_static"] for row in rows)
    return {
        "manifest": str(STATIC),
        "clips_rows": len(rows),
        "clips_unique_sha256": len(hashes),
        "static_letter_classes": len(labels),
        "clips_per_static_letter_min": min(labels.values()),
        "clips_per_static_letter_max": max(labels.values()),
        "clips_per_static_letter": dict(sorted(labels.items())),
        "performer_index_is_unverified": True,
        "license": sorted({row["license"] for row in rows}),
        "canonical_manifest_hash_overlap": len(hashes & canonical_hashes),
        "canonical_manifest_unique_hashes": len(canonical_hashes),
        "deduplication_rule": "ccby_v2 y canonical_v1 se consideran el mismo origen cuando comparten SHA-256; no se suman entre sí.",
    }


def main() -> None:
    word_rows = read_rows(WORD)
    auxiliary_rows = read_rows(AUXILIARY)
    static_rows = read_rows(STATIC)
    canonical_rows = read_rows(STATIC_CANONICAL)
    words = summarize_words(word_rows)
    report = {
        "scope": "Inventario local pasivo; no descarga archivos, no usa S09 para inferencia y no altera corpus.",
        "words_210": words,
        "auxiliary_39_classes": summarize_auxiliary(auxiliary_rows, {row["label_lsm"] for row in word_rows}),
        "static_letters_3d": summarize_static(static_rows, canonical_rows),
        "dynamic_letters": {
            "local_labeled_training_corpus_found": False,
            "count": 0,
            "meaning": "La app declara seis letras dinámicas por contrato de producto, pero el workspace local no contiene un manifiesto de clips etiquetados para entrenarlas o auditarlas. No se debe inventar un conteo ni reutilizar el corpus 3D estático como si fuera dinámico.",
        },
        "no_repeat_constraint": "Cualquier fuente nueva debe deduplicarse por licencia/origen, hash de archivo cuando exista, URL/DOI, etiqueta, firmante y clip antes de cualquier descarga o integración.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()