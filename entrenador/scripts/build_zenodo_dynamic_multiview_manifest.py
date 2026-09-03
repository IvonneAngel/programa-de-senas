"""Empareja landmarks frontal/perfil Zenodo para un control externo aislado."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LETTERS = frozenset(("J", "K", "Ñ", "Q", "X", "Z"))
VIEWS = frozenset(("frontal", "profile"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"participant_id", "letter_lsm", "repetition", "split_original", "feature_path", "feature_status"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifiesto inválido: {path}")
    rows = [row for row in rows if row["feature_status"] == "ok"]
    if len(rows) != 600:
        raise ValueError(f"Se requieren 600 landmarks correctos en {path}; hay {len(rows)}")
    return rows


def sample_key(row: dict[str, str]) -> tuple[int, str, int]:
    return (int(row["participant_id"]), row["letter_lsm"].upper(), int(row["repetition"]))


def validate_source(rows: list[dict[str, str]], source_name: str) -> dict[tuple[int, str, int], dict[str, str]]:
    indexed = {sample_key(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Muestras duplicadas en {source_name}")
    participants = {key[0] for key in indexed}
    labels = {key[1] for key in indexed}
    repetitions = {key[2] for key in indexed}
    if len(participants) != 20 or labels != LETTERS or repetitions != {1, 2, 3, 4, 5}:
        raise ValueError(f"Cobertura canónica inválida en {source_name}")
    if {row["split_original"] for row in rows} != {"train", "test"}:
        raise ValueError(f"Split oficial ausente en {source_name}")
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontal-manifest", type=Path, required=True)
    parser.add_argument("--profile-manifest", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()

    frontal = validate_source(read_rows(args.frontal_manifest), "frontal")
    profile = validate_source(read_rows(args.profile_manifest), "profile")
    if set(frontal) != set(profile):
        raise ValueError("Frontal y perfil no emparejan uno-a-uno por participante, letra y repetición")
    for key in frontal:
        if frontal[key]["split_original"] != profile[key]["split_original"]:
            raise ValueError("Fuga de split entre vistas emparejadas")

    official_train = sorted({key[0] for key, row in frontal.items() if row["split_original"] == "train"})
    official_test = sorted({key[0] for key, row in frontal.items() if row["split_original"] == "test"})
    if len(official_train) != 18 or len(official_test) != 2:
        raise ValueError("El split oficial debe contener 18 firmantes train y 2 test")
    validation_signers = set(official_train[-2:])
    train_signers = set(official_train[:-2])
    rows: list[dict[str, str]] = []
    for participant, letter, repetition in sorted(frontal):
        original = frontal[(participant, letter, repetition)]["split_original"]
        split = "test" if original == "test" else ("validation" if participant in validation_signers else "train")
        pair_id = f"S{participant:02d}-{letter}-{repetition}"
        for view, indexed in (("frontal", frontal), ("profile", profile)):
            source = indexed[(participant, letter, repetition)]
            rows.append({
                "sample_id": f"zenodo_dynamic_{pair_id}-{view}",
                "pair_id": pair_id,
                "participant_id": f"S{participant:02d}",
                "letter_lsm": letter,
                "repetition": str(repetition),
                "view": view,
                "split_original": original,
                "split_external": split,
                "feature_path": source["feature_path"],
                "source_manifest": str(args.frontal_manifest if view == "frontal" else args.profile_manifest),
            })

    counts = {split: sum(row["split_external"] == split for row in rows) for split in ("train", "validation", "test")}
    if counts != {"train": 960, "validation": 120, "test": 120}:
        raise ValueError(f"Particiones inesperadas: {counts}")
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.out_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "kind": "zenodo_dynamic_lsm_multiview_manifest",
        "rows": len(rows),
        "pairs": len(rows) // 2,
        "views": sorted(VIEWS),
        "labels": sorted(LETTERS),
        "counts": counts,
        "signers": {"train": sorted(f"S{value:02d}" for value in train_signers), "validation": sorted(f"S{value:02d}" for value in validation_signers), "test": sorted(f"S{value:02d}" for value in official_test)},
        "frontal_profile_paired": True,
        "benchmark_210_words_touched": False,
        "s08_read": False,
        "s09_read": False,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()