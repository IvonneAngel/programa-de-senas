"""Audita por API el catálogo de hmsc33hmkz sin descargar ni deserializar archivos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests


DATASET = "hmsc33hmkz"
VERSION = 1


def folder_listing() -> list[dict[str, Any]]:
    response = requests.get(
        f"https://data.mendeley.com/public-api/datasets/{DATASET}/folders/{VERSION}",
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("Respuesta de carpetas Mendeley inesperada")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw_folders = folder_listing()
    nodes = {str(item["id"]): item for item in raw_folders}
    if not nodes:
        raise ValueError("El catálogo de carpetas no puede estar vacío")

    def resolve_path(folder_id: str, visiting: set[str] | None = None) -> str:
        visiting = set() if visiting is None else visiting
        if folder_id in visiting:
            raise ValueError(f"Ciclo de carpetas detectado: {folder_id}")
        visiting.add(folder_id)
        item = nodes[folder_id]
        name = str(item["name"])
        parent_id = str(item.get("parent_id", ""))
        return name if parent_id not in nodes else f"{resolve_path(parent_id, visiting)}/{name}"

    folders = [{"id": folder_id, "path": resolve_path(folder_id), "parent_id": item.get("parent_id")} for folder_id, item in nodes.items()]
    folders.sort(key=lambda item: str(item["path"]))
    dynamic_root = next((item for item in folders if str(item["path"]).endswith("MP_Data_KEYPOINTS_videos")), None)
    dynamic_children = [] if dynamic_root is None else [item for item in folders if item["parent_id"] == dynamic_root["id"]]
    model_markers = ("dynamic_data", "static_data")
    report = {
        "kind": "mendeley_hmsc33hmkz_folder_catalog",
        "dataset": DATASET,
        "version": VERSION,
        "downloaded_payload_files": 0,
        "deserialized": False,
        "executed": False,
        "folders": folders,
        "summary": {
            "folders": len(folders),
            "dynamic_keypoint_root": None if dynamic_root is None else dynamic_root["path"],
            "dynamic_keypoint_labels": [item["path"].split("/")[-1] for item in dynamic_children],
            "model_directories": [item["path"] for item in folders if any(marker in str(item["path"]).lower() for marker in model_markers)],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"folders": len(folders), "dynamic_keypoint_labels": report["summary"]["dynamic_keypoint_labels"], "downloaded_payload_files": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()