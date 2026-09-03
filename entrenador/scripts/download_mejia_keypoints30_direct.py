"""Descarga reanudable del corpus público Mejía de keypoints LSM."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from gdown.download_folder import download_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def download_one(file_id: str, output: Path) -> str:
    if output.exists() and output.stat().st_size > 0:
        return "skipped"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    url = "https://drive.usercontent.google.com/download"
    with requests.get(
        url,
        params={"id": file_id, "export": "download", "confirm": "t"},
        stream=True,
        timeout=60,
    ) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            raise RuntimeError(f"Drive devolvió HTML para {file_id}")
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)
    if temporary.stat().st_size == 0:
        raise RuntimeError(f"Descarga vacía para {file_id}")
    temporary.replace(output)
    return "downloaded"


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 6:
        raise ValueError("--workers debe estar entre 1 y 6")
    files = download_folder(
        id=args.folder_id,
        output=str(args.output),
        quiet=True,
        skip_download=True,
    )
    selected = files[: args.limit] if args.limit else files
    report = {
        "folder_id": args.folder_id,
        "expected": len(files),
        "selected": len(selected),
        "downloaded": 0,
        "skipped": 0,
        "errors": [],
        "source_partition_preserved": True,
        "benchmark_210_words_touched": False,
        "s08_read": False,
        "s09_read": False,
    }
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, item.id, Path(item.local_path)): item
            for item in selected
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                report[future.result()] += 1
            except Exception as error:  # pragma: no cover - red de terceros
                report["errors"].append({"path": item.path, "error": str(error)})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "direct_download_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()