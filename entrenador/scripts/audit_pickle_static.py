"""Inspección estática de pickle: no deserializa ni ejecuta objetos."""
from __future__ import annotations

import argparse
import collections
import json
import pickletools
from pathlib import Path


EXECUTION_RELATED = {"GLOBAL", "STACK_GLOBAL", "REDUCE", "NEWOBJ", "NEWOBJ_EX", "OBJ", "INST", "BUILD", "PERSID", "BINPERSID"}


def inspect(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    operations: collections.Counter[str] = collections.Counter()
    positions: list[dict[str, object]] = []
    try:
        for opcode, argument, position in pickletools.genops(raw):
            operations[opcode.name] += 1
            if opcode.name in EXECUTION_RELATED:
                positions.append({"opcode": opcode.name, "position": position, "argument": repr(argument)[:240]})
    except Exception as exc:  # Rechaza en vez de intentar cargar contenido ambiguo.
        return {"file": path.name, "bytes": len(raw), "static_parse_ok": False, "error": f"{type(exc).__name__}: {exc}", "deserialized": False, "executed": False, "training_eligible": False}
    return {"file": path.name, "bytes": len(raw), "static_parse_ok": True, "opcode_counts": dict(sorted(operations.items())), "execution_related_opcodes": positions, "deserialized": False, "executed": False, "training_eligible": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {"kind": "static_pickle_quarantine_audit", "deserialized": False, "executed": False, "training_eligible": False, "files": [inspect(path) for path in args.files]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(report["files"]), "deserialized": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()