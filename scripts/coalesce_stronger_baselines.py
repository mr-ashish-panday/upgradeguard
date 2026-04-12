from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill missing stronger-baseline fields in a target JSON from one or more source JSON files."
    )
    parser.add_argument("--target", required=True, help="Target stronger_baselines.json to create/update.")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source JSON file to merge from. Earlier sources have higher fill priority.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _should_fill(current: Any, candidate: Any) -> bool:
    if candidate is None:
        return False
    if current is None:
        return True
    return False


def main() -> None:
    args = parse_args()
    target_path = Path(args.target).resolve()
    payload: dict[str, Any] = {}
    if target_path.exists():
        payload.update(_load_json(target_path))

    for source_str in args.source:
        source_path = Path(source_str).resolve()
        source_payload = _load_json(source_path)
        for key, value in source_payload.items():
            if _should_fill(payload.get(key), value):
                payload[key] = value

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(target_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
