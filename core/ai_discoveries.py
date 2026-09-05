"""Persist successful AI solves for the static web showcase."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path


DISCOVERIES_FILE_NAME = "ai-discoveries.json"


def default_discoveries_path() -> Path:
    """Return the web project's discovery feed when it is available locally."""
    configured_path = os.environ.get("TWISTY_WEB_DISCOVERIES_PATH")
    if configured_path:
        return Path(configured_path).expanduser()

    project_root = Path(__file__).resolve().parents[1]
    for parent in project_root.parents:
        for name in ("ルービックキューブマニュアル", "ルービックキューブマニュアル"):
            candidate = parent / name / "assets" / "data" / DISCOVERIES_FILE_NAME
            if candidate.parent.parent.parent.exists():
                return candidate
    return project_root / "exports" / DISCOVERIES_FILE_NAME


def _clean_moves(moves) -> list[str]:
    return [str(move).strip() for move in moves if str(move).strip()]


def _record_id(puzzle: str, setup: list[str]) -> str:
    payload = "\0".join((puzzle, *setup)).encode("utf-8")
    return sha256(payload).hexdigest()[:16]


class AiDiscoveryStore:
    """Store the shortest successful solve for each puzzle and start position."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else default_discoveries_path()

    def save(self, puzzle: str, setup, moves) -> str:
        """Save a discovery and return ``added``, ``shorter``, or ``unchanged``."""
        normalized_puzzle = str(puzzle).strip()
        clean_setup = _clean_moves(setup)
        clean_moves = _clean_moves(moves)
        if not normalized_puzzle:
            raise ValueError("puzzle is required")
        if not clean_moves:
            raise ValueError("moves is required")

        payload = self._read()
        discoveries = payload["discoveries"]
        record_id = _record_id(normalized_puzzle, clean_setup)
        current = next((item for item in discoveries if item.get("id") == record_id), None)
        if current is not None and len(current.get("moves", ())) <= len(clean_moves):
            return "unchanged"

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        record = {
            "id": record_id,
            "puzzle": normalized_puzzle,
            "setup": clean_setup,
            "moves": clean_moves,
            "moveCount": len(clean_moves),
            "foundAt": current.get("foundAt", now) if current else now,
            "updatedAt": now,
        }
        if current is None:
            discoveries.append(record)
            outcome = "added"
        else:
            discoveries[discoveries.index(current)] = record
            outcome = "shorter"

        discoveries.sort(key=lambda item: (item["moveCount"], item["updatedAt"], item["id"]))
        payload["updatedAt"] = now
        self._write(payload)
        return outcome

    def _read(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "updatedAt": None, "discoveries": []}
        with self.path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict) or not isinstance(payload.get("discoveries"), list):
            raise ValueError(f"Invalid discovery file: {self.path}")
        payload.setdefault("version", 1)
        payload.setdefault("updatedAt", None)
        return payload

    def _write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary_path.replace(self.path)
