"""Small, local artifact and provenance utilities for this research project."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value, *, version: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, ensure_ascii=False, default=json_default, allow_nan=False) + "\n"
    if version and path.exists():
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
        version_path = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        version_path.write_text(content, encoding="utf-8")
    path.write_text(content, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(path: Path, description: str, stage: str = "implementation", skill: str | None = None) -> None:
    """Append a relative-path scientific artifact record after generating results."""
    path = Path(path).resolve()
    try:
        label = path.relative_to(ROOT).as_posix()
    except ValueError:
        label = path.name
    target = ROOT / "analysis" / "artifact_manifest.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {"created_at": now(), "path": label, "stage": stage, "description": description}
    if path.is_file():
        record["sha256"] = sha256(path)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
