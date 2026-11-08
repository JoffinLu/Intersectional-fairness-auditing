"""Download the three official source archives and verify their recorded SHA256 values."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
MANIFEST = DATA / "download_manifest.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check local archives without using the network")
    args = parser.parse_args()
    metadata = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, info in metadata.items():
        if Path(name).name != name or not name.endswith(".zip"):
            raise ValueError("Unexpected source filename")
        path = DATA / name
        if not path.exists():
            if args.verify_only:
                raise FileNotFoundError(path)
            partial = path.with_suffix(".zip.part")
            request = urllib.request.Request(info["source_url"], headers={"User-Agent": "Intersectional-fairness-auditing-reproduction"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if digest(partial) != info["sha256"]:
                raise ValueError(f"Source archive hash differs from recorded data: {name}; partial file retained")
            partial.replace(path)
            info["downloaded_utc"] = datetime.now(timezone.utc).isoformat()
        if digest(path) != info["sha256"]:
            raise ValueError(f"Source archive hash differs from recorded data: {name}")
        if path.stat().st_size != info["bytes"]:
            raise ValueError(f"Source archive length differs: {name}")
        info["path"] = path.relative_to(ROOT).as_posix()
        print(f"Verified {name}: {info['sha256']}")
    if not args.verify_only:
        MANIFEST.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
