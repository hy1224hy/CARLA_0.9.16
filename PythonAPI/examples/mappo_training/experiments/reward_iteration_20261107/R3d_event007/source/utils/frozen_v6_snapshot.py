"""Verify/extract the pinned pre-base-driving V6 runtime; never connect to CARLA."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reproducibility/v6_before_base_driving_20261020.zip"
MANIFEST = ARCHIVE.with_suffix(".json")


def verify():
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != record["archive_sha256"]:
        raise ValueError("frozen V6 archive hash mismatch")
    with zipfile.ZipFile(ARCHIVE) as archive:
        if sorted(archive.namelist()) != sorted(record["files"]):
            raise ValueError("frozen V6 archive inventory mismatch")
        for name, digest in record["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"frozen V6 source mismatch: {name}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-to", type=Path)
    args = parser.parse_args()
    record = verify()
    if args.extract_to:
        destination = args.extract_to.resolve()
        if destination.exists():
            raise FileExistsError("extract requires a new directory; no overwrite")
        with zipfile.ZipFile(ARCHIVE) as archive:
            for name in archive.namelist():
                target = (destination / name).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError("unsafe frozen source path")
            archive.extractall(destination)
    print(json.dumps({"status": "VERIFIED_FILES_ONLY", "files": len(record["files"]),
                      "frozen_sources": len(record["registered_sources"]),
                      "extract_to": str(args.extract_to or "")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
