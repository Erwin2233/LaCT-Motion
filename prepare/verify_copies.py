"""Verify local files, optional large resources, and preserved source hashes."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def tree_digest(path):
    """Hash sorted relative filenames, sizes, and individual file hashes."""
    checksum = hashlib.sha256()
    paths = sorted(p for p in path.rglob("*") if p.is_file())
    total_bytes = 0
    for item in paths:
        size = item.stat().st_size
        total_bytes += size
        name = item.relative_to(path).as_posix()
        checksum.update(f"{name}\0{size}\0{digest(item)}\n".encode())
    return checksum.hexdigest(), len(paths), total_bytes


def check(path, record, expected_hash, label, errors):
    if record.get("kind") == "tree":
        if not path.is_dir():
            errors.append(f"Missing {label}: {path}")
            return
        actual_hash, count, size = tree_digest(path)
        if count != record["files"] or size != record["bytes"]:
            errors.append(f"Changed tree contents for {label}: {path}")
    else:
        if not path.is_file():
            errors.append(f"Missing {label}: {path}")
            return
        actual_hash = digest(path)
    if actual_hash != expected_hash:
        errors.append(f"Changed {label}: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sources", action="store_true")
    parser.add_argument("--include-resources", action="store_true",
                        help="Also hash model weights and raw dataset trees")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs" / "COPY_MANIFEST.json").read_text())
    source_roots = {
        Path(path).name: (root / path).resolve()
        for path in manifest["source_roots"].values()
    }
    errors = []
    checked_sources = 0
    records = list(manifest["files"])
    if args.include_resources:
        records.extend(manifest.get("resources", []))
    for record in records:
        check(root / record["destination"], record, record["sha256"], "copy", errors)
        if args.check_sources:
            for source in [record] + record.get("additional_sources", []):
                path = source_roots[source["source_project"]] / source["source_path"]
                expected = source.get("source_sha256", source.get("sha256", record["sha256"]))
                check(path, source, expected, "source", errors)
                checked_sources += 1
    for record in manifest.get("added_files", []):
        check(root / record["destination"], record, record["sha256"], "addition", errors)
    for error in errors:
        print(error)
    if errors:
        raise SystemExit(1)
    suffix = f", {checked_sources} source records" if args.check_sources else ""
    print(f"Verified {len(records)} copied file/tree records, "
          f"{len(manifest.get('added_files', []))} additions{suffix}; "
          "all SHA-256 checks passed.")


if __name__ == "__main__":
    main()
