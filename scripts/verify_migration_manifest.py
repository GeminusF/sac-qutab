from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_migration_manifest.py MANIFEST ROOT")
    manifest_path, root = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != "sac-migration-manifest-v1":
        raise ValueError("unsupported migration manifest")
    for row in manifest.get("files", []):
        relative = Path(row["path"])
        candidate = (root / relative).resolve()
        if root != candidate and root not in candidate.parents:
            raise ValueError(f"manifest path escapes restore root: {relative}")
        if not candidate.is_file() or candidate.stat().st_size != int(row["size"]):
            raise ValueError(f"missing or wrong-sized file: {relative}")
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {relative}")
    if len(manifest.get("files", [])) != int(manifest.get("file_count", -1)):
        raise ValueError("migration manifest file count mismatch")
    print(json.dumps({"status": "verified", "files": manifest["file_count"], "root": str(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
