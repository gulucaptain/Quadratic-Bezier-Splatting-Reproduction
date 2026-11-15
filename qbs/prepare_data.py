"""Re-download the exact pilot inputs from the recorded immutable manifest."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    args = p.parse_args()
    for row in json.loads(args.manifest.read_text()):
        path = Path(row["file"])
        if not path.exists() or args.force:
            data = urllib.request.urlopen(row["url"], timeout=60).read()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("Download hash mismatch")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Local hash mismatch: {path}")
        print("verified", path)


if __name__ == "__main__":
    main()
