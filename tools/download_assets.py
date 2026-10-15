"""Download weights only from official repositories or byte-verified mirrors."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("plan", type=Path)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    def fetch(item):
        dst = (args.root / item["destination"]).resolve()
        if not dst.is_relative_to(args.root.resolve()):
            raise ValueError("Unsafe download destination")
        dst.parent.mkdir(parents=True, exist_ok=True)
        valid = lambda f: f.exists() and f.stat().st_size == item["bytes"] and (not item["sha256"] or sha(f) == item["sha256"])
        if not valid(dst):
            part = dst.with_name(dst.name + ".partial")
            for attempt in range(3):
                # Small config/license files come from the immutable upstream revision.
                url = f'https://huggingface.co/{item["repo"]}/resolve/{item["revision"]}/{item["source"]}'
                if item["sha256"] and attempt < 2:
                    url = f'https://modelscope.cn/models/{item["mirror_repo"]}/resolve/master/{item["source"]}'
                try:
                    with requests.get(url, stream=True, timeout=(20, 90)) as r:
                        r.raise_for_status()
                        with part.open("wb") as f:
                            for chunk in r.iter_content(2 * 1024**2):
                                f.write(chunk)
                    if not valid(part):
                        raise ValueError("Size/hash mismatch: " + item["destination"])
                    part.replace(dst)
                    break
                except Exception as exc:
                    print("retry", item["destination"], attempt, str(exc)[:150], flush=True)
                    if attempt == 2:
                        raise
                    time.sleep(2)
        result = {**item, "verified_sha256": sha(dst)}
        print("verified", item["destination"], dst.stat().st_size, flush=True)
        return result
    with ThreadPoolExecutor(max_workers=6) as pool:
        verified = list(pool.map(fetch, plan["files"]))
    (args.root / "asset-verification.json").write_text(json.dumps({"repositories": plan["repositories"], "files": verified}, indent=2))

if __name__ == "__main__":
    main()
