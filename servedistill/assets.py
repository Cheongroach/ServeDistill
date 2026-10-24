"""Download official Qwen mirror files in parallel, pinned by SHA256."""
from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

import requests

PATTERNS = (
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "tokenizer.model", "chat_template.jinja", "model*.safetensors",
    "model.safetensors.index.json",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(root: Path, model: str, entry: dict) -> tuple[str, dict]:
    name = entry["Path"]
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Unsafe repository path")
    expected, size, revision = entry["Sha256"], entry["Size"], entry["Revision"]
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Missing repository SHA256")
    target = root / "model" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not (target.is_file() and target.stat().st_size == size and sha256(target) == expected):
        partial = target.with_name(target.name + ".prefetch")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError("Partial file exceeds source size")
        url = (f"https://modelscope.cn/models/{model}/resolve/"
               f"{quote(revision, safe='')}/{quote(name, safe='/')}")
        if offset < size:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise RuntimeError("Source ignored Range; partial preserved")
                with partial.open("ab" if offset else "wb") as stream:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        stream.write(chunk)
        if partial.stat().st_size != size or sha256(partial) != expected:
            raise ValueError("Repository size/SHA256 mismatch: " + name)
        partial.replace(target)
    print(json.dumps({"event": "asset_verified", "model": model, "file": name,
                      "bytes": size}), flush=True)
    return name, {"sha256": expected, "bytes": size, "file_revision": revision}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--exclude", nargs="*", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pin = args.output / "source-listing.json"
    if pin.exists():
        source = json.loads(pin.read_text())
        if source["model"] != args.model or source["revision"] != args.revision:
            raise ValueError("Cached model source differs from requested source")
        entries = source["entries"]
    else:
        response = requests.get(
            f"https://modelscope.cn/api/v1/models/{args.model}/repo/files",
            params={"Revision": args.revision, "Recursive": "true"}, timeout=60)
        response.raise_for_status()
        listing = response.json()
        if listing.get("Code") != 200:
            raise RuntimeError("Repository listing failed")
        entries = [e for e in listing["Data"]["Files"] if e["Type"] == "blob"
                   and any(fnmatch.fnmatch(e["Path"], p) for p in PATTERNS)]
        pin.write_text(json.dumps({"model": args.model, "revision": args.revision,
                                   "entries": entries}, indent=2) + "\n")
    selected = [e for e in entries if e["Path"] not in args.exclude]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, args.output, args.model, e) for e in selected]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    if args.exclude:
        return  # A cooperating download owns the excluded files.
    files = {e["Path"]: {"sha256": e["Sha256"], "bytes": e["Size"],
                         "file_revision": e["Revision"]} for e in entries}
    manifest = {"repo_id": args.model, "source": "modelscope",
                "requested_revision": args.revision, "path": "model", "files": files}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
