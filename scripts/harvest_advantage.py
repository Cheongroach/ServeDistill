"""Collect small persisted evidence without copying model weights or credentials."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import subprocess


def harvest(remote: str, output: Path):
    environment = {**os.environ, "HADOOP_CONF_SAFE_REFRESH_RUNNING": "true"}
    result = subprocess.run(
        ["hdfs", "dfs", "-ls", "-R", remote], capture_output=True, text=True,
        env=environment, timeout=45)
    if result.returncode:
        return {"available": False, "listing_returncode": result.returncode, "files": 0}
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[0].startswith("-"):
            continue
        path = fields[-1]
        prefix = remote.rstrip("/") + "/"
        if not path.startswith(prefix):
            # HDFS listings can omit the scheme/authority.
            prefix = "/" + remote.split("/", 3)[-1].strip("/") + "/"
        if not path.startswith(prefix):
            raise ValueError("Listing escaped the evidence root")
        relative = PurePosixPath(path[len(prefix):])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe relative evidence path")
        if relative.suffix not in (".json", ".jsonl", ".log"):
            continue
        if relative.name.endswith((".tmp", ".upload")) or "teacher-export" in relative.parts:
            continue
        target = output / str(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".download")
        with temporary.open("wb") as stdout:
            copied = subprocess.run(
                ["hdfs", "dfs", "-cat", path], stdout=stdout, stderr=subprocess.PIPE,
                env=environment, timeout=30)
        if copied.returncode:
            temporary.unlink(missing_ok=True)
            continue
        temporary.replace(target)
        files.append(str(relative))
    summary = {"available": True, "files": len(files), "relative_paths": files}
    (output / "harvest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = harvest(args.remote, args.output)
    print(json.dumps({key: result[key] for key in ("available", "files")}))


if __name__ == "__main__":
    main()
