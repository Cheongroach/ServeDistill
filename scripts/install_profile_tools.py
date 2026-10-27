"""Extract SHA256-pinned official Nsight CLI packages without root."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import urllib.request

from servedistill.profiling import digest, locate


def install(lock: dict, name: str, root: Path) -> str:
    if platform.machine() != "x86_64":
        raise ValueError("Pinned packages support Linux x86_64 only")
    package = lock["packages"][name]
    root.mkdir(parents=True, exist_ok=True)
    archive = root / package["filename"]
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        with urllib.request.urlopen(lock["repository"] + package["filename"], timeout=60) as response:
            with temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        temporary.replace(archive)
    if digest(archive) != package["sha256"]:
        raise ValueError("Official package checksum mismatch: " + name)
    if "bytes" in package and archive.stat().st_size != package["bytes"]:
        raise ValueError("Official package size mismatch: " + name)
    target = root / name
    subprocess.run(["dpkg-deb", "-x", str(archive), str(target)], check=True, timeout=120)
    matches = [p for p in target.rglob(name) if p.is_file() and p.stat().st_mode & 0o111]
    binary = str(sorted(matches)[0]) if matches else None
    if not binary:
        raise FileNotFoundError("Extracted profiler executable missing: " + name)
    return binary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path("configs/monitoring/tools.lock.json"))
    parser.add_argument("--tools", nargs="+", choices=("nsys", "ncu"), default=["nsys", "ncu"])
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    records = {}
    for name in args.tools:
        binary = install(lock, name, args.root)
        version = subprocess.run([binary, "--version"], capture_output=True, text=True, check=True)
        records[name] = {"binary": binary, "version": version.stdout, **lock["packages"][name]}
    (args.root / "installation-audit.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
