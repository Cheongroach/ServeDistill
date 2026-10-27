"""Materialize exact executed DEV sources while verifying their frozen hashes."""
import argparse
import json
from pathlib import Path
import tarfile

from servedistill.profiling import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", choices=("anchor-development-v1", "anchor-weight-development-v2"),
                        default="anchor-development-v1")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = repo / "experiments" / args.experiment
    manifest = json.loads((root / "manifest.json").read_text())
    archive_path = root / manifest["file"]
    protocol_path = repo / "configs" / (args.experiment + ".json")
    if (digest(archive_path) != manifest["sha256"]
            or digest(protocol_path) != manifest["protocol_sha256"]):
        raise ValueError("Frozen source archive or protocol changed")
    support_path = root / manifest["support_file"]
    if digest(support_path) != manifest["support_sha256"]:
        raise ValueError("Frozen worker support archive changed")
    protocol = json.loads(protocol_path.read_text())
    source_files = protocol["source_files"].copy()
    base_path = None
    if "base_protocol_sha256" in protocol:
        base_path = repo / "configs/anchor-development-v1.json"
        if digest(base_path) != protocol["base_protocol_sha256"]:
            raise ValueError("Original base protocol changed")
        source_files.update(json.loads(base_path.read_text())["source_files"])
    args.output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                raise ValueError("Unsafe source archive entry")
        archive.extractall(args.output)
    with tarfile.open(support_path) as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                raise ValueError("Unsafe worker support entry")
        archive.extractall(args.output)
    for relative, expected in source_files.items():
        if digest(args.output / relative) != expected:
            raise ValueError("Materialized source hash mismatch")
    (args.output / "anchor-protocol.json").write_bytes((base_path or protocol_path).read_bytes())
    if base_path:
        (args.output / "anchor-weight-protocol.json").write_bytes(protocol_path.read_bytes())
    (args.output / "pyproject.toml").write_bytes((repo / "pyproject.toml").read_bytes())
    (args.output / "LICENSE").write_bytes((repo / "LICENSE").read_bytes())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
