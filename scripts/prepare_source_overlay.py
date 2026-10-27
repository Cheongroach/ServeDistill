"""Safely apply a SHA256-bound source/data archive to an isolated worker copy."""
import argparse
import hashlib
from pathlib import Path
import tarfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--code", type=Path, required=True)
    args = parser.parse_args()
    with args.archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != args.sha256:
        raise ValueError("Source overlay checksum mismatch")
    with tarfile.open(args.archive) as archive:
        members = archive.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe overlay path")
        archive.extractall(args.code, members=members)


if __name__ == "__main__":
    main()
