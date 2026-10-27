"""Publish stable files atomically, reusing verified identical destinations."""
from pathlib import Path
import hashlib
import shutil

def signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def content_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if not path.is_file() or path.name.endswith((".tmp", ".upload")):
            continue
        target = destination / path.relative_to(source)
        before = signature(path)
        existing = signature(target) if target.is_file() else None
        expected = content_digest(path)
        if signature(path) != before:
            raise RuntimeError("Publication source changed while hashing")
        if existing is not None and before[2] == existing[2]:
            actual = content_digest(target)
            if signature(path) != before or signature(target) != existing:
                raise RuntimeError("Publication input changed during comparison")
            if actual == expected:
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".upload")
        shutil.copyfile(path, temporary)
        if content_digest(temporary) != expected or signature(path) != before:
            raise RuntimeError("Publication copied content differs from stable source")
        temporary.replace(target)
