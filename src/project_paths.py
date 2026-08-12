from __future__ import annotations

import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path) -> str:
    """返回不泄露作者本机目录的项目相对路径。"""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(portable_path(path).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()
