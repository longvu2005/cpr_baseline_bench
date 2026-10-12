"""Content-validated stage caches with separate read roots and a write root."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resolve(value) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@contextlib.contextmanager
def atomic_file(path: Path, mode="wb"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, mode) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def atomic_json(path, payload):
    with atomic_file(Path(path), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


@contextlib.contextmanager
def cache_lock(directory: Path):
    """OS locks are released on a crash; no stale-lock deletion heuristics."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


class StageCache:
    def __init__(self, stage, fingerprint, *, write_root=None, read_roots=None):
        self.stage, self.fingerprint = stage, fingerprint
        self.key = canonical_hash(fingerprint)
        self.write_root = resolve(write_root or os.environ.get("CPR_SHARED_CACHE_ROOT", "cache/shared"))
        if read_roots is None:
            read_roots = [p for p in os.environ.get("CPR_SHARED_CACHE_READ_ROOTS", "").split(os.pathsep) if p]
        # Prefer a repaired writable entry over a corrupt mounted entry.
        self.read_roots = list(dict.fromkeys([self.write_root, *(resolve(p) for p in read_roots)]))
        self.directory = self.write_root / stage / self.key

    def find(self, filename):
        for root in self.read_roots:
            path = root / self.stage / self.key / filename
            meta = read_json(path.with_name(path.name + ".meta.json"))
            try:
                if (isinstance(meta, dict) and meta.get("fingerprint") == self.fingerprint
                        and meta.get("size") == path.stat().st_size
                        and meta.get("sha256") == sha256_file(path)):
                    return path
            except OSError:
                pass
        return None

    def seal(self, path, **extra):
        path = Path(path)
        atomic_json(path.with_name(path.name + ".meta.json"), {
            "fingerprint": self.fingerprint, "sha256": sha256_file(path),
            "size": path.stat().st_size, **extra,
        })
        return path

    def write_json(self, filename, payload):
        path = self.directory / filename
        atomic_json(path, payload)
        return self.seal(path)


def feature_config_hash(path):
    """Exclude execution and SetMatch settings from expensive native embeddings."""
    import yaml
    cfg = yaml.safe_load(Path(path).read_text())
    return canonical_hash({k: v for k, v in cfg.items()
                           if k not in {"setmatch", "runtime", "output", "cache", "display_name", "group"}})
