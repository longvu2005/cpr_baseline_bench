#!/usr/bin/env python3
"""Canonical CPR benchmark data-path validation and gallery-link repair.

The manifests intentionally store portable repository-relative paths such as
``data/gallery/<filename>``.  Hosted runtimes (notably Kaggle) usually expose
the actual images elsewhere and link ``data/gallery`` to that directory.

This module keeps that machine-local concern out of individual baselines.  It
never rewrites the manifests and it never accepts a candidate gallery root
unless every manifest image exists below it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence


GALLERY_ENV = "CPR_GALLERY_SOURCE"
CANONICAL_GALLERY_PREFIX = ("data", "gallery")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no}: JSONL row must be an object")
            rows.append(row)
    return rows


def _gallery_relpath(row: dict[str, Any], index: int) -> Path:
    value = row.get("path")
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"Gallery row {index} has no usable 'path'")

    manifest_path = Path(value)
    if manifest_path.is_absolute():
        raise ValueError(
            f"Gallery row {index} uses an absolute path: {value!r}. "
            "Canonical gallery.jsonl paths must be repository-relative."
        )

    parts = manifest_path.parts
    if tuple(parts[:2]) != CANONICAL_GALLERY_PREFIX or len(parts) <= 2:
        raise ValueError(
            f"Gallery row {index} has non-canonical path {value!r}; expected "
            "'data/gallery/<relative-image-path>'."
        )

    relative = Path(*parts[2:])
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Gallery row {index} has unsafe path {value!r}")
    return relative


def _relative_paths(gallery_rows: Sequence[dict[str, Any]]) -> list[Path]:
    if not gallery_rows:
        raise ValueError("Gallery manifest is empty")
    return [_gallery_relpath(row, index) for index, row in enumerate(gallery_rows)]


def _probe_paths(relative_paths: Sequence[Path]) -> list[Path]:
    """Pick deterministic probes before paying for a full 17k-file audit."""
    n = len(relative_paths)
    positions = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    return [relative_paths[index] for index in positions]


def _root_has(root: Path, relative_paths: Sequence[Path]) -> bool:
    return root.is_dir() and all((root / rel).is_file() for rel in relative_paths)


def _missing(root: Path, relative_paths: Sequence[Path], limit: int = 10) -> list[Path]:
    missing: list[Path] = []
    for relative in relative_paths:
        if not (root / relative).is_file():
            missing.append(relative)
            if len(missing) >= limit:
                break
    return missing


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            normalized = str(path.expanduser().resolve())
        except OSError:
            normalized = str(path.expanduser().absolute())
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(Path(normalized))
    return result


def _kaggle_candidates(relative_paths: Sequence[Path]) -> list[Path]:
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.is_dir():
        return []

    candidates: list[Path] = [
        kaggle_input / "cir-data" / "data" / "raw" / "images" / "train",
        kaggle_input / "cir-data" / "gallery",
        kaggle_input / "cir-data" / "data" / "gallery",
    ]

    # Dataset mount layouts can change (or be nested by a helper such as
    # kagglehub).  Locate one canonical probe, infer its root, then validate the
    # other probes and finally the entire manifest before accepting it.
    probe = relative_paths[0]
    try:
        matches = kaggle_input.rglob(probe.name)
        for match in matches:
            if not match.is_file():
                continue
            root = match
            for _ in probe.parts:
                root = root.parent
            candidates.append(root)
    except OSError:
        pass

    return _dedupe_paths(candidates)


def _discover_gallery_root(relative_paths: Sequence[Path]) -> Path | None:
    probes = _probe_paths(relative_paths)

    override = os.environ.get(GALLERY_ENV, "").strip()
    if override:
        root = Path(override).expanduser()
        if not _root_has(root, probes):
            missing = _missing(root, probes)
            details = "\n".join(f"  - {root / rel}" for rel in missing)
            raise FileNotFoundError(
                f"{GALLERY_ENV} points to an incompatible gallery root: {root}\n"
                f"Missing probe image(s):\n{details}"
            )
        missing = _missing(root, relative_paths, limit=1)
        if missing:
            raise FileNotFoundError(
                f"{GALLERY_ENV} does not contain the complete gallery; first missing: "
                f"{root / missing[0]}"
            )
        return root.resolve()

    for root in _kaggle_candidates(relative_paths):
        if not _root_has(root, probes):
            continue
        if _missing(root, relative_paths, limit=1):
            continue
        return root.resolve()

    return None


def _placeholder_only(directory: Path) -> bool:
    if not directory.is_dir() or directory.is_symlink():
        return False
    try:
        entries = list(directory.iterdir())
    except OSError:
        return False
    return all(entry.is_file() and entry.name == ".gitkeep" for entry in entries)


def _install_gallery_link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)

    if link.is_symlink():
        link.unlink()
    elif link.exists():
        if not _placeholder_only(link):
            raise RuntimeError(
                f"Refusing to replace non-placeholder gallery directory: {link}. "
                f"Set {GALLERY_ENV} and create data/gallery as a symlink manually."
            )
        placeholder = link / ".gitkeep"
        placeholder.unlink(missing_ok=True)
        link.rmdir()

    link.symlink_to(target, target_is_directory=True)


def ensure_gallery_layout(
    repo_root: Path,
    *,
    gallery_rows: Sequence[dict[str, Any]] | None = None,
    manifest_path: Path | None = None,
    repair: bool = True,
) -> Path:
    """Return a fully validated gallery root and repair a stale local link.

    Validation is intentionally strict: success means every path listed by the
    canonical gallery manifest exists.  This prevents a long GPU job from
    failing on a later DataLoader batch because a mount points at the wrong
    dataset directory.
    """

    repo_root = Path(repo_root).resolve()
    manifest = manifest_path or (repo_root / "data" / "gallery.jsonl")
    rows = list(gallery_rows) if gallery_rows is not None else _read_jsonl(manifest)
    relative_paths = _relative_paths(rows)
    gallery_link = repo_root / "data" / "gallery"

    if _root_has(gallery_link, _probe_paths(relative_paths)):
        missing = _missing(gallery_link, relative_paths, limit=1)
        if not missing:
            return gallery_link.resolve()

    current_target: str | None = None
    if gallery_link.is_symlink():
        try:
            current_target = str(gallery_link.resolve(strict=False))
        except OSError:
            current_target = "<unresolvable symlink>"

    discovered = _discover_gallery_root(relative_paths)
    if discovered is None:
        first_relative = relative_paths[0]
        current = gallery_link / first_relative
        message = [
            "Canonical CPR gallery is unavailable.",
            f"Expected {len(relative_paths):,} images below: {gallery_link}",
            f"First required image: {current}",
        ]
        if current_target is not None:
            message.append(f"Current data/gallery symlink resolves to: {current_target}")
        message.extend(
            [
                "No complete gallery root was found under /kaggle/input.",
                f"Set {GALLERY_ENV} to the directory that directly contains the gallery images.",
            ]
        )
        raise FileNotFoundError("\n".join(message))

    if repair:
        _install_gallery_link(gallery_link, discovered)
        missing = _missing(gallery_link, relative_paths, limit=1)
        if missing:
            raise RuntimeError(
                "Gallery link repair completed but validation still failed: "
                f"{gallery_link / missing[0]}"
            )

    return discovered


def describe_gallery_link(repo_root: Path) -> str:
    link = Path(repo_root) / "data" / "gallery"
    if link.is_symlink():
        return f"{link} -> {link.resolve(strict=False)}"
    return str(link)
