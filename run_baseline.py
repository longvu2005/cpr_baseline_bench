#!/usr/bin/env python3
"""Run one benchmark method end to end from a single method name.

Pipeline:
    method-local checkpoint download -> inference -> evaluation -> tables

Examples:
    python run_baseline.py clip_image
    python run_baseline.py 01_clip_image
    python run_baseline.py fafa_setmatch --force-checkpoint
    python run_baseline.py --list
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METHOD_ROOTS = (
    ROOT / "methods" / "simple",
    ROOT / "methods" / "published",
)
NUMERIC_PREFIX_RE = re.compile(r"^\d+_(.+)$")


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    directory: Path

    @property
    def relative_directory(self) -> Path:
        return self.directory.relative_to(ROOT)

    @property
    def aliases(self) -> tuple[str, ...]:
        names = {self.method_id, self.directory.name}
        match = NUMERIC_PREFIX_RE.match(self.directory.name)
        if match:
            names.add(match.group(1))
        return tuple(sorted(names))


def read_method_id(config_path: Path) -> str:
    """Read the top-level `method:` value without adding a YAML dependency."""
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line[0].isspace():
            continue
        if not raw_line.startswith("method:"):
            continue

        value = raw_line.split(":", 1)[1].split("#", 1)[0].strip()
        value = value.strip("'\"")
        if not value:
            break
        return value

    raise ValueError(f"Missing top-level `method:` in {config_path.relative_to(ROOT)}")


def discover_methods() -> list[MethodSpec]:
    methods: list[MethodSpec] = []
    seen_ids: dict[str, Path] = {}

    for method_root in METHOD_ROOTS:
        if not method_root.is_dir():
            continue

        for directory in sorted(path for path in method_root.iterdir() if path.is_dir()):
            config_path = directory / "config.yaml"
            run_path = directory / "run.py"
            if not config_path.is_file() or not run_path.is_file():
                continue

            method_id = read_method_id(config_path)
            previous = seen_ids.get(method_id)
            if previous is not None:
                raise RuntimeError(
                    f"Duplicate method id {method_id!r}: "
                    f"{previous.relative_to(ROOT)} and {directory.relative_to(ROOT)}"
                )

            seen_ids[method_id] = directory
            methods.append(MethodSpec(method_id=method_id, directory=directory))

    return methods


def resolve_method(name: str, methods: list[MethodSpec]) -> MethodSpec:
    exact_matches = [method for method in methods if name in method.aliases]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        choices = ", ".join(method.method_id for method in exact_matches)
        raise RuntimeError(f"Ambiguous method name {name!r}: {choices}")

    available = ", ".join(method.method_id for method in methods) or "<none>"
    raise KeyError(f"Unknown method {name!r}. Available methods: {available}")


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_step(index: int, total: int, title: str, command: list[str]) -> None:
    print()
    print(f"[{index}/{total}] {title}")
    print(f"$ {format_command(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def run_pipeline(method: MethodSpec, force_checkpoint: bool) -> None:
    print(f"Method : {method.method_id}")
    print(f"Path   : {method.relative_directory}")

    downloader = method.directory / "download_checkpoint.py"
    if downloader.is_file():
        command = [sys.executable, str(downloader.relative_to(ROOT))]
        if force_checkpoint:
            command.append("--force")
        run_step(1, 4, "Checkpoint", command)
    else:
        print()
        print("[1/4] Checkpoint")
        print(
            "[skip] This method has no download_checkpoint.py; "
            "continuing with its existing checkpoint/source setup."
        )

    run_step(
        2,
        4,
        "Inference",
        [sys.executable, str((method.directory / "run.py").relative_to(ROOT))],
    )
    run_step(
        3,
        4,
        "Evaluation",
        [sys.executable, "evaluate.py", "--method", method.method_id],
    )
    run_step(
        4,
        4,
        "Build benchmark tables",
        [sys.executable, "build_tables.py"],
    )

    print()
    print("Done")
    print("----")
    print(f"runs/{method.method_id}/scores.npy")
    print(f"outputs/{method.method_id}/metrics.json")
    print("tables/table1_main.csv")
    print("tables/table2_cases.csv")


def print_methods(methods: list[MethodSpec]) -> None:
    if not methods:
        print("No runnable methods found.")
        return

    print("Available methods")
    print("-----------------")
    for method in methods:
        aliases = [alias for alias in method.aliases if alias != method.method_id]
        alias_text = f"  aliases: {', '.join(aliases)}" if aliases else ""
        print(f"{method.method_id:<24} {method.relative_directory}{alias_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a method checkpoint when supported, run inference, "
            "evaluate it, and rebuild the benchmark tables."
        )
    )
    parser.add_argument(
        "method",
        nargs="?",
        help="Method id or method directory name, e.g. clip_image or 01_clip_image.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered runnable methods and exit.",
    )
    parser.add_argument(
        "--force-checkpoint",
        action="store_true",
        help="Pass --force to the method-local checkpoint downloader.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = discover_methods()

    if args.list:
        print_methods(methods)
        return

    if not args.method:
        raise SystemExit("Missing method name. Use `python run_baseline.py --list` to see options.")

    try:
        method = resolve_method(args.method, methods)
    except (KeyError, RuntimeError) as error:
        raise SystemExit(str(error)) from error

    run_pipeline(method, args.force_checkpoint)


if __name__ == "__main__":
    main()
