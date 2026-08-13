#!/usr/bin/env python3
"""Run one benchmark baseline end to end from a single method name.

Default pipeline:
    install method requirements -> checkpoint preparation -> inference
    -> official evaluation -> table rebuilding

Examples:
    python run_baseline.py clip_image
    python run_baseline.py 01_clip_image
    python run_baseline.py fafa_setmatch --force-checkpoint
    python run_baseline.py clip_image --skip-install
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
    def config_path(self) -> Path:
        return self.directory / "config.yaml"

    @property
    def requirements_path(self) -> Path:
        return self.directory / "requirements.txt"

    @property
    def run_path(self) -> Path:
        return self.directory / "run.py"

    @property
    def checkpoint_path(self) -> Path:
        return self.directory / "download_checkpoint.py"

    @property
    def aliases(self) -> tuple[str, ...]:
        names = {self.method_id, self.directory.name}
        match = NUMERIC_PREFIX_RE.match(self.directory.name)
        if match:
            names.add(match.group(1))
        return tuple(sorted(names))


def read_method_id(config_path: Path) -> str:
    """Read the top-level `method:` value without requiring PyYAML."""
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
    incomplete: list[str] = []

    for method_root in METHOD_ROOTS:
        if not method_root.is_dir():
            continue

        for directory in sorted(path for path in method_root.iterdir() if path.is_dir()):
            config_path = directory / "config.yaml"
            if not config_path.is_file():
                continue

            missing = [
                filename
                for filename in ("requirements.txt", "run.py")
                if not (directory / filename).is_file()
            ]
            if missing:
                incomplete.append(
                    f"{directory.relative_to(ROOT)}: missing {', '.join(missing)}"
                )
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

    if incomplete:
        details = "\n".join(f"  - {item}" for item in incomplete)
        raise RuntimeError(
            "Incomplete benchmark method integration(s) found:\n"
            f"{details}\n"
            "Every configured method must provide requirements.txt and run.py."
        )

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


def print_skip(index: int, total: int, title: str, message: str) -> None:
    print()
    print(f"[{index}/{total}] {title}")
    print(f"[skip] {message}")


def run_pipeline(
    method: MethodSpec,
    force_checkpoint: bool,
    skip_install: bool,
) -> None:
    total = 5
    print(f"Method : {method.method_id}")
    print(f"Path   : {method.relative_directory}")

    if skip_install:
        print_skip(
            1,
            total,
            "Install requirements",
            "requested by --skip-install; using the current environment as-is.",
        )
    else:
        run_step(
            1,
            total,
            "Install requirements",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(method.requirements_path.relative_to(ROOT)),
            ],
        )

    if method.checkpoint_path.is_file():
        command = [sys.executable, str(method.checkpoint_path.relative_to(ROOT))]
        if force_checkpoint:
            command.append("--force")
        run_step(2, total, "Prepare checkpoint", command)
    else:
        print_skip(
            2,
            total,
            "Prepare checkpoint",
            "this method has no download_checkpoint.py and declares no automated checkpoint preparation.",
        )

    run_step(
        3,
        total,
        "Inference",
        [sys.executable, str(method.run_path.relative_to(ROOT))],
    )
    run_step(
        4,
        total,
        "Official evaluation",
        [sys.executable, "evaluate.py", "--method", method.method_id],
    )
    run_step(
        5,
        total,
        "Build benchmark tables",
        [sys.executable, "build_tables.py"],
    )

    print()
    print("Done")
    print("----")
    print(f"runs/{method.method_id}/scores.npy")
    print(f"runs/{method.method_id}/run.json")
    print(f"outputs/{method.method_id}/metrics.json")
    print(f"outputs/{method.method_id}/run.json")
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
            "Install the selected method requirements, prepare its checkpoint when "
            "supported, run inference, evaluate it, and rebuild benchmark tables."
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
        "--skip-install",
        action="store_true",
        help=(
            "Skip the default `python -m pip install -r <method>/requirements.txt` "
            "step. Use only when the active environment is already prepared."
        ),
    )
    parser.add_argument(
        "--force-checkpoint",
        action="store_true",
        help="Pass --force to the method-local checkpoint preparer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        methods = discover_methods()
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    if args.list:
        print_methods(methods)
        return

    if not args.method:
        raise SystemExit("Missing method name. Use `python run_baseline.py --list` to see options.")

    try:
        method = resolve_method(args.method, methods)
    except (KeyError, RuntimeError) as error:
        raise SystemExit(str(error)) from error

    run_pipeline(method, args.force_checkpoint, args.skip_install)


if __name__ == "__main__":
    main()
