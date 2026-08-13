#!/usr/bin/env python3
"""Small, dependency-free progress helpers shared by benchmark scripts.

The helper deliberately owns only phase-level logging. Fine-grained batch
progress remains in each method's tqdm loops so model code keeps full control
of iteration and no benchmark result depends on this module.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterator


def format_duration(seconds: float) -> str:
    """Format elapsed wall time compactly for terminal progress messages."""
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def progress_bar(iterable, *, desc: str, total: int | None = None, unit: str = "it"):
    """Create a consistent tqdm bar without making tqdm a runner dependency."""
    from tqdm.auto import tqdm

    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        dynamic_ncols=True,
        mininterval=0.5,
        smoothing=0.1,
        leave=True,
    )

def byte_progress(*, desc: str, total: int | None):
    """Create a byte-scaled tqdm progress object for streamed downloads."""
    from tqdm.auto import tqdm

    return tqdm(
        total=total,
        desc=desc,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        mininterval=0.5,
        smoothing=0.1,
        leave=True,
    )


@dataclass
class PhaseTracker:
    """Print deterministic phase start/end messages with elapsed time."""

    name: str
    total: int
    current: int = 0
    _active_title: str | None = field(default=None, init=False, repr=False)
    _active_started: float | None = field(default=None, init=False, repr=False)
    _active_prefix: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError("PhaseTracker.total must be > 0")

    def log(self, message: str) -> None:
        print(f"[{self.name}] {message}", flush=True)

    def _close_active(self) -> None:
        if self._active_title is None or self._active_started is None or self._active_prefix is None:
            return
        elapsed = format_duration(perf_counter() - self._active_started)
        print(f"{self._active_prefix} done in {elapsed}: {self._active_title}", flush=True)
        self._active_title = None
        self._active_started = None
        self._active_prefix = None

    def advance(self, title: str, detail: str | None = None) -> None:
        """Start the next sequential phase without indenting the caller's code."""
        self._close_active()
        if self.current >= self.total:
            raise RuntimeError(
                f"PhaseTracker for {self.name!r} exceeded declared total={self.total}"
            )
        self.current += 1
        prefix = f"[{self.name}] [{self.current}/{self.total}]"
        suffix = f" — {detail}" if detail else ""
        print(f"\n{prefix} {title}{suffix}", flush=True)
        self._active_title = title
        self._active_started = perf_counter()
        self._active_prefix = prefix

    @contextmanager
    def phase(self, title: str, detail: str | None = None) -> Iterator[None]:
        if self.current >= self.total:
            raise RuntimeError(
                f"PhaseTracker for {self.name!r} exceeded declared total={self.total}"
            )
        self.current += 1
        prefix = f"[{self.name}] [{self.current}/{self.total}]"
        suffix = f" — {detail}" if detail else ""
        print(f"\n{prefix} {title}{suffix}", flush=True)
        started = perf_counter()
        try:
            yield
        except Exception:
            elapsed = format_duration(perf_counter() - started)
            print(f"{prefix} FAILED after {elapsed}: {title}", flush=True)
            raise
        else:
            elapsed = format_duration(perf_counter() - started)
            print(f"{prefix} done in {elapsed}: {title}", flush=True)

    def finish(self) -> None:
        self._close_active()
        if self.current != self.total:
            raise RuntimeError(
                f"PhaseTracker for {self.name!r} finished at {self.current}/{self.total}"
            )
        print(f"\n[{self.name}] Completed all {self.total} phases.", flush=True)
