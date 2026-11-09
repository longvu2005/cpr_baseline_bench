#!/usr/bin/env python3
"""FTI4CIR full benchmark inference."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cpr_baselines.runner import main

if __name__ == "__main__":
    main(Path(__file__).with_name("config.yaml"))
