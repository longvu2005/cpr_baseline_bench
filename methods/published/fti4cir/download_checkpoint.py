#!/usr/bin/env python3
"""Prepare actual FTI4CIR model assets."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from cpr_baselines.prepare import main

if __name__ == "__main__":
    main(Path(__file__).with_name("config.yaml"))
