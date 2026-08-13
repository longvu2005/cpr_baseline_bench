#!/usr/bin/env python3
"""Prepare the pinned Qwen2.5-VL verifier for S11 by reusing S7's preparer."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from benchmark_progress import PhaseTracker  # noqa: E402
S7=ROOT/'methods/simple/07_qwen25vl_rewrite_clip/download_checkpoint.py'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--force',action='store_true'); args=p.parse_args()
    if not S7.is_file():
        raise FileNotFoundError('S11 shares the pinned Qwen snapshot with S7; apply/install S7 first.')
    tracker=PhaseTracker('s11_prepare',total=1); tracker.advance('Prepare shared pinned Qwen2.5-VL snapshot')
    cmd=[sys.executable,'-u',str(S7.relative_to(ROOT))]
    if args.force: cmd.append('--force')
    subprocess.run(cmd,cwd=ROOT,check=True)
    tracker.finish()
if __name__=='__main__': main()
