"""Global CLIP baseline over the shared frozen RCR scene pipeline."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cpr_baselines.global_clip import main
if __name__ == '__main__':
    main(Path(__file__).with_name('config.yaml'))
