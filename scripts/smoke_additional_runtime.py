#!/usr/bin/env python3
"""Check actual native forwards on synthetic images; never report benchmark metrics."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from cpr_baselines.backends import QwenGenerator, SentenceEncoder, build_backend
from cpr_baselines.captions import parse_caption, parse_structure, structure_text
from cpr_baselines.common import load_config, normalized, seed_runtime
from cpr_baselines.prepare import validate_prepared
from run_baseline import discover_methods, resolve_method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    spec = resolve_method(args.method, discover_methods())
    cfg = load_config(spec.config_path)
    cfg['runtime']['device'] = args.device
    validate_prepared(cfg)
    device = seed_runtime(cfg['runtime'])
    images = [Image.new('RGB', (224, 224), (160, 50, 30)),
              Image.new('RGB', (320, 224), (30, 90, 160))]
    if cfg['backend'] in {'caption_to_caption', 'structured_caption'}:
        generator = QwenGenerator(cfg, device)
        raw = generator.generate(cfg['prompts']['gallery'], images[0])
        parse = parse_structure if cfg['backend'] == 'structured_caption' else parse_caption
        result = parse(raw)
        del generator
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        encoder = SentenceEncoder(cfg, device)
        text = structure_text(result) if isinstance(result, dict) else result
        features = normalized(encoder.texts([text]))
        assert features.ndim == 2 and features.shape[0] == 1
    else:
        backend = build_backend(cfg, device)
        gallery = normalized(backend.images(images))
        queries = normalized(backend.queries(images, ['a person holding a cup', 'a person wearing a green shirt'],
                                              reference_features=gallery))
        assert gallery.shape == queries.shape and gallery.shape[0] == 2
        assert np.isfinite(queries @ gallery.T).all()
    print('Native forward smoke passed on synthetic inputs. No benchmark metric was computed.')


if __name__ == '__main__':
    main()
