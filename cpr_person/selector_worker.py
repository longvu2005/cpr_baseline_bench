"""Shared CLIP localization worker, isolated from native tokenizer imports."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .cache import StageCache, read_json, resolve
from .detection import Detections, protocol_config
from .localization import (ClipSelector, localize_queries, localization_fingerprint,
                           validate_localization)
from .tokenizer_runtime import validate_tokenizer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["gallery-manifest","query-manifest","detections","cache-dir","device"]:
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    cfg=protocol_config()
    root=validate_tokenizer(cfg["selector"])
    sys.path.insert(0,str(root))
    import ftfy
    import wcwidth
    for name,module in [("ftfy",ftfy),("wcwidth",wcwidth)]:
        if (not resolve(module.__file__).is_relative_to(root)
                or module.__version__!=cfg["selector"]["tokenizer_packages"][name]):
            raise RuntimeError("CLIP selector imported a native model's tokenizer dependency")
    gallery=[json.loads(line) for line in resolve(args.gallery_manifest).read_text().splitlines() if line.strip()]
    queries=[json.loads(line) for line in resolve(args.query_manifest).read_text().splitlines() if line.strip()]
    detection_path=resolve(args.detections)
    meta=read_json(detection_path.with_name(detection_path.name+".meta.json"))
    with np.load(detection_path,allow_pickle=False) as arrays:
        detections=Detections(**{name:np.array(arrays[name]) for name in
            ("offsets","boxes","confidences","fallback","image_sizes")},fingerprint=meta["fingerprint"],path=detection_path)
    detections.validate(len(gallery))
    fp=localization_fingerprint(args.query_manifest,detections,cfg)
    directory=resolve(args.cache_dir)
    cache=StageCache("query_localization",fp,write_root=directory.parents[1])
    if cache.directory!=directory:
        raise ValueError("Selector output directory differs from its immutable inputs")
    selector=ClipSelector(cfg["selector"],args.device)
    members,stats=localize_queries(queries,gallery,detections,cfg["selector"],selector)
    validate_localization(members,queries,gallery,detections)
    cache.write_json("members.json",{"members":members,"stats":stats})


if __name__=="__main__":
    main()
