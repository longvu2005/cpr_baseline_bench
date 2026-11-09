import json

import numpy as np
import pytest

from cpr_person import detection
from cpr_person.cache import canonical_hash, sha256_file
from cpr_person.rcr_cache import read_rcr_index


def test_project_completed_rcr_index_without_running_detector(
    toy_protocol, rcr_detection_index, tmp_path, monkeypatch,
):
    cfg, gallery, _, gm, _, calls, _ = toy_protocol
    ids = ["train-image", "a", "b", "other-test-image"]
    boxes = [[0, 0, 10, 20], [20.5, 1.75, 41.25, 30.0], [-0.5, 2.25, 19.75, 31.25], [1, 1, 2, 2]]
    index = rcr_detection_index(ids, boxes, [0, 1, 3, 3, 4])

    canonical = tmp_path / "data" / "test" / "gallery.jsonl"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(gm.read_bytes())
    metadata = canonical.parents[1] / "manifest.json"
    metadata.write_text(json.dumps({"rcr_gallery_ids_sha256": canonical_hash(ids)}))
    monkeypatch.setattr(detection, "ROOT", tmp_path)

    def unexpected_detector(*args):
        pytest.fail("An existing RCR index must bypass the detector worker")

    result = detection.get_detections(gallery, canonical, "cpu", runner=unexpected_detector, cfg=cfg)
    np.testing.assert_array_equal(result.boxes, np.array([boxes[1], boxes[2], [0, 0, 40, 30]], np.float32))
    assert result.offsets.tolist() == [0, 2, 3]
    assert result.fallback.tolist() == [False, False, True]
    assert result.confidences.tolist() == [1.0, 1.0, -1.0]
    assert result.fingerprint["rcr_cache"]["index_sha256"] == sha256_file(index)
    assert detection.get_detections(gallery, canonical, "cpu", require_cache=True, cfg=cfg).cache_hit
    assert not calls

    rcr_detection_index(["train-image", "b", "a", "other-test-image"], boxes, [0, 1, 3, 3, 4])
    with pytest.raises(ValueError, match="gallery IDs/order"):
        detection.get_detections(gallery, canonical, "cpu", cfg=cfg)


def test_reject_changed_rcr_index(rcr_detection_index):
    import torch

    path = rcr_detection_index(["a"], [[1, 1, 20, 20]], [0, 1])
    index = torch.load(path, weights_only=True)
    index["boxes_pixel"][0, 0] = 2.0
    torch.save(index, path)
    with pytest.raises(ValueError, match="cache_id"):
        read_rcr_index(path)
