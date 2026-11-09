"""Original RCR indexes (no image digests) are reusable, and drift is still caught."""

import numpy as np
import pytest
from PIL import Image

from cpr_baselines.scene_cache import spot_check_globals, spot_check_rows
from cpr_person.cache import sha256_file
from cpr_person.detection import bind_rcr_images
from cpr_person.protocol import IMAGE_BINDINGS, rcr_detection_source
from cpr_person import PROTOCOL_VERSION


def images(tmp_path, count=3):
    rows = []
    for i in range(count):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (20 + i, 10), (40 * i, 0, 0)).save(path)
        rows.append({"image_id": str(i), "path": str(path)})
    return rows


def test_rcr_detection_index_without_digests_binds_by_ids(tmp_path):
    rows = images(tmp_path)
    assert bind_rcr_images(["2", "0", "1", "x"], None, rows) == "rcr_ids_and_image_sizes"
    with pytest.raises(ValueError, match="absent"):
        bind_rcr_images(["0", "1"], None, rows)


def test_build_time_digests_are_still_verified(tmp_path):
    rows = images(tmp_path)
    hashes = [sha256_file(row["path"]) for row in rows]
    assert bind_rcr_images(["0", "1", "2"], hashes, rows) == "build_time_sha256"
    Image.new("RGB", (21, 10), "white").save(rows[1]["path"])
    with pytest.raises(ValueError, match="different image bytes"):
        bind_rcr_images(["0", "1", "2"], hashes, rows)
    with pytest.raises(ValueError, match="Invalid"):
        bind_rcr_images(["0", "1", "2"], hashes[:2], rows)


def test_run_alignment_accepts_only_declared_bindings():
    def provenance(**source):
        base = {"layout": "person-detections-v1", "index_sha256": "a" * 64,
                "cache_id": "b" * 64, "gallery_ids_sha256": "c" * 64}
        return {"version": PROTOCOL_VERSION, "detections_fingerprint": {
            "protocol": {"backend": "rcr_detection_cache"}, "rcr_cache": {**base, **source}}}

    for binding in IMAGE_BINDINGS:
        assert rcr_detection_source(provenance(image_binding=binding)) is not None
    assert rcr_detection_source(provenance(image_content_verified=True)) is None
    assert rcr_detection_source(provenance(image_binding="unchecked")) is None


def test_scene_spot_check_sample_is_label_free_and_deterministic():
    gallery = [{"image_id": f"img{i}", "path": "unused", "person_ids": ["x"]} for i in range(40)]
    picked = spot_check_rows(gallery, 16)
    assert picked == sorted(picked) and len(set(picked)) == 16
    relabelled = [{**row, "person_ids": ["other"], "path": "elsewhere"} for row in gallery]
    assert spot_check_rows(relabelled, 16) == picked
    assert spot_check_rows(gallery[:5], 16) == [0, 1, 2, 3, 4]


def test_scene_spot_check_rejects_different_images_or_preprocessing():
    rng = np.random.default_rng(3)
    cached = rng.normal(size=(16, 512)).astype(np.float32)
    # Precision-level differences (e.g. TF32 on another GPU) pass.
    noisy = cached + 1e-3 * rng.normal(size=cached.shape).astype(np.float32)
    result = spot_check_globals(cached, noisy)
    assert result["num_images"] == 16 and result["min_cosine"] > 0.999
    # A different crop/letterbox or image gives a clearly different embedding.
    shifted = cached.copy()
    shifted[3] = 0.6 * cached[3] + 0.8 * rng.normal(size=512)
    with pytest.raises(ValueError, match="differ from the shared HF letterbox"):
        spot_check_globals(cached, shifted)
    with pytest.raises(ValueError):
        spot_check_globals(cached, cached[:3])
