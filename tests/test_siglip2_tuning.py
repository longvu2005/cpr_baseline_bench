"""SigLIP2 val-only alpha selection and frozen composition contracts."""
import json

import numpy as np
import pytest
import yaml

from benchmark_data import manifest_fingerprint
from cpr_baselines import siglip2_tuning
from cpr_baselines.backends import SigLIPCompose
from cpr_baselines.common import compose_linear
from cpr_person.cache import StageCache


def test_siglip2_query_composition_uses_selected_alpha():
    encoder = object.__new__(SigLIPCompose)
    encoder.cfg = {"composition": {"alpha": 0.25}}
    encoder.texts = lambda texts: np.array([[0., 1.]], np.float32)
    reference = np.array([[1., 0.]], np.float32)
    actual = encoder.queries([], ["modify this subject"], reference_features=reference)
    np.testing.assert_allclose(actual, compose_linear(reference, np.array([[0., 1.]]), 0.25))


def test_siglip2_tunes_on_val_only_and_reuses_verified_receipt(tmp_path, monkeypatch):
    val = tmp_path / "data" / "val"
    val.mkdir(parents=True)
    gallery = [
        {"image_id": "reference", "path": "ref.png"},
        {"image_id": "negative", "path": "negative.png"},
        {"image_id": "positive", "path": "positive.png"},
    ]
    queries = [{"query_id": "q", "image_id": "reference", "text": "target condition",
                "full_positive_ids": ["positive"], "target_ids": ["ID"],
                "case": "INDIVIDUAL"}]
    for name, rows in (("gallery", gallery), ("queries", queries)):
        (val / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(siglip2_tuning, "ROOT", tmp_path)
    monkeypatch.setattr(siglip2_tuning, "verify_manifest",
                        lambda root, split: {"fingerprint": manifest_fingerprint(root, split)})
    monkeypatch.setattr(siglip2_tuning, "image_digest", lambda rows: "verified-image-content")
    monkeypatch.setattr(siglip2_tuning, "source_images",
                        lambda rows, start, end: [r["image_id"] for r in rows[start:end]])
    monkeypatch.setattr(siglip2_tuning, "feature_cache",
                        lambda stage, fp, runtime: StageCache(stage, fp, write_root=tmp_path / "cache"))

    class Encoder:
        calls = 0
        def images(self, images):
            self.calls += 1
            vectors = {"reference": [1., 0.], "negative": [1., 0.],
                       "positive": [0., 1.]}
            return np.array([vectors[name] for name in images], np.float32)
        def texts(self, texts):
            self.calls += 1
            assert texts == ["target condition"]
            return np.array([[0., 1.]], np.float32)
    encoder = Encoder()
    cfg = {"method": "siglip2_compose", "cpr_supervision": "Val only",
           "composition": {"alpha": 0.5, "alpha_selection": "validation_full_map",
                           "text_field": "text", "max_text_tokens": 64,
                           "formula": "normalize(alpha*normalize(image)+(1-alpha)*normalize(text))"},
           "tuning": {"image_weights": [i / 20 for i in range(21)]},
           "runtime": {"image_batch_size": 2, "text_batch_size": 1,
                       "score_query_batch_size": 1}}
    output = tmp_path / "runs" / "siglip2_compose"
    output.mkdir(parents=True)
    first = siglip2_tuning.tune_siglip2(cfg, {"pinned": True}, "cpu", output,
                                      backend_factory=lambda *_: encoder)
    assert 0 <= first["image_weight"] < 0.5
    assert first["validation_full_map"] == pytest.approx(1.0)
    assert first["fingerprint"]["validation_fingerprint"] == manifest_fingerprint(tmp_path, "val")
    assert first["split"] == "val" and len(first["trials"]) == 21
    calls = encoder.calls
    second = siglip2_tuning.tune_siglip2(cfg, {"pinned": True}, "cpu", output,
                                       backend_factory=lambda *_: pytest.fail("cache not reused"))
    assert second == first and encoder.calls == calls
    with pytest.raises(ValueError, match="alpha grid"):
        siglip2_tuning.tune_siglip2({**cfg, "tuning": {"image_weights": [0.5]}},
                                   {"pinned": True}, "cpu", output)
