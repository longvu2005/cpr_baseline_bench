"""Dataset and score regressions for the RCR test protocol."""

import json

import numpy as np
import pytest

from benchmark_data import CASE_TYPES, ensure_gallery_layout, load_jsonl, split_paths, verify_manifest
from cpr_baselines.clip_fusion import late_zscore, select_image_weight
from cpr_person.validation import verify_tuning_manifests
from evaluate import evaluate_scores, rank_scores
from prepare_data import import_final, write_jsonl


def test_rcr_final_import_preserves_split_order_and_filters_positives(tmp_path):
    final = tmp_path / "final"
    (final / "splits").mkdir(parents=True)
    images = [
        {"image_id": i, "path": f"{split}/{i}.jpg"}
        for i, split in (("v2", "val"), ("t2", "test"), ("train", "train"), ("v1", "val"),
                         ("left", "leftover"), ("t1", "test"))
    ]
    write_jsonl(final / "images.jsonl", images)
    write_jsonl(final / "gallery.jsonl", [{"image_id": row["image_id"]} for row in images])
    write_jsonl(final / "head_boxes.jsonl", [
        {"image_id": row["image_id"], "identity_id": identity}
        for row in images for identity in ("a", "b")
    ])
    desc = "Identify Subject 1 as the people in blue"
    change = "then retrieve target images where the members of Subject 1 are wearing red"
    val = {
        "sample_id": "v", "case_type": "GROUP", "query_image_id": "v1",
        "target_image_id": "v2", "positive_image_ids": ["left", "t2", "v2"],
        "subjects": [{"subject_id": 1, "identity_ids": ["a", "b"]}],
        "final_desc": desc, "final_change": change, "final_instruction": f"{desc}; {change}.",
    }
    extra_val = {**val, "sample_id": "v-first"}
    test = {**val, "sample_id": "t", "query_image_id": "t1", "target_image_id": "t2",
            "positive_image_ids": ["t2", "v2"]}
    write_jsonl(final / "samples.jsonl", [{"sample_id": "train"}, val, extra_val, test])
    (final / "splits" / "val.txt").write_text("v-first\nv\n")
    (final / "splits" / "test.txt").write_text("t\n")
    (final / "manifest.json").write_text(json.dumps({
        "partial": False, "version": "example", "splits": {"train": 1, "val": 2, "test": 1}
    }))
    root = tmp_path / "baseline"
    import_final(final, root)

    val_gallery_path, val_query_path = split_paths(root, "val")
    test_gallery_path, test_query_path = split_paths(root)
    assert [row["image_id"] for row in load_jsonl(val_gallery_path)] == ["v2", "v1"]
    assert [row["image_id"] for row in load_jsonl(test_gallery_path)] == ["t2", "t1"]
    val_queries = load_jsonl(val_query_path)
    assert [row["query_id"] for row in val_queries] == ["v-first", "v"]
    assert val_queries[0]["full_positive_ids"] == ["v2"]
    assert val_queries[0]["subjects"][0]["identity_ids"] == ["a", "b"]
    assert len(val_queries[0]["subjects"]) == 1
    assert not (root / "data" / "train").exists()
    image_root = tmp_path / "images"
    for split in ("val", "test"):
        for row in load_jsonl(split_paths(root, split)[0]):
            path = image_root / split / f"{row['image_id']}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    assert ensure_gallery_layout(root, source_root=image_root) == image_root
    assert (root / val_queries[0]["path"]).is_file()
    assert verify_tuning_manifests(
        root, test_gallery_path, test_query_path, val_gallery_path, val_query_path
    ) == verify_manifest(root, "val")["fingerprint"]
    with pytest.raises(ValueError, match="canonical RCR val"):
        verify_tuning_manifests(root, test_gallery_path, test_query_path, test_gallery_path, test_query_path)
    val_query_path.write_text(val_query_path.read_text() + "\n")
    with pytest.raises(ValueError, match="val manifest files differ"):
        verify_manifest(root, "val")


def test_macro_weights_present_cases_equally_and_ties_use_gallery_order():
    gallery = [{"image_id": i, "person_ids": ["a", "b"]} for i in ("ref", "first", "second")]
    queries = [
        {"query_id": str(i), "image_id": "ref", "case": case, "target_ids": ["a"],
         "full_positive_ids": [positive]}
        for i, (case, positive) in enumerate((
            ("INDIVIDUAL", "first"), ("INDIVIDUAL", "first"), ("GROUP", "second")
        ))
    ]
    scores = np.array([[100.0, 0.0, 0.0]] * 3)
    metrics = evaluate_scores(scores, gallery, queries)
    assert CASE_TYPES == ("INDIVIDUAL", "GROUP", "DUAL", "RELATIONAL")
    assert tuple(metrics["cases"]) == ("INDIVIDUAL", "GROUP")
    assert metrics["cases"]["GROUP"]["num_queries"] == 1
    assert metrics["overall"]["Full-mAP"] == pytest.approx(5 / 6)
    assert metrics["overall"]["Macro-Full-mAP"] == pytest.approx(0.75)
    assert metrics["overall"]["ID-mAP"] == 1.0
    assert rank_scores(scores[0], 0).tolist() == [1, 2]


def test_clip_fusion_selection_uses_eligible_scores():
    pytest.importorskip("torch")
    scores = np.array([[10.0, 1.0, 3.0]])
    np.testing.assert_allclose(late_zscore(scores, [0], 1e-6), [[0.0, -1.0, 1.0]])
    scores[0, 0] = -500.0
    np.testing.assert_allclose(late_zscore(scores, [0], 1e-6), [[0.0, -1.0, 1.0]])

    gallery = [{"image_id": i} for i in ("ref", "bad", "positive")]
    queries = [{"query_id": "val", "image_id": "ref", "full_positive_ids": ["positive"]}]
    alpha, best, _ = select_image_weight(
        scores, np.array([[10.0, 2.0, 0.0]]), gallery, queries, [0.0, 0.5, 1.0]
    )
    assert alpha == 1.0
    assert best == 1.0


def test_s11_reorders_only_the_eligible_top_k(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import sys
    from types import SimpleNamespace

    # The rerank writer is pure NumPy. Only the model-facing verifier needs torch.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=lambda: lambda fn: fn))
    path = Path(__file__).resolve().parents[1] / "methods/simple/11_qwen_vl_rerank/run.py"
    spec = importlib.util.spec_from_file_location("test_s11_rcr", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first_scores = np.array([[0.7, 1.0, 0.6, 0.4]], dtype=np.float32)
    scores = module.write_final(
        first_scores, np.array([[0, 2]]), np.array([[0.0, 1.0]], dtype=np.float32),
        [1], 2, 0.0, tmp_path / "scores.npy",
    )
    assert np.isfinite(scores).all()
    assert rank_scores(scores[0], 1).tolist() == [2, 0, 3]
    with pytest.raises(RuntimeError, match="eligible first-stage ranking"):
        module.write_final(
            first_scores, np.array([[1, 0]]), np.array([[0.0, 1.0]], dtype=np.float32),
            [1], 2, 0.0, tmp_path / "scores.npy",
        )
    assert rank_scores(np.load(tmp_path / "scores.npy")[0], 1).tolist() == [2, 0, 3]
