"""Torch-free parity with an RCR checkout: data, text routing, selection, matching, metrics.

The RCR functions are read from the checkout itself. Modules that import torch only
for unrelated helpers are loaded function-by-function from their source.
These are protocol checks; they do not run any pretrained model.
"""

import ast
import hashlib
import importlib
import importlib.util
import os
import re
import sys
import types
from collections import Counter
from pathlib import Path
from statistics import fmean

import numpy as np
import pytest
from PIL import Image
from scipy.optimize import linear_sum_assignment

RCR = Path(os.environ.get("CPR_TEST_RCR_REPO", "/nonexistent"))
FINAL = Path(os.environ.get("CPR_TEST_RCR_FINAL", str(RCR / "dataset/data/final")))
pytestmark = pytest.mark.skipif(
    not (RCR / "src/rcr").is_dir(),
    reason="Set CPR_TEST_RCR_REPO for direct cross-repo parity",
)


@pytest.fixture(autouse=True)
def rcr_path(monkeypatch):
    monkeypatch.syspath_prepend(str(RCR / "src"))


def rcr_functions(relative, *names):
    """Execute selected top-level functions of an RCR module, without its imports."""
    tree = ast.parse((RCR / "src/rcr" / relative).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    assert sorted(n.name for n in nodes) == sorted(names)
    namespace = {"re": re, "np": np, "linear_sum_assignment": linear_sum_assignment}
    exec(compile(ast.Module(nodes, []), relative, "exec"), namespace)
    return [namespace[name] for name in names]


def rcr_module(relative):
    spec = importlib.util.spec_from_file_location(
        "rcr_parity_" + relative.replace("/", "_")[:-3], RCR / "src/rcr" / relative
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def imported(tmp_path_factory):
    if not (FINAL / "manifest.json").is_file():
        pytest.skip("RCR final export is not available")
    import prepare_data

    root = tmp_path_factory.mktemp("rcr-import")
    prepare_data.import_final(FINAL, root)
    return root


def test_import_matches_rcr_split_loader(imported):
    from benchmark_data import load_jsonl, split_paths
    from rcr.common.data import (
        load_rcr_data, required_identity_ids, sample_selection_texts,
        split_image_ids, split_samples,
    )

    data = load_rcr_data(FINAL)
    for split in ("val", "test"):
        gallery_path, query_path = split_paths(imported, split)
        gallery, queries = load_jsonl(gallery_path), load_jsonl(query_path)
        samples = split_samples(data, split)
        assert [g["image_id"] for g in gallery] == split_image_ids(data, split)
        assert [q["query_id"] for q in queries] == [s["sample_id"] for s in samples]
        for query, sample in zip(queries, samples, strict=True):
            assert query["image_id"] == sample["query_image_id"]
            assert query["case"] == sample["case_type"]
            assert query["text"] == sample["final_instruction"]
            assert query["full_positive_ids"] == sample["positive_image_ids"]
            assert query["target_ids"] == required_identity_ids(sample)
            assert [s["select_text"] for s in query["subjects"]] == sample_selection_texts(sample)
        heads = data.gt_head_boxes_by_image
        for row in gallery:
            assert row["person_ids"] == sorted({h["identity_id"] for h in heads.get(row["image_id"], [])})


def test_modification_routing_matches_rcr_fafa_on_every_query(imported):
    from benchmark_data import load_jsonl, split_paths

    (subject_changes,) = rcr_functions("baselines/fafa_adapter.py", "subject_changes")
    samples = {s["sample_id"]: s for s in load_jsonl(FINAL / "samples.jsonl")}
    for split in ("val", "test"):
        for query in load_jsonl(split_paths(imported, split)[1]):
            assert [s["modify_text"] for s in query["subjects"]] == subject_changes(
                samples[query["query_id"]]
            )


def test_evaluator_matches_rcr_on_complete_splits_with_ties(imported, monkeypatch):
    from benchmark_data import load_jsonl, split_paths
    from evaluate import evaluate_scores, rank_scores

    try:
        import torch  # noqa: F401
    except ImportError:  # evaluate_rankings itself never touches torch.
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.delitem(sys.modules, "rcr.evaluation.evaluate", raising=False)
    evaluate_rankings = importlib.import_module("rcr.evaluation.evaluate").evaluate_rankings
    samples = {s["sample_id"]: s for s in load_jsonl(FINAL / "samples.jsonl")}
    keys = {"ID-mAP": "id_map", "Full-mAP": "full_map",
            **{f"{a}-R@{k}": f"{b}_r{k}" for a, b in (("ID", "id"), ("Full", "full")) for k in (1, 5, 10)}}
    for split in ("val", "test"):
        gallery_path, query_path = split_paths(imported, split)
        gallery, queries = load_jsonl(gallery_path), load_jsonl(query_path)
        scores = np.random.default_rng(5).integers(0, 4, (len(queries), len(gallery))).astype(np.float32)
        ids = [g["image_id"] for g in gallery]
        index = {key: i for i, key in enumerate(ids)}
        rankings = {q["query_id"]: [ids[i] for i in rank_scores(row, index[q["image_id"]])]
                    for q, row in zip(queries, scores, strict=True)}
        native = evaluate_rankings(
            [{**samples[q["query_id"]], "positive_image_ids": q["full_positive_ids"]} for q in queries],
            ids, {g["image_id"]: set(g["person_ids"]) for g in gallery}, rankings, candidate_ks=(),
        )
        ours = evaluate_scores(scores, gallery, queries)
        for key, native_key in keys.items():
            assert ours["overall"][key] == pytest.approx(native["overall"][native_key], abs=1e-12)
        assert set(ours["cases"]) == set(native["by_case"])
        for case, row in ours["cases"].items():
            assert row["num_queries"] == native["by_case"][case]["num_queries"]
            for key in ("ID-mAP", "ID-R@1", "Full-mAP", "Full-R@1"):
                assert row[key] == pytest.approx(native["by_case"][case][keys[key]], abs=1e-12)
        assert ours["overall"]["Macro-Full-mAP"] == pytest.approx(
            fmean(row["full_map"] for row in native["by_case"].values()), abs=1e-12)


class SeededSelector:
    """Stand-in for CLIP features: a deterministic function of crop pixels and text."""

    def __init__(self, spread):
        self.spread = spread

    @staticmethod
    def _seed(payload):
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")

    def images(self, crops):
        rows = []
        for crop in crops:
            rng = np.random.default_rng(self._seed(np.asarray(crop).tobytes() + str(crop.size).encode()))
            rows.append(np.ones(8) + self.spread * rng.normal(size=8))
        rows = np.asarray(rows, np.float32)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    def texts(self, texts):
        rows = []
        for text in texts:
            rng = np.random.default_rng(self._seed(text.encode()))
            rows.append(np.ones(8) + self.spread * rng.normal(size=8))
        rows = np.asarray(rows, np.float32)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


@pytest.mark.parametrize("spread", [0.02, 0.3, 2.0])
def test_group_dual_and_shortage_membership_match_rcr_fafa(tmp_path, spread):
    """Hungarian anchors, cap 5, disjoint extras, fallbacks and padding vs RCR."""
    from cpr_person.detection import Detections, crop_person, protocol_config, read_rgb
    from cpr_person.localization import localize_queries, validate_localization

    select_rcr, choose_boxes = rcr_functions(
        "baselines/fafa_adapter.py", "select_subject_members", "choose_boxes")
    rng = np.random.default_rng(17)
    counts = [8, 0, 1, 3, 6]  # crowded GROUP, empty, single, small, mid
    gallery, offsets, boxes, flags, sizes = [], [0], [], [], []
    for i, count in enumerate(counts):
        width, height = 90 + 7 * i, 60 + 3 * i
        path = tmp_path / f"{i}.png"
        Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(path)
        gallery.append({"image_id": str(i), "path": str(path)})
        rows = [[float(rng.uniform(-3, width - 12)), float(rng.uniform(-2, height - 12)), 0.0, 0.0]
                for _ in range(count)]
        for row in rows:
            row[2], row[3] = row[0] + 10 + 1.5 * len(boxes), row[1] + 9 + 0.5 * len(boxes)
        if not rows:
            rows, fallback = [[0.0, 0.0, float(width), float(height)]], [True]
        else:
            fallback = [False] * len(rows)
        boxes += rows
        flags += fallback
        offsets.append(len(boxes))
        sizes.append((width, height))
    detections = Detections(
        np.asarray(offsets, np.int64), np.asarray(boxes, np.float32),
        np.where(flags, -1.0, 1.0).astype(np.float32), np.asarray(flags, bool),
        np.asarray(sizes, np.int64), {"protocol": {"backend": "rcr_detection_cache"}})
    detections.validate(len(gallery))
    queries = []
    for i in range(len(gallery)):
        for subjects in (1, 2):
            queries.append({"query_id": f"q{i}-{subjects}", "image_id": str(i), "subjects": [
                {"subject_id": s + 1, "select_text": f"person {s} in image {i} wearing {spread}"}
                for s in range(subjects)]})
    cfg = protocol_config()["selector"]
    assert int(cfg["max_members_per_subject"]) == 5
    selector = SeededSelector(spread)
    members, stats = localize_queries(queries, gallery, detections, cfg, selector)
    validate_localization(members, queries, gallery, detections, 5)
    assert stats["uses_case_labels_or_identity_counts"] is False

    candidates = [[{"box": list(map(float, detections.boxes[j])), "score": float(detections.confidences[j]),
                    "fallback": bool(detections.fallback[j])}
                   for j in range(offsets[g], offsets[g + 1])] for g in range(len(gallery))]
    for query, ours in zip(queries, members, strict=True):
        g = int(query["image_id"])
        texts = [s["select_text"] for s in query["subjects"]]
        chosen = choose_boxes(candidates[g], 0.3, len(texts))
        while len(chosen) < len(texts):
            chosen.append({"box": [0, 0, *sizes[g]], "score": -1.0, "fallback": True})
        image = read_rgb(gallery[g]["path"])
        sim = selector.texts(texts) @ selector.images([crop_person(image, b["box"]) for b in chosen]).T
        expected = select_rcr(
            sim, threshold=float(cfg["membership_threshold"]), margin=float(cfg["membership_margin"]),
            max_members=5, valid_extra=np.array([not b["fallback"] for b in chosen]))
        assert [[list(map(float, m["box"])) for m in subject] for subject in ours] == [
            [list(map(float, chosen[j]["box"])) for j in subject] for subject in expected]
        assert [[m["fallback"] for m in subject] for subject in ours] == [
            [chosen[j]["fallback"] for j in subject] for subject in expected]
    if spread == 0.02:  # Near-identical crops: the crowded image saturates the cap.
        histogram = Counter(len(subject) for query in members for subject in query)
        assert histogram[5] >= 1 and max(histogram) == 5


def test_one_to_many_partial_setmatch_matches_rcr():
    from cpr_person.matching import aggregate_people, score_feature_sets

    native = rcr_module("baselines/fafa_matching.py")
    rng = np.random.default_rng(29)
    for k in range(1, 11):  # Up to two Subjects with five predicted members each.
        counts = rng.integers(0, 9, size=40)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        for b in (-1.0, -3.5e3):
            scores = rng.normal(0, 1, (k, int(offsets[-1]))).astype(np.float32)
            scores[:, ::7] = scores[:, :1]  # exact ties inside an image
            np.testing.assert_array_equal(
                aggregate_people(scores, offsets, b), native.aggregate_people(scores, offsets, b))
        features = rng.normal(size=(int(offsets[-1]), 6)).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        query = features[: max(1, min(k, len(features)))]
        expected = native.aggregate_people(query @ features.T, offsets, -1.0)
        np.testing.assert_allclose(score_feature_sets(query, features, offsets, -1.0, 5), expected, atol=1e-6)
