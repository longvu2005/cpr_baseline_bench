"""Direct comparisons against an explicitly supplied RCR checkout."""

from collections import Counter
import os
from pathlib import Path
from statistics import fmean

import numpy as np
from PIL import Image
import pytest

RCR = Path(os.environ.get("CPR_TEST_RCR_REPO", "/nonexistent"))
pytestmark = pytest.mark.skipif(
    not (RCR / "src/rcr").is_dir(),
    reason="Set CPR_TEST_RCR_REPO for direct cross-repo parity",
)


@pytest.fixture(autouse=True)
def rcr_import(monkeypatch):
    monkeypatch.syspath_prepend(str(RCR / "src"))


def test_subject_memberships_and_partial_matching_match_rcr():
    pytest.importorskip("torch")  # RCR's FAFA adapter imports it.
    from rcr.baselines.fafa_adapter import select_subject_members as expected_members
    from rcr.baselines.fafa_matching import partial_setmatch as expected_score
    from cpr_person.localization import select_subject_members
    from cpr_person.matching import partial_setmatch
    from cpr_person.detection import BoxCandidate

    rng = np.random.default_rng(943)
    for subjects in (1, 2, 3):
        for candidates in (subjects, 6, 11):
            for cap in (1, 5, 8):
                sim = rng.uniform(-1, 1, (subjects, candidates)).astype(np.float32)
                sim[:, -1] = sim[:, 0]
                valid = np.arange(candidates) % 3 != 0
                boxes = [BoxCandidate((0, 0, 5, 7), 1, not bool(v)) for v in valid]
                got, _ = select_subject_members(
                    sim, boxes, threshold=0.2, margin=0.03, max_members=cap
                )
                expected = expected_members(
                    sim, threshold=0.2, margin=0.03, max_members=cap, valid_extra=valid
                )
                assert got == expected
    sim = np.full((1, 9), 0.8, np.float32)
    boxes = [BoxCandidate((0, 0, 5, 7), 1) for _ in range(9)]
    assert select_subject_members(sim, boxes, max_members=5)[0] == [[0, 1, 2, 3, 4]]
    for k in range(1, 10):
        for n in range(0, 10):
            matrix = rng.normal(-2e4, 1e4, (k, n)).astype(np.float32)
            assert partial_setmatch(matrix, -5e4) == expected_score(matrix, -5e4)


def test_crop_and_stored_pixel_orientation_match_rcr(tmp_path):
    from cpr_person.detection import read_rgb, crop_person

    rng = np.random.default_rng(2)
    image = Image.fromarray(rng.integers(0, 256, (39, 71, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "exif.jpg"
    image.save(path, exif=exif)
    with Image.open(path) as source:
        expected = source.convert("RGB")
    actual = read_rgb(path)
    assert actual.size == (71, 39)
    for box in [(-2.5, 0.5, 72.5, 40.5), (1.25, 2.5, 26.75, 38.5), (0, 0, 71, 39)]:
        np.testing.assert_array_equal(crop_person(actual, box), expected.crop(box))


def test_complete_final_splits_and_evaluator_match_rcr(tmp_path):
    pytest.importorskip("torch")  # RCR's evaluator module imports it.
    import prepare_data
    from benchmark_data import load_jsonl, split_paths
    from cpr_baselines.common import public_inputs
    from evaluate import evaluate_scores, rank_scores
    from rcr.evaluation.evaluate import evaluate_rankings
    import json

    final = Path(os.environ.get("CPR_TEST_RCR_FINAL", str(RCR / "dataset/data/final")))
    metadata = prepare_data.import_final(final, tmp_path)
    source = {x["sample_id"]: x for x in load_jsonl(final / "samples.jsonl")}
    expected = {
        "val": (1799, 5684, dict(INDIVIDUAL=1320, GROUP=168, DUAL=149, RELATIONAL=162)),
        "test": (
            2307,
            7868,
            dict(INDIVIDUAL=1319, GROUP=329, DUAL=318, RELATIONAL=341),
        ),
    }
    mapping = {
        "ID-mAP": "id_map",
        "Full-mAP": "full_map",
        **{
            f"{a}-R@{k}": f"{b}_r{k}"
            for a, b in [("ID", "id"), ("Full", "full")]
            for k in (1, 5, 10)
        },
    }
    for split in ("val", "test"):
        gm, qm = split_paths(tmp_path, split)
        gallery, queries = load_jsonl(gm), load_jsonl(qm)
        nq, ng, cases = expected[split]
        assert (
            len(queries),
            len(gallery),
            dict(Counter(q["case"] for q in queries)),
        ) == (nq, ng, cases)
        assert [q["query_id"] for q in queries] == (
            final / "splits" / f"{split}.txt"
        ).read_text().split()
        assert metadata["splits"][split]["num_queries"] == nq
        pg, pq, _ = public_inputs(gallery, queries)
        public = json.dumps([pg, pq])
        for forbidden in (
            "identity_ids",
            "person_ids",
            "target_ids",
            "full_positive_ids",
            "target_image_id",
            '"case"',
        ):
            assert forbidden not in public
        # Deliberate ties, multiple Full/ID positives and all four cases.
        chosen = []
        for case in cases:
            chosen.extend([q for q in queries if q["case"] == case][:3])
        if os.environ.get("CPR_TEST_FULL_EVALUATOR") == "1":
            chosen = queries
        scores = (
            np.random.default_rng(8)
            .integers(-2, 3, (len(chosen), len(gallery)))
            .astype(np.float32)
        )
        actual = evaluate_scores(scores, gallery, chosen)
        gids = [g["image_id"] for g in gallery]
        gindex = {key: i for i, key in enumerate(gids)}
        rankings = {
            q["query_id"]: [gids[i] for i in rank_scores(row, gindex[q["image_id"]])]
            for q, row in zip(chosen, scores)
        }
        samples = [
            {**source[q["query_id"]], "positive_image_ids": q["full_positive_ids"]}
            for q in chosen
        ]
        native = evaluate_rankings(
            samples,
            gids,
            {g["image_id"]: set(g["person_ids"]) for g in gallery},
            rankings,
            candidate_ks=(),
        )
        for key, nkey in mapping.items():
            assert actual["overall"][key] == pytest.approx(
                native["overall"][nkey], abs=1e-14
            )
        for case, values in actual["cases"].items():
            for key in ("ID-mAP", "ID-R@1", "Full-mAP", "Full-R@1"):
                assert values[key] == pytest.approx(
                    native["by_case"][case][mapping[key]], abs=1e-14
                )
        assert actual["overall"]["Macro-Full-mAP"] == fmean(
            v["full_map"] for v in native["by_case"].values()
        )


def test_packed_detection_cache_projects_identical_rcr_fafa_geometry(
    tmp_path, monkeypatch
):
    import json
    torch = pytest.importorskip("torch")
    from rcr.common import detection
    from rcr.baselines.fafa_detector import candidates_from_cache
    from cpr_person.detection import get_detections
    from cpr_person.rcr_cache import RCR_DETECTOR, read_rcr_index

    images = []
    for i in range(2):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (27, 19), ("red", "blue")[i]).save(path)
        images.append(path)
    values = iter(
        [
            torch.tensor([[-2.5, 0.5, 15.25, 20.75], [10.5, 2.0, 28.5, 18.5]]),
            torch.empty(0, 4),
        ]
    )
    monkeypatch.setattr(detection, "detect_persons", lambda *args: next(values))
    source = tmp_path / "rcr-detections"
    detection.build_detection_cache(
        ["a", "b"], images, source, RCR_DETECTOR, model=torch.nn.Identity()
    )
    native = detection.DetectionCache(source)
    index, _ = read_rcr_index(source / "index.pt")
    assert index["cache_id"] == native.cache_id
    expected = candidates_from_cache(None, ["b", "a"], native, {})
    gallery = [
        {"image_id": key, "path": str(images[i])} for key, i in [("b", 1), ("a", 0)]
    ]
    gm = tmp_path / "gallery.jsonl"
    gm.write_text("".join(json.dumps(row) + "\n" for row in gallery))
    monkeypatch.setenv("CPR_RCR_DETECTION_CACHE", str(source))
    monkeypatch.setenv("CPR_SHARED_CACHE_ROOT", str(tmp_path / "shared"))
    got = get_detections(gallery, gm, "cpu")
    for actual, wanted in zip(got.candidates(), expected):
        assert [
            {"box": list(c.box), "score": c.score, "fallback": c.fallback}
            for c in actual
        ] == wanted
    assert got.fingerprint["rcr_cache"]["image_binding"] == "rcr_ids_and_image_sizes"
