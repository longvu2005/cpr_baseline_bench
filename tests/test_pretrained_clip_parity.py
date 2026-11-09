"""Real pretrained component forwards; no claim of full native 29-method inference."""

import gc
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
RCR = Path(os.environ.get("CPR_TEST_RCR_REPO", "/nonexistent"))
HF = Path(
    os.environ.get(
        "CPR_TEST_HF_CLIP", str(ROOT / "checkpoints/clip/hf-vit-base-patch32")
    )
)
OPENAI = Path(
    os.environ.get("CPR_TEST_SELECTOR_CLIP", str(ROOT / "checkpoints/clip/ViT-B-32.pt"))
)
pytestmark = pytest.mark.skipif(
    os.environ.get("CPR_TEST_PRETRAINED_CLIP") != "1",
    reason="Opt in to real pretrained CLIP component inference",
)


@pytest.fixture(autouse=True)
def actual_models(monkeypatch):
    if (
        not (RCR / "src/rcr").is_dir()
        or not (HF / "prepared.json").is_file()
        or not OPENAI.is_file()
    ):
        pytest.fail(
            "Real CLIP parity requested but pinned local assets/RCR checkout are missing"
        )
    monkeypatch.syspath_prepend(str(RCR / "src"))
    monkeypatch.setenv("CPR_SCENE_CLIP_PYTHON", str(Path(os.sys.executable).absolute()))
    monkeypatch.setenv("USE_TF", "0")
    monkeypatch.setenv("USE_FLAX", "0")
    import torch

    torch.set_num_threads(2)
    yield
    gc.collect()


def pictures(tmp_path):
    rng = np.random.default_rng(52)
    paths = []
    for i, shape in enumerate(((37, 139, 3), (121, 41, 3), (67, 67, 3))):
        path = tmp_path / f"fixture-{i}.png"
        Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8)).save(path)
        paths.append(path)
    return paths


def test_hf_letterbox_globals_and_rcr_cache_parity_with_real_weights(
    tmp_path, monkeypatch
):
    import torch
    from transformers import AutoImageProcessor, CLIPVisionModelWithProjection
    from rcr.proposed.cache.scene import _LetterboxProcessor, build_cache
    from rcr.proposed.nn.encoders import ImageEncoder
    from cpr_baselines.scene_worker import encode, LetterboxProcessor
    from cpr_baselines.scene_clip import (
        normalize,
        gallery_features,
        gallery_fingerprint,
        load_scene_clip,
    )
    from cpr_baselines.scene_cache import read_scene_globals
    from cpr_baselines.scene_protocol import MODEL, REVISION
    from cpr_person.cache import canonical_hash
    import json

    paths = pictures(tmp_path)
    images = [Image.open(path).convert("RGB") for path in paths]
    processor = AutoImageProcessor.from_pretrained(
        str(HF), local_files_only=True, use_fast=False
    )
    expected_processor = _LetterboxProcessor(processor, [224, 224])
    actual_processor = LetterboxProcessor(processor)
    torch.testing.assert_close(
        expected_processor(images=images, return_tensors="pt")["pixel_values"],
        actual_processor(images=images)["pixel_values"],
        rtol=0,
        atol=0,
    )
    model = (
        CLIPVisionModelWithProjection.from_pretrained(str(HF), local_files_only=True)
        .float()
        .eval()
    )
    native = ImageEncoder(model).eval()
    with torch.inference_mode():
        expected = native(
            expected_processor(images=images, return_tensors="pt")["pixel_values"]
        )[1].numpy()
    request = {
        "kind": "images",
        "checkpoint": str(HF),
        "device": "cpu",
        "inputs": [str(p) for p in paths],
        "batch_size": 2,
    }
    got = encode(request, image_model=model, processor=actual_processor)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert not any(p.requires_grad for p in model.parameters())
    ids = ["im0", "im1", "im2"]
    cache = tmp_path / "native-scene"
    build_cache(
        ids,
        paths,
        cache,
        native,
        expected_processor,
        "cpu",
        "float16",
        {
            "image_encoder": {
                "model": MODEL,
                "revision": REVISION,
                "scene_size": [224, 224],
            }
        },
    )
    rows = [{"image_id": ids[i], "path": str(paths[i])} for i in (2, 0)]
    copied, proof = read_scene_globals(cache / "index.pt", rows, canonical_hash(ids))
    np.testing.assert_allclose(copied, expected[[2, 0]], rtol=1e-5, atol=1e-5)
    # RCR's own builder writes no image digests; reuse must still work.
    assert proof["native_global_features_only"]
    assert proof["image_binding"] == "rcr_gallery_order"
    monkeypatch.setenv("CPR_RCR_SCENE_CACHE", str(cache))
    monkeypatch.setenv("CPR_SCENE_CACHE_ROOT", str(tmp_path / "shared"))
    gm = tmp_path / "gallery.jsonl"
    gm.write_text("".join(json.dumps(row) + "\n" for row in rows))
    scene, _ = load_scene_clip(HF, "cpu")
    fp = gallery_fingerprint(HF, gm, MODEL)
    a, _, hit = gallery_features(scene, None, rows, None, {"batch_size": 2}, "cpu", fp)
    b, _, warm = gallery_features(scene, None, rows, None, {"batch_size": 2}, "cpu", fp)
    assert not hit and warm
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, normalize(expected[[2, 0]]), rtol=1e-5, atol=1e-5)


def test_hf_text_and_subprocess_facade_match_rcr_native_globals(tmp_path):
    import torch
    from transformers import CLIPTokenizerFast, CLIPTextModelWithProjection
    from rcr.proposed.nn.encoders import TextEncoder
    from cpr_baselines.scene_worker import encode
    from cpr_baselines.scene_clip import load_scene_clip

    texts = [
        "Subject 1 wears a red jacket.",
        "A woman standing beside Subject 2.",
        "café Tiếng Việt " + "walking " * 110,
    ]
    tokenizer = CLIPTokenizerFast.from_pretrained(str(HF), local_files_only=True)
    tokens = tokenizer(
        texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
    )
    assert (
        tokens.input_ids.shape[1] == 77
        and tokens.input_ids[-1, -1] == tokenizer.eos_token_id
    )
    model = (
        CLIPTextModelWithProjection.from_pretrained(str(HF), local_files_only=True)
        .float()
        .eval()
    )
    native = TextEncoder(model, dim=17).eval()
    with torch.inference_mode():
        expected = native(tokens.input_ids, tokens.attention_mask)[2].numpy()
    request = {
        "kind": "texts",
        "checkpoint": str(HF),
        "device": "cpu",
        "inputs": texts,
        "batch_size": 2,
    }
    got = encode(request, text_model=model, tokenizer=tokenizer)
    with torch.no_grad():
        native.proj.weight.add_(5)
    with torch.inference_mode():
        after = native(tokens.input_ids, tokens.attention_mask)[2].numpy()
    np.testing.assert_array_equal(after, expected)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    scene, _ = load_scene_clip(HF, "cpu")
    facade = scene.texts(texts, 2)
    np.testing.assert_allclose(facade, expected, rtol=1e-5, atol=1e-5)


def test_openai_selector_matches_rcr_pretrained_crop_and_text_features(tmp_path):
    import torch
    from rcr.baselines.clip import load_clip
    from cpr_person.localization import ClipSelector
    from cpr_person.detection import crop_person
    from cpr_person.cache import sha256_file
    from cpr_person.protocol import RCR_CLIP_SHA256

    assert sha256_file(OPENAI) == RCR_CLIP_SHA256
    paths = pictures(tmp_path)
    crops = [
        crop_person(Image.open(paths[0]).convert("RGB"), box)
        for box in ((-3.5, 0.5, 140.5, 38.5), (0, 0, 41.25, 36.75))
    ]
    texts = ["the person on the left", "a woman wearing a red hat"]
    clip, model, preprocess = load_clip(OPENAI, torch.device("cpu"))
    with torch.inference_mode():
        expected_i = torch.nn.functional.normalize(
            model.encode_image(torch.stack([preprocess(i) for i in crops])).float(),
            dim=-1,
        ).numpy()
        expected_t = torch.nn.functional.normalize(
            model.encode_text(clip.tokenize(texts, truncate=True)).float(), dim=-1
        ).numpy()
    del model
    gc.collect()
    selector = ClipSelector({"checkpoint": str(OPENAI)}, "cpu")
    np.testing.assert_allclose(
        selector.images(crops).numpy(), expected_i, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        selector.texts(texts).numpy(), expected_t, rtol=1e-6, atol=1e-6
    )


def test_actual_selector_worker_and_warm_cache_match_direct_selector(
    tmp_path, monkeypatch
):
    import json
    import torch
    from cpr_person.cache import canonical_hash, sha256_file
    from cpr_person.detection import get_detections, protocol_config
    from cpr_person.localization import ClipSelector, localize_queries, get_localization
    from cpr_person.rcr_cache import _rcr_cache_id, RCR_DETECTOR

    paths = pictures(tmp_path)
    gallery = [
        {"image_id": str(i), "path": str(path)} for i, path in enumerate(paths[:2])
    ]
    subjects = [
        {
            "subject_id": 1,
            "select_text": "a person in a red shirt",
            "identity_ids": ["forbidden"],
        }
    ]
    queries = [
        {
            "query_id": "group",
            "image_id": "0",
            "subjects": subjects,
            "case": "GROUP",
            "target_ids": ["forbidden"],
        },
        {
            "query_id": "dual",
            "image_id": "1",
            "subjects": subjects
            + [{"subject_id": 2, "select_text": "the person on the right"}],
        },
    ]
    gm, qm = tmp_path / "gallery.jsonl", tmp_path / "queries.jsonl"
    for manifest, rows in ((gm, gallery), (qm, queries)):
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    # Geometry is a fixture; only CLIP weights and the actual selector worker run here.
    index = {
        "layout": "person-detections-v1",
        "prompt": "person",
        "detector": RCR_DETECTOR,
        "image_ids": ["0", "1"],
        "image_content_sha256": [sha256_file(p) for p in paths[:2]],
        "image_sizes": torch.tensor([Image.open(p).size for p in paths[:2]]),
        "offsets": torch.tensor([0, 2, 2]),
        "boxes_pixel": torch.tensor(
            [[-2.5, 0.5, 68.5, 37.5], [70.5, 0.0, 140.5, 38.5]]
        ),
    }
    index["cache_id"] = _rcr_cache_id(index)
    source = tmp_path / "index.pt"
    torch.save(index, source)
    monkeypatch.setenv("CPR_RCR_DETECTION_CACHE", str(source))
    monkeypatch.setenv("CPR_SHARED_CACHE_ROOT", str(tmp_path / "shared"))
    detections = get_detections(gallery, gm, "cpu")
    cfg = protocol_config()["selector"]
    direct = ClipSelector(cfg, "cpu")
    expected, stats = localize_queries(queries, gallery, detections, cfg, direct)
    del direct
    gc.collect()
    actual = get_localization(queries, qm, gallery, detections, "cpu")
    assert (
        actual.members == expected
        and canonical_hash(actual.stats) == canonical_hash(stats)
        and not actual.cache_hit
    )
    warm = get_localization(queries, qm, gallery, detections, "cpu")
    assert warm.cache_hit and canonical_hash(warm.members) == canonical_hash(expected)
