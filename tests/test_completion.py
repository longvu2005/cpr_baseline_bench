"""Adapter, provenance and validation contracts; these are not native benchmark runs."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from cpr_person.matching import aggregate_people
from cpr_person.unmatched import select_unmatched, native_distance_grid
from cpr_person.cache import canonical_hash, read_json, sha256_file
from cpr_baselines.api_trace import complete, validate_cached


def test_val_penalty_selection_preserves_native_scores_and_native_scale(tmp_path):
    gallery = [{"image_id": str(i)} for i in range(4)]
    queries = [{"query_id": "q", "image_id": "0", "full_positive_ids": ["2"]}]
    offsets = np.array([0, 2, 3, 5, 5])
    pairs = np.array(
        [[-2e4, -3e4, -1e3, -1.5e4, -2e4], [-3e4, -2e4, -2e3, -2e4, -1.5e4]], np.float32
    )
    result = select_unmatched(
        [-1e6, -2e4, -1e3],
        gallery,
        queries,
        offsets,
        lambda _: pairs,
        tmp_path / "selection.json",
        {"val": "only"},
    )
    assert result["selected_unmatched_score"] == -1e6
    assert (
        result["selection_split"] == "val"
        and result["pairwise_metric"] == "negative_squared_euclidean"
    )
    assert result["trials"][0]["Full-mAP"] == 1.0
    assert result["num_queries_with_any_shortage"] == 1
    assert read_json(tmp_path / "selection.json") == result
    from cpr_person.matching import partial_setmatch_sum

    rng = np.random.default_rng(43)
    for k in range(1, 13):
        p = rng.normal(-20000, 50000, (k, offsets[-1])).astype(np.float32)
        sums = np.array(
            [
                partial_setmatch_sum(p[:, a:b])[0]
                for a, b in zip(offsets[:-1], offsets[1:])
            ]
        )
        missing = np.maximum(k - np.diff(offsets), 0)
        for b in [-1e6, -123.4, 0.0]:
            affine = ((sums + missing * b) / k).astype(np.float32)
            np.testing.assert_array_equal(affine, aggregate_people(p, offsets, b))


def test_native_distance_grid_is_deterministic_without_normalization():
    q = np.array([[100, 0], [0, 100]], np.float32)
    g = np.array([[200, 0], [0, 300]], np.float32)
    a, policy = native_distance_grid(q, g)
    b, _ = native_distance_grid(q, g)
    np.testing.assert_array_equal(a, b)
    assert a[0] == -100000 and a[-1] == -10000 and not policy["feature_normalization"]
    assert len(policy["quantiles"]) == 21
    with pytest.raises(ValueError):
        native_distance_grid(q * float("nan"), g)


def test_native_api_request_is_unchanged_and_receipt_is_bound(tmp_path):
    calls = []

    class Response:
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(content="Suggestion: keep the red jacket")
            )
        ]

        def model_dump(self, mode):
            return {
                "id": "resp-actual",
                "model": "gpt-4o-2024-08-06",
                "created": 123,
                "system_fingerprint": "fp-1",
                "choices": [{"message": {"content": self.choices[0].message.content}}],
            }

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kw: calls.append(kw) or Response()
            )
        )
    )
    messages = [{"role": "user", "content": "original native prompt"}]
    output, receipt = complete(client, "gpt-4o", messages, tmp_path / "identity.json")
    assert calls == [{"model": "gpt-4o", "messages": messages}]
    validate_cached(receipt, "gpt-4o", messages, output, tmp_path / "identity.json")
    altered = deepcopy(receipt)
    altered["response"]["choices"][0]["message"]["content"] = "changed"
    with pytest.raises(ValueError):
        validate_cached(altered, "gpt-4o", messages, output, tmp_path / "identity.json")
    changed = deepcopy(receipt)
    changed["response"]["model"] = "different"
    changed["response_sha256"] = canonical_hash(changed["response"])
    with pytest.raises(RuntimeError, match="model identity changed"):
        validate_cached(changed, "gpt-4o", messages, output, tmp_path / "identity.json")


def test_word4per_receipt_rejects_renamed_overlap_and_changed_weights(
    tmp_path, monkeypatch, write_benchmark_views
):
    from PIL import Image
    from cpr_person.reproduction import audit_word4per_data, admit_word4per
    import cpr_person.reproduction as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    image = tmp_path / "rcr.png"
    Image.new("RGB", (9, 8), "red").save(image)
    val = tmp_path / "val.png"
    Image.new("RGB", (9, 8), "blue").save(val)
    g = [{"image_id": "test", "path": str(image)}]
    q = [{"query_id": "test"}]
    write_benchmark_views(
        tmp_path,
        g,
        q,
        val_gallery=[{"image_id": "val", "path": str(val)}],
        val_queries=[{"query_id": "val"}],
    )
    external = tmp_path / "external"
    (external / "CUHK-PEDES/imgs").mkdir(parents=True)
    extra = external / "CUHK-PEDES/imgs/external.png"
    Image.new("RGB", (9, 8), "green").save(extra)
    (external / "CUHK-PEDES/reid_raw.json").write_text(
        json.dumps([{"split": "train", "file_path": "external.png"}])
    )
    for name in ["query.json", "gallery.json"]:
        (external / name).write_text(
            json.dumps([{"file_path": "CUHK-PEDES/imgs/external.png"}])
        )
    audit = audit_word4per_data(external, tmp_path)
    weights, config = tmp_path / "stage2", tmp_path / "config.yaml"
    weights.write_bytes(b"native weights")
    config.write_text("native: true")
    receipt = tmp_path / "receipt.json"
    record = {
        "official_commit": "source",
        "stage2_checkpoint_sha256": sha256_file(weights),
        "stage2_config_sha256": sha256_file(config),
        "stage1_sha256": "a" * 64,
        "cpr_supervision": False,
        "external_data_audit": audit,
        "selection": "external ITCPR",
    }
    receipt.write_text(json.dumps(record))
    cfg = {
        "source": {"commit": "source"},
        "checkpoint": {"reproduction_provenance": str(receipt)},
    }
    assert (
        admit_word4per(cfg, weights, config, tmp_path)["external_inventory_sha256"]
        == audit["inventory_sha256"]
    )
    weights.write_bytes(b"other weights")
    with pytest.raises(ValueError, match="receipt"):
        admit_word4per(cfg, weights, config, tmp_path)
    extra.write_bytes(image.read_bytes())
    with pytest.raises(ValueError, match="overlap"):
        audit_word4per_data(external, tmp_path)


def test_preflight_does_not_claim_model_inference_or_expose_credentials(
    tmp_path, monkeypatch
):
    from cpr_baselines.preflight import method_readiness

    cfg = {
        "method": "wiser",
        "runtime": {"device": "cuda"},
        "models": {"refiner": {"api_key_env": "PRIVATE_NATIVE_KEY"}},
        "isolated_env": {"minimum_free_disk_gib": 70},
    }
    monkeypatch.delenv("PRIVATE_NATIVE_KEY", raising=False)
    result = method_readiness(cfg, tmp_path, gpu_mib=[], free_bytes=1024)
    assert len(result["blockers"]) == 3 and not result["native_inference_verified"]
    monkeypatch.setenv("PRIVATE_NATIVE_KEY", "secret-do-not-print")
    result = method_readiness(cfg, tmp_path, gpu_mib=[81920], free_bytes=100 * 1024**3)
    assert result["prerequisites_passed"] and not result["native_inference_verified"]
    assert "secret-do-not-print" not in repr(result)


def test_actual_state_digest_binds_weights_dtype_and_scalar_buffers():
    torch = pytest.importorskip("torch")
    from cpr_baselines.provenance import module_state_sha256

    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    before = module_state_sha256(model)
    with torch.no_grad():
        model[0].weight[0, 0] += 1
    assert module_state_sha256(model) != before
    current = module_state_sha256(model)
    assert module_state_sha256(model.half()) != current


def test_evidence_moves_receipts_without_changing_their_content(tmp_path):
    from cpr_baselines.evidence import persist_evidence

    original = tmp_path / "cache/selection.json"
    original.parent.mkdir()
    original.write_text('{"selected":-5000}')
    proof = {
        "selection_split": "val",
        "artifact": str(original),
        "artifact_sha256": sha256_file(original),
    }
    run = {
        "method": "instruct_reid_setmatch",
        "setmatch": {"calibration_provenance": proof},
    }
    result = persist_evidence(run, tmp_path / "runs/instruct_reid_setmatch", tmp_path)
    destination = tmp_path / result["setmatch"]["calibration_provenance"]["artifact"]
    assert destination.read_bytes() == original.read_bytes()
    assert not destination.is_relative_to(tmp_path / "cache")


def test_publication_check_cannot_treat_synthetic_metrics_as_native(
    tmp_path, write_benchmark_views
):
    from PIL import Image
    import runpy
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    checker = runpy.run_path(str(root / "scripts/check_publication.py"))[
        "audit_artifacts"
    ]
    image = tmp_path / "im.png"
    Image.new("RGB", (4, 4), "blue").save(image)
    rows = [
        {"image_id": "a", "path": str(image), "person_ids": ["id"]},
        {"image_id": "b", "path": str(image), "person_ids": ["id"]},
    ]
    queries = [
        {
            "query_id": "q",
            "image_id": "a",
            "target_ids": ["id"],
            "full_positive_ids": ["b"],
            "case": "INDIVIDUAL",
        }
    ]
    write_benchmark_views(
        tmp_path, rows, queries, val_gallery=rows, val_queries=queries
    )
    result = checker(tmp_path, [SimpleNamespace(method_id="clip_image")])
    assert (
        not result["all_methods_complete"]
        and not result["results"][0]["artifact_checks_passed"]
    )
    assert "missing completed" in result["results"][0]["blockers"][0]


def test_instruct_validation_glue_freezes_val_and_ignores_test_labels(
    tmp_path, monkeypatch, write_benchmark_views
):
    pytest.importorskip("torch")
    import importlib.util
    from pathlib import Path
    import yaml
    from PIL import Image
    import cpr_person.instruct_validation as validation
    from benchmark_data import manifest_fingerprint

    root = Path(__file__).resolve().parents[1]
    path = root / "methods/published/03_instruct_reid_setmatch/run.py"
    spec = importlib.util.spec_from_file_location("native_instruct_contract", path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    cfg = yaml.safe_load(path.with_name("config.yaml").read_text())
    cfg["model"]["feature_dim"] = 2
    cfg["runtime"]["device"] = "cpu"
    monkeypatch.setattr(validation, "ROOT", tmp_path)
    monkeypatch.setenv("CPR_SHARED_CACHE_ROOT", str(tmp_path / "shared"))
    groups = {}
    for split in ("val", "test"):
        groups[split] = []
        for i, color in enumerate(("red", "green", "blue")):
            image = tmp_path / f"{split}-{i}.png"
            Image.new("RGB", (9, 7), color).save(image)
            groups[split].append(
                {
                    "image_id": f"{split}{i}",
                    "path": str(image),
                    "person_ids": ["a", "b"],
                }
            )

    def q(split):
        return [
            {
                "query_id": split,
                "image_id": split + "0",
                "text": "Subject 1 red and Subject 2 blue",
                "subjects": [
                    {"subject_id": 1, "select_text": "red", "modify_text": "red hat"},
                    {"subject_id": 2, "select_text": "blue", "modify_text": "blue hat"},
                ],
                "target_ids": ["a", "b"],
                "full_positive_ids": [split + "2"],
                "case": "DUAL",
            }
        ]

    write_benchmark_views(
        tmp_path,
        groups["test"],
        q("test"),
        val_gallery=groups["val"],
        val_queries=q("val"),
    )
    dpath, lpath = tmp_path / "d.npz", tmp_path / "members.json"
    dpath.write_bytes(b"geometry")
    lpath.write_text("{}")

    def person(*args):
        d = SimpleNamespace(
            path=dpath,
            offsets=np.array([0, 1, 2, 3]),
            boxes=np.array([[0, 0, 9, 7]] * 3),
        )
        localization = SimpleNamespace(
            path=lpath,
            flat=lambda _: [
                (0, {"global_person_index": 0, "box": [0, 0, 9, 7]}),
                (1, {"global_person_index": None, "box": [0, 0, 9, 7]}),
            ],
        )
        return SimpleNamespace(
            detections=d,
            localization=localization,
            metadata=lambda: {"localization_fingerprint": {"val": "same"}},
        )

    monkeypatch.setattr(validation, "load_person_context", person)
    calls = []

    def gallery(**kw):
        calls.append("gallery")
        values = np.array([[100, 0], [0, 100], [50, 50]], np.float16)
        np.save(kw["cache_path"], values)
        native.write_json(native.meta_path(kw["cache_path"]), kw["cache_meta"])
        return values

    def query(**kw):
        calls.append("query")
        offsets = np.array([0, 2])
        values = np.array([[100, 0], [0, 100]], np.float16)
        np.savez(kw["cache_path"], offsets=offsets, features=values)
        native.write_json(native.meta_path(kw["cache_path"]), kw["cache_meta"])
        return offsets, values

    monkeypatch.setattr(native, "compute_gallery_instruct_features", gallery)
    monkeypatch.setattr(native, "compute_query_instruct_features", query)
    files = [tmp_path / name for name in ["checkpoint", "bert", "config_bert"]]
    for file in files:
        file.write_bytes(file.name.encode())

    def select():
        return validation.select_for_instruct(native, cfg, object(), "cpu", *files)

    selected = select()
    proof = deepcopy(cfg["setmatch"]["calibration_provenance"])
    assert (
        selected == manifest_fingerprint(tmp_path, "val")
        and cfg["cpr_supervision"] == "Val only"
    )
    assert cfg["setmatch"]["unmatched_score"] < -1.0
    changed = q("test")
    changed[0]["full_positive_ids"] = ["test1"]
    write_benchmark_views(
        tmp_path,
        groups["test"],
        changed,
        val_gallery=groups["val"],
        val_queries=q("val"),
    )
    assert select() == selected and cfg["setmatch"]["calibration_provenance"] == proof
    assert calls == ["gallery", "query"]


def test_wiser_wrapper_preserves_native_prompts_outputs_and_warm_cache(tmp_path):
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "methods/published/11_wiser/run.py"
    tree = ast.parse(path.read_text())
    names = {
        "build_gallery_index",
        "eligible_top",
        "t2i_refine_prompt",
        "i2i_refine_prompt",
        "get_suggestions",
    }
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    calls = []

    class Response:
        choices = [
            SimpleNamespace(
                message=SimpleNamespace(content="Suggestion: keep the original jacket")
            )
        ]

        def model_dump(self, mode):
            return {
                "id": "actual-response-shape",
                "model": "gpt-4o-2024-08-06",
                "choices": [{"message": {"content": self.choices[0].message.content}}],
            }

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: calls.append(kwargs) or Response()
            )
        )
    )
    env = {
        "np": np,
        "os": __import__("os"),
        "json": json,
        "resolve": Path,
        "load_jsonl": lambda p: [
            json.loads(line) for line in p.read_text().splitlines()
        ],
        "progress_bar": lambda values, **kwargs: values,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    cfg = {
        "cache": {
            "suggestions_dir": str(tmp_path / "suggestions"),
            "dir": str(tmp_path),
        },
        "wiser": {"topk_per_path": 1},
        "models": {"refiner": {"model": "gpt-4o"}},
    }
    generation = [
        {
            "source_caption": "the original scene",
            "instruction": "the full native instruction",
        }
    ]
    kwargs = {
        "cfg": cfg,
        "client": client,
        "queries": [{"image_id": "a"}],
        "generation": generation,
        "gallery": [{"image_id": "a"}, {"image_id": "b"}],
        "captions": {"a": "reference", "b": "candidate caption"},
        "t_rank": np.array([[0, 1]]),
        "i_rank": np.array([[0, 1]]),
        "t_uncertain": [True],
        "i_uncertain": [False],
        "loop": 0,
    }
    expected_prompt = env["t2i_refine_prompt"](
        "the original scene", "the full native instruction", ["candidate caption"]
    )
    first = env["get_suggestions"](**kwargs)
    assert calls == [
        {"model": "gpt-4o", "messages": [{"role": "user", "content": expected_prompt}]}
    ]
    assert first == (
        ["Suggestion: keep the original jacket"],
        ["Good retrieval, no more loops needed"],
    )
    assert env["get_suggestions"](**kwargs) == first and len(calls) == 1


def test_legacy_cache_verification_rejects_mismatch_before_binding_images(tmp_path):
    from pathlib import Path
    import runpy
    torch = pytest.importorskip("torch")
    from PIL import Image

    root = Path(__file__).resolve().parents[1]
    verify = runpy.run_path(str(root / "scripts/verify_rcr_cache_provenance.py"))[
        "verify_copy"
    ]
    image = tmp_path / "image.png"
    Image.new("RGB", (9, 7), "green").save(image)
    index = {
        "global_features": torch.arange(512, dtype=torch.float32).reshape(1, 512),
        "image_ids": ["a"],
        "cache_id": "original",
    }
    source = tmp_path / "source.pt"
    torch.save(index, source)
    destination = tmp_path / "verified/index.pt"
    with pytest.raises(ValueError, match="tensor differs"):
        verify(index, [image], lambda _: np.ones(512), source, destination, "scene", {})
    assert not destination.exists() and "image_content_sha256" not in index
    original = source.read_bytes()
    receipt = verify(
        index,
        [image],
        lambda _: index["global_features"][0].numpy(),
        source,
        destination,
        "scene",
        {},
    )
    copy = torch.load(destination, weights_only=True)
    assert source.read_bytes() == original and copy["cache_id"] == "original"
    assert (
        copy["image_content_sha256"] == [sha256_file(image)]
        and receipt["all_original_tensors_compared"]
    )
    torch.testing.assert_close(
        copy["global_features"], index["global_features"], rtol=0, atol=0
    )


def test_word4per_preflight_requires_the_reproduction_receipt(tmp_path, monkeypatch):
    from cpr_baselines.preflight import method_readiness

    for name in (
        "WORD4PER_STAGE2_CHECKPOINT",
        "WORD4PER_STAGE2_CONFIG",
        "WORD4PER_STAGE2_PROVENANCE",
    ):
        monkeypatch.delenv(name, raising=False)
    folder = tmp_path / "word4per"
    folder.mkdir()
    (folder / "weights.pth").write_bytes(b"native checkpoint fixture")
    (folder / "configs.yaml").write_text("native: true")
    cfg = {
        "method": "word4per_setmatch",
        "runtime": {"device": "cuda"},
        "checkpoint": {
            "stage2": str(folder / "weights.pth"),
            "stage2_config": str(folder / "configs.yaml"),
        },
    }
    result = method_readiness(cfg, tmp_path, gpu_mib=[81920])
    assert not result["prerequisites_passed"] and "receipt" in result["blockers"][0]
    (folder / "word4per_cuhk_pedes_stage2_reproduction.json").write_text("{}")
    # Presence is only a preflight; admit_word4per validates its actual content.
    result = method_readiness(cfg, tmp_path, gpu_mib=[81920])
    assert result["prerequisites_passed"] and not result["native_inference_verified"]
