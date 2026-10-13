"""Full inference/evaluation contracts with deterministic toy encoders, no weights."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

import build_tables
import evaluate
from benchmark_data import manifest_fingerprint
from cpr_baselines import captions, common, prepare, runner
from cpr_baselines.person_cache import require_shared_context
from cpr_person.cache import StageCache, atomic_json
from cpr_person.detection import detection_fingerprint
from cpr_person.matching import aggregate_people
from run_baseline import discover_methods, resolve_method

ROOT = Path(__file__).resolve().parents[1]
NEW = {"siglip2_compose", "searle_xl", "context_i2w", "fti4cir", "qwen_caption_to_caption",
       "structured_caption_retrieval", "gme_qwen2_vl_2b"}


def config(method):
    spec = resolve_method(method, discover_methods())
    return yaml.safe_load(spec.config_path.read_text())


def test_seven_integrations_and_assets_are_pinned():
    methods = discover_methods()
    for method in NEW:
        spec = resolve_method(method, methods)
        assert resolve_method(spec.directory.name, methods) == spec
        cfg = config(method)
        for asset in cfg["assets"].values():
            if asset["kind"] in {"hf", "hf_file", "git"}:
                assert len(asset["revision"]) == 40 and all(c in '0123456789abcdef' for c in asset["revision"])
        assert cfg["output"]["dir"] == f"runs/{method}"
        assert cfg["data"] == {"gallery_manifest": "data/gallery.jsonl", "query_manifest": "data/queries.jsonl"}
        assert cfg["cpr_supervision"] == "No"
    assert config("structured_caption_retrieval")["person_protocol"] == "cpr_person/protocol.yaml"
    assert config("siglip2_compose")["composition"]["alpha_selection"] == "fixed_no_benchmark_labels"


def test_model_inputs_cannot_observe_labels(toy_protocol):
    _, gallery, queries, *_ = toy_protocol
    queries = [{**row, "query_id": str(i), "text": "public instruction", "full_positive_ids": ["label"],
                "target_image_id": "oracle", "target_ids": ["GT"], "final_change": "public target"}
               for i, row in enumerate(queries)]
    first = common.public_inputs(gallery, queries)
    other_gallery, other_queries = deepcopy(gallery), deepcopy(queries)
    for row in other_gallery:
        row["person_ids"] = ["poison"]
    for row in other_queries:
        for key in ("case", "target_ids", "full_positive_ids", "target_image_id"):
            row[key] = {"poison": [1, 2, 3]}
        for subject in row["subjects"]:
            subject["identity_ids"] = ["poison"]
            subject["identity_id"] = "poison"
    assert common.public_inputs(other_gallery, other_queries) == first
    assert all(set(row) <= {"image_id", "path"} for row in first[0])


def test_scene_and_person_image_digests_are_identical(toy_protocol):
    cfg, gallery, _, gm, *_ = toy_protocol
    assert common.image_digest(gallery) == detection_fingerprint(gm, cfg)["gallery_image_content_sha256"]


def test_cosine_composition_and_chunked_scene_scores(tmp_path):
    image = np.asarray([[3, 0], [0, 2]], np.float32)
    text = np.asarray([[0, 4], [4, 0]], np.float32)
    expected = np.full((2, 2), 1 / np.sqrt(2), np.float32)
    np.testing.assert_allclose(common.compose_linear(image, text, .5), expected)
    g = common.normalized(np.asarray([[1, 0], [0, 1], [1, 1]], np.float32))
    for qb, gb in [(1, 1), (8, 2), (1, 128)]:
        path = tmp_path / f"{qb}-{gb}.npy"
        common.write_scene_scores(expected, g, path, qb, gb)
        np.testing.assert_allclose(np.load(path), expected @ g.T)
    with pytest.raises(ValueError):
        common.compose_linear(image, text, np.nan)
    with pytest.raises(ValueError):
        common.normalized(np.zeros((1, 3)))


def test_feature_cache_reuses_rows_and_detects_corruption(tmp_path):
    cache = StageCache('test', {'version': 1}, write_root=tmp_path)
    calls = []
    def make(start, end):
        calls.append((start, end))
        return np.ones((end - start, 3), np.float32)
    common.cached_features(cache, 'features.npy', 3, make, 2)
    common.cached_features(cache, 'features.npy', 3, lambda *_: pytest.fail('unnecessary forward'), 2)
    assert calls == [(0, 2), (2, 3)]
    (cache.directory/'features.npy').write_bytes(b'corrupted bytes')
    common.cached_features(cache, 'features.npy', 3, make, 2)
    assert len(calls) == 4


def test_failed_batch_keeps_prior_artifact_intact(tmp_path):
    cache = StageCache('test', {'version': 1}, write_root=tmp_path)
    def fail(start, end):
        if start:
            raise RuntimeError('native inference failure')
        return np.ones((end-start, 3))
    with pytest.raises(RuntimeError):
        common.cached_features(cache, 'features.npy', 4, fail, 2)
    assert not (cache.directory/'features.npy').exists()
    assert not (cache.directory/'features.npy.partial').exists()


@pytest.mark.parametrize('raw', ['{}', '{"appearance": 1}', 'invalid',
    '{"appearance":"a","clothing":"b","action":"c","objects":"d","identity_ids":[1]}'])
def test_structure_rejects_incomplete_or_extra_fields(raw):
    with pytest.raises((ValueError, TypeError)):
        captions.parse_structure(raw)


def test_structure_has_fixed_field_order():
    raw = '```json\n{"objects":"cup","action":"sitting","clothing":"red","appearance":"short hair"}\n```'
    parsed = captions.parse_structure(raw)
    assert captions.structure_text(parsed) == 'appearance: short hair; clothing: red; action: sitting; objects: cup'


def test_caption_cache_resumes_after_failure_and_uses_no_ground_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(captions, 'StageCache', lambda stage, fp, **kw: StageCache(stage, fp, write_root=tmp_path))
    cfg = config('qwen_caption_to_caption')
    class Generator:
        calls = []
        def generate(self, prompt, image):
            self.calls.append(prompt)
            if prompt == 'second':
                raise RuntimeError('interrupted')
            return 'A visible person with a hat.'
    generator = Generator()
    store = captions.CaptionStore(cfg, 'verified-assets', lambda: generator)
    assert store.request('first', Image.new('RGB', (2, 2), 'red'))
    with pytest.raises(RuntimeError):
        store.request('second')
    restored = captions.CaptionStore(cfg, 'verified-assets', lambda: pytest.fail('should reuse complete item'))
    assert restored.request('first', Image.new('RGB', (2, 2), 'red'))
    assert restored.hits == 1
    assert len(generator.calls) == 2


def test_structured_people_keep_grounding_multiplicity_and_partial_matching(toy_protocol, tmp_path):
    from cpr_person.adapter import load_person_context
    _, gallery, queries, gm, qm, calls, _ = toy_protocol
    raw = [{**q, 'query_id': str(i), 'text': 'public instruction'} for i,q in enumerate(queries)]
    # The actual shared adapter consumes the original manifests, including all
    # multi-member subjects and padded fallback slots.
    person = load_person_context(gallery, queries, gm, qm, 'cpu')
    g, q, index = common.public_inputs(gallery, raw)
    class Store:
        calls = []
        def request(self, prompt, image=None, structured=False):
            self.calls.append((prompt, image is not None))
            return dict(appearance='person', clothing='shirt', action='standing', objects='')
    store = Store()
    records, target, offsets = captions.person_captions(config('structured_caption_retrieval'), g, q, index, person, store)
    assert len(records) == 3
    assert np.diff(offsets).tolist() == [2, 2, 3]
    assert len(calls) == 1
    assert all('oracle' not in prompt and 'GT' not in prompt for prompt,_ in store.calls)
    features = common.normalized(np.asarray([[1, 0], [0, 1], [1, 1]], np.float32))
    components = common.normalized(np.asarray([[1, 0], [0, 1]] * 3 + [[1, 1]], np.float32))
    cfg = config('structured_caption_retrieval')
    path = tmp_path / 'scores.npy'
    runner.write_person_scores(components, features, offsets, person.detections.offsets, path, cfg)
    expected = np.stack([aggregate_people(components[offsets[i]:offsets[i+1]] @ features.T,
                                           person.detections.offsets, -1) for i in range(3)])
    np.testing.assert_allclose(np.load(path), expected)


def test_captioner_is_forbidden_to_recompute_grounding(toy_protocol, monkeypatch):
    _, _, _, gm, qm, *_ = toy_protocol
    cfg = {'data': {'gallery_manifest': str(gm), 'query_manifest': str(qm)}}
    monkeypatch.setenv('CPR_REQUIRE_SHARED_CACHE', 'previous')
    def loader(*args, **kwargs):
        import os
        assert kwargs['require_cache'] is True
        assert os.environ['CPR_REQUIRE_SHARED_CACHE'] == '1'
        return 'shared-context'
    assert require_shared_context(cfg, 'cpu', loader=loader) == 'shared-context'
    import os
    assert os.environ['CPR_REQUIRE_SHARED_CACHE'] == 'previous'


def test_prepared_asset_validation_rejects_changed_bytes(tmp_path):
    p = tmp_path/'weights.pt';p.write_bytes(b'official bytes')
    cfg = {'assets': {'checkpoint': {'kind': 'url', 'path': str(p), 'url': 'https://example.invalid/weights'}},
           'prepared_marker': str(tmp_path/'prepared.json')}
    info = {'kind': 'url', **prepare.file_info(p)}
    atomic_json(tmp_path/'prepared.json', {'spec_hash': prepare.canonical_hash(prepare.asset_spec(cfg)),
                                          'assets': {'checkpoint': info}})
    assert prepare.validate_prepared(cfg)['assets']['checkpoint']['size'] == len(b'official bytes')
    p.write_bytes(b'tampered bytes')
    with pytest.raises(RuntimeError, match='bytes changed'):
        prepare.validate_prepared(cfg)


def test_skip_install_preflight_never_installs_packages(monkeypatch):
    import cpr_baselines.person_cache
    script = common.load_source_file('cpr_shared_cache_preflight_test',
        ROOT/'methods/simple/14_structured_caption_retrieval/prepare_person_cache.py')
    monkeypatch.setattr('sys.argv', ['prepare_person_cache.py', '--check-only'])
    def missing(*args, **kwargs):
        raise RuntimeError('cache is missing')
    monkeypatch.setattr(cpr_baselines.person_cache, 'require_shared_context', missing)
    monkeypatch.setattr(script.subprocess, 'run', lambda *_args, **_kwargs: pytest.fail('skip-install must not invoke pip'))
    with pytest.raises(SystemExit, match='without --skip-install'):
        script.main()


def test_alternate_manifests_cannot_be_mistaken_for_evaluator_inputs(tmp_path):
    cfg = config('siglip2_compose')
    cfg['data']['gallery_manifest'] = str(tmp_path/'different_gallery.jsonl')
    with pytest.raises(ValueError, match='canonical manifests'):
        runner.run(cfg)


def test_fusion_scene_adapter_evaluates_multiple_positives(toy_protocol, tmp_path, monkeypatch):
    # Full adapter -> existing evaluator -> table builder, with a deterministic
    # toy backbone. No native GPU claim and no modification to relevance formulas.
    _, original_gallery, _, *_ = toy_protocol
    data=tmp_path/'data';data.mkdir()
    gallery = [original_gallery[0], original_gallery[1],
               {**original_gallery[1], 'image_id': 'c'}]
    gallery = [{**row, 'person_ids': ['identity']} for row in gallery]
    queries = [{'query_id': 'q', 'image_id': 'a', 'text': 'desired people', 'subjects': [],
                'target_ids': ['identity'], 'full_positive_ids': ['b','c'], 'case': 'GROUP'}]
    for name,rows in [('gallery', gallery), ('queries', queries)]:
        (data/f'{name}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    g,q,index=common.public_inputs(gallery, queries)
    cfg=config('siglip2_compose')
    monkeypatch.setattr(runner, 'feature_cache', lambda stage, fp, runtime: StageCache(stage, fp, write_root=tmp_path/'cache'))
    class Backend:
        def images(self, images):
            return np.asarray([np.asarray(image).mean(axis=(0,1)) for image in images], np.float32)
        def queries(self, images, texts, reference_features=None):
            return np.array([[0,0,1]], np.float32)
    out=tmp_path/'runs'/'siglip2_compose';out.mkdir(parents=True)
    runner.run_scene(cfg,g,q,index,{'images':'same'},'cpu',out,backend_factory=lambda *_:Backend())
    score=np.load(out/'scores.npy')
    assert score.shape == (1,3) and np.isfinite(score).all()
    assert score[0,0] == 0  # Raw reference entry is finite and preserved.
    atomic_json(out/'run.json',{'method':'siglip2_compose','higher_is_better':True,
                              'data_fingerprint':manifest_fingerprint(tmp_path)})
    for key,value in {'ROOT':tmp_path,'DATA_DIR':data,'RUNS_DIR':tmp_path/'runs','OUTPUTS_DIR':tmp_path/'outputs'}.items():
        monkeypatch.setattr(evaluate,key,value)
    monkeypatch.setattr('sys.argv',['evaluate.py','--method','siglip2_compose'])
    evaluate.main()
    metrics=json.loads((tmp_path/'outputs/siglip2_compose/metrics.json').read_text())
    assert metrics['overall']['Full-mAP'] == 1.0
    assert metrics['overall']['ID-mAP'] == 1.0
    monkeypatch.setattr(build_tables,'ROOT',tmp_path)
    monkeypatch.setattr(build_tables,'OUTPUTS_DIR',tmp_path/'outputs')
    assert len(build_tables.collect_results()) == 1
