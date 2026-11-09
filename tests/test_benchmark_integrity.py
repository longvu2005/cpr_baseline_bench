"""Regressions for score provenance, public inputs and completed-run status."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

import build_tables
import evaluate
import run_baseline
from benchmark_data import gallery_image_digest, manifest_fingerprint
from cpr_person.cache import atomic_json, canonical_hash, sha256_file
from cpr_person.protocol import require_score_artifact

ROOT = Path(__file__).resolve().parents[1]


def functions(path, names, namespace=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    assert len(nodes) == len(names)
    ns = {'np': np, 'Path': Path, 'Any': object, 'Sequence': list,
          'progress_bar': lambda values, **_: values, **(namespace or {})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec',
                 dont_inherit=False), ns)
    return ns


def test_scene_cache_tracks_same_size_image_changes_with_preserved_mtime(tmp_path, monkeypatch):
    from cpr_baselines import scene_clip
    monkeypatch.setattr(scene_clip, 'verify_scene_clip', lambda *_: 'a'*64)
    monkeypatch.setattr(scene_clip, 'scene_identity', lambda *_: {'assets_sha256':'a'*64})
    monkeypatch.setattr(scene_clip, 'ROOT', tmp_path)
    image = tmp_path / 'image'; image.write_bytes(b'old pixels')
    checkpoint = tmp_path / 'clip.pt'; checkpoint.write_bytes(b'checkpoint')
    gallery = tmp_path / 'gallery.jsonl'
    gallery.write_text(json.dumps({'image_id': 'i', 'path': 'image'}) + '\n')
    before = image.stat()
    first = scene_clip.gallery_fingerprint(checkpoint, gallery, 'ViT-B/32')
    image.write_bytes(b'new pixels')
    os.utime(image, ns=(before.st_atime_ns, before.st_mtime_ns))
    second = scene_clip.gallery_fingerprint(checkpoint, gallery, 'ViT-B/32')
    assert image.stat().st_size == before.st_size
    assert first != second


def test_wiser_verifier_excludes_reference_before_top_k_and_refinement(tmp_path):
    calls = []
    gallery = [{'image_id': name, 'path': tmp_path / name} for name in ('self', 'a', 'b')]
    query = {'image_id': 'self', 'text': 'red jacket'}
    cfg = {'wiser': {'topk_per_path': 2, 'confidence_threshold': .7},
           'adaptation': {'query_text_field': 'text'}, 'cache': {'verifier_dir': str(tmp_path)}}
    def score(reference, instruction, candidate):
        calls.append(candidate)
        return .9 if candidate.name == 'self' else .2
    ns = functions('methods/published/wiser/run.py',
                   ['eligible_top', 'verify_candidates'], {
        'verify_cache_path': lambda *_: tmp_path / 'verify.jsonl',
        'gallery_path': lambda row, _: row['path'],
        'load_jsonl': lambda path: [json.loads(line) for line in path.read_text().splitlines()],
        'json': json, 'os': os,
    })
    ranked = np.array([[0, 1, 2]])
    result = ns['verify_candidates'](cfg, SimpleNamespace(score=score), [query], gallery,
                                     {'self': 0, 'a': 1, 'b': 2}, ranked, ranked, 0)
    assert [p.name for p in calls] == ['a', 'b', 'a', 'b']
    assert result[2] == [True] and result[3] == [True]
    calls.clear()
    ns['verify_candidates'](cfg, SimpleNamespace(score=score), [query], gallery,
                            {'self': 0, 'a': 1, 'b': 2}, ranked, ranked, 0)
    assert calls == []


def test_wiser_cache_namespace_tracks_images_config_and_prepared_models(tmp_path):
    module = runpy.run_path(str(ROOT / 'methods/published/wiser/cache_protocol.py'))
    namespace = module['namespace_caches']
    image = tmp_path / 'image'; image.write_bytes(b'old')
    gallery = tmp_path / 'gallery.jsonl'
    gallery.write_text(json.dumps({'image_id': 'i', 'path': 'image'}) + '\n')
    queries = tmp_path / 'queries.jsonl'; queries.write_text('{}\n')
    marker = tmp_path / 'prepared.json'; marker.write_text('{}')
    cfg = {'data': {'gallery_manifest': str(gallery), 'query_manifest': str(queries)},
           'checkpoint': {'prepared_marker': str(marker)},
           'cache': {'dir': 'runs/wiser/cache', 'verifier_dir': 'runs/wiser/cache/verifier'}}
    first, _ = namespace(cfg, tmp_path)
    image.write_bytes(b'new')
    second, _ = namespace(cfg, tmp_path)
    assert first['cache'] != second['cache']
    marker.write_text('{"model": "new"}')
    third, _ = namespace(cfg, tmp_path)
    assert second['cache'] != third['cache']
    assert cfg['cache']['dir'] == 'runs/wiser/cache'


def test_source_score_reuse_rejects_new_values_with_same_shape(tmp_path):
    path = tmp_path / 'scores.npy'; np.save(path, np.array([[1., 2.]], np.float32))
    run = {'scores_sha256': sha256_file(path)}
    require_score_artifact(run, path)
    np.save(path, np.array([[2., 1.]], np.float32))
    with pytest.raises(RuntimeError, match='Source scores differ'):
        require_score_artifact(run, path)


def test_person_cache_keeps_precision_but_not_matching_or_score_chunks(tmp_path):
    import yaml
    from cpr_person.cache import feature_config_hash
    path = tmp_path / 'config.yaml'
    cfg = {'method': 'person', 'runtime': {'feature_cache_dtype': 'float16', 'score_chunk': 100},
           'setmatch': {'unmatched_score': -1}}
    path.write_text(yaml.safe_dump(cfg)); original = feature_config_hash(path)
    cfg['setmatch']['unmatched_score'] = -.5
    cfg['runtime']['score_chunk'] = 200
    path.write_text(yaml.safe_dump(cfg))
    assert feature_config_hash(path) == original
    cfg['runtime']['feature_cache_dtype'] = 'float32'
    path.write_text(yaml.safe_dump(cfg))
    assert feature_config_hash(path) != original


def test_evaluation_binds_scores_and_rejects_row_order_changes(tmp_path, monkeypatch, write_benchmark_views):
    from PIL import Image
    Image.new('RGB', (2, 2), 'red').save(tmp_path / 'ref.png')
    Image.new('RGB', (2, 2), 'blue').save(tmp_path / 'target.png')
    gallery = [{'image_id': name, 'path': str(tmp_path / (name + '.png')), 'person_ids': ['id']}
               for name in ('ref', 'target')]
    queries = [{'query_id': 'q', 'image_id': 'ref', 'case': 'INDIVIDUAL', 'target_ids': ['id'],
                'full_positive_ids': ['target']}]
    write_benchmark_views(tmp_path, gallery, queries)
    method = 'siglip2_compose'; directory = tmp_path / 'runs' / method; directory.mkdir(parents=True)
    path = directory / 'scores.npy'; np.save(path, np.array([[9., 1.]], np.float32))
    run = {'result_protocol': evaluate.RESULT_PROTOCOL, 'implementation': {'evaluate.py': sha256_file(ROOT/'evaluate.py')}, 'method': method, 'higher_is_better': True, 'data_fingerprint': manifest_fingerprint(tmp_path),
           'image_provenance': {'gallery_image_content_sha256': gallery_image_digest(tmp_path, gallery)},
           'scores_sha256': sha256_file(path), 'gallery_image_ids': ['ref', 'target'], 'query_ids': ['q']}
    atomic_json(directory / 'run.json', run)
    for name, value in {'ROOT': tmp_path, 'RUNS_DIR': tmp_path / 'runs',
                        'OUTPUTS_DIR': tmp_path / 'outputs'}.items():
        monkeypatch.setattr(evaluate, name, value)
    monkeypatch.setattr('sys.argv', ['evaluate.py', '--method', method])
    evaluate.main()
    output = tmp_path / 'outputs' / method
    metrics = json.loads((output / 'metrics.json').read_text())
    evaluated_run = json.loads((output / 'run.json').read_text())
    assert metrics['run_metadata_sha256'] == canonical_hash(evaluated_run)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    np.save(path, np.array([[1., 9.]], np.float32))
    with pytest.raises(ValueError, match='Score bytes differ'):
        evaluate.main()
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    run['scores_sha256'] = sha256_file(path)
    run['gallery_image_ids'] = ['target', 'ref']
    atomic_json(directory / 'run.json', run)
    with pytest.raises(ValueError, match='ordering'):
        evaluate.main()


@pytest.mark.parametrize('all_methods', [False, True])
def test_checkpoint_unavailable_is_nonzero(monkeypatch, tmp_path, all_methods):
    monkeypatch.setattr('sys.argv', ['run_baseline.py', '--all'] if all_methods
                        else ['run_baseline.py', 'clip_image'])
    monkeypatch.setattr(run_baseline, 'check_table_builder', lambda: None)
    monkeypatch.setattr(run_baseline, 'run_pipeline', lambda *_: False)
    # --all writes batch status and removes blocked dependents' run.json under
    # ROOT; never let a test touch real results in the checkout.
    methods = run_baseline.discover_methods()
    monkeypatch.setattr(run_baseline, 'discover_methods', lambda: methods)
    monkeypatch.setattr(run_baseline, 'ROOT', tmp_path)
    with pytest.raises(SystemExit) as error:
        run_baseline.main()
    assert error.value.code != 0
    if not all_methods:
        assert error.value.code == 42


def test_s11_checkpoint_dependency_resolves_existing_adapter():
    module = runpy.run_path(str(ROOT / 'methods/simple/qwen_vl_rerank/download_checkpoint.py'))
    assert module['S7'].is_file()


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.01, 1.01, True, '0.5'])
def test_table_rejects_invalid_numeric_metrics(value):
    with pytest.raises(ValueError):
        build_tables.format_metric(value)
