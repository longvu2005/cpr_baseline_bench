"""Regression coverage for reporting after a scene baseline such as LinCIR."""

import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import build_tables
import run_baseline
from benchmark_data import DATA_PROTOCOL, manifest_fingerprint
from cpr_person import PROTOCOL_VERSION
from cpr_person.protocol import PERSON_METHODS, TEXTFREE_DIAGNOSTIC_METHOD

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reporting_repo(tmp_path, monkeypatch, write_benchmark_views):
    # Copy only reporting code, not adapters, models, images or checkpoints.
    files = ["build_tables.py", "run_baseline.py", "benchmark_data.py",
             "benchmark_progress.py", "prepare_data.py", "cpr_person/__init__.py",
             "cpr_person/protocol.py", "cpr_person/cache.py", "cpr_person/rcr_cache.py"]
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write_benchmark_views(tmp_path, [{"id": "gallery"}], [{"id": "queries"}])
    monkeypatch.setattr(build_tables, "ROOT", tmp_path)
    monkeypatch.setattr(build_tables, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(build_tables, "TABLES_DIR", tmp_path / "tables")
    return tmp_path


def write_result(root, method="lincir", provenance=None):
    output = root / "outputs" / method
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = manifest_fingerprint(root)
    run = {"method": method, "display_name": "LinCIR" if method == "lincir" else method,
           "group": "Published / SOTA Baselines", "cpr_supervision": "No",
           "higher_is_better": True, "data_fingerprint": fingerprint,
           "split": "test", "data_protocol": DATA_PROTOCOL}
    if provenance is not None:
        run["shared_person_protocol"] = provenance
    metrics = {"method": method, "data_fingerprint": fingerprint,
               "split": "test", "data_protocol": DATA_PROTOCOL,
               "overall": {"Macro-Full-mAP": 0.25, **{
                   f"{kind}-{metric}": 0.125 for kind in ("ID", "Full")
                   for metric in ("mAP", "R@1", "R@5", "R@10")
               }},
               "cases": {case: {"Full-mAP": 0.25, "Full-R@1": 0.5}
                         for case in build_tables.CASE_TYPES}}
    (output / "run.json").write_text(json.dumps(run))
    (output / "metrics.json").write_text(json.dumps(metrics))
    return output


def shared_provenance(root):
    fingerprint = manifest_fingerprint(root)
    detection = {"adapter_version": PROTOCOL_VERSION,
                 "gallery_manifest_sha256": fingerprint["gallery_sha256"],
                 "gallery_image_content_sha256": "a" * 64,
                 "protocol": {"backend": "rcr_detection_cache", "threshold": 0.30, "text_threshold": 0.25},
                 "rcr_cache": {"layout": "person-detections-v1", "index_sha256": "1" * 64,
                               "cache_id": "2" * 64, "gallery_ids_sha256": "3" * 64}}
    boxes = "b" * 64
    return {"version": PROTOCOL_VERSION, "detections_fingerprint": detection,
            "detection_array_sha256": boxes, "localization_components_sha256": "c" * 64,
            "localization_fingerprint": {"adapter_version": PROTOCOL_VERSION,
                                         "queries_manifest_sha256": fingerprint["queries_sha256"],
                                         "detections": copy.deepcopy(detection),
                                         "detection_array_sha256": boxes,
                                         "selector": {"membership_threshold": 0.2}}}


def cli(root, script, *args):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, "-S", script, *args], cwd=root, env=env,
                          capture_output=True, text=True, timeout=15)


def test_scene_tables_run_without_site_packages_or_person_adapters(reporting_repo):
    write_result(reporting_repo)
    completed = cli(reporting_repo, "build_tables.py")
    assert completed.returncode == 0, completed.stderr
    assert not (reporting_repo / "cpr_person" / "adapter.py").exists()
    with (reporting_repo / "tables" / "table1_main.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["Method"] == "LinCIR"
    assert rows[0]["Protocol"] == "Native scene v1"
    assert rows[0]["ID-mAP"] == "12.50"
    assert rows[0]["Macro-Full-mAP"] == "25.00"
    with (reporting_repo / "tables" / "table2_cases.csv").open() as handle:
        cases = list(csv.DictReader(handle))
    assert cases[0]["GROUP Full-mAP"] == "25.00"
    assert cases[0]["RELATIONAL Full-R@1"] == "50.00"


def test_check_needs_neither_manifests_nor_evaluated_results(reporting_repo):
    shutil.rmtree(reporting_repo / "data")
    completed = cli(reporting_repo, "build_tables.py", "--check")
    assert completed.returncode == 0, completed.stderr
    assert "imports OK" in completed.stdout
    assert not (reporting_repo / "tables").exists()


def test_tables_only_reuses_metrics_without_method_discovery_or_inference(reporting_repo):
    output = write_result(reporting_repo)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    completed = cli(reporting_repo, "run_baseline.py", "--tables-only")
    assert completed.returncode == 0, completed.stderr
    assert "LinCIR" in completed.stdout
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}
    assert not (reporting_repo / "runs").exists()
    assert not (reporting_repo / "checkpoints").exists()
    assert not (reporting_repo / "methods").exists()


def test_missing_reporting_module_fails_before_gallery_or_model_work(reporting_repo):
    method = reporting_repo / "methods" / "published" / "05_lincir"
    method.mkdir(parents=True)
    (method / "config.yaml").write_text("method: lincir\n")
    (method / "requirements.txt").write_text("")
    (method / "run.py").write_text("raise AssertionError('inference must not run')\n")
    (reporting_repo / "cpr_person" / "protocol.py").unlink()
    completed = cli(reporting_repo, "run_baseline.py", "05_lincir", "--skip-install")
    assert completed.returncode != 0
    assert "No module named 'cpr_person.protocol'" in completed.stderr
    assert "Validate gallery data" not in completed.stdout
    assert "inference must not run" not in completed.stderr
    assert not (reporting_repo / "tables").exists()


def test_all_mode_checks_reporting_once_before_starting_methods(monkeypatch):
    methods = run_baseline.discover_methods()
    events = []
    monkeypatch.setattr(sys, "argv", ["run_baseline.py", "--all", "--skip-install"])
    monkeypatch.setattr(run_baseline, "check_table_builder", lambda: events.append("check"))
    monkeypatch.setattr(run_baseline, "run_pipeline",
                        lambda method, *args: events.append(method.method_id))
    run_baseline.main()
    assert events == ["check"] + [method.method_id for method in methods]


def test_evaluated_scene_result_requires_inference_manifest_identity(reporting_repo):
    output = write_result(reporting_repo)
    run = json.loads((output / "run.json").read_text())
    run.pop("data_fingerprint")
    (output / "run.json").write_text(json.dumps(run))
    with pytest.raises(RuntimeError, match="Run metadata does not match"):
        build_tables.collect_results()


def test_scene_and_same_protocol_person_runs_share_a_table(reporting_repo):
    write_result(reporting_repo)
    for method in ("fafa_setmatch", "basic_setmatch"):
        write_result(reporting_repo, method, shared_provenance(reporting_repo))
    results = build_tables.collect_results()
    assert len(results) == 3
    assert sum(result["protocol"] == "Shared RCR boxes" for result in results) == 2


@pytest.mark.parametrize("provenance", [None, {}, {"version": "legacy"}])
def test_legacy_person_results_do_not_block_a_valid_scene_result(reporting_repo, provenance):
    write_result(reporting_repo)
    write_result(reporting_repo, "fafa_setmatch", provenance)
    assert [result["method_id"] for result in build_tables.collect_results()] == ["lincir"]


@pytest.mark.parametrize("change,error", [
    ("images", "different gallery image bytes"),
    ("boxes", "different actual detector boxes"),
    ("membership", "different selected people"),
    ("selector", "different selected people"),
])
def test_person_runs_with_different_shared_inputs_cannot_be_compared(reporting_repo, change, error):
    first = shared_provenance(reporting_repo)
    second = copy.deepcopy(first)
    if change == "images":
        second["detections_fingerprint"]["gallery_image_content_sha256"] = "d" * 64
        second["localization_fingerprint"]["detections"] = copy.deepcopy(second["detections_fingerprint"])
    elif change == "boxes":
        second["detection_array_sha256"] = "d" * 64
        second["localization_fingerprint"]["detection_array_sha256"] = "d" * 64
    elif change == "membership":
        second["localization_components_sha256"] = "d" * 64
    else:
        second["localization_fingerprint"]["selector"]["membership_threshold"] = 0.3
    write_result(reporting_repo, "basic_setmatch", first)
    write_result(reporting_repo, "fafa_setmatch", second)
    with pytest.raises(RuntimeError, match=error):
        build_tables.collect_results()


@pytest.mark.parametrize("change,error", [
    ("detections_fingerprint", "Missing shared detector/localization fingerprints"),
    ("localization_fingerprint", "Missing shared detector/localization fingerprints"),
    ("detection_array_sha256", "detector boxes checksum"),
    ("localization_components_sha256", "selected membership checksum"),
    ("images", "gallery image bytes checksum"),
    ("gallery", "does not match the evaluated manifests"),
    ("queries", "does not match the evaluated manifests"),
    ("version", "fingerprint version mismatch"),
    ("linked_detection", "refers to different detector boxes"),
    ("linked_boxes", "refers to different detector boxes"),
])
def test_incomplete_or_inconsistent_current_person_provenance_is_rejected(reporting_repo, change, error):
    provenance = shared_provenance(reporting_repo)
    if change in {"detections_fingerprint", "localization_fingerprint",
                  "detection_array_sha256", "localization_components_sha256"}:
        provenance.pop(change)
    elif change == "images":
        provenance["detections_fingerprint"]["gallery_image_content_sha256"] = None
    elif change == "gallery":
        provenance["detections_fingerprint"]["gallery_manifest_sha256"] = "d" * 64
    elif change == "queries":
        provenance["localization_fingerprint"]["queries_manifest_sha256"] = "d" * 64
    elif change == "version":
        provenance["localization_fingerprint"]["adapter_version"] = "legacy"
    elif change == "linked_detection":
        provenance["localization_fingerprint"]["detections"] = {}
    else:
        provenance["localization_fingerprint"]["detection_array_sha256"] = "d" * 64
    write_result(reporting_repo, "fafa_setmatch", provenance)
    with pytest.raises(RuntimeError, match=error):
        build_tables.collect_results()


def test_diagnostic_is_excluded_from_primary_comparison(reporting_repo):
    write_result(reporting_repo)
    write_result(reporting_repo, TEXTFREE_DIAGNOSTIC_METHOD)
    assert [result["method_id"] for result in build_tables.collect_results()] == ["lincir"]
    assert TEXTFREE_DIAGNOSTIC_METHOD not in PERSON_METHODS


@pytest.mark.parametrize("file,key,value,error", [
    ("run.json", "method", "clip_image", "Method mismatch"),
    ("run.json", "data_fingerprint", {}, "Run metadata does not match"),
])
def test_evaluated_run_metadata_must_match_metrics(reporting_repo, file, key, value, error):
    output = write_result(reporting_repo)
    path = output / file
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises((ValueError, RuntimeError), match=error):
        build_tables.collect_results()


def test_missing_adjacent_run_metadata_has_a_clear_error(reporting_repo):
    output = write_result(reporting_repo)
    (output / "run.json").unlink()
    with pytest.raises(RuntimeError, match="Missing evaluated run metadata"):
        build_tables.collect_results()


def test_old_manifest_results_do_not_overwrite_existing_tables(reporting_repo, write_benchmark_views):
    write_result(reporting_repo)
    write_benchmark_views(reporting_repo, [{"id": "gallery"}], [{"changed": True}])
    tables = reporting_repo / "tables"
    tables.mkdir()
    old = tables / "table1_main.csv"
    old.write_text("previous table\n")
    completed = cli(reporting_repo, "build_tables.py")
    assert completed.returncode != 0
    assert "No evaluated results match" in completed.stderr
    assert old.read_text() == "previous table\n"


@pytest.mark.parametrize('field,value,error', [
    ('detection_array_sha256', 'b'*63, 'detector boxes checksum'),
    ('localization_components_sha256', 'z'*64, 'selected membership checksum'),
    ('localization_components_sha256', 123, 'selected membership checksum'),
])
def test_person_checksums_must_identify_actual_arrays(reporting_repo, field, value, error):
    provenance = shared_provenance(reporting_repo)
    provenance[field] = value
    write_result(reporting_repo, 'fafa_setmatch', provenance)
    with pytest.raises(RuntimeError, match=error):
        build_tables.collect_results()


def test_current_person_result_requires_selector_provenance(reporting_repo):
    provenance = shared_provenance(reporting_repo)
    provenance['localization_fingerprint']['selector'] = None
    write_result(reporting_repo, 'fafa_setmatch', provenance)
    with pytest.raises(RuntimeError, match='Missing shared selector fingerprint'):
        build_tables.collect_results()


def test_single_method_checks_reporting_before_pipeline(monkeypatch):
    events = []
    monkeypatch.setattr(sys, 'argv', ['run_baseline.py', 'fafa_setmatch', '--skip-install'])
    monkeypatch.setattr(run_baseline, 'check_table_builder', lambda: events.append('check'))
    monkeypatch.setattr(run_baseline, 'run_pipeline', lambda *a: events.append('pipeline'))
    run_baseline.main()
    assert events == ['check', 'pipeline']


@pytest.mark.parametrize('arguments', [
    ('--tables-only', '--all'), ('--tables-only', '--list'),
    ('fafa_setmatch', '--tables-only'),
])
def test_tables_only_cannot_start_methods(reporting_repo, arguments):
    completed = cli(reporting_repo, 'run_baseline.py', *arguments)
    assert completed.returncode != 0
    assert not (reporting_repo/'tables').exists()
    assert not (reporting_repo/'runs').exists()
    assert not (reporting_repo/'checkpoints').exists()


def test_invalid_case_metric_preserves_both_existing_tables(reporting_repo, monkeypatch):
    output = write_result(reporting_repo)
    path = output/'metrics.json'
    metrics = json.loads(path.read_text())
    metrics['cases']['GROUP']['Full-mAP'] = 'invalid metric'
    path.write_text(json.dumps(metrics))
    tables = reporting_repo/'tables';tables.mkdir()
    for name in ('table1_main.csv', 'table2_cases.csv'):
        (tables/name).write_text('previous table\n')
    monkeypatch.setattr(sys, 'argv', ['build_tables.py'])
    with pytest.raises(ValueError):
        build_tables.main()
    assert all((tables/name).read_text() == 'previous table\n'
               for name in ('table1_main.csv', 'table2_cases.csv'))


def test_interrupted_csv_write_preserves_completed_file(tmp_path, monkeypatch):
    path = tmp_path/'table.csv';path.write_text('previous table\n')
    class InterruptedWriter:
        def __init__(self, handle, **kwargs):
            self.handle = handle
        def writeheader(self):
            self.handle.write('partial header\n')
        def writerows(self, rows):
            raise OSError('interrupted')
    monkeypatch.setattr(build_tables.csv, 'DictWriter', InterruptedWriter)
    with pytest.raises(OSError, match='interrupted'):
        build_tables.write_csv(path, ['Method'], [{'Method':'test'}])
    assert path.read_text() == 'previous table\n'
    assert list(tmp_path.iterdir()) == [path]


def test_stale_results_do_not_create_tables_directory(reporting_repo, write_benchmark_views):
    write_result(reporting_repo)
    write_benchmark_views(reporting_repo, [{"id": "gallery"}], [{"changed": True}])
    completed = cli(reporting_repo, 'build_tables.py')
    assert completed.returncode != 0
    assert 'No evaluated results match' in completed.stderr
    assert not (reporting_repo/'tables').exists()


def test_legacy_person_metadata_does_not_block_current_scene_tables(reporting_repo):
    write_result(reporting_repo)
    output = write_result(reporting_repo, 'fafa_setmatch', {'version':'legacy'})
    path = output/'run.json'
    run = json.loads(path.read_text());run['data_fingerprint'] = {}
    path.write_text(json.dumps(run))
    assert [row['method_id'] for row in build_tables.collect_results()] == ['lincir']
