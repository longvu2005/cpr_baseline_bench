import ast
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cpr_person.adapter import load_person_context
from cpr_person.cache import canonical_hash
from cpr_person.cache_tools import import_shared, inspect_roots
from cpr_person.detection import Detections
from cpr_person.validation import ensure_separate_queries
from cpr_person.calibration import freeze, apply_instruct_calibration
from cpr_person.localization import get_localization


def test_verified_cache_import_does_not_run_models(toy_protocol,tmp_path,monkeypatch):
    cfg,gallery,queries,gm,qm,calls,selector=toy_protocol
    p=load_person_context(gallery,queries,gm,qm,"cpu")
    original=tmp_path/"shared"
    monkeypatch.setenv("CPR_SHARED_CACHE_ROOT",str(tmp_path/"new-shared"))
    result=import_shared(original,gm,qm,cfg=cfg)
    assert Path(result["detections"]).is_file() and Path(result["query_localization"]).is_file()
    p2=load_person_context(gallery,queries,gm,qm,"cpu")
    assert p2.localization.members==p.localization.members and len(calls)==1


def test_legacy_cache_not_reused_by_tensor_shape(toy_protocol,tmp_path,monkeypatch):
    cfg,gallery,queries,gm,qm,calls,selector=toy_protocol
    root=tmp_path/"legacy";root.mkdir()
    np.savez(root/"person_detections.npz",offsets=[0,2,3],boxes=np.zeros((3,4)))
    with pytest.raises(ValueError):import_shared(root,gm,qm,cfg=cfg)
    rows=inspect_roots([root])
    assert rows[0]["kind"]=="legacy_or_other_metadata_no_automatic_reuse"
    assert not calls


def test_changed_image_bytes_invalidate_shared_detection(toy_protocol):
    cfg,gallery,queries,gm,qm,calls,selector=toy_protocol
    p=load_person_context(gallery,queries,gm,qm,"cpu")
    Image.new("RGB",(40,30),"blue").save(gallery[0]["path"])
    other=load_person_context(gallery,queries,gm,qm,"cpu")
    assert len(calls)==2 and p.detections.path!=other.detections.path


def test_validation_rejects_relabelled_eval_inputs(toy_protocol,tmp_path):
    cfg,gallery,queries,gm,qm,calls,selector=toy_protocol
    altered=[{**q,"query_id":"different","target_ids":["changed"],"case":"GROUP"} for q in queries]
    val=tmp_path/"val.jsonl";val.write_text("".join(json.dumps(q)+"\n" for q in altered))
    with pytest.raises(ValueError):ensure_separate_queries(qm,val)
    altered[0]["subjects"][0]["modify_text"]="independent validation change"
    val.write_text(json.dumps(altered[0])+"\n")
    ensure_separate_queries(qm,val)


def test_worker_recovers_only_checksummed_ordered_prefix(tmp_path):
    # Exercise the real worker recovery code without importing torch/models.
    source=Path(__file__).resolve().parents[1]/"cpr_person/worker.py"
    tree=ast.parse(source.read_text());names={"recover_prefix","rows_to_detections"}
    module=ast.Module(body=[n for n in tree.body if getattr(n,"name",None) in names],type_ignores=[])
    env={"np":np,"json":json,"Detections":Detections,"canonical_hash":canonical_hash}
    exec(compile(module,str(source),"exec"),env)
    row={"image_id":"a","size":[10,10],"boxes":[[0,0,10,10]],"scores":[-1],"fallback":[True]}
    encoded=json.dumps({**row,"sha256":canonical_hash(row)})+"\n"
    journal=tmp_path/"partial.jsonl";journal.write_text(encoded+'{"truncated"')
    assert env["recover_prefix"](journal,[{"image_id":"a"},{"image_id":"b"}])==[row]
    assert journal.read_text()==encoded
    broken={**row,"scores":[.4]}
    journal.write_text(json.dumps({**broken,"sha256":canonical_hash(broken)})+"\n")
    assert env["recover_prefix"](journal,[{"image_id":"a"}])==[]


def test_external_unmatched_calibration_is_fixed_and_checkpoint_bound(toy_protocol,tmp_path):
    cfg,gallery,queries,gm,qm,calls,selector=toy_protocol
    external=tmp_path/"external.png";Image.new("RGB",(10,10),"black").save(external)
    eg,eq=tmp_path/"external-gallery.jsonl",tmp_path/"external-queries.jsonl"
    eg.write_text(json.dumps({"image_id":"external","path":str(external)})+"\n")
    eq.write_text(json.dumps({"image_id":"external","subjects":[{"select_text":"new","modify_text":"new"}]})+"\n")
    values=tmp_path/"external-scores.npy";np.save(values,np.array([-1.,-4.,-9.]))
    marker=tmp_path/"calibration.json";checkpoint=Path(cfg["detector"]["checkpoint"])
    artifact=freeze(values,eq,eg,checkpoint,"native-commit","independent-dataset",marker,
                    benchmark_queries=qm,benchmark_gallery=gm)
    native={"source":{"commit":"native-commit"},"setmatch":{"unmatched_calibration":str(marker)}}
    apply_instruct_calibration(native,checkpoint,qm,gm)
    assert native["setmatch"]["unmatched_score"]==pytest.approx(artifact["unmatched_score"])
    assert native["setmatch"]["calibration_provenance"]["dataset_id"]=="independent-dataset"
    native["source"]["commit"]="different-model"
    with pytest.raises(ValueError):apply_instruct_calibration(native,checkpoint,qm,gm)


def test_calibration_refuses_eval_images(toy_protocol,tmp_path):
    cfg,gallery,queries,gm,qm,calls,selector=toy_protocol
    eq=tmp_path/"external-queries.jsonl"
    eq.write_text(json.dumps({"image_id":"other","subjects":[{"select_text":"new","modify_text":"new"}]})+"\n")
    scores=tmp_path/"pairs.npy";np.save(scores,np.array([-2.]))
    with pytest.raises(ValueError):freeze(scores,eq,gm,Path(cfg["detector"]["checkpoint"]),"commit","external",
        tmp_path/"b.json",benchmark_queries=qm,benchmark_gallery=gm)


def test_selector_worker_consumes_the_correct_split_and_writes_verified_cache(toy_protocol,tmp_path,monkeypatch):
    from types import SimpleNamespace
    import cpr_person.selector_worker as worker
    from conftest import ToySelector
    cfg,gallery,queries,gm,qm,calls,selector=toy_protocol
    p=load_person_context(gallery,queries,gm,qm,"cpu")
    p.localization.path.unlink()
    p.localization.path.with_name(p.localization.path.name+".meta.json").unlink()
    runtime=tmp_path/"fixed-tokenizer";runtime.mkdir()
    for name,version in cfg["selector"]["tokenizer_packages"].items():
        monkeypatch.setitem(__import__("sys").modules,name,SimpleNamespace(__file__=str(runtime/(name+".py")),__version__=version))
    monkeypatch.setattr(worker,"protocol_config",lambda:cfg)
    monkeypatch.setattr(worker,"validate_tokenizer",lambda selector:runtime)
    monkeypatch.setattr(worker,"ClipSelector",lambda *args:ToySelector())
    def subprocess_run(command,**kwargs):
        assert command[command.index("--gallery-manifest")+1]==str(gm)
        monkeypatch.setattr("sys.argv",["selector_worker",*command[3:]])
        worker.main()
    monkeypatch.setattr("cpr_person.localization.subprocess.run",subprocess_run)
    result=get_localization(queries,qm,gallery,p.detections,"cpu",cfg=cfg)
    assert result.members==p.localization.members and not result.cache_hit
