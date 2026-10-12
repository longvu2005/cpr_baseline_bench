"""CPU adapter contracts. Native neural forwards are deliberately not mocked as GPU passes."""
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import build_tables
import evaluate
from benchmark_data import manifest_fingerprint
from cpr_person.adapter import PERSON_METHODS, expand_targets, load_person_context, require_person_branch, require_manifest_branch
from cpr_person.cache import read_json
from cpr_person.matching import aggregate_people, aggregate_component_scores, score_feature_sets

ROOT=Path(__file__).resolve().parents[1]


def function(path,name,namespace=None):
    tree=ast.parse((ROOT/path).read_text())
    n=next(n for n in tree.body if getattr(n,"name",None)==name)
    n.decorator_list=[]
    # Postponed annotations avoid importing heavy frameworks for CPU adapters.
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),n],type_ignores=[])
    env={"np":np,"os":os,"score_feature_sets":score_feature_sets,"aggregate_people":aggregate_people,
         "progress_bar":lambda values,**kwargs:values,**(namespace or {})}
    exec(compile(ast.fix_missing_locations(module),str(path),"exec"),env)
    return env[name]


DIRECT={
 "instruct_reid_setmatch":("methods/published/03_instruct_reid_setmatch/run.py","negative_squared_euclidean"),
 "groundingdino_clipreid_set":("methods/simple/05_reid_set/run.py","cosine"),
 "per_person_clip_compose_setmatch":("methods/simple/10_person_compose_setmatch/run.py","cosine"),
}


@pytest.mark.parametrize("method",sorted(DIRECT))
def test_adapter_preserves_native_geometry_and_partial_credit(method):
    path,metric=DIRECT[method]
    score=function(path,"score_one_query")
    q=np.array([[1.,0],[0,1.]],np.float32)
    g=np.array([[1.,0],[0,1.],[.2,.3]],np.float32)
    offsets=np.array([0,2,3,3]);counts=np.diff(offsets)
    kwargs={"query_features":q,"offsets":offsets,"counts":counts,
            "person_image_index":np.array([0,0,1]),"unmatched_score":-4.}
    if metric=="negative_squared_euclidean":kwargs.update(gallery_features=g,person_chunk_size=1)
    else:kwargs.update(all_features=g,feature_chunk_size=1)
    raw=q@g.T if metric=="cosine" else -((q[:,None,:]-g[None,:,:])**2).sum(axis=-1)
    np.testing.assert_allclose(score(**kwargs),aggregate_people(raw,offsets,-4))


def test_basic_and_adafocal_keep_their_pair_scores():
    # Official score spaces are passed through unchanged, including negatives.
    s=np.array([[4.,-3.,-2.],[-7.,3.,-1.]])
    offsets=np.array([0,2,3,3]);expected=aggregate_people(s,offsets,-19)
    basic=function("methods/published/08_basic_setmatch/run.py","collapse_setmatch")
    ada=function("methods/published/09_adafocal_setmatch/run.py","aggregate_setmatch")
    np.testing.assert_allclose(basic(s,None,None,offsets,-19),expected)
    np.testing.assert_allclose(ada(pair_scores=s,offsets=offsets,unmatched_score=-19),expected)


@pytest.mark.parametrize("method",["word4per","fafa"])
def test_published_component_output_contract(method,tmp_path):
    directory="01_word4per_setmatch" if method=="word4per" else "02_fafa_setmatch"
    score=function(f"methods/published/{directory}/run.py","score_all_queries", {
        "flatten_query_features":lambda rows:(np.concatenate(rows),np.r_[0,np.cumsum([len(x) for x in rows])]),
        "atomic_save":lambda path,array:np.save(path,array),
        "aggregate_component_scores":aggregate_component_scores,
    })
    q=[np.array([[1.,0],[0,1.]],np.float32),np.array([[1.,0]],np.float32)]
    gallery=np.array([[1.,0],[0,1.],[.3,.6]],np.float32)
    token_gallery=np.stack([gallery,gallery*.9],axis=1)
    gp=tmp_path/"gallery.npy";np.save(gp,gallery if method=="word4per" else token_gallery)
    cp,out=tmp_path/"components.npy",tmp_path/"scores.npy"
    def component_scores(**kwargs):
        features=kwargs["component_features"]
        # Only the heavy tensor forward is substituted. Native score source is
        # independently guarded below, and the real adapter/aggregator executes.
        raw=features@gallery.T if method=="word4per" else np.einsum("qd,pkd->qpk",features,token_gallery).mean(axis=-1)
        np.save(kwargs["cache_path"],raw.astype(np.float32))
        return kwargs["cache_path"]
    score.__globals__["compute_component_person_scores"]=component_scores
    kwargs={"query_features":q,"gallery_feature_path":gp,"offsets":np.array([0,2,3]),
        "component_score_path":cp,"scores_path":out,"person_batch_size":2,"query_batch_size":2,
        "unmatched_score":-1.,"device":"cpu"}
    if method=="fafa":kwargs["fda_k"]=2
    score(**kwargs)
    assert np.load(out).shape==(2,2) and np.isfinite(np.load(out)).all()


def test_s5_s6_s9_share_actual_cached_context_and_score_contract(toy_protocol,tmp_path):
    cfg,gallery,queries,gm,qm,calls,selector=toy_protocol
    person=load_person_context(gallery,queries,gm,qm,"cpu")
    features=np.eye(3,dtype=np.float32)
    index={r["image_id"]:i for i,r in enumerate(gallery)}
    query_sets=[np.stack([features[m["global_person_index"]] if m["global_person_index"] is not None
                         else np.array([0,0,1]) if index[q["image_id"]]==1 else np.array([.707,.707,0])
                         for si,m in person.localization.flat(qi)]) for qi,q in enumerate(queries)]
    bcfg={"setmatch":{"unmatched_score":-1.,"unmatched_score_source":"analytic_cosine_lower_bound"},
          "runtime":{"score_feature_chunk_size":2}}
    one=function("methods/simple/05_reid_set/run.py","score_one_query")
    compute=function("methods/simple/05_reid_set/run.py","compute_scores",{"score_one_query":one})
    def prepare_branch(**kwargs):
        p=load_person_context(gallery,queries,gm,qm,"cpu")
        return p,features,query_sets,{"shared_person_protocol":p.metadata()}
    s5=SimpleNamespace(ADAPTER_VERSION="test-native", prepare_person_branch=prepare_branch,
        device_from=lambda value:value,
        compute_scores=compute, query_gallery_indices=lambda qs,idx:np.array([idx[q["image_id"]] for q in qs]),
        build_gallery_index=lambda rows:{r["image_id"]:i for i,r in enumerate(rows)},
        meta_path=lambda p:p.with_suffix(p.suffix+".meta.json"))
    output=tmp_path/"s5.npy"
    got=compute(cfg=bcfg,queries=queries,query_indices=np.array([0,1,0]),features=features,
                offsets=person.detections.offsets,output_path=output,query_features_list=query_sets)
    assert got.shape==(len(queries),len(gallery)) and np.isfinite(got).all()
    six=function("methods/simple/06_reid_set_text/run.py","prepare_s5_scores", {
        "read_json":read_json,"meta_path":s5.meta_path,
        "write_json":lambda p,x:p.write_text(json.dumps(x)),
        "validate_score_matrix":lambda arr,shape,label:evaluate.validate_score_matrix(arr,shape)})
    s6,stats=six(s5=s5,s5_cfg=bcfg,s5_cfg_path=gm,gallery_manifest=gm,query_manifest=qm,
        detection_cache=tmp_path/"obsolete.npz",feature_cache=tmp_path/"features.npy",
        score_cache=tmp_path/"s6.npy",device="cpu")
    np.testing.assert_array_equal(s6,got)
    assert len(calls)==1 and stats["shared_person_protocol"]["version"]
    require_person_branch({"shared_person_protocol":person.metadata()},gm,qm)
    with pytest.raises(RuntimeError):require_person_branch({"shared_person_protocol":{}},gm,qm)
    basecfg=tmp_path/"base.yaml";basecfg.write_text(yaml.safe_dump(bcfg))
    nine=function("methods/simple/09_reid_qwen_proxy/run.py","build_validation_reid_scores", {
        "resolve_path":lambda v:Path(v),"load_yaml":lambda p:yaml.safe_load(p.read_text()),"read_json":read_json,
        "load_cached_scores":lambda *args:None,
        "save_scores":lambda path,arr,fp:arr})
    result=nine(cfg={"base_set":{"config":str(basecfg)},"runtime":{"device":"cpu"},
                    "cache":{"validation":{"reid_features":str(tmp_path/"vf.npy"),"reid_scores":str(tmp_path/"s9.npy")}}},
                config_path=basecfg,val_gallery_manifest=gm,val_query_manifest=qm,
                val_gallery=gallery,val_queries=queries,s5=s5,tracker=None)
    np.testing.assert_array_equal(result,got)


def test_expanded_members_keep_native_subject_texts(toy_protocol):
    cfg,gallery,queries,gm,qm,calls,selector=toy_protocol
    p=load_person_context(gallery,queries,gm,qm,"cpu")
    targets=[[SimpleNamespace(modify_text=f"subject-{si}") for si,s in enumerate(q["subjects"])] for q in queries]
    expanded=expand_targets(targets,p.localization)
    assert [t.modify_text for t in expanded[0]]==["subject-0","subject-0"]
    assert [t.modify_text for t in expanded[2]]==["subject-0","subject-1","subject-2"]


def test_official_evaluator_and_table_builder_accept_adapter_scores(toy_protocol,tmp_path,monkeypatch):
    cfg,gallery,queries,gm,qm,calls,selector=toy_protocol
    person=load_person_context(gallery,queries,gm,qm,"cpu")
    data=tmp_path/"data";data.mkdir()
    gallery=[{**g,"person_ids":["id"]} for g in gallery]
    queries=[{**q,"query_id":str(i),"target_ids":["id"],
              "full_positive_ids":["b" if q["image_id"]=="a" else "a"],"case":"GROUP"} for i,q in enumerate(queries)]
    for name,rows in [("gallery",gallery),("queries",queries)]:
        (data/f"{name}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    person=load_person_context(gallery,queries,data/"gallery.jsonl",data/"queries.jsonl","cpu")
    runs,outputs=tmp_path/"runs",tmp_path/"outputs"
    run_dir=runs/"fafa_setmatch";run_dir.mkdir(parents=True)
    np.save(run_dir/"scores.npy",np.array([[1,.2],[.3,1],[1,.1]],np.float32))
    # Use the official evaluator exactly as written; only its filesystem roots
    # point at the tiny test fixture. Positive sets and formulas are untouched.
    (run_dir/"run.json").write_text(json.dumps({"method":"fafa_setmatch","higher_is_better":True,
        "shared_person_protocol":person.metadata(),"data_fingerprint":manifest_fingerprint(tmp_path)}))
    for key,val in {"ROOT":tmp_path,"DATA_DIR":data,"RUNS_DIR":runs,"OUTPUTS_DIR":outputs}.items():
        monkeypatch.setattr(evaluate,key,val)
    monkeypatch.setattr("sys.argv",["evaluate.py","--method","fafa_setmatch"])
    evaluate.main()
    metrics=json.loads((outputs/"fafa_setmatch"/"metrics.json").read_text())
    assert metrics["overall"]["Full-mAP"]==1.0
    monkeypatch.setattr(build_tables,"ROOT",tmp_path);monkeypatch.setattr(build_tables,"OUTPUTS_DIR",outputs)
    result=build_tables.collect_results()
    assert len(result)==1 and result[0]["protocol"]=="Shared person v1"


def test_all_configs_and_execution_paths_are_integrated():
    discovered={}
    for p in (ROOT/"methods").glob("*/*/config.yaml"):
        cfg=yaml.safe_load(p.read_text());discovered[cfg["method"]]=(p,cfg)
    assert PERSON_METHODS<=discovered.keys()
    for method in PERSON_METHODS:
        p,cfg=discovered[method]
        assert cfg["person_protocol"]=="cpr_person/protocol.yaml"
        source=p.with_name("run.py").read_text()
        if method not in {"groundingdino_clipreid_set_text","reid_set_qwen_edit_proxy"}:
            assert "load_person_context(" in source
            assert "aggregate_people(" in source or "score_feature_sets(" in source or "aggregate_component_scores(" in source
        else:
            assert "prepare_person_branch(" in source
        assert "shared_person_protocol" in source
        assert "fasterrcnn" not in source and "strict_minimum" not in source


def test_requirement_and_dependency_paths_exist():
    for p in (ROOT/"methods").glob("*/*/requirements.txt"):
        for line in p.read_text().splitlines():
            if line.startswith("-r "):assert (p.parent/line[3:]).resolve().is_file()
    for p in (ROOT/"methods").glob("*/*/config.yaml"):
        def walk(value):
            if isinstance(value,dict):
                for v in value.values():walk(v)
            elif isinstance(value,list):
                for v in value:walk(v)
            elif isinstance(value,str) and value.startswith("methods/"):
                assert (ROOT/value).exists(),(p,value)
        walk(yaml.safe_load(p.read_text()))


def test_native_encoders_composition_and_scores_remain_unchanged():
    signatures=json.loads((ROOT/"tests/native_architecture_signatures.json").read_text())
    for path,names in signatures.items():
        tree=ast.parse((ROOT/path).read_text())
        for name,digest in names.items():
            node=next(n for n in tree.body if getattr(n,"name",None)==name)
            assert hashlib.sha256(ast.dump(node).encode()).hexdigest()==digest,(path,name)


def test_proxy_branch_cannot_reuse_scores_from_different_manifests(toy_protocol):
    from cpr_person.cache import sha256_file
    cfg,gallery,queries,gm,qm,calls,selector=toy_protocol
    expected={"gallery_sha256":sha256_file(gm),"queries_sha256":sha256_file(qm)}
    require_manifest_branch({"data_fingerprint":expected},gm,qm)
    with pytest.raises(RuntimeError):require_manifest_branch({"data_fingerprint":{}},gm,qm)
