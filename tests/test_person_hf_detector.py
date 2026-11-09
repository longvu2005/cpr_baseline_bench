"""HF detector, floating-box and runtime boundaries; no weights or network."""
from __future__ import annotations

import ast
import builtins
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from baseline_environment import _requirements_digest
from cpr_person import assets, runtime, worker
from cpr_person.cache import StageCache, atomic_json
from cpr_person.detection import (BoxCandidate, crop_person, detection_fingerprint,
                                  get_detections, protocol_config, save_detections)
from cpr_person.protocol import PERSON_METHODS

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_rcr_configuration_and_native_dependencies():
    cfg = protocol_config()
    d = cfg['detector']
    assert (d['backend'], d['model'], d['text_prompt'], d['threshold'], d['text_threshold']) == (
        'huggingface_grounding_dino', 'IDEA-Research/grounding-dino-tiny', 'person', .30, .25)
    assert d['revision'] == d['processor_revision'] == 'a2bb814dd30d776dcf7e30523b00659f4f141c71'
    assert not {'nms_iou_threshold', 'min_box_size_px', 'max_persons_per_image', 'box_threshold'} & d.keys()
    _, native = _requirements_digest(ROOT/'methods/published/02_fafa_setmatch/requirements.txt')
    _, detector = _requirements_digest(ROOT/'cpr_person/detector-requirements.txt')
    assert 'transformers==4.39.3' in native and 'transformers==4.56.2' not in native
    assert 'transformers==4.56.2' in detector and 'transformers==4.39.3' not in detector
    assert set(d['runtime_versions'].items()) <= {
        tuple(line.split('==')) for line in detector.splitlines() if '==' in line and not line.startswith('#')}
    assert PERSON_METHODS == __import__('cpr_person.adapter', fromlist=['PERSON_METHODS']).PERSON_METHODS


def test_auto_model_and_processor_use_same_pinned_model_offline(monkeypatch):
    calls = []
    model = SimpleNamespace(to=lambda device: model, eval=lambda: model)
    def load_model(name, **kwargs):
        calls.append(('model', name, kwargs)); return model
    processor = object()
    def load_processor(name, **kwargs):
        calls.append(('processor', name, kwargs)); return processor
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
        AutoModelForZeroShotObjectDetection=SimpleNamespace(from_pretrained=load_model)))
    monkeypatch.setattr(runtime, 'check_runtime', lambda d: {})
    monkeypatch.setattr(assets, 'require_snapshot', lambda d: {})
    d = protocol_config()['detector']
    assert assets.load_detector(d, 'cpu') == (model, processor)
    assert [c[:2] for c in calls] == [('processor', d['model']), ('model', d['model'])]
    for _, _, kwargs in calls:
        assert kwargs['local_files_only'] is True
        assert kwargs['revision'] == d['revision']


def test_fractional_unclipped_pil_crops_match_fafa_and_rcr():
    image = Image.fromarray(np.arange(20*30*3, dtype=np.uint8).reshape(20,30,3))
    box = (-1.25, 2.5, 10.5, 18.75)
    got = crop_person(image, box)
    expected = image.crop(box)
    np.testing.assert_array_equal(got, expected)
    assert got.size == (11,17)  # PIL nearest/even, with black out-of-image padding.
    assert not np.any(np.asarray(got)[:,0])


def test_hf_call_label_filtering_order_confidence_and_no_post_filters():
    torch = pytest.importorskip('torch')
    image = Image.new('RGB', (40,30))
    boxes = torch.tensor([[2.25,1.5,3.0,2.25], [-1.,0.,10.25,31.5], [0,0,40,30]] * 5)
    scores = torch.tensor([.45,.95,.8] * 5)
    labels = ['person','person','a person'] * 5
    calls = []
    class Batch(dict):
        input_ids = torch.tensor([[101,102]])
        def to(self, device):
            calls.append(('device', device)); return self
    class Processor:
        def __call__(self, **kwargs):
            calls.append(('inputs', kwargs)); return Batch()
        def post_process_grounded_object_detection(self, outputs, **kwargs):
            calls.append(('post', kwargs))
            return [{'boxes':boxes, 'scores':scores, 'text_labels':labels}]
    b, s = worker.detect_image(lambda **kw: object(), Processor(), image, protocol_config()['detector'], 'cpu')
    keep = [i for i, label in enumerate(labels) if label == 'person']
    np.testing.assert_array_equal(b, boxes[keep])
    np.testing.assert_array_equal(s, scores[keep])
    assert len(b) == 10 and b[0,0] == 2.25 and s[0] < s[1]
    assert calls[0][1]['text'] == [['person']]
    post = calls[-1][1]
    assert post['threshold'] == .3 and post['text_threshold'] == .25
    assert post['target_sizes'] == [(30,40)]


def test_uncapped_fractional_overlapping_detections_validate():
    image = Image.new('RGB',(40,30))
    boxes = np.array([[-.25,.5,10.75,31.25]] * 17, np.float32)
    scores = np.linspace(.4,.95,17,dtype=np.float32)  # Ascending is legal model order.
    d = worker.rows_to_detections([worker.detection_row('g',image,boxes,scores)],
                                 {'protocol':protocol_config()['detector']})
    d.validate(1)
    assert len(d.candidates()[0]) == 17
    np.testing.assert_array_equal(d.boxes, boxes)
    np.testing.assert_array_equal(d.confidences, scores)


def test_zero_detection_full_image_is_explicit_fallback():
    row = worker.detection_row('g', Image.new('RGB',(40,30)), np.empty((0,4)), np.empty(0))
    assert row == {'image_id':'g','size':[40,30], 'boxes':[[0,0,40,30]],'scores':[-1], 'fallback':[True]}
    worker.rows_to_detections([row],{'protocol':protocol_config()['detector']}).validate(1)


@pytest.mark.parametrize('key,value', [
    ('model','example/other-model'), ('revision','f'*40), ('processor_revision','e'*40),
    ('threshold',.31), ('text_threshold',.26), ('text_prompt','human'),
    ('runtime_versions',{'transformers':'4.57.0'}), ('attention_backend','different'),
])
def test_fingerprint_tracks_detector_processing_provenance(toy_protocol,key,value):
    cfg, gallery, queries, gm, qm, calls, selector = toy_protocol
    before = detection_fingerprint(gm,cfg)
    changed = copy.deepcopy(cfg); changed['detector'][key] = value
    assert before != detection_fingerprint(gm,changed)


def test_old_swin_cache_cannot_be_reused(toy_protocol,monkeypatch):
    cfg,gallery,queries,gm,qm,calls,selector = toy_protocol
    fp = detection_fingerprint(gm,cfg)
    legacy = copy.deepcopy(fp)
    legacy['adapter_version'] = '2026-10-12-person-v1'
    legacy['protocol']['backend'] = 'groundingdino_swint_ogc'
    cache = StageCache('detections', legacy)
    row = worker.detection_row('a',Image.new('RGB',(40,30)),np.empty((0,4)),np.empty(0))
    detections = worker.rows_to_detections([row,row],legacy)
    save_detections(cache,detections)
    monkeypatch.setattr(runtime,'detector_python',lambda:pytest.fail('cache-only must not select a worker'))
    with pytest.raises(RuntimeError,match='cache guard'):
        get_detections(gallery,gm,'cpu',cfg=cfg,require_cache=True)
    with pytest.raises(ValueError,match='Retired'):
        detections.validate(2)


def test_cached_read_needs_neither_model_assets_nor_transformers(toy_protocol,monkeypatch):
    from cpr_person.adapter import load_person_context
    cfg,gallery,queries,gm,qm,calls,selector = toy_protocol
    context = load_person_context(gallery,queries,gm,qm,'cpu')
    Path(cfg['detector']['checkpoint']).unlink()
    real_import = builtins.__import__
    def blocked(name,*args,**kwargs):
        if name.split('.')[0] in {'torch','transformers','groundingdino'}:
            pytest.fail('a cache consumer imported detector runtime')
        return real_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',blocked)
    monkeypatch.setenv('CPR_DETECTOR_PYTHON','/missing/detector/python')
    got = get_detections(gallery,gm,'cpu',cfg=cfg,require_cache=True)
    assert got.path == context.detections.path and got.cache_hit and len(calls) == 1


def test_miss_uses_explicit_worker_python_and_scrubs_native_cache(toy_protocol,monkeypatch,tmp_path):
    cfg,gallery,queries,gm,qm,calls,selector = toy_protocol
    python = tmp_path/'isolated/bin/python';python.parent.mkdir(parents=True);python.touch()
    monkeypatch.setenv('CPR_DETECTOR_PYTHON',str(python))
    monkeypatch.setenv('HF_HOME','/native/fafa/cache')
    def launch(command,**kwargs):
        assert command[0] == str(python) and command[2] == 'cpr_person.worker'
        assert 'HF_HOME' not in kwargs['env'] and kwargs['env']['PYTHONNOUSERSITE'] == '1'
        fp = detection_fingerprint(gm,cfg)
        row = worker.detection_row('a',Image.new('RGB',(40,30)),np.empty((0,4)),np.empty(0))
        save_detections(StageCache('detections',fp),worker.rows_to_detections([row,row],fp))
    monkeypatch.setattr('cpr_person.detection.subprocess.run',launch)
    assert not get_detections(gallery,gm,'cpu',cfg=cfg).cache_hit


def test_old_fafa_runtime_is_rejected_by_detector_not_upgraded(monkeypatch):
    d = protocol_config()['detector']
    monkeypatch.setattr(runtime,'version',lambda name:'4.39.3' if name=='transformers' else d['runtime_versions'][name])
    with pytest.raises(RuntimeError,match='Incompatible shared detector runtime'):
        runtime.check_runtime(d)


def test_missing_worker_does_not_fall_back_to_method_python(monkeypatch):
    monkeypatch.setenv('CPR_DETECTOR_PYTHON','/missing/shared-detector/python')
    with pytest.raises(FileNotFoundError,match='Missing detector Python'):
        runtime.detector_python()


def test_complete_journal_seals_without_model_initialization(toy_protocol,monkeypatch):
    cfg,gallery,queries,gm,qm,calls,selector = toy_protocol
    cache = StageCache('detections',detection_fingerprint(gm,cfg))
    cache.directory.mkdir(parents=True)
    rows = [worker.detection_row(g['image_id'],Image.new('RGB',(40,30)),np.empty((0,4)),np.empty(0)) for g in gallery]
    from cpr_person.cache import canonical_hash
    (cache.directory/'persons.partial.jsonl').write_text(''.join(
        json.dumps({**row,'sha256':canonical_hash(row)})+'\n' for row in rows))
    monkeypatch.setattr(worker,'protocol_config',lambda:cfg)
    monkeypatch.setattr(assets,'load_detector',lambda *args:pytest.fail('complete journal must not load a model'))
    worker.detect(gallery,cache,'cpu')
    assert get_detections(gallery,gm,'cpu',cfg=cfg,require_cache=True).cache_hit


def test_fafa_query_and_gallery_crop_same_fractional_boxes(toy_protocol):
    cfg,gallery,queries,gm,qm,calls,selector = toy_protocol
    path = ROOT/'methods/published/02_fafa_setmatch/run.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if getattr(n,'name',None) in
             {'PersonCropDataset','GalleryPersonDataset','QueryTargetDataset'}]
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
    env = {'Dataset':object,'np':np,'Image':Image,'image_path':lambda row,gi:Path(row['path'])}
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),env)
    box = (1.25,2.5,15.75,25.5)
    gp = env['GalleryPersonDataset'](gallery,[[BoxCandidate(box,.8)],[]],np.asarray)
    query = [{'image_id':gallery[0]['image_id']}]
    qp = env['QueryTargetDataset'](query,[[SimpleNamespace(modify_text='public change')]],
        [[[box]]],gallery,{gallery[0]['image_id']:0},np.asarray)
    pixels,caption,owner = qp[0]
    np.testing.assert_array_equal(gp[0],pixels)
    assert caption == 'public change' and owner == 0


def test_asset_inventory_detects_tampering(tmp_path):
    d = copy.deepcopy(protocol_config()['detector'])
    d['runtime_cache'] = str(tmp_path)
    d['runtime_assets_marker'] = str(tmp_path/'prepared.json')
    asset = tmp_path/'config.json';asset.write_text('{}')
    from cpr_person.cache import sha256_file
    atomic_json(tmp_path/'prepared.json',{'identity':assets.asset_identity(d),
        'files':[{'path':'config.json','sha256':sha256_file(asset)}]})
    assert assets.require_snapshot(d)['files']
    asset.write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='Changed/incomplete'):
        assets.require_snapshot(d)


def test_real_hf_processor_thresholds_and_pixel_conversion(tmp_path):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    if transformers.__version__ != '4.56.2':
        pytest.skip('run in the pinned detector runtime')
    vocab = tmp_path/'vocab.txt'
    vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nperson\n.\n')
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab))
    processor = transformers.GroundingDinoProcessor(transformers.GroundingDinoImageProcessor(),tokenizer)
    image = Image.new('RGB',(40,30))
    inputs = processor(images=image,text=[['person']],return_tensors='pt')
    assert tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True) == ['person.']
    assert tuple(inputs.pixel_values.shape) == (1,3,800,1066)
    logits = torch.full((1,4,len(inputs.input_ids[0])),-10.)
    logits[0,0,1] = torch.logit(torch.tensor(.8))
    logits[0,1,1] = torch.logit(torch.tensor(.4))
    logits[0,2,1] = torch.logit(torch.tensor(.299))  # Below box threshold.
    logits[0,3,0] = torch.logit(torch.tensor(.9))  # Special token -> empty phrase.
    output = SimpleNamespace(logits=logits,pred_boxes=torch.tensor([
        [[.5,.5,.4,.8],[.25,.25,.1,.1],[.6,.6,.2,.2],[.1,.1,.1,.1]]]))
    model = lambda **kw: output
    cfg = protocol_config()['detector']
    boxes,scores = worker.detect_image(model,processor,image,cfg,'cpu')
    np.testing.assert_allclose(boxes,[[12,3,28,27],[8,6,12,9]],atol=1e-6)
    np.testing.assert_allclose(scores,[.8,.4])
    changed = {**cfg,'text_threshold':.5}
    boxes,scores = worker.detect_image(model,processor,image,changed,'cpu')
    assert len(boxes) == 1 and scores[0] == pytest.approx(.8)
    changed = {**cfg,'threshold':.5}
    assert len(worker.detect_image(model,processor,image,changed,'cpu')[0]) == 1


@pytest.mark.parametrize('method', ['fafa_setmatch','structured_caption_retrieval','clip_image'])
def test_runner_requires_rcr_index_only_for_person_methods(tmp_path,monkeypatch,method):
    import run_baseline as runner
    directory = tmp_path/'methods'/method
    directory.mkdir(parents=True)
    for name in ('config.yaml','requirements.txt','run.py','download_checkpoint.py'):
        (directory/name).touch()
    if method == 'structured_caption_retrieval':
        (directory/'prepare_person_cache.py').touch()
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.delenv('CPR_DETECTOR_PYTHON',raising=False)
    monkeypatch.setattr(runner,'ensure_gallery_layout',lambda *a,**k:tmp_path/'gallery')
    monkeypatch.setattr(runner,'describe_gallery_link',lambda *a:'test gallery')
    events = []
    index = tmp_path / "index.pt"
    monkeypatch.delenv("CPR_RCR_DETECTION_CACHE", raising=False)

    def detection_index(*, required):
        assert required
        events.append("rcr index")
        return index

    def native(**kwargs):
        events.append("native")
        if method in PERSON_METHODS:
            assert __import__("os").environ["CPR_RCR_DETECTION_CACHE"] == str(index)
        return "/native/bin/python", {}

    monkeypatch.setattr(runner, "rcr_index_path", detection_index)
    monkeypatch.setattr(runtime, "prepare_runtime", lambda **kw: pytest.fail("must reuse RCR boxes"))
    monkeypatch.setattr('baseline_environment.prepare_method_python',native)
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:None)
    monkeypatch.setattr(runner,'run_checkpoint_step',lambda *a,**k:False)
    runner.run_pipeline(runner.MethodSpec(method,directory),False,True)
    assert events == (["rcr index", "native"] if method in PERSON_METHODS else ["native"])


def test_worker_builds_gallery_then_cache_hit_avoids_forward(toy_protocol,monkeypatch):
    cfg,gallery,queries,gm,qm,calls,selector = toy_protocol
    fp = detection_fingerprint(gm,cfg)
    cache = StageCache('detections',fp)
    cache.directory.mkdir(parents=True)
    loads, forwards = [], []
    monkeypatch.setattr(worker,'protocol_config',lambda:cfg)
    monkeypatch.setattr(runtime,'check_runtime',lambda d:d['runtime_versions'])
    def load(*args):
        loads.append(1);return object(),object()
    def detect(*args):
        forwards.append(1)
        if len(forwards) == 1:
            return np.array([[.25,1.5,10.75,20.25],[5.5,2.25,25.5,30.5]],np.float32),np.array([.4,.9],np.float32)
        return np.empty((0,4),np.float32),np.empty(0,np.float32)
    monkeypatch.setattr(assets,'load_detector',load)
    monkeypatch.setattr(worker,'detect_image',detect)
    worker.detect(gallery,cache,'cpu')
    result = get_detections(gallery,gm,'cpu',cfg=cfg,require_cache=True)
    assert result.cache_hit and len(loads) == 1 and len(forwards) == 2
    assert result.offsets.tolist() == [0,2,3] and result.fallback.tolist() == [False,False,True]
    assert result.boxes[0,0] == .25 and result.confidences[0] < result.confidences[1]


def test_detection_change_invalidates_localization_and_fafa_features(toy_protocol,tmp_path):
    from cpr_person.adapter import load_person_context
    from test_person_integrations import function
    import hashlib
    cfg,gallery,queries,gm,qm,calls,selector = toy_protocol
    first = load_person_context(gallery,queries,gm,qm,'cpu')
    cfg['detector']['threshold'] += .01
    second = load_person_context(gallery,queries,gm,qm,'cpu')
    assert first.detections.path != second.detections.path
    assert first.localization.path != second.localization.path
    build = function('methods/published/02_fafa_setmatch/run.py','build_stage_caches',{
        'hashlib':hashlib,'json':json,'resolve_config_path':lambda p:Path(p),
        'checkpoint_identity':lambda p:'same-native-checkpoint'})
    native = {'checkpoint':{'path':'unchanged.pt','max_txt_len':128},'source':{'commit':'unchanged'}}
    before = build(native,gm,qm,tmp_path,first)[1]['keys']
    after = build(native,gm,qm,tmp_path,second)[1]['keys']
    for stage in ('gallery_features','query_features','component_scores'):
        assert before[stage] != after[stage]
