import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cpr_person.detection import Detections, detection_fingerprint, filter_person_boxes, get_detections, protocol_config, crop_person


@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setenv("CPR_SHARED_CACHE_ROOT", str(tmp_path/"shared"))
    cfg=protocol_config()
    checkpoint=tmp_path/"gd.pth";checkpoint.write_bytes(b"test detector identity")
    cfg["detector"]["checkpoint"]=str(checkpoint)
    gallery=[{"image_id":f"g{i}","path":str(tmp_path/f"{i}.jpg")} for i in range(2)]
    for row in gallery:Image.new("RGB",(40,30)).save(row["path"])
    manifest=tmp_path/"gallery.jsonl";manifest.write_text("".join(json.dumps(r)+"\n" for r in gallery))
    calls=[]
    def runner(rows,fingerprint):
        calls.append(1)
        return Detections(np.array([0,2,3]), np.array([[0,0,10,30],[20,0,40,30],[0,0,40,30]],np.float32),
                          np.array([.9,.8,-1],np.float32),np.array([False,False,True]),
                          np.array([[40,30],[40,30]]),fingerprint)
    return cfg,gallery,manifest,runner,calls


def test_consistent_shared_reuse_without_inference(setup):
    cfg,gallery,manifest,runner,calls=setup
    a=get_detections(gallery,manifest,"cpu",cfg=cfg,runner=runner)
    b=get_detections(gallery,manifest,"cpu",cfg=cfg,runner=runner)
    assert not a.cache_hit and b.cache_hit and len(calls)==1
    assert a.path==b.path
    np.testing.assert_array_equal(a.boxes,b.boxes)
    assert a.candidates()[1][0].fallback


def test_detector_change_invalidates_but_matching_does_not(setup):
    cfg,gallery,manifest,runner,calls=setup
    a=get_detections(gallery,manifest,"cpu",cfg=cfg,runner=runner)
    cfg["unrelated_setmatch"]={"aggregation":"mean","unmatched":-123}
    b=get_detections(gallery,manifest,"cpu",cfg=cfg,runner=runner)
    assert a.path==b.path and len(calls)==1
    cfg["detector"]["box_threshold"]+=.01
    c=get_detections(gallery,manifest,"cpu",cfg=cfg,runner=runner)
    assert c.path!=a.path and len(calls)==2


def test_gallery_order_and_checkpoint_invalidation(setup):
    cfg,gallery,manifest,runner,calls=setup
    fp=detection_fingerprint(manifest,cfg)
    manifest.write_text("".join(json.dumps(r)+"\n" for r in reversed(gallery)))
    assert detection_fingerprint(manifest,cfg)!=fp
    Path(cfg["detector"]["checkpoint"]).write_bytes(b"different weights")
    assert detection_fingerprint(manifest,cfg)["checkpoint_sha256"]!=fp["checkpoint_sha256"]


def test_corruption_is_detected_and_rebuilt(setup):
    cfg,gallery,manifest,runner,calls=setup
    a=get_detections(gallery,manifest,"cpu",cfg=cfg,runner=runner)
    a.path.write_bytes(b"truncated")
    with pytest.raises(RuntimeError):get_detections(gallery,manifest,"cpu",cfg=cfg,runner=runner,require_cache=True)
    b=get_detections(gallery,manifest,"cpu",cfg=cfg,runner=runner)
    assert len(calls)==2 and b.boxes.shape==(3,4)


def test_read_only_root_is_consumed_without_writes(setup,monkeypatch,tmp_path):
    cfg,gallery,manifest,runner,calls=setup
    a=get_detections(gallery,manifest,"cpu",cfg=cfg,runner=runner)
    original=tmp_path/"shared"
    monkeypatch.setenv("CPR_SHARED_CACHE_ROOT",str(tmp_path/"writable"))
    monkeypatch.setenv("CPR_SHARED_CACHE_READ_ROOTS",str(original))
    b=get_detections(gallery,manifest,"cpu",cfg=cfg,runner=runner,require_cache=True)
    assert b.path==a.path and len(calls)==1
    assert not (tmp_path/"writable").exists()


def test_deterministic_nms_order_and_filtering():
    d=protocol_config()["detector"]
    # Tied confidences: coordinates choose the left person consistently.
    boxes=np.array([[.75,.5,.2,.8],[.25,.5,.2,.8],[.25,.5,.2,.8],
                    [.5,.5,.01,.01],[.5,.5,.2,.2],[np.nan,.5,.2,.2]])
    scores=np.array([.8,.8,.8,.9,.35,.9])
    a,s=filter_person_boxes(boxes,scores,100,100,d)
    assert len(a)==2 and a[0,0]<a[1,0]
    for _ in range(5):np.testing.assert_array_equal(filter_person_boxes(boxes,scores,100,100,d)[0],a)


def test_person_limit_clipping_and_consistent_crop():
    d=protocol_config()["detector"]
    boxes=np.array([[x,.5,.02,.8] for x in np.linspace(.02,.98,20)])
    a,s=filter_person_boxes(boxes,np.linspace(.99,.4,20),200,100,d)
    assert len(a)==10 and np.all(np.diff(s)<=0)
    assert np.array_equal(a,np.round(a))
    image=Image.new("RGB",(200,100))
    assert crop_person(image,a[0]).size==image.crop(tuple(a[0])).size


def test_fallback_semantic_validation(setup):
    cfg,gallery,manifest,runner,calls=setup
    d=runner(gallery,detection_fingerprint(manifest,cfg))
    d.validate(2)
    d.boxes[-1]=[0,0,10,10]
    with pytest.raises(ValueError):d.validate(2)
