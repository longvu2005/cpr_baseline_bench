from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pytest
from PIL import Image

from cpr_person.cache import sha256_file
from cpr_person.detection import Detections, get_detections, protocol_config
from cpr_person.localization import get_localization


class ToySelector:
    def __init__(self):
        self.image_calls = self.text_calls = 0

    def images(self, images):
        self.image_calls += 1
        features = np.asarray([np.asarray(image).mean(axis=(0,1)) for image in images], dtype=np.float32)
        return features / np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)

    def texts(self, texts):
        self.text_calls += 1
        vectors = {"red": [1,0,0], "green": [0,1,0], "blue": [0,0,1], "group": [1,1,0]}
        features = np.asarray([vectors[t] for t in texts], dtype=np.float32)
        return features / np.linalg.norm(features,axis=1,keepdims=True)


@pytest.fixture
def toy_protocol(tmp_path,monkeypatch):
    monkeypatch.setenv("CPR_SHARED_CACHE_ROOT",str(tmp_path/"shared"))
    monkeypatch.delenv("CPR_SHARED_CACHE_READ_ROOTS",raising=False)
    cfg=protocol_config()
    for section in ["detector","selector"]:
        p=tmp_path/(section+".pt");p.write_bytes(section.encode())
        cfg[section]["checkpoint"]=str(p)
        cfg[section]["checkpoint_sha256"]=sha256_file(p)
    a=Image.new("RGB",(40,30),"red");a.paste(Image.new("RGB",(20,30),(0,255,0)),(20,0))
    a.save(tmp_path/"a.png");Image.new("RGB",(40,30),"blue").save(tmp_path/"b.png")
    gallery=[{"image_id":"a","path":str(tmp_path/"a.png"),"person_ids":["oracle"]},
             {"image_id":"b","path":str(tmp_path/"b.png"),"person_ids":["oracle"]}]
    def subject(text):return {"select_text":text,"modify_text":"new clothes","identity_ids":["GT"]}
    queries=[{"image_id":"a","subjects":[subject("group")],"case":"INDIVIDUAL","target_ids":["wrong"]},
             {"image_id":"b","subjects":[subject("red"),subject("green")],"case":"GROUP"},
             {"image_id":"a","subjects":[subject("red"),subject("green"),subject("group")],"case":"RELATIONAL"}]
    gm,qm=tmp_path/"gallery.jsonl",tmp_path/"queries.jsonl"
    gm.write_text("".join(json.dumps(row)+"\n" for row in gallery))
    qm.write_text("".join(json.dumps(row)+"\n" for row in queries))
    calls=[]
    def runner(rows,fp):
        calls.append(1)
        return Detections(np.array([0,2,3]),np.array([[0,0,20,30],[20,0,40,30],[0,0,40,30]],np.float32),
            np.array([.9,.8,-1],np.float32),np.array([False,False,True]),np.array([[40,30],[40,30]]),fp)
    selector=ToySelector()
    def detections(rows,path,device,**kwargs):
        return get_detections(rows,path,device,cfg=cfg,runner=runner,**kwargs)
    def localization(qs,path,rows,d,device):
        return get_localization(qs,path,rows,d,device,cfg=cfg,selector=selector)
    monkeypatch.setattr("cpr_person.adapter.get_detections",detections)
    monkeypatch.setattr("cpr_person.adapter.get_localization",localization)
    return cfg,gallery,queries,gm,qm,calls,selector
