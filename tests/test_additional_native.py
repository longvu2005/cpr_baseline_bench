"""Optional real upstream forwards on tiny random CLIP backbones, no metrics.

Uses downloaded official source/mapper weights when available. The random
backbones are deliberately not presented as a pretrained end-to-end model.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cpr_baselines.common import load_config, load_source_file, normalized
from cpr_baselines.published import ContextI2W, FTI4CIR, SEARLE, torch_load
from cpr_person.cache import resolve, sha256_file
from run_baseline import discover_methods, resolve_method


def cfg(method):
    return load_config(resolve_method(method, discover_methods()).config_path)


def path_or_skip(env, default):
    path = Path(os.environ.get(env, str(default))).resolve()
    if not path.exists():
        pytest.skip(f'Official source/checkpoint not downloaded: {path.name}')
    return path


def openai_clip(monkeypatch):
    pytest.importorskip('pkg_resources')
    source = path_or_skip('CPR_OPENAI_CLIP_SOURCE', resolve(cfg('searle_xl')['assets']['clip_code']['path']))
    monkeypatch.syspath_prepend(str(source))
    clip = importlib.import_module('clip')
    assert Path(clip.__file__).resolve().is_relative_to(source)
    return clip


def test_searle_official_phi_and_pseudo_encoder(monkeypatch):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(2)
    clip = openai_clip(monkeypatch)
    c = cfg('searle_xl')
    source = path_or_skip('CPR_SEARLE_SOURCE', resolve(c['source']['local_checkout']))
    checkpoint = path_or_skip('CPR_SEARLE_CHECKPOINT', resolve(c['assets']['checkpoint']['path']))
    assert sha256_file(checkpoint) == c['assets']['checkpoint']['sha256']
    phi_mod = load_source_file('test_searle_phi', source/'src/phi.py')
    phi = phi_mod.Phi(768, 3072, 768, .5).eval()
    phi.load_state_dict(torch_load(checkpoint)['Phi'], strict=True)
    native = SEARLE.__new__(SEARLE)
    native.cfg = c
    native.torch, native.device, native.clip, native.phi = torch, torch.device('cpu'), clip, phi
    from clip.model import CLIP
    from torchvision.transforms import Compose, Resize, ToTensor
    native.model = CLIP(768,32,1,64,16,77,49408,768,12,1).float().eval()
    native.preprocess = Compose([Resize((32,32)), ToTensor()])
    native.pseudo_encoder = load_source_file('test_searle_pseudo', source/'src/encode_with_pseudo_tokens.py').encode_with_pseudo_tokens
    images = [Image.new('RGB',(40,32),'red'), Image.new('RGB',(32,40),'blue')]
    result = normalized(native.queries(images,['wearing a red shirt','holding a glass']))
    assert result.shape == (2,768) and np.isfinite(result).all()
    with pytest.raises(ValueError, match='reserved'):
        native.queries(images,['bad $ placeholder','shirt'])


def test_context_i2w_official_custom_tokenizer_and_mapper(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('einops_exts')
    torch.set_num_threads(2)
    c=cfg('context_i2w')
    source=path_or_skip('CPR_CONTEXT_I2W_SOURCE', resolve(c['source']['local_checkout']))
    monkeypatch.syspath_prepend(str(source))
    model_mod=importlib.import_module('model.model')
    tok_mod=importlib.import_module('third_party.open_clip.simple_tokenizer')
    assert Path(model_mod.__file__).resolve().is_relative_to(source)
    native=ContextI2W.__new__(ContextI2W)
    native.cfg=c
    native.torch,native.device,native.tokenizer=torch,torch.device('cpu'),tok_mod.SimpleTokenizer()
    native.model=model_mod.CLIP(embed_dim=64,image_resolution=32,vision_layers=1,vision_width=64,
        vision_patch_size=16,context_length=77,vocab_size=49408,transformer_width=64,
        transformer_heads=1,transformer_layers=1).float().eval()
    native.mapper=model_mod.IM2TEXT_OWM_Attn(transformer_width=64,vision_width=64,
        embed_dim=64,middle_dim=32,output_dim=64,n_layer=2).float().eval()
    native.replace_id=native.tokenizer.encoder['<|replace|>']
    assert native.replace_id == 49405 and native.model.replace_id == 49405
    from torchvision.transforms import Compose, Resize, ToTensor
    native.preprocess=Compose([Resize((32,32)),ToTensor()])
    images=[Image.new('RGB',(32,32),'green'),Image.new('RGB',(32,32),'blue')]
    result=normalized(native.queries(images,['with a red shirt','with a white hat']))
    assert result.shape == (2,64)
    tokens=native.tokenize(['a photo of <|replace|> , '+'shirt '*200])
    assert tokens.shape == (1,77) and int((tokens==49405).sum()) == 1
    assert int(tokens[0,-1]) == native.model.end_id


def test_fti4cir_official_serialized_mapper_and_token_replacement(monkeypatch):
    torch=pytest.importorskip('torch')
    torch.set_num_threads(2)
    clip=openai_clip(monkeypatch)
    c=cfg('fti4cir')
    source=path_or_skip('CPR_FTI4CIR_SOURCE',resolve(c['source']['local_checkout']))
    checkpoint=path_or_skip('CPR_FTI4CIR_CHECKPOINT',resolve(c['assets']['checkpoint']['path']))
    assert sha256_file(checkpoint) == c['assets']['checkpoint']['sha256']
    monkeypatch.syspath_prepend(str(source/'src'))
    mapper_mod=importlib.import_module('i2t_model')
    mapper=torch_load(checkpoint).float().eval()
    assert isinstance(mapper,mapper_mod.IMG2TEXT)
    native=FTI4CIR.__new__(FTI4CIR)
    native.torch,native.device,native.clip,native.mapper=torch,torch.device('cpu'),clip,mapper
    native.utils=importlib.import_module('utils')
    from clip.model import CLIP
    from torchvision.transforms import Compose, Resize, ToTensor
    native.model=CLIP(768,32,1,1024,16,77,49408,768,12,1).float().eval()
    native.preprocess=Compose([Resize((32,32)),ToTensor()])
    result=normalized(native.queries([Image.new('RGB',(32,32),'red')],['wearing a blue shirt']))
    assert result.shape == (1,768) and np.isfinite(result).all()
    with pytest.raises(ValueError,match='reserved'):
        native.queries([Image.new('RGB',(32,32),'red')],['bad * placeholder'])
