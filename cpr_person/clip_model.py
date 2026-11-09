"""Local OpenAI CLIP weights and preprocessing shared by RCR-facing adapters."""
from __future__ import annotations

from pathlib import Path

from .cache import cache_lock, resolve, sha256_file
from .protocol import RCR_CLIP_NAME, RCR_CLIP_SHA256

OFFICIAL_CLIP_WEIGHTS = {
    RCR_CLIP_NAME: (RCR_CLIP_SHA256, "ViT-B-32.pt"),
    "ViT-L/14": ("b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836", "ViT-L-14.pt"),
}


def prepare_openai_clip(model_name, checkpoint, force=False):
    from .assets import download_stream

    digest, filename = OFFICIAL_CLIP_WEIGHTS[model_name]
    path = resolve(checkpoint)
    if not force and path.is_file() and sha256_file(path) == digest:
        print(f"[skip] {path} (checksum valid)", flush=True)
        return
    with cache_lock(path.parent / ".clip-prepare"):
        download_stream(
            url=f"https://openaipublic.azureedge.net/clip/models/{digest}/{filename}",
            path=path, force=force, expected_sha256=digest,
        )


def load_openai_clip(checkpoint, device):
    # Same loader as RCR baselines/clip.py, including raw state-dict support.
    import torch
    from clip.clip import _transform
    from clip.model import build_model

    checkpoint = Path(checkpoint)
    device = torch.device(device)
    try:
        state = torch.jit.load(str(checkpoint), map_location="cpu").state_dict()
    except RuntimeError:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(state).to(device)
    if device.type != "cuda":
        model.float()
    return model.eval(), _transform(model.visual.input_resolution)
