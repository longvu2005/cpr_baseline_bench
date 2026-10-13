"""CPU-only regressions for deterministic Qwen token-limit handling."""


import pytest
import yaml
from PIL import Image

from cpr_baselines import captions
from cpr_baselines.backends import CaptionLengthExceeded
from cpr_person.cache import StageCache


def config():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    path = root / "methods/simple/13_qwen_caption_to_caption/config.yaml"
    return yaml.safe_load(path.read_text())


def store_for(tmp_path, monkeypatch, generator, cfg=None):
    monkeypatch.setattr(captions, "StageCache", lambda stage, fp, **kw:
                        StageCache(stage, fp, write_root=tmp_path))
    return captions.CaptionStore(cfg or config(), "fake-frozen-assets", lambda: generator)


def test_length_overflow_retries_compact_prompt_and_seals_only_complete_output(tmp_path, monkeypatch):
    class Fake:
        prompts = []
        def generate(self, prompt, image):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                raise CaptionLengthExceeded("length budget reached")
            return "A person in a blue shirt holding a cup."

    generator = Fake()
    store = store_for(tmp_path, monkeypatch, generator)
    image = Image.new("RGB", (2, 2), "white")
    assert store.request("Describe the scene", image) == "A person in a blue shirt holding a cup."
    assert len(generator.prompts) == 2
    assert "40 words" in generator.prompts[1]
    assert "Describe the scene" in generator.prompts[1]
    assert store.metadata()["length_repairs"] == 1
    assert store.metadata()["generated_records"] == 1
    cached = captions.CaptionStore(config(), "fake-frozen-assets", lambda:
                                   pytest.fail("should not regenerate"))
    assert cached.request("Describe the scene", image) == "A person in a blue shirt holding a cup."
    assert cached.hits == 1


def test_exhausted_length_retry_does_not_cache_partial_output(tmp_path, monkeypatch):
    class AlwaysLong:
        calls = 0
        def generate(self, prompt, image):
            self.calls += 1
            raise CaptionLengthExceeded("overflow")

    generator = AlwaysLong()
    store = store_for(tmp_path, monkeypatch, generator)
    with pytest.raises(ValueError, match="no incomplete caption was cached"):
        store.request("Describe", None)
    assert generator.calls == 2
    assert list(tmp_path.rglob("response.json")) == []


def test_structured_and_format_retries_are_independent(tmp_path, monkeypatch):
    class Fake:
        prompts = []
        def generate(self, prompt, image):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                raise CaptionLengthExceeded("overflow")
            if len(self.prompts) == 2:
                return "not json"
            return '{"appearance":"short hair","clothing":"red shirt","action":"sitting","objects":"cup"}'

    generator = Fake()
    store = store_for(tmp_path, monkeypatch, generator)
    record = store.request("Describe one person in JSON", structured=True)
    assert record["clothing"] == "red shirt"
    assert "exactly the keys" in generator.prompts[1]
    assert "Previous answer:" in generator.prompts[2]
    assert store.metadata()["length_repairs"] == 1
    assert store.metadata()["format_repairs"] == 1


def test_unrelated_backend_error_is_not_swallowed(tmp_path, monkeypatch):
    class Fake:
        def generate(self, prompt, image):
            raise RuntimeError("CUDA failure")

    store = store_for(tmp_path, monkeypatch, Fake())
    with pytest.raises(RuntimeError, match="CUDA failure"):
        store.request("Describe")
    assert list(tmp_path.rglob("response.json")) == []
