"""CPU-only validation that CPR's FAFA uses RCR's 128-token policy."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from cpr_person.fafa_text import apply_fafa_text_policy, resolve_fafa_text_policy

CONFIG = Path(__file__).resolve().parents[1] / "methods/published/02_fafa_setmatch/config.yaml"


def test_cpr_config_128_and_max_sum_mean():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["checkpoint"]["max_txt_len"] == 128
    assert cfg["setmatch"]["assignment"] == "hungarian_max_sum"
    assert cfg["setmatch"]["aggregation"] == "mean"
    assert cfg["setmatch"]["unmatched_score"] == -1.0


def test_tokenizer_and_caption_limits_actually_overridden():
    model = SimpleNamespace(max_txt_len=32)
    processor = SimpleNamespace(max_words=50)
    policy = apply_fafa_text_policy(model, {"eval": processor}, {"max_txt_len": 128})
    assert policy == {"max_txt_len": 128, "max_words": 128}
    assert model.max_txt_len == 128
    assert processor.max_words == 128


@pytest.mark.parametrize("bad", [None, 0, 1, 513, True, 128.0, "128"])
def test_bad_lengths_fail_closed(bad):
    with pytest.raises(ValueError):
        resolve_fafa_text_policy({"max_txt_len": bad})
