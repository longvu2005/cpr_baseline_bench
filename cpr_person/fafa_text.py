"""Explicit native FAFA query text-length policy shared with RCR (128 tokens)."""


def resolve_fafa_text_policy(checkpoint_cfg: dict) -> dict[str, int]:
    length = checkpoint_cfg["max_txt_len"]
    if type(length) is not int or not 2 <= length <= 512:
        raise ValueError("checkpoint.max_txt_len must be an integer in [2, 512]")
    return {"max_txt_len": length, "max_words": max(50, length)}


def apply_fafa_text_policy(model, text_processors, checkpoint_cfg: dict):
    """Set both native Q-Former tokenizer and eval caption word cap."""
    policy = resolve_fafa_text_policy(checkpoint_cfg)
    model.max_txt_len = policy["max_txt_len"]
    text_processors["eval"].max_words = policy["max_words"]
    return policy
