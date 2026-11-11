"""FAFA query length and RCR Proposed full-condition text routing."""

import re

RCR_PROPOSED_SOURCE_COMMIT = "da764bc3c158dc5286e4b65bfa4c3bbd8cf22296"
PROPOSED_TEXT_POLICY = "rcr-proposed-complete-relative-subject-v1"


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


def subject_conditions(sample: dict) -> list[str]:
    """Anchor the complete condition to each public Subject, without splitting.

    Exact RCR Proposed src/rcr/proposed/cache/native.py logic, pinned at
    RCR_PROPOSED_SOURCE_COMMIT. No case/identity/GT membership enters this.
    """
    ids = [subject["subject_id"] for subject in sample["subjects"]]
    text = re.sub(
        r"^then retrieve target images where\s+",
        "",
        sample["final_change"],
        flags=re.IGNORECASE,
    ).strip()
    result = []
    for sid in ids:

        def replace(match, current=sid):
            return (
                "the reference person"
                if int(match[1]) == current
                else f"the other referenced person (Subject {match[1]})"
            )

        condition = re.sub(
            r"(?:the\s+members\s+of\s+|members\s+of\s+)?\bSubject\s+(\d+)\b",
            replace,
            text,
        )
        # Per-crop FAFA describes one person even when a public Subject is a group.
        # Keep the condition itself; only fix adjacent plural auxiliary agreement.
        condition = re.sub(
            r"(the reference person|the other referenced person \(Subject \d+\)) "
            r"(are|were|have)\b",
            lambda match: (
                match[1] + " " + {"are": "is", "were": "was", "have": "has"}[match[2]]
            ),
            condition,
        )
        result.append(condition)
    return result
