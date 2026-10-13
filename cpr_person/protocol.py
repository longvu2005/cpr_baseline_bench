"""Lightweight method registry for reporting and person adapters.

This module deliberately uses only the standard library so scene baselines can
build tables without installing person-detector or assignment dependencies.
"""

PERSON_METHODS = frozenset({
    "word4per_setmatch",
    "fafa_setmatch",
    "instruct_reid_setmatch",
    "basic_setmatch",
    "adafocal_setmatch",
    "groundingdino_clipreid_set",
    "groundingdino_clipreid_set_text",
    "reid_set_qwen_edit_proxy",
    "per_person_clip_compose_setmatch",
})

TEXTFREE_DIAGNOSTIC_METHOD = "groundingdino_clipreid_set_textfree_diagnostic"
