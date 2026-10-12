import copy

import numpy as np
import pytest

from cpr_person.adapter import load_person_context
from cpr_person.detection import BoxCandidate
from cpr_person.localization import select_subject_members, localize_queries, get_localization


def candidates(n, fallback=()):
    return [BoxCandidate((0,0,10,10),.9,i in fallback) for i in range(n)]


@pytest.mark.parametrize("similarity,expected", [
    ([[.8]],[[0]]),
    ([[.8,.79,.75]],[[0,1]]),
    ([[.7,.69,.68]],[[0,1,2]]),
    ([[.8,.79,.78],[.77,.9,.89]],[[0],[1,2]]),
    ([[.8,.8,.79],[.8,.8,.79]],[[0,2],[1]]),
    ([[.19,.18]],[[0]]),
    ([[-.9,-.7]],[[1]]),
])
def test_membership_anchors_threshold_margin_conflicts(similarity,expected):
    s=np.asarray(similarity)
    members,anchors=select_subject_members(s,candidates(s.shape[1]))
    assert members==expected
    assert all(people[0]==anchor for people,anchor in zip(members,anchors))
    assert len({p for people in members for p in people})==sum(map(len,members))


def test_fallback_never_expands_and_cap_is_deterministic():
    s=np.full((1,8),.7)
    members,_=select_subject_members(s,candidates(8,fallback=[2]),max_members=5)
    assert members==[[0,1,3,4,5]]
    for _ in range(10):assert select_subject_members(s,candidates(8,fallback=[2]),max_members=5)[0]==members


def test_full_reference_padding_group_and_no_case_or_gt_usage(toy_protocol):
    cfg,gallery,queries,gm,qm,calls,selector=toy_protocol
    person=load_person_context(gallery,queries,gm,qm,"cpu")
    members=person.localization.members
    assert len(members[0][0])==2  # GROUP-like wording expands even when case says INDIVIDUAL.
    assert len(members[1])==2 and all(people[0]["fallback"] for people in members[1])
    assert sum(people[0]["global_person_index"] is None for people in members[1])==1
    assert len(members[2])==3 and members[2][2][0]["global_person_index"] is None
    changed=copy.deepcopy(queries)
    for q in changed:
        q["case"]="DUAL";q["case_type"]="bogus";q["target_ids"]=["oracle"]*20
        q["full_positive_ids"]=["a"];q["boxes"]=[[0,0,1,1]]
        for s in q["subjects"]:s["identity_ids"]=["GT"]*40
    result,stats=localize_queries(changed,gallery,person.detections,cfg["selector"],selector)
    assert result==members


def test_cross_method_localization_cache_skips_selector(toy_protocol):
    cfg,gallery,queries,gm,qm,calls,selector=toy_protocol
    a=load_person_context(gallery,queries,gm,qm,"cpu")
    count=(selector.image_calls,selector.text_calls)
    for _ in range(8):
        b=load_person_context(gallery,queries,gm,qm,"cpu")
        assert b.localization.members==a.localization.members and b.localization.cache_hit
        assert b.detections.path==a.detections.path
    assert len(calls)==1 and (selector.image_calls,selector.text_calls)==count


def test_localization_settings_invalidate_without_detector_rerun(toy_protocol):
    cfg,gallery,queries,gm,qm,calls,selector=toy_protocol
    p=load_person_context(gallery,queries,gm,qm,"cpu")
    cfg["selector"]["membership_margin"]=.01
    other=get_localization(queries,qm,gallery,p.detections,"cpu",cfg=cfg,selector=selector)
    assert other.path!=p.localization.path and len(calls)==1


def test_localization_corruption_is_rebuilt(toy_protocol):
    cfg,gallery,queries,gm,qm,calls,selector=toy_protocol
    a=load_person_context(gallery,queries,gm,qm,"cpu")
    a.localization.path.write_text("{broken")
    b=load_person_context(gallery,queries,gm,qm,"cpu")
    assert not b.localization.cache_hit and b.localization.members==a.localization.members


def test_complete_manifest_order_is_required(toy_protocol):
    cfg,gallery,queries,gm,qm,calls,selector=toy_protocol
    with pytest.raises(ValueError):load_person_context(list(reversed(gallery)),queries,gm,qm,"cpu")
    with pytest.raises(ValueError):load_person_context(gallery,queries[:1],gm,qm,"cpu")


@pytest.mark.parametrize("similarity", [np.array([[np.nan]]),np.zeros((0,1)),np.ones((2,1))])
def test_invalid_similarity_rejected(similarity):
    with pytest.raises(ValueError):select_subject_members(similarity,candidates(similarity.shape[1]))
