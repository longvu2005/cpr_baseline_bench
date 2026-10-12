import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from cpr_person.matching import aggregate_people, partial_setmatch, score_feature_sets
from cpr_person.penalties import basic_unmatched_bound


def reference(s, b):
    if not s.shape[1]:
        return b
    rows, cols = linear_sum_assignment(-s.astype(np.float64))
    return (s[rows, cols].astype(np.float64).sum() + (s.shape[0] - len(rows))*b) / s.shape[0]


@pytest.mark.parametrize("k,n", [(1,0),(1,1),(1,6),(2,0),(2,1),(2,2),(2,7),(3,1),
                                  (5,2),(4,3),(3,3),(4,9),(9,4)])
@pytest.mark.parametrize("b", [-1.,0.,-.27,-1000000.])
def test_scalar_hungarian_cases(k,n,b):
    rng = np.random.default_rng(17+k+n)
    for s in [rng.normal(size=(k,n)), np.zeros((k,n)), -np.ones((k,n))]:
        assert partial_setmatch(s,b) == pytest.approx(reference(s,b),abs=1e-10)


@pytest.mark.parametrize("k", range(1,11))
@pytest.mark.parametrize("b", [-1.,.9,-1000000.])
def test_batched_fast_paths_and_generic(k,b):
    rng = np.random.default_rng(k)
    counts = np.tile(np.arange(12), 3)
    offsets = np.r_[0,np.cumsum(counts)]
    for s in [rng.normal(size=(k,int(offsets[-1]))),
              rng.integers(-2,3,size=(k,int(offsets[-1]))).astype(float)]:
        got = aggregate_people(s,offsets,b)
        expected = [reference(s[:,a:z],b) for a,z in zip(offsets[:-1],offsets[1:])]
        np.testing.assert_allclose(got,expected,rtol=1e-7,atol=1e-6)
        assert np.isfinite(got).all()


def test_forced_real_matches_and_no_person_reuse():
    # b is deliberately better than all real pairs: optional unmatching is forbidden.
    s = np.array([[-3.],[-2.]])
    assert partial_setmatch(s,1.) == -.5
    # Both components prefer person 0; the gallery person can be used only once.
    s = np.array([[1.,.2],[.9,.1]])
    assert partial_setmatch(s,-1.) == pytest.approx(.55)
    assert aggregate_people(s,np.array([0,2]),-1.)[0] == pytest.approx(.55)


def test_mean_gives_partial_credit():
    s=np.array([[.8],[.6]])
    assert partial_setmatch(s,-1.) == pytest.approx(-.1)
    assert partial_setmatch(s,0.) == pytest.approx(.4)


@pytest.mark.parametrize("metric", ["cosine","negative_squared_euclidean"])
def test_chunked_native_pairwise_score(metric):
    rng=np.random.default_rng(16)
    q,g=rng.normal(size=(4,5)),rng.normal(size=(6,5))
    offsets=np.array([0,0,1,3,6])
    raw=q@g.T if metric=="cosine" else -((q[:,None,:]-g[None,:,:])**2).sum(axis=-1)
    expected=aggregate_people(raw,offsets,-23)
    for chunk in [1,2,100]:
        np.testing.assert_allclose(score_feature_sets(q,g,offsets,-23,chunk,metric=metric),expected,rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize("s,b", [(np.zeros((0,2)),-1), (np.array([[np.nan]]),0),
                                 (np.array([[np.inf]]),0),(np.ones((1,2)),np.inf)])
def test_invalid_inputs_are_rejected(s,b):
    with pytest.raises(ValueError):partial_setmatch(s,b)


def test_basic_resource_bound():
    rng=np.random.default_rng(44)
    for _ in range(50):
        im,tm=rng.uniform(0,1,size=2)
        mi,mt=-rng.uniform(.1,2,size=2)
        lam=.1
        b=basic_unmatched_bound(im,tm,mi,mt,lam)
        x=rng.uniform(0,((1+im)**2-mi)/abs(mi),size=100)
        y=rng.uniform(0,((1+im)*(1+tm)-mt)/abs(mi),size=100)
        assert np.all(x*y-lam*(x+y)**2 >= b)
