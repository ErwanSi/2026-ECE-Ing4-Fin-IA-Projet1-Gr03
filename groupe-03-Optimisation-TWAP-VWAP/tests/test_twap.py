from src.strategies.twap import twap_schedule


def test_twap_exact_split():
    res = twap_schedule(Q=10, N=4)
    assert res.feasible is True
    assert sum(res.slices) == 10
    assert max(res.slices) - min(res.slices) <= 1


def test_twap_with_caps_infeasible():
    res = twap_schedule(Q=10, N=4, max_per_slice=2)
    assert res.feasible is False
    assert sum(res.slices) == 8
