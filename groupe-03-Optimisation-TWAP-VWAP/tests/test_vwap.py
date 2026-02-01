from src.strategies.vwap import vwap_schedule

def test_vwap_proportional_example():
    res = vwap_schedule(Q=100, volumes=[10, 40, 100, 30, 20])
    assert res.feasible
    assert sum(res.slices) == 100
    # Expected exact in this case
    assert res.slices == [5, 20, 50, 15, 10]

def test_vwap_infeasible_participation_cap():
    # participation_rate=0.2 -> caps = [2,8,20,6,4] sum=40, impossible to reach 100
    res = vwap_schedule(Q=100, volumes=[10, 40, 100, 30, 20], participation_rate=0.2)
    assert not res.feasible
    assert sum(res.slices) == 40
