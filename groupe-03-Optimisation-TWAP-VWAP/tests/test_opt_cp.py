from src.strategies.constrained_opt_cp import constrained_opt_cp_schedule


def test_opt_conserves_volume():
    """The optimizer must conserve the total traded volume Q."""
    Q = 100
    volumes = [10, 20, 30, 20, 20]

    result = constrained_opt_cp_schedule(
        Q=Q,
        volumes=volumes,
        participation_rate=1.0,
        w_impact=1,
        w_track=0
    )

    assert result.feasible
    assert sum(result.slices) == Q


def test_opt_respects_caps():
    """The optimizer must respect participation caps."""
    Q = 50
    volumes = [10, 10, 10, 10, 10]
    participation_rate = 0.5  # cap = 5 per slice

    result = constrained_opt_cp_schedule(
        Q=Q,
        volumes=volumes,
        participation_rate=participation_rate,
        w_impact=1,
        w_track=0
    )

    caps = [int(participation_rate * v) for v in volumes]

    for x, cap in zip(result.slices, caps):
        assert x <= cap


def test_opt_moves_towards_vwap_when_tracking_enabled():
    """With tracking enabled, OPT should be closer to VWAP than TWAP is."""
    Q = 100
    volumes = [5, 15, 60, 15, 5]

    # OPT without tracking
    opt_no_track = constrained_opt_cp_schedule(
        Q=Q,
        volumes=volumes,
        participation_rate=1.0,
        w_impact=1,
        w_track=0
    ).slices

    # OPT with strong tracking
    opt_track = constrained_opt_cp_schedule(
        Q=Q,
        volumes=volumes,
        participation_rate=1.0,
        w_impact=1,
        w_track=20
    ).slices

    # VWAP target
    total_vol = sum(volumes)
    target = [int(Q * v / total_vol) for v in volumes]
    diff = Q - sum(target)
    i = 0
    while diff != 0:
        target[i] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1
        i = (i + 1) % len(target)

    def tracking_error(x):
        return sum((x[i] - target[i]) ** 2 for i in range(len(x)))

    assert tracking_error(opt_track) <= tracking_error(opt_no_track)
