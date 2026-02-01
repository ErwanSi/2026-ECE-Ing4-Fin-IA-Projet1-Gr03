# src/strategies/vwap.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ScheduleResult:
    slices: List[int]
    total: int
    feasible: bool
    message: str


def _round_proportional_allocation(Q: int, weights: List[float]) -> List[int]:
    """
    Allocate integer quantities proportionally to weights using:
    - floor allocation
    - then distribute remainder to largest fractional parts
    """
    if Q == 0:
        return [0] * len(weights)

    raw = [Q * w for w in weights]
    floored = [int(x) for x in raw]
    remainder = Q - sum(floored)

    # fractional parts
    fracs = [(raw[i] - floored[i], i) for i in range(len(weights))]
    fracs.sort(reverse=True)  # largest fractional first

    out = floored[:]
    for k in range(remainder):
        out[fracs[k][1]] += 1
    return out


def vwap_schedule(
    Q: int,
    volumes: List[float],
    min_per_slice: int = 0,
    max_per_slice: Optional[int] = None,
    participation_rate: Optional[float] = None,
    caps: Optional[List[int]] = None,
) -> ScheduleResult:
    """
    VWAP baseline:
    - Allocate Q proportionally to market volumes.

    Constraints:
    - min_per_slice (>= 0)
    - max_per_slice (global cap)
    - participation_rate: cap each slice to p * volume_t (p in (0,1])
    - caps: explicit per-slice caps (overrides max_per_slice/participation caps if provided)

    Returns integer schedule.
    If constraints make Q infeasible, returns the closest feasible capped schedule and feasible=False.
    """
    if Q < 0:
        raise ValueError("Q must be >= 0")
    if min_per_slice < 0:
        raise ValueError("min_per_slice must be >= 0")
    if len(volumes) == 0:
        raise ValueError("volumes must be non-empty")

    N = len(volumes)

    # Build per-slice caps
    caps_vec = [10**18] * N

    # participation cap
    if participation_rate is not None:
        if not (0 < participation_rate <= 1):
            raise ValueError("participation_rate must be in (0, 1]")
        caps_vec = [min(caps_vec[i], int(participation_rate * volumes[i])) for i in range(N)]

    # max_per_slice cap
    if max_per_slice is not None:
        if max_per_slice < 0:
            raise ValueError("max_per_slice must be >= 0")
        caps_vec = [min(caps_vec[i], max_per_slice) for i in range(N)]

    # explicit caps override everything if provided
    if caps is not None:
        if len(caps) != N:
            raise ValueError("caps must have length == len(volumes)")
        caps_vec = caps[:]

    # Check feasibility bounds
    min_total = N * min_per_slice
    max_total = sum(caps_vec)

    if Q < min_total:
        return ScheduleResult(
            slices=[min_per_slice] * N,
            total=min_total,
            feasible=False,
            message=f"Infeasible: Q={Q} < N*min_per_slice={min_total}. Returning minimum schedule."
        )
    if Q > max_total:
        return ScheduleResult(
            slices=caps_vec,
            total=max_total,
            feasible=False,
            message=f"Infeasible: Q={Q} > sum(caps)={max_total}. Returning capped schedule."
        )

    # Start from minimum allocation
    remaining = Q - min_total

    # If all market volumes are zero, fallback to uniform split for remaining
    total_vol = sum(volumes)
    if total_vol <= 0:
        # uniform weights
        weights = [1.0 / N] * N
    else:
        weights = [v / total_vol for v in volumes]

    add = _round_proportional_allocation(remaining, weights)
    x = [min_per_slice + add[i] for i in range(N)]

    # Apply caps and redistribute overflow
    overflow = 0
    for i in range(N):
        if x[i] > caps_vec[i]:
            overflow += x[i] - caps_vec[i]
            x[i] = caps_vec[i]

    # Redistribute overflow to slices with remaining capacity
    i = 0
    while overflow > 0:
        if x[i] < caps_vec[i]:
            x[i] += 1
            overflow -= 1
        i = (i + 1) % N

    return ScheduleResult(slices=x, total=sum(x), feasible=True, message="OK")
