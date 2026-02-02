# src/strategies/twap.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ScheduleResult:
    """Execution schedule result."""
    slices: List[int]           # quantity per interval
    total: int                  # total executed (should equal Q)
    feasible: bool              # whether constraints were satisfied exactly
    message: str                # info if infeasible/adjusted


def _distribute_remainder(base: List[int], remainder: int) -> List[int]:
    """
    Distribute +1 over the first 'remainder' slots (deterministic).
    Assumes remainder >= 0.
    """
    out = base[:]
    for i in range(remainder):
        out[i] += 1
    return out


def _cap_and_redistribute(x: List[int], caps: List[int]) -> List[int]:
    """
    Apply caps elementwise and redistribute any leftover volume to intervals
    that still have capacity, in a greedy round-robin way.
    """
    n = len(x)
    x = [min(x[i], caps[i]) for i in range(n)]
    return x


def twap_schedule(
    Q: int,
    N: int,
    min_per_slice: int = 0,
    max_per_slice: Optional[int] = None,
    caps: Optional[List[int]] = None,
) -> ScheduleResult:
    """
    Compute a TWAP schedule: split Q into N slices as evenly as possible.

    Constraints supported:
    - min_per_slice: lower bound per interval
    - max_per_slice: upper bound per interval (same for all intervals)
    - caps: per-interval cap (overrides max_per_slice if provided)

    Returns integer quantities per interval.

    Notes:
    - If constraints make exact execution impossible, returns the closest feasible
      schedule (greedy) and feasible=False with message.
    """
    if Q < 0:
        raise ValueError("Q must be >= 0")
    if N <= 0:
        raise ValueError("N must be >= 1")
    if min_per_slice < 0:
        raise ValueError("min_per_slice must be >= 0")

    # Build caps vector
    if caps is not None:
        if len(caps) != N:
            raise ValueError("caps must have length N")
        caps_vec = caps[:]
    else:
        if max_per_slice is None:
            caps_vec = [10**18] * N  # effectively unbounded
        else:
            if max_per_slice < 0:
                raise ValueError("max_per_slice must be >= 0")
            caps_vec = [max_per_slice] * N

    # Quick feasibility checks for exact fulfillment
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
    x = [min_per_slice] * N
    remaining = Q - min_total

    # Evenly distribute remaining (TWAP spirit)
    base_add = remaining // N
    rem = remaining % N

    x = [x[i] + base_add for i in range(N)]
    x = _distribute_remainder(x, rem)

    # Apply caps; if we cap some intervals, we need to redistribute leftover
    # until no leftover remains.
    # Compute overflow to redistribute
    overflow = 0
    for i in range(N):
        if x[i] > caps_vec[i]:
            overflow += x[i] - caps_vec[i]
            x[i] = caps_vec[i]

    # Redistribute overflow to intervals with remaining capacity
    # Greedy round-robin
    i = 0
    while overflow > 0:
        if x[i] < caps_vec[i]:
            x[i] += 1
            overflow -= 1
        i = (i + 1) % N

    return ScheduleResult(
        slices=x,
        total=sum(x),
        feasible=True,
        message="OK"
    )
