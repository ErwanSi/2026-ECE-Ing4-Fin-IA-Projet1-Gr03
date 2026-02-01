from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from ortools.sat.python import cp_model


@dataclass(frozen=True)
class ScheduleResult:
    slices: List[int]
    total: int
    feasible: bool
    message: str


def constrained_opt_cp_schedule(
    Q: int,
    volumes: List[int],
    min_per_slice: int = 0,
    max_per_slice: Optional[int] = None,
    participation_rate: Optional[float] = None,
    caps: Optional[List[int]] = None,
    w_impact: int = 1,
    w_track: int = 0,
    time_limit_s: float = 30.0,
    forbidden_slices: Optional[List[int]] = None,
) -> ScheduleResult:
    if Q < 0:
        raise ValueError("Q must be >= 0")
    if min_per_slice < 0:
        raise ValueError("min_per_slice must be >= 0")
    if len(volumes) == 0:
        raise ValueError("volumes must be non-empty")
    if w_impact < 0 or w_track < 0:
        raise ValueError("weights must be >= 0")

    N = len(volumes)

    # ---- caps per slice
    caps_vec = [10**9] * N

    if participation_rate is not None:
        if not (0 < participation_rate <= 1):
            raise ValueError("participation_rate must be in (0, 1]")
        for i in range(N):
            caps_vec[i] = min(caps_vec[i], int(participation_rate * volumes[i]))

    if max_per_slice is not None:
        if max_per_slice < 0:
            raise ValueError("max_per_slice must be >= 0")
        for i in range(N):
            caps_vec[i] = min(caps_vec[i], max_per_slice)

    if caps is not None:
        if len(caps) != N:
            raise ValueError("caps must have length == len(volumes)")
        caps_vec = caps[:]

    # Timing hard constraint (optional)
    if forbidden_slices is not None:
        for idx in forbidden_slices:
            if 0 <= idx < N:
                caps_vec[idx] = 0
            else:
                raise ValueError(f"forbidden_slices contains out-of-range index {idx}")

    min_total = N * min_per_slice
    max_total = sum(caps_vec)

    if Q < min_total:
        return ScheduleResult([min_per_slice] * N, min_total, False,
                              f"Infeasible: Q={Q} < N*min_per_slice={min_total}")
    if Q > max_total:
        return ScheduleResult(caps_vec, max_total, False,
                              f"Infeasible: Q={Q} > sum(caps)={max_total}")

    model = cp_model.CpModel()

    x = [model.NewIntVar(min_per_slice, caps_vec[i], f"x_{i}") for i in range(N)]
    model.Add(sum(x) == Q)

    obj_terms = []

    # ---- Impact term: sum( x^2 / volume ) approx with integer weights
    if w_impact > 0:
        SCALE = 1000  # ✅ petit SCALE => solve stable
        for i in range(N):
            sq = model.NewIntVar(0, caps_vec[i] * caps_vec[i], f"x2_{i}")
            model.AddMultiplicationEquality(sq, [x[i], x[i]])

            vol = max(1, int(volumes[i]))
            w_i = max(1, SCALE // vol)  # avoid 0 weight

            obj_terms.append(w_impact * w_i * sq)

    # ---- Tracking term (✅ L1): sum |x - target|
    if w_track > 0:
        total_vol = sum(volumes)
        if total_vol <= 0:
            targets = [Q // N] * N
            targets[0] += Q - sum(targets)
        else:
            raw = [Q * volumes[i] / total_vol for i in range(N)]
            targets = [int(r) for r in raw]
            diff = Q - sum(targets)
            fracs = [(raw[i] - int(raw[i]), i) for i in range(N)]
            fracs.sort(reverse=True)
            for k in range(abs(diff)):
                idx = fracs[k][1]
                targets[idx] += 1 if diff > 0 else -1

        for i in range(N):
            # ✅ must allow large negative values when target > cap
            d = model.NewIntVar(-Q, Q, f"d_{i}")
            model.Add(d == x[i] - targets[i])

            ad = model.NewIntVar(0, Q, f"abs_d_{i}")
            model.AddAbsEquality(ad, d)

            obj_terms.append(w_track * ad)

    model.Minimize(sum(obj_terms) if obj_terms else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ScheduleResult([], 0, False, f"Solver status={status_name}")

    slices = [int(solver.Value(x[i])) for i in range(N)]
    return ScheduleResult(slices, sum(slices), True, f"OK ({status_name})")
