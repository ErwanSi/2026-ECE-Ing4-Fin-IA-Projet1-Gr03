import math

def vwap_target(Q, volumes):
    """Compute an integer VWAP target that sums exactly to Q."""
    total = sum(volumes)
    n = len(volumes)

    if total <= 0:
        base = [Q // n] * n
        for i in range(Q - sum(base)):
            base[i] += 1
        return base

    raw = [Q * v / total for v in volumes]
    base = [int(x) for x in raw]
    diff = Q - sum(base)

    fracs = [(raw[i] - base[i], i) for i in range(n)]
    fracs.sort(reverse=True)

    for k in range(abs(diff)):
        idx = fracs[k % n][1]
        base[idx] += 1 if diff > 0 else -1

    return base


class ExecutionEnv:
    """
    Simple online execution environment for discrete Q-learning.
    State: (t, q_remaining_disc)
    Action: pick a fraction of cap (0, 25%, 50%, 75%, 100%)
    Reward: - (lambda_impact * a^2 + lambda_track * (a - target_t)^2)
    Terminal penalty if not fully executed.
    """

    def __init__(
        self,
        Q,
        volumes,
        participation_rate=1.0,
        lambda_impact=1.0,
        lambda_track=20.0,
        terminal_penalty=50000.0,
        q_bin=5,
        action_fracs=None
    ):
        self.Q = int(Q)
        self.volumes = [int(v) for v in volumes]
        self.N = len(self.volumes)

        self.participation_rate = float(participation_rate)
        self.lambda_impact = float(lambda_impact)
        self.lambda_track = float(lambda_track)
        self.terminal_penalty = float(terminal_penalty)

        self.q_bin = int(q_bin)
        self.action_fracs = action_fracs if action_fracs is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
        


        self.target = vwap_target(self.Q, self.volumes)

        self.t = 0
        self.q_remaining = self.Q

    def reset(self):
        self.t = 0
        self.q_remaining = self.Q
        return self._state()

    def _cap_at(self, t):
        cap = int(math.floor(self.participation_rate * self.volumes[t]))
        return max(0, cap)

    def _state(self):
        q_disc = int(self.q_remaining // self.q_bin)
        return (self.t, q_disc)

    def valid_actions(self):
        return list(range(len(self.action_fracs)))

    def step(self, action_idx):
        cap = self._cap_at(self.t)
        frac = self.action_fracs[action_idx]

        a = int(round(frac * cap))
        a = max(0, min(a, cap, self.q_remaining))

        tgt = self.target[self.t]

        impact_cost = a * a
        track_cost = (a - tgt) * (a - tgt)

        reward = -(self.lambda_impact * impact_cost + self.lambda_track * track_cost)

        self.q_remaining -= a
        self.t += 1
        done = (self.t >= self.N)

        if done and self.q_remaining != 0:
            reward -= self.terminal_penalty * (abs(self.q_remaining) / max(self.Q, 1))

        return self._state(), float(reward), done

    def rollout_greedy(self, Q_table):
        self.reset()
        slices = []

        for _ in range(self.N):
            s = self._state()
            qvals = Q_table.get(s)

            if qvals is None:
                a_idx = 0
            else:
                a_idx = max(range(len(qvals)), key=lambda i: qvals[i])

            cap = self._cap_at(self.t)
            a = int(round(self.action_fracs[a_idx] * cap))
            a = max(0, min(a, cap, self.q_remaining))

            slices.append(a)
            self.step(a_idx)

        return slices


__all__ = ["ExecutionEnv", "vwap_target"]
