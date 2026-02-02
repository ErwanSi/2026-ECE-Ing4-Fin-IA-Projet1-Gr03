import pickle
from src.rl.env import ExecutionEnv, vwap_target

from src.strategies.twap import twap_schedule
from src.strategies.vwap import vwap_schedule
from src.strategies.constrained_opt_cp import constrained_opt_cp_schedule


def impact_cost(x):
    return sum(v * v for v in x)

def track_cost(x, target):
    return sum((x[i] - target[i]) ** 2 for i in range(len(x)))


def main():
    Q = 200
    volumes = [10, 40, 200, 100, 100]
    participation_rate = 1

    target = vwap_target(Q, volumes)

    twap = twap_schedule(Q=Q, N=len(volumes)).slices
    vwap = vwap_schedule(Q=Q, volumes=volumes, participation_rate=participation_rate).slices
    opt = constrained_opt_cp_schedule(
        Q=Q, volumes=volumes, participation_rate=participation_rate, w_impact=1, w_track=20
    ).slices

    with open("rl_qtable.pkl", "rb") as f:
        Q_table = pickle.load(f)

    env = ExecutionEnv(
        Q=Q,
        volumes=volumes,
        participation_rate=participation_rate,
        lambda_impact=1.0,
        lambda_track=20.0,
        terminal_penalty=50000.0,
        q_bin=5,
    )

    rl_slices = env.rollout_greedy(Q_table)

    print("Volumes:", volumes)
    print("VWAP target:", target)
    print()

    rows = [
        ("TWAP", twap),
        ("VWAP", vwap),
        ("OPT (CP)", opt),
        ("RL (Q-learn)", rl_slices),
    ]

    for name, x in rows:
        print(f"{name:12} -> {x} | sum={sum(x)} impact={impact_cost(x)} track={track_cost(x, target)}")


if __name__ == "__main__":
    main()
