import random
from src.rl.env import ExecutionEnv
from src.rl.qlearning import qlearning_train
import pickle

def main():
    Q = 200
    base = [10, 40, 200, 100, 100]

    # on va entraîner sur plusieurs envs et merge dans une seule Q-table
    Q_table = {}

    for k in range(20):  # 20 profils différents
        volumes = [max(1, int(v * random.uniform(0.7, 1.3))) for v in base]

        env = ExecutionEnv(
            Q=Q,
            volumes=volumes,
            participation_rate=1.0,
            lambda_impact=1.0,
            lambda_track=20.0,
            terminal_penalty=10000.0,
            q_bin=5,
        )

        Q_part = qlearning_train(env, episodes=10000, seed=42+k)

        # merge (simple): overwrite, ou average si tu veux faire mieux
        Q_table.update(Q_part)

    with open("rl_qtable.pkl", "wb") as f:
        pickle.dump(Q_table, f)

    print("Training done. Saved rl_qtable.pkl")

if __name__ == "__main__":
    main()
