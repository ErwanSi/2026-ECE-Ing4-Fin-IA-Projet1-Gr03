import random

def qlearning_train(
    env,
    episodes=4000,
    alpha=0.15,
    gamma=0.98,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    seed=42
):
    random.seed(seed)

    n_actions = len(env.valid_actions())
    Q = {}
    epsilon = float(epsilon_start)

    def ensure_state(s):
        if s not in Q:
            Q[s] = [0.0] * n_actions

    for _ in range(episodes):
        s = env.reset()
        ensure_state(s)
        done = False

        while not done:
            if random.random() < epsilon:
                a = random.randrange(n_actions)
            else:
                a = max(range(n_actions), key=lambda i: Q[s][i])

            s2, r, done = env.step(a)
            ensure_state(s2)

            best_next = max(Q[s2])
            Q[s][a] = (1 - alpha) * Q[s][a] + alpha * (r + gamma * best_next)

            s = s2

        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return Q
