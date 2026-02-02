from src.strategies.twap import twap_schedule
from src.strategies.vwap import vwap_schedule
from src.strategies.constrained_opt_cp import constrained_opt_cp_schedule


def impact_cost(x, volumes, eps=1e-9):
    # Impact proxy : sum(x_t^2 / volume_t)
    return sum((x[i] * x[i]) / (volumes[i] + eps) for i in range(len(x)))


def tracking_cost_l2(x, target):
    # Tracking "classique" : somme des erreurs au carré
    return sum((x[i] - target[i]) ** 2 for i in range(len(x)))


def vwap_target(Q, volumes):
    tot = sum(volumes)
    raw = [Q * v / tot for v in volumes]
    base = [int(r) for r in raw]
    diff = Q - sum(base)

    fracs = sorted(
        [(raw[i] - int(raw[i]), i) for i in range(len(volumes))],
        reverse=True
    )

    for k in range(abs(diff)):
        idx = fracs[k][1]
        base[idx] += 1 if diff > 0 else -1

    return base


def main():
    Q = 340
    volumes = [10, 40, 200, 100, 100, 239, 20, 78, 286]
    N = len(volumes)

    # -----------------------
    # Baselines
    # -----------------------
    twap = twap_schedule(Q=Q, N=N, max_per_slice=None).slices
    vwap = vwap_schedule(
        Q=Q,
        volumes=volumes,
        participation_rate=None,  # ✅ VWAP idéal, sans contrainte
        caps=None
    ).slices

    target = vwap_target(Q, volumes)

    # -----------------------
    # CSP constraints (réalistes)
    # -----------------------
    participation_rate_csp = 0.40
    caps_vec = [int(participation_rate_csp * v) for v in volumes]

    # Fenêtre de news / no-trade partiel sur slice 2
    caps_vec[2] = min(caps_vec[2], 15)

    if Q > sum(caps_vec):
        print("❌ INFEASIBLE CSP")
        print("Q =", Q)
        print("sum caps =", sum(caps_vec))
        print("caps_vec =", caps_vec)
        return

    # -----------------------
    # CSP optimisation (impact + tracking)
    # -----------------------
    res_mix = constrained_opt_cp_schedule(
        Q=Q,
        volumes=volumes,
        caps=caps_vec,
        w_impact=1,
        w_track=30,     # tracking L1 vers VWAP
        time_limit_s=30.0,
    )

    # -----------------------
    # Affichage
    # -----------------------
    print("\n--------------------------------------------------------------------------")
    print("Q =", Q)
    print("volumes =", volumes)
    print("VWAP target (ideal) =", target)
    print("CSP caps_vec =", caps_vec)
    print("OPT status:", res_mix.message)
    print()

    rows = [
        ("TWAP", twap),
        ("VWAP (no cap)", vwap),
        ("OPT impact+track (CSP)", res_mix.slices),
    ]

    for name, x in rows:
        if not x:
            print(f"{name:24} -> []")
            continue

        print(
            f"{name:24} -> {x} | sum={sum(x)} "
            f"| impact={impact_cost(x, volumes):.6f} "
            f"| track(L2)={tracking_cost_l2(x, target)}"
        )


if __name__ == "__main__":
    main()
