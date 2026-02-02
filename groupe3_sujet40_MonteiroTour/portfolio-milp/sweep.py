import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from runner import RunConfig, load_returns, run_walkforward


def run_sweep(out_dir: str = "runs") -> tuple[str, str]:
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ", "NVDA", "PG"]
    sector_map = {
        "AAPL": "TECH", "MSFT": "TECH", "AMZN": "CONS", "GOOGL": "TECH", "META": "TECH",
        "NVDA": "TECH", "JPM": "FIN", "XOM": "ENERGY", "JNJ": "HEALTH", "PG": "CONS",
    }
    sectors = [sector_map[t] for t in tickers]
    sector_bounds = {
        "TECH": (0.00, 0.70),
        "FIN": (0.00, 0.30),
        "ENERGY": (0.00, 0.30),
        "HEALTH": (0.00, 0.30),
        "CONS": (0.00, 0.50),
        "OTHER": (0.00, 0.30),
    }

    turnover_caps = [None, 800, 400, 200, 100]
    lambdas = [0.1, 0.3, 1.0]

    base = RunConfig(
        tickers=tickers,
        sectors=sectors,
        sector_bounds=sector_bounds,
        lambda_cvar=0.3,          # overwritten
        turnover_cap_lots=400,    # overwritten
        cap_disabled_first_window=True,
        cost_rate=0.001,
        cps_time_limit=10.0,
        seed=42,
        n_scenarios=200,
    )

    print("Loading returns once for the sweep...")
    rets = load_returns(base)

    rows = []
    for cap in turnover_caps:
        for lam in lambdas:
            cfg = RunConfig(**{**base.__dict__, "turnover_cap_lots": cap, "lambda_cvar": lam})
            out = run_walkforward(cfg, rets=rets, save_curves=False)

            m = out["metrics"]["cpsat"]
            d = out["diagnostics"]

            rows.append({
                "turnover_cap_lots": "None" if cap is None else int(cap),
                "lambda_cvar": float(lam),
                "mean": m["mean"],
                "vol": m["vol"],
                "sharpe": m["sharpe"],
                "maxDD": m["maxDD"],
                "avg_turnover_lots": d["avg_turnover_lots"],
                "avg_turnover_weight": d["avg_turnover_weight"],
                "avg_cost": d["avg_cost"],
                "avg_selected_assets": d["avg_selected_assets"],
                "pct_optimal": d["pct_optimal"],
                "avg_solve_time_s": d["avg_solve_time_s"],
            })

    df = pd.DataFrame(rows)

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"sweep_{ts}.csv")
    df.to_csv(csv_path, index=False)

    # Plot Pareto: Sharpe vs turnover (weight)
    plt.figure(figsize=(9.5, 5.5))
    for lam in lambdas:
        sub = df[df["lambda_cvar"] == float(lam)].copy()

        def cap_sort(x):
            return 10**9 if str(x) == "None" else int(x)

        sub["cap_sort"] = sub["turnover_cap_lots"].apply(cap_sort)
        sub = sub.sort_values("cap_sort")

        plt.plot(sub["avg_turnover_weight"], sub["sharpe"], marker="o", label=f"lambda={lam}")

    plt.title("Trade-off: Sharpe vs average turnover (CP-SAT)")
    plt.xlabel("Avg turnover (weight)")
    plt.ylabel("Sharpe (annualised approx)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    png_path = os.path.join(out_dir, f"pareto_sharpe_turnover_{ts}.png")
    plt.savefig(png_path, dpi=170)
    plt.close()

    print(f"Saved sweep CSV : {os.path.abspath(csv_path)}")
    print(f"Saved Pareto PNG: {os.path.abspath(png_path)}")
    return csv_path, png_path


if __name__ == "__main__":
    run_sweep()
