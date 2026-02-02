import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from runner import RunConfig, load_returns, run_walkforward


def main():
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ", "NVDA", "PG"]

    sector_map = {
        "AAPL": "TECH",
        "MSFT": "TECH",
        "AMZN": "CONS",
        "GOOGL": "TECH",
        "META": "TECH",
        "NVDA": "TECH",
        "JPM": "FIN",
        "XOM": "ENERGY",
        "JNJ": "HEALTH",
        "PG": "CONS",
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

    # ---- GRID (bon ratio effort/note)
    turnover_caps = [None, 800, 400, 200, 100]  # None = pas de cap
    lambdas = [0.1, 0.3, 1.0]

    base_cfg = RunConfig(
        tickers=tickers,
        sectors=sectors,
        sector_bounds=sector_bounds,
        train_len=252 * 2,
        test_len=21,
        step=21,
        Q=1000,
        K=5,
        q_min=1,
        w_max=0.60,
        beta=0.95,
        lambda_cvar=1.0,  # overwritten in loop
        turnover_cap_lots=200,  # overwritten in loop
        cap_disabled_first_window=True,
        cost_rate=0.001,
        cps_time_limit=10.0,
        seed=42,
    )

    print("Loading returns once...")
    rets = load_returns(base_cfg)

    rows = []
    for cap in turnover_caps:
        for lam in lambdas:
            cfg = RunConfig(**{**base_cfg.__dict__, "turnover_cap_lots": cap, "lambda_cvar": lam})
            print(f"Running cap={cap}, lambda={lam} ...")
            out = run_walkforward(cfg, rets=rets, save_curves=False)

            m = out["metrics"]["cpsat"]
            d = out["diagnostics"]

            rows.append({
                "turnover_cap_lots": cap if cap is not None else "None",
                "lambda_cvar": lam,
                "mean": m["mean"],
                "vol": m["vol"],
                "sharpe": m["sharpe"],
                "maxDD": m["maxDD"],
                "avg_turnover_lots": d["avg_turnover_cps_lots"],
                "avg_turnover_weight": d["avg_turnover_cps_weight"],
                "avg_cost": d["avg_cost_cps"],
                "avg_selected_assets": d["avg_selected_assets_cps"],
                "pct_optimal": d["pct_optimal_cps"],
            })

    df = pd.DataFrame(rows)

    os.makedirs("runs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join("runs", f"sweep_{ts}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved sweep CSV: {os.path.abspath(out_csv)}")

    # ---- Plot: Sharpe vs turnover (weight)
    plt.figure(figsize=(9.5, 5.5))
    for lam in lambdas:
        sub = df[df["lambda_cvar"] == lam].copy()
        # ensure numeric for sorting: treat "None" as very large cap
        def cap_sort(x):
            return 10**9 if x == "None" else int(x)
        sub["cap_sort"] = sub["turnover_cap_lots"].apply(cap_sort)
        sub = sub.sort_values("cap_sort")

        plt.plot(sub["avg_turnover_weight"], sub["sharpe"], marker="o", label=f"lambda={lam}")

    plt.title("Trade-off: Sharpe vs average turnover (CP-SAT)")
    plt.xlabel("Avg turnover (weight)")
    plt.ylabel("Sharpe (annualised approx)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join("runs", f"pareto_sharpe_turnover_{ts}.png")
    plt.savefig(out_png, dpi=170)
    print(f"Saved plot: {os.path.abspath(out_png)}")
    plt.show(block=True)


if __name__ == "__main__":
    main()
