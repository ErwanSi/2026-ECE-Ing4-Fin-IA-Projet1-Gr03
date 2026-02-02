import glob
import os

import pandas as pd
import matplotlib.pyplot as plt


def latest_file(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def plot_wealth(curves):
    ew = curves["ew"]
    cps = curves["cps"]

    plt.figure(figsize=(11, 5.5))
    plt.plot(ew.index, ew.values, label="Equal-weight")
    plt.plot(cps.index, cps.values, label="CP-SAT CVaR (lots + K + sector + TC)")
    plt.title("OOS Walk-forward performance (cumulative)")
    plt.xlabel("Date")
    plt.ylabel("Wealth (start=1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_pareto_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)

    # Required columns
    required = {"lambda_cvar", "avg_turnover_weight", "sharpe"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Got columns={list(df.columns)}")

    # turnover_cap_lots might be absent in old sweeps
    if "turnover_cap_lots" not in df.columns:
        df["turnover_cap_lots"] = "None"

    # Parse numeric columns robustly
    df["lambda_cvar"] = pd.to_numeric(df["lambda_cvar"], errors="coerce")
    df["avg_turnover_weight"] = pd.to_numeric(df["avg_turnover_weight"], errors="coerce")
    df["sharpe"] = pd.to_numeric(df["sharpe"], errors="coerce")

    # Treat NaN/None/blank as "None" (i.e. very large cap)
    def cap_sort(x):
        if pd.isna(x):
            return 10**9
        sx = str(x).strip()
        if sx.lower() in ("none", "nan", ""):
            return 10**9
        try:
            return int(float(sx))  # handles "200.0"
        except Exception:
            return 10**9

    # Drop broken rows
    df = df.dropna(subset=["lambda_cvar", "avg_turnover_weight", "sharpe"]).copy()

    plt.figure(figsize=(9.5, 5.5))
    for lam in sorted(df["lambda_cvar"].dropna().unique()):
        sub = df[df["lambda_cvar"] == lam].copy()
        sub["cap_sort"] = sub["turnover_cap_lots"].apply(cap_sort)
        sub = sub.sort_values("cap_sort")

        plt.plot(sub["avg_turnover_weight"], sub["sharpe"], marker="o", label=f"lambda={lam}")

    plt.title("Trade-off: Sharpe vs average turnover (CP-SAT)")
    plt.xlabel("Avg turnover (weight)")
    plt.ylabel("Sharpe (annualised approx)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
