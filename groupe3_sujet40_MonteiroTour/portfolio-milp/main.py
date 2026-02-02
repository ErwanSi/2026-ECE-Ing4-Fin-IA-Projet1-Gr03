import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from runner import RunConfig, load_returns, run_walkforward
from sweep import run_sweep
from present import latest_file, plot_pareto_from_csv, plot_wealth


# -----------------------------
# Pretty printing helpers
# -----------------------------
def _fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _pct(x, nd=2):
    try:
        return f"{100.0 * float(x):.{nd}f}%"
    except Exception:
        return str(x)


def _print_block(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def _cagr_from_wealth(wealth_series: pd.Series) -> float:
    # wealth starts near 1
    if wealth_series is None or len(wealth_series) < 5:
        return float("nan")
    w0 = float(wealth_series.iloc[0])
    wT = float(wealth_series.iloc[-1])
    n_days = len(wealth_series)
    years = n_days / 252.0
    if years <= 0 or w0 <= 0:
        return float("nan")
    return (wT / w0) ** (1.0 / years) - 1.0


def _calmar(cagr: float, maxdd: float) -> float:
    if maxdd >= 0 or np.isnan(cagr):
        return float("nan")
    return cagr / abs(maxdd)


def _hit_ratio(daily_returns: pd.Series) -> float:
    if daily_returns is None or len(daily_returns) == 0:
        return float("nan")
    return float((daily_returns > 0).mean())


def _ann_turnover(avg_turnover_weight: float, step_days: int) -> float:
    # avg_turnover_weight is per rebalance; annualise by number of rebalances per year
    return float(avg_turnover_weight) * (252.0 / float(step_days))


def _best_from_sweep(csv_path: str):
    try:
        df = pd.read_csv(csv_path)

        # Ensure columns exist
        if "turnover_cap_lots" not in df.columns:
            df["turnover_cap_lots"] = "None"

        df["sharpe"] = pd.to_numeric(df.get("sharpe"), errors="coerce")
        df["avg_turnover_weight"] = pd.to_numeric(df.get("avg_turnover_weight"), errors="coerce")
        df["lambda_cvar"] = pd.to_numeric(df.get("lambda_cvar"), errors="coerce")

        df = df.dropna(subset=["sharpe", "avg_turnover_weight", "lambda_cvar"]).copy()
        if df.empty:
            return None

        best = df.loc[df["sharpe"].idxmax()].to_dict()
        return best
    except Exception:
        return None


# -----------------------------
# MAIN (presentation entrypoint)
# -----------------------------
def main():
    # ---- Config (chosen for presentation: reasonable trade-off)
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

    # Sector diversification bounds (simple, transparent)
    sector_bounds = {
        "TECH": (0.00, 0.70),
        "FIN": (0.00, 0.30),
        "ENERGY": (0.00, 0.30),
        "HEALTH": (0.00, 0.30),
        "CONS": (0.00, 0.50),
        "OTHER": (0.00, 0.30),
    }

    cfg = RunConfig(
        tickers=tickers,
        sectors=sectors,
        sector_bounds=sector_bounds,
        start="2018-01-01",
        train_len=252 * 2,
        test_len=21,
        step=21,
        Q=1000,
        K=5,
        q_min=1,
        w_max=0.60,
        beta=0.95,
        lambda_cvar=0.3,          # tuned in sweep
        turnover_cap_lots=400,    # tuned in sweep
        cap_disabled_first_window=True,
        cost_rate=0.001,
        cps_time_limit=10.0,
        seed=42,
        n_scenarios=200,
    )

    # ---- 1) Run walk-forward (no plots inside runner)
    rets = load_returns(cfg)
    out = run_walkforward(cfg, rets=rets, save_curves=True)

    # Extract curves & reconstruct daily returns
    ew_w = out["curves"]["ew"]
    cps_w = out["curves"]["cps"]
    ew_r = ew_w.pct_change().dropna()
    cps_r = cps_w.pct_change().dropna()

    # Metrics & diagnostics
    ew_m = out["metrics"]["equal_weight"]
    cps_m = out["metrics"]["cpsat"]
    d = out["diagnostics"]

    # Extra KPIs
    ew_cagr = _cagr_from_wealth(ew_w)
    cps_cagr = _cagr_from_wealth(cps_w)
    ew_calmar = _calmar(ew_cagr, ew_m["maxDD"])
    cps_calmar = _calmar(cps_cagr, cps_m["maxDD"])
    ew_hit = _hit_ratio(ew_r)
    cps_hit = _hit_ratio(cps_r)

    cps_turn_ann = _ann_turnover(d["avg_turnover_weight"], step_days=cfg.step)
    cps_cost_ann = float(cfg.cost_rate) * cps_turn_ann  # approx annual cost drag

    # ---- 2) Ensure sweep exists (creates CSV+PNG but does NOT show)
    os.makedirs("runs", exist_ok=True)
    csv_latest = latest_file(os.path.join("runs", "sweep_*.csv"))
    if csv_latest is None:
        csv_latest, _ = run_sweep(out_dir="runs")

    best = _best_from_sweep(csv_latest)

    # ---- Terminal output (presentation-ready)
    _print_block("FINAL RUN — RESULTS (presentation-ready)")

    print("CONFIG")
    print(f"- Universe: {len(cfg.tickers)} tickers | K={cfg.K} | Q={cfg.Q} lots | q_min={cfg.q_min} | w_max={cfg.w_max}")
    print(f"- Walk-forward: train={cfg.train_len}d | test={cfg.test_len}d | step={cfg.step}d | scenarios={cfg.n_scenarios}")
    print(f"- Risk/Cost: beta(CVaR)={cfg.beta} | lambda={cfg.lambda_cvar} | turnover_cap_lots={cfg.turnover_cap_lots} | cost_rate={cfg.cost_rate}")
    print(f"- CP-SAT: time_limit={cfg.cps_time_limit}s | avg_solve_time={_fmt(d['avg_solve_time_s'],3)}s | pct_optimal={_pct(d['pct_optimal'],0)}")

    _print_block("PERFORMANCE (annualised approx)")
    header = f"{'Strategy':<18} {'CAGR':>10} {'Mean':>10} {'Vol':>10} {'Sharpe':>10} {'MaxDD':>10} {'Calmar':>10} {'Hit%':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'Equal-weight':<18} {_pct(ew_cagr):>10} {_pct(ew_m['mean']):>10} {_pct(ew_m['vol']):>10} {_fmt(ew_m['sharpe'],3):>10} {_pct(ew_m['maxDD']):>10} {_fmt(ew_calmar,3):>10} {_pct(ew_hit):>10}")
    print(f"{'CP-SAT CVaR':<18} {_pct(cps_cagr):>10} {_pct(cps_m['mean']):>10} {_pct(cps_m['vol']):>10} {_fmt(cps_m['sharpe'],3):>10} {_pct(cps_m['maxDD']):>10} {_fmt(cps_calmar,3):>10} {_pct(cps_hit):>10}")

    _print_block("TRADING / CONSTRAINTS (CP-SAT)")
    print(f"- Avg selected assets     : {d['avg_selected_assets']:.2f} (<=K={cfg.K})")
    print(f"- Avg turnover (lots)     : {d['avg_turnover_lots']:.2f}")
    print(f"- Avg turnover (weight)   : {d['avg_turnover_weight']:.4f} per rebalance")
    print(f"- Annualised turnover     : {cps_turn_ann:.2f} (approx)")
    print(f"- Avg cost (per rebalance): {d['avg_cost']:.6f}")
    print(f"- Annualised cost impact  : {cps_cost_ann:.4f} (approx)")
    print(f"- Statuses (tail)         : {d['statuses_tail']}")

    if best is not None:
        _print_block("SWEEP — BEST CONFIG (from latest sweep CSV)")
        print(f"- Best Sharpe : {_fmt(best.get('sharpe'),3)}")
        print(f"- lambda      : {best.get('lambda_cvar')}")
        print(f"- cap_lots     : {best.get('turnover_cap_lots')}")
        print(f"- avg_turnover : {_fmt(best.get('avg_turnover_weight'),4)}")

    # ---- SHOW ONLY TWO FIGURES (presentation requirement)
    plot_wealth(out["curves"])
    plt.show()

    plot_pareto_from_csv(csv_latest)
    plt.show()


if __name__ == "__main__":
    main()
