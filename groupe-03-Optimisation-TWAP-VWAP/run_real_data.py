from datetime import datetime, date
import numpy as np
import pandas as pd

import yfinance as yf
import matplotlib.pyplot as plt

from src.strategies.twap import twap_schedule
from src.strategies.vwap import vwap_schedule
from src.strategies.constrained_opt_cp import constrained_opt_cp_schedule


def impact_cost(x, volumes, eps=1e-9):
    return sum((x[i] * x[i]) / (volumes[i] + eps) for i in range(len(x)))


def tracking_cost_l2(x, target):
    return sum((x[i] - target[i]) ** 2 for i in range(len(x)))


def vwap_target(Q, volumes):
    tot = sum(volumes)
    raw = [Q * v / tot for v in volumes]
    base = [int(r) for r in raw]
    diff = Q - sum(base)

    fracs = sorted([(raw[i] - int(raw[i]), i) for i in range(len(volumes))], reverse=True)

    for k in range(abs(diff)):
        idx = fracs[k][1]
        base[idx] += 1 if diff > 0 else -1

    return base


def yahoo_volumes_into_slices(
    ticker: str,
    day: str,
    N: int,
    interval: str = "1m",
    market_tz: str = "America/New_York",
):
    d0 = pd.Timestamp(day).tz_localize(market_tz)
    d1 = d0 + pd.Timedelta(days=1)

    df = yf.download(
        tickers=ticker,
        start=d0.tz_convert("UTC").tz_localize(None),
        end=d1.tz_convert("UTC").tz_localize(None),
        interval=interval,
        auto_adjust=False,
        progress=False,
        prepost=False,
        group_by="column",
        threads=True,
    )

    if df is None or df.empty:
        raise RuntimeError(f"Aucune donnée intraday récupérée pour {ticker} le {day}.")

    df.index = pd.to_datetime(df.index, utc=True).tz_convert(market_tz)
    df = df.between_time("09:30", "16:00")

    if "Volume" not in df.columns:
        if isinstance(df.columns, pd.MultiIndex):
            vol_cols = [c for c in df.columns if (isinstance(c, tuple) and "Volume" in c)]
            if vol_cols:
                df["Volume"] = df[vol_cols[0]]
            else:
                raise RuntimeError("Colonne Volume introuvable dans les données Yahoo.")
        else:
            raise RuntimeError("Colonne Volume introuvable dans les données Yahoo.")

    vol_series = df["Volume"].dropna()
    if vol_series.empty:
        raise RuntimeError(f"Volume vide après filtrage heures de marché pour {ticker} le {day}.")

    vols = vol_series.to_numpy(dtype=float)
    splits = np.array_split(vols, N)
    slice_volumes = [int(np.round(s.sum())) for s in splits]
    return [max(1, v) for v in slice_volumes]


def yahoo_live_price(ticker: str):
    t = yf.Ticker(ticker)
    try:
        price = t.fast_info["last_price"]
        if price is not None:
            return float(price)
    except Exception:
        pass

    hist = t.history(period="1d", interval="1m")
    if hist.empty:
        raise RuntimeError(f"Impossible de récupérer le prix live pour {ticker}")
    return float(hist["Close"].iloc[-1])


def plot_results(volumes, target, schedules_dict):
    """
    Graphiques:
      (1) Impact global (bar)
      (2) Sous-ordres par slice + liquidité (volume marché sur axe secondaire)
      (3) Tracking par slice (x_t - target_t)^2
    """
    t = np.arange(1, len(volumes) + 1)
    names = list(schedules_dict.keys())
    impacts = [impact_cost(schedules_dict[n], volumes) for n in names]

    # (1) Impact global
    plt.figure()
    plt.bar(names, impacts)
    plt.title("Impact cost (global)")
    plt.ylabel("sum(x_t^2 / volume_t)")
    plt.xticks(rotation=15)
    plt.tight_layout()

    # (2) Sous-ordres + liquidité
    fig, ax1 = plt.subplots()
    for n in names:
        ax1.plot(t, schedules_dict[n], marker="o", label=n)
    ax1.set_title("Sous-ordres par slice vs liquidité marché")
    ax1.set_xlabel("Slice t")
    ax1.set_ylabel("Shares exécutées (x_t)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(t, volumes, linestyle="--", marker="x")
    ax2.set_ylabel("Volume marché (shares)")
    ax2.grid(False)
    plt.tight_layout()

    # (3) Tracking par slice
    plt.figure()
    for n in names:
        err2 = [(schedules_dict[n][i] - target[i]) ** 2 for i in range(len(target))]
        plt.plot(t, err2, marker="o", label=n)
    plt.title("Tracking par slice : (x_t - target_t)^2")
    plt.xlabel("Slice t")
    plt.ylabel("Erreur^2")
    plt.legend()
    plt.tight_layout()

    plt.show()


def apply_cap_overrides(caps_vec, overrides_1based):
    """
    overrides_1based : dict {slice_index (1..N): cap_value}
    Exemple: {2: 100, 5: 250}
    """
    for t1, cap in overrides_1based.items():
        idx = t1 - 1  # conversion 1-based -> 0-based
        if 0 <= idx < len(caps_vec):
            caps_vec[idx] = min(caps_vec[idx], int(cap))
    return caps_vec


def main():
    ticker = "AAPL"
    trading_day = "2026-01-30"
    N = 10
    Q = 3400

    volumes = yahoo_volumes_into_slices(
        ticker=ticker,
        day=trading_day,
        N=N,
        interval="1m",
        market_tz="America/New_York",
    )

    live_price = yahoo_live_price(ticker)

    twap = twap_schedule(Q=Q, N=N, max_per_slice=None).slices
    vwap = vwap_schedule(Q=Q, volumes=volumes, participation_rate=None, caps=None).slices
    target = vwap_target(Q, volumes)

    participation_rate_csp = 0.20
    caps_vec = [int(participation_rate_csp * v) for v in volumes]

    # ✅ Ici tu peux mettre plusieurs caps custom (slices en 1..N)
    cap_overrides = {
        2: 100,   # slice 2 capée à 100
        5: 220,   # slice 5 capée à 250
        7: 90,   # exemple no-trade total sur slice 7
    }
    caps_vec = apply_cap_overrides(caps_vec, cap_overrides)

    if Q > sum(caps_vec):
        print("❌ INFEASIBLE CSP")
        print("Q =", Q)
        print("sum caps =", sum(caps_vec))
        print("caps_vec =", caps_vec)
        return

    res_mix = constrained_opt_cp_schedule(
        Q=Q,
        volumes=volumes,
        caps=caps_vec,
        w_impact=1,
        w_track=2000,
        time_limit_s=30.0,
    )

    print("\n--------------------------------------------------------------------------")
    print(f"Stock = {ticker} | volumes day = {trading_day} | interval=1m -> slices N={N}")
    print(f"Live Price = {live_price:.2f} USD")
    print("Q =", Q)
    print("volumes =", volumes)
    print("max participation_rate du volume=", caps_vec)
    print()

    rows = [
        ("TWAP", twap),
        ("VWAP (ideal)", vwap),
        ("OPT (CSP)", res_mix.slices),
    ]

    schedules = {}
    for name, x in rows:
        schedules[name] = x
        if not x:
            print(f"{name:24} -> []")
            continue

        print(
            f"{name:24} -> {x} | sum={sum(x)} "
            f"| impact={impact_cost(x, volumes):.6f} "
            f"| track(L2)={tracking_cost_l2(x, target)}"
        )

    plot_results(volumes, target, schedules)


if __name__ == "__main__":
    main()
