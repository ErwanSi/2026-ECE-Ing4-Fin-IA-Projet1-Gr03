from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import yfinance as yf


def load_recent_intraday(
    symbol: str,
    interval: str = "1m",
    period: str = "1d",
    last_n: int = 30,
) -> Tuple[List[float], List[int]]:
    """
    Load recent intraday bars (close, volume).
    If 1m fetch fails or is empty, fallback to 5m.
    """

    def _download(sym: str, inter: str, per: str) -> pd.DataFrame:
        return yf.download(sym, interval=inter, period=per, progress=False, threads=False)

    # Try primary
    df = _download(symbol, interval, period)

    # If empty, fallback
    if df is None or df.empty:
        # 1m sometimes fails; fallback to 5m over longer period
        df = _download(symbol, "5m", "5d")
        interval = "5m"
        period = "5d"

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {symbol}. Try another symbol or interval.")

    # Normalize columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if "Close" not in df.columns or "Volume" not in df.columns:
        raise RuntimeError(f"Unexpected columns from yfinance: {df.columns.tolist()}")

    df = df.dropna(subset=["Close", "Volume"]).tail(last_n)

    prices = df["Close"].astype(float).tolist()
    volumes = df["Volume"].fillna(0).astype(int).tolist()

    if len(prices) == 0:
        raise RuntimeError("Not enough bars after cleaning.")

    # Useful debug info (optional)
    # print(f"Loaded {len(volumes)} bars for {symbol} using interval={interval}, period={period}")

    return prices, volumes
