import os
import time
import hashlib
import pandas as pd
import yfinance as yf


def _cache_key(tickers: list[str], start: str) -> str:
    payload = (start + "|" + ",".join(tickers)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _cache_path(tickers: list[str], start: str) -> str:
    os.makedirs("data_cache", exist_ok=True)
    key = _cache_key(tickers, start)
    return os.path.join("data_cache", f"prices_{start}_{key}.csv")


def _load_cache(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        px = pd.read_csv(path, index_col=0, parse_dates=True)
        # basic sanity
        if px is None or px.empty:
            return None
        return px
    except Exception:
        return None


def _save_cache(px: pd.DataFrame, path: str) -> None:
    try:
        px.to_csv(path)
    except Exception:
        # cache is a robustness feature; never crash on cache save
        pass


def download_prices(tickers: list[str], start: str) -> pd.DataFrame:
    """
    Robust price loader:
    1) Try yfinance (auto_adjust=True) with retries.
    2) If yfinance fails, fallback to last cached prices if available.

    Returns a DataFrame indexed by date, columns=tickers, with no NaNs.
    """
    cache_path = _cache_path(tickers, start)

    # Try yfinance download with retries (most common fix for transient yfinance issues)
    last_err = None
    for attempt in range(1, 4):
        try:
            df = yf.download(
                tickers=tickers,
                start=start,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=False,  # important: avoids some yfinance multi-thread edge bugs
            )

            if df is None or len(df) == 0:
                raise RuntimeError("yfinance returned empty dataframe")

            # Handle multi-index columns (common for multiple tickers)
            if isinstance(df.columns, pd.MultiIndex):
                if "Adj Close" in df.columns.get_level_values(0):
                    px = df["Adj Close"].copy()
                elif "Close" in df.columns.get_level_values(0):
                    px = df["Close"].copy()
                else:
                    # fallback: first level
                    px = df.xs(df.columns.levels[0][0], axis=1, level=0).copy()
            else:
                px = df.copy()

            # Keep only requested tickers that are present
            cols = [t for t in tickers if t in px.columns]
            px = px.loc[:, cols].dropna(how="all")

            if px is None or px.empty or px.shape[1] == 0:
                raise RuntimeError("yfinance returned no usable price columns")

            # Strict cleaning: keep only complete rows
            px = px.dropna(how="any")

            # If still too small, consider it a failure
            if px.shape[0] < 50:
                raise RuntimeError("downloaded price history too short after cleaning")

            # Cache and return
            _save_cache(px, cache_path)
            return px

        except Exception as e:
            last_err = e
            # short backoff
            time.sleep(1.0 * attempt)

    # If we reach here, yfinance failed 3 times -> fallback to cache
    px_cache = _load_cache(cache_path)
    if px_cache is not None:
        print(f"[WARN] yfinance failed; using cached prices: {os.path.abspath(cache_path)}")
        return px_cache

    # No cache available -> hard fail with clear error
    raise RuntimeError(
        "yfinance returned no data and no cache was available.\n"
        f"- Last error: {repr(last_err)}\n"
        "Fixes:\n"
        "1) Re-run later (Yahoo/yfinance can fail temporarily)\n"
        "2) Ensure you have internet access / no firewall blocking\n"
        "3) Delete and recreate venv, then reinstall requirements\n"
    )
