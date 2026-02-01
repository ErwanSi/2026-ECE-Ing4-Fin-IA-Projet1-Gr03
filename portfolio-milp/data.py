import pandas as pd
import yfinance as yf


def load_prices(tickers, start="2021-01-01", end=None):
    """
    Download adjusted close prices using yfinance.
    Settings chosen to reduce Windows 'database is locked' issues.
    """
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,  # important on Windows
    )["Close"]

    # Ensure DataFrame even if single ticker
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Drop tickers with no data
    df = df.dropna(axis=1, how="all")

    # Forward fill then drop remaining missing
    df = df.ffill().dropna()

    return df


def returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple arithmetic returns."""
    return prices.pct_change().dropna()
