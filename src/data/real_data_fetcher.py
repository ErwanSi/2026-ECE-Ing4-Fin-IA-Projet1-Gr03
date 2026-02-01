import ccxt
import pandas as pd
import time
from datetime import datetime
import os

class BinanceDataFetcher:
    """
    Fetches historical market data from Binance using CCXT.
    """
    
    def __init__(self, symbol='BTC/USDT', timeframe='1m'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        
    def fetch_historical_data(self, limit=1000):
        """
        Fetch OHLCV data. Handles pagination for limits > 1000.
        Returns a DataFrame with ['time', 'open', 'high', 'low', 'close', 'volume'].
        """
        print(f"Fetching {limit} candles for {self.symbol} ({self.timeframe})...")
        
        # Calculate start time (approximate)
        # 1m = 60 seconds * 1000 ms
        timeframe_duration_ms = 60 * 1000 
        if self.timeframe == '1h': timeframe_duration_ms *= 60
        elif self.timeframe == '1d': timeframe_duration_ms *= 60 * 24
        
        now = self.exchange.milliseconds()
        since = now - (limit * timeframe_duration_ms)
        
        all_ohlcv = []
        fetched = 0
        
        try:
            while fetched < limit:
                batch_limit = min(1000, limit - fetched)
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since, limit=batch_limit)
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                fetched += len(ohlcv)
                print(f"Fetched {fetched}/{limit} candles...")
                
                # Update since to the timestamp of the last candle + 1 timeframe unit
                since = ohlcv[-1][0] + 1 # +1ms to avoid duplicates (though usually fine)
                
                # Rate limit sleep
                time.sleep(self.exchange.rateLimit / 1000.0)
                
                if len(ohlcv) < batch_limit:
                    break # No more data available
            
            # Trim if we got slightly more or duplicates (rare with since logic but safe)
            all_ohlcv = all_ohlcv[:limit]
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Save for caching
            os.makedirs("data", exist_ok=True)
            filename = f"data/{self.symbol.replace('/', '_')}_{self.timeframe}.csv"
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
            
            return df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Try to load cached
            filename = f"data/{self.symbol.replace('/', '_')}_{self.timeframe}.csv"
            if os.path.exists(filename):
                print("Loading cached data instead.")
                return pd.read_csv(filename)
            else:
                raise e

    def get_price_path(self, limit=1000):
        """
        Returns numpy array of close prices for simulation.
        """
        df = self.fetch_historical_data(limit)
        return df['close'].values
