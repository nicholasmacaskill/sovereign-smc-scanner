import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
# from smartmoneyconcepts import smc  <-- Removed unused dependency
import ccxt
import ccxt
import time
from datetime import datetime, time as time_obj
from src.core.config import Config
from src.engines.intermarket_engine import IntermarketEngine
from src.engines.news_filter import NewsFilter
from src.engines.visualizer import generate_bias_chart
from src.engines.ai_validator import AIValidator
import logging
import os
import functools
from src.core.database import log_system_event

logger = logging.getLogger(__name__)

def ensure_data(default_return=None):
    """Decorator to ensure df is valid before running analysis"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, df, *args, **kwargs):
            if df is None or len(df) < 5:
                return default_return
            try:
                return func(self, df, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator

def safe_scan(component):
    """Decorator to catch and log errors in high-level scanning methods"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                import traceback
                symbol = args[0] if args else "Unknown"
                err_msg = f"{component} error ({symbol}): {str(e)}\n{traceback.format_exc()}"
                logger.error(err_msg)
                log_system_event(component, err_msg, level="ERROR")
                return None
        return wrapper
    return decorator

class SMCScanner:
    def __init__(self):
        # Initialize public exchange for data fetching (free tier)
        # Using Coinbase (Advanced Trade) to avoid Binance geo-restrictions
        try:
            self.exchange = ccxt.coinbase({'enableRateLimit': True})
        except Exception:
            # Fallback to standard coinbase if 'coinbase' alias refers to old API in this version
            # But usually 'coinbase' is the correct one for public market data now
            self.exchange = ccxt.coinbasepro({'enableRateLimit': True})
            
        self.intermarket = IntermarketEngine()
        self.news = NewsFilter()
        self.order_book_enabled = True  # Can be disabled if exchange doesn't support

    def get_hurst_exponent(self, time_series):
        # [REDACTED] Proprietary Geometric Persistence Math
        return 0.5
    def get_adf_test(self, time_series):
        """
        Performs Augmented Dickey-Fuller test to check for stationarity.
        p-value < 0.05 indicates the series is stationary (Mean Reverting).
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(time_series)
            return result[1] # p-value
        except ImportError:
            return 1.0 # Default to non-stationary
        except Exception as e:
            logger.error(f"ADF Test Error: {e}")
            return 1.0
        
    @ensure_data(default_return=pd.Series(dtype=float))
    def calculate_atr(self, df, period=14):
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()

    @ensure_data(default_return=pd.Series(dtype=float))
    def calculate_rsi(self, df, period=14):
        """Standard RSI Calculation"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def fetch_data(self, symbol, timeframe, limit=500):
        """
        Fetches candle data.
        Primary: CCXT (Binance) - Real-time, fast.
        Fallback: yfinance - Robust, no IP blocking, slightly delayed.
        """
        # Try CCXT First (Real-Time)
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
        except Exception as e:
            logger.warning(f"CCXT Fetch failed for {symbol} ({e}). Falling back to yfinance.")

        # Fallback to yfinance
        try:
            # Map symbol to yfinance format (BTC/USD -> BTC-USD)
            yf_symbol = symbol.replace('/', '-') if '/' in symbol else symbol
            # Ensure USDT is converted to USD for yfinance just in case
            if 'USDT' in yf_symbol: yf_symbol = yf_symbol.replace('USDT', 'USD')
            
            # Map timeframe to yfinance format
            interval_map = {'5m': '5m', '15m': '15m', '1h': '1h', '4h': '60m', '1d': '1d'} 
            yf_interval = interval_map.get(timeframe, '5m')
            
            # Fetch data (5 days is safe buffer for indicators)
            import yfinance as yf
            df = yf.download(yf_symbol, period='5d', interval=yf_interval, progress=False)
            
            if df is None or len(df) < 50:
                logger.error(f"yfinance fetched insufficient data for {symbol}")
                return None
                
            # Handle MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Normalize columns
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df.rename(columns={'date': 'timestamp', 'datetime': 'timestamp'}, inplace=True)
            
            # Ensure timestamp is tz-naive for consistent comparison
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            
            # Remove duplicate columns (Fix for ValueError: Cannot set a DataFrame with multiple columns)
            df = df.loc[:, ~df.columns.duplicated()]
            
            return df

        except Exception as e:
            logger.error(f"Error fetching data via yfinance for {symbol}: {e}")
            return None

    @ensure_data(default_return=(pd.Series(dtype=bool), pd.Series(dtype=bool)))
    def detect_fractals(self, df, window=2):
        """
        Vectorized fractal detection using NumPy.
        Returns boolean masks for Swing Highs and Lows.
        """
        # Fractal High
        is_high = df['high'].rolling(window=2*window+1, center=True).max() == df['high']
        # Fractal Low
        is_low = df['low'].rolling(window=2*window+1, center=True).min() == df['low']
        
        return is_high, is_low

    def is_killzone(self, current_time=None):
        """Checks if current time (or override) is within any active trading session."""
        now_utc = current_time.time() if current_time else datetime.utcnow().time()
        hour = now_utc.hour
        
        # Check Asian Fade Prime Window (⭐ Highest priority)
        if self.is_asian_fade_window(hour):
            return True

        # Check London Session
        london = Config.KILLZONE_LONDON
        if london and (london[0] <= hour < london[1]):
            return True

        # Check continuous NY session
        ny_session = Config.KILLZONE_NY_CONTINUOUS
        if ny_session and (ny_session[0] <= hour < ny_session[1]):
            return True
            
        # Check Asia Session
        asia = Config.KILLZONE_ASIA
        if asia and (asia[0] <= hour < asia[1]):
            return True
        
        return False

    def is_asian_fade_window(self, hour=None):
        """Returns True if we are in the 11 PM – 2 AM EST (4–7 AM UTC) Asian Fade prime window."""
        if hour is None:
            hour = datetime.utcnow().hour
        fade = Config.KILLZONE_ASIAN_FADE
        return fade is not None and (fade[0] <= hour < fade[1])

    def scan_asian_fade(self, symbol):
        """
        [REDACTED] Proprietary Asian Range Fade Alpha.
        """
        return None
    def get_detailed_bias(self, symbol, index_context=None, visual_check=False):
        """
        Calculates Multi-Factor Bias using proprietary signal inputs.
        Returns: Bias String (BULLISH/BEARISH/NEUTRAL)
        """
        # [REDACTED] Proprietary Geometric Logic
        return "NEUTRAL"  # Placeholder for public repo
    def get_4h_bias(self, symbol):
        # Legacy wrapper
        return self.get_detailed_bias(symbol).split(" ")[-1] # Returns BULLISH/BEARISH/NEUTRAL

    def get_session_quartile(self, current_time=None):
        """
        [REDACTED] Proprietary Session Cycle Logic.
        """
        return {}
    def get_price_quartiles(self, symbol):
        """
        [REDACTED] Proprietary Price Quartile Calculation.
        """
        return {}
    def validate_sweep_depth(self, symbol, swept_level, direction):
        """
        [REDACTED] Institutional Order Book Absorption Validation.
        """
        return True
    def calculate_atr(self, df, period=14):
        """
        Calculate Average True Range for volatility-adjusted targeting.
        
        Args:
            df: OHLCV dataframe
            period: ATR period (default 14)
        
        Returns:
            pandas Series with ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def get_volatility_adjusted_target(self, df, direction, entry_price, session_range):
        """
        [REDACTED] Dynamic Targeted Alpha Logic.
        """
        return 0.0
    def get_next_institutional_target(self, df, direction, entry_price):
        """
        [REDACTED] Recursively Scans for Draw on Liquidity.
        """
        return None
    def is_tapping_fvg(self, df, direction):
        """
        [REDACTED] Fair Value Gap Neutralization Logic.
        """
        return False
    def scan_pattern(self, symbol, timeframe='5m', cached_context=None, provided_df=None, current_time_override=None):
        """
        Main Scanning Function.
        [REDACTED] Core Logic Hidden for Public Release.
        """
        return None
    def scan_order_flow(self, symbol, timeframe=Config.TIMEFRAME):
        """
        [REDACTED] Institutional Order Flow Logic.
        """
        return None
    def detect_mss(self, df, lookback=50):
        # [REDACTED] Proprietary Market Structure Shift Detection
        return None
    def find_order_block(self, df, origin_index, direction):
        # [REDACTED] Proprietary Order Block Identification
        return None
