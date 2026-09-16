"""
Market Data Collector with local parquet caching and incremental update support.
Uses FinanceDataReader with truststore SSL injection for robust Windows execution.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import pandas as pd
import numpy as np
import FinanceDataReader as fdr

from src.database.models import CACHE_DIR

BENCHMARK_CODE = "069500"  # KODEX 200


def get_cache_path(code: str) -> str:
    """Returns absolute path to cached parquet file for given stock code."""
    return os.path.join(CACHE_DIR, f"{code}.parquet")


def fetch_ohlcv(
    code: str,
    start: str = "2017-01-01",
    end: Optional[str] = None,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Fetches adjusted OHLCV for given stock code.
    Loads from parquet cache if present, fetching only missing dates incrementally.
    """
    cache_path = get_cache_path(code)
    today_str = datetime.now().strftime("%Y-%m-%d")
    target_end = end or today_str

    df_cached: Optional[pd.DataFrame] = None

    if not force_refresh and os.path.exists(cache_path):
        try:
            df_cached = pd.read_parquet(cache_path)
            if not df_cached.empty:
                df_cached.index = pd.to_datetime(df_cached.index)
                last_cached_date = df_cached.index.max().strftime("%Y-%m-%d")

                # If cache already covers target_end or is within 1 day (e.g. today/yesterday)
                if last_cached_date >= target_end:
                    filtered = df_cached.loc[start:target_end]
                    return filtered

                # Incremental fetch from day after last cached date
                next_day = (df_cached.index.max() + timedelta(days=1)).strftime("%Y-%m-%d")
                if next_day <= target_end:
                    df_new = fdr.DataReader(code, next_day, target_end)
                    if df_new is not None and not df_new.empty:
                        df_new.index = pd.to_datetime(df_new.index)
                        df_combined = pd.concat([df_cached, df_new])
                        df_combined = df_combined[~df_combined.index.duplicated(keep="last")].sort_index()
                        df_combined.to_parquet(cache_path)
                        return df_combined.loc[start:target_end]

                return df_cached.loc[start:target_end]
        except Exception as e:
            # Fall back to full fetch if cache reading fails
            pass

    # Full fetch
    try:
        df = fdr.DataReader(code, start, target_end)
    except Exception as e:
        # If network error and cache exists, fallback to cache
        if df_cached is not None and not df_cached.empty:
            return df_cached.loc[start:target_end]
        raise RuntimeError(f"Failed to fetch data for {code}: {e}")

    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "Change"])

    df.index = pd.to_datetime(df.index)
    # Ensure column casing
    rename_dict = {}
    for col in df.columns:
        c_lower = col.lower()
        if c_lower == "open":
            rename_dict[col] = "Open"
        elif c_lower == "high":
            rename_dict[col] = "High"
        elif c_lower == "low":
            rename_dict[col] = "Low"
        elif c_lower == "close":
            rename_dict[col] = "Close"
        elif c_lower == "volume":
            rename_dict[col] = "Volume"
        elif c_lower in ("change", "chg"):
            rename_dict[col] = "Change"
    df = df.rename(columns=rename_dict)

    # Save to parquet cache
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(cache_path)
    except Exception:
        pass

    return df.loc[start:target_end]


def get_benchmark_ohlcv(start: str = "2017-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Fetches and caches KODEX 200 (069500) ETF daily OHLCV."""
    return fetch_ohlcv(BENCHMARK_CODE, start=start, end=end)


def get_latest_market_info(code: str) -> Dict[str, Any]:
    """
    Returns latest price, daily change %, volume, and date for given ticker.
    Used for rapid watchlist display.
    """
    try:
        df = fetch_ohlcv(code)
        if df.empty:
            return {"close": 0.0, "change_pct": 0.0, "volume": 0, "date": "N/A"}
        latest = df.iloc[-1]
        prev_close = df.iloc[-2]["Close"] if len(df) > 1 else latest["Close"]
        change_pct = ((latest["Close"] - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0

        return {
            "close": float(latest["Close"]),
            "change_pct": float(change_pct),
            "volume": int(latest["Volume"]),
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "high": float(latest["High"]),
            "low": float(latest["Low"]),
            "open": float(latest["Open"]),
        }
    except Exception as e:
        return {"close": 0.0, "change_pct": 0.0, "volume": 0, "date": "Error", "error": str(e)}


if __name__ == "__main__":
    print("Testing market_data collector...")
    df_samsung = fetch_ohlcv("005930", "2024-01-01", "2024-01-10")
    print("Samsung Electronics (005930):")
    print(df_samsung.head(2))
    info = get_latest_market_info("005930")
    print("Latest info:", info)
