"""
Benchmark matching and excess return (Alpha) calculation.
Matches trade execution dates with KODEX 200 (069500) ETF intraday performance.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


def match_benchmark_returns(
    trades_df: pd.DataFrame,
    benchmark_ohlcv: pd.DataFrame
) -> pd.DataFrame:
    """
    Matches each trade with KODEX 200 (069500) intraday return on the exact trade entry/exit date.
    Intraday return for KODEX 200: (Close - Open) / Open * 100.
    Adds 'bm_return_pct' and 'excess_return_pct' to trades_df.
    """
    if trades_df.empty or benchmark_ohlcv.empty:
        trades_df["bm_return_pct"] = 0.0
        trades_df["excess_return_pct"] = trades_df.get("net_return_pct", 0.0)
        return trades_df

    df = trades_df.copy()

    # Pre-calculate benchmark intraday return: (Close - Open) / Open
    bm = benchmark_ohlcv.copy()
    bm.index = pd.to_datetime(bm.index)
    bm_intraday = ((bm["Close"] - bm["Open"]) / bm["Open"].replace(0, np.nan)) * 100.0
    bm_map = {idx.strftime("%Y-%m-%d"): float(val) for idx, val in bm_intraday.dropna().items()}

    bm_rets = []
    excess_rets = []

    for _, row in df.iterrows():
        # Trade date is entry_date (intraday trading executes on entry_date)
        t_date = str(row.get("entry_date", row.get("date", "")))[:10]
        bm_ret = bm_map.get(t_date, 0.0)
        strat_ret = float(row.get("net_return_pct", 0.0))
        excess = strat_ret - bm_ret

        bm_rets.append(round(bm_ret, 4))
        excess_rets.append(round(excess, 4))

    df["bm_return_pct"] = bm_rets
    df["excess_return_pct"] = excess_rets
    return df


def build_cumulative_equity_curves(
    trades_df: pd.DataFrame,
    benchmark_ohlcv: pd.DataFrame,
    initial_capital: float = 10_000_000.0
) -> pd.DataFrame:
    """
    Constructs aligned time-series comparison of Strategy cumulative return vs KODEX 200 buy-and-hold
    and KODEX 200 matched intraday return.
    """
    if trades_df.empty:
        return pd.DataFrame()

    df_t = trades_df.copy()
    df_t["date"] = pd.to_datetime(df_t["entry_date"])
    df_t = df_t.sort_values("date").reset_index(drop=True)

    # Strategy compounded equity
    df_t["strat_mult"] = 1.0 + (df_t["net_return_pct"] / 100.0)
    df_t["strat_cum_return"] = (df_t["strat_mult"].cumprod() - 1.0) * 100.0

    # Matched benchmark compounded return
    df_t["bm_mult"] = 1.0 + (df_t["bm_return_pct"] / 100.0)
    df_t["bm_cum_return"] = (df_t["bm_mult"].cumprod() - 1.0) * 100.0

    df_t["alpha_cum"] = df_t["strat_cum_return"] - df_t["bm_cum_return"]

    return df_t


if __name__ == "__main__":
    from src.collectors.market_data import get_benchmark_ohlcv
    bm = get_benchmark_ohlcv("2024-01-01", "2024-01-10")
    dummy_trades = pd.DataFrame([
        {"entry_date": "2024-01-02", "net_return_pct": 2.5},
        {"entry_date": "2024-01-03", "net_return_pct": -1.8},
    ])
    matched = match_benchmark_returns(dummy_trades, bm)
    print("Matched returns:")
    print(matched[["entry_date", "net_return_pct", "bm_return_pct", "excess_return_pct"]])
