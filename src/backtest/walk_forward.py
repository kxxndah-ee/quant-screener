"""
Walk-Forward Validation Backtest Engine.
Rolling 250-trading-day In-Sample (IS) / 20-trading-day Out-of-Sample (OOS) slices.
Simulates realistic next-day execution, feasibility checks, dynamic exits, and costs.
Evaluates statistical significance against KODEX 200 benchmark.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from src.collectors.market_data import fetch_ohlcv, get_benchmark_ohlcv
from src.core.scoring import compute_point_in_time_indicators, calculate_score_for_row
from src.core.execution import is_trade_feasible, calculate_net_trade_return
from src.backtest.exit_rules import simulate_intraday_exit
from src.backtest.benchmark import match_benchmark_returns, build_cumulative_equity_curves
from src.backtest.statistical import run_bootstrap_analysis


def run_walk_forward_backtest(
    code: str,
    start_year: int = 2017,
    min_entry_score: float = 70.0,
    strategy_mode: str = "NORMAL",  # "NORMAL", "SNIPER", "CLOSING_BET"
    is_window_size: int = 250,   # 250 trading days In-Sample
    oos_window_size: int = 20,   # 20 trading days Out-of-Sample
    custom_sl_pct: float = 2.0,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Executes a rolling Walk-Forward Validation backtest on historical OHLCV data.
    Supports 3 strategy modes:
    1. 'NORMAL': Standard quantitative scoring with dynamic TP/SL.
    2. 'SNIPER': High-conviction 80%-target model (Market regime + 20MA pullback + Gap limit + TP +1.2%).
    3. 'CLOSING_BET': Leader overnight close-to-open bet (15:20 close enter -> Next morning open/spike exit).
    """
    start_date = f"{start_year}-01-01"
    df = fetch_ohlcv(code, start=start_date, force_refresh=force_refresh)
    if df.empty or len(df) < (is_window_size + oos_window_size + 10):
        return {
            "error": f"데이터 표본 부족 ({len(df)}개 봉). 최소 {is_window_size + oos_window_size + 10}개 일봉 필요.",
            "trades_df": pd.DataFrame(),
            "stats": {},
        }

    # Pre-compute rolling technical indicators
    df_ind = compute_point_in_time_indicators(df)
    if df_ind.empty:
        return {"error": "지표 계산 실패", "trades_df": pd.DataFrame(), "stats": {}}

    bm_ohlcv = get_benchmark_ohlcv(start=start_date)

    total_bars = len(df_ind)
    oos_trades: List[Dict[str, Any]] = []

    step = oos_window_size
    # Roll forward window
    for window_start in range(0, total_bars - is_window_size - oos_window_size, step):
        is_end = window_start + is_window_size
        oos_end = min(is_end + oos_window_size, total_bars - 1)

        # Iterate through Out-of-Sample window
        for i in range(is_end, oos_end):
            signal_row = df_ind.iloc[i]
            signal_date = df_ind.index[i].strftime("%Y-%m-%d")
            next_row = df_ind.iloc[i + 1]
            next_date = df_ind.index[i + 1].strftime("%Y-%m-%d")

            prev_close = float(signal_row["Close"])
            next_open = float(next_row["Open"])
            next_high = float(next_row["High"])
            next_low = float(next_row["Low"])
            next_close = float(next_row["Close"])
            prev_prev_close = float(df_ind.iloc[i - 1]["Close"]) if i > 0 else prev_close
            avg_val = float(signal_row.get("VolMA20", 1.0) * prev_close)

            # -------------------------------------------------------------
            # STRATEGY MODE 1: SNIPER HIGH-CONVICTION (80% TARGET)
            # -------------------------------------------------------------
            if strategy_mode == "SNIPER":
                # A. Market regime check: KODEX 200 >= 20MA
                if not bm_ohlcv.empty:
                    bm_slice = bm_ohlcv.loc[:signal_date]
                    if len(bm_slice) >= 20:
                        bm_ma20 = float(bm_slice["Close"].rolling(20).mean().iloc[-1])
                        bm_close = float(bm_slice["Close"].iloc[-1])
                        if bm_close < bm_ma20:
                            continue  # Market in correction/downtrend -> Skip

                # B. Signal stock technical filters
                ma5 = float(signal_row["MA5"])
                ma20 = float(signal_row["MA20"])
                rsi = float(signal_row["RSI14"])
                vol_ratio = float(signal_row["VolRatio"])
                disparity = (prev_close / ma20) * 100.0 if ma20 > 0 else 0.0

                # 20MA pullback support zone + MA5 > MA20 + RSI healthy + Vol expansion
                if not (98.0 <= disparity <= 104.0):
                    continue
                if not (ma5 > ma20):
                    continue
                if not (45.0 <= rsi <= 68.0):
                    continue
                if vol_ratio < 1.1:
                    continue

                # C. Next-day execution check: Strict open gap filter (-1.5% ~ +1.5%)
                open_gap_pct = ((next_open - prev_close) / prev_close) * 100.0
                if open_gap_pct > 1.5 or open_gap_pct < -1.5:
                    continue  # Gap-up chase or crash dump prevented

                # Feasibility check
                feasible, _ = is_trade_feasible(prev_close, next_open, prev_prev_close)
                if not feasible:
                    continue

                # D. Target TP +1.2%, SL -2.0%
                tp_pct = 1.2
                sl_pct = custom_sl_pct if custom_sl_pct else 2.0
                exit_sim = simulate_intraday_exit(
                    entry_open=next_open,
                    day_high=next_high,
                    day_low=next_low,
                    day_close=next_close,
                    score=95.0,
                    custom_tp_pct=tp_pct,
                    custom_sl_pct=sl_pct
                )

                ret_data = calculate_net_trade_return(
                    raw_entry_price=exit_sim["raw_entry_price"],
                    raw_exit_price=exit_sim["raw_exit_price"],
                    avg_daily_val=avg_val
                )

                oos_trades.append({
                    "signal_date": signal_date,
                    "entry_date": next_date,
                    "code": code,
                    "score": 95.0,
                    "label": "스나이퍼",
                    "raw_entry": exit_sim["raw_entry_price"],
                    "raw_exit": exit_sim["raw_exit_price"],
                    "target_tp": exit_sim["target_tp"],
                    "target_sl": exit_sim["target_sl"],
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "exit_reason": exit_sim["exit_reason"],
                    "exit_code": exit_sim["exit_code"],
                    "gross_return_pct": ret_data["gross_return_pct"],
                    "net_return_pct": ret_data["net_return_pct"],
                    "slippage_rate": ret_data["slippage_rate"],
                })

            # -------------------------------------------------------------
            # STRATEGY MODE 2: CLOSING BET (15:20 OVERNIGHT)
            # -------------------------------------------------------------
            elif strategy_mode == "CLOSING_BET":
                # Candidate evaluation on day T:
                high_p = float(signal_row["High"])
                low_p = float(signal_row["Low"])
                vol_p = float(signal_row["Volume"])
                day_val = vol_p * prev_close

                # 1. Day return >= +2.5%
                day_return = ((prev_close - prev_prev_close) / prev_prev_close) * 100.0 if prev_prev_close > 0 else 0.0
                if day_return < 2.5:
                    continue

                # 2. Strong close near High (top 20% range)
                if (high_p - low_p) <= 0:
                    continue
                high_close_ratio = (prev_close - low_p) / (high_p - low_p)
                if high_close_ratio < 0.80:
                    continue

                # 3. Above MAs
                ma5 = float(signal_row["MA5"])
                ma20 = float(signal_row["MA20"])
                if not (prev_close > ma5 and prev_close > ma20):
                    continue

                # Execute Closing Bet: Enter at Day T Close -> Exit at Day T+1
                tp_pct = 1.5
                sl_pct = custom_sl_pct if custom_sl_pct else 2.0
                target_tp = prev_close * (1.0 + tp_pct / 100.0)
                target_sl = prev_close * (1.0 - sl_pct / 100.0)

                if next_open >= target_tp:
                    raw_exit = next_open
                    reason = "TAKE_PROFIT (시초가 갭익절)"
                    exit_code = "TP_GAP"
                elif next_high >= target_tp and next_low <= target_sl:
                    raw_exit = target_sl
                    reason = "STOP_LOSS (동시도달 손절우선)"
                    exit_code = "SL_TIE"
                elif next_high >= target_tp:
                    raw_exit = target_tp
                    reason = "TAKE_PROFIT (장초반 슈팅)"
                    exit_code = "TP"
                elif next_low <= target_sl:
                    raw_exit = target_sl
                    reason = "STOP_LOSS (손절)"
                    exit_code = "SL"
                else:
                    raw_exit = next_close
                    reason = "CLOSE_EXIT (종가청산)"
                    exit_code = "CLOSE"

                ret_data = calculate_net_trade_return(
                    raw_entry_price=prev_close,
                    raw_exit_price=raw_exit,
                    avg_daily_val=avg_val
                )

                oos_trades.append({
                    "signal_date": signal_date,
                    "entry_date": next_date,
                    "code": code,
                    "score": 95.0,
                    "label": "종가배팅",
                    "raw_entry": prev_close,
                    "raw_exit": raw_exit,
                    "target_tp": target_tp,
                    "target_sl": target_sl,
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "exit_reason": reason,
                    "exit_code": exit_code,
                    "gross_return_pct": ret_data["gross_return_pct"],
                    "net_return_pct": ret_data["net_return_pct"],
                    "slippage_rate": ret_data["slippage_rate"],
                })

            # -------------------------------------------------------------
            # STRATEGY MODE 3: SURGE 5PCT TARGET (익일 5% 돌파 타겟 모드)
            # -------------------------------------------------------------
            elif strategy_mode == "SURGE_5PCT":
                # Market regime check
                if not bm_ohlcv.empty:
                    bm_slice = bm_ohlcv.loc[:signal_date]
                    if len(bm_slice) >= 20:
                        bm_ma20 = float(bm_slice["Close"].rolling(20).mean().iloc[-1])
                        bm_close = float(bm_slice["Close"].iloc[-1])
                        if bm_close < bm_ma20:
                            continue

                high_p = float(signal_row["High"])
                low_p = float(signal_row["Low"])
                vol_p = float(signal_row["Volume"])

                # 1. Day return >= +12%
                day_return = ((prev_close - prev_prev_close) / prev_prev_close) * 100.0 if prev_prev_close > 0 else 0.0
                if day_return < 12.0:
                    continue

                # 2. High-close ratio >= 0.88
                if (high_p - low_p) <= 0:
                    continue
                high_close_ratio = (prev_close - low_p) / (high_p - low_p)
                if high_close_ratio < 0.88:
                    continue

                # 3. Minimum trading value (200억 이상)
                if (vol_p * prev_close) < 20_000_000_000:
                    continue

                # Check Next Day (and Day 2 if held)
                tp_pct = 5.0
                sl_pct = custom_sl_pct if custom_sl_pct else 4.0
                target_tp = prev_close * (1.0 + tp_pct / 100.0)
                target_sl = prev_close * (1.0 - sl_pct / 100.0)

                # Day 2 data if available
                has_day2 = (i + 2) < total_bars
                day2_high = float(df_ind.iloc[i + 2]["High"]) if has_day2 else next_high
                day2_low = float(df_ind.iloc[i + 2]["Low"]) if has_day2 else next_low
                day2_close = float(df_ind.iloc[i + 2]["Close"]) if has_day2 else next_close

                if next_open >= target_tp:
                    raw_exit = next_open
                    reason = "TAKE_PROFIT (익일 시초가 +5% 갭익절)"
                    exit_code = "TP_GAP"
                elif next_high >= target_tp and next_low <= target_sl:
                    raw_exit = target_sl
                    reason = "STOP_LOSS (동시도달 손절우선)"
                    exit_code = "SL_TIE"
                elif next_high >= target_tp:
                    raw_exit = target_tp
                    reason = "TAKE_PROFIT (익일 장중 +5% 슈팅 익절)"
                    exit_code = "TP"
                elif has_day2 and day2_high >= target_tp and day2_low > target_sl:
                    raw_exit = target_tp
                    reason = "TAKE_PROFIT (2일차 +5% 도달 익절)"
                    exit_code = "TP_DAY2"
                elif next_low <= target_sl:
                    raw_exit = target_sl
                    reason = "STOP_LOSS (손절)"
                    exit_code = "SL"
                elif has_day2 and day2_low <= target_sl:
                    raw_exit = target_sl
                    reason = "STOP_LOSS (2일차 손절)"
                    exit_code = "SL_DAY2"
                else:
                    raw_exit = day2_close if has_day2 else next_close
                    reason = "CLOSE_EXIT (종가청산)"
                    exit_code = "CLOSE"

                ret_data = calculate_net_trade_return(
                    raw_entry_price=prev_close,
                    raw_exit_price=raw_exit,
                    avg_daily_val=avg_val
                )

                oos_trades.append({
                    "signal_date": signal_date,
                    "entry_date": next_date,
                    "code": code,
                    "score": 96.0,
                    "label": "5%급등",
                    "raw_entry": prev_close,
                    "raw_exit": raw_exit,
                    "target_tp": target_tp,
                    "target_sl": target_sl,
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "exit_reason": reason,
                    "exit_code": exit_code,
                    "gross_return_pct": ret_data["gross_return_pct"],
                    "net_return_pct": ret_data["net_return_pct"],
                    "slippage_rate": ret_data["slippage_rate"],
                })

            # -------------------------------------------------------------
            # STRATEGY MODE: REAL-TIME DAY TRADING (당일 매수 -> 당일 +5% 익절 / 당일 전량청산)
            # -------------------------------------------------------------
            elif strategy_mode == "DAY_TRADE_5PCT":
                if next_open <= 0 or next_high <= next_low:
                    continue

                if avg_val < 10_000_000_000:
                    continue

                # Entry at Open * 1.03 (+3% breakout entry)
                entry_p = next_open * 1.03
                tp_pct = 5.0
                sl_pct = custom_sl_pct if custom_sl_pct else 2.5
                target_tp = entry_p * (1.0 + tp_pct / 100.0)
                target_sl = entry_p * (1.0 - sl_pct / 100.0)

                if next_high < entry_p:
                    continue

                # 1. If day high reached +5.0% target and close held entry: Take Profit!
                if next_high >= target_tp and next_close >= entry_p:
                    raw_exit = target_tp
                    reason = "TAKE_PROFIT (당일 장중 +5% 슈팅 익절)"
                    exit_code = "TP"
                # 2. If day dumped below stop loss and closed below entry: Stop Loss!
                elif next_close <= target_sl or (next_close < entry_p and next_low <= target_sl):
                    raw_exit = target_sl
                    reason = "STOP_LOSS (손절)"
                    exit_code = "SL"
                # 3. If reached TP before late pullback:
                elif next_high >= target_tp:
                    raw_exit = target_tp
                    reason = "TAKE_PROFIT (장중 지정가 익절 체결)"
                    exit_code = "TP_LIMIT"
                # 4. Otherwise: exit at 15:15 market close
                else:
                    raw_exit = next_close
                    reason = "CLOSE_EXIT (당일 15:15 종가 전량 청산)"
                    exit_code = "CLOSE"

                ret_data = calculate_net_trade_return(
                    raw_entry_price=entry_p,
                    raw_exit_price=raw_exit,
                    avg_daily_val=avg_val
                )

                oos_trades.append({
                    "signal_date": signal_date,
                    "entry_date": next_date,
                    "code": code,
                    "score": 95.0,
                    "label": "당일단타",
                    "raw_entry": entry_p,
                    "raw_exit": raw_exit,
                    "target_tp": target_tp,
                    "target_sl": target_sl,
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "exit_reason": reason,
                    "exit_code": exit_code,
                    "gross_return_pct": ret_data["gross_return_pct"],
                    "net_return_pct": ret_data["net_return_pct"],
                    "slippage_rate": ret_data["slippage_rate"],
                })

            # -------------------------------------------------------------
            # STRATEGY MODE 4: NORMAL QUANT SCORING (DEFAULT)
            # -------------------------------------------------------------
            else:
                score_res = calculate_score_for_row(signal_row)
                score = score_res["score"]

                if score < min_entry_score:
                    continue

                feasible, reason = is_trade_feasible(prev_close, next_open, prev_prev_close)
                if not feasible:
                    continue

                exit_sim = simulate_intraday_exit(
                    entry_open=next_open,
                    day_high=next_high,
                    day_low=next_low,
                    day_close=next_close,
                    score=score,
                    custom_sl_pct=custom_sl_pct
                )

                ret_data = calculate_net_trade_return(
                    raw_entry_price=exit_sim["raw_entry_price"],
                    raw_exit_price=exit_sim["raw_exit_price"],
                    avg_daily_val=avg_val
                )

                oos_trades.append({
                    "signal_date": signal_date,
                    "entry_date": next_date,
                    "code": code,
                    "score": score,
                    "label": score_res["label"],
                    "raw_entry": exit_sim["raw_entry_price"],
                    "raw_exit": exit_sim["raw_exit_price"],
                    "target_tp": exit_sim["target_tp"],
                    "target_sl": exit_sim["target_sl"],
                    "tp_pct": exit_sim["tp_pct"],
                    "sl_pct": exit_sim["sl_pct"],
                    "exit_reason": exit_sim["exit_reason"],
                    "exit_code": exit_sim["exit_code"],
                    "gross_return_pct": ret_data["gross_return_pct"],
                    "net_return_pct": ret_data["net_return_pct"],
                    "slippage_rate": ret_data["slippage_rate"],
                })

    trades_df = pd.DataFrame(oos_trades)
    if trades_df.empty:
        mode_desc = "스나이퍼" if strategy_mode == "SNIPER" else ("주도주 종가배팅" if strategy_mode == "CLOSING_BET" else f"최소 점수 {min_entry_score}점")
        return {
            "warning": f"설정된 조건({mode_desc})을 만족하는 유효 거래 신호가 없습니다.",
            "trades_df": pd.DataFrame(),
            "stats": {},
            "equity_curve": pd.DataFrame(),
        }

    # Match benchmark returns
    matched_trades = match_benchmark_returns(trades_df, bm_ohlcv)

    # Run Bootstrap analysis (1,000 iterations)
    stats = run_bootstrap_analysis(matched_trades, n_iterations=1000)

    # Build cumulative equity curves
    equity_curve = build_cumulative_equity_curves(matched_trades, bm_ohlcv)

    # Exit reason breakdown
    exit_counts = matched_trades["exit_code"].value_counts().to_dict()

    return {
        "trades_df": matched_trades,
        "stats": stats,
        "equity_curve": equity_curve,
        "exit_counts": exit_counts,
        "total_evaluated_days": total_bars,
        "oos_days": total_bars - is_window_size,
        "strategy_mode": strategy_mode,
    }


if __name__ == "__main__":
    print("Testing Walk-Forward Engine on Samsung Electronics (005930)...")
    res = run_walk_forward_backtest("005930", start_year=2020, min_entry_score=70.0)
    print("Backtest finished!")
    if "stats" in res:
        print("Stats:", {k: v for k, v in res["stats"].items() if k != "bootstrap_win_rates"})
    if "exit_counts" in res:
        print("Exit counts:", res["exit_counts"])
