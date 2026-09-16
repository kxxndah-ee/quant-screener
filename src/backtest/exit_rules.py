"""
Exit Rules Engine for next-day intraday trading.
Implements:
- Dynamic Take-Profit (TP) based on Technical Score (+1.5% ~ +3.0%)
- Base Stop-Loss (SL) (-2.0%)
- Conservative tie-breaking rule: if High hits TP and Low hits SL on the same bar,
  Stop-Loss takes priority.
- Fallback exit at Close on the same trading day (intraday round-trip).
"""

from typing import Dict, Any, Tuple, Optional


def get_target_percentages(
    score: float,
    custom_sl_pct: float = 2.0,
    custom_tp_pct: Optional[float] = None
) -> Tuple[float, float]:
    """
    Returns (tp_pct, sl_pct) as positive percentages (e.g. 3.0, 2.0).
    """
    if custom_tp_pct is not None:
        tp_pct = abs(custom_tp_pct)
    elif score >= 80.0:
        tp_pct = 3.0
    elif score >= 70.0:
        tp_pct = 2.0
    elif score >= 60.0:
        tp_pct = 1.5
    else:
        tp_pct = 1.0

    sl_pct = abs(custom_sl_pct)
    return tp_pct, sl_pct


def simulate_intraday_exit(
    entry_open: float,
    day_high: float,
    day_low: float,
    day_close: float,
    score: float,
    custom_sl_pct: float = 2.0,
    custom_tp_pct: Optional[float] = None
) -> Dict[str, Any]:
    """
    Simulates next-day intraday exit:
    1. Enter at Open.
    2. Check target TP and SL levels:
       target_tp = entry_open * (1 + tp_pct / 100)
       target_sl = entry_open * (1 - sl_pct / 100)
    3. If both High >= target_tp and Low <= target_sl:
       Conservative execution: STOP LOSS triggered.
    4. Else if High >= target_tp:
       TAKE PROFIT triggered.
    5. Else if Low <= target_sl:
       STOP LOSS triggered.
    6. Else:
       CLOSE EXIT triggered at day_close.
    """
    tp_pct, sl_pct = get_target_percentages(score, custom_sl_pct, custom_tp_pct)

    target_tp = entry_open * (1.0 + tp_pct / 100.0)
    target_sl = entry_open * (1.0 - sl_pct / 100.0)

    hit_tp = day_high >= target_tp
    hit_sl = day_low <= target_sl

    if hit_tp and hit_sl:
        # Conservative principle: Stop-Loss priority
        raw_exit_price = target_sl
        exit_reason = "STOP_LOSS (동시도달 손절우선)"
        exit_code = "SL_TIE"
    elif hit_sl:
        raw_exit_price = target_sl
        exit_reason = "STOP_LOSS (손절)"
        exit_code = "SL"
    elif hit_tp:
        raw_exit_price = target_tp
        exit_reason = "TAKE_PROFIT (익절)"
        exit_code = "TP"
    else:
        raw_exit_price = day_close
        exit_reason = "CLOSE_EXIT (종가청산)"
        exit_code = "CLOSE"

    return {
        "raw_entry_price": entry_open,
        "raw_exit_price": raw_exit_price,
        "target_tp": target_tp,
        "target_sl": target_sl,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "exit_reason": exit_reason,
        "exit_code": exit_code,
        "hit_tp": hit_tp,
        "hit_sl": hit_sl,
    }


if __name__ == "__main__":
    res = simulate_intraday_exit(
        entry_open=10000,
        day_high=10350,
        day_low=9700,
        day_close=10100,
        score=82.0
    )
    print("Simulated exit (Tie test):", res)
