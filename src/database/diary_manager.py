"""
Prediction Diary and Forward Testing Log Manager.
Handles:
- Recording daily predictions at 15:40 (Score, TP/SL targets, indicators)
- Next-day auto-settlement using actual market OHLCV (Hit/Miss, actual return)
- Forward testing performance tracking and Model Decay detection
"""

import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

from src.database.models import get_db_connection
from src.collectors.market_data import fetch_ohlcv
from src.backtest.exit_rules import simulate_intraday_exit
from src.core.execution import calculate_net_trade_return


def record_daily_prediction(
    date_str: str,
    code: str,
    name: str,
    score: float,
    label: str,
    target_tp: float,
    target_sl: float,
    rsi: float = 0.0,
    ma_align: int = 0,
    vol_surge: float = 0.0,
    bb_pct: float = 0.0
) -> int:
    """
    Inserts a new prediction into the predictions table.
    Avoids duplicate entries for the same date and code.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM predictions WHERE date = ? AND code = ?",
            (date_str, code)
        )
        existing = cursor.fetchone()
        if existing:
            pred_id = existing[0]
            cursor.execute("""
                UPDATE predictions SET
                    score = ?, label = ?, rsi = ?, ma_align = ?, vol_surge = ?, bb_pct = ?,
                    target_tp = ?, target_sl = ?, created_at = ?
                WHERE id = ?
            """, (score, label, rsi, ma_align, vol_surge, bb_pct, target_tp, target_sl, now_str, pred_id))
            conn.commit()
            return pred_id

        cursor.execute("""
            INSERT INTO predictions (
                date, code, name, score, label, rsi, ma_align, vol_surge, bb_pct,
                target_tp, target_sl, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
        """, (date_str, code, name, score, label, rsi, ma_align, vol_surge, bb_pct, target_tp, target_sl, now_str))
        conn.commit()
        return cursor.lastrowid


def settle_pending_predictions() -> List[Dict[str, Any]]:
    """
    Finds all PENDING predictions and checks if subsequent day OHLCV is available to settle them.
    Computes Hit/Miss, actual return, exit reason, and inserts into settlements table.
    """
    settled_results = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, date, code, name, score, target_tp, target_sl
            FROM predictions
            WHERE status = 'PENDING'
            ORDER BY date ASC
        """)
        pending_rows = cursor.fetchall()

        for row in pending_rows:
            pred_id = row["id"]
            pred_date = row["date"]
            code = row["code"]
            score = row["score"]

            # Fetch OHLCV around pred_date
            try:
                df = fetch_ohlcv(code, start=pred_date)
            except Exception:
                continue

            if df.empty or len(df) < 2:
                continue

            # Target execution day is the trading day right after pred_date
            trading_dates = [d.strftime("%Y-%m-%d") for d in df.index]
            if pred_date not in trading_dates:
                # If pred_date was not a trading date, find first date >= pred_date
                future_dates = [d for d in trading_dates if d >= pred_date]
                if len(future_dates) < 2:
                    continue
                exec_date = future_dates[1]
            else:
                idx = trading_dates.index(pred_date)
                if idx + 1 >= len(trading_dates):
                    # Next trading day hasn't occurred yet
                    continue
                exec_date = trading_dates[idx + 1]

            exec_row = df.loc[exec_date]
            open_p = float(exec_row["Open"])
            high_p = float(exec_row["High"])
            low_p = float(exec_row["Low"])
            close_p = float(exec_row["Close"])

            if open_p <= 0:
                continue

            # Simulate intraday exit
            sim = simulate_intraday_exit(
                entry_open=open_p,
                day_high=high_p,
                day_low=low_p,
                day_close=close_p,
                score=score
            )

            # Net return
            net_ret = calculate_net_trade_return(
                raw_entry_price=sim["raw_entry_price"],
                raw_exit_price=sim["raw_exit_price"]
            )

            actual_ret_pct = net_ret["net_return_pct"]
            hit_status = "HIT" if actual_ret_pct > 0 else "MISS"

            # Insert into settlements table
            cursor.execute("""
                INSERT OR REPLACE INTO settlements (
                    prediction_id, code, target_date,
                    actual_open, actual_high, actual_low, actual_close,
                    actual_return, hit_status, exit_type, settled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pred_id, code, exec_date,
                open_p, high_p, low_p, close_p,
                actual_ret_pct, hit_status, sim["exit_reason"], now_str
            ))

            # Mark prediction as SETTLED
            cursor.execute("UPDATE predictions SET status = 'SETTLED' WHERE id = ?", (pred_id,))
            conn.commit()

            settled_results.append({
                "prediction_id": pred_id,
                "code": code,
                "target_date": exec_date,
                "actual_return": actual_ret_pct,
                "hit_status": hit_status,
                "exit_type": sim["exit_reason"]
            })

    return settled_results


def get_diary_history(limit: int = 50) -> pd.DataFrame:
    """Returns combined DataFrame of predictions and settlement results."""
    with get_db_connection() as conn:
        query = """
            SELECT 
                p.id, p.date as prediction_date, p.code, p.name, p.score, p.label,
                p.target_tp, p.target_sl, p.status,
                s.target_date as execution_date, s.actual_open, s.actual_high, s.actual_low, s.actual_close,
                s.actual_return, s.hit_status, s.exit_type
            FROM predictions p
            LEFT JOIN settlements s ON p.id = s.prediction_id
            ORDER BY p.date DESC, p.score DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        return df


def evaluate_model_decay(expected_ci_lower: float = 50.0) -> Dict[str, Any]:
    """
    Evaluates whether live forward testing win rate has decayed significantly
    below the backtest 95% confidence interval lower bound.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT actual_return, hit_status FROM settlements WHERE hit_status IS NOT NULL")
        rows = cursor.fetchall()

    total_settled = len(rows)
    if total_settled == 0:
        return {
            "total_settled": 0,
            "live_win_rate": 0.0,
            "decay_detected": False,
            "message": "아직 정산된 라이브 예측 표본이 없습니다.",
            "status": "INSUFFICIENT_DATA"
        }

    hits = sum(1 for r in rows if r["hit_status"] == "HIT")
    live_win_rate = (hits / total_settled) * 100.0

    # If at least 10 forward test trades and win rate is < expected_ci_lower:
    decay_detected = (total_settled >= 10) and (live_win_rate < expected_ci_lower)

    if decay_detected:
        msg = f"⚠️ [모델 열화 경고] 실전 승률({live_win_rate:.1f}%)이 백테스트 95% 신뢰구간 하한({expected_ci_lower:.1f}%)을 밑돌고 있습니다. 시장 국면 전환 또는 파라미터 재조정이 권장됩니다."
        status = "DECAY_WARNING"
    elif total_settled < 10:
        msg = f"ℹ️ 현재 누적 정산 {total_settled}건, 실현 승률 {live_win_rate:.1f}% (최소 10건 이상 누적 시 통계적 열화 판정 가능)"
        status = "NORMAL"
    else:
        msg = f"✅ 실전 승률({live_win_rate:.1f}%)이 백테스트 신뢰구간 하한을 안정적으로 상회하고 있습니다."
        status = "HEALTHY"

    return {
        "total_settled": total_settled,
        "hits": hits,
        "live_win_rate": round(live_win_rate, 1),
        "decay_detected": decay_detected,
        "message": msg,
        "status": status,
    }


if __name__ == "__main__":
    init_id = record_daily_prediction(
        date_str="2024-01-02",
        code="005930",
        name="삼성전자",
        score=78.5,
        label="강세",
        target_tp=2.0,
        target_sl=2.0
    )
    print("Recorded prediction id:", init_id)
    settled = settle_pending_predictions()
    print("Settled results:", settled)
    df_hist = get_diary_history(5)
    print(df_hist.head())
