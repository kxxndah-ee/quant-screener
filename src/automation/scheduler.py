"""
Automated Batch Scheduler using APScheduler (KST timezone).
Runs 2 automated jobs on Korean trading days (Monday - Friday):
1) 15:40 (Post-market): Cache update -> Settle T-1 predictions -> Compute T+1 predictions -> Load diary.
2) 08:30 (Pre-market): Reset daily risk state -> Screen high-conviction candidates (Score >= 70) -> Briefing log.
"""

import os
import sys
from datetime import datetime
from typing import List, Dict, Any

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.database.models import get_db_connection, init_db
from src.collectors.market_data import fetch_ohlcv, get_benchmark_ohlcv
from src.collectors.krx_universe import get_universe
from src.core.scoring import evaluate_stock_latest
from src.core.high_winrate_strategies import (
    evaluate_5pct_surge_candidate,
    evaluate_sniper_candidate,
    evaluate_market_regime
)
from src.core.risk_manager import PortfolioRiskManager
from src.database.diary_manager import (
    record_daily_prediction,
    settle_pending_predictions,
    evaluate_model_decay
)


def run_post_market_job() -> Dict[str, Any]:
    """
    15:40 Batch Job:
    1. Update price caches for all watchlist items.
    2. Settle yesterday's pending predictions.
    3. Calculate score & dynamic TP/SL for tomorrow, and record to Prediction Diary.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"[{now_str}] Starting 15:40 Post-Market Automation Job...")

    # 1. Settle pending predictions first
    settled = settle_pending_predictions()
    print(f"  - Settled {len(settled)} previous predictions.")

    # 2. Fetch Watchlist
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT code, name FROM watchlist")
        watchlist_stocks = [dict(r) for r in cursor.fetchall()]

    new_predictions = []
    bm_ohlcv = get_benchmark_ohlcv()
    # 3. Update cache & score for tomorrow
    for item in watchlist_stocks:
        code = item["code"]
        name = item["name"]
        try:
            df = fetch_ohlcv(code, force_refresh=True)
            if df.empty or len(df) < 60:
                continue

            score_res = evaluate_stock_latest(df)
            if not score_res:
                continue

            # Check 80% Win-Rate Strategies
            s5 = evaluate_5pct_surge_candidate(df, df_benchmark=bm_ohlcv, min_today_val_krw=10_000_000_000, min_day_return=10.0)
            snp = evaluate_sniper_candidate(df, df_benchmark=bm_ohlcv, min_daily_val_krw=5_000_000_000)

            score = score_res["score"]
            label = score_res["label"]
            tp = score_res["tp_pct"]
            sl = score_res["sl_pct"]

            if s5:
                label = "🚀 5% 급등 타겟 (1~2일 스윙)"
                tp = 5.0
                sl = 4.0
                score = max(score, 96.0)
            elif snp:
                label = "🎯 스나이퍼 고확신 (시초갭 제한)"
                tp = 1.2
                sl = 2.0
                score = max(score, 95.0)

            pred_id = record_daily_prediction(
                date_str=today_str,
                code=code,
                name=name,
                score=score,
                label=label,
                target_tp=tp,
                target_sl=sl,
                rsi=score_res["breakdown"]["rsi_val"],
                ma_align=1 if "정배열" in score_res["breakdown"]["ma_status"] else 0,
                vol_surge=score_res["breakdown"]["vol_ratio"],
                bb_pct=score_res["breakdown"]["bb_pct"]
            )
            new_predictions.append({
                "code": code,
                "name": name,
                "score": score,
                "label": label,
                "tp": tp,
                "sl": sl
            })
        except Exception as e:
            print(f"  - Error scoring {code} ({name}): {e}")

    print(f"  - Created {len(new_predictions)} next-day predictions in diary.")
    return {
        "job": "POST_MARKET_1540",
        "settled_count": len(settled),
        "new_predictions_count": len(new_predictions),
        "timestamp": now_str,
    }


def run_pre_market_job() -> Dict[str, Any]:
    """
    08:30 Batch Job:
    1. Reset daily risk manager state (daily PnL to 0, circuit breaker cleared).
    2. Screen watchlist & candidates with Score >= 70.
    3. Generate pre-market briefing log.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"[{now_str}] Starting 08:30 Pre-Market Automation Job...")

    # 1. Reset Risk State in DB
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE risk_state SET
                date = ?,
                daily_realized_loss = 0.0,
                is_circuit_breaker_active = 0
            WHERE id = 1
        """, (today_str,))
        conn.commit()

    # 2. Check candidates from diary
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT code, name, score, label, target_tp, target_sl
            FROM predictions
            WHERE date <= ? AND status = 'PENDING' AND score >= 70.0
            ORDER BY score DESC
        """, (today_str,))
        bullish_candidates = [dict(r) for r in cursor.fetchall()]

    briefing_lines = [
        f"=== [08:30 장전 퀀트 공략 브리핑 ({today_str})] ===",
        f"포트폴리오 일일 손익 리셋 완료 | 신규 매수 한도: 최대 5종목 (종목당 20% 이내)",
        f"금일 고확신(스코어 70점 이상) 진입 후보 종목 수: {len(bullish_candidates)}건",
    ]
    for c in bullish_candidates:
        briefing_lines.append(
            f"  - [{c['code']}] {c['name']}: 스코어 {c['score']:.1f}점 ({c['label']}) | 목표익절 +{c['target_tp']:.1f}% | 손절 -{c['target_sl']:.1f}%"
        )

    briefing_text = "\n".join(briefing_lines)
    print(briefing_text)

    return {
        "job": "PRE_MARKET_0830",
        "candidates_count": len(bullish_candidates),
        "briefing": briefing_text,
        "timestamp": now_str,
    }


def start_scheduler(blocking: bool = True):
    """Starts the APScheduler with Korea Standard Time (Asia/Seoul)."""
    init_db()
    sched = BlockingScheduler(timezone="Asia/Seoul") if blocking else BackgroundScheduler(timezone="Asia/Seoul")

    # Mon-Fri 15:40 KST
    sched.add_job(
        run_post_market_job,
        CronTrigger(day_of_week="mon-fri", hour=15, minute=40, timezone="Asia/Seoul"),
        id="job_post_market_1540",
        replace_existing=True
    )

    # Mon-Fri 08:30 KST
    sched.add_job(
        run_pre_market_job,
        CronTrigger(day_of_week="mon-fri", hour=8, minute=30, timezone="Asia/Seoul"),
        id="job_pre_market_0830",
        replace_existing=True
    )

    print("APScheduler configured for Mon-Fri 08:30 and 15:40 KST.")
    if blocking:
        try:
            sched.start()
        except (KeyboardInterrupt, SystemExit):
            pass
    else:
        sched.start()
        return sched


if __name__ == "__main__":
    print("Testing manual job execution:")
    res_pm = run_post_market_job()
    print("Post-market result:", res_pm)
    res_pre = run_pre_market_job()
    print("Pre-market result:", res_pre["briefing"])
