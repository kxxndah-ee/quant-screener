"""
Verification Script:
Evaluates Point-in-Time Technical Scores using data up to YESTERDAY (2026-09-09 close).
Screens candidate stocks, then verifies their performance against TODAY's (2026-09-10) actual market OHLCV.
Calculates realized accuracy (win rate), net return, and Alpha vs KODEX 200.
"""

import os
import sys

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np

# Inject truststore for SSL
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database.models import init_db
from src.collectors.market_data import fetch_ohlcv, get_benchmark_ohlcv
from src.collectors.krx_universe import get_universe, sync_krx_universe
from src.core.scoring import compute_point_in_time_indicators, calculate_score_for_row
from src.core.execution import is_trade_feasible, calculate_net_trade_return
from src.backtest.exit_rules import simulate_intraday_exit

init_db()


def run_point_in_time_verification():
    print("=== [어제(2026-09-09) 데이터 기준 익일 상승 후보 스크리닝 및 금일(2026-09-10) 실제 적중률 검증] ===")

    # 1. Benchmark today's return
    bm = get_benchmark_ohlcv()
    bm_yesterday = bm.loc["2026-09-09"] if "2026-09-09" in bm.index else None
    bm_today = bm.loc["2026-09-10"] if "2026-09-10" in bm.index else None

    if bm_today is not None:
        bm_intraday_ret = ((bm_today["Close"] - bm_today["Open"]) / bm_today["Open"]) * 100.0
        bm_day_ret = float(bm_today["Change"]) * 100.0 if "Change" in bm_today else 0.0
        print(f"📌 시장 벤치마크 (KODEX 200) 금일 성과: 시가 대비 종가 {bm_intraday_ret:+.2f}%, 전일대비 {bm_day_ret:+.2f}%")
    else:
        bm_intraday_ret = -2.0
        print("Warning: KODEX 200 today bar not found.")

    # 2. Get Universe pool (sample liquid top stocks from KOSPI / KOSDAQ)
    df_univ = get_universe(active_only=True)
    if df_univ.empty or len(df_univ) < 50:
        sync_krx_universe()
        df_univ = get_universe(active_only=True)

    # Take liquid candidate pool
    test_pool = df_univ.head(100)
    print(f"총 {len(test_pool)}개 대표 유니버스 종목 대상 검증 시작...")

    results = []

    for _, row in test_pool.iterrows():
        code = row["code"]
        name = row["name"]
        market = row["market"]
        sector = row.get("sector", "기타")

        try:
            df = fetch_ohlcv(code, start="2024-01-01")
            if df.empty or len(df) < 60:
                continue

            # Check if 2026-09-09 and 2026-09-10 exist
            dates = [d.strftime("%Y-%m-%d") for d in df.index]
            if "2026-09-09" not in dates or "2026-09-10" not in dates:
                continue

            # Point-in-time truncation: strictly up to 2026-09-09
            df_pit = df.loc[:"2026-09-09"].copy()
            df_ind = compute_point_in_time_indicators(df_pit)
            if df_ind.empty:
                continue

            # Yesterday's closing evaluation
            yesterday_row = df_ind.iloc[-1]
            score_res = calculate_score_for_row(yesterday_row)
            score = score_res["score"]
            label = score_res["label"]
            tp_target = score_res["tp_pct"]
            sl_target = score_res["sl_pct"]
            vol_ratio = score_res["breakdown"]["vol_ratio"]
            ma_status = score_res["breakdown"]["ma_status"]
            rsi_val = score_res["breakdown"]["rsi_val"]

            # Today's actual market bar (2026-09-10)
            today_row = df.loc["2026-09-10"]
            today_open = float(today_row["Open"])
            today_high = float(today_row["High"])
            today_low = float(today_row["Low"])
            today_close = float(today_row["Close"])

            if today_open <= 0:
                continue

            # 1. Feasibility check
            prev_close = float(yesterday_row["Close"])
            prev_prev = float(df_ind.iloc[-2]["Close"]) if len(df_ind) > 1 else prev_close
            feasible, reason = is_trade_feasible(prev_close, today_open, prev_prev)
            if not feasible:
                continue

            # 2. Simulated Intraday Exit (system rules)
            exit_sim = simulate_intraday_exit(
                entry_open=today_open,
                day_high=today_high,
                day_low=today_low,
                day_close=today_close,
                score=score,
                custom_sl_pct=2.0
            )

            # 3. Cost & Net return
            avg_val = float(yesterday_row.get("VolMA20", 1.0) * prev_close)
            net_ret = calculate_net_trade_return(
                raw_entry_price=exit_sim["raw_entry_price"],
                raw_exit_price=exit_sim["raw_exit_price"],
                avg_daily_val=avg_val
            )

            # Metrics
            gross_intraday_ret = ((today_close - today_open) / today_open) * 100.0
            max_potential_gain = ((today_high - today_open) / today_open) * 100.0
            net_return_pct = net_ret["net_return_pct"]
            is_hit = net_return_pct > 0
            excess_return = net_return_pct - bm_intraday_ret

            results.append({
                "code": code,
                "name": name,
                "market": market,
                "sector": sector,
                "yesterday_score": score,
                "yesterday_label": label,
                "rsi": rsi_val,
                "vol_ratio": vol_ratio,
                "ma_status": ma_status,
                "today_open": today_open,
                "today_high": today_high,
                "today_low": today_low,
                "today_close": today_close,
                "target_tp": tp_target,
                "target_sl": sl_target,
                "exit_reason": exit_sim["exit_reason"],
                "exit_code": exit_sim["exit_code"],
                "max_potential_gain": round(max_potential_gain, 2),
                "gross_intraday_ret": round(gross_intraday_ret, 2),
                "net_return_pct": round(net_return_pct, 2),
                "is_hit": is_hit,
                "excess_return": round(excess_return, 2),
            })
        except Exception as e:
            continue

    if not results:
        print("검증 가능한 종목 데이터가 없습니다.")
        return

    df_res = pd.DataFrame(results)

    # Breakdown by score groups
    high_conviction = df_res[df_res["yesterday_score"] >= 70.0]
    mid_conviction = df_res[(df_res["yesterday_score"] >= 50.0) & (df_res["yesterday_score"] < 70.0)]
    low_conviction = df_res[df_res["yesterday_score"] < 50.0]

    def summarize_group(grp: pd.DataFrame, name: str):
        if grp.empty:
            print(f"\n[{name}] 종목 수: 0개")
            return
        total = len(grp)
        hits = grp["is_hit"].sum()
        win_rate = (hits / total) * 100.0
        avg_net_ret = grp["net_return_pct"].mean()
        avg_excess_ret = grp["excess_return"].mean()
        avg_max_gain = grp["max_potential_gain"].mean()
        tp_hits = sum(grp["exit_code"] == "TP")
        sl_hits = sum(grp["exit_code"].isin(["SL", "SL_TIE"]))
        close_exits = sum(grp["exit_code"] == "CLOSE")

        print(f"\n=======================================================")
        print(f"📊 [{name}] 그룹 성과 분석 (총 {total}종목)")
        print(f"=======================================================")
        print(f"  • 실현 정확도(승률): {win_rate:.1f}% ({hits}/{total} 적중)")
        print(f"  • 목표 익절(+{grp['target_tp'].iloc[0]}%↑) 달성: {tp_hits}건 ({(tp_hits/total)*100:.1f}%)")
        print(f"  • 손절(-2.0%) 처리: {sl_hits}건 ({(sl_hits/total)*100:.1f}%)")
        print(f"  • 종가 청산: {close_exits}건")
        print(f"  • 평균 실현 순수익률: {avg_net_ret:+.2f}%")
        print(f"  • KODEX 200 대비 초과수익률(알파): {avg_excess_ret:+.2f}%")
        print(f"  • 장중 최대 상승폭 평균: {avg_max_gain:+.2f}%")

    summarize_group(high_conviction, "🔥 고확신 상승 후보 (스코어 70점 이상 / 강세)")
    summarize_group(mid_conviction, "⚖️ 중립 후보 (스코어 50~69점)")
    summarize_group(low_conviction, "❄️ 비추천/약세 (스코어 50점 미만)")

    # Print Top High Conviction screening table
    if not high_conviction.empty:
        print("\n[🔥 어제 스크리닝 발굴된 강세 후보 종목별 금일 실제 성적표]")
        cols_to_print = ["name", "code", "yesterday_score", "max_potential_gain", "net_return_pct", "exit_reason", "excess_return"]
        print(high_conviction[cols_to_print].sort_values("yesterday_score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    run_point_in_time_verification()
