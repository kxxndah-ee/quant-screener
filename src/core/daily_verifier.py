"""
Daily Point-in-Time Verification and Self-Tuning Optimization Engine.
Evaluates historical screener recommendations against actual market outcomes,
diagnoses reasons for failure, and provides actionable parameter auto-tuning recommendations.
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.collectors.market_data import fetch_ohlcv, get_benchmark_ohlcv
from src.collectors.krx_universe import get_universe, sync_krx_universe
from src.core.scoring import compute_point_in_time_indicators, calculate_score_for_row
from src.core.high_winrate_strategies import (
    evaluate_sniper_candidate,
    evaluate_closing_bet_candidate,
    evaluate_5pct_surge_candidate,
    evaluate_intraday_daytrade_candidate
)
from src.core.execution import is_trade_feasible, calculate_net_trade_return
from src.backtest.exit_rules import simulate_intraday_exit


def get_available_trading_dates(limit: int = 30) -> List[str]:
    """Returns recent trading dates from benchmark OHLCV."""
    try:
        bm = get_benchmark_ohlcv()
        if not bm.empty:
            dates = [d.strftime("%Y-%m-%d") for d in bm.index]
            return dates[-limit:]
    except Exception:
        pass
    dates = []
    curr = datetime.now()
    while len(dates) < limit:
        curr -= timedelta(days=1)
        if curr.weekday() < 5:
            dates.append(curr.strftime("%Y-%m-%d"))
    return dates


def run_daily_point_in_time_verification(
    pred_date: str,
    exec_date: Optional[str] = None,
    strategy_mode: str = "⚡ 실시간 당일 단타 (5% 익절)",
    min_val_krw: float = 10_000_000_000,
    score_cutoff: float = 65.0,
    sample_pool_size: int = 150
) -> Dict[str, Any]:
    """
    Performs Point-in-Time verification for predictions made on pred_date (T-1)
    and evaluated against actual market outcomes on exec_date (T).
    """
    bm = get_benchmark_ohlcv()
    trading_dates = [d.strftime("%Y-%m-%d") for d in bm.index] if not bm.empty else []

    if exec_date is None:
        if pred_date in trading_dates:
            p_idx = trading_dates.index(pred_date)
            if p_idx + 1 < len(trading_dates):
                exec_date = trading_dates[p_idx + 1]
            else:
                return {"error": f"{pred_date} 이후의 실제 거래일 체결 데이터가 아직 없습니다."}
        else:
            return {"error": f"{pred_date}는 거래일 데이터에 존재하지 않습니다."}

    # Benchmark metrics on exec_date
    bm_intraday_ret = 0.0
    bm_day_ret = 0.0
    if not bm.empty and exec_date in bm.index:
        bm_bar = bm.loc[exec_date]
        if bm_bar["Open"] > 0:
            bm_intraday_ret = ((bm_bar["Close"] - bm_bar["Open"]) / bm_bar["Open"]) * 100.0
        bm_day_ret = float(bm_bar.get("Change", 0.0)) * 100.0

    # Get active universe
    df_univ = get_universe(active_only=True)
    if df_univ.empty or len(df_univ) < 50:
        sync_krx_universe()
        df_univ = get_universe(active_only=True)

    pool = df_univ.head(sample_pool_size)
    results = []

    for _, row in pool.iterrows():
        code = row["code"]
        name = row["name"]
        market = row["market"]
        sector = row.get("sector", "기타")

        try:
            df = fetch_ohlcv(code, start="2024-01-01")
            if df.empty or len(df) < 30:
                continue

            dates = [d.strftime("%Y-%m-%d") for d in df.index]
            if pred_date not in dates or exec_date not in dates:
                continue

            # Point-in-time slice up to pred_date
            df_pit = df.loc[:pred_date].copy()
            df_ind = compute_point_in_time_indicators(df_pit)
            if df_ind.empty:
                continue

            t1_row = df_ind.iloc[-1]
            t1_close = float(t1_row["Close"])
            t1_val = float(t1_row.get("VolMA20", 1.0) * t1_close)

            # Strategy screening at T-1
            is_matched = False
            strat_label = ""
            target_tp = 5.0
            target_sl = 2.5

            if "당일 단타" in strategy_mode:
                strat_label = "실시간 당일 단타"
                target_tp = 5.0
                target_sl = 2.5
                c_cand = evaluate_intraday_daytrade_candidate(
                    df_pit, bm,
                    min_today_val_krw=min_val_krw,
                    min_day_return=2.0
                )
                if c_cand:
                    is_matched = True
            elif "스나이퍼" in strategy_mode:
                strat_label = "스나이퍼 고확신"
                target_tp = 1.2
                target_sl = 2.0
                s_cand = evaluate_sniper_candidate(df_pit, bm, min_daily_val_krw=min_val_krw)
                if s_cand:
                    is_matched = True
            elif "5% 급등" in strategy_mode:
                strat_label = "5% 급등 타겟"
                target_tp = 5.0
                target_sl = 4.0
                s5_cand = evaluate_5pct_surge_candidate(df_pit, bm, min_today_val_krw=min_val_krw)
                if s5_cand:
                    is_matched = True
            elif "종가배팅" in strategy_mode:
                strat_label = "주도주 종가배팅"
                target_tp = 1.5
                target_sl = 2.0
                cb_cand = evaluate_closing_bet_candidate(df_pit, min_today_val_krw=min_val_krw)
                if cb_cand:
                    is_matched = True
            else:
                score_res = calculate_score_for_row(t1_row)
                if score_res["score"] >= score_cutoff and t1_val >= min_val_krw:
                    is_matched = True
                    strat_label = f"퀀트 {score_res['score']:.0f}점"
                    target_tp = score_res["tp_pct"]
                    target_sl = score_res["sl_pct"]

            score_res = calculate_score_for_row(t1_row)
            score = score_res["score"]
            if not is_matched and score >= max(score_cutoff, 70.0) and t1_val >= min_val_krw:
                is_matched = True
                strat_label = f"고확신 ({score:.0f}점)"
                target_tp = score_res["tp_pct"]
                target_sl = score_res["sl_pct"]

            if not is_matched:
                continue

            # Actual performance on exec_date (T)
            t_row = df.loc[exec_date]
            t_open = float(t_row["Open"])
            t_high = float(t_row["High"])
            t_low = float(t_row["Low"])
            t_close = float(t_row["Close"])

            if t_open <= 0:
                continue

            open_gap_pct = ((t_open - t1_close) / t1_close) * 100.0

            exit_sim = simulate_intraday_exit(
                entry_open=t_open,
                day_high=t_high,
                day_low=t_low,
                day_close=t_close,
                score=score,
                custom_sl_pct=target_sl
            )

            net_ret = calculate_net_trade_return(
                raw_entry_price=exit_sim["raw_entry_price"],
                raw_exit_price=exit_sim["raw_exit_price"],
                avg_daily_val=t1_val
            )

            max_gain_pct = ((t_high - t_open) / t_open) * 100.0
            close_gain_pct = ((t_close - t_open) / t_open) * 100.0
            net_return_pct = net_ret["net_return_pct"]
            is_hit = max_gain_pct >= target_tp or net_return_pct > 0
            excess_return = net_return_pct - bm_intraday_ret

            results.append({
                "code": code,
                "name": name,
                "market": market,
                "sector": sector,
                "t1_score": score,
                "strategy": strat_label,
                "t1_close": t1_close,
                "t1_val_krw": t1_val,
                "rsi": score_res["breakdown"]["rsi_val"],
                "vol_ratio": score_res["breakdown"]["vol_ratio"],
                "t_open": t_open,
                "t_high": t_high,
                "t_low": t_low,
                "t_close": t_close,
                "open_gap_pct": round(open_gap_pct, 2),
                "max_gain_pct": round(max_gain_pct, 2),
                "close_gain_pct": round(close_gain_pct, 2),
                "net_return_pct": round(net_return_pct, 2),
                "target_tp": target_tp,
                "target_sl": target_sl,
                "is_hit": is_hit,
                "exit_code": exit_sim["exit_code"],
                "exit_reason": exit_sim["exit_reason"],
                "excess_return": round(excess_return, 2),
            })
        except Exception:
            continue

    df_res = pd.DataFrame(results)
    if df_res.empty:
        return {
            "pred_date": pred_date,
            "exec_date": exec_date,
            "total_screened": 0,
            "df_results": pd.DataFrame(),
            "kpi": {},
            "diagnosis": {"issues": [], "summary": "해당 조건으로 발굴된 종목이 없습니다."},
            "tuning": {"proposals": [], "simulation": {}}
        }

    total = len(df_res)
    hits = int(df_res["is_hit"].sum())
    win_rate = (hits / total) * 100.0 if total > 0 else 0.0
    avg_net_ret = float(df_res["net_return_pct"].mean())
    avg_max_gain = float(df_res["max_gain_pct"].mean())
    avg_excess_ret = float(df_res["excess_return"].mean())
    tp_count = int(sum(df_res["exit_code"] == "TP"))
    sl_count = int(sum(df_res["exit_code"].isin(["SL", "SL_TIE"])))

    kpi = {
        "total": total,
        "hits": hits,
        "win_rate": round(win_rate, 1),
        "avg_net_ret": round(avg_net_ret, 2),
        "avg_max_gain": round(avg_max_gain, 2),
        "avg_excess_ret": round(avg_excess_ret, 2),
        "tp_count": tp_count,
        "sl_count": sl_count,
        "bm_day_ret": round(bm_day_ret, 2),
        "bm_intraday_ret": round(bm_intraday_ret, 2)
    }

    diagnosis = diagnose_failure_reasons(df_res, bm_day_ret)
    tuning = generate_auto_tuning_recommendations(df_res, {
        "min_val_krw": min_val_krw,
        "score_cutoff": score_cutoff
    })

    return {
        "pred_date": pred_date,
        "exec_date": exec_date,
        "total_screened": total,
        "df_results": df_res,
        "kpi": kpi,
        "diagnosis": diagnosis,
        "tuning": tuning
    }


def diagnose_failure_reasons(df_results: pd.DataFrame, bm_change_pct: float = 0.0) -> Dict[str, Any]:
    """
    Analyzes root causes of failure for Miss/Loss stocks vs Hit/Profit stocks.
    """
    if df_results.empty:
        return {"issues": [], "summary": "분석 가능한 종목 데이터가 없습니다.", "stats_comparison": {}}

    hits_df = df_results[df_results["is_hit"]]
    miss_df = df_results[~df_results["is_hit"]]

    issues = []
    total_miss = len(miss_df)
    if total_miss == 0:
        return {
            "issues": [],
            "summary": "🎉 모든 추천 종목이 목표 익절 또는 양의 수익률을 달성하였습니다! 결함 패턴 없음.",
            "stats_comparison": {}
        }

    # 1. Open Gap Check
    miss_high_gap = miss_df[miss_df["open_gap_pct"] >= 3.0]
    if len(miss_high_gap) > 0 and (len(miss_high_gap) / total_miss) >= 0.35:
        issues.append({
            "type": "HIGH_OPEN_GAP",
            "severity": "HIGH",
            "title": "시초가 과대 갭상승으로 인한 차익 매물 출회",
            "description": f"손실/미도달 종목의 {(len(miss_high_gap)/total_miss)*100:.0f}%({len(miss_high_gap)}건)가 시초가 +3.0% 이상 갭상승 출발 후 음봉 전환되었습니다.",
            "action": "시초가 갭상승 상한선 필터를 2.5% 이하로 보수적 제한 권장"
        })

    # 2. Volume / Liquidity Check
    miss_low_val = miss_df[miss_df["t1_val_krw"] < 15_000_000_000]
    if len(miss_low_val) > 0 and (len(miss_low_val) / total_miss) >= 0.35:
        issues.append({
            "type": "LOW_LIQUIDITY",
            "severity": "MEDIUM",
            "title": "거래대금 부족으로 인한 호가 탄력 저하",
            "description": f"실패 종목의 {(len(miss_low_val)/total_miss)*100:.0f}%({len(miss_low_val)}건)가 일평균 거래대금 150억원 미만의 비주도주였습니다.",
            "action": "최소 거래대금 필터를 150억원 이상으로 상향 조정 권장"
        })

    # 3. RSI Overbought Check
    miss_high_rsi = miss_df[miss_df["rsi"] >= 70.0]
    if len(miss_high_rsi) > 0 and (len(miss_high_rsi) / total_miss) >= 0.3:
        issues.append({
            "type": "OVERBOUGHT_RSI",
            "severity": "MEDIUM",
            "title": "RSI 과열권(70 이상) 고점 추격 진입",
            "description": f"실패 종목 중 {len(miss_high_rsi)}건이 이미 RSI 과매수(70 이상) 영역에서 추천되어 상방 탄력이 둔화되었습니다.",
            "action": "RSI 70 미만 눌림목/첫돌파 종목 우선 필터링 적용 권장"
        })

    # 4. Market Systematic Risk
    if bm_change_pct <= -1.0:
        issues.append({
            "type": "MARKET_CRASH",
            "severity": "HIGH",
            "title": f"시장 지수 급락 영향 (KODEX 200 {bm_change_pct:+.2f}%)",
            "description": f"검증일 시장 대표 지수가 {bm_change_pct:+.2f}% 하락하여 개별 종목 수급이 동반 위축되었습니다.",
            "action": "시장 지수 급락 시 스크리너 최소 스코어 컷오프를 +5점 상향하거나 비중 50% 축소 권장"
        })

    stats_comparison = {
        "metric": ["종목 수", "평균 시초가 갭 (%)", "평균 거래대금 (억원)", "평균 RSI", "평균 장중 최대상승 (%)"],
        "hits": [
            f"{len(hits_df)}개",
            f"{hits_df['open_gap_pct'].mean():+.2f}%" if not hits_df.empty else "-",
            f"{hits_df['t1_val_krw'].mean()/1e8:.0f}억" if not hits_df.empty else "-",
            f"{hits_df['rsi'].mean():.1f}" if not hits_df.empty else "-",
            f"{hits_df['max_gain_pct'].mean():+.2f}%" if not hits_df.empty else "-",
        ],
        "misses": [
            f"{len(miss_df)}개",
            f"{miss_df['open_gap_pct'].mean():+.2f}%" if not miss_df.empty else "-",
            f"{miss_df['t1_val_krw'].mean()/1e8:.0f}억" if not miss_df.empty else "-",
            f"{miss_df['rsi'].mean():.1f}" if not miss_df.empty else "-",
            f"{miss_df['max_gain_pct'].mean():+.2f}%" if not miss_df.empty else "-",
        ]
    }

    summary = f"총 {len(df_results)}건 중 성공 {len(hits_df)}건, 실패/손절 {len(miss_df)}건 감지됨. 주요 개선 항목 {len(issues)}건 도출."

    return {
        "issues": issues,
        "summary": summary,
        "stats_comparison": stats_comparison
    }


def generate_auto_tuning_recommendations(
    df_results: pd.DataFrame,
    current_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates optimized screener parameter updates and simulates Before vs After performance.
    """
    if df_results.empty:
        return {"proposals": [], "simulation": {}}

    proposals = []
    opt_mask = pd.Series(True, index=df_results.index)

    miss_df = df_results[~df_results["is_hit"]]
    if not miss_df.empty and (miss_df["open_gap_pct"] >= 3.0).sum() >= 1:
        proposals.append({
            "param_name": "시초가 대비 상승률 구간 상한",
            "param_key": "max_open_gain",
            "current_val": "4.5%",
            "recommended_val": "2.5%",
            "target_val_float": 2.5,
            "reason": "시초 갭 3.0% 이상 고점 추격 종목 사전 차단"
        })
        opt_mask = opt_mask & (df_results["open_gap_pct"] <= 2.5)

    cur_min_val = current_params.get("min_val_krw", 10_000_000_000)
    if not miss_df.empty and (miss_df["t1_val_krw"] < 15_000_000_000).sum() >= 1:
        proposals.append({
            "param_name": "최소 일평균 거래대금",
            "param_key": "min_trading_val",
            "current_val": f"{cur_min_val/1e8:.0f}억원",
            "recommended_val": "150억원",
            "target_val_float": 150.0,
            "reason": "호가 탄력이 검증된 시장 주도주로 압축"
        })
        opt_mask = opt_mask & (df_results["t1_val_krw"] >= 15_000_000_000)

    cur_cutoff = current_params.get("score_cutoff", 65.0)
    low_score_misses = miss_df[miss_df["t1_score"] < 72.0]
    if len(low_score_misses) >= 1:
        proposals.append({
            "param_name": "최소 퀀트 스코어 컷오프",
            "param_key": "min_score_cutoff",
            "current_val": f"{cur_cutoff:.0f}점",
            "recommended_val": "72점",
            "target_val_float": 72.0,
            "reason": "승률이 불확실한 60점대 경계 종목 배제"
        })
        opt_mask = opt_mask & (df_results["t1_score"] >= 72.0)

    if not proposals:
        proposals.append({
            "param_name": "최소 퀀트 스코어 컷오프",
            "param_key": "min_score_cutoff",
            "current_val": f"{cur_cutoff:.0f}점",
            "recommended_val": f"{cur_cutoff + 2.0:.0f}점",
            "target_val_float": cur_cutoff + 2.0,
            "reason": "현행 파라미터가 양호하여 미세 고확신 종목 집중 유지"
        })

    df_opt = df_results[opt_mask]
    before_total = len(df_results)
    before_win_rate = (df_results["is_hit"].sum() / before_total) * 100.0 if before_total > 0 else 0.0
    before_avg_ret = float(df_results["net_return_pct"].mean()) if before_total > 0 else 0.0

    after_total = len(df_opt)
    after_win_rate = (df_opt["is_hit"].sum() / after_total) * 100.0 if after_total > 0 else before_win_rate
    after_avg_ret = float(df_opt["net_return_pct"].mean()) if after_total > 0 else before_avg_ret

    filtered_losses = int((~df_results.loc[~opt_mask, "is_hit"]).sum())

    simulation = {
        "before_total": before_total,
        "before_win_rate": round(before_win_rate, 1),
        "before_avg_ret": round(before_avg_ret, 2),
        "after_total": after_total,
        "after_win_rate": round(after_win_rate, 1),
        "after_avg_ret": round(after_avg_ret, 2),
        "filtered_losses": filtered_losses,
        "win_rate_boost": round(after_win_rate - before_win_rate, 1)
    }

    return {
        "proposals": proposals,
        "simulation": simulation
    }
