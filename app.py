"""
개인용 관심종목 익일 전망 및 상승 후보 스크리닝 앱
Streamlit 3-Tab 반응형 대시보드
Tab 1: 관심종목 & 실시간 모니터링 / 상승 후보 스크리너
Tab 2: 워크포워드 검증 & 벤치마크 리포트 (부트스트랩 95% CI)
Tab 3: 예측 다이어리 & 라이브 추적 (Model Decay 감지)
"""

import os
import sys
from datetime import datetime, timedelta

# Fix Windows SSL certificate verification before importing network libraries
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Root package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database.models import (
    init_db,
    get_db_connection,
    update_watchlist_group,
    batch_update_watchlist_groups,
    get_watchlist_groups
)
from src.collectors.market_data import fetch_ohlcv, get_latest_market_info, get_benchmark_ohlcv
from src.collectors.krx_universe import (
    sync_krx_universe,
    get_universe,
    search_ticker,
    get_stock_metadata,
    batch_add_to_watchlist,
    clear_watchlist,
    normalize_sector
)
from src.core.scoring import evaluate_stock_latest, compute_point_in_time_indicators, calculate_score_for_row
from src.core.high_winrate_strategies import (
    evaluate_market_regime,
    evaluate_sniper_candidate,
    evaluate_closing_bet_candidate,
    simulate_closing_bet_trade,
    evaluate_5pct_surge_candidate,
    simulate_5pct_surge_trade,
    evaluate_intraday_daytrade_candidate,
    simulate_intraday_daytrade
)
from src.core.risk_manager import calculate_half_kelly_weight, PortfolioRiskManager
from src.backtest.walk_forward import run_walk_forward_backtest
from src.database.diary_manager import (
    record_daily_prediction,
    settle_pending_predictions,
    get_diary_history,
    evaluate_model_decay
)
from src.automation.scheduler import run_post_market_job, run_pre_market_job
from src.core.daily_verifier import (
    get_available_trading_dates,
    run_daily_point_in_time_verification,
    diagnose_failure_reasons,
    generate_auto_tuning_recommendations
)


# Page configuration (Office stealth friendly)
st.set_page_config(
    page_title="Analytics Workspace",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for badges, cards, and Korean fonts (Office Stealth Mode)
st.markdown("""
<style>
    /* Clean compact layout without wasted viewport padding, ensuring top tabs are clearly visible below Streamlit header */
    .block-container {
        padding-top: 3.8rem !important;
        padding-bottom: 1.0rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.35rem !important;
    }
    hr {
        margin: 0.35rem 0 !important;
    }

    /* Office Stealth Mode: Discreet compact typography */
    .office-heading {
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        margin: 2px 0 6px 0 !important;
        letter-spacing: -0.2px;
    }
    .office-subheading {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        margin: 4px 0 4px 0 !important;
    }
    h1, h2, h3, [data-testid="stSubheader"] {
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        margin-top: 2px !important;
        margin-bottom: 4px !important;
        padding: 0 !important;
    }
    [data-testid="stSubheader"] * {
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }
    h4, h5, h6 {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        margin-top: 2px !important;
        margin-bottom: 3px !important;
    }
    div[data-testid="stTabs"] {
        margin-top: 0px !important;
        margin-bottom: 8px !important;
    }
    div[data-baseweb="tab-list"] {
        gap: 6px !important;
    }
    button[data-baseweb="tab"] {
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        padding: 4px 14px !important;
        color: #64748B !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0F172A !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #475569 !important;
    }
    .stMarkdown table {
        font-size: 0.78rem !important;
        line-height: 1.3 !important;
    }
    .stMarkdown th, .stMarkdown td {
        padding: 3px 6px !important;
    }
    .main-title {
        display: none !important;
    }
    .sub-title {
        display: none !important;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 6px;
        padding: 8px 12px;
        border-left: 3px solid #64748B;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .badge-verified {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-experimental {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-bullish {
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
        padding: 2px 6px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-neutral {
        background-color: #F8FAFC;
        color: #475569;
        border: 1px solid #E2E8F0;
        padding: 2px 6px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    .badge-bearish {
        background-color: #F8FAFC;
        color: #64748B;
        border: 1px solid #E2E8F0;
        padding: 2px 6px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.75rem;
    }
    /* Compact KPI Metrics styling for Image 1 */
    div[data-testid="stMetric"] {
        background-color: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 6px !important;
        padding: 4px 10px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    }
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        margin-bottom: 0px !important;
    }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        color: #1E293B !important;
        line-height: 1.15 !important;
    }
    div[data-testid="stMetricDelta"], div[data-testid="stMetricDelta"] * {
        font-size: 0.68rem !important;
    }

    /* Office Stealth Mode: Button Styling (Discreet corporate slate/neutral) */
    div.stButton > button {
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        padding: 0.35rem 0.8rem !important;
        min-height: 2rem !important;
        border-radius: 5px !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button p {
        font-size: 0.80rem !important;
        font-weight: 600 !important;
    }
    /* Primary buttons: Subdued Slate Grey instead of neon/coral red */
    div.stButton > button[kind="primary"],
    [data-testid="stBaseButton-primary"] {
        background-color: #475569 !important;
        border: 1px solid #334155 !important;
        color: #FFFFFF !important;
        box-shadow: none !important;
    }
    div.stButton > button[kind="primary"] p,
    [data-testid="stBaseButton-primary"] p {
        color: #FFFFFF !important;
    }
    div.stButton > button[kind="primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {
        background-color: #334155 !important;
        border-color: #1E293B !important;
        color: #FFFFFF !important;
    }
    div.stButton > button[kind="primary"]:active,
    [data-testid="stBaseButton-primary"]:active {
        background-color: #1E293B !important;
        border-color: #0F172A !important;
    }
    div.stButton > button[kind="primary"]:focus,
    [data-testid="stBaseButton-primary"]:focus {
        box-shadow: 0 0 0 2px rgba(71, 85, 105, 0.25) !important;
    }

    /* Secondary buttons: Clean subtle border, soft grey background */
    div.stButton > button[kind="secondary"],
    [data-testid="stBaseButton-secondary"] {
        background-color: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        color: #334155 !important;
        box-shadow: none !important;
    }
    div.stButton > button[kind="secondary"] p,
    [data-testid="stBaseButton-secondary"] p {
        color: #334155 !important;
    }
    div.stButton > button[kind="secondary"]:hover,
    [data-testid="stBaseButton-secondary"]:hover {
        background-color: #F1F5F9 !important;
        border-color: #94A3B8 !important;
        color: #0F172A !important;
    }
    div.stButton > button[kind="secondary"]:focus,
    [data-testid="stBaseButton-secondary"]:focus {
        box-shadow: 0 0 0 2px rgba(203, 213, 225, 0.5) !important;
    }

    /* Office Stealth Sliders: Slate grey accents instead of bright red */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #475569 !important;
        border-color: #334155 !important;
        box-shadow: none !important;
    }
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="rgb(255, 75, 75)"],
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="#ff4b4b"],
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="RGB(255, 75, 75)"] {
        background-color: #475569 !important;
    }
    div[data-testid="stSlider"] div[data-testid="stThumbValue"],
    div[data-testid="stSlider"] div[data-testid="stThumbValue"] * {
        color: #334155 !important;
        font-size: 0.76rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSliderTickBar"] > div {
        background: #CBD5E1 !important;
    }
    div[data-testid="stSlider"] label p {
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }

    /* Progress indicators: Soft slate styling */
    div[data-testid="stProgress"] > div > div > div > div {
        background-color: #475569 !important;
    }

    /* Segmented pill radio buttons for crisp layout */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        background: #F8FAFC !important;
        padding: 4px !important;
        border-radius: 6px !important;
        gap: 6px !important;
        border: 1px solid #E2E8F0 !important;
        display: flex !important;
        flex-wrap: wrap !important;
        align-items: center !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: #FFFFFF !important;
        padding: 4px 12px !important;
        border-radius: 5px !important;
        border: 1px solid #E2E8F0 !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: #F1F5F9 !important;
        border-color: #CBD5E1 !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        margin: 0 !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background: #334155 !important;
        border-color: #1E293B !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06) !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database on app start
init_db()


# ---------------------------------------------------------
# Sidebar Controls & Portfolio Risk Status
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### AlphaQuant Engine")
    st.caption("로컬 데이터 분석 & 스크리너")

    st.markdown("---")
    st.markdown("#### 포트폴리오 리스크 상태")

    # Load Risk State from DB
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM risk_state WHERE id = 1")
        risk_row = cursor.fetchone()

    equity = float(risk_row["current_equity"]) if risk_row else 10_000_000.0
    daily_loss = float(risk_row["daily_realized_loss"]) if risk_row else 0.0
    cb_active = bool(risk_row["is_circuit_breaker_active"]) if risk_row else False
    cooldown = bool(risk_row["cooldown_active"]) if risk_row else False

    col_e1, col_e2 = st.columns(2)
    col_e1.metric("총 운용자산", f"{equity:,.0f}원")
    col_e2.metric("당일 누적손실", f"{daily_loss:,.0f}원", delta=f"{(daily_loss/equity)*100:.2f}%" if daily_loss != 0 else "0.0%")

    if cb_active:
        st.error("일일 서킷브레이커 발동 (-3% 초과 손실). 신규 매수 차단")
    elif cooldown:
        st.warning("3회 연속 손절 쿨다운 활성화 중")
    else:
        st.success("포트폴리오 상태: 정상 (진입 가능)")

    st.markdown("---")
    with st.expander("시스템 수동 관리 도구 (선택 사항)", expanded=False):
        st.caption("※ 평소에는 스케줄러가 자동 처리하므로 누르실 필요가 없습니다.")
        if st.button("15:40 마감 정산 & 예측 수동 실행", use_container_width=True):
            with st.spinner("15:40 마감 정산 및 익일 예측 실행 중..."):
                res = run_post_market_job()
                st.toast(f"마감 정산 완료 (정산 {res['settled_count']}건, 예측 {res['new_predictions_count']}건)")
                st.rerun()

        if st.button("08:30 장전 브리핑 수동 생성", use_container_width=True):
            with st.spinner("08:30 장전 브리핑 생성 중..."):
                res = run_pre_market_job()
                st.toast(f"장전 브리핑 완료 (후보 {res['candidates_count']}건)")
                st.info(res["briefing"])

        if st.button("KRX 유니버스 최신 동기화", use_container_width=True):
            with st.spinner("KRX 상장 및 상폐 종목 동기화 중..."):
                cnt = sync_krx_universe(force=True)
                st.toast(f"유니버스 {cnt}종목 동기화 완료!")


# Top Header: Removed for Office Stealth Mode


# ---------------------------------------------------------
# High-performance Cached Metrics Loader for Large Watchlists (300+ stocks)
# ---------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_all_watchlist_metrics(codes_tuple):
    """
    Fast batch evaluator for watchlist stocks.
    Loads parquet caches in parallel/batch and computes point-in-time scores.
    """
    results = []
    bm_ohlcv = get_benchmark_ohlcv()
    for item in codes_tuple:
        if len(item) == 6:
            code, name, market, sector, notes, group_name = item
        else:
            code, name, market, sector, notes = item[:5]
            group_name = "기본그룹"

        try:
            df = fetch_ohlcv(code)
            price_info = get_latest_market_info(code)
            score_info = evaluate_stock_latest(df)
        except Exception:
            price_info = {"close": 0.0, "change_pct": 0.0, "volume": 0, "date": "N/A"}
            score_info = None

        if score_info:
            score = score_info["score"]
            label = score_info["label"]
            tp = score_info["tp_pct"]
            sl = score_info["sl_pct"]
            bk = score_info["breakdown"]
            intraday_info = score_info.get("intraday_outlook", {})
            nextday_info = score_info.get("nextday_outlook", {})
            intraday_tag = intraday_info.get("tag", "⏸️ 거래소강")
            intraday_desc = intraday_info.get("desc", "장중 시세 데이터 대기")
            intraday_open_pct = intraday_info.get("open_gain_pct", 0.0)
            nextday_tag = nextday_info.get("tag", f"{label} ({score:.1f}점)")
            nextday_desc = nextday_info.get("desc", "기술적 추세 기반 전망")
        else:
            score = 50.0
            label = "중립"
            tp = 1.5
            sl = 2.0
            bk = {
                "rsi_val": 50.0,
                "ma_status": "-",
                "vol_ratio": 1.0,
                "bb_pct": 0.5,
                "rsi_score": 0.0,
                "ma_score": 0.0,
                "vol_score": 0.0,
                "bb_score": 0.0
            }
            intraday_tag = "⏸️ 거래소강"
            intraday_desc = "시세 분석 대기"
            intraday_open_pct = 0.0
            nextday_tag = "중립 (50.0점)"
            nextday_desc = "기본 퀀트 점수"

        is_sniper = False
        is_closing_bet = False
        is_surge_5pct = False
        is_daytrade = False
        if df is not None and not df.empty and len(df) >= 30:
            try:
                dtrade = evaluate_intraday_daytrade_candidate(df, bm_ohlcv, min_today_val_krw=10_000_000_000)
                if dtrade:
                    is_daytrade = True
                s5 = evaluate_5pct_surge_candidate(df, bm_ohlcv, min_today_val_krw=10_000_000_000, min_day_return=10.0)
                if s5:
                    is_surge_5pct = True
                snp = evaluate_sniper_candidate(df, bm_ohlcv, min_daily_val_krw=5_000_000_000)
                if snp:
                    is_sniper = True
                cbet = evaluate_closing_bet_candidate(df, min_today_val_krw=10_000_000_000)
                if cbet:
                    is_closing_bet = True
            except Exception:
                pass

        badge = "● 검증됨" if score >= 75.0 else "● 실험적"
        if is_daytrade:
            strat_tag = "당일단타 (당일 +5%)"
            tp = 5.0
            sl = 2.5
        elif is_surge_5pct:
            strat_tag = "5%타겟 (마감 직전)"
            tp = 5.0
            sl = 4.0
        elif is_sniper:
            strat_tag = "스나이퍼 (오전 9시)"
            tp = 1.2
            sl = 2.0
        elif is_closing_bet:
            strat_tag = "종가배팅 (마감 직전)"
            tp = 1.5
            sl = 2.0
        else:
            strat_tag = "-"

        results.append({
            "code": code,
            "name": name,
            "group_name": group_name or "기본그룹",
            "market": market,
            "sector": sector or notes or "기타",
            "close": price_info["close"],
            "change_pct": price_info["change_pct"],
            "volume": price_info["volume"],
            "score": score,
            "label": label,
            "intraday_tag": intraday_tag,
            "intraday_desc": intraday_desc,
            "intraday_open_pct": intraday_open_pct,
            "nextday_tag": nextday_tag,
            "nextday_desc": nextday_desc,
            "strategy_tag": strat_tag,
            "is_sniper": is_sniper,
            "is_closing_bet": is_closing_bet,
            "is_surge_5pct": is_surge_5pct,
            "is_daytrade": is_daytrade,
            "is_high_conviction": is_sniper or is_surge_5pct or is_daytrade,
            "tp_pct": tp,
            "sl_pct": sl,
            "rsi": bk.get("rsi_val", 50.0),
            "vol_ratio": bk.get("vol_ratio", 1.0),
            "ma_status": bk.get("ma_status", "-"),
            "bb_pct": bk.get("bb_pct", 0.5),
            "verified_badge": badge,
            "breakdown": bk,
        })
    return results


# ---------------------------------------------------------
# 3 Main Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. 시장 전체 익일 상승 후보 & 고확신 스크리너",
    "2. 관심종목 실시간 스코어보드",
    "3. 워크포워드 검증 & 벤치마크 리포트",
    "4. 예측 다이어리 & 라이브 추적",
    "5. 전일 성과 자가검증 & AI 전략 최적화",
])


# =========================================================
# TAB 1: 시장 전체 익일 상승 후보 & 고확신 스크리너
# =========================================================
with tab1:
    st.markdown('<div class="office-heading">1. 시장 전체 익일 상승 후보 & 고확신 스크리너</div>', unsafe_allow_html=True)
    
    
    sc_mode = st.radio(
        "스크리닝 전략 모드 선택",
        [
            "⚡ 실시간 당일 단타 (5% 익절)",
            "🎯 스나이퍼 고확신 (눌림목 반등)",
            "🚀 5% 급등 타겟 (1~2일 스윙)",
            "🌙 주도주 종가배팅 (익일 시초 갭)",
            "📊 일반 퀀트 스코어링"
        ],
        horizontal=True,
        key="sc_mode_radio_sel"
    )

    if "당일 단타" in sc_mode:
        m_tag = '<span style="color:#0284C7; font-weight:700;">[실시간 당일 단타]</span> 장중 09:10~14:30 수급 폭발 주도주 포착 · 당일 +5.0% 익절 / 15:15 미도달 시 종가 전량 청산 (No Overnight)'
    elif "스나이퍼" in sc_mode:
        m_tag = '<span style="color:#059669; font-weight:700;">[스나이퍼 고확신]</span> 오전 09:00 시초가 진입 · 20일선 첫 눌림목 반등 공략 (익절 +1.2% / 손절 -2.0% 엄수, 승률 80% 타겟)'
    elif "5% 급등" in sc_mode or "5% 돌파" in sc_mode:
        m_tag = '<span style="color:#D97706; font-weight:700;">[5% 급등 타겟]</span> 마감 직전 15:20 종가 진입 · 상한가/초대형 거래대금 돌파봉 1~2일 내 5% 급등 수확 (손절 -4.0%)'
    elif "종가배팅" in sc_mode:
        m_tag = '<span style="color:#7C3AED; font-weight:700;">[주도주 종가배팅]</span> 마감 직전 15:20 동시호가 매수 -> 익일 09:00 시초 갭상승(+1.5%↑) 즉시 익절 (Overnight)'
    else:
        m_tag = '<span style="color:#475569; font-weight:700;">[일반 퀀트]</span> Point-in-Time 복합 팩터 100점 만점 랭킹 시스템'

    st.markdown(f'<div style="font-size:0.79rem; color:#475569; padding:2px 0 6px 2px;">{m_tag}</div>', unsafe_allow_html=True)

    # 시간대별 최적 거래대금 설정 가이드 및 실시간 자동 세팅 엔진
    def get_auto_time_recommendation(dt=None):
        if dt is None:
            dt = datetime.now()
        cur_hm = dt.hour * 100 + dt.minute

        if 910 <= cur_hm < 940:
            return {
                "slot_name": "09:10 ~ 09:40 [장초반 수급 집중]",
                "val": 100,
                "gain": (3.0, 6.5),
                "vol": 0.6,
                "highlight_idx": 0,
                "badge_title": "09:10 ~ 09:40 장초반 수급 집중 구간",
                "badge_desc": "거래대금 100억↑, 시초대비 +3.0%~+6.5% (초입 돌파 포착)",
                "tip": "장초반 윗꼬리 물림 방지 및 +5% 익절 공간 확보"
            }
        elif 940 <= cur_hm < 1100:
            return {
                "slot_name": "09:40 ~ 11:00 [★골든타임]",
                "val": 200,
                "gain": (3.0, 8.5),
                "vol": 0.6,
                "highlight_idx": 1,
                "badge_title": "09:40 ~ 11:00 [★골든타임] 주도주 2차 돌파 구간",
                "badge_desc": "거래대금 200억↑, 시초대비 +3.0%~+8.5% (최고 승률 권장값)",
                "tip": "눌림목 지지 확인 후 2차 파동 돌파 적중률 극대화"
            }
        elif 1100 <= cur_hm < 1300:
            return {
                "slot_name": "11:00 ~ 13:00 [점심 횡보장]",
                "val": 200,
                "gain": (4.5, 9.0),
                "vol": 0.8,
                "highlight_idx": 2,
                "badge_title": "11:00 ~ 13:00 점심 횡보장 대장주 압축 구간",
                "badge_desc": "거래대금 200억↑, 시초대비 +4.5%~+9.0% (고가 지지력 확인)",
                "tip": "점심 거래량 감소 구간, 탄력 유지 대장주만 선별"
            }
        elif 1300 <= cur_hm < 1430:
            return {
                "slot_name": "13:00 ~ 14:30 [오후 2차 수급]",
                "val": 300,
                "gain": (5.0, 10.0),
                "vol": 1.0,
                "highlight_idx": 3,
                "badge_title": "13:00 ~ 14:30 오후 2차 수급 슈팅 구간",
                "badge_desc": "거래대금 300억↑, 시초대비 +5.0%~+10.0% (오후 2차 돌파)",
                "tip": "상한가/VI 직행 추진력을 갖춘 최상위 주도주 공략"
            }
        elif 1430 <= cur_hm < 1535:
            return {
                "slot_name": "14:30 ~ 15:35 [장마감 임박/동시호가]",
                "val": 200,
                "gain": (3.0, 8.5),
                "vol": 0.6,
                "highlight_idx": 4,
                "badge_title": "14:30 ~ 15:35 장마감 임박 (당일단타 신규진입 OFF)",
                "badge_desc": "신규 진입 중단 및 보유분 15:15 전량 청산 집중",
                "tip": "신규 매수는 마감 직전 '5% 급등 타겟' 또는 '종가배팅' 모드 권장"
            }
        else:
            return {
                "slot_name": "장외/마감 정산 (내일 실전 준비)",
                "val": 200,
                "gain": (3.0, 8.5),
                "vol": 0.6,
                "highlight_idx": 1,
                "badge_title": f"현재 {dt.strftime('%H:%M')} (장외/마감 시간)",
                "badge_desc": "골든타임 표준 권장값 (거래대금 200억↑, 시초대비 +3.0%~+8.5%) 자동 세팅",
                "tip": "마감 데이터 기준 최적 백테스트 및 내일 장초반 준비"
            }

    rec_time = get_auto_time_recommendation()
    now_str = datetime.now().strftime("%H:%M")
    cur_slot_id = f"slot_{rec_time['highlight_idx']}"

    # Auto-sync session state when slot changes or sync is triggered
    if "sc_last_applied_slot" not in st.session_state or st.session_state.get("sc_trigger_time_sync") or st.session_state.get("sc_last_applied_slot") != cur_slot_id:
        st.session_state["sc_min_daytrade_val"] = rec_time["val"]
        st.session_state["sc_intraday_gain_range"] = rec_time["gain"]
        st.session_state["sc_min_daytrade_vol_ratio"] = rec_time["vol"]
        st.session_state["sc_last_applied_slot"] = cur_slot_id
        st.session_state["sc_trigger_time_sync"] = False

    # 1) 전략 원칙 & 시간대별 설정 가이드 통합 접이식(Expander)
    with st.expander("📖 [전략 핵심 원칙 & 시간대별 설정 가이드] (필요 시 클릭하여 열기)", expanded=False):
        g_tab1, g_tab2 = st.tabs(["전략 핵심 원칙 & 시장 체제 점검", "시간대별 권장 설정 가이드"])
        with g_tab1:
            if "당일 단타" in sc_mode:
                bm_ohlcv = get_benchmark_ohlcv()
                mkt_ok, mkt_msg, mkt_meta = evaluate_market_regime(bm_ohlcv)
                if mkt_ok:
                    st.success(f"**[시장 체제 확인]** KODEX 200 지수가 20일 이동평균선 상단에 위치하여 **당일 단타 모드 신규 매수가 적극 허용**됩니다. ({mkt_msg})")
                else:
                    st.warning(f"**[시장 체제 경고]** KODEX 200 지수가 20일선 아래(조정/하락장)에 위치합니다. 하락장에서는 보수적인 비중 조절 및 칼손절(-2.5%) 준수가 필수입니다! ({mkt_msg})")

                st.markdown(
                    """
### [실시간 당일 단타] 5% 돌파 모드 핵심 원칙 (당일 완결 / No Overnight)
1. **진입 타이밍**: 장중 09:10 ~ 14:30 실시간 수급 폭발 종목 포착
2. **수급 & 거래대금**: 당일 거래대금 200억↑ & 장중 이미 전일 거래량 60% 이상 돌파 (거래량 폭발 주도주)
3. **시초가 돌파 모멘텀**: 시초가 대비 +3.0% ~ +8.5% 구간 돌파 (고점 뇌동매매 방지 및 상승 초입 포착)
4. **탄탄한 양봉 지지**: 종가가 당일 진폭 상위 65% 이상 유지 (윗꼬리 긴 투매형 캔들 배제)
5. **목표 및 당일 청산**: 진입 즉시 **+5.0% 익절 자동주문 걸어두기 / 손절 -2.5% 엄수 / 15:15 미도달 시 종가 전량 청산 (오버나잇 리스크 제로!)**
                    """,
                    unsafe_allow_html=True
                )
            elif "스나이퍼" in sc_mode:
                bm_ohlcv = get_benchmark_ohlcv()
                mkt_ok, mkt_msg, mkt_meta = evaluate_market_regime(bm_ohlcv)
                if mkt_ok:
                    st.success(f"**[시장 체제 확인]** KODEX 200 지수가 20일 이동평균선 상단에 위치하여 **스나이퍼 신규 매수가 허용**됩니다. ({mkt_msg})")
                else:
                    st.warning(f"**[시장 체제 경고]** KODEX 200 지수가 20일선 아래(조정/하락장)에 위치합니다. 스나이퍼 전략은 하락장에서 무리한 진입을 전면 차단합니다! ({mkt_msg})")

                st.markdown(
                    """
### [스나이퍼 고확신 모드] 핵심 원칙 (80% 승률 타겟 - 20일선 눌림목 반등)
- **시장 필터**: KODEX 200 > 20일선 상승 추세에서만 진입 (지수 급락 리스크 회피)
- **우량 유동성**: 20일 평균 거래대금 최소 100억~200억 이상 (호가 왜곡 방지 및 풍부한 체결력)
- **눌림목 이격도**: 20일선 대비 주가 이격도 **98% ~ 103.5%** (상승 5일선 골든 후 20일선 첫 지지 반등)
- **핵심 체결 룰 (시초가 갭 제한)**: 익일 시초가가 전일종가 대비 **-1.5% ~ +1.5%** 이내일 때만 진입 (+1.5% 초과 갭상승 시 뇌동매매 진입 금지!)
- **비대칭 목표**: 익절 +1.2% / 손절 -2.0%
                    """,
                    unsafe_allow_html=True
                )
            elif "종가배팅" in sc_mode:
                st.markdown(
                    """
### [주도주 종가배팅 모드] 핵심 원칙 (Overnight Close-to-Open)
- **탐색 시점**: 매일 15:20 ~ 15:30 (장 마감 직전 동시호가 전후)
- **거래대금 폭발**: 당일 거래대금 최소 200억~300억 이상 (당일 시장 주도 테마 대장주)
- **장대양봉 고가마감**: 당일 주가 +3.0% 이상 장대양봉 & 일중 고가 부근 마감 (윗꼬리 18% 이하로 종가 고가마감)
- **청산 룰**: 당일 15:20 종가 매수 -> 익일 09:00 시초가 갭상승(+1.5%↑) 시 즉시 시초가 익절 또는 장초반 +1.5% 슈팅 익절 (손절 -2.0%)
                    """,
                    unsafe_allow_html=True
                )
            elif "5% 급등" in sc_mode or "5% 돌파" in sc_mode:
                st.markdown(
                    """
### [5% 급등 타겟 모드] 핵심 원칙 (마감 직전 15:20 진입: 1~2일 5% 급등 수확)
- **빅데이터 검증**: 일반 종목의 익일 5% 도달률은 12%에 불과하지만, **[거래대금 500억↑ + 신고가 첫 상한가/준상한가(+15%~+29.8%) + 윗꼬리 8% 이하 고가마감]** 주도주는 익일 +5% 도달률 **68.8%**, **1~2일 스윙 보유 시 +5% 도달률이 79.2% ~ 81.2% (승률 80%대 달성)**에 달합니다.
- **탐색 & 진입 시점**: 매일 마감 직전 15:20 (상한가 굳히기 or 초대형 거래대금 돌파봉 종가 매수)
- **목표 및 손절**: 장중 **+5.0%** 도달 즉시 전량 자동 익절 (1일차 미도달 시 2일차까지 홀딩, 손절선 **-4.0%** 엄수)
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
### [일반 퀀트 스코어링 모드] 가이드 (기본 종합 점수제)
- **특징**: Point-in-Time 스코어링 엔진으로 수급, 모멘텀, 거래대금, 변동성, 재무 건전성을 복합 평가하여 100점 만점으로 순위를 산출합니다.
- **권장 거래대금**: 최소 50억~100억 원 이상 (유동성 부족으로 인한 슬리피지 방지).
                    """,
                    unsafe_allow_html=True
                )

        with g_tab2:
            if "당일 단타" in sc_mode:
                active_tag = ' <span style="color:#0F172A; font-weight:700; font-size:0.68rem; background:#E2E8F0; padding:1px 5px; border-radius:4px; border:1px solid #94A3B8;">▶ 현재 자동 세팅됨</span>'
                r0_t = active_tag if rec_time["highlight_idx"] == 0 else ""
                r1_t = active_tag if rec_time["highlight_idx"] == 1 else ""
                r2_t = active_tag if rec_time["highlight_idx"] == 2 else ""
                r3_t = active_tag if rec_time["highlight_idx"] == 3 else ""
                r4_t = active_tag if rec_time["highlight_idx"] == 4 else ""

                st.markdown(
                    f"""
| 시간대 | 장세 특성 | 권장 누적 거래대금 | 권장 시초가대비 상승률 구간 | 매매 행동 요령 & 슬라이더 설정 팁 |
| :--- | :--- | :--- | :--- | :--- |
| **09:10 ~ 09:40**{r0_t} | **장초반 거래 집중**<br>(변동성/거래량 극대화) | **50억 ~ 100억 원 이상** | **+3.0% ~ +6.5%**<br>*(상승 초입 포착)* | 장초반 고점 윗꼬리(설거지) 물림을 방지하기 위해 슬라이더 상한을 +6.5%로 낮춰 잡습니다. 시초가를 막 뚫고 올라서는 초입을 잡아야 +5% 익절 공간이 확보됩니다. |
| **09:40 ~ 11:00**<br>**[골든타임]**{r1_t} | **주도주 압축 & 2차 돌파**<br>(당일 단타 승률 최고 구간) | **150억 ~ 200억 원 이상**<br>*(가장 추천: 200억)* | **+3.0% ~ +8.5%**<br>*(★최고 승률 기본 권장값)* | 장초반 1차 슈팅 후 눌림목을 거쳐 재차 치고 나가는 2차 파동 구간입니다. 거래대금 200억 이상 유입 + 슬라이더 기본 권장값(+3.0% ~ +8.5%)에서 적중률이 극대화됩니다. |
| **11:00 ~ 13:00**{r2_t} | **점심 횡보장**<br>(거래량 급감, 소강상태) | **200억 ~ 300억 원 이상** | **+4.5% ~ +9.0%**<br>*(고가 지지력 확인)* | 거래량이 마르는 점심 시간대에는 애매한 +2~3%대 종목은 흘러내립니다. 이미 +4.5% 이상 상승 탄력을 유지하며 당일 중심선을 지키는 대장주만 진입합니다. |
| **13:00 ~ 14:30**{r3_t} | **오후 2차 수급 유입**<br>(마감 전 주도 섹터 랠리) | **300억 원 이상** | **+5.0% ~ +10.0%**<br>*(오후 2차 슈팅)* | 오후장 돌파 매매는 상한가(VI)로 직행하는 강한 추진력이 필요합니다. 시초가 대비 최소 +5% 이상 올라서서 전고점을 재돌파하는 주도주를 선별합니다. |
| **14:30 이후**{r4_t} | **장마감 임박**<br>(당일 청산 준비 구간) | **신규 매수 진입 금지 (OFF)** | **신규 진입 금지 (OFF)** | 당일 15:15 전량 청산 원칙이므로 신규 진입 시 5% 익절할 시간적 여유가 부족합니다. 보유분 익절/손절 청산에만 집중하세요. |
                    """,
                    unsafe_allow_html=True
                )
            elif "스나이퍼" in sc_mode:
                st.markdown(
                    """
| 분석/실행 시점 | 데이터 기준 | 권장 설정값 (20일 평균 거래대금) | 매매 행동 요령 & 주의사항 |
| :--- | :--- | :--- | :--- |
| **08:30 ~ 08:50**<br>(장 시작 전 분석) | **전일까지의 확정 20일 평균 거래대금** | **100억 ~ 200억 원 이상**<br>*(기본 100억 권장)* | **핵심**: 장 시작 전에는 당일 거래대금이 0원이므로, 반드시 **'20일 평균 거래대금'** 기준으로 필터링해야 정상 검색됩니다. 유동성이 풍부한 우량주 위주로 후보군을 확정합니다. |
| **09:00 ~ 09:05**<br>(시초가 체결 확인) | **시초가 갭 검증** | **체결 갭: 전일종가 대비 -1.5% ~ +1.5%** | 후보 종목의 시초가가 **+1.5%를 초과하여 갭상승하면 절대 매수 금지** (갭 메우기 급락 위험). 갭 범위 내 체결 시 시초가 매수 진입. |
| **09:05 ~ 15:20**<br>(장중 자동 청산) | **장중 시세 감시** | **익절 +1.2% / 손절 -2.0%** | 주문 체결 즉시 MTS/HTS에 자동 감시 매도(스탑로스)를 걸어두고, 장 마감까지 미도달 시 종가에 정리합니다. |
                    """,
                    unsafe_allow_html=True
                )
            elif "종가배팅" in sc_mode:
                st.markdown(
                    """
| 분석/실행 시점 | 데이터 기준 | 권장 설정값 (당일 최종 거래대금) | 매매 행동 요령 & 주의사항 |
| :--- | :--- | :--- | :--- |
| **15:15 ~ 15:25**<br>(마감 직전 탐색) | **당일 마감 누적 거래대금** | **200억 ~ 300억 원 이상** | 장 마감 무렵 당일 거래대금이 200억~300억 이상 터진 종목을 스크리닝하여 테마 1등주를 확정합니다. |
| **15:20 ~ 15:30**<br>(동시호가 종가 체결) | **종가 고가마감 확인** | **당일 주가 +3.0% 이상 & 윗꼬리 18% 이하** | 윗꼬리가 길지 않고 일봉상 고가 부근에서 마감할 때 15:20 동시호가에 매수합니다. |
| **익일 09:00 ~ 09:15**<br>(시초가 갭 청산) | **익일 시초가 및 장초반** | **+1.5% 갭상승 시 즉시 청산** | 오버나잇 후 익일 09:00 시초가에 갭상승(+1.5% 이상) 출발 시 즉시 전량 익절하고 현금화합니다. |
                    """,
                    unsafe_allow_html=True
                )
            elif "5% 급등" in sc_mode or "5% 돌파" in sc_mode:
                st.markdown(
                    """
| 분석/실행 시점 | 데이터 기준 | 권장 설정값 (당일 최종 거래대금) | 매매 행동 요령 & 주의사항 |
| :--- | :--- | :--- | :--- |
| **15:15 ~ 15:25**<br>(마감 직전 탐색) | **당일 마감 누적 거래대금** | **500억 원 이상**<br>*(중소형주는 최소 300억↑)* | 시장의 모든 유동성이 쏠린 최상위 대장주만 진입합니다. 500억 이상 집중된 신고가 돌파봉은 익일 강한 후속 매수세를 유발합니다. |
| **15:20 ~ 15:30**<br>(동시호가 종가 체결) | **종가 고가마감 확인** | **당일 상승률 +15%↑ & 윗꼬리 8% 이하** | 당일 최고가 부근(상한가 또는 준상한가)에서 탄탄하게 마감하는 종목을 15:20 동시호가에 종가 매수합니다. |
| **익일 ~ 2일차**<br>(스윙 자동 익절) | **장중 시세 감시** | **장중 +5.0% 도달 시 전량 자동 익절** | 매수 다음 날 장중 +5% 급등 시 즉시 익절합니다. 1일차에 +5%에 못 미치더라도 추세가 살아있으면 2일차까지 홀딩하며, 손절 기준(-4.0%) 이탈 시 칼손절합니다. |
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.caption("Point-in-Time 스코어링 엔진으로 전체 상장 유니버스를 일괄 스크리닝하고 테마 집중 위험을 감지합니다. (권장 거래대금: 최소 50억~100억 원 이상)")

    # 2) 스크리너 필터 파라미터 툴바
    if "당일 단타" in sc_mode:
        f_hdr_c1, f_hdr_c2 = st.columns([4.2, 1.3])
        with f_hdr_c1:
            st.markdown(
                f'<div style="display:flex; align-items:center; gap:8px; padding-top:4px;">'
                f'<span class="office-subheading" style="margin:0 !important;">스크리너 필터 설정</span>'
                f'<span style="background:#F1F5F9; color:#1E293B; font-weight:700; font-size:0.72rem; padding:2px 8px; border-radius:4px; border:1px solid #CBD5E1;">🕒 {now_str} {rec_time["slot_name"]}</span>'
                f'<span style="color:#64748B; font-size:0.75rem;">(권장: 거래대금 {rec_time["val"]}억↑ · 시초대비 +{rec_time["gain"][0]}%~+{rec_time["gain"][1]}% 자동 세팅됨)</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        with f_hdr_c2:
            if st.button("🔄 현재시간 권장값 재적용", key="btn_reapply_auto_time", use_container_width=True):
                st.session_state["sc_min_daytrade_val"] = rec_time["val"]
                st.session_state["sc_intraday_gain_range"] = rec_time["gain"]
                st.session_state["sc_min_daytrade_vol_ratio"] = rec_time["vol"]
                st.session_state["sc_trigger_time_sync"] = True
                st.toast(f"현재 시각({now_str}) 권장 설정이 자동 적용되었습니다.")
                st.rerun()
    else:
        st.markdown('<div class="office-subheading" style="margin-bottom:4px !important;">스크리너 필터 설정</div>', unsafe_allow_html=True)

    sc_c1, sc_c2, sc_c3, sc_c4, sc_c5 = st.columns([1.1, 1.3, 1.2, 1.3, 1.1])

    with sc_c1:
        target_market = st.selectbox(
            "시장 구분",
            ["전체", "KOSPI", "KOSDAQ"],
            index=0,
            key="sc_target_market"
        )
    with sc_c2:
        sc_scope = st.selectbox(
            "탐색 대상 범위",
            [
                "내 관심종목 138선 (초고속/추천)",
                "주도주/우량주 300선 (권장)",
                "전 상장사 전수조사 (약 2,700개)"
            ],
            index=0,
            key="sc_scope_sel"
        )

    if "당일 단타" in sc_mode:
        with sc_c3:
            val_options = [50, 100, 200, 300]
            if "sc_min_daytrade_val" not in st.session_state or st.session_state["sc_min_daytrade_val"] not in val_options:
                st.session_state["sc_min_daytrade_val"] = rec_time["val"]
            min_daytrade_val_b = st.selectbox(
                "당일 최소 거래대금",
                val_options,
                format_func=lambda x: f"{x}억 원 이상",
                key="sc_min_daytrade_val"
            )
        with sc_c4:
            if "sc_intraday_gain_range" not in st.session_state:
                st.session_state["sc_intraday_gain_range"] = rec_time["gain"]
            intraday_gain_range = st.slider(
                "시초가 대비 상승률 구간 (%)",
                2.0, 12.0,
                step=0.5,
                help="권장 설정: 골든타임(09:40~11:00) 3.0%~8.5% | 장초반(09:10~09:40) 3.0%~6.5% | 점심 4.5%~9.0%",
                key="sc_intraday_gain_range"
            )
        with sc_c5:
            if "sc_min_daytrade_vol_ratio" not in st.session_state:
                st.session_state["sc_min_daytrade_vol_ratio"] = rec_time["vol"]
            min_daytrade_vol_ratio = st.slider(
                "전일대비 거래량 비율",
                0.4, 1.5,
                step=0.1,
                format="%.1f배 이상",
                help="권장 설정: 최소 0.6배(60%) 이상 | 거래량 폭발 대장주 1.0배 이상",
                key="sc_min_daytrade_vol_ratio"
            )
    elif "스나이퍼" in sc_mode:
        with sc_c3:
            min_val_krw_b = st.selectbox("최소 20일 평균 거래대금", [50, 100, 200, 300], index=1, format_func=lambda x: f"{x}억 원 이상", key="sc_min_val_krw")
        with sc_c4:
            max_disparity_val = st.slider("20일선 최대 이격도 (%)", 101.0, 105.0, 103.5, step=0.5, key="sc_max_disparity")
        with sc_c5:
            ignore_market_filter = st.checkbox("시장 하락장 무시", value=False, key="sc_ignore_mkt_filter")
    elif "종가배팅" in sc_mode:
        with sc_c3:
            min_today_val_b = st.selectbox("당일 최소 거래대금", [200, 300, 500], index=0, format_func=lambda x: f"{x}억 원 이상", key="sc_cb_min_today_val")
        with sc_c4:
            min_day_ret_val = st.slider("당일 최소 주가 상승률 (%)", 2.0, 5.0, 3.0, step=0.5, key="sc_cb_min_day_ret")
        with sc_c5:
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("15:20~15:30 권장")
    elif "5% 급등" in sc_mode or "5% 돌파" in sc_mode:
        with sc_c3:
            min_surge_val_b = st.selectbox("당일 최소 거래대금", [200, 300, 500, 800], index=1, format_func=lambda x: f"{x}억 원 이상", key="sc_surge_min_val")
        with sc_c4:
            min_day_surge_pct = st.slider("당일 최소 주가 상승률 (%)", 8.0, 25.0, 10.0, step=1.0, key="sc_surge_day_pct")
        with sc_c5:
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("마감 직전 15:20 진입 -> TP +5%")
    else:
        with sc_c3:
            min_screener_score = st.slider("최소 스코어 기준", 60.0, 90.0, 70.0, step=5.0, key="sc_min_score")
        with sc_c4:
            min_vol_surge = st.slider("거래량 급증 비율 (20일선 대비)", 1.0, 3.0, 1.5, step=0.2, key="sc_min_vol_surge")
        with sc_c5:
            require_ma_align = st.checkbox("이평 정배열/골든 필수", value=True, key="sc_require_ma")

    if st.button("후보 스크리닝 실행", type="primary", use_container_width=True):
        with st.spinner("상장 유니버스 데이터를 로드하고 전략별 필터링 중입니다..."):
            mkt_param = None if target_market == "전체" else target_market
            df_univ = get_universe(market=mkt_param, active_only=True)

            if df_univ.empty:
                st.warning("유니버스 데이터가 없습니다. 사이드바의 'KRX 유니버스 최신 동기화'를 먼저 실행해주세요.")
            else:
                if "관심종목" in sc_scope:
                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT w.code, w.name, w.market, COALESCE(u.sector, '기타') as sector FROM watchlist w LEFT JOIN universe u ON w.code = u.code")
                        wl_recs = cursor.fetchall()
                    sample_pool = pd.DataFrame([dict(r) for r in wl_recs]) if wl_recs else df_univ.head(100)
                elif "300선" in sc_scope:
                    top_lead_codes = []
                    try:
                        import FinanceDataReader as fdr
                        df_krx_lead = fdr.StockListing("KRX")
                        if not df_krx_lead.empty and "Amount" in df_krx_lead.columns:
                            if target_market in ("KOSPI", "KOSDAQ"):
                                df_krx_lead = df_krx_lead[df_krx_lead["Market"].str.upper() == target_market]
                            top_lead_codes = df_krx_lead.sort_values("Amount", ascending=False)["Code"].astype(str).str.zfill(6).tolist()[:350]
                    except Exception:
                        top_lead_codes = []

                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT code FROM watchlist")
                        wl_codes = [r["code"] for r in cursor.fetchall()]

                    # Combine true top trading value leaders + watchlist
                    major_lead_codes = list(dict.fromkeys(top_lead_codes + wl_codes))
                    df_top = df_univ[df_univ["code"].isin(major_lead_codes)]
                    df_rest = df_univ[~df_univ["code"].isin(major_lead_codes)]
                    sample_pool = pd.concat([df_top, df_rest]).head(300).reset_index(drop=True)
                else:
                    sample_pool = df_univ

                screened_results = []
                bm_ohlcv = get_benchmark_ohlcv() if ("스나이퍼" in sc_mode or "당일 단타" in sc_mode) else None
                prog_bar = st.progress(0, text=f"총 {len(sample_pool)}개 종목 시세 로드 및 전략 필터링 중...")
                total_pool_cnt = len(sample_pool)

                for idx, (_, row) in enumerate(sample_pool.iterrows()):
                    if idx % 25 == 0 or idx == total_pool_cnt - 1:
                        prog_bar.progress((idx + 1) / total_pool_cnt, text=f"전수 탐색 중 ({idx+1}/{total_pool_cnt}): {row['name']}")
                    code = row["code"]
                    name = row["name"]
                    sector = row.get("sector", "기타")
                    market = row["market"]

                    try:
                        df_stock = fetch_ohlcv(code)
                        if df_stock.empty or len(df_stock) < 30:
                            continue

                        close_p = float(df_stock["Close"].iloc[-1])
                        prev_p = float(df_stock["Close"].iloc[-2]) if len(df_stock) >= 2 else close_p
                        chg_pct = ((close_p - prev_p) / prev_p) * 100.0 if prev_p > 0 else 0.0
                        vol_p = int(df_stock["Volume"].iloc[-1])

                        # Detect intraday session (before 15:30 on trading day)
                        now_dt = datetime.now()
                        last_bar_dt = df_stock.index[-1]
                        is_today_intraday = (last_bar_dt.strftime("%Y-%m-%d") == now_dt.strftime("%Y-%m-%d") and now_dt.hour < 15)

                        # For Sniper (09:00 market open entry):
                        # Always evaluate on the completed daily bar (Day T-1) so volume & 20MA pullback are exact
                        df_snp_eval = df_stock.iloc[:-1] if is_today_intraday and len(df_stock) >= 31 else df_stock

                        if "당일 단타" in sc_mode:
                            bm_for_eval = bm_ohlcv
                            dtrade = evaluate_intraday_daytrade_candidate(
                                df_stock,
                                df_benchmark=bm_for_eval,
                                min_today_val_krw=min_daytrade_val_b * 100_000_000,
                                min_intraday_gain=intraday_gain_range[0],
                                max_intraday_gain=intraday_gain_range[1]
                            )
                            if not dtrade:
                                continue

                            m_dt = dtrade["metrics"]
                            if m_dt.get("vol_ratio_vs_prev", 0) < min_daytrade_vol_ratio:
                                continue

                            today_val_b_val = m_dt.get('today_trading_val_억', 0)
                            gain_from_open_val = m_dt.get('gain_from_open', 0)
                            supp_ratio_val = m_dt.get('support_ratio', 0)
                            vol_ratio_prev_val = m_dt.get('vol_ratio_vs_prev', 1.0)
                            room_limit_val = m_dt.get('room_to_limit_pct', 0)

                            reason_summary = f"대금 {today_val_b_val:,.0f}억 유입 | 시초대비 {gain_from_open_val:+.1f}% 돌파 | 양봉지지 {supp_ratio_val:.0f}%"
                            reason_core = "장중 대규모 수급(거래대금)이 유입되며 시초가를 강력하게 상향 돌파하였고, 윗꼬리가 짧은 견고한 양봉 지지력을 유지하여 당일 장중 +5% 추가 슈팅 확률이 매우 높은 당일단타 주도주입니다."
                            reason_criteria = [
                                f"당일 거래대금 {today_val_b_val:,.0f}억 원 유입 (최소 기준 {min_daytrade_val_b}억 원 충족)",
                                f"시초가 대비 {gain_from_open_val:+.1f}% 돌파 (유효 돌파 구간 +{intraday_gain_range[0]}% ~ +{intraday_gain_range[1]}% 충족)",
                                f"장중 전일 거래량 대비 {vol_ratio_prev_val*100:.0f}% 돌파 (수급 집중 기준 {min_daytrade_vol_ratio*100:.0f}% 충족)",
                                f"당일 진폭 중 고점 지지율 {supp_ratio_val:.1f}% (윗꼬리 짧은 탄탄한 양봉 지지, 65% 이상 충족)",
                                f"상한가(+30%)까지 잔여 상승폭 +{room_limit_val:.1f}% 확보 (목표 +5% 익절 공간 여유)"
                            ]

                            screened_results.append({
                                "code": code,
                                "name": name,
                                "market": market,
                                "sector": sector,
                                "close": close_p,
                                "change_pct": chg_pct,
                                "volume": vol_p,
                                "timing_label": "장중 실시간 (09:10~14:30)",
                                "strategy_mode": "DAY_TRADE_5PCT",
                                "strategy_tag": "당일단타 (당일 +5%)",
                                "score": dtrade["score"],
                                "label": dtrade["label"],
                                "tp_pct": dtrade["tp_pct"],
                                "sl_pct": dtrade["sl_pct"],
                                "vol_ratio": vol_ratio_prev_val,
                                "rsi": 65.0,
                                "ma_status": "장중 수급폭발 돌파",
                                "rule_note": dtrade["rule_note"],
                                "reason_summary": reason_summary,
                                "reason_core": reason_core,
                                "reason_criteria": reason_criteria,
                                "breakdown": {
                                    "rsi_val": 65.0,
                                    "ma_status": f"시초대비 {gain_from_open_val:+.1f}% 돌파봉",
                                    "vol_ratio": vol_ratio_prev_val,
                                    "bb_pct": supp_ratio_val / 100.0
                                }
                            })
                        elif "스나이퍼" in sc_mode:
                            bm_for_eval = None if ignore_market_filter else bm_ohlcv
                            snp = evaluate_sniper_candidate(df_snp_eval, df_benchmark=bm_for_eval, min_daily_val_krw=min_val_krw_b * 100_000_000)
                            if not snp:
                                continue
                            if snp["metrics"]["disparity"] > max_disparity_val:
                                continue

                            m_snp = snp["metrics"]
                            disp_val = m_snp.get("disparity", 100.0)
                            avg_val_b = m_snp.get("avg_trading_val_억", 0)
                            rsi_val = m_snp.get("rsi", 50.0)
                            vol_r_val = m_snp.get("vol_ratio", 1.0)

                            reason_summary = f"20일선 이격도 {disp_val:.1f}% 눌림목 | 20일평균 대금 {avg_val_b:,.0f}억 | RSI {rsi_val:.1f}"
                            reason_core = "KODEX 200 지수 상승장 확인 후, 일평균 거래대금이 풍부한 우량주가 20일 이동평균선 눌림목에서 첫 반등을 시작하여 시초가 갭 기준 부합 시 승률 80%를 타겟하는 고확신 스나이퍼 종목입니다."
                            reason_criteria = [
                                "KODEX 200 지수 20일 이동평균선 상회 (지수 하락장 시스템 위험 사전 차단 필터 통과)" if not ignore_market_filter else "시장 필터 미적용 (개별 종목 기술적 지표 단독 검증)",
                                f"20일 일평균 거래대금 {avg_val_b:,.0f}억 원 (기준 {min_val_krw_b}억 원 이상 풍부한 유동성)",
                                f"20일선 이격도 {disp_val:.1f}% (설정 기준 98.0% ~ {max_disparity_val:.1f}% 내 20일선 첫 지지 반등)",
                                "5일선 > 20일선 상단 위치 (단기 상승 모멘텀 유지)",
                                f"RSI(14) {rsi_val:.1f} (과열 없는 건전한 에너지 응축 구간, 45~68 충족)",
                                f"20일 평균 대비 거래량 {vol_r_val:.2f}배 반등 유입 (1.2배 이상 기준 충족)"
                            ]

                            screened_results.append({
                                "code": code,
                                "name": name,
                                "market": market,
                                "sector": sector,
                                "close": close_p,
                                "change_pct": chg_pct,
                                "volume": vol_p,
                                "timing_label": "장초 (09:00 시초가)",
                                "strategy_mode": "SNIPER",
                                "strategy_tag": "스나이퍼 (오전 9시)",
                                "score": snp["score"],
                                "label": snp["label"],
                                "tp_pct": snp["tp_pct"],
                                "sl_pct": snp["sl_pct"],
                                "vol_ratio": vol_r_val,
                                "rsi": rsi_val,
                                "ma_status": m_snp.get("ma_status", "20일선 지지"),
                                "rule_note": "익일 시초가 갭 -1.5%~+1.5% 이내 시에만 체결 (+1.5% 초과 갭상승 시 진입금지)",
                                "reason_summary": reason_summary,
                                "reason_core": reason_core,
                                "reason_criteria": reason_criteria,
                                "breakdown": {
                                    "rsi_val": rsi_val,
                                    "ma_status": "20일선 지지 첫반등",
                                    "vol_ratio": vol_r_val,
                                    "bb_pct": 0.5
                                }
                            })
                        elif "종가배팅" in sc_mode:
                            cbet = evaluate_closing_bet_candidate(df_stock, min_today_val_krw=min_today_val_b * 100_000_000)
                            if not cbet:
                                continue
                            if cbet["metrics"]["day_return"] < min_day_ret_val:
                                continue

                            m_cb = cbet["metrics"]
                            cb_val_b = m_cb.get("today_trading_val_억", 0)
                            cb_ret = m_cb.get("day_return", 0)
                            cb_hc = m_cb.get("high_close_ratio", 0)

                            reason_summary = f"당일대금 {cb_val_b:,.0f}억 폭발 | 당일 +{cb_ret:.1f}% | 고가마감 {cb_hc:.1f}%"
                            reason_core = "장마감 시점(15:20) 대규모 주도 거래대금과 함께 종가를 최고가 부근으로 마감하여, 익일 09:00 시초가 갭상승 및 장초반 슈팅(+1.5% 이상) 청산 확률이 극대화된 종가배팅 후보입니다."
                            reason_criteria = [
                                f"당일 거래대금 {cb_val_b:,.0f}억 원 폭발 (기준 {min_today_val_b}억 원 이상 시장 주도 수급)",
                                f"당일 주가 +{cb_ret:.1f}% 장대양봉 마감 (기준 +{min_day_ret_val:.1f}% 이상 충족)",
                                f"당일 고점 대비 종가 지지율 {cb_hc:.1f}% (기준 82% 이상, 장마감까지 차익실현 없는 탄탄한 종가)",
                                "5일선 및 20일선 상단 돌파 (단기 이평선 정배열 확인)"
                            ]

                            screened_results.append({
                                "code": code,
                                "name": name,
                                "market": market,
                                "sector": sector,
                                "close": close_p,
                                "change_pct": chg_pct,
                                "volume": vol_p,
                                "timing_label": "마감 직전 (15:20 종가)",
                                "strategy_mode": "CLOSING_BET",
                                "strategy_tag": "종가배팅 (마감 직전)",
                                "score": cbet["score"],
                                "label": cbet["label"],
                                "tp_pct": cbet["tp_pct"],
                                "sl_pct": cbet["sl_pct"],
                                "vol_ratio": 2.0,
                                "rsi": 65.0,
                                "ma_status": "5·20일선 상단 고가마감",
                                "rule_note": "15:20 종가 진입 -> 익일 09:00 시초가 갭익절(+1.5%↑) 또는 장초반 +1.5% 슈팅 청산",
                                "reason_summary": reason_summary,
                                "reason_core": reason_core,
                                "reason_criteria": reason_criteria,
                                "breakdown": {
                                    "rsi_val": 65.0,
                                    "ma_status": "주도주 고가마감",
                                    "vol_ratio": 2.5,
                                    "bb_pct": 0.8
                                }
                            })
                        elif "5% 급등" in sc_mode or "5% 돌파" in sc_mode:
                            s5 = evaluate_5pct_surge_candidate(
                                df_stock,
                                min_today_val_krw=min_surge_val_b * 100_000_000,
                                min_day_return_pct=min_day_surge_pct
                            )
                            if not s5 and is_today_intraday and len(df_stock) >= 31:
                                s5 = evaluate_5pct_surge_candidate(
                                    df_stock.iloc[:-1],
                                    min_today_val_krw=min_surge_val_b * 100_000_000,
                                    min_day_return_pct=min_day_surge_pct
                                )
                            if not s5:
                                continue

                            m_s5 = s5["metrics"]
                            s5_val_b = m_s5.get("today_trading_val_억", 0)
                            s5_ret = m_s5.get("day_return", 0)
                            s5_hc = m_s5.get("high_close_ratio", 0)

                            reason_summary = f"초대형 대금 {s5_val_b:,.0f}억 | 당일 +{s5_ret:.1f}% 급등 | 종가마감 {s5_hc:.1f}%"
                            reason_core = "시장의 주도 자금이 집중된 초대형 거래대금과 함께 장대양봉(상한가/준상한가 마루보즈)으로 최고가권 마감하여, 1~2일 내 장중 +5.0% 연속 돌파 슈팅 가능성이 검증된 초강세 대장주입니다."
                            reason_criteria = [
                                f"당일 거래대금 {s5_val_b:,.0f}억 원 집중 (기준 {min_surge_val_b}억 원 이상 최상위 주도주 수급)",
                                f"당일 주가 +{s5_ret:.1f}% 급등 마감 (기준 +{min_day_surge_pct:.1f}% 이상 강력한 모멘텀)",
                                f"당일 고점 대비 종가 유지율 {s5_hc:.1f}% (기준 88% 이상, 윗꼬리 극소화 마루보즈봉)",
                                "매매 원칙: 15:20 종가 매수 -> 1~2일 내 장중 +5.0% 터치 시 즉시 전량 자동 익절"
                            ]

                            screened_results.append({
                                "code": code,
                                "name": name,
                                "market": market,
                                "sector": sector,
                                "close": close_p,
                                "change_pct": chg_pct,
                                "volume": vol_p,
                                "timing_label": "마감 직전 (15:20 종가)",
                                "strategy_mode": "SURGE_5PCT",
                                "strategy_tag": "5%타겟 (마감 직전)",
                                "score": s5["score"],
                                "label": s5["label"],
                                "tp_pct": s5["tp_pct"],
                                "sl_pct": s5["sl_pct"],
                                "vol_ratio": m_s5.get("vol_ratio", 3.0),
                                "rsi": 70.0,
                                "ma_status": "신고가/상한가 돌파",
                                "rule_note": "전일 15:20 종가 매수 -> 1~2일 내 장중 +5.0% 돌파 시 즉시 전량 익절 (손절 -4.0%)",
                                "reason_summary": reason_summary,
                                "reason_core": reason_core,
                                "reason_criteria": reason_criteria,
                                "breakdown": {
                                    "rsi_val": 70.0,
                                    "ma_status": "주도주 상한가/급등돌파",
                                    "vol_ratio": m_s5.get("vol_ratio", 3.0),
                                    "bb_pct": 0.95
                                }
                            })
                        else:
                            score_res = evaluate_stock_latest(df_stock)
                            if not score_res:
                                continue
                            score = score_res["score"]
                            vol_ratio = score_res["breakdown"]["vol_ratio"]
                            ma_status = score_res["breakdown"]["ma_status"]

                            if score < min_screener_score:
                                continue
                            if vol_ratio < min_vol_surge:
                                continue
                            if require_ma_align and ("정배열" not in ma_status and "골든" not in ma_status):
                                continue

                            reason_summary = f"종합 스코어 {score:.1f}점 | 거래량 {vol_ratio:.1f}배 급증 | {ma_status}"
                            reason_core = "이동평균선 추세, 거래량 급증, RSI 모멘텀, 볼린저 밴드 지표의 다중 팩터 가중치 종합 퀀트 분석 결과 상위권에 랭크된 기술적 우량 후보입니다."
                            reason_criteria = [
                                f"종합 퀀트 스코어 {score:.1f}점 달성 (기준 {min_screener_score}점 이상 {score_res['label']} 등급)",
                                f"20일 평균 대비 거래량 {vol_ratio:.1f}배 급증 (기준 {min_vol_surge}배 이상 수급 유입)",
                                f"이평선 배열: {ma_status} (추세 상승 국면 확인)",
                                f"RSI(14) {score_res['breakdown']['rsi_val']:.1f} (과열 없는 안정적 상승 탄력)"
                            ]

                            screened_results.append({
                                "code": code,
                                "name": name,
                                "market": market,
                                "sector": sector,
                                "close": close_p,
                                "change_pct": chg_pct,
                                "volume": vol_p,
                                "timing_label": "장초 (09:00 시초가)",
                                "strategy_mode": "NORMAL",
                                "strategy_tag": "일반퀀트",
                                "score": score,
                                "label": score_res["label"],
                                "tp_pct": score_res["tp_pct"],
                                "sl_pct": score_res["sl_pct"],
                                "vol_ratio": vol_ratio,
                                "rsi": score_res["breakdown"]["rsi_val"],
                                "ma_status": ma_status,
                                "rule_note": "익일 시초가 진입 -> 점수 연동 동적 익절선(+1.5%~+3.0%)",
                                "reason_summary": reason_summary,
                                "reason_core": reason_core,
                                "reason_criteria": reason_criteria,
                                "breakdown": score_res["breakdown"]
                            })
                    except Exception:
                        continue

                prog_bar.empty()

                if not screened_results:
                    st.session_state["screened_results_df"] = pd.DataFrame()
                    with st.container(border=True):
                        st.markdown(
                            f"<div style='font-size:0.92rem; font-weight:700; color:#1e293b; margin-bottom:6px;'>"
                            f"조건 만족 후보 종목이 없습니다 (0건)"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f"<div style='font-size:0.83rem; line-height:1.5; color:#475569; margin-bottom:8px;'>"
                            f"선택하신 <strong>{sc_mode}</strong>의 필터 조건이 엄격하여, 현재 탐색 대상 범위(<strong>{sc_scope}</strong>, 시장: <strong>{target_market}</strong>) 내에 오늘 기준 충족 종목이 없습니다.<br>"
                            f"시스템 오류가 아니며, 시장의 자금 쏠림이나 탐색 범위에 따른 정상적인 퀀트 필터링 결과입니다."
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f"<div style='font-size:0.80rem; line-height:1.5; color:#334155; background-color:#f8fafc; padding:8px 12px; border-radius:4px; border-left:3px solid #64748b;'>"
                            f"<strong>해결 방법 가이드</strong>:<br>"
                            f"• <strong>탐색 대상 범위 변경</strong>: '내 관심종목 138선'에 오늘 급등주가 없다면, 탐색 대상을 <strong>'주도주/우량주 300선 (권장)'</strong> 또는 <strong>'전 상장사 전수조사'</strong>로 변경 후 재실행해보세요.<br>"
                            f"• <strong>스크리닝 시장</strong>: 코스닥 테마 급등주를 함께 탐색하려면 <strong>'스크리닝 시장: 전체'</strong>로 설정하세요.<br>"
                            f"• <strong>필터 기준 완화</strong>: 당일 최소 거래대금을 <strong>200억~300억 원</strong>, 최소 상승률을 <strong>10%</strong> 수준으로 1단계 완화해보세요."
                            f"</div>",
                            unsafe_allow_html=True
                        )
                else:
                    df_screened = pd.DataFrame(screened_results).sort_values("score", ascending=False).reset_index(drop=True)
                    st.session_state["screened_results_df"] = df_screened
                    st.session_state["screened_mode_label"] = sc_mode

    # Display Persistent Screened Results like Watchlist
    if "screened_results_df" in st.session_state and st.session_state["screened_results_df"] is not None and not st.session_state["screened_results_df"].empty:
        df_screened = st.session_state["screened_results_df"]
        mode_label = st.session_state.get("screened_mode_label", "선택 전략")

        st.markdown("---")

        # Theme Clustering Risk Check
        top_n = df_screened.head(10)
        sector_counts = top_n["sector"].value_counts()
        if not sector_counts.empty:
            max_sector = sector_counts.index[0]
            max_count = sector_counts.iloc[0]
            max_ratio = max_count / len(top_n)

            if max_ratio >= 0.40:
                st.warning(
                    f"**[테마 몰림 경고]** 상위 10개 후보 중 **{max_sector}** 업종이 {max_count}건({max_ratio*100:.0f}%)으로 고도로 편중되어 있습니다. "
                    f"동일 테마/업종 동반 급락 리스크에 주의하여 분산 투자를 준수하세요!"
                )

        # Action Bar: Total Count + Batch Add All to Watchlist
        hdr_c1, hdr_c2 = st.columns([3, 1.8])
        with hdr_c1:
            st.markdown(f"### {mode_label} 스크리닝 결과 (**{len(df_screened):,}개 종목 발굴**)")
            st.caption("표에서 종목(행)을 클릭하면 하단에 상세 **선정 이유(발굴 근거 및 수치 팩트체크)**와 매매 실행 가이드가 즉시 표시됩니다.")
        with hdr_c2:
            if st.button("발굴 후보 전체 관심종목 일괄 추가", type="primary", use_container_width=True):
                codes_to_add = df_screened["code"].tolist()
                added_cnt = batch_add_to_watchlist(codes_to_add)
                st.cache_data.clear()
                st.toast(f"총 {added_cnt}개 종목이 관심종목에 등록되었습니다!")
                st.rerun()

        # Filter & View Toolbar
        sc_t1, sc_t2, sc_t3, sc_t4, sc_t5 = st.columns([1.8, 1.6, 1.5, 1.1, 1.6])
        with sc_t1:
            sc_timing_filter = st.selectbox(
                "매수 타이밍 분류 필터",
                [
                    "전체 타이밍",
                    "장중 실시간 (09:10~14:30) 당일단타 후보",
                    "장초 (09:00 시초가) 스나이퍼 후보",
                    "마감 직전 (15:20 종가) 5%급등 후보"
                ],
                index=0,
                key="sc_timing_sel"
            )
        with sc_t2:
            sc_sort = st.selectbox(
                "정렬 기준",
                ["익일 스코어 높은 순", "전일대비 등락률 높은 순", "목표 익절가 높은 순", "종목명 가나다순"],
                index=0,
                key="sc_sort_sel"
            )
        with sc_t3:
            avail_sectors = ["전체"] + sorted(list(set(df_screened["sector"].dropna().unique())))
            sc_sector_filter = st.selectbox("업종/테마 필터", avail_sectors, index=0, key="sc_sector_sel")
        with sc_t4:
            sc_display_rows = st.selectbox("표시 높이", [10, 15, 20, 30], index=0, format_func=lambda x: f"{x}줄 보기", key="sc_display_rows_sel")
        with sc_t5:
            sc_view_mode = st.radio("화면 모드", ["고밀도 테이블 뷰 (권장)", "카드 페이징 뷰"], horizontal=True, key="sc_view_radio")

        # Apply Sorting & Sector Filter & Timing Filter
        df_sc_disp = df_screened.copy()
        if "reason_summary" not in df_sc_disp.columns:
            df_sc_disp["reason_summary"] = "전략 조건 충족"
        if "reason_core" not in df_sc_disp.columns:
            df_sc_disp["reason_core"] = "설정된 퀀트 전략 기준을 모두 통과하여 선정되었습니다."
        if "reason_criteria" not in df_sc_disp.columns:
            df_sc_disp["reason_criteria"] = [[] for _ in range(len(df_sc_disp))]

        if "장중 실시간" in sc_timing_filter:
            df_sc_disp = df_sc_disp[df_sc_disp["timing_label"].str.contains("장중|09:10", na=False)]
        elif "장초 (09:00" in sc_timing_filter:
            df_sc_disp = df_sc_disp[df_sc_disp["timing_label"].str.contains("09:00", na=False)]
        elif "마감 직전 (15:20" in sc_timing_filter:
            df_sc_disp = df_sc_disp[df_sc_disp["timing_label"].str.contains("15:20", na=False)]

        if sc_sector_filter != "전체":
            df_sc_disp = df_sc_disp[df_sc_disp["sector"] == sc_sector_filter]

        if sc_sort == "익일 스코어 높은 순":
            df_sc_disp = df_sc_disp.sort_values("score", ascending=False)
        elif sc_sort == "전일대비 등락률 높은 순":
            df_sc_disp = df_sc_disp.sort_values("change_pct", ascending=False)
        elif sc_sort == "목표 익절가 높은 순":
            df_sc_disp = df_sc_disp.sort_values("tp_pct", ascending=False)
        elif sc_sort == "종목명 가나다순":
            df_sc_disp = df_sc_disp.sort_values("name", ascending=True)

        df_sc_disp = df_sc_disp.reset_index(drop=True)
        total_sc_items = len(df_sc_disp)

        # Screener Display Table Height (10 rows fixed height with smooth internal scrolling)
        sc_row_limit = int(sc_display_rows)
        sc_table_height = min(38 + sc_row_limit * 35, 38 + total_sc_items * 35)

        # -------------------------------------------------
        # Screener View A: High-Density Interactive Data Table
        # -------------------------------------------------
        if "테이블" in sc_view_mode:
            st.caption(f"스크리닝 결과: **총 {total_sc_items:,}개** 발굴됨 (10줄 높이 고정 · 표 내부 마우스 스크롤로 전체 탐색 및 다중 체크박스 선택 지원)")
            disp_table_sc = df_sc_disp[[
                "code", "name", "timing_label", "strategy_tag", "reason_summary", "market", "sector", "close", "change_pct",
                "score", "label", "tp_pct", "sl_pct", "vol_ratio", "rsi", "ma_status"
            ]].copy()

            grid_sc = st.dataframe(
                disp_table_sc,
                column_config={
                    "code": st.column_config.TextColumn("코드", width="small"),
                    "name": st.column_config.TextColumn("종목명", width="medium"),
                    "timing_label": st.column_config.TextColumn("매수 타이밍", width="medium"),
                    "strategy_tag": st.column_config.TextColumn("전략 태그", width="medium"),
                    "reason_summary": st.column_config.TextColumn("선정 핵심 근거", width="large"),
                    "market": st.column_config.TextColumn("시장", width="small"),
                    "sector": st.column_config.TextColumn("업종", width="small"),
                    "close": st.column_config.NumberColumn("현재가", format="%,d원"),
                    "change_pct": st.column_config.NumberColumn("전일대비", format="%+.2f%%"),
                    "score": st.column_config.ProgressColumn("익일 스코어", min_value=0, max_value=100, format="%.1f점"),
                    "label": st.column_config.TextColumn("전망"),
                    "tp_pct": st.column_config.NumberColumn("목표익절", format="+%.1f%%"),
                    "sl_pct": st.column_config.NumberColumn("손절기준", format="-%.1f%%"),
                    "vol_ratio": st.column_config.NumberColumn("거래량비", format="%.1f배"),
                    "rsi": st.column_config.NumberColumn("RSI(14)", format="%.1f"),
                    "ma_status": st.column_config.TextColumn("이평배열"),
                },
                hide_index=True,
                use_container_width=True,
                selection_mode="multi-row",
                on_select="rerun",
                height=sc_table_height,
                key="screener_result_grid"
            )

            # Row Selection Inspector Panel
            sel_sc_rows = grid_sc.selection.rows if hasattr(grid_sc, "selection") else []
            if sel_sc_rows and len(sel_sc_rows) > 0:
                if len(sel_sc_rows) > 1:
                    sel_sc_df = df_sc_disp.iloc[sel_sc_rows]
                    st.markdown(f"### 선택 후보 종목 일괄 관리 (총 **{len(sel_sc_df)}**개 종목 선택됨)")

                    # Batch Add to Watchlist Container
                    avail_grps_sc = get_watchlist_groups()
                    with st.container(border=True):
                        b_c1, b_c2, b_c3 = st.columns([3, 3, 2])
                        with b_c1:
                            target_grp_sc = st.selectbox(
                                "관심종목 등록 대상 그룹",
                                ["기본그룹"] + [g for g in avail_grps_sc if g != "기본그룹"] + ["+ 새 그룹 직접입력..."],
                                key="batch_sc_grp_sel"
                            )
                        with b_c2:
                            custom_grp_sc = st.text_input(
                                "새 그룹명 입력",
                                "",
                                key="batch_sc_custom_grp",
                                disabled=(target_grp_sc != "+ 새 그룹 직접입력...")
                            )
                        with b_c3:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button(f"{len(sel_sc_df)}개 관심종목 일괄 추가", key="btn_batch_add_sc_wl", type="primary", use_container_width=True):
                                applied_grp = custom_grp_sc.strip() if target_grp_sc == "+ 새 그룹 직접입력..." and custom_grp_sc.strip() else target_grp_sc
                                codes_to_add = sel_sc_df["code"].tolist()
                                batch_add_to_watchlist(codes_to_add, group_name=applied_grp)
                                st.cache_data.clear()
                                st.toast(f"{len(codes_to_add)}개 종목이 '{applied_grp}' 그룹에 일괄 추가되었습니다!")
                                st.rerun()

                    # Summary Metrics for Selected Screener Rows
                    sm_c1, sm_c2, sm_c3, sm_c4 = st.columns(4)
                    sm_c1.metric("선택 종목 평균 스코어", f"{sel_sc_df['score'].mean():.1f}점")
                    sm_c2.metric("선택 종목 평균 등락률", f"{sel_sc_df['change_pct'].mean():+.2f}%")
                    sm_c3.metric("평균 목표 익절", f"+{sel_sc_df['tp_pct'].mean():.1f}%")
                    sm_c4.metric("평균 손절 기준", f"-{sel_sc_df['sl_pct'].mean():.1f}%")

                    st.markdown("---")
                    st.markdown("#### 선택 종목 중 개별 팩트체크 & 상세 진단")
                    target_sc_name = st.selectbox(
                        "상세 진단할 종목 선택",
                        [f"{r['name']} ({r['code']}) - {r.get('strategy_tag', '')}" for _, r in sel_sc_df.iterrows()],
                        key="sel_multi_inspect_sc"
                    )
                    sel_sc_code = target_sc_name.split("(")[-1].split(")")[0].strip()
                    sel_sc = sel_sc_df[sel_sc_df["code"] == sel_sc_code].iloc[0]
                else:
                    sel_sc = df_sc_disp.iloc[sel_sc_rows[0]]

                st.markdown(f"### 선택 후보 종목 정밀 진단: **{sel_sc['name']}** (`{sel_sc['code']}`)")

                # 1. Detailed Selection Reason & Fact Check Box
                with st.container(border=True):
                    r_hdr_c1, r_hdr_c2 = st.columns([3.2, 1.3])
                    with r_hdr_c1:
                        st.markdown(
                            f"<div style='font-size:0.92rem; font-weight:700; color:#1e293b; margin-bottom:4px;'>"
                            f"선정 근거 요약: {sel_sc.get('reason_summary', '조건 충족')}"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    with r_hdr_c2:
                        st.caption(f"전략: {sel_sc.get('strategy_tag', '')} | {sel_sc.get('timing_label', '')}")

                    st.markdown(
                        f"<div style='font-size:0.84rem; line-height:1.55; color:#334155; margin-bottom:10px; background-color:#f8fafc; padding:10px 14px; border-radius:4px; border-left:3px solid #3b82f6;'>"
                        f"<strong>[핵심 발굴 근거]</strong> {sel_sc.get('reason_core', '설정된 퀀트 전략 기준을 모두 통과하여 선정되었습니다.')}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    rc_list = sel_sc.get("reason_criteria", [])
                    if rc_list:
                        st.markdown("<div style='font-size:0.80rem; font-weight:700; color:#475569; margin-bottom:4px;'>전략 통과 세부 팩트체크:</div>", unsafe_allow_html=True)
                        r_cols = st.columns(2)
                        for idx, crit in enumerate(rc_list):
                            with r_cols[idx % 2]:
                                st.markdown(f"<div style='font-size:0.79rem; line-height:1.45; color:#334155;'>• {crit}</div>", unsafe_allow_html=True)

                # 2. Timing & Target Alert Box
                timing_val = sel_sc.get("timing_label", "")
                target_tp_val = sel_sc["close"] * (1.0 + sel_sc["tp_pct"] / 100.0)
                target_sl_val = sel_sc["close"] * (1.0 - sel_sc["sl_pct"] / 100.0)

                if "장중" in timing_val or "09:10" in timing_val:
                    st.success(
                        f"**[실시간 당일단타 5% 돌파 모드: 현재 분석 시점 진입 기준]**\n\n"
                        f"• **진입 기준가 (현재 체결가)**: **{sel_sc['close']:,.0f}원** (분석 요청 시점 기준)\n\n"
                        f"• **당일 장중 목표 익절가**: **{target_tp_val:,.0f}원 (+{sel_sc['tp_pct']:.1f}%)** 도달 즉시 전량 자동 익절\n\n"
                        f"• **당일 원칙 손절가**: **{target_sl_val:,.0f}원 (-{sel_sc['sl_pct']:.1f}%)** 이탈 시 즉시 칼손절\n\n"
                        f"• **당일 마감 청산**: 15:15까지 목표가 미도달 시 종가 전량 시장가 청산 (**오버나잇 리스크 제로!**)"
                    )
                elif "09:00" in timing_val:
                    st.success(f"**[매수 타이밍: 장초 09:00 시초가 진입]** {sel_sc['name']}은 20일선 눌림목 반등 포착 종목입니다. 09:00 시초가 갭이 **-1.5% ~ +1.5%** 이내일 때만 체결 (+1.5% 초과 갭상승 시 뇌동매매 금지) **목표익절가 {target_tp_val:,.0f}원(+1.2%) / 손절가 {target_sl_val:,.0f}원(-2.0%)**")
                elif "15:20" in timing_val:
                    st.success(f"**[매수 타이밍: 마감 직전 15:20 종가 진입]** {sel_sc['name']}은 당일 거래대금 폭발 주도 대장주입니다. 15:20 동시호가 종가 매수 후 1~2일 내 **장중 +5.0% 도달 시 즉시 자동 익절** **목표익절가 {target_tp_val:,.0f}원(+5.0%) / 손절가 {target_sl_val:,.0f}원(-4.0%)**")

                # 3. Numeric Metrics & Actions
                insp_c1, insp_c2, insp_c3, insp_c4, insp_btn1, insp_btn2 = st.columns([2, 2, 2, 2, 2.5, 2])
                with insp_c1:
                    st.metric("익일 전망 스코어", f"{sel_sc['score']:.1f}점", delta=sel_sc['label'])
                with insp_c2:
                    st.metric("현재가 / 등락률", f"{sel_sc['close']:,.0f}원", delta=f"{sel_sc['change_pct']:+.2f}%")
                with insp_c3:
                    st.metric("목표 익절가", f"{target_tp_val:,.0f}원", delta=f"+{sel_sc['tp_pct']:.1f}%")
                with insp_c4:
                    st.metric("원칙 손절가", f"{target_sl_val:,.0f}원", delta=f"-{sel_sc['sl_pct']:.1f}%")

                with insp_btn1:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button(f"{sel_sc['name']} 7년 백테스트 실행", type="primary", key=f"bt_sc_btn_{sel_sc['code']}", use_container_width=True):
                        st.session_state["target_backtest_code"] = sel_sc["code"]
                        st.toast(f"탭 3(워크포워드 검증)로 이동하여 {sel_sc['name']} 백테스트를 확인하세요!")
                with insp_btn2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("관심종목 추가", key=f"add_sc_insp_{sel_sc['code']}", use_container_width=True):
                        batch_add_to_watchlist([sel_sc["code"]])
                        st.cache_data.clear()
                        st.toast(f"{sel_sc['name']} 관심종목에 추가 완료!")
                        st.rerun()

                # 4. Technical Indicator Breakdown Cards
                bk = sel_sc.get("breakdown", {})
                b1, b2, b3, b4 = st.columns(4)
                b1.info(f"**RSI(14)**: {bk.get('rsi_val', sel_sc.get('rsi', 0)):.1f}")
                b2.info(f"**이평배열**: {sel_sc.get('ma_status', '-')}")
                b3.info(f"**거래량 급증비**: {sel_sc.get('vol_ratio', 1.0):.2f}배")
                b4.info(f"**체결 원칙**: {sel_sc.get('rule_note', '-')}")

        # -------------------------------------------------
        # Screener View B: Paginated Cards View
        # -------------------------------------------------
        else:
            card_page_size = int(sc_display_rows)
            total_card_pages = max(1, (total_sc_items - 1) // card_page_size + 1)
            p_c1, p_c2 = st.columns([3.5, 1.2])
            with p_c1:
                st.caption(f"스크리닝 결과 카드 뷰 (페이지당 {card_page_size}개 표시)")
            with p_c2:
                sc_curr_page = st.number_input("카드 페이지", min_value=1, max_value=total_card_pages, value=1, step=1, key="sc_card_page_num_in")
            start_sc = (sc_curr_page - 1) * card_page_size
            end_sc = min(start_sc + card_page_size, total_sc_items)
            paged_sc_cards = df_sc_disp.iloc[start_sc:end_sc].reset_index(drop=True)
            card_cols = st.columns(3)
            for i, (_, srow) in enumerate(paged_sc_cards.iterrows()):
                with card_cols[i % 3]:
                    with st.container(border=True):
                        st.markdown(f"**{srow['name']}** (`{srow['code']}`)")
                        st.caption(f"{srow['market']} | {srow['sector']} | {srow['strategy_tag']}")
                        chg_c = "red" if srow["change_pct"] > 0 else ("blue" if srow["change_pct"] < 0 else "gray")
                        st.markdown(f"**{srow['close']:,.0f}원** <span style='color:{chg_c}; font-weight:600;'>{srow['change_pct']:+.2f}%</span>", unsafe_allow_html=True)
                        st.markdown(f"**스코어 {srow['score']:.1f}점** ({srow['label']})")
                        st.progress(min(1.0, max(0.0, srow["score"] / 100.0)))
                        st.caption(f"목표: +{srow['tp_pct']:.1f}% | 손절: -{srow['sl_pct']:.1f}%")
                        st.caption(f"{srow.get('rule_note', '')}")

                        with st.expander("선정 이유 및 팩트체크", expanded=False):
                            st.markdown(f"<div style='font-size:0.80rem; line-height:1.4; margin-bottom:6px;'><strong>핵심 근거</strong>: {srow.get('reason_core', '조건 충족')}</div>", unsafe_allow_html=True)
                            for crit in srow.get('reason_criteria', []):
                                st.markdown(f"<div style='font-size:0.77rem; line-height:1.35; color:#475569;'>• {crit}</div>", unsafe_allow_html=True)

                        c_b1, c_b2 = st.columns(2)
                        with c_b1:
                            if st.button("관심추가", key=f"card_add_sc_{srow['code']}_{i}", use_container_width=True):
                                batch_add_to_watchlist([srow['code']])
                                st.cache_data.clear()
                                st.toast(f"{srow['name']} 관심종목 추가 완료!")
                                st.rerun()
                        with c_b2:
                            if st.button("백테스트", key=f"card_bt_sc_{srow['code']}_{i}", use_container_width=True):
                                st.session_state["target_backtest_code"] = srow["code"]
                                st.toast(f"{srow['name']} 백테스트 설정 완료! 탭 3을 확인하세요.")

# =========================================================
# TAB 2: 관심종목 실시간 스코어보드 & 관리
# =========================================================
with tab2:
    st.markdown('<div class="office-heading">2. 관심종목 실시간 스코어보드</div>', unsafe_allow_html=True)
    

    # 1. Load Watchlist from DB & Render Summary KPIs Row (상단 KPI 카드)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                w.code, 
                w.name, 
                w.market, 
                COALESCE(NULLIF(u.sector, ''), NULLIF(w.notes, ''), '기타') AS raw_sector,
                COALESCE(u.industry, '') AS industry,
                w.notes,
                COALESCE(NULLIF(w.group_name, ''), '기본그룹') AS group_name
            FROM watchlist w 
            LEFT JOIN universe u ON w.code = u.code 
            ORDER BY w.added_at DESC
        """)
        watchlist_rows = cursor.fetchall()

    if watchlist_rows:
        tuple_codes = tuple(
            (r["code"], r["name"], r["market"], normalize_sector(r["raw_sector"], r["name"], r["industry"]), r["notes"], r["group_name"])
            for r in watchlist_rows
        )
        raw_wl_data = load_all_watchlist_metrics(tuple_codes)
        df_all = pd.DataFrame(raw_wl_data)

        total_cnt = len(df_all)
        bullish_cnt = len(df_all[df_all["score"] >= 70.0])
        neutral_cnt = len(df_all[(df_all["score"] >= 50.0) & (df_all["score"] < 70.0)])
        bearish_cnt = len(df_all[df_all["score"] < 50.0])
        avg_score = df_all["score"].mean() if not df_all.empty else 0.0

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("총 관심종목", f"{total_cnt:,}개")
        kpi2.metric("평균 익일 스코어", f"{avg_score:.1f}점 / 100")
        kpi3.metric(
            "강세 (70점↑)",
            f"{bullish_cnt}개",
            delta=f"비중 {(bullish_cnt/total_cnt)*100:.1f}%" if total_cnt else None,
            delta_color="off",
            help="전체 관심종목 중 강세 종목 비율 (점유율)"
        )
        kpi4.metric(
            "중립 (50~69점)",
            f"{neutral_cnt}개",
            delta=f"비중 {(neutral_cnt/total_cnt)*100:.1f}%" if total_cnt else None,
            delta_color="off",
            help="전체 관심종목 중 중립 종목 비율 (점유율)"
        )
        kpi5.metric(
            "약세 (50점↓)",
            f"{bearish_cnt}개",
            delta=f"비중 {(bearish_cnt/total_cnt)*100:.1f}%" if total_cnt else None,
            delta_color="off",
            help="전체 관심종목 중 약세 종목 비율 (점유율)"
        )
    else:
        df_all = pd.DataFrame()
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("총 관심종목", "0개")
        kpi2.metric("평균 익일 스코어", "0.0점")
        kpi3.metric("강세 (70점↑)", "0개")
        kpi4.metric("중립 (50~69점)", "0개")
        kpi5.metric("약세 (50점↓)", "0개")

    st.markdown("---")

    # 2. Search, Group & Batch Add Toolbar
    col_search, col_grp, col_batch_toggle = st.columns([2.6, 1.4, 1.2])
    avail_groups = get_watchlist_groups()
    with col_search:
        search_kw = st.text_input("개별 종목 검색 및 빠른 추가 (종목명 또는 6자리 코드)", "", placeholder="예: 삼성전자, 000660, 에코프로, 카카오...", key="wl_search_ticker_kw")
    with col_grp:
        quick_target_grp = st.selectbox("추가할 대상 그룹", avail_groups + ["+ 새 그룹 추가..."], key="quick_target_grp_sel")
        if quick_target_grp == "+ 새 그룹 추가...":
            new_quick_grp = st.text_input("새 그룹명 입력", "", placeholder="예: 바이오, 단타...", key="quick_new_grp_in")
            assigned_quick_grp = new_quick_grp.strip() if new_quick_grp.strip() else "기본그룹"
        else:
            assigned_quick_grp = quick_target_grp
    with col_batch_toggle:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        show_batch_panel = st.checkbox("대량 등록 / 일괄 관리", value=False, key="wl_batch_panel_chk")

    if search_kw:
        matched_stocks = search_ticker(search_kw, limit=6)
        if matched_stocks:
            cols = st.columns(min(len(matched_stocks), 4))
            for i, stock in enumerate(matched_stocks[:4]):
                with cols[i]:
                    code = stock["code"]
                    name = stock["name"]
                    market = stock["market"]
                    sector = stock.get("sector", "")
                    st.markdown(f"**{name}** (`{code}`)")
                    st.caption(f"{market} | {sector} | 📁 {assigned_quick_grp}")
                    if st.button(f"관심종목 추가", key=f"add_search_{code}"):
                        batch_add_to_watchlist([code], group_name=assigned_quick_grp)
                        st.cache_data.clear()
                        st.toast(f"{name} 종목이 '{assigned_quick_grp}' 그룹에 등록되었습니다!")
                        st.rerun()

    # Batch Add / Management Expander
    if show_batch_panel:
        with st.expander("종목 대량 등록 & 프리셋 관리 (수백 개 종목 일괄 처리)", expanded=True):
            b_c1, b_c2 = st.columns([3, 2])
            with b_c1:
                st.markdown("**1) 종목코드 텍스트 일괄 붙여넣기**")
                st.caption("쉼표(,), 띄어쓰기, 또는 줄바꿈(엔터)으로 구분된 6자리 종목코드들을 붙여넣으세요.")
                b_grp_c1, b_grp_c2 = st.columns([2, 2])
                with b_grp_c1:
                    batch_grp_sel = st.selectbox("일괄 등록 대상 그룹", avail_groups + ["+ 새 그룹 직접입력..."], key="batch_grp_sel")
                with b_grp_c2:
                    batch_custom_grp = st.text_input("새 그룹명", "", key="batch_custom_grp", disabled=(batch_grp_sel != "+ 새 그룹 직접입력..."))
                batch_final_grp = batch_custom_grp.strip() if batch_grp_sel == "+ 새 그룹 직접입력..." and batch_custom_grp.strip() else (batch_grp_sel if batch_grp_sel != "+ 새 그룹 직접입력..." else "기본그룹")

                batch_text = st.text_area("종목코드 입력창", "", placeholder="005930, 000660, 035420, 005380, 068270, 000270, 105560...", height=100)
                if st.button("종목 일괄 추가 실행", type="primary"):
                    if batch_text.strip():
                        import re
                        raw_codes = re.findall(r"\b\d{6}\b", batch_text)
                        if raw_codes:
                            added = batch_add_to_watchlist(raw_codes, group_name=batch_final_grp)
                            st.cache_data.clear()
                            st.success(f"총 {added}개 종목이 '{batch_final_grp}' 그룹에 일괄 등록되었습니다!")
                            st.rerun()
                        else:
                            st.error("유효한 6자리 숫자 종목코드를 찾을 수 없습니다.")

            with b_c2:
                st.markdown("**2) 시장 대표 우량주 프리셋 추가**")
                st.caption("클릭 한 번으로 시장 핵심 종목군을 즉시 관심종목으로 구성합니다.")
                if st.button("KOSPI 시총 상위 30종목 원클릭 추가", use_container_width=True):
                    kospi_top30 = [
                        "005930", "000660", "373220", "207940", "005380", "000270", "068270", "005490",
                        "105560", "055550", "035420", "028260", "012330", "032830", "003670", "035720",
                        "086790", "011200", "010130", "009150", "018260", "010950", "033780", "000810",
                        "015760", "329180", "034730", "003550", "000100", "006400"
                    ]
                    batch_add_to_watchlist(kospi_top30, group_name="KOSPI30")
                    st.cache_data.clear()
                    st.toast("코스피 상위 30종목 (KOSPI30 그룹) 추가 완료!")
                    st.rerun()

                if st.button("KOSDAQ 시총 상위 30종목 원클릭 추가", use_container_width=True):
                    kosdaq_top30 = [
                        "247540", "086520", "196170", "058470", "022100", "041510", "277810", "091990",
                        "066970", "028300", "293490", "036930", "263750", "145020", "039030", "005290",
                        "214150", "035900", "112040", "357780", "078600", "141080", "253450", "195940",
                        "095660", "067160", "036570", "048410", "034220", "042000"
                    ]
                    batch_add_to_watchlist(kosdaq_top30, group_name="KOSDAQ30")
                    st.cache_data.clear()
                    st.toast("코스닥 상위 30종목 (KOSDAQ30 그룹) 추가 완료!")
                    st.rerun()

                st.markdown("---")
                del_confirm = st.checkbox("관심종목 전체 삭제 확인", value=False)
                if st.button("관심종목 전체 비우기", disabled=not del_confirm, use_container_width=True):
                    clear_watchlist()
                    st.cache_data.clear()
                    st.warning("모든 관심종목이 삭제되었습니다.")
                    st.rerun()

    if not watchlist_rows:
        st.info("현재 등록된 관심종목이 없습니다. 위 검색창이나 '대량 등록'을 통해 종목을 등록하세요.")
    else:
        # 4. Multi-dimensional Filter & View Toolbar
        all_groups = ["전체"] + sorted(list(set(df_all["group_name"].dropna().unique())))
        all_sectors = ["전체"] + sorted(list(set(df_all["sector"].dropna().unique())))

        t_col0, t_col1, t_col2, t_col3, t_col4, t_col5, t_col6 = st.columns([1.2, 1.6, 1.0, 1.2, 1.4, 1.0, 1.4])
        with t_col0:
            filter_group = st.selectbox("그룹 필터", all_groups, index=0, key="wl_filter_group")
        with t_col1:
            filter_label = st.selectbox(
                "전망/전략 필터",
                [
                    "전체",
                    "실시간 당일단타 적격 (장중 진입: 당일 +5% 익절)",
                    "당일 수급돌파 강세주",
                    "당일 눌림목 지지주",
                    "스나이퍼 적격 (오전 9시 진입: 눌림목 반등)",
                    "5% 급등 타겟 적격 (마감 직전 진입: 1~2일 스윙)",
                    "종가배팅 적격 (마감 직전 진입: 익일 갭익절)",
                    "강세(70점↑)",
                    "중립(50~69점)",
                    "약세(50점↓)"
                ],
                index=0,
                key="wl_filter_label"
            )
        with t_col2:
            filter_market = st.selectbox("시장 구분", ["전체", "KOSPI", "KOSDAQ"], index=0, key="wl_filter_market")
        with t_col3:
            filter_sector = st.selectbox("업종/테마 필터", all_sectors, index=0, key="wl_filter_sector")
        with t_col4:
            sort_by = st.selectbox(
                "정렬 기준",
                ["익일 스코어 높은 순", "전일대비 등락률 높은 순", "시초가대비 등락률 높은 순", "거래량 많은 순", "목표 익절가 높은 순", "종목명 가나다순"],
                index=0,
                key="wl_sort_by"
            )
        with t_col5:
            wl_display_rows = st.selectbox("표시 높이", [10, 15, 20, 30], index=0, format_func=lambda x: f"{x}줄 보기", key="wl_display_rows_sel")
        with t_col6:
            view_mode = st.radio("화면 모드", ["고밀도 테이블 뷰 (권장)", "카드 페이징 뷰"], horizontal=True, key="wl_view_mode")

        # Apply Filters
        df_filtered = df_all.copy()
        if filter_group != "전체":
            df_filtered = df_filtered[df_filtered["group_name"] == filter_group]

        if "당일단타" in filter_label:
            df_filtered = df_filtered[df_filtered["is_daytrade"] == True]
        elif "수급돌파" in filter_label:
            df_filtered = df_filtered[df_filtered["intraday_tag"].str.contains("수급돌파")]
        elif "눌림목" in filter_label:
            df_filtered = df_filtered[df_filtered["intraday_tag"].str.contains("눌림지지")]
        elif "5% 급등" in filter_label:
            df_filtered = df_filtered[df_filtered["is_surge_5pct"] == True]
        elif "스나이퍼" in filter_label:
            df_filtered = df_filtered[df_filtered["is_sniper"] == True]
        elif "종가배팅" in filter_label:
            df_filtered = df_filtered[df_filtered["is_closing_bet"] == True]
        elif filter_label == "강세(70점↑)":
            df_filtered = df_filtered[df_filtered["score"] >= 70.0]
        elif filter_label == "중립(50~69점)":
            df_filtered = df_filtered[(df_filtered["score"] >= 50.0) & (df_filtered["score"] < 70.0)]
        elif filter_label == "약세(50점↓)":
            df_filtered = df_filtered[df_filtered["score"] < 50.0]

        if filter_market != "전체":
            df_filtered = df_filtered[df_filtered["market"] == filter_market]

        if filter_sector != "전체":
            df_filtered = df_filtered[df_filtered["sector"] == filter_sector]

        # Apply Sorting
        if sort_by == "익일 스코어 높은 순":
            df_filtered = df_filtered.sort_values("score", ascending=False)
        elif sort_by == "전일대비 등락률 높은 순":
            df_filtered = df_filtered.sort_values("change_pct", ascending=False)
        elif sort_by == "시초가대비 등락률 높은 순":
            df_filtered = df_filtered.sort_values("intraday_open_pct", ascending=False)
        elif sort_by == "거래량 많은 순":
            df_filtered = df_filtered.sort_values("volume", ascending=False)
        elif sort_by == "목표 익절가 높은 순":
            df_filtered = df_filtered.sort_values("tp_pct", ascending=False)
        elif sort_by == "종목명 가나다순":
            df_filtered = df_filtered.sort_values("name", ascending=True)

        df_filtered = df_filtered.reset_index(drop=True)
        total_wl_items = len(df_filtered)

        # -------------------------------------------------
        # MODE A: High-Density Interactive Data Table (Default)
        # -------------------------------------------------
        if "테이블" in view_mode:
            rows_to_show_wl = int(wl_display_rows)
            wl_table_height = min(38 + rows_to_show_wl * 35, 38 + total_wl_items * 35)
            st.caption(f"필터 결과: **총 {total_wl_items:,}개** 종목 (10줄 높이 고정 · 표 내부 마우스 스크롤로 전체 탐색 및 다중 체크박스 선택 지원)")

            disp_table = df_filtered[[
                "code", "name", "group_name", "market", "sector", "strategy_tag", "close", "change_pct",
                "intraday_tag", "nextday_tag", "tp_pct", "sl_pct", "vol_ratio", "rsi", "ma_status", "verified_badge"
            ]].copy()

            grid_event = st.dataframe(
                disp_table,
                column_config={
                    "code": st.column_config.TextColumn("코드", width="small"),
                    "name": st.column_config.TextColumn("종목명", width="medium"),
                    "group_name": st.column_config.TextColumn("그룹", width="small"),
                    "market": st.column_config.TextColumn("시장", width="small"),
                    "sector": st.column_config.TextColumn("업종", width="small"),
                    "strategy_tag": st.column_config.TextColumn("고확신 전략", width="small"),
                    "close": st.column_config.NumberColumn("현재가", format="%,d원"),
                    "change_pct": st.column_config.NumberColumn("전일대비", format="%+.2f%%"),
                    "intraday_tag": st.column_config.TextColumn("당일 전망 (장중)", width="medium"),
                    "nextday_tag": st.column_config.TextColumn("익일 전망 (스윙)", width="medium"),
                    "tp_pct": st.column_config.NumberColumn("목표익절", format="+%.1f%%"),
                    "sl_pct": st.column_config.NumberColumn("손절기준", format="-%.1f%%"),
                    "vol_ratio": st.column_config.NumberColumn("거래량 급증비", format="%.1f배"),
                    "rsi": st.column_config.NumberColumn("RSI(14)", format="%.1f"),
                    "ma_status": st.column_config.TextColumn("이평배열"),
                    "verified_badge": st.column_config.TextColumn("검증상태"),
                },
                hide_index=True,
                use_container_width=True,
                selection_mode="multi-row",
                on_select="rerun",
                height=wl_table_height,
                key="watchlist_grid"
            )

            # Row Selection Detail Inspector Panel
            selected_indices = grid_event.selection.rows if hasattr(grid_event, "selection") else []
            if selected_indices and len(selected_indices) > 0:
                if len(selected_indices) > 1:
                    sel_rows_df = df_filtered.iloc[selected_indices]
                    st.markdown(f"### 선택 관심종목 일괄 관리 (총 **{len(sel_rows_df)}**개 종목 선택됨)")

                    with st.container(border=True):
                        b_col1, b_col2, b_col3, b_col4 = st.columns([2.5, 2.5, 1.8, 1.8])
                        with b_col1:
                            batch_grp_choice = st.selectbox(
                                "선택 종목 일괄 그룹 변경",
                                ["(그룹 선택)"] + [g for g in avail_groups] + ["+ 새 그룹 직접입력..."],
                                key="batch_insp_grp_sel"
                            )
                        with b_col2:
                            batch_custom_grp = st.text_input(
                                "새 그룹명 입력",
                                "",
                                key="batch_insp_custom_grp",
                                disabled=(batch_grp_choice != "+ 새 그룹 직접입력...")
                            )
                        with b_col3:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button("그룹 일괄 변경", key="btn_batch_grp_apply", use_container_width=True):
                                applied_grp = batch_custom_grp.strip() if batch_grp_choice == "+ 새 그룹 직접입력..." and batch_custom_grp.strip() else batch_grp_choice
                                if applied_grp and applied_grp != "(그룹 선택)":
                                    codes_to_update = sel_rows_df["code"].tolist()
                                    batch_update_watchlist_groups(codes_to_update, applied_grp)
                                    st.cache_data.clear()
                                    st.toast(f"{len(codes_to_update)}개 종목이 '{applied_grp}' 그룹으로 일괄 변경되었습니다!")
                                    st.rerun()
                                else:
                                    st.warning("유효한 그룹을 선택하거나 입력해주세요.")
                        with b_col4:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button(f"{len(sel_rows_df)}개 일괄 삭제", key="btn_batch_del", use_container_width=True):
                                codes_to_del = sel_rows_df["code"].tolist()
                                placeholders = ",".join(["?"] * len(codes_to_del))
                                with get_db_connection() as conn:
                                    cursor = conn.cursor()
                                    cursor.execute(f"DELETE FROM watchlist WHERE code IN ({placeholders})", codes_to_del)
                                    conn.commit()
                                st.cache_data.clear()
                                st.toast(f"{len(codes_to_del)}개 종목이 관심종목에서 삭제되었습니다!")
                                st.rerun()

                    # Multi-stock Summary Metrics
                    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                    m_c1.metric("선택 종목 평균 스코어", f"{sel_rows_df['score'].mean():.1f}점")
                    m_c2.metric("선택 종목 평균 등락률", f"{sel_rows_df['change_pct'].mean():+.2f}%")
                    m_c3.metric("상승 / 보합 / 하락", f"{(sel_rows_df['change_pct'] > 0).sum()} / {(sel_rows_df['change_pct'] == 0).sum()} / {(sel_rows_df['change_pct'] < 0).sum()}")
                    m_c4.metric("고확신(70점↑) 종목", f"{(sel_rows_df['score'] >= 70).sum()}개")

                    st.markdown("---")
                    st.markdown("#### 선택 종목 중 개별 상세 진단")
                    target_stock_name = st.selectbox(
                        "상세 진단할 종목 선택",
                        [f"{r['name']} ({r['code']}) [그룹: {r.get('group_name', '기본그룹')}]" for _, r in sel_rows_df.iterrows()],
                        key="sel_multi_inspect_stock"
                    )
                    sel_stock_code = target_stock_name.split("(")[-1].split(")")[0].strip()
                    sel_row = sel_rows_df[sel_rows_df["code"] == sel_stock_code].iloc[0]
                else:
                    sel_row = df_filtered.iloc[selected_indices[0]]

                st.markdown(f"### 선택 종목 정밀 진단: **{sel_row['name']}** (`{sel_row['code']}`) [소속: `{sel_row.get('group_name', '기본그룹')}`]")

                # 1. Group Manager Row
                with st.container(border=True):
                    g_c1, g_c2, g_c3 = st.columns([2.5, 2.5, 1.5])
                    with g_c1:
                        target_grp_sel = st.selectbox(
                            "소속 그룹 변경",
                            ["(선택 유지)"] + [g for g in avail_groups if g != sel_row.get('group_name')] + ["+ 새 그룹 직접입력..."],
                            key=f"insp_grp_sel_{sel_row['code']}"
                        )
                    with g_c2:
                        custom_grp_name = st.text_input(
                            "새 그룹명 입력",
                            "",
                            key=f"insp_grp_in_{sel_row['code']}",
                            disabled=(target_grp_sel != "+ 새 그룹 직접입력...")
                        )
                    with g_c3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("그룹 변경 저장", key=f"insp_grp_btn_{sel_row['code']}", use_container_width=True):
                            applied_new_grp = custom_grp_name.strip() if target_grp_sel == "+ 새 그룹 직접입력..." and custom_grp_name.strip() else target_grp_sel
                            if applied_new_grp and applied_new_grp != "(선택 유지)":
                                update_watchlist_group(sel_row['code'], applied_new_grp)
                                st.cache_data.clear()
                                st.toast(f"{sel_row['name']} 종목이 '{applied_new_grp}' 그룹으로 변경되었습니다!")
                                st.rerun()

                # 2. Key Metrics & Action Buttons
                insp_c1, insp_c2, insp_c3, insp_c4, insp_btn1, insp_btn2 = st.columns([2, 2, 2, 2, 2.5, 1.5])
                with insp_c1:
                    st.metric("익일 스코어", f"{sel_row['score']:.1f}점", delta=sel_row['label'])
                with insp_c2:
                    st.metric("현재가 / 전일대비", f"{sel_row['close']:,.0f}원", delta=f"{sel_row['change_pct']:+.2f}%")
                with insp_c3:
                    st.metric("시초가대비 등락", f"{sel_row.get('intraday_open_pct', 0.0):+.2f}%", help="오늘 시초가 대비 현재가 변동률")
                with insp_c4:
                    st.metric("목표 / 손절", f"+{sel_row['tp_pct']:.1f}% / -{sel_row['sl_pct']:.1f}%")

                with insp_btn1:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button(f"{sel_row['name']} 7년 백테스트 실행", type="primary", key=f"bt_wl_btn_{sel_row['code']}", use_container_width=True):
                        st.session_state["target_backtest_code"] = sel_row["code"]
                        st.toast(f"탭 3(워크포워드 검증)로 이동하여 {sel_row['name']} 백테스트를 확인하세요!")
                with insp_btn2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("삭제", key=f"del_sel_{sel_row['code']}", use_container_width=True):
                        with get_db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute("DELETE FROM watchlist WHERE code = ?", (sel_row["code"],))
                            conn.commit()
                        st.cache_data.clear()
                        st.toast(f"{sel_row['name']} 삭제 완료!")
                        st.rerun()

                # 3. Dual Outlook Diagnosis Cards (당일 실시간 vs 익일 스윙)
                rep_c1, rep_c2 = st.columns(2)
                with rep_c1:
                    with st.container(border=True):
                        st.markdown(f"**⚡ [당일 장중 실시간 진단]**: `{sel_row.get('intraday_tag', '-')}`")
                        st.caption(f"{sel_row.get('intraday_desc', '')}")
                        st.markdown(f"- **시초가 대비 등락**: `{sel_row.get('intraday_open_pct', 0.0):+.2f}%`")
                        st.markdown(f"- **장중 거래량 급증비**: `{sel_row.get('vol_ratio', 1.0):.2f}배`")
                        st.markdown(f"- **고확신 전략 부합**: `{sel_row.get('strategy_tag', '-')}`")
                with rep_c2:
                    with st.container(border=True):
                        st.markdown(f"**📈 [익일 스윙 퀀트 진단]**: `{sel_row.get('nextday_tag', '-')}`")
                        st.caption(f"{sel_row.get('nextday_desc', '')}")
                        st.markdown(f"- **익일 퀀트 스코어**: `{sel_row['score']:.1f}점 / 100점` ({sel_row['label']})")
                        st.markdown(f"- **이평배열 상태**: `{sel_row.get('ma_status', '-')}`")
                        st.markdown(f"- **권장 목표/손절선**: `익절 +{sel_row['tp_pct']:.1f}% / 손절 -{sel_row['sl_pct']:.1f}%`")

                # Detailed Technical Breakdown
                bk = sel_row["breakdown"]
                b1, b2, b3, b4 = st.columns(4)
                b1.info(f"**RSI(14)**: {bk.get('rsi_val', 0):.1f} (기여: {bk.get('rsi_score', 0)}점)")
                b2.info(f"**이평배열**: {bk.get('ma_status', '-')} (기여: {bk.get('ma_score', 0)}점)")
                b3.info(f"**거래량 급증**: {bk.get('vol_ratio', 1.0):.2f}배 (기여: {bk.get('vol_score', 0)}점)")
                b4.info(f"**볼린저 %b**: {bk.get('bb_pct', 0.5):.2f} (기여: {bk.get('bb_score', 0)}점)")

        # -------------------------------------------------
        # MODE B: Paginated Card View
        # -------------------------------------------------
        else:
            page_size = int(wl_display_rows) if 'wl_display_rows' in locals() else 12
            total_pages = max(1, (len(df_filtered) - 1) // page_size + 1)
            p_col1, p_col2 = st.columns([4, 1])
            with p_col2:
                page_num = st.number_input("페이지 선택", min_value=1, max_value=total_pages, value=1, step=1, key="wl_card_page_num_in")

            start_idx = (page_num - 1) * page_size
            end_idx = min(start_idx + page_size, len(df_filtered))
            paged_data = df_filtered.iloc[start_idx:end_idx]

            card_cols = st.columns(3)
            for i, (_, item) in enumerate(paged_data.iterrows()):
                with card_cols[i % 3]:
                    with st.container(border=True):
                        st.markdown(f"**{item['name']}** (`{item['code']}`)")
                        st.caption(f"{item['market']} | {item['sector']} | 📁 {item.get('group_name', '기본그룹')}")
                        chg_c = "red" if item["change_pct"] > 0 else ("blue" if item["change_pct"] < 0 else "gray")
                        st.markdown(f"**{item['close']:,.0f}원** <span style='color:{chg_c}; font-weight:600;'>{item['change_pct']:+.2f}%</span>", unsafe_allow_html=True)
                        st.markdown(f"**당일**: `{item.get('intraday_tag', '-')}`")
                        st.markdown(f"**익일**: `{item.get('nextday_tag', '-')}`")
                        st.progress(min(1.0, max(0.0, item["score"] / 100.0)))
                        st.caption(f"목표 익절: +{item['tp_pct']:.1f}% | 손절: -{item['sl_pct']:.1f}%")

                        btn_c1, btn_c2 = st.columns(2)
                        with btn_c1:
                            if st.button("백테스트", key=f"bt_card_{item['code']}", use_container_width=True):
                                st.session_state["target_backtest_code"] = item["code"]
                                st.toast(f"{item['name']} 백테스트 설정됨! 탭 3을 확인하세요.")
                        with btn_c2:
                            if st.button("삭제", key=f"del_card_{item['code']}", use_container_width=True):
                                with get_db_connection() as conn:
                                    cursor = conn.cursor()
                                    cursor.execute("DELETE FROM watchlist WHERE code = ?", (item["code"],))
                                    conn.commit()
                                st.cache_data.clear()
                                st.rerun()



# =========================================================
# TAB 3: 워크포워드 검증 & 벤치마크 리포트 (95% CI 부트스트랩)
# =========================================================
with tab3:
    st.markdown('<div class="office-heading">3. 워크포워드 백테스트 & KODEX 200 벤치마크 검증 리포트</div>', unsafe_allow_html=True)
    st.caption("250거래일 In-Sample 학습 구간 롤링 후 20거래일 Out-of-Sample 순차 테스트 (과최적화 방지 & 부트스트랩 1,000회 95% 신뢰구간 산출)")

    bt_c1, bt_c2, bt_c3, bt_c4, bt_c5 = st.columns([2.5, 2.2, 1.8, 1.5, 1.5])
    with bt_c1:
        # Pre-populate with watchlist codes or direct input
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT code, name FROM watchlist ORDER BY name ASC")
            wl_opts = [f"{r['name']} ({r['code']})" for r in cursor.fetchall()]
        if not wl_opts:
            wl_opts = ["삼성전자 (005930)", "SK하이닉스 (000660)"]

        # If a stock was selected from Tab 1 inspector
        target_override = st.session_state.get("target_backtest_code")
        default_idx = 0
        if target_override:
            for idx, opt in enumerate(wl_opts):
                if target_override in opt:
                    default_idx = idx
                    break

        selected_wl = st.selectbox("검증 대상 종목", wl_opts, index=default_idx, key="bt_target_wl")
        target_code = selected_wl.split("(")[-1].replace(")", "").strip()
    with bt_c2:
        bt_strategy = st.selectbox(
            "검증 전략 모드",
            [
                "일반 퀀트 모드 (기본 점수제)",
                "실시간 당일 단타 모드 (당일 +5.0% 익절 / 당일 전량청산)",
                "스나이퍼 고확신 모드 (오전 9시 진입: 80% 승률 타겟)",
                "5% 급등 타겟 모드 (마감 직전 15:20 진입: 1~2일 스윙)",
                "주도주 종가배팅 모드 (마감 직전 15:20 매수 -> 익일 시초 갭익절)"
            ],
            index=0,
            key="bt_strategy_sel"
        )
    with bt_c3:
        bt_min_score = st.slider("최소 점수 (일반 모드)", 60.0, 85.0, 70.0, step=5.0, key="bt_min_score_slider")
    with bt_c4:
        bt_start_year = st.selectbox("시작 연도", [2017, 2018, 2019, 2020, 2021, 2022], index=3, key="bt_start_year_sel")
    with bt_c5:
        if "당일 단타" in bt_strategy:
            default_sl = 2.5
        elif ("5% 급등" in bt_strategy or "5% 돌파" in bt_strategy):
            default_sl = 4.0
        else:
            default_sl = 2.0
        bt_sl = st.slider("손절선 (%)", 1.0, 6.0, default_sl, step=0.5, key="bt_sl_slider")

    if st.button("워크포워드 백테스트 & 부트스트랩 검증 실행", type="primary", use_container_width=True):
        if "당일 단타" in bt_strategy:
            mode_key = "DAY_TRADE_5PCT"
        elif ("5% 급등" in bt_strategy or "5% 돌파" in bt_strategy):
            mode_key = "SURGE_5PCT"
        elif "스나이퍼" in bt_strategy:
            mode_key = "SNIPER"
        elif "종가배팅" in bt_strategy:
            mode_key = "CLOSING_BET"
        else:
            mode_key = "NORMAL"
        with st.spinner(f"[{target_code}] {bt_strategy} 롤링 워크포워드 검증 및 1,000회 부트스트랩 리샘플링 중..."):
            bt_results = run_walk_forward_backtest(
                code=target_code,
                start_year=bt_start_year,
                min_entry_score=bt_min_score,
                strategy_mode=mode_key,
                custom_sl_pct=bt_sl
            )

        if "error" in bt_results:
            st.error(bt_results["error"])
        elif "warning" in bt_results:
            st.warning(bt_results["warning"])
        else:
            stats = bt_results["stats"]
            trades_df = bt_results["trades_df"]
            equity_curve = bt_results["equity_curve"]
            exit_counts = bt_results["exit_counts"]

            st.markdown("---")
            # 1. Big Status Badge
            badge_col1, badge_col2 = st.columns([1, 3])
            with badge_col1:
                if stats["is_verified"]:
                    st.markdown("### 상태 배지: <br><span class='badge-verified' style='font-size:1.3rem; padding:8px 16px;'>● 검증됨 (Verified)</span>", unsafe_allow_html=True)
                else:
                    st.markdown("### 상태 배지: <br><span class='badge-experimental' style='font-size:1.3rem; padding:8px 16px;'>● 실험적 (Experimental)</span>", unsafe_allow_html=True)
            with badge_col2:
                st.markdown(
                    f"**검증 판정 기준**: KODEX 200 대비 초과수익률 95% 신뢰구간 하한 > 0 **AND** 승률 95% 신뢰구간 하한 > 50%<br>"
                    f"- 현재 초과수익 95% CI 하한: **{stats['excess_return_ci'][0]:+.2f}%** ({'충족' if stats['excess_return_ci'][0] > 0 else '미충족'})<br>"
                    f"- 현재 승률 95% CI 하한: **{stats['win_rate_ci'][0]:.1f}%** ({'충족' if stats['win_rate_ci'][0] > 50 else '미충족'})",
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            # 2. Key Metrics Row
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric(
                "실현 승률",
                f"{stats['win_rate']:.1f}%",
                help=f"95% 부트스트랩 신뢰구간: [{stats['win_rate_ci'][0]:.1f}% ~ {stats['win_rate_ci'][1]:.1f}%]"
            )
            m2.metric(
                "KODEX 200 초과수익(알파)",
                f"{stats['mean_excess_return']:+.2f}%",
                help=f"95% 부트스트랩 신뢰구간: [{stats['excess_return_ci'][0]:+.2f}% ~ {stats['excess_return_ci'][1]:+.2f}%]"
            )
            m3.metric("Profit Factor (PF)", f"{stats['profit_factor']:.2f}")
            m4.metric("Max Drawdown (MDD)", f"-{stats['max_drawdown_pct']:.2f}%")
            m5.metric("총 OOS 거래 건수", f"{stats['total_trades']}건")

            st.markdown("---")

            # 3. Interactive Charts: Equity Curve & Distribution
            chart_col1, chart_col2 = st.columns([2, 1])
            with chart_col1:
                st.markdown("#### 누적 수익률 비교 (전략 vs KODEX 200 ETF)")
                if not equity_curve.empty:
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=equity_curve["date"],
                        y=equity_curve["strat_cum_return"],
                        mode="lines",
                        name="전략 누적 순수익률 (%)",
                        line=dict(color="#2563EB", width=2.5)
                    ))
                    fig_eq.add_trace(go.Scatter(
                        x=equity_curve["date"],
                        y=equity_curve["bm_cum_return"],
                        mode="lines",
                        name="KODEX 200 동시보유 수익률 (%)",
                        line=dict(color="#94A3B8", width=1.8, dash="dash")
                    ))
                    fig_eq.update_layout(
                        margin=dict(l=20, r=20, t=30, b=20),
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_eq, use_container_width=True)

            with chart_col2:
                st.markdown("#### 청산 사유 분포")
                if exit_counts:
                    labels = list(exit_counts.keys())
                    values = list(exit_counts.values())
                    fig_pie = px.pie(
                        names=labels,
                        values=values,
                        hole=0.45,
                        color_discrete_sequence=["#22C55E", "#EF4444", "#3B82F6"]
                    )
                    fig_pie.update_layout(margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_pie, use_container_width=True)

            # 4. Bootstrap Win Rate Distribution Histogram
            st.markdown("#### 1,000회 부트스트랩 승률 분포 (95% 신뢰구간)")
            boot_wr = stats.get("bootstrap_win_rates", [])
            if boot_wr:
                fig_hist = px.histogram(
                    x=boot_wr,
                    nbins=25,
                    title=f"승률 95% 신뢰구간: [{stats['win_rate_ci'][0]:.1f}% ~ {stats['win_rate_ci'][1]:.1f}%]",
                    labels={"x": "리샘플링 승률 (%)", "y": "빈도수"},
                    color_discrete_sequence=["#6366F1"]
                )
                fig_hist.add_vline(x=50.0, line_dash="dash", line_color="red", annotation_text="기준선 50%")
                fig_hist.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_hist, use_container_width=True)

            # 5. Recent Trades Detailed Table
            st.markdown("#### 최근 Out-of-Sample 상세 매매 내역 (최근 15건)")
            recent_trades = trades_df.tail(15).iloc[::-1]
            st.dataframe(
                recent_trades[[
                    "entry_date", "score", "raw_entry", "raw_exit", "exit_reason",
                    "net_return_pct", "bm_return_pct", "excess_return_pct"
                ]].rename(columns={
                    "entry_date": "진입일자",
                    "score": "신호 스코어",
                    "raw_entry": "진입가",
                    "raw_exit": "청산가",
                    "exit_reason": "청산 사유",
                    "net_return_pct": "전략 순수익률(%)",
                    "bm_return_pct": "KODEX 200(%)",
                    "excess_return_pct": "초과수익(알파)(%)"
                }),
                use_container_width=True
            )


# =========================================================
# TAB 4: 예측 다이어리 & 라이브 추적 (Forward Testing Log)
# =========================================================
with tab4:
    st.markdown('<div class="office-heading">4. 실거래 예측 다이어리 & 라이브 포워드 테스팅</div>', unsafe_allow_html=True)
    st.caption("매일 15:40 마감 후 생성된 예측 건과 익일 실제 체결 시세(시/고/저/종)를 일대일 대조하여 모델 열화(Model Decay)를 실시간 감지합니다.")

    # 1. Model Decay Status Card
    decay_info = evaluate_model_decay(expected_ci_lower=50.0)
    
    col_dec1, col_dec2 = st.columns([1.2, 3])
    with col_dec1:
        if decay_info["status"] == "DECAY_WARNING":
            st.error("모델 열화(Model Decay) 감지")
        elif decay_info["status"] == "HEALTHY":
            st.success("모델 성능 정상 (Healthy)")
        else:
            st.info("표본 축적 진행 중")
    with col_dec2:
        st.markdown(f"**진단 결과**: {decay_info['message']}")
        st.caption(f"누적 정산: {decay_info['total_settled']}건 | 적중: {decay_info.get('hits', 0)}건 | 실현 승률: {decay_info['live_win_rate']:.1f}%")

    st.markdown("---")

    # Manual Settlement / Refresh Action
    act_col1, act_col2 = st.columns([3, 1])
    with act_col1:
        st.markdown("#### 일자별 예측 및 익일 정산 내역")
    with act_col2:
        if st.button("전일 예측 즉시 정산", use_container_width=True):
            with st.spinner("미정산 예측 건 시세 조회 및 정산 처리 중..."):
                settled = settle_pending_predictions()
                st.toast(f"{len(settled)}건의 예측이 정산되었습니다.")
                st.rerun()

    # Load Diary History
    diary_df = get_diary_history(limit=50)

    if diary_df.empty:
        st.info("현재 기록된 예측 다이어리가 없습니다. 15:40 마감 배치를 실행하거나 사이드바의 '15:40 마감 배치' 버튼을 눌러보세요.")
    else:
        # Display Table
        disp_df = diary_df.copy()
        disp_df["hit_status"] = disp_df["hit_status"].fillna("정산대기 (PENDING)")
        
        st.dataframe(
            disp_df[[
                "prediction_date", "code", "name", "score", "label",
                "target_tp", "target_sl", "status", "execution_date",
                "actual_return", "hit_status", "exit_type"
            ]].rename(columns={
                "prediction_date": "예측일자",
                "code": "종목코드",
                "name": "종목명",
                "score": "스코어",
                "label": "전망",
                "target_tp": "목표익절(%)",
                "target_sl": "손절기준(%)",
                "status": "상태",
                "execution_date": "체결일자",
                "actual_return": "실현수익률(%)",
                "hit_status": "적중여부",
                "exit_type": "청산유형"
            }),
            use_container_width=True
        )

        # Forward Testing Hit Rate Chart
        settled_subset = diary_df[diary_df["hit_status"].isin(["HIT", "MISS"])].copy()
        if not settled_subset.empty:
            st.markdown("#### 라이브 포워드 테스팅 누적 적중률 추이")
            settled_subset["is_hit"] = (settled_subset["hit_status"] == "HIT").astype(int)
            settled_subset = settled_subset.iloc[::-1].reset_index(drop=True)
            settled_subset["cum_hit_rate"] = (settled_subset["is_hit"].cumsum() / (settled_subset.index + 1)) * 100.0

            fig_live = px.line(
                settled_subset,
                x="execution_date",
                y="cum_hit_rate",
                title="라이브 예측 실현 승률 추이 (%)",
                labels={"execution_date": "체결일자", "cum_hit_rate": "누적 승률 (%)"},
                markers=True
            )
            fig_live.add_hline(y=50.0, line_dash="dash", line_color="gray", annotation_text="50% 기준선")
            fig_live.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_live, use_container_width=True)


# =========================================================
# TAB 5: 전일 성과 자가검증 & AI 전략 최적화 (Self-Tuning)
# =========================================================
with tab5:
    st.markdown('<div class="office-heading">5. 전일 성과 사후 검증 & AI 전략 자가최적화 (Self-Tuning)</div>', unsafe_allow_html=True)
    st.caption("매일 전일 추천 종목의 당일 실제 체결 성과(승률/수익률/알파)를 사후 검증하고, 실패 요인을 스스로 학습하여 최적 스크리닝 파라미터로 자동 보정합니다.")

    # 1. Date & Strategy Controls
    avail_dates = get_available_trading_dates(limit=30)

    v_c1, v_c2, v_c3 = st.columns([1.5, 2.0, 1.2])
    with v_c1:
        sel_verif_date = st.selectbox(
            "검증 기준일 (T-1 추천 발굴일)",
            options=avail_dates[::-1] if avail_dates else ["2026-09-09"],
            index=1 if len(avail_dates) > 1 else 0,
            key="verif_pred_date_sel"
        )
    with v_c2:
        sel_verif_strat = st.selectbox(
            "검증 대상 전략 모드",
            [
                "⚡ 실시간 당일 단타 (5% 익절)",
                "🎯 스나이퍼 고확신 (눌림목 반등)",
                "🚀 5% 급등 타겟 (1~2일 스윙)",
                "🌙 주도주 종가배팅 (익일 시초 갭)",
                "📊 일반 퀀트 스코어링"
            ],
            index=0,
            key="verif_strat_mode_sel"
        )
    with v_c3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_verif_btn = st.button("🔍 전일 성과 검증 & 자가 진단 실행", type="primary", use_container_width=True, key="btn_run_daily_verif")

    # Auto-run on first load or when user clicks run button
    cache_key = f"{sel_verif_date}_{sel_verif_strat}"
    cached_key = st.session_state.get("daily_verif_cache_key")

    if run_verif_btn or ("daily_verif_cache" not in st.session_state):
        with st.spinner(f"[{sel_verif_date}] 기준 스크리닝 및 익일 실제 체결 데이터 사후 검증 중..."):
            v_res = run_daily_point_in_time_verification(
                pred_date=sel_verif_date,
                strategy_mode=sel_verif_strat,
                min_val_krw=10_000_000_000,
                score_cutoff=65.0,
                sample_pool_size=150
            )
            st.session_state["daily_verif_cache"] = v_res
            st.session_state["daily_verif_cache_key"] = cache_key

    # Display results if available in session state
    verif_data = st.session_state.get("daily_verif_cache")

    if not verif_data:
        st.info("💡 상단의 **[🔍 전일 성과 검증 & 자가 진단 실행]** 버튼을 누르시면, 선택하신 날짜의 추천 종목과 익일 실제 체결 성과를 대조 분석하여 실패 요인 진단 및 최적화 파라미터를 도출합니다.")
    elif "error" in verif_data:
        st.warning(f"⚠️ {verif_data['error']}")
    elif verif_data.get("total_screened", 0) == 0:
        st.info(f"선택일({verif_data.get('pred_date', sel_verif_date)})에 해당 전략 조건으로 포착된 종목이 없습니다. 다른 일자나 전략을 선택해 보세요.")
    else:
        kpi = verif_data["kpi"]
        diag = verif_data["diagnosis"]
        tuning = verif_data["tuning"]
        df_res = verif_data["df_results"]

        st.markdown("---")

        # 1. KPI Metric Row
        k_c1, k_c2, k_c3, k_c4 = st.columns(4)
        with k_c1:
            st.metric(
                "실현 적중률 (승률)",
                f"{kpi['win_rate']:.1f}%",
                delta=f"{kpi['hits']} / {kpi['total']} 종목 적중"
            )
        with k_c2:
            st.metric(
                "평균 실현 순수익률",
                f"{kpi['avg_net_ret']:+.2f}%",
                delta=f"목표익절 {kpi['tp_count']}건 / 손절 {kpi['sl_count']}건"
            )
        with k_c3:
            st.metric(
                "KODEX 200 대비 알파",
                f"{kpi['avg_excess_ret']:+.2f}%p",
                delta=f"시장수익률: {kpi['bm_day_ret']:+.2f}%"
            )
        with k_c4:
            st.metric(
                "장중 최대상승폭 평균",
                f"{kpi['avg_max_gain']:+.2f}%",
                delta="장중 최고가 기준"
            )

        # 2. AI Root Cause Diagnosis & Failure Analysis
        st.markdown("---")
        st.markdown("#### 🧠 AI 자가 진단 & 실패 원인 분석 보고서")
        st.caption(f"검증 기준일: **{verif_data['pred_date']}** ➔ 체결 검증일: **{verif_data['exec_date']}** | {diag.get('summary', '')}")

        issues = diag.get("issues", [])
        if not issues:
            st.success("🎉 **[결함 요인 없음]** 모든 추천 종목이 목표 익절 또는 안정적인 양의 수익률을 달성하였습니다! 현행 파라미터가 장세와 완벽하게 일치합니다.")
        else:
            for iss in issues:
                sev_icon = "🚨" if iss["severity"] == "HIGH" else "⚠️"
                with st.expander(f"{sev_icon} [{iss['title']}]", expanded=True):
                    st.markdown(f"**진단 내역**: {iss['description']}")
                    st.markdown(f"**권장 조치**: `{iss['action']}`")

        # Feature Comparison Table (Hits vs Misses)
        stats_cmp = diag.get("stats_comparison", {})
        if stats_cmp and "metric" in stats_cmp:
            with st.expander("📊 성공 종목 vs 실패 종목 핵심 팩터 비교표", expanded=False):
                cmp_df = pd.DataFrame(stats_cmp).rename(columns={
                    "metric": "비교 지표",
                    "hits": "성공/익절 종목군",
                    "misses": "실패/손절 종목군"
                })
                st.table(cmp_df)

        # 3. Strategy Auto-Tuning Proposals & 1-Click Apply
        st.markdown("---")
        st.markdown("#### ⚙️ 전략 자가 수정(Auto-Tuning) 제안 & 원클릭 최적화 반영")
        st.caption("발굴된 결함 요인을 보정하기 위해 도출된 최적 스크리너 파라미터입니다. 적용 시 스크리너 필터에 즉시 반영됩니다.")

        proposals = tuning.get("proposals", [])
        sim = tuning.get("simulation", {})

        if proposals:
            t_col1, t_col2 = st.columns([1.6, 1.2])
            with t_col1:
                prop_rows = []
                for p in proposals:
                    prop_rows.append({
                        "조정 파라미터": p["param_name"],
                        "현재 설정값": p["current_val"],
                        "AI 제안 최적값": p["recommended_val"],
                        "개선 근거": p["reason"]
                    })
                st.table(pd.DataFrame(prop_rows))

            with t_col2:
                st.markdown("**💡 자가 수정 시뮬레이션 개선 효과**")
                st.markdown(
                    f"""
                    - **적용 전 승률**: `{sim.get('before_win_rate', 0):.1f}%` ({sim.get('before_total', 0)}종목)
                    - **최적화 후 승률**: **`{sim.get('after_win_rate', 0):.1f}%`** ({sim.get('after_total', 0)}종목)
                    - **승률 향상폭**: **`+{sim.get('win_rate_boost', 0):.1f}%p`**
                    - **손실 종목 사전 차단**: **`{sim.get('filtered_losses', 0)}개`** 손실 유발 종목 원천 배제
                    """
                )

                if st.button("⚡ 진단된 최적 파라미터를 스크리너에 즉시 자동 적용", type="primary", use_container_width=True, key="btn_apply_auto_tune"):
                    for p in proposals:
                        pkey = p.get("param_key")
                        tval = p.get("target_val_float")
                        if pkey == "max_open_gain" and tval:
                            cur_g = st.session_state.get("sc_intraday_gain_range", (3.0, 8.5))
                            st.session_state["sc_intraday_gain_range"] = (min(cur_g[0], tval - 0.5), float(tval))
                        elif pkey == "min_trading_val" and tval:
                            val_options = [50, 100, 200, 300]
                            best_val = min(val_options, key=lambda x: abs(x - int(tval)))
                            st.session_state["sc_min_daytrade_val"] = best_val
                            st.session_state["sc_min_val_krw"] = best_val
                    st.toast("✅ 자가 최적화 파라미터가 스크리너(탭 1)에 성공적으로 자동 반영되었습니다!")
                    st.success("✅ **[적용 완료]** 최적화 파라미터가 반영되었습니다! 탭 1로 이동하시면 한층 정밀해진 스크리닝 결과를 바로 확인하실 수 있습니다.")

        # 4. Detailed Stock Performance Table
        st.markdown("---")
        st.markdown("#### 📋 검증 대상 종목별 상세 성적표")

        disp_cols = [
            "code", "name", "market", "t1_score", "strategy", "open_gap_pct",
            "t_open", "t_high", "t_close", "max_gain_pct", "net_return_pct",
            "is_hit", "exit_code", "exit_reason", "excess_return"
        ]
        disp_df = df_res[disp_cols].copy()
        disp_df["is_hit"] = disp_df["is_hit"].apply(lambda x: "✅ 적중" if x else "❌ 손절/미달")

        st.dataframe(
            disp_df.rename(columns={
                "code": "종목코드",
                "name": "종목명",
                "market": "시장",
                "t1_score": "T-1스코어",
                "strategy": "전략구분",
                "open_gap_pct": "시초갭(%)",
                "t_open": "시초가",
                "t_high": "고가",
                "t_close": "종가",
                "max_gain_pct": "장중최대상승(%)",
                "net_return_pct": "실현수익률(%)",
                "is_hit": "적중여부",
                "exit_code": "청산코드",
                "exit_reason": "청산상세",
                "excess_return": "알파(%)"
            }),
            height=388,
            use_container_width=True
        )
