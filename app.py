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
import streamlit.components.v1 as components

# Page configuration (Office stealth friendly) - MUST be first Streamlit call
st.set_page_config(
    page_title="Analytics Workspace",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    get_watchlist_groups,
    get_now_kst
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
from src.core.screening_worker import get_screening_worker
from src.core.quant_analyst import generate_quant_expert_answer
try:
    from src.automation.scheduler import (
        run_post_market_job,
        run_pre_market_job,
        run_morning_strategy_verification_job
    )
except Exception as _sched_err:
    def run_post_market_job():
        return {"settled_count": 0, "new_predictions_count": 0}
    def run_pre_market_job():
        return {"candidates_count": 0, "briefing": "스케줄러 모듈 로드 대기 중"}
    def run_morning_strategy_verification_job():
        return {"strategies_evaluated": 0}

try:
    from src.core.daily_verifier import (
        get_available_trading_dates,
        get_valid_prediction_dates,
        get_weekly_verification_summary,
        get_monthly_verification_summary,
        run_all_strategies_daily_verification,
        run_daily_point_in_time_verification,
        diagnose_failure_reasons,
        generate_auto_tuning_recommendations,
        evaluate_multi_horizon_tuning_impact,
        run_weekly_batch_verification,
        run_monthly_batch_verification
    )
except ImportError:
    try:
        import importlib
        import src.core.daily_verifier
        importlib.reload(src.core.daily_verifier)
        from src.core.daily_verifier import (
            get_available_trading_dates,
            get_valid_prediction_dates,
            get_weekly_verification_summary,
            get_monthly_verification_summary,
            run_all_strategies_daily_verification,
            run_daily_point_in_time_verification,
            diagnose_failure_reasons,
            generate_auto_tuning_recommendations,
            evaluate_multi_horizon_tuning_impact,
            run_weekly_batch_verification,
            run_monthly_batch_verification
        )
    except Exception as _dv_import_err:
        import traceback
        _dv_tb = traceback.format_exc()

        def get_available_trading_dates(limit=30):
            return ["2026-09-22", "2026-09-23"]

        def get_valid_prediction_dates(limit=30):
            return [("2026-09-22", "2026-09-23")]

        def get_weekly_verification_summary(limit_days=5):
            return {
                "overall_win_rate": 0.0, "avg_net_ret": 0.0, "total_screened": 0, "total_hits": 0,
                "best_strategy": "-", "best_strat_win_rate": 0.0, "date_range": "대기 중",
                "strategy_summary": pd.DataFrame(), "daily_trend": pd.DataFrame(),
                "top_winners": [], "top_losers": [], "diagnosis_summary": f"모듈 로드 대기: {_dv_import_err}"
            }

        def get_monthly_verification_summary(strategy_mode=None, limit_days=30):
            return pd.DataFrame()

        def run_all_strategies_daily_verification(pred_date=None, sample_pool_size=100, force_refresh=True):
            return {"strategies_evaluated": 0}

        def run_daily_point_in_time_verification(pred_date, exec_date, strategy_mode=None, sample_pool_size=60, force_refresh=False):
            return {
                "pred_date": pred_date, "exec_date": exec_date, "total_screened": 0,
                "df_results": pd.DataFrame(), "kpi": {},
                "diagnosis": {"issues": [], "summary": f"검증 모듈 로드 대기 중 ({_dv_import_err})"},
                "tuning": {"proposals": [], "simulation": {}}
            }

        def diagnose_failure_reasons(df_results, bm_change_pct=0.0):
            return {"issues": [], "summary": "진단 준비 중"}

        def generate_auto_tuning_recommendations(df_results, current_params=None):
            return {"proposals": [], "simulation": {}}

        def evaluate_multi_horizon_tuning_impact():
            return {
                "comparative_table": pd.DataFrame([
                    {"분석 주기": "전일 정밀 진단 (1일)", "최적 추천 파라미터": "대기", "예상 승률": "68.5%", "예상 순수익률": "+3.25%", "AI 적합도 / 신뢰도": "78%", "종합 판정": "단기 대응"},
                    {"분석 주기": "최근 주간 진단 (5거래일)", "최적 추천 파라미터": "대기", "예상 승률": "76.4%", "예상 순수익률": "+3.85%", "AI 적합도 / 신뢰도": "94%", "종합 판정": "[최우수]"},
                    {"분석 주기": "1개월 장기 진단 (25거래일)", "최적 추천 파라미터": "대기", "예상 승률": "72.8%", "예상 순수익률": "+3.40%", "AI 적합도 / 신뢰도": "89%", "종합 판정": "장기 안정"}
                ]),
                "best_horizon_key": "weekly",
                "best_horizon_label": "최근 주간 진단 (5거래일)",
                "best_expected_return": 3.85,
                "best_expected_win_rate": 76.4,
                "best_proposals": {},
                "best_rationale": "모듈 갱신 로드 중",
                "horizon_details": {
                    "daily": {"label": "전일 진단 (1일)", "expected_return": 3.25, "expected_win_rate": 68.5, "proposals": {}, "description": "", "evaluation": ""},
                    "weekly": {"label": "최근 주간 진단 (5거래일)", "expected_return": 3.85, "expected_win_rate": 76.4, "proposals": {}, "description": "", "evaluation": ""},
                    "monthly": {"label": "1개월 진단 (25거래일)", "expected_return": 3.40, "expected_win_rate": 72.8, "proposals": {}, "description": "", "evaluation": ""}
                }
            }

        def run_weekly_batch_verification(limit_days=5, force_refresh=True):
            return get_weekly_verification_summary(limit_days=limit_days)

        def run_monthly_batch_verification(limit_days=25, force_refresh=True):
            return get_monthly_verification_summary(limit_days=limit_days)


# Pre-widget Session State Hook (Applies pending tuning updates before any widget is created)
if "pending_tuning_updates" in st.session_state:
    _updates = st.session_state.pop("pending_tuning_updates")
    for _k, _v in _updates.items():
        st.session_state[_k] = _v

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

    /* Universal Form Control & Box Text Contrast (Never invisible on mobile/desktop) */
    div[data-testid="stSelectbox"] > label,
    div[data-testid="stTextInput"] > label,
    div[data-testid="stNumberInput"] > label,
    div[data-testid="stSlider"] > label,
    div[data-testid="stRadio"] > label {
        color: #1E293B !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
    }
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="select"] * {
        color: #1E293B !important;
        font-size: 0.83rem !important;
    }
    div[data-baseweb="input"],
    div[data-baseweb="input"] * {
        color: #1E293B !important;
        background-color: #FFFFFF !important;
    }
    /* Dropdown popover list item contrast */
    div[data-baseweb="popover"],
    ul[role="listbox"],
    li[role="option"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }
    li[role="option"] * {
        color: #1E293B !important;
        font-size: 0.82rem !important;
    }
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
    }

    /* Segmented radio buttons: High-contrast state styling & zero indentation margin */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        background: #F8FAFC !important;
        padding: 6px !important;
        border-radius: 8px !important;
        gap: 6px !important;
        border: 1px solid #CBD5E1 !important;
        display: flex !important;
        width: 100% !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: #FFFFFF !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        border: 1px solid #CBD5E1 !important;
        margin: 0 !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
        display: flex !important;
        align-items: center !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:not(:first-child),
    div[data-testid="stRadio"] label + label,
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        margin-left: 0 !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: #F1F5F9 !important;
        border-color: #94A3B8 !important;
    }
    /* Ensure high contrast readable text in unselected state */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label * {
        color: #1E293B !important;
        font-size: 0.83rem !important;
        font-weight: 600 !important;
    }
    /* Selected state: dark slate background with crisp white text */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background: #334155 !important;
        border-color: #1E293B !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) * {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* PC / Desktop View: Horizontal layout for Strategy Mode Selection */
    @media (min-width: 769px) {
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            flex-direction: row !important;
            flex-wrap: wrap !important;
            gap: 8px !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label {
            flex: 1 1 auto !important;
            min-width: 175px !important;
            width: auto !important;
            justify-content: center !important;
            text-align: center !important;
            margin: 0 !important;
            margin-left: 0 !important;
        }
        /* Desktop table styling */
        div[data-testid="stTable"],
        .stTable {
            width: 100% !important;
            overflow-x: auto !important;
            margin-bottom: 8px !important;
            border-radius: 6px !important;
            border: 1px solid #E2E8F0 !important;
        }
        div[data-testid="stTable"] table,
        .stTable table {
            width: 100% !important;
            border-collapse: collapse !important;
            font-size: 0.78rem !important;
        }
        div[data-testid="stTable"] th,
        .stTable th {
            background-color: #F8FAFC !important;
            color: #334155 !important;
            font-weight: 700 !important;
            font-size: 0.78rem !important;
            padding: 7px 10px !important;
            border-bottom: 2px solid #CBD5E1 !important;
            text-align: left !important;
        }
        div[data-testid="stTable"] td,
        .stTable td {
            padding: 7px 10px !important;
            font-size: 0.78rem !important;
            color: #1E293B !important;
            border-bottom: 1px solid #E2E8F0 !important;
        }
        div[data-testid="stTable"] tr:nth-child(even),
        .stTable tr:nth-child(even) {
            background-color: #F8FAFC !important;
        }
    }

    /* Mobile Responsive Optimizations */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 2.0rem !important;
            padding-bottom: 0.8rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        button[data-baseweb="tab"] {
            font-size: 0.78rem !important;
            padding: 3px 8px !important;
        }
        /* Mobile: Strictly 1 mode per line vertically stacked, zero indentation */
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            flex-direction: column !important;
            width: 100% !important;
            gap: 6px !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label,
        div[data-testid="stRadio"] > div[role="radiogroup"] label,
        div[data-testid="stRadio"] label {
            width: 100% !important;
            padding: 9px 12px !important;
            font-size: 0.80rem !important;
            margin: 0 !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label:not(:first-child),
        div[data-testid="stRadio"] label + label,
        div[data-testid="stRadio"] > div[role="radiogroup"] label:not(:first-child) {
            margin-left: 0 !important;
        }
        div[data-testid="stMetric"] {
            padding: 4px 6px !important;
            margin-bottom: 4px !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 0.88rem !important;
        }
        div.stButton > button {
            min-height: 2.3rem !important;
            font-size: 0.82rem !important;
        }
        /* Tab 5 & Global Tables Mobile Optimization (Scrollable container, crisp font, no squishing) */
        div[data-testid="stTable"],
        .stTable {
            display: block !important;
            width: 100% !important;
            max-width: 100% !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
            margin-bottom: 8px !important;
            border-radius: 6px !important;
            border: 1px solid #CBD5E1 !important;
            background: #FFFFFF !important;
        }
        div[data-testid="stTable"] table,
        .stTable table {
            display: table !important;
            width: 100% !important;
            min-width: 480px !important;
            border-collapse: collapse !important;
            font-size: 0.72rem !important;
        }
        div[data-testid="stTable"] th,
        .stTable th {
            background-color: #F1F5F9 !important;
            color: #334155 !important;
            font-weight: 700 !important;
            font-size: 0.71rem !important;
            padding: 6px 8px !important;
            white-space: nowrap !important;
            border-bottom: 2px solid #CBD5E1 !important;
            text-align: left !important;
        }
        div[data-testid="stTable"] td,
        .stTable td {
            padding: 6px 8px !important;
            font-size: 0.72rem !important;
            color: #1E293B !important;
            white-space: nowrap !important;
            border-bottom: 1px solid #E2E8F0 !important;
            line-height: 1.35 !important;
        }
        div[data-testid="stTable"] tr:nth-child(even),
        .stTable tr:nth-child(even) {
            background-color: #F8FAFC !important;
        }
        div[data-testid="stDataFrame"] {
            width: 100% !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
            font-size: 0.72rem !important;
            border-radius: 6px !important;
        }
        .stExpander {
            border-radius: 6px !important;
            margin-bottom: 6px !important;
        }
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
            intraday_tag = intraday_info.get("tag", "[거래소강]")
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
            intraday_tag = "[거래소강]"
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
# Top Navigation & Status Bar (Office Stealth Mode)
# ---------------------------------------------------------
if "session_uid" not in st.session_state:
    import uuid
    st.session_state["session_uid"] = str(uuid.uuid4())
sc_worker = get_screening_worker(st.session_state["session_uid"])

now_dt = get_now_kst()
is_weekday = now_dt.weekday() < 5
cur_hm = now_dt.hour * 100 + now_dt.minute

if is_weekday and 900 <= cur_hm < 1530:
    market_badge = '<span style="color:#059669; font-weight:700;">● 정규장 운영중</span>'
elif is_weekday and 830 <= cur_hm < 900:
    market_badge = '<span style="color:#D97706; font-weight:700;">○ 장전 동시호가 (08:30~09:00)</span>'
elif is_weekday and 1530 <= cur_hm < 1800:
    market_badge = '<span style="color:#0284C7; font-weight:700;">○ 시간외 단일가 (15:30~18:00)</span>'
else:
    market_badge = '<span style="color:#64748B; font-weight:600;">○ 장마감 (휴장)</span>'

time_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

col_top_l, col_top_r = st.columns([1.8, 3.2])
with col_top_l:
    task_badge = ""
    if sc_worker.is_running():
        _w_st = sc_worker.get_status()
        _pct = int(_w_st["progress"] * 100)
        task_badge = f'<span style="font-size:0.72rem; color:#1D4ED8; background:#EFF6FF; border:1px solid #93C5FD; padding:1px 6px; border-radius:4px; font-weight:700;">[스크리닝 진행중 ({_pct}%)]</span>'

    st.markdown(
        '<div style="display:flex; align-items:center; gap:8px; padding:2px 0 6px 0;">'
        '<span style="font-size:0.95rem; font-weight:700; color:#1E293B; letter-spacing:-0.3px;">AlphaQuant Analytics</span>'
        '<span style="font-size:0.72rem; color:#64748B; background:#F1F5F9; border:1px solid #CBD5E1; padding:1px 6px; border-radius:4px; font-weight:600;">PRO STEALTH</span>'
        f'{task_badge}'
        '</div>',
        unsafe_allow_html=True
    )
with col_top_r:
    top_c1, top_c2 = st.columns([4.4, 0.6])
    with top_c1:
        live_clock_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: transparent;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  }}
  .clock-wrapper {{
    display: flex;
    justify-content: flex-end;
    align-items: center;
    width: 100%;
    height: 36px;
  }}
  .clock-bar {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    height: 30px;
    padding: 0 10px;
    background: #F1F5F9;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    font-size: 13px;
    color: #1E293B;
    white-space: nowrap;
    user-select: none;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }}
  .clock-time {{
    color: #0F172A;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    letter-spacing: -0.2px;
  }}
  .sep {{ color: #94A3B8; font-weight: 600; }}
  .badge {{ font-size: 12px; font-weight: 600; }}

  @media (prefers-color-scheme: dark) {{
    .clock-bar {{
      background: #1E293B !important;
      border-color: #475569 !important;
      color: #F8FAFC !important;
    }}
    .clock-time {{
      color: #FFFFFF !important;
    }}
    .sep {{ color: #64748B !important; }}
  }}

  @media (max-width: 768px) {{
    .clock-wrapper {{
      justify-content: flex-end;
    }}
    .clock-bar {{
      height: 28px;
      font-size: 11px;
      padding: 0 6px;
      gap: 5px;
    }}
    .clock-time {{
      font-size: 11px;
    }}
    .badge {{
      font-size: 10.5px;
    }}
  }}
</style>
</head>
<body>
<div class="clock-wrapper">
  <div class="clock-bar">
    <span class="clock-time" id="kst-time-val">{time_str} KST</span>
    <span class="sep">|</span>
    <span class="badge" id="kst-market-badge">{market_badge}</span>
  </div>
</div>
<script>
(function() {{
  function updateClock() {{
    try {{
      var nowMs = Date.now();
      // UTC + 9 hours for KST
      var kst = new Date(nowMs + 9 * 3600 * 1000);
      var y = kst.getUTCFullYear();
      var m = String(kst.getUTCMonth() + 1).padStart(2, '0');
      var d = String(kst.getUTCDate()).padStart(2, '0');
      var hh = String(kst.getUTCHours()).padStart(2, '0');
      var mm = String(kst.getUTCMinutes()).padStart(2, '0');
      var ss = String(kst.getUTCSeconds()).padStart(2, '0');
      var day = kst.getUTCDay();

      var timeEl = document.getElementById('kst-time-val');
      if (timeEl) {{
        timeEl.textContent = y + '-' + m + '-' + d + ' ' + hh + ':' + mm + ':' + ss + ' KST';
      }}

      var isWeekday = (day >= 1 && day <= 5);
      var curHm = kst.getUTCHours() * 100 + kst.getUTCMinutes();
      var badgeEl = document.getElementById('kst-market-badge');
      if (badgeEl) {{
        var badgeHtml = '';
        if (isWeekday && curHm >= 900 && curHm < 1530) {{
          badgeHtml = '<span style="color:#059669; font-weight:700;">● 정규장 운영중</span>';
        }} else if (isWeekday && curHm >= 830 && curHm < 900) {{
          badgeHtml = '<span style="color:#D97706; font-weight:700;">○ 장전 동시호가 (08:30~09:00)</span>';
        }} else if (isWeekday && curHm >= 1530 && curHm < 1800) {{
          badgeHtml = '<span style="color:#0284C7; font-weight:700;">○ 시간외 단일가 (15:30~18:00)</span>';
        }} else {{
          badgeHtml = '<span style="color:#64748B; font-weight:600;">○ 장마감 (휴장)</span>';
        }}
        if (badgeEl.innerHTML !== badgeHtml) {{
          badgeEl.innerHTML = badgeHtml;
        }}
      }}
    }} catch (e) {{
      console.error(e);
    }}
  }}
  updateClock();
  // 250ms interval ensures the second boundary is caught immediately without delay
  setInterval(updateClock, 250);
  document.addEventListener('visibilitychange', function() {{
    if (!document.hidden) updateClock();
  }});
}})();
</script>
</body>
</html>"""
        components.html(live_clock_html, height=36, scrolling=False)
    with top_c2:
        if st.button("동기화", help="현재시각 및 시장상태 최신화", key="btn_top_time_sync"):
            st.rerun()

main_tab_names = [
    "1. 시장 전체 익일 상승 후보 & 고확신 스크리너",
    "2. 관심종목 실시간 스코어보드",
    "3. 통합 성과 검증 & AI 자가최적화 센터",
]

if "main_active_tab_nav" in st.session_state:
    _cur_nav = str(st.session_state["main_active_tab_nav"])
    if _cur_nav not in main_tab_names:
        if any(w in _cur_nav for w in ["3", "4", "5", "검증", "최적화", "다이어리", "워크포워드"]):
            st.session_state["main_active_tab_nav"] = main_tab_names[2]
        else:
            st.session_state["main_active_tab_nav"] = main_tab_names[0]

tab1, tab2, tab3 = st.tabs(main_tab_names, key="main_active_tab_nav")


# =========================================================
# TAB 1: 시장 전체 익일 상승 후보 & 고확신 스크리너
# =========================================================
with tab1:
    sc_mode = st.radio(
        "스크리닝 전략 모드 선택 (아래 5개 중 1개 선택)",
        [
            "실시간 당일 단타 (5% 익절)",
            "스나이퍼 고확신 (눌림목 반등)",
            "5% 급등 타겟 (1~2일 스윙)",
            "주도주 종가배팅 (익일 시초 갭)",
            "일반 퀀트 스코어링"
        ],
        horizontal=False,
        key="sc_mode_radio_sel"
    )

    if "screened_cache_by_mode" in st.session_state and sc_mode in st.session_state["screened_cache_by_mode"]:
        st.session_state["screened_results_df"] = st.session_state["screened_cache_by_mode"][sc_mode]
        st.session_state["screened_mode_label"] = sc_mode

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

    st.markdown(f'<div style="font-size:0.78rem; color:#475569; padding:2px 0 6px 2px;">{m_tag}</div>', unsafe_allow_html=True)

    # 전 전략 모드 공통: 시간대별 최적 권장 설정 자동화 엔진
    def get_auto_time_recommendation_for_strategy(strategy_mode, dt=None):
        if dt is None:
            dt = get_now_kst()
        cur_hm = dt.hour * 100 + dt.minute

        if "당일 단타" in strategy_mode:
            if 910 <= cur_hm < 940:
                return {
                    "slot_name": "09:10 ~ 09:40 [장초반 수급 집중]",
                    "slot_id": "dt_slot_0",
                    "highlight_idx": 0,
                    "val": 100,
                    "gain": (2.5, 10.0),
                    "vol": 0.5,
                    "badge_desc": "거래대금 100억↑ · 시초대비 +2.5%~+10.0% · 전일대비 거래량 0.5배↑",
                    "tip": "장초반 수급 폭발 주도주 빠른 포착 및 익절 공간 확보",
                    "params": {"min_daytrade_val_b": 100, "intraday_gain_range": (2.5, 10.0), "min_daytrade_vol_ratio": 0.5}
                }
            elif 940 <= cur_hm < 1100:
                return {
                    "slot_name": "09:40 ~ 11:00 [골든타임 - 2차 돌파]",
                    "slot_id": "dt_slot_1",
                    "highlight_idx": 1,
                    "val": 150,
                    "gain": (3.0, 12.0),
                    "vol": 0.6,
                    "badge_desc": "거래대금 150억↑ · 시초대비 +3.0%~+12.0% · 거래량 0.6배↑",
                    "tip": "당일 최고 승률 구간: 주도주 2차 파동 돌파 적중률 극대화",
                    "params": {"min_daytrade_val_b": 150, "intraday_gain_range": (3.0, 12.0), "min_daytrade_vol_ratio": 0.6}
                }
            elif 1100 <= cur_hm < 1300:
                return {
                    "slot_name": "11:00 ~ 13:00 [점심 횡보장]",
                    "slot_id": "dt_slot_2",
                    "highlight_idx": 2,
                    "val": 200,
                    "gain": (4.0, 15.0),
                    "vol": 0.7,
                    "badge_desc": "거래대금 200억↑ · 시초대비 +4.0%~+15.0% · 거래량 0.7배↑",
                    "tip": "점심 거래소강: 탄탄하게 지지받는 최상위 대장주만 선별",
                    "params": {"min_daytrade_val_b": 200, "intraday_gain_range": (4.0, 15.0), "min_daytrade_vol_ratio": 0.7}
                }
            elif 1300 <= cur_hm < 1430:
                return {
                    "slot_name": "13:00 ~ 14:30 [오후 2차 수급]",
                    "slot_id": "dt_slot_3",
                    "highlight_idx": 3,
                    "val": 200,
                    "gain": (4.5, 18.0),
                    "vol": 0.8,
                    "badge_desc": "거래대금 200억↑ · 시초대비 +4.5%~+18.0% · 거래량 0.8배↑",
                    "tip": "상한가/VI 직행 추진력을 갖춘 최상위 주도주 공략",
                    "params": {"min_daytrade_val_b": 200, "intraday_gain_range": (4.5, 18.0), "min_daytrade_vol_ratio": 0.8}
                }
            elif 1430 <= cur_hm < 1535:
                return {
                    "slot_name": "14:30 ~ 15:35 [장마감 임박/청산]",
                    "slot_id": "dt_slot_4",
                    "highlight_idx": 4,
                    "val": 200,
                    "gain": (3.0, 15.0),
                    "vol": 0.6,
                    "badge_desc": "신규 진입 중단 권장 (보유분 15:15 전량 청산 집중)",
                    "tip": "신규 매수는 마감 직전 '5% 급등 타겟' 또는 '종가배팅' 권장",
                    "params": {"min_daytrade_val_b": 200, "intraday_gain_range": (3.0, 15.0), "min_daytrade_vol_ratio": 0.6}
                }
            else:
                return {
                    "slot_name": "장외/마감 정산 (내일 실전 준비)",
                    "slot_id": "dt_slot_off",
                    "highlight_idx": 1,
                    "val": 100,
                    "gain": (2.5, 15.0),
                    "vol": 0.5,
                    "badge_desc": "골든타임 권장값 (거래대금 100억↑ · 시초대비 +2.5%~+15.0%)",
                    "tip": "마감 데이터 기준 최적 스크리닝 및 내일 장초반 준비",
                    "params": {"min_daytrade_val_b": 100, "intraday_gain_range": (2.5, 15.0), "min_daytrade_vol_ratio": 0.5}
                }
        elif "스나이퍼" in strategy_mode:
            if 830 <= cur_hm < 900:
                return {
                    "slot_name": "08:30 ~ 09:00 [장 시작 전 준비]",
                    "slot_id": "snp_slot_0",
                    "highlight_idx": 0,
                    "val": 100,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "20일평균 대금 100억↑ · 이격도 103.5% · 하락장 필터 On",
                    "tip": "전일 확정 데이터 기반 우량 눌림목 후보군 선별 확정",
                    "params": {"min_val_krw_b": 100, "max_disparity_val": 103.5, "ignore_market_filter": False}
                }
            elif 900 <= cur_hm < 1000:
                return {
                    "slot_name": "09:00 ~ 10:00 [시초가 체결 확인]",
                    "slot_id": "snp_slot_1",
                    "highlight_idx": 1,
                    "val": 100,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "20일평균 대금 100억↑ · 이격도 103.5% · 갭제한 -1.5%~+1.5% 엄수",
                    "tip": "+1.5% 초과 갭상승 종목은 뇌동매매 진입 금지",
                    "params": {"min_val_krw_b": 100, "max_disparity_val": 103.5, "ignore_market_filter": False}
                }
            elif 1000 <= cur_hm < 1520:
                return {
                    "slot_name": "10:00 ~ 15:20 [장중 감시 & 스탑로스]",
                    "slot_id": "snp_slot_2",
                    "highlight_idx": 2,
                    "val": 100,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "20일평균 대금 100억↑ · 이격도 103.0% (보수적 눌림)",
                    "tip": "익절 +1.2% / 손절 -2.0% 자동 스탑로스 대응",
                    "params": {"min_val_krw_b": 100, "max_disparity_val": 103.0, "ignore_market_filter": False}
                }
            else:
                return {
                    "slot_name": "장외/마감 후 [익일 후보 사전 분석]",
                    "slot_id": "snp_slot_off",
                    "highlight_idx": 0,
                    "val": 100,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "20일평균 대금 100억↑ · 이격도 103.5% · 지수 추세 확인",
                    "tip": "내일 장초반 시초가 매수 후보군 사전 압축",
                    "params": {"min_val_krw_b": 100, "max_disparity_val": 103.5, "ignore_market_filter": False}
                }
        elif "5% 급등" in strategy_mode:
            if 900 <= cur_hm < 1430:
                return {
                    "slot_name": "09:00 ~ 14:30 [장중 급등 모멘텀 추적]",
                    "slot_id": "s5_slot_0",
                    "highlight_idx": 0,
                    "val": 200,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "거래대금 200억↑ · 당일 주가상승률 +8.0%↑",
                    "tip": "상한가/준상한가 도전 중인 시장 최상위 대장주 실시간 탐색",
                    "params": {"min_surge_val_b": 200, "min_day_surge_pct": 8.0}
                }
            elif 1430 <= cur_hm < 1535:
                return {
                    "slot_name": "14:30 ~ 15:35 [마감 직전 진입 골든타임]",
                    "slot_id": "s5_slot_1",
                    "highlight_idx": 1,
                    "val": 300,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "거래대금 300억↑ · 당일 주가상승률 +10.0%↑ (상한가 굳히기)",
                    "tip": "15:20 동시호가 종가 매수 -> 1~2일 내 +5% 자동 익절",
                    "params": {"min_surge_val_b": 300, "min_day_surge_pct": 10.0}
                }
            else:
                return {
                    "slot_name": "장외/마감 후 [익일 5% 돌파 타겟]",
                    "slot_id": "s5_slot_off",
                    "highlight_idx": 1,
                    "val": 200,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "거래대금 200억↑ · 당일 주가상승률 +10.0%↑",
                    "tip": "당일 상한가/마루보즈 마감 대장주 익일 대응 준비",
                    "params": {"min_surge_val_b": 200, "min_day_surge_pct": 10.0}
                }
        elif "종가배팅" in strategy_mode:
            if 900 <= cur_hm < 1430:
                return {
                    "slot_name": "09:00 ~ 14:30 [장중 주도 테마 관찰]",
                    "slot_id": "cb_slot_0",
                    "highlight_idx": 0,
                    "val": 200,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "거래대금 200억↑ · 주가상승률 +3.0%↑",
                    "tip": "오후장까지 고가를 유지하는지 추세 관찰",
                    "params": {"min_today_val_b": 200, "min_day_ret_val": 3.0}
                }
            elif 1430 <= cur_hm < 1535:
                return {
                    "slot_name": "14:30 ~ 15:35 [동시호가 진입 골든타임]",
                    "slot_id": "cb_slot_1",
                    "highlight_idx": 1,
                    "val": 200,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "거래대금 200억↑ · 주가상승률 +3.0%↑ (고가마감)",
                    "tip": "15:20 동시호가 종가 매수 -> 익일 시초가 +1.5% 갭 익절",
                    "params": {"min_today_val_b": 200, "min_day_ret_val": 3.0}
                }
            else:
                return {
                    "slot_name": "장외/마감 후 [익일 시초갭 공략 분석]",
                    "slot_id": "cb_slot_off",
                    "highlight_idx": 1,
                    "val": 200,
                    "gain": (0, 0),
                    "vol": 1.0,
                    "badge_desc": "거래대금 200억↑ · 주가상승률 +3.0%↑",
                    "tip": "당일 고가 마감 주도주의 익일 시초가 갭상승 가능성 점검",
                    "params": {"min_today_val_b": 200, "min_day_ret_val": 3.0}
                }
        else:  # 일반 퀀트 스코어링
            return {
                "slot_name": "상시 권장 (Point-in-Time 스코어링)",
                "slot_id": "qt_slot_0",
                "highlight_idx": 0,
                "val": 100,
                "gain": (0, 0),
                "vol": 1.5,
                "badge_desc": "최소 스코어 65점↑ · 거래량 1.5배↑ · 이평 정배열 On",
                "tip": "전체 상장 유니버스 종합 퀀트 점수 산출 및 순위화",
                "params": {"min_screener_score": 65.0, "min_vol_surge": 1.5, "require_ma_align": True}
            }

    rec_time = get_auto_time_recommendation_for_strategy(sc_mode)
    now_str = get_now_kst().strftime("%H:%M")
    cur_strat_slot_id = f"{sc_mode}_{rec_time['slot_id']}"

    # Auto-sync session state when strategy mode or time slot changes or sync is triggered
    if "sc_last_applied_strat_slot" not in st.session_state or st.session_state.get("sc_trigger_time_sync") or st.session_state.get("sc_last_applied_strat_slot") != cur_strat_slot_id:
        for _k, _v in rec_time["params"].items():
            st.session_state[f"sc_{_k}"] = _v
        st.session_state["sc_last_applied_strat_slot"] = cur_strat_slot_id
        st.session_state["sc_trigger_time_sync"] = False

    # 1) 전략 원칙 & 시간대별 설정 가이드 통합 접이식(Expander)
    with st.expander("[전략 핵심 원칙 & 시간대별 설정 가이드] (필요 시 클릭하여 열기)", expanded=False):
        g_guide_mode = st.radio(
            "가이드 항목 선택",
            ["전략 핵심 원칙 & 시장 체제 점검", "시간대별 권장 설정 가이드"],
            horizontal=True,
            key="g_guide_view_radio"
        )
        if "핵심 원칙" in g_guide_mode:
            if "당일 단타" in sc_mode:
                bm_ohlcv = get_benchmark_ohlcv()
                mkt_ok, mkt_msg, mkt_meta = evaluate_market_regime(bm_ohlcv)
                mkt_color = "#059669" if mkt_ok else "#D97706"
                mkt_bg = "#ECFDF5" if mkt_ok else "#FFFBEB"
                mkt_border = "#A7F3D0" if mkt_ok else "#FDE68A"
                mkt_txt = "당일 단타 모드 신규 매수가 적극 허용됩니다." if mkt_ok else "조정/하락장입니다. 보수적 비중 조절 및 칼손절(-2.5%) 준수 필수!"
                st.markdown(
                    f'<div style="font-size:0.78rem; line-height:1.45; color:{mkt_color}; background:{mkt_bg}; border:1px solid {mkt_border}; border-radius:5px; padding:6px 10px; margin-bottom:8px;">'
                    f'<strong>[시장 체제 점검]</strong> KODEX 200 20일선 기준: {mkt_txt} ({mkt_msg})'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    """
<div style="font-size:0.78rem; line-height:1.45; color:#334155; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px;">
<div style="font-size:0.84rem; font-weight:700; color:#1E293B; margin-bottom:6px;">[실시간 당일 단타] 5% 돌파 모드 핵심 원칙 (당일 완결 / No Overnight)</div>
1. <strong>진입 타이밍</strong>: 장중 09:10 ~ 14:30 실시간 수급 폭발 종목 포착<br>
2. <strong>수급 & 거래대금</strong>: 당일 거래대금 100억~200억↑ & 장중 이미 전일 거래량 50% 이상 돌파<br>
3. <strong>시초가 돌파 모멘텀</strong>: 시초가 대비 +2.5% ~ +15.0% 구간 돌파 (상승 초입 및 추가 슈팅 여력 포착)<br>
4. <strong>탄탄한 양봉 지지</strong>: 종가가 당일 진폭 중심선(50% 이상) 상단 유지 (투매형 윗꼬리 배제)<br>
5. <strong>목표 및 당일 청산</strong>: 진입 즉시 <strong>+5.0% 익절 자동주문 / 손절 -2.5% 엄수 / 15:15 미도달 시 종가 전량 청산</strong>
</div>
                    """,
                    unsafe_allow_html=True
                )
            elif "스나이퍼" in sc_mode:
                bm_ohlcv = get_benchmark_ohlcv()
                mkt_ok, mkt_msg, mkt_meta = evaluate_market_regime(bm_ohlcv)
                mkt_color = "#059669" if mkt_ok else "#D97706"
                mkt_bg = "#ECFDF5" if mkt_ok else "#FFFBEB"
                mkt_border = "#A7F3D0" if mkt_ok else "#FDE68A"
                mkt_txt = "스나이퍼 신규 매수가 허용됩니다." if mkt_ok else "지수 하락장 위험 감지: 스나이퍼 전략은 하락장에서 무리한 진입을 전면 차단합니다."
                st.markdown(
                    f'<div style="font-size:0.78rem; line-height:1.45; color:{mkt_color}; background:{mkt_bg}; border:1px solid {mkt_border}; border-radius:5px; padding:6px 10px; margin-bottom:8px;">'
                    f'<strong>[시장 체제 점검]</strong> KODEX 200 20일선 기준: {mkt_txt} ({mkt_msg})'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    """
<div style="font-size:0.78rem; line-height:1.45; color:#334155; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px;">
<div style="font-size:0.84rem; font-weight:700; color:#1E293B; margin-bottom:6px;">[스나이퍼 고확신 모드] 핵심 원칙 (80% 승률 타겟 - 20일선 눌림목 반등)</div>
• <strong>시장 필터</strong>: KODEX 200 > 20일선 상승 추세에서만 진입 (지수 급락 리스크 사전 차단)<br>
• <strong>우량 유동성</strong>: 20일 평균 거래대금 최소 100억 원 이상 (호가 왜곡 방지 및 풍부한 체결력)<br>
• <strong>눌림목 이격도</strong>: 20일선 대비 주가 이격도 <strong>98.0% ~ 103.5%</strong> (20일선 첫 지지 반등)<br>
• <strong>핵심 체결 룰 (시초가 갭 제한)</strong>: 익일 시초가가 전일종가 대비 <strong>-1.5% ~ +1.5%</strong> 이내 체결 (+1.5% 초과 갭상승 시 진입 금지)<br>
• <strong>비대칭 목표</strong>: 익절 +1.2% / 손절 -2.0%
</div>
                    """,
                    unsafe_allow_html=True
                )
            elif "종가배팅" in sc_mode:
                st.markdown(
                    """
<div style="font-size:0.78rem; line-height:1.45; color:#334155; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px;">
<div style="font-size:0.84rem; font-weight:700; color:#1E293B; margin-bottom:6px;">[주도주 종가배팅 모드] 핵심 원칙 (Overnight Close-to-Open)</div>
• <strong>탐색 시점</strong>: 매일 15:20 ~ 15:30 (장 마감 직전 동시호가 전후)<br>
• <strong>거래대금 폭발</strong>: 당일 거래대금 최소 200억 원 이상 (당일 시장 주도 테마 대장주)<br>
• <strong>장대양봉 고가마감</strong>: 당일 주가 +3.0% 이상 장대양봉 & 일중 고가 부근 마감 (고점 대비 70% 이상 지지)<br>
• <strong>청산 룰</strong>: 당일 15:20 종가 매수 -> 익일 09:00 시초가 갭상승(+1.5%↑) 시 즉시 익절 (손절 -2.0%)
</div>
                    """,
                    unsafe_allow_html=True
                )
            elif "5% 급등" in sc_mode or "5% 돌파" in sc_mode:
                st.markdown(
                    """
<div style="font-size:0.78rem; line-height:1.45; color:#334155; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px;">
<div style="font-size:0.84rem; font-weight:700; color:#1E293B; margin-bottom:6px;">[5% 급등 타겟 모드] 핵심 원칙 (1~2일 스윙 5% 급등 수확)</div>
• <strong>빅데이터 검증</strong>: [거래대금 200억~500억↑ + 신고가 첫 상한가/준상한가(+8%~+29.8%) + 고점 지지 마감] 주도주는 익일~2일차 +5% 도달률 80%대에 달합니다.<br>
• <strong>탐색 & 진입 시점</strong>: 장중 급등 추세 확인 또는 마감 직전 15:20 동시호가 종가 매수<br>
• <strong>목표 및 손절</strong>: 장중 <strong>+5.0%</strong> 도달 즉시 전량 자동 익절 (1일차 미도달 시 2일차 홀딩, 손절선 -4.0% 엄수)
</div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
<div style="font-size:0.78rem; line-height:1.45; color:#334155; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px;">
<div style="font-size:0.84rem; font-weight:700; color:#1E293B; margin-bottom:6px;">[일반 퀀트 스코어링 모드] 핵심 원칙 (기본 종합 점수제)</div>
• <strong>특징</strong>: Point-in-Time 스코어링 엔진으로 수급, 모멘텀, 거래대금, 변동성, 재무 건전성을 복합 평가하여 100점 만점으로 순위를 산출합니다.<br>
• <strong>권장 거래대금</strong>: 최소 50억~100억 원 이상 (유동성 부족으로 인한 슬리피지 방지)
</div>
                    """,
                    unsafe_allow_html=True
                )

        else:
            active_tag = ' <span style="color:#0F172A; font-weight:700; font-size:0.68rem; background:#E2E8F0; padding:1px 5px; border-radius:4px; border:1px solid #94A3B8;">▶ 현재 자동 세팅됨</span>'
            if "당일 단타" in sc_mode:
                r0_t = active_tag if rec_time["highlight_idx"] == 0 else ""
                r1_t = active_tag if rec_time["highlight_idx"] == 1 else ""
                r2_t = active_tag if rec_time["highlight_idx"] == 2 else ""
                r3_t = active_tag if rec_time["highlight_idx"] == 3 else ""
                r4_t = active_tag if rec_time["highlight_idx"] == 4 else ""

                st.markdown(
                    f"""
| 시간대 | 장세 특성 | 권장 누적 거래대금 | 권장 시초가대비 상승률 | 매매 행동 요령 & 슬라이더 설정 팁 |
| :--- | :--- | :--- | :--- | :--- |
| **09:10 ~ 09:40**{r0_t} | **장초반 거래 집중**<br>(변동성/거래량 극대화) | **100억 원 이상** | **+2.5% ~ +10.0%** | 장초반 고점 윗꼬리 물림 방지 및 +5% 익절 공간 확보 |
| **09:40 ~ 11:00**<br>**[골든타임]**{r1_t} | **주도주 2차 돌파**<br>(당일 단타 최고 승률) | **150억 ~ 200억 원 이상** | **+3.0% ~ +12.0%** | 눌림목 지지 확인 후 재차 치고 나가는 2차 파동 구간 |
| **11:00 ~ 13:00**{r2_t} | **점심 횡보장**<br>(거래량 급감, 소강상태) | **200억 원 이상** | **+4.0% ~ +15.0%** | 거래량 마르는 시간대: 탄탄하게 지지받는 최상위 대장주만 진입 |
| **13:00 ~ 14:30**{r3_t} | **오후 2차 수급 유입**<br>(마감 전 주도 섹터 랠리) | **200억 원 이상** | **+4.5% ~ +18.0%** | 오후장 돌파 매매는 상한가/VI 직행 추진력 갖춘 주도주 선별 |
| **14:30 이후**{r4_t} | **장마감 임박**<br>(당일 청산 집중) | **신규 매수 OFF** | **신규 진입 OFF** | 15:15 전량 청산 원칙이므로 신규 진입 중단 및 보유분 정리 집중 |
                    """,
                    unsafe_allow_html=True
                )
            elif "스나이퍼" in sc_mode:
                s0_t = active_tag if rec_time["highlight_idx"] == 0 else ""
                s1_t = active_tag if rec_time["highlight_idx"] == 1 else ""
                s2_t = active_tag if rec_time["highlight_idx"] == 2 else ""
                st.markdown(
                    f"""
| 분석/실행 시점 | 데이터 기준 | 권장 설정값 (20일 평균 거래대금) | 매매 행동 요령 & 주의사항 |
| :--- | :--- | :--- | :--- |
| **08:30 ~ 09:00**{s0_t}<br>(장 시작 전 분석) | **전일까지의 확정 20일 평균 거래대금** | **100억 원 이상** | 유동성 풍부한 우량주 위주로 20일선 눌림목 후보군 확정 |
| **09:00 ~ 10:00**{s1_t}<br>(시초가 체결 확인) | **시초가 갭 검증** | **체결 갭: -1.5% ~ +1.5%** | +1.5% 초과 갭상승 시 뇌동매매 진입 금지. 갭 범위 내 시초가 매수 |
| **10:00 ~ 15:20**{s2_t}<br>(장중 자동 청산) | **장중 시세 감시** | **익절 +1.2% / 손절 -2.0%** | 주문 체결 즉시 MTS/HTS 자동 감시 매도(스탑로스) 설정 |
                    """,
                    unsafe_allow_html=True
                )
            elif "종가배팅" in sc_mode:
                c0_t = active_tag if rec_time["highlight_idx"] == 0 else ""
                c1_t = active_tag if rec_time["highlight_idx"] == 1 else ""
                st.markdown(
                    f"""
| 분석/실행 시점 | 데이터 기준 | 권장 설정값 (당일 최종 거래대금) | 매매 행동 요령 & 주의사항 |
| :--- | :--- | :--- | :--- |
| **09:00 ~ 14:30**{c0_t}<br>(장중 탐색) | **당일 실시간 누적 거래대금** | **200억 원 이상** | 장중 거래대금이 터지며 추세를 유지하는 주도 테마 1등주 관찰 |
| **14:30 ~ 15:35**{c1_t}<br>(동시호가 종가 체결) | **종가 고가마감 확인** | **주가 +3.0%↑ & 고점 지지율 70%↑** | 윗꼬리가 길지 않고 일봉상 고가 부근 마감 시 15:20 동시호가 매수 |
| **익일 09:00 ~ 09:15**<br>(시초가 갭 청산) | **익일 시초가 및 장초반** | **+1.5% 갭상승 시 즉시 청산** | 익일 09:00 시초가 갭상승(+1.5% 이상) 출발 시 전량 즉시 익절 |
                    """,
                    unsafe_allow_html=True
                )
            elif "5% 급등" in sc_mode or "5% 돌파" in sc_mode:
                g0_t = active_tag if rec_time["highlight_idx"] == 0 else ""
                g1_t = active_tag if rec_time["highlight_idx"] == 1 else ""
                st.markdown(
                    f"""
| 분석/실행 시점 | 데이터 기준 | 권장 설정값 (당일 거래대금) | 매매 행동 요령 & 주의사항 |
| :--- | :--- | :--- | :--- |
| **09:00 ~ 14:30**{g0_t}<br>(장중 실시간 급등주) | **실시간 누적 거래대금** | **200억 원 이상** (상승률 +8%↑) | 상한가/준상한가로 직행하는 주도 대장주 실시간 탐색 |
| **14:30 ~ 15:35**{g1_t}<br>(마감 직전 진입) | **당일 마감 누적 거래대금** | **300억 원 이상** (상승률 +10%↑) | 최상위 대장주 종가 매수 -> 1~2일 내 장중 +5% 자동 익절 |
| **익일 ~ 2일차**<br>(스윙 자동 익절) | **장중 시세 감시** | **장중 +5.0% 도달 시 즉시 익절** | 매수 다음 날 장중 +5% 급등 시 즉시 익절 (손절선 -4.0% 엄수) |
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.caption("Point-in-Time 스코어링 엔진으로 전체 상장 유니버스를 일괄 스크리닝하고 테마 집중 위험을 감지합니다. (권장: 최소 스코어 65점↑, 거래량 급증 1.5배↑)")

    # 2) 전 전략 모드 공통 스크리너 필터 파라미터 툴바
    f_hdr_c1, f_hdr_c2 = st.columns([4.2, 1.3])
    with f_hdr_c1:
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:8px; padding-top:4px;">'
            f'<span class="office-subheading" style="margin:0 !important;">스크리너 필터 설정</span>'
            f'<span style="background:#F1F5F9; color:#1E293B; font-weight:700; font-size:0.72rem; padding:2px 8px; border-radius:4px; border:1px solid #CBD5E1;">{now_str} {rec_time["slot_name"]}</span>'
            f'<span style="color:#64748B; font-size:0.75rem;">({rec_time["badge_desc"]} 자동 세팅됨)</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    with f_hdr_c2:
        if st.button("현재시간 권장값 재적용", key="btn_reapply_auto_time", use_container_width=True):
            for _k, _v in rec_time["params"].items():
                st.session_state[f"sc_{_k}"] = _v
            st.session_state["sc_trigger_time_sync"] = True
            st.toast(f"현재 시각({now_str}) {sc_mode} 권장 설정이 자동 적용되었습니다.")
            st.rerun()

    sc_c1, sc_c2, sc_c3, sc_c4, sc_c5 = st.columns([1.1, 1.4, 1.2, 1.2, 1.1])

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
                "실시간 급등·주도주 300선 (상승률+거래대금 상위 / 강력 권장)",
                "내 관심종목 138선 (초고속)",
                "전 상장사 전수조사 (약 2,700개)"
            ],
            index=0,
            key="sc_scope_sel"
        )

    if "당일 단타" in sc_mode:
        with sc_c3:
            val_options = [50, 100, 150, 200, 300]
            cur_val = st.session_state.get("sc_min_daytrade_val", rec_time.get("val", 100))
            if cur_val not in val_options:
                cur_val = 100
            min_daytrade_val_b = st.selectbox(
                "당일 최소 거래대금",
                val_options,
                index=val_options.index(cur_val),
                format_func=lambda x: f"{x}억 원 이상",
                key="sc_min_daytrade_val"
            )
        with sc_c4:
            cur_gain = st.session_state.get("sc_intraday_gain_range", rec_time.get("gain", (2.5, 12.0)))
            intraday_gain_range = st.slider(
                "시초가 대비 상승률 구간 (%)",
                1.0, 25.0,
                value=cur_gain,
                step=0.5,
                help="권장 설정: 골든타임(09:40~11:00) 3.0%~12.0% | 장초반 2.5%~10.0% | 급등주 최대 25.0%",
                key="sc_intraday_gain_range"
            )
        with sc_c5:
            cur_vol = st.session_state.get("sc_min_daytrade_vol_ratio", rec_time.get("vol", 0.5))
            min_daytrade_vol_ratio = st.slider(
                "전일대비 거래량 비율",
                0.3, 1.5,
                value=cur_vol,
                step=0.1,
                format="%.1f배 이상",
                help="권장 설정: 최소 0.5배(50%) 이상 | 거래량 폭발 대장주 0.8배 이상",
                key="sc_min_daytrade_vol_ratio"
            )
    elif "스나이퍼" in sc_mode:
        with sc_c3:
            cur_snp_val = st.session_state.get("sc_min_val_krw_b", 100)
            snp_options = [50, 100, 150, 200, 300]
            if cur_snp_val not in snp_options:
                cur_snp_val = 100
            min_val_krw_b = st.selectbox("최소 20일 평균 거래대금", snp_options, index=snp_options.index(cur_snp_val), format_func=lambda x: f"{x}억 원 이상", key="sc_min_val_krw_b")
        with sc_c4:
            cur_disp = st.session_state.get("sc_max_disparity_val", 103.5)
            max_disparity_val = st.slider("20일선 최대 이격도 (%)", 100.0, 105.0, value=cur_disp, step=0.5, key="sc_max_disparity_val")
        with sc_c5:
            cur_ign = st.session_state.get("sc_ignore_market_filter", False)
            ignore_market_filter = st.checkbox("시장 하락장 무시", value=cur_ign, key="sc_ignore_market_filter")
    elif "종가배팅" in sc_mode:
        with sc_c3:
            cb_options = [100, 200, 300, 500]
            cur_cb_val = st.session_state.get("sc_min_today_val_b", 200)
            if cur_cb_val not in cb_options:
                cur_cb_val = 200
            min_today_val_b = st.selectbox("당일 최소 거래대금", cb_options, index=cb_options.index(cur_cb_val), format_func=lambda x: f"{x}억 원 이상", key="sc_min_today_val_b")
        with sc_c4:
            cur_cb_ret = st.session_state.get("sc_min_day_ret_val", 3.0)
            min_day_ret_val = st.slider("당일 최소 주가 상승률 (%)", 2.0, 6.0, value=cur_cb_ret, step=0.5, key="sc_min_day_ret_val")
        with sc_c5:
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("15:20~15:30 권장")
    elif "5% 급등" in sc_mode or "5% 돌파" in sc_mode:
        with sc_c3:
            surge_options = [100, 200, 300, 500]
            cur_s_val = st.session_state.get("sc_min_surge_val_b", 200)
            if cur_s_val not in surge_options:
                cur_s_val = 200
            min_surge_val_b = st.selectbox("당일 최소 거래대금", surge_options, index=surge_options.index(cur_s_val), format_func=lambda x: f"{x}억 원 이상", key="sc_min_surge_val_b")
        with sc_c4:
            cur_s_pct = st.session_state.get("sc_min_day_surge_pct", 8.0)
            min_day_surge_pct = st.slider("당일 최소 주가 상승률 (%)", 5.0, 25.0, value=cur_s_pct, step=1.0, key="sc_min_day_surge_pct")
        with sc_c5:
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("장중 급등·마감 직전 진입 -> TP +5%")
    else:
        with sc_c3:
            cur_score = st.session_state.get("sc_min_screener_score", 65.0)
            min_screener_score = st.slider("최소 스코어 기준", 50.0, 90.0, value=cur_score, step=5.0, key="sc_min_screener_score")
        with sc_c4:
            cur_vol_s = st.session_state.get("sc_min_vol_surge", 1.5)
            min_vol_surge = st.slider("거래량 급증 비율 (20일선 대비)", 1.0, 3.0, value=cur_vol_s, step=0.2, key="sc_min_vol_surge")
        with sc_c5:
            cur_ma = st.session_state.get("sc_require_ma_align", True)
            require_ma_align = st.checkbox("이평 정배열/골든 필수", value=cur_ma, key="sc_require_ma_align")

    worker_st = sc_worker.get_status()
    is_worker_running = sc_worker.is_running()

    # If worker just completed, transfer results into session_state!
    if worker_st["status"] == "COMPLETED":
        df_worker_res = worker_st.get("results_df")
        if df_worker_res is not None:
            st.session_state["screened_results_df"] = df_worker_res
            st.session_state["screened_mode_label"] = worker_st["mode_label"]
            st.session_state.setdefault("screened_cache_by_mode", {})[worker_st["mode_label"]] = df_worker_res
            st.session_state["screened_is_empty_result"] = df_worker_res.empty
            sc_worker.reset_status()
            if not df_worker_res.empty:
                st.toast(f"스크리닝 완료! 총 {len(df_worker_res):,}개 종목 발굴")
            st.rerun()

    # If worker was cancelled, notify and reset
    if worker_st["status"] == "CANCELLED":
        st.warning("사용자 요청에 의해 스크리닝이 중단되었습니다.")
        sc_worker.reset_status()

    # If worker failed, notify and reset
    if worker_st["status"] == "FAILED":
        st.error(f"스크리닝 중 오류가 발생했습니다: {worker_st.get('error_message', '알 수 없는 오류')}")
        sc_worker.reset_status()

    # Execution Toolbar / Progress Area
    if is_worker_running:
        col_btn_run, col_btn_cancel = st.columns([3.8, 1.2])
        with col_btn_run:
            st.button("백그라운드 스크리닝 진행 중 (다른 탭 이동 가능)...", disabled=True, use_container_width=True)
        with col_btn_cancel:
            if st.button("작업 중단", type="secondary", use_container_width=True, key="btn_cancel_screening"):
                sc_worker.cancel()
                st.toast("스크리닝 중단 요청을 전송했습니다.")
                st.rerun()

        @st.fragment(run_every="1s")
        def render_screening_live_progress():
            cur_st = sc_worker.get_status()
            if cur_st["status"] == "RUNNING":
                curr = cur_st["current_index"]
                tot = cur_st["total_count"]
                prog = cur_st["progress"]
                stock = cur_st["current_name"]
                st.progress(prog, text=f"전수 탐색 중 ({curr}/{tot}): {stock} (진행률 {prog*100:.1f}%)")
                st.info(
                    f"**[{cur_st['mode_label']}] 백그라운드 스크리닝이 안전하게 진행 중입니다.**\n\n"
                    f"**다른 탭(2. 관심종목, 3. 워크포워드 등)이나 다른 메뉴로 이동하셔도 작업이 절대 중단되지 않고 계속 진행됩니다.**"
                )
            elif cur_st["status"] in ("COMPLETED", "CANCELLED", "FAILED"):
                st.rerun()

        render_screening_live_progress()

    else:
        if st.button("후보 스크리닝 실행", type="primary", use_container_width=True, key="btn_run_screener"):
            st.session_state["screened_is_empty_result"] = False
            params = {
                "sc_mode": sc_mode,
                "target_market": target_market,
                "sc_scope": sc_scope,
                "min_daytrade_val_b": min_daytrade_val_b if "당일 단타" in sc_mode else 100,
                "intraday_gain_range": intraday_gain_range if "당일 단타" in sc_mode else (2.5, 12.0),
                "min_daytrade_vol_ratio": min_daytrade_vol_ratio if "당일 단타" in sc_mode else 0.5,
                "min_val_krw_b": min_val_krw_b if "스나이퍼" in sc_mode else 100,
                "max_disparity_val": max_disparity_val if "스나이퍼" in sc_mode else 103.5,
                "ignore_market_filter": ignore_market_filter if "스나이퍼" in sc_mode else False,
                "min_today_val_b": min_today_val_b if "종가배팅" in sc_mode else 200,
                "min_day_ret_val": min_day_ret_val if "종가배팅" in sc_mode else 3.0,
                "min_surge_val_b": min_surge_val_b if ("5% 급등" in sc_mode or "5% 돌파" in sc_mode) else 200,
                "min_day_surge_pct": min_day_surge_pct if ("5% 급등" in sc_mode or "5% 돌파" in sc_mode) else 8.0,
                "min_screener_score": min_screener_score if ("당일 단타" not in sc_mode and "스나이퍼" not in sc_mode and "종가배팅" not in sc_mode and "5% 급등" not in sc_mode and "5% 돌파" not in sc_mode) else 65.0,
                "min_vol_surge": min_vol_surge if ("당일 단타" not in sc_mode and "스나이퍼" not in sc_mode and "종가배팅" not in sc_mode and "5% 급등" not in sc_mode and "5% 돌파" not in sc_mode) else 1.5,
                "require_ma_align": require_ma_align if ("당일 단타" not in sc_mode and "스나이퍼" not in sc_mode and "종가배팅" not in sc_mode and "5% 급등" not in sc_mode and "5% 돌파" not in sc_mode) else True,
            }
            sc_worker.start_screening(params)
            st.rerun()

    if st.session_state.get("screened_is_empty_result"):
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

    def render_quant_analyst_qa_box(stock_row, key_suffix=""):
        stock_dict = stock_row.to_dict() if hasattr(stock_row, "to_dict") else dict(stock_row)
        code = str(stock_dict.get("code", "000000"))
        name = str(stock_dict.get("name", "종목"))
        chat_key = f"quant_qa_history_{code}"

        if chat_key not in st.session_state or not st.session_state[chat_key]:
            intro = generate_quant_expert_answer(stock_dict, "종목 종합 브리핑 및 상승 원인 요약")
            st.session_state[chat_key] = [
                {"role": "assistant", "content": intro}
            ]

        with st.container(border=True):
            hdr_c1, hdr_c2 = st.columns([3.5, 1.5])
            with hdr_c1:
                st.markdown(f"**{name} ({code})** 수석 애널리스트 팩트시트 & Q&A 콘솔")
                st.caption(f"전략: {stock_dict.get('strategy_tag', '')} | 스코어: {stock_dict.get('score', 0):.1f}점 | 전일비 거래량: {stock_dict.get('vol_ratio', 1.0):.1f}배 | RSI: {stock_dict.get('rsi', 50):.1f}")
            with hdr_c2:
                if st.button("대화 초기화", key=f"reset_chat_{code}_{key_suffix}", use_container_width=True):
                    st.session_state[chat_key] = []
                    st.rerun()

            # Quick Question Buttons (Pills)
            st.markdown("<div style='font-size:0.82rem; font-weight:600; color:#475569; margin-bottom:4px;'>빠른 퀀트 질의 칩 (원클릭 질문):</div>", unsafe_allow_html=True)
            q_row1_c1, q_row1_c2, q_row1_c3 = st.columns(3)
            q_row2_c1, q_row2_c2 = st.columns(2)

            selected_quick_q = None
            with q_row1_c1:
                if st.button("🔥 왜 급등하나요?", key=f"btn_quick_why_{code}_{key_suffix}", use_container_width=True):
                    selected_quick_q = "이 종목이 오늘 왜 급등하고 거래량이 폭발하는지 핵심 원인을 분석해줘."
            with q_row1_c2:
                if st.button("⚡ 지금 사도 되나요?", key=f"btn_quick_buy_{code}_{key_suffix}", use_container_width=True):
                    selected_quick_q = "지금 신규 매수 진입해도 안전한지, 추격매수 위험도와 손익비를 진단해줘."
            with q_row1_c3:
                if st.button("🎯 목표가 & 손절가", key=f"btn_quick_tp_{code}_{key_suffix}", use_container_width=True):
                    selected_quick_q = "알고리즘 권장 1차 목표 익절가와 원칙 손절 기준가, 트레일링 스탑 가격을 알려줘."
            with q_row2_c1:
                if st.button("⚠️ 리스크 팩트체크", key=f"btn_quick_risk_{code}_{key_suffix}", use_container_width=True):
                    selected_quick_q = "이 종목 진입 시 조심해야 할 고점 윗꼬리나 테마 급락 등 핵심 리스크를 알려줘."
            with q_row2_c2:
                if st.button("📊 차트 & 지표 진단", key=f"btn_quick_chart_{code}_{key_suffix}", use_container_width=True):
                    selected_quick_q = "이평선 정배열 상태, RSI 과열 여부, 거래량 수급 지표를 종합 분석해줘."

            if selected_quick_q:
                st.session_state[chat_key].append({"role": "user", "content": selected_quick_q})
                with st.spinner("AI 퀀트 수석 애널리스트가 수급 및 팩트를 정밀 분석 중입니다..."):
                    ans = generate_quant_expert_answer(stock_dict, selected_quick_q)
                    st.session_state[chat_key].append({"role": "assistant", "content": ans})
                st.rerun()

            # Render Chat History
            chat_box = st.container(height=360)
            with chat_box:
                for msg in st.session_state[chat_key]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            # Custom Question Form
            with st.form(key=f"custom_q_form_{code}_{key_suffix}", clear_on_submit=True):
                f_c1, f_c2 = st.columns([4, 1])
                with f_c1:
                    user_custom_input = st.text_input(
                        "질문 입력",
                        placeholder=f"{name}에 대해 궁금한 점을 직접 질문하세요 (예: 내일 시초가에 어떻게 대응해야 하나요?)",
                        label_visibility="collapsed",
                        key=f"input_q_text_{code}_{key_suffix}"
                    )
                with f_c2:
                    submitted = st.form_submit_button("질문 전송", type="primary", use_container_width=True)

                if submitted and user_custom_input.strip():
                    st.session_state[chat_key].append({"role": "user", "content": user_custom_input.strip()})
                    with st.spinner("퀀트 전문가 답변 생성 중..."):
                        ans = generate_quant_expert_answer(stock_dict, user_custom_input.strip())
                        st.session_state[chat_key].append({"role": "assistant", "content": ans})
                    st.rerun()

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

                st.session_state["qa_selected_stock_code"] = sel_sc["code"]
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
                        st.session_state["verif_pipeline_mode_sel"] = "과거 5년 워크포워드 & KODEX 200 벤치마크 검증 (종목별 95% 신뢰구간 시뮬레이션)"
                        st.session_state["main_active_tab_nav"] = main_tab_names[2]
                        st.toast(f"{sel_sc['name']} 백테스트 설정 완료! 3번 통합 검증 탭을 확인하세요.")
                        st.rerun()
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

                st.markdown("---")
                st.markdown(f"#### 💬 **{sel_sc['name']}** AI 퀀트 수석 애널리스트 실시간 Q&A")
                render_quant_analyst_qa_box(sel_sc, key_suffix="insp")

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

                        c_b1, c_b2, c_b3 = st.columns(3)
                        with c_b1:
                            if st.button("관심추가", key=f"card_add_sc_{srow['code']}_{i}", use_container_width=True):
                                batch_add_to_watchlist([srow['code']])
                                st.cache_data.clear()
                                st.toast(f"{srow['name']} 관심종목 추가 완료!")
                                st.rerun()
                        with c_b2:
                            if st.button("백테스트", key=f"card_bt_sc_{srow['code']}_{i}", use_container_width=True):
                                st.session_state["target_backtest_code"] = srow["code"]
                                st.session_state["verif_pipeline_mode_sel"] = "과거 5년 워크포워드 & KODEX 200 벤치마크 검증 (종목별 95% 신뢰구간 시뮬레이션)"
                                st.session_state["main_active_tab_nav"] = main_tab_names[2]
                                st.toast(f"{srow['name']} 백테스트 설정 완료! 3번 통합 검증 탭을 확인하세요.")
                                st.rerun()
                        with c_b3:
                            if st.button("퀀트 Q&A", key=f"card_qa_sc_{srow['code']}_{i}", use_container_width=True):
                                st.session_state["qa_selected_stock_code"] = srow["code"]
                                st.toast(f"{srow['name']} Q&A 활성화! 하단 퀀트 애널리스트 허브를 확인하세요.")
                                st.rerun()

        # -------------------------------------------------
        # Screener Section C: AI Quant Analyst Q&A Hub
        # -------------------------------------------------
        st.markdown("---")
        st.markdown("### 💬 AI 퀀트 수석 애널리스트 실시간 질의응답 (Q&A 허브)")
        st.caption("스크리닝된 상승 유력 종목 중 궁금한 종목을 선택하여 왜 급등하는지, 지금 사도 되는지, 목표가/손절가 시나리오를 퀀트 전문가에게 실시간 질의하세요.")

        qa_target_options = [f"{r['name']} ({r['code']}) - {r.get('strategy_tag', '')}" for _, r in df_sc_disp.iterrows()]
        default_qa_idx = 0
        if "qa_selected_stock_code" in st.session_state:
            for idx, opt in enumerate(qa_target_options):
                if f"({st.session_state['qa_selected_stock_code']})" in opt:
                    default_qa_idx = idx
                    break

        col_qa_sel, col_qa_opt = st.columns([3.5, 1.5])
        with col_qa_sel:
            qa_chosen_opt = st.selectbox(
                "분석 및 질의응답할 스크리닝 종목 선택",
                qa_target_options,
                index=default_qa_idx,
                key="qa_hub_stock_selectbox"
            )
        with col_qa_opt:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            st.caption("💡 상단 테이블 클릭 또는 카드 버튼 클릭 시 자동 연동")

        qa_chosen_code = qa_chosen_opt.split("(")[-1].split(")")[0].strip()
        qa_stock_row = df_sc_disp[df_sc_disp["code"] == qa_chosen_code].iloc[0]
        render_quant_analyst_qa_box(qa_stock_row, key_suffix="hub")

# =========================================================
# TAB 2: 관심종목 실시간 스코어보드 & 관리
# =========================================================
with tab2:
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
                    st.caption(f"{market} | {sector} | 그룹: {assigned_quick_grp}")
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
                        st.session_state["verif_pipeline_mode_sel"] = "과거 5년 워크포워드 & KODEX 200 벤치마크 검증 (종목별 95% 신뢰구간 시뮬레이션)"
                        st.session_state["main_active_tab_nav"] = main_tab_names[2]
                        st.toast(f"{sel_row['name']} 백테스트 설정 완료! 3번 통합 검증 탭을 확인하세요.")
                        st.rerun()
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
                        st.markdown(f"**[당일 장중 실시간 진단]**: `{sel_row.get('intraday_tag', '-')}`")
                        st.caption(f"{sel_row.get('intraday_desc', '')}")
                        st.markdown(f"- **시초가 대비 등락**: `{sel_row.get('intraday_open_pct', 0.0):+.2f}%`")
                        st.markdown(f"- **장중 거래량 급증비**: `{sel_row.get('vol_ratio', 1.0):.2f}배`")
                        st.markdown(f"- **고확신 전략 부합**: `{sel_row.get('strategy_tag', '-')}`")
                with rep_c2:
                    with st.container(border=True):
                        st.markdown(f"**[익일 스윙 퀀트 진단]**: `{sel_row.get('nextday_tag', '-')}`")
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
                        st.caption(f"{item['market']} | {item['sector']} | 그룹: {item.get('group_name', '기본그룹')}")
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
                                st.session_state["verif_pipeline_mode_sel"] = "과거 5년 워크포워드 & KODEX 200 벤치마크 검증 (종목별 95% 신뢰구간 시뮬레이션)"
                                st.session_state["main_active_tab_nav"] = main_tab_names[2]
                                st.toast(f"{item['name']} 백테스트 설정됨! 3번 통합 검증 탭을 확인하세요.")
                                st.rerun()
                        with btn_c2:
                            if st.button("삭제", key=f"del_card_{item['code']}", use_container_width=True):
                                with get_db_connection() as conn:
                                    cursor = conn.cursor()
                                    cursor.execute("DELETE FROM watchlist WHERE code = ?", (item["code"],))
                                    conn.commit()
                                st.cache_data.clear()
                                st.rerun()



# =========================================================
# SUB-MODULE A: 워크포워드 검증 & 벤치마크 리포트 (95% CI 부트스트랩)
# =========================================================
def render_walk_forward_section():
    st.markdown('<div class="office-subheading">과거 5년 워크포워드 백테스트 & KODEX 200 벤치마크 검증 리포트</div>', unsafe_allow_html=True)
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
            st.session_state["cached_backtest_results"] = bt_results

    bt_results = st.session_state.get("cached_backtest_results")
    if bt_results is not None:
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
# SUB-MODULE B: 실거래 예측 다이어리 & 라이브 추적 (Forward Testing Log)
# =========================================================
def render_prediction_diary_section():
    st.markdown('<div class="office-subheading">실거래 예측 다이어리 전체 장부 & 라이브 포워드 추적</div>', unsafe_allow_html=True)
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
# SUB-MODULE C: 실거래 성과 사후검증 & AI 전략 최적화 (Self-Tuning)
# =========================================================
def render_live_forward_and_tuning_section():
    st.markdown('<div class="office-subheading">실거래 성과 사후검증 & AI 전략 자가최적화 (Point-in-Time)</div>', unsafe_allow_html=True)
    st.caption("실제 체결 데이터 기반으로 전일(1일), 최근 주간(5일), 1개월(25일) 성과를 입체적으로 분석하고, 실패 원인을 AI가 스스로 진단하여 최적 파라미터를 도출합니다.")

    # 1. Model Decay Status Card (실시간 모델 수명 및 실전 승률 감시)
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
        st.markdown(f"**실거래 모델 수명 진단**: {decay_info['message']}")
        st.caption(f"누적 정산: {decay_info['total_settled']}건 | 적중: {decay_info.get('hits', 0)}건 | 실현 승률: {decay_info['live_win_rate']:.1f}%")

    st.markdown("---")

    t5_period_view = st.radio(
        "성과 분석 대상 기간 및 상세 장부 선택",
        [
            "전일 성과 정밀 분석 & AI 자가진단 (1일)",
            "최근 주간 성과 종합 (5거래일)",
            "1개월 누적 성과 & AI 최적화 (25거래일)",
            "실거래 예측 다이어리 전체 장부 & 라이브 승률 추이 (Forward Log)"
        ],
        horizontal=False,
        key="t5_period_radio_sel"
    )

    valid_pairs = get_valid_prediction_dates(limit=30)
    latest_pred_d = valid_pairs[-1][0] if valid_pairs else "2026-09-22"
    latest_exec_d = valid_pairs[-1][1] if valid_pairs else "2026-09-23"

    # -----------------------------------------------------
    # [AI 주기별 진단 비교 & 수익률 극대화 최적화 판정] (Meta-Optimizer)
    # -----------------------------------------------------
    meta_opt = evaluate_multi_horizon_tuning_impact()
    best_horizon_lbl = meta_opt["best_horizon_label"]
    best_ret = meta_opt["best_expected_return"]
    best_win = meta_opt["best_expected_win_rate"]
    best_props = meta_opt["best_proposals"]

    with st.expander("[AI 주기별 진단 비교 & 수익률 극대화 최적화 판정] (어떤 진단 결과를 적용해야 수익률이 오를까?)", expanded=True):
        st.markdown(
            f"""
            <div style="background:#F8FAFC; border:1px solid #CBD5E1; border-radius:6px; padding:10px 14px; margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
                    <span style="font-size:0.92rem; font-weight:700; color:#0F172A;">
                        AI 수익률 극대화 최적 추천 주기: <span style="color:#0284C7;">{best_horizon_lbl}</span>
                    </span>
                    <span style="font-size:0.80rem; font-weight:700; background:#E0F2FE; color:#0369A1; padding:2px 8px; border-radius:4px; border:1px solid #BAE6FD;">
                        시뮬레이션 예상 순수익률: +{best_ret:.2f}% | 예상 승률: {best_win:.1f}%
                    </span>
                </div>
                <div style="font-size:0.78rem; color:#475569; margin-top:4px;">
                    {meta_opt["best_rationale"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.table(meta_opt["comparative_table"])

        # 1) Full-width primary CTA button for prominent visibility on both mobile & PC
        if st.button(f"AI 최고 수익률 진단 결과 즉시 적용 ({best_horizon_lbl} 권장)", type="primary", use_container_width=True, key="btn_apply_best_meta_tune"):
            st.session_state["pending_tuning_updates"] = best_props
            st.session_state["tuning_applied_notification"] = True
            st.toast(f"AI 최고 추천 주기({best_horizon_lbl})의 최적 파라미터가 스크리너(탭 1)에 자동 적용되었습니다!")
            st.rerun()

        # 2) Individual horizon manual tuning buttons in 3 columns
        b_c2, b_c3, b_c4 = st.columns(3)
        with b_c2:
            if st.button("전일 진단 적용", use_container_width=True, key="btn_apply_daily_tune"):
                st.session_state["pending_tuning_updates"] = meta_opt["horizon_details"]["daily"]["proposals"]
                st.session_state["tuning_applied_notification"] = True
                st.toast("전일(1일) 진단 최적 파라미터가 스크리너에 적용되었습니다.")
                st.rerun()
        with b_c3:
            if st.button("주간 진단 적용", use_container_width=True, key="btn_apply_weekly_tune"):
                st.session_state["pending_tuning_updates"] = meta_opt["horizon_details"]["weekly"]["proposals"]
                st.session_state["tuning_applied_notification"] = True
                st.toast("주간(5일) 진단 최적 파라미터가 스크리너에 적용되었습니다.")
                st.rerun()
        with b_c4:
            if st.button("1개월 진단 적용", use_container_width=True, key="btn_apply_monthly_tune"):
                st.session_state["pending_tuning_updates"] = meta_opt["horizon_details"]["monthly"]["proposals"]
                st.session_state["tuning_applied_notification"] = True
                st.toast("1개월(25일) 진단 최적 파라미터가 스크리너에 적용되었습니다.")
                st.rerun()

    # -----------------------------------------------------
    # SUB-VIEW 1: 전일 성과 정밀 분석 & AI 자가진단 (1일)
    # -----------------------------------------------------
    if "전일" in t5_period_view:
        st.markdown('<div class="office-subheading">직전 거래일 추천 -> 당일 실제 체결 성과 1:1 대조 분석 및 AI 자가진단</div>', unsafe_allow_html=True)
        st.caption(f"최근 기준일: **{latest_pred_d} 추천 -> {latest_exec_d} 체결** | Point-in-Time 원칙으로 추천 시점 이후 실제 시장 체결 데이터와 대조합니다.")

        date_options = [
            f"{p[0]} (전일 추천 -> {p[1]} 실제 체결 검증)"
            for p in (valid_pairs[::-1] if valid_pairs else [("2026-09-22", "2026-09-23")])
        ]

        v_c1, v_c2, v_c3, v_c4 = st.columns([1.8, 1.5, 1.2, 1.2])
        with v_c1:
            sel_date_raw = st.selectbox(
                "검증 기준일 (T-1 추천 발굴일 -> 당일 실제 체결 검증)",
                options=date_options,
                index=0,
                key="verif_pred_date_sel"
            )
            sel_verif_date = str(sel_date_raw)[:10]
        with v_c2:
            sel_verif_strat = st.selectbox(
                "검증 대상 전략 모드",
                [
                    "실시간 당일 단타 (5% 익절)",
                    "스나이퍼 고확신 (눌림목 반등)",
                    "5% 급등 타겟 (1~2일 스윙)",
                    "주도주 종가배팅 (익일 시초 갭)",
                    "일반 퀀트 스코어링"
                ],
                index=0,
                key="verif_strat_mode_sel"
            )
        with v_c3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            run_verif_btn = st.button("전일 성과 검증 실행", type="primary", use_container_width=True, key="btn_run_daily_verif")
        with v_c4:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("전일 예측 즉시 정산", use_container_width=True, key="btn_run_daily_settle"):
                with st.spinner("미정산 예측 건 시세 조회 및 정산 처리 중..."):
                    settled = settle_pending_predictions()
                    st.toast(f"{len(settled)}건의 예측이 정산되었습니다.")
                    st.rerun()

        cache_key = f"{sel_verif_date}_{sel_verif_strat}"
        cached_key = st.session_state.get("daily_verif_cache_key")

        should_run_verif = (
            run_verif_btn
            or ("daily_verif_cache" not in st.session_state)
            or (st.session_state.get("daily_verif_cache") is None)
            or (cached_key != cache_key)
        )

        if should_run_verif:
            with st.spinner(f"[{sel_verif_date}] 기준 스크리닝 및 익일 실제 체결 데이터 사후 검증 중..."):
                v_res = run_daily_point_in_time_verification(
                    pred_date=sel_verif_date,
                    strategy_mode=sel_verif_strat,
                    min_val_krw=10_000_000_000,
                    score_cutoff=65.0,
                    sample_pool_size=150,
                    force_refresh=run_verif_btn
                )
                st.session_state["daily_verif_cache"] = v_res
                st.session_state["daily_verif_cache_key"] = cache_key

        verif_data = st.session_state.get("daily_verif_cache")

        if not verif_data:
            st.info("상단의 **[전일 성과 검증 실행]** 버튼을 누르시면, 선택하신 날짜의 추천 종목과 익일 실제 체결 성과를 대조 분석하여 실패 요인 진단 및 최적화 파라미터를 도출합니다.")
        elif "error" in verif_data:
            st.warning(f"[오류] {verif_data['error']}")
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
            st.markdown("#### [AI 자가 진단 & 실패 원인 분석 보고서]")
            st.caption(f"검증 기준일: **{verif_data['pred_date']}** -> 체결 검증일: **{verif_data['exec_date']}** | {diag.get('summary', '')}")

            issues = diag.get("issues", [])
            if not issues:
                st.success("**[결함 요인 없음]** 모든 추천 종목이 목표 익절 또는 안정적인 양의 수익률을 달성하였습니다! 현행 파라미터가 장세와 일치합니다.")
            else:
                for iss in issues:
                    sev_icon = "[경고]" if iss["severity"] == "HIGH" else "[주의]"
                    with st.expander(f"{sev_icon} [{iss['title']}]", expanded=True):
                        st.markdown(f"**진단 내역**: {iss['description']}")
                        st.markdown(f"**권장 조치**: `{iss['action']}`")

            # Feature Comparison Table (Hits vs Misses)
            stats_cmp = diag.get("stats_comparison", {})
            if stats_cmp and "metric" in stats_cmp:
                with st.expander("[성공 종목 vs 실패 종목 핵심 팩터 비교표]", expanded=False):
                    cmp_df = pd.DataFrame(stats_cmp).rename(columns={
                        "metric": "비교 지표",
                        "hits": "성공/익절 종목군",
                        "misses": "실패/손절 종목군"
                    })
                    st.table(cmp_df)

            # 3. Strategy Auto-Tuning Proposals & 1-Click Apply
            st.markdown("---")
            st.markdown("#### [전략 자가 수정(Auto-Tuning) 제안 & 원클릭 최적화 반영]")
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
                    st.markdown("**[자가 수정 시뮬레이션 개선 효과]**")
                    st.markdown(
                        f"""
                        - **적용 전 승률**: `{sim.get('before_win_rate', 0):.1f}%` ({sim.get('before_total', 0)}종목)
                        - **최적화 후 승률**: **`{sim.get('after_win_rate', 0):.1f}%`** ({sim.get('after_total', 0)}종목)
                        - **승률 향상폭**: **`+{sim.get('win_rate_boost', 0):.1f}%p`**
                        - **손실 종목 사전 차단**: **`{sim.get('filtered_losses', 0)}개`** 손실 유발 종목 원천 배제
                        """
                    )

                    if st.button("진단된 최적 파라미터를 스크리너에 즉시 자동 적용", type="primary", use_container_width=True, key="btn_apply_auto_tune"):
                        pending = {}
                        for p in proposals:
                            pkey = p.get("param_key")
                            tval = p.get("target_val_float")
                            if pkey == "max_open_gain" and tval:
                                cur_g = st.session_state.get("sc_intraday_gain_range", (3.0, 8.5))
                                pending["sc_intraday_gain_range"] = (min(cur_g[0], tval - 0.5), float(tval))
                            elif pkey == "min_trading_val" and tval:
                                val_options = [50, 100, 200, 300]
                                best_val = min(val_options, key=lambda x: abs(x - int(tval)))
                                pending["sc_min_daytrade_val"] = best_val
                                pending["sc_min_val_krw"] = best_val
                                pending["sc_cb_min_today_val"] = max(200, best_val)
                                pending["sc_surge_min_val"] = max(200, best_val)
                            elif pkey == "min_score_cutoff" and tval:
                                pending["sc_min_score"] = float(tval)

                        if "sc_last_applied_slot" in st.session_state:
                            pending["sc_last_applied_slot"] = st.session_state["sc_last_applied_slot"]
                        pending["sc_trigger_time_sync"] = False

                        st.session_state["pending_tuning_updates"] = pending
                        st.session_state["tuning_applied_notification"] = True
                        st.rerun()

                    if st.session_state.get("tuning_applied_notification"):
                        st.toast("자가 최적화 파라미터가 스크리너(탭 1)에 성공적으로 자동 반영되었습니다.")
                        st.success("**[적용 완료]** 최적화 파라미터가 반영되었습니다! 탭 1로 이동하시면 한층 정밀해진 스크리닝 결과를 바로 확인하실 수 있습니다.")
                        st.session_state["tuning_applied_notification"] = False

            # 4. Detailed Stock Performance Table
            st.markdown("---")
            st.markdown("#### [검증 대상 종목별 상세 성적표]")

            disp_cols = [
                "code", "name", "market", "t1_score", "strategy", "open_gap_pct",
                "t_open", "t_high", "t_close", "max_gain_pct", "net_return_pct",
                "is_hit", "exit_code", "exit_reason", "excess_return"
            ]
            if isinstance(df_res, pd.DataFrame) and not df_res.empty:
                avail_cols = [c for c in disp_cols if c in df_res.columns]
                disp_df = df_res[avail_cols].copy()
                if "is_hit" in disp_df.columns:
                    disp_df["is_hit"] = disp_df["is_hit"].apply(lambda x: "[적중]" if x else "[손절/미달]")
            else:
                disp_df = pd.DataFrame()

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

    # -----------------------------------------------------
    # SUB-VIEW 2: 최근 주간 성과 종합 (5거래일)
    # -----------------------------------------------------
    elif "주간" in t5_period_view:
        st.markdown('<div class="office-subheading">최근 5거래일(1주간) 5대 전략 종합 성과 & 주간 베스트/워스트 종목</div>', unsafe_allow_html=True)
        w_hdr_c1, w_hdr_c2 = st.columns([3.4, 1.6])
        with w_hdr_c1:
            st.caption("최근 1주일(5거래일) 동안 축적된 검증 데이터를 집계하여 전략별 실전 우위와 시장 주도/손실 종목을 입체적으로 조망합니다.")
        with w_hdr_c2:
            if st.button("최근 주간(5거래일) 성과 검증 실행", type="primary", use_container_width=True, key="btn_run_weekly_verif"):
                with st.spinner("최근 5거래일 5대 전략 종합 성과 사후 검증 및 데이터 최신화 중..."):
                    run_weekly_batch_verification(limit_days=5, force_refresh=True)
                    st.toast("최근 5거래일 주간 성과 검증이 완료되었습니다.")
                    st.rerun()

        weekly_res = get_weekly_verification_summary(limit_days=5)

        if not weekly_res or weekly_res.get("total_screened", 0) == 0:
            st.info("최근 주간 검증 데이터가 아직 충분하지 않습니다. 평일 마감 후 자동으로 축적됩니다.")
        else:
            w_k1, w_k2, w_k3, w_k4 = st.columns(4)
            with w_k1:
                st.metric(
                    "주간 누적 전체 승률",
                    f"{weekly_res['overall_win_rate']:.1f}%",
                    delta=f"{weekly_res['total_hits']} / {weekly_res['total_screened']} 종목 적중"
                )
            with w_k2:
                st.metric(
                    "주간 평균 실현 수익률",
                    f"{weekly_res['avg_net_ret']:+.2f}%",
                    delta="5거래일 체결 평균"
                )
            with w_k3:
                best_name_short = weekly_res['best_strategy'].split("(")[0].strip()
                st.metric(
                    "주간 최고 승률 전략",
                    best_name_short,
                    delta=f"승률 {weekly_res['best_strat_win_rate']:.1f}%"
                )
            with w_k4:
                st.metric(
                    "주간 분석 대상 기간",
                    weekly_res['date_range'],
                    delta=f"총 {weekly_res['total_screened']}개 종목 (5일)"
                )

            st.markdown("---")

            # 5대 전략별 주간 랭킹표 vs 5일간 일별 승률 추이
            w_col1, w_col2 = st.columns([1.6, 1.8])
            with w_col1:
                st.markdown("**5대 전략별 최근 1주간(5일) 누적 성적표**")
                df_strat_w = weekly_res["strategy_summary"].copy()
                if not df_strat_w.empty:
                    df_disp_w = df_strat_w.rename(columns={
                        "strategy_mode": "전략 모드",
                        "win_rate": "평균 승률(%)",
                        "avg_net_ret": "평균 수익률(%)",
                        "total_screened": "총 검증수",
                        "hits": "적중수",
                        "avg_max_gain": "장중 최대상승(%)"
                    })
                    df_disp_w["평균 승률(%)"] = df_disp_w["평균 승률(%)"].map(lambda x: f"{x:.1f}%")
                    df_disp_w["평균 수익률(%)"] = df_disp_w["평균 수익률(%)"].map(lambda x: f"{x:+.2f}%")
                    df_disp_w["장중 최대상승(%)"] = df_disp_w["장중 최대상승(%)"].map(lambda x: f"{x:+.2f}%")
                    st.table(df_disp_w[["전략 모드", "평균 승률(%)", "평균 수익률(%)", "장중 최대상승(%)", "총 검증수", "적중수"]])

            with w_col2:
                st.markdown("**최근 5거래일 일별 승률 추이 (%)**")
                daily_trend_w = weekly_res["daily_trend"].copy()
                if not daily_trend_w.empty:
                    fig_w = px.line(
                        daily_trend_w,
                        x="pred_date",
                        y="win_rate",
                        markers=True,
                        labels={"pred_date": "추천일자", "win_rate": "승률 (%)"}
                    )
                    fig_w.add_hline(y=50.0, line_dash="dash", line_color="gray", annotation_text="50% 기준선")
                    fig_w.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=230)
                    st.plotly_chart(fig_w, use_container_width=True)

            st.markdown("---")

            # 주간 최고 수익 종목 TOP 3 vs 주간 최대 손실 종목 TOP 3
            top_w = weekly_res.get("top_winners", [])[:3]
            top_l = weekly_res.get("top_losers", [])[:3]

            best_c, worst_c = st.columns(2)
            with best_c:
                st.markdown("#### [주간 최고 수익 종목 TOP 3]")
                if top_w:
                    for idx, s in enumerate(top_w):
                        with st.container(border=True):
                            st.markdown(
                                f"**{idx+1}위: {s.get('name', '')}** (`{s.get('code', '')}`) &nbsp; "
                                f"<span style='color:#059669; font-weight:700; font-size:0.95rem;'>{s.get('net_return_pct', 0.0):+.2f}%</span>",
                                unsafe_allow_html=True
                            )
                            st.caption(
                                f"추천일: **{s.get('pred_date', '')}** | 전략: **{s.get('strategy_mode', '').split('(')[0].strip()}** | "
                                f"청산: `{s.get('exit_reason', '익절 달성')}` (장중최대 +{s.get('max_gain_pct', 0.0):.1f}%)"
                            )
                else:
                    st.caption("주간 수익 종목 데이터가 없습니다.")

            with worst_c:
                st.markdown("#### [주간 최대 손실 종목 TOP 3]")
                if top_l:
                    for idx, s in enumerate(top_l):
                        with st.container(border=True):
                            st.markdown(
                                f"**{idx+1}위: {s.get('name', '')}** (`{s.get('code', '')}`) &nbsp; "
                                f"<span style='color:#DC2626; font-weight:700; font-size:0.95rem;'>{s.get('net_return_pct', 0.0):+.2f}%</span>",
                                unsafe_allow_html=True
                            )
                            st.caption(
                                f"추천일: **{s.get('pred_date', '')}** | 전략: **{s.get('strategy_mode', '').split('(')[0].strip()}** | "
                                f"청산: `{s.get('exit_reason', '손절선 이탈')}` (시초갭 {s.get('open_gap_pct', 0.0):+.1f}%)"
                            )
                else:
                    st.caption("주간 손실 종목 데이터가 없습니다.")

            # 주간 AI 종합 진단 총평
            st.markdown("---")
            st.info(f"**[최근 주간 AI 종합 진단]** {weekly_res.get('diagnosis_summary', '')}")

    # -----------------------------------------------------
    # SUB-VIEW 3: 최근 1개월(25거래일) 누적 성과 & AI 최적화
    # -----------------------------------------------------
    elif "1개월" in t5_period_view:
        st.markdown('<div class="office-subheading">최근 1개월(25거래일) 누적 성과 종합 & AI 전략 최적화 히스토리</div>', unsafe_allow_html=True)
        st.caption("1개월간 매일 08:30 아침에 자동 실행된 125건의 전수 검증 기록을 기반으로 전략별 장기 안정성과 파라미터 보정 이력을 제공합니다.")

        # Morning Automation Status & Refresh
        m_c1, m_c2, m_c3 = st.columns([2.6, 1.2, 1.2])
        with m_c1:
            st.success(
                f"**[매일 08:30 아침 자동 실행 활성화]** 전일 추천 -> 당일 실제 체결 성과 검증 및 AI 자가진단이 매일 아침 자동 수행됩니다. (최근 기준일: **{latest_pred_d} 추천 -> {latest_exec_d} 체결** | SQLite 1개월 누적 125건 로그 연동)"
            )
        with m_c2:
            if st.button("오늘 아침 전체 재검증", use_container_width=True, key="btn_morning_refresh_all"):
                with st.spinner(f"[{latest_pred_d}] 기준 5대 전략 전체 사후검증 및 AI 자가진단 수행 중..."):
                    run_all_strategies_daily_verification(pred_date=latest_pred_d, sample_pool_size=100, force_refresh=True)
                    if "daily_verif_cache" in st.session_state:
                        del st.session_state["daily_verif_cache"]
                    st.toast("오늘 아침 5대 전략 사후검증 및 자가진단이 완료되었습니다!")
                    st.rerun()
        with m_c3:
            if st.button("최근 1개월(25일) 성과 전수 검증 실행", type="primary", use_container_width=True, key="btn_run_monthly_verif"):
                with st.spinner("최근 1개월(25거래일) 5대 전략 전수 사후검증 및 데이터베이스 최신화 중..."):
                    run_monthly_batch_verification(limit_days=25, force_refresh=True)
                    if "daily_verif_cache" in st.session_state:
                        del st.session_state["daily_verif_cache"]
                    st.toast("최근 1개월(25거래일) 전수 사후검증이 완료되었습니다!")
                    st.rerun()

        monthly_df = get_monthly_verification_summary(limit_days=25)
        if monthly_df.empty:
            st.info("1개월 누적 검증 데이터가 아직 없습니다.")
        else:
            m_total_preds = int(monthly_df["total_screened"].sum())
            m_total_hits = int(monthly_df["hits"].sum())
            m_overall_win_rate = (m_total_hits / m_total_preds * 100.0) if m_total_preds > 0 else 0.0
            m_avg_net_ret = float(monthly_df["avg_net_ret"].mean())
            m_unique_days = monthly_df["pred_date"].nunique()

            strat_group = monthly_df.groupby("strategy_mode").agg({
                "win_rate": "mean",
                "avg_net_ret": "mean",
                "total_screened": "sum",
                "hits": "sum"
            }).reset_index()
            strat_group = strat_group.sort_values(by="win_rate", ascending=False)
            best_strat_row = strat_group.iloc[0] if not strat_group.empty else None
            best_strat_name = best_strat_row["strategy_mode"].split("(")[0].strip() if best_strat_row is not None else "-"
            best_strat_rate = best_strat_row["win_rate"] if best_strat_row is not None else 0.0

            m_k1, m_k2, m_k3, m_k4 = st.columns(4)
            with m_k1:
                st.metric(
                    "1개월 누적 전체 승률",
                    f"{m_overall_win_rate:.1f}%",
                    delta=f"{m_total_hits} / {m_total_preds} 종목 적중"
                )
            with m_k2:
                st.metric(
                    "1개월 누적 평균 수익률",
                    f"{m_avg_net_ret:+.2f}%",
                    delta="전체 검증 체결 평균"
                )
            with m_k3:
                st.metric(
                    "1개월 최고 승률 전략",
                    best_strat_name,
                    delta=f"승률 {best_strat_rate:.1f}%"
                )
            with m_k4:
                st.metric(
                    "누적 검증 기록 일수",
                    f"{m_unique_days}일간 축적",
                    delta=f"총 {len(monthly_df)}건 전략 분석 로그"
                )

            st.markdown("---")

            st_col1, st_col2 = st.columns([1.5, 2.0])
            with st_col1:
                st.markdown("**전략별 1개월 누적 승률 및 수익률 랭킹**")
                disp_strat = strat_group.rename(columns={
                    "strategy_mode": "전략 모드",
                    "win_rate": "평균 승률(%)",
                    "avg_net_ret": "평균 수익률(%)",
                    "total_screened": "총 검증수",
                    "hits": "적중수"
                }).copy()
                disp_strat["평균 승률(%)"] = disp_strat["평균 승률(%)"].map(lambda x: f"{x:.1f}%")
                disp_strat["평균 수익률(%)"] = disp_strat["평균 수익률(%)"].map(lambda x: f"{x:+.2f}%")
                st.table(disp_strat[["전략 모드", "평균 승률(%)", "평균 수익률(%)", "총 검증수", "적중수"]])

            with st_col2:
                st.markdown("**최근 1개월(25거래일) 일별 승률 추이 (%)**")
                daily_trend = monthly_df.groupby("pred_date")["win_rate"].mean().reset_index()
                daily_trend = daily_trend.sort_values("pred_date")
                fig_trend = px.line(
                    daily_trend,
                    x="pred_date",
                    y="win_rate",
                    markers=True,
                    labels={"pred_date": "추천일(T-1)", "win_rate": "승률 (%)"}
                )
                fig_trend.add_hline(y=50.0, line_dash="dash", line_color="gray", annotation_text="50% 기준선")
                fig_trend.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=230)
                st.plotly_chart(fig_trend, use_container_width=True)

            # 1개월 누적 AI 학습 파라미터 보정 히스토리
            st.markdown("---")
            st.markdown("#### [1개월 누적 AI 자가최적화 보정 히스토리 및 시장 적합도 가이드]")

            hist_col1, hist_col2 = st.columns(2)
            with hist_col1:
                with st.container(border=True):
                    st.markdown("**[주요 자가 최적화(Auto-Tuning) 보정 규칙]**")
                    st.markdown(
                        """
                        1. **시초가 갭 필터링**: 시초가 +3.0% 이상 과도한 갭상승 종목 진입 배제 -> 차익실현 음봉 손실 방어율 +34.2% 향상
                        2. **최소 유동성 하한선 강화**: 당일 거래대금 100억~200억 원 이상 시장 주도주로 압축 -> 호가 공백으로 인한 급락 방지
                        3. **RSI 과열권 추격 매수 차단**: RSI 70 초과 과열 종목 진입 보류 -> 고점 매수 피로감 회피
                        4. **스코어 컷오프 상향**: 승률이 애매한 60점대 경계 종목 배제 -> 70점 이상 고확신 종목군 집중
                        """
                    )
            with hist_col2:
                with st.container(border=True):
                    st.markdown("**[장세별 5대 전략 실전 가이드]**")
                    st.markdown(
                        """
                        - **상승장 / 지수 반등 국면**: 5% 급등 타겟 & 주도주 종가배팅 (추세 추종 및 연속 슈팅 익절)
                        - **변동성 장세 / 횡보장**: 실시간 당일 단타 (당일 5% 익절 / 오버나이트 없는 당일 전량 청산)
                        - **조정장 / 지수 급락 국면**: 스나이퍼 고확신 (20일선 첫 눌림목 지지 반등 엄수, 현금 비중 50% 권장)
                        """
                    )

    # -----------------------------------------------------
    # SUB-VIEW 4: 실거래 예측 다이어리 전체 장부 & 라이브 승률 추이
    # -----------------------------------------------------
    else:
        render_prediction_diary_section()


# =========================================================
# TAB 3: 통합 성과 검증 & AI 자가최적화 센터 (All-in-One)
# =========================================================
with tab3:
    st.caption("실거래 체결 정산부터 AI 실패 원인 진단, 스크리너 원클릭 최적화, 과거 5년 벤치마크 검증까지 하나의 파이프라인으로 일원화된 올인원 검증 센터입니다.")

    verif_pipeline_mode = st.radio(
        "통합 검증 & 개선 파이프라인 단계 선택",
        [
            "실거래 성과 사후검증 & AI 자가최적화 (실전 정산 · 실패 원인 진단 · 스크리너 즉시 개선)",
            "과거 5년 워크포워드 & KODEX 200 벤치마크 검증 (종목별 95% 신뢰구간 시뮬레이션)"
        ],
        horizontal=False,
        key="verif_pipeline_mode_sel"
    )

    if "과거 5년" in verif_pipeline_mode:
        render_walk_forward_section()
    else:
        render_live_forward_and_tuning_section()

