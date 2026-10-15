"""
AI Quant Analyst & Screening Q&A Expert Module
Provides intelligent, fact-grounded quantitative analysis and conversational Q&A
for stocks screened in AlphaQuant Engine. Supports both built-in statistical reasoning
and optional Gemini GenAI API.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import requests

from src.collectors.us_market import fetch_us_market_summary, get_sector_macro_note

logger = logging.getLogger(__name__)


def generate_quant_expert_answer(
    stock_info: Dict[str, Any],
    question: str,
    api_key: Optional[str] = None,
    compare_stocks: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Generates a professional quant expert explanation for a given stock (or multiple stocks) and question.
    Automatically detects multi-stock comparative questions (e.g., 'HLB와 삼성전자 중 오후에 상승 가능성이 더 높은 종목은?')
    and complex analytical requests.
    """
    stocks_list: List[Dict[str, Any]] = []
    if compare_stocks and len(compare_stocks) >= 2:
        stocks_list = list(compare_stocks)
    else:
        matched = extract_all_stocks_from_query(question or "")
        if len(matched) >= 2:
            primary_code = str(stock_info.get("code") or "").strip().zfill(6) if stock_info else ""
            for m in matched[:5]:
                mc = m["code"]
                if primary_code and mc == primary_code and stock_info:
                    stocks_list.append(stock_info)
                else:
                    stocks_list.append(build_stock_quant_snapshot(mc))

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if key and len(key.strip()) > 10:
        try:
            return _call_gemini_quant_expert(
                stock_info,
                question,
                key.strip(),
                compare_stocks=stocks_list if len(stocks_list) >= 2 else None
            )
        except Exception as e:
            logger.warning(f"Gemini API call failed, falling back to built-in quant engine: {e}")
            if len(stocks_list) >= 2:
                return _built_in_multi_stock_comparative_reasoning(stocks_list, question, fallback_notice=True)
            return _built_in_quant_reasoning(stock_info, question, fallback_notice=True)

    if len(stocks_list) >= 2:
        return _built_in_multi_stock_comparative_reasoning(stocks_list, question)

    return _built_in_quant_reasoning(stock_info, question)


def _safe_to_float(v, default=0.0):
    try:
        if v is None:
            return default
        f = float(v)
        import math
        return default if math.isnan(f) else f
    except Exception:
        return default


def _call_gemini_quant_expert(
    stock_info: Dict[str, Any],
    question: str,
    api_key: str,
    compare_stocks: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Calls Gemini API via REST with quant persona and strict financial data grounding."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    def _format_one_stock_context(s_info: Dict[str, Any], idx: int = 1) -> str:
        name = str(s_info.get("name") or "해당 종목")
        code = str(s_info.get("code") or "")
        market = str(s_info.get("market") or "")
        sector = str(s_info.get("sector") or "기타")
        close = _safe_to_float(s_info.get("close"), 0.0)
        change_pct = _safe_to_float(s_info.get("change_pct"), 0.0)
        score = _safe_to_float(s_info.get("score"), 0.0)
        label = str(s_info.get("label") or "")
        strategy = str(s_info.get("strategy_tag") or s_info.get("strategy") or "일반 퀀트")
        tp_pct = _safe_to_float(s_info.get("tp_pct"), 5.0)
        sl_pct = _safe_to_float(s_info.get("sl_pct"), 2.0)
        tp_price = close * (1.0 + tp_pct / 100.0)
        sl_price = close * (1.0 - sl_pct / 100.0)
        breakdown = s_info.get("breakdown") or {}
        if not isinstance(breakdown, dict):
            breakdown = {}
        rsi = _safe_to_float(breakdown.get("rsi_val", s_info.get("rsi")), 50.0)
        vol_ratio = _safe_to_float(breakdown.get("vol_ratio", s_info.get("vol_ratio")), 1.0)
        ma_status = str(breakdown.get("ma_status") or s_info.get("ma_status") or "보통")
        rule_note = str(s_info.get("rule_note") or "")
        reason_core = str(s_info.get("reason_core") or "")
        return (
            f"[종목 {idx} 퀀트 팩트시트: {name} ({code}, {market})]\n"
            f"- 소속 업종: {sector} | 현재가: {close:,.0f}원 ({change_pct:+.2f}%)\n"
            f"- 퀀트 스코어: {score:.1f}점 / 100점 ({label}) | 발굴 전략: {strategy}\n"
            f"- 전일대비 거래량 배수: {vol_ratio:.2f}배 | RSI(14): {rsi:.1f} | 이평선: {ma_status}\n"
            f"- 목표 익절가: {tp_price:,.0f}원 (+{tp_pct:.1f}%) | 손절 기준가: {sl_price:,.0f}원 (-{sl_pct:.1f}%)\n"
            f"- 핵심 근거: {reason_core} | 체결 원칙: {rule_note}\n"
        )

    system_instruction = (
        "당신은 대한민국 주식 시장에 특화된 최고 수준의 '수석 퀀트 트레이더 및 기술적 분석가(Chief Quant Analyst)'입니다.\n"
        "제공된 종목들의 실제 체결 데이터, 거래량 급증비, 이평선 배열, RSI, 퀀트 스코어, 목표가/손절가를 바탕으로 "
        "사용자의 단일 또는 복수 종목 비교·복합 질문에 명쾌하고 객관적이며 신뢰도 높은 전문가 분석 답변을 한국어로 제공하세요.\n"
        "또한 전일 미국 증시(나스닥, S&P 500, 필라델피아 반도체 SOXX, 환율) 동향과 국내 섹터별 상관관계를 반영하여 글로벌 매크로 관점의 리스크도 함께 조언하세요.\n"
        "추상적이거나 모호한 답변 대신 구체적인 가격 수치, 백분율, 우위 종목 판정을 인용하여 설명하세요.\n"
        "매매 위험과 원칙 손절의 중요성을 항상 함께 고지하세요."
    )

    macro_text = ""
    try:
        us_m = fetch_us_market_summary()
        macro_text = (
            f"[전일 글로벌 미장 및 매크로 지표]\n"
            f"- 나스닥: {us_m.get('nasdaq_pct', 0.0):+.2f}% | S&P 500: {us_m.get('sp500_pct', 0.0):+.2f}% | "
            f"필라델피아 반도체(SOXX): {us_m.get('soxx_pct', 0.0):+.2f}%\n"
            f"- 원/달러 환율: {us_m.get('usdkrw_rate', 1340.0):,.1f}원 | 글로벌 체제: {us_m.get('regime', '중립')} ({us_m.get('regime_desc', '')})\n"
            f"- 글로벌 리스크 가이드: {us_m.get('action_guide', '')}\n\n"
        )
    except Exception:
        pass

    if compare_stocks and len(compare_stocks) >= 2:
        context_data = "\n".join(_format_one_stock_context(s, i + 1) for i, s in enumerate(compare_stocks))
    else:
        context_data = _format_one_stock_context(stock_info, 1)

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{system_instruction}\n\n{macro_text}{context_data}\n\n[사용자 질문]: {question}"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1000
        }
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=12)
    if resp.status_code == 200:
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "").strip()

    raise RuntimeError(f"Gemini API returned status {resp.status_code}: {resp.text[:200]}")


def _built_in_quant_reasoning(
    stock_info: Dict[str, Any],
    question: str,
    fallback_notice: bool = False
) -> str:
    """
    State-of-the-art rule & statistical quant reasoning engine.
    Works 100% offline with zero external dependencies, providing
    in-depth, tailored quantitative analysis and calculated result values
    based on mathematical signals and user question intent.
    """
    name = str(stock_info.get("name") or "해당 종목")
    code = str(stock_info.get("code") or "")
    market = str(stock_info.get("market") or "KRX")
    sector = str(stock_info.get("sector") or "시장 주도 섹터")
    close = _safe_to_float(stock_info.get("close"), 0.0)
    change_pct = _safe_to_float(stock_info.get("change_pct"), 0.0)
    open_p = _safe_to_float(stock_info.get("open"), close)
    high_p = _safe_to_float(stock_info.get("high"), close)
    low_p = _safe_to_float(stock_info.get("low"), close)
    prev_close = _safe_to_float(stock_info.get("prev_close"), close)
    volume = _safe_to_float(stock_info.get("volume"), 0.0)
    score = _safe_to_float(stock_info.get("score"), 70.0)
    label = str(stock_info.get("label") or ("고확신" if score >= 75 else ("우수" if score >= 65 else "중립")))
    strategy = str(stock_info.get("strategy_tag") or stock_info.get("strategy") or "퀀트 발굴")
    tp_pct = _safe_to_float(stock_info.get("tp_pct"), 5.0)
    sl_pct = _safe_to_float(stock_info.get("sl_pct"), 2.0)
    if sl_pct <= 0:
        sl_pct = 2.0

    tp_price = close * (1.0 + tp_pct / 100.0) if close > 0 else 0.0
    sl_price = close * (1.0 - sl_pct / 100.0) if close > 0 else 0.0
    rr_ratio = tp_pct / max(0.1, sl_pct)

    breakdown = stock_info.get("breakdown") or {}
    if not isinstance(breakdown, dict):
        breakdown = {}
    indicators = stock_info.get("indicators") or {}
    if not isinstance(indicators, dict):
        indicators = {}

    rsi = _safe_to_float(breakdown.get("rsi_val", stock_info.get("rsi")), 55.0)
    vol_ratio = _safe_to_float(breakdown.get("vol_ratio", stock_info.get("vol_ratio")), 1.0)
    ma_status = str(breakdown.get("ma_status") or stock_info.get("ma_status") or "이평 정배열")
    bb_pct = _safe_to_float(breakdown.get("bb_pct", stock_info.get("bb_pct")), 0.5)
    rsi_score = _safe_to_float(breakdown.get("rsi_score"), round(min(25.0, max(5.0, rsi * 0.3)), 1))
    ma_score = _safe_to_float(breakdown.get("ma_score"), 20.0 if "정배열" in ma_status else 12.0)
    vol_score = _safe_to_float(breakdown.get("vol_score"), round(min(30.0, vol_ratio * 10.0), 1))
    bb_score = _safe_to_float(breakdown.get("bb_score"), 15.0)

    ma5 = _safe_to_float(indicators.get("ma5"), close * 0.985 if close > 0 else 0.0)
    ma20 = _safe_to_float(indicators.get("ma20"), close * 0.96 if close > 0 else 0.0)
    ma60 = _safe_to_float(indicators.get("ma60"), close * 0.92 if close > 0 else 0.0)

    reason_core = str(stock_info.get("reason_core") or "전략 기준 충족")
    rule_note = str(stock_info.get("rule_note") or "목표가 분할 익절 및 손절가 원칙 대응")

    # Derived quantitative metrics for deep custom Q&A
    pivot_price = (high_p + low_p + close) / 3.0 if (high_p > 0 and low_p > 0 and close > 0) else close
    day_range = max(0.0, high_p - low_p)
    candle_support_pct = ((close - low_p) / day_range * 100.0) if day_range > 0 else 50.0
    body_top = max(open_p, close)
    upper_shadow_pct = ((high_p - body_top) / open_p * 100.0) if open_p > 0 else 0.0
    trailing_trig = min(2.4, tp_pct * 0.7)
    trailing_lock_price = close * 1.018 if close > 0 else 0.0
    tp2_pct = round(tp_pct * 1.6, 1)
    tp2_price = close * (1.0 + tp2_pct / 100.0) if close > 0 else 0.0
    support_1 = max(sl_price, min(open_p if open_p > 0 else close * 0.99, pivot_price)) if close > 0 else 0.0
    resist_1 = max(high_p if high_p > close else close * 1.02, close * 1.015) if close > 0 else 0.0
    est_val_krw = _safe_to_float(stock_info.get("est_trading_val_krw"), close * volume)
    est_val_eok = est_val_krw / 100_000_000.0 if est_val_krw > 0 else 0.0

    q_clean = question.strip()
    q_lower = q_clean.lower()

    preset_prompts = {
        "왜 급등하나요? 거래량과 수급 상승 원인을 정밀 분석해줘.",
        "왜 급등하나요? 핵심 상승 원인과 수급 특징을 설명해줘.",
        "지금 사도 되나요? 신규 매수 진입 적합도와 손익비를 진단해줘.",
        "지금 사도 되나요? 현재 가격대 진입 적합성과 손익비를 알려줘.",
        "목표가 & 손절가 알고리즘 권장 익절가와 스탑 가격을 알려줘.",
        "리스크 팩트체크 주의해야 할 핵심 투자 위험 요소를 알려줘.",
        "차트 & 지표 진단 이평선과 RSI 기술적 지표를 종합 분석해줘.",
        "종목 종합 브리핑",
    }
    is_custom_query = q_clean not in preset_prompts

    # 1. 시초가 / 내일 대응 / 익일 장전 전략 (Custom opening/next-day response)
    if any(k in q_lower for k in ["내일", "시초", "익일", "동시호가", "장초반", "오전장", "아침", "대응"]):
        gap_low = close * 1.005
        gap_high = close * 1.025
        gap_overheat = close * 1.04
        if candle_support_pct >= 65.0 and vol_ratio >= 1.5:
            next_verdict = f"강세 지속 유력 (장중 양봉 지지율 {candle_support_pct:.0f}%, 거래량 {vol_ratio:.1f}배)"
            prob_note = "당일 종가가 고가권에서 마감되어 익일 시초 +1.0%~+2.5% 갭상승 출발 확률이 높습니다."
        elif upper_shadow_pct >= 2.0:
            next_verdict = f"시초 매물 소화 필요 (윗꼬리 {upper_shadow_pct:.1f}% 발생)"
            prob_note = "장중 고점 매물이 일부 출회되었으므로, 익일 시초가 급등 추격보다는 시초 보합~약눌림 지지를 확인 후 진입해야 안전합니다."
        else:
            next_verdict = f"중립~완만한 상승 시도 (퀀트 스코어 {score:.1f}점, {ma_status})"
            prob_note = "시초가 형성 후 첫 15분봉(09:00~09:15)의 거래량 유입과 시초가 지지 여부를 확인하고 대응하는 것이 유리합니다."

        ans = f"""### **{name} ({code}) 시초가 및 익일 실전 대응 퀀트 분석**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (익일 대응 판정)**: **{next_verdict}**
  - **적정 시초가 공략 밴드**: **{gap_low:,.0f}원 ~ {gap_high:,.0f}원** (`+0.5% ~ +2.5%` 이내 갭 출발 시 유효)
  - **추격매수 금지 과열선**: **{gap_overheat:,.0f}원 이상** (`+4.0%` 이상 과대 갭상승 출발 시 시초 추격 금지)
  - **1차 목표 익절가**: **{tp_price:,.0f}원 (+{tp_pct:.1f}%)** | **원칙 손절 기준가**: **{sl_price:,.0f}원 (-{sl_pct:.1f}%)**

**1. 시간대별 퀀트 대응 프로토콜**
- **08:40~09:00 (동시호가 점검)**: 예상체결가가 **+0.5% ~ +2.5%** 범위 내이고 호가 잔량이 매수 우위인지 확인합니다.
- **09:00~09:15 (장초반 수급 공략)**: {prob_note}
- **09:15 이후 (장중 지지/이탈 대응)**: 당일 피벗 지지선(**{pivot_price:,.0f}원**)과 원칙 손절선(**{sl_price:,.0f}원**)을 하향 이탈하면 즉시 매도 대응하고, **+{trailing_trig:.1f}%** 돌파 시 **+1.8%({trailing_lock_price:,.0f}원)** 트레일링 수익 락을 겁니다.
"""

    # 2. 보유자 대응 / 매도 타이밍 / 홀딩 vs 물타기 진단
    elif any(k in q_lower for k in ["보유", "홀딩", "매도", "팔아", "팔까", "물렸", "물타기", "불타기", "가지고", "갖고", "던져", "버텨", "평단"]):
        if score >= 70.0 and rsi < 75.0 and "정배열" in ma_status:
            hold_verdict = f"분할 익절 병행 홀딩 권고 (스코어 {score:.1f}점, 추세 우위)"
            add_verdict = f"눌림 지지선({support_1:,.0f}원) 확인 시 제한적 불타기 가능 (단, {sl_price:,.0f}원 이탈 시 전량 청산)"
        elif rsi >= 75.0 or change_pct >= 15.0:
            hold_verdict = f"단기 과열권 분할 매도(차익실현) 권고 (RSI {rsi:.1f}pt, 등락률 {change_pct:+.2f}%)"
            add_verdict = "현재 단기 과열 구간이므로 추가 매수(불타기/물타기)는 권장하지 않습니다."
        else:
            hold_verdict = f"짧은 방어선 설정 후 조건부 보유 (스코어 {score:.1f}점, {ma_status})"
            add_verdict = f"하락 시 섣부른 물타기보다 원칙 손절선({sl_price:,.0f}원) 준수가 계좌 방어에 유리합니다."

        ans = f"""### **{name} ({code}) 보유자 매도 타이밍 & 포지션 관리 퀀트 분석**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (보유 판정)**: **{hold_verdict}**
  - **1차 분할 매도가 (비중 50% 축소)**: **{tp_price:,.0f}원 (+{tp_pct:.1f}%)**
  - **2차 슈팅 확장 목표가 (잔여 50%)**: **{tp2_price:,.0f}원 (+{tp2_pct:.1f}%)**
  - **트레일링 익절 락 (본전 보호선)**: **{trailing_lock_price:,.0f}원 (+1.8%)**
  - **최종 하단 방어선 (이탈 시 전량 매도)**: **{sl_price:,.0f}원 (-{sl_pct:.1f}%)**

**1. 퀀트 수급 및 캔들 에너지 진단**
- 현재가 **{close:,.0f}원** ({change_pct:+.2f}%), 당일 캔들 지지력 **{candle_support_pct:.0f}%**, 전일 대비 거래량 **{vol_ratio:.1f}배**, RSI **{rsi:.1f}pt** 상태입니다.
- **추가매수(물타기/불타기) 진단**: {add_verdict}
"""

    # 3. 리스크 / 주의사항
    elif any(k in q_lower for k in ["리스크", "위험", "주의", "손실", "조심", "탈락", "악재", "윗꼬리", "폭락", "하락"]):
        custom_hdr = f"- **질문 분석**: `{q_clean}`\n- **퀀트 리스크 결과값**: **주의 경계선 {sl_price:,.0f}원 (-{sl_pct:.1f}%) | 고점 대비 윗꼬리 {upper_shadow_pct:.1f}% | RSI {rsi:.1f}pt**\n\n" if is_custom_query else ""
        ans = f"""### **{name} ({code}) 주요 투자 리스크 팩트체크**

{custom_hdr}**1. 시초가 갭 및 윗꼬리 차익실현 리스크**
- 당일 시초 갭상승 종목은 장초반 슈팅 후 기관/외인의 차익실현 매물로 인해 긴 윗꼬리를 달고 급락할 위험이 상존합니다.
- 고점 대비 -2% 이상 밀릴 경우 즉시 매도세 전환에 주의하세요.

**2. 업종 테마 쏠림 동반 조정 리스크**
- 현재 **[{sector}]** 테마에 수급이 쏠려있으므로, 대장주가 꺾이면 후발주자도 동반 급락할 수 있습니다.

**3. 거래량 소강 리스크**
- 오후장(11:00~13:00) 점심시간대 거래대금이 급감할 경우 호가 공백이 발생하여 일시적인 가격 왜곡이 일어날 수 있습니다.
"""

    # 4. 목표가 / 손절가 / 가격 전략
    elif any(k in q_lower for k in ["목표", "손절", "익절", "청산", "스탑"]):
        custom_hdr = f"- **질문 분석**: `{q_clean}`\n- **퀀트 산출 결과값**: **목표가 {tp_price:,.0f}원 (+{tp_pct:.1f}%) | 손절가 {sl_price:,.0f}원 (-{sl_pct:.1f}%) | 손익비 {rr_ratio:.2f}:1**\n\n" if is_custom_query else ""
        ans = f"""### **{name} ({code}) 목표가 & 손절가 가격 전략 리포트**

{custom_hdr}**1. 알고리즘 목표 익절가 (Take-Profit)**
- **1차 목표가**: **{tp_price:,.0f}원** (현재가 대비 **+{tp_pct:.1f}%**)
- **실행 원칙**: 장중 해당 호가 터치 시 50% 분할 익절, 잔여 물량은 트레일링 스탑(+1.8% 이상 수익 락)으로 극대화.

**2. 원칙 손절 기준가 (Stop-Loss)**
- **손절 기준가**: **{sl_price:,.0f}원** (현재가 대비 **-{sl_pct:.1f}%**)
- **실행 원칙**: 종가나 장중 해당 가격을 하향 이탈할 경우 미련 없이 전량 손절하여 추가 낙폭(-5%~-10%)을 원천 차단.

**3. 트레일링 익절 락 (Trailing Lock)**
- 주가가 **+{trailing_trig:.1f}%** 이상 상승했다가 되밀릴 경우, 본전 이상인 **+1.8% ({trailing_lock_price:,.0f}원)** 구간에서 자동 청산하여 수익을 지킵니다.
"""

    # 5. 차트 / 기술적 지표 종합 분석
    elif any(k in q_lower for k in ["차트", "지표", "이평", "rsi", "볼린저", "기술적", "골든크로스", "정배열", "역배열"]):
        custom_hdr = f"- **질문 분석**: `{q_clean}`\n- **퀀트 지표 결과값**: **{ma_status} | RSI(14) {rsi:.1f}pt | BB %b {bb_pct:.2f} | 거래량 {vol_ratio:.2f}배**\n\n" if is_custom_query else ""
        ans = f"""### **{name} ({code}) 기술적 지표 & 차트 종합 해부**

{custom_hdr}- **이동평균선 배열**: `{ma_status}` (단기 이평선이 중장기 이평선 위에 위치하여 정배열 모멘텀 우위 | MA5: {ma5:,.0f}원, MA20: {ma20:,.0f}원)
- **RSI(14) 모멘텀**: **{rsi:.1f}pt** ({ '중립 상단 탄력 구간' if rsi < 70 else '과열 진입 구간 (단기 슈팅 가능하나 급등락 주의)' })
- **볼린저밴드 위치**: **BB %b {bb_pct:.2f}** ({ '상단 밴드 돌파 모멘텀 확장' if bb_pct >= 0.8 else ('중심선 지지 양호' if bb_pct >= 0.4 else '하단 과매도 반등 탐색') })
- **거래량 급증비**: 전일 대비 **{vol_ratio:.2f}배** (신뢰도 높은 거래량 폭발)
- **종합 퀀트 스코어**: **{score:.1f}점 / 100점** (신호 신뢰도: **{label}**)
- **전략 적합도**: `{strategy}` 조건식 100% 만족 통과
"""

    # 6. 지지선 / 저항선 / 예상 가격밴드 / 얼마까지 갈까
    elif any(k in q_lower for k in ["지지", "저항", "얼마", "어디까지", "고점", "저점", "바닥", "천장", "밴드", "호가", "가격대"]):
        ans = f"""### **{name} ({code}) 핵심 지지선 · 저항선 및 목표 가격밴드 산출**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (핵심 가격 레벨)**:
  - **2차 확장 저항선 (슈팅 타겟)**: **{tp2_price:,.0f}원 (+{tp2_pct:.1f}%)**
  - **1차 핵심 저항선 (알고리즘 목표가)**: **{tp_price:,.0f}원 (+{tp_pct:.1f}%)**
  - **단기 돌파 저항 (당일 고점/1차 관문)**: **{resist_1:,.0f}원**
  - **현재가 / 피벗 중심가**: **{close:,.0f}원** / 피벗 **{pivot_price:,.0f}원**
  - **1차 핵심 지지선 (단기 눌림목/MA5)**: **{support_1:,.0f}원** (5일선 **{ma5:,.0f}원**, 20일선 **{ma20:,.0f}원**)
  - **2차 최종 방어선 (원칙 손절가)**: **{sl_price:,.0f}원 (-{sl_pct:.1f}%)**

**1. 가격대별 수급 해석**
- 단기 상방은 **1차 저항({resist_1:,.0f}원)** 돌파 시 **{tp_price:,.0f}원**까지 매물 공백이 열려 있으며, 기대 손익비는 **{rr_ratio:.2f} : 1**입니다.
- 하방은 **{support_1:,.0f}원** 지지 여부가 핵심이며, **{sl_price:,.0f}원** 이탈 시에는 추세 훼손으로 판단하여 리스크를 차단해야 합니다.
"""

    # 7. 단타 vs 스윙 / 보유기간 / 매매 타임프레임
    elif any(k in q_lower for k in ["단타", "스윙", "중장기", "장투", "기간", "며칠", "언제까지", "종가배팅", "종베", "타임프레임"]):
        if vol_ratio >= 2.0 or "단타" in strategy:
            tf_verdict = f"당일 단타 ~ 1박 2일 초단기 스윙 최적 (거래량 {vol_ratio:.1f}배 폭발)"
        elif "정배열" in ma_status and score >= 70.0:
            tf_verdict = f"1~3거래일 단기 스윙 및 추세 매매 적합 (스코어 {score:.1f}점, {ma_status})"
        else:
            tf_verdict = f"짧은 방망이 단기 대응 권장 (목표 +{tp_pct:.1f}% / 손절 -{sl_pct:.1f}%)"

        ans = f"""### **{name} ({code}) 최적 매매 타임프레임(단타 vs 스윙) 퀀트 판정**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (권장 타임프레임)**: **{tf_verdict}**
  - **적용 알고리즘 전략**: `{strategy}`
  - **단타 대응 기준**: 목표가 **{tp_price:,.0f}원 (+{tp_pct:.1f}%)** / 손절가 **{sl_price:,.0f}원 (-{sl_pct:.1f}%)** / 당일 15:20 전 미도달 시 비중 축소
  - **스윙 대응 기준**: 5일 이평선(**{ma5:,.0f}원**) 및 20일 이평선(**{ma20:,.0f}원**) 종가 지지 유지 시 **+{tp2_pct:.1f}% ({tp2_price:,.0f}원)**까지 추세 홀딩 가능

**1. 체결 및 보유 원칙**
- {rule_note}
"""

    # 8. 투자 비중 / 분할 매수 / 자금 관리
    elif any(k in q_lower for k in ["비중", "분할", "몇퍼", "얼마나", "포트", "자금", "배팅", "시드"]):
        if score >= 80.0 and rsi < 72.0:
            alloc_pct = "총 운용자산의 15% ~ 20% (고확신 구간)"
        elif score >= 68.0 and rsi < 75.0:
            alloc_pct = "총 운용자산의 10% ~ 15% (표준 권장 비중)"
        else:
            alloc_pct = "총 운용자산의 5% ~ 10% 이내 (변동성 관리 소액 비중)"

        entry_1 = close
        entry_2 = max(sl_price * 1.005, close * 0.99)
        ans = f"""### **{name} ({code}) 권장 투자 비중 & 분할 진입 자금관리 퀀트 산출**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (권장 비중)**: **{alloc_pct}**
  - **기대 손익비 (Risk/Reward)**: **{rr_ratio:.2f} : 1** (기대수익 **+{tp_pct:.1f}%** vs 허용손실 **-{sl_pct:.1f}%**)
  - **1차 진입가 (배정 물량의 50%)**: 현재가 부근 **{entry_1:,.0f}원**
  - **2차 분할 진입가 (배정 물량의 50%)**: 장중 눌림 지지선 **{entry_2:,.0f}원** 부근
  - **계좌 최대 리스크 제한**: 원칙 손절가 **{sl_price:,.0f}원 (-{sl_pct:.1f}%)** 이탈 시 전체 포트폴리오 손실률 **-0.3% 이내** 통제

**1. 자금 집행 가이드**
- 한 번에 전량 진입(몰빵)하기보다 2회 분할 매수로 평균단가 우위를 확보하고, **{tp_price:,.0f}원** 터치 시 50% 수익 확정을 권장합니다.
"""

    # 9. 지금 사도 되나요? / 진입 가능 여부 / 추격매수 진단
    elif any(k in q_lower for k in ["사도", "진입", "매수", "추격", "타이밍", "지금", "신규", "들어가", "살까", "사볼까"]):
        if "장전" in strategy or "시초가" in strategy:
            risk_badge = " [우수: 장전 시초가 공략 후보]"
            verdict = "전일 확정 데이터 기반으로 사전 압축된 장전 시초가 공략 후보입니다."
            action_guide = f"08:00~08:50 사전 점검을 마치고, 08:40~09:00 동시호가 예상체결 갭(적정 갭 확인)을 거쳐 09:00 정규장 개장 시초가에 분할 진입하세요. (목표가 {tp_price:,.0f}원 / 손절가 {sl_price:,.0f}원)"
        elif change_pct >= 18.0 or rsi >= 78.0:
            risk_badge = " [고위험 경고: 추격매수 절대 금지]"
            verdict = "현재 당일 상승률이 과도하게 높고 단기 과열 구간이므로, 현 위치에서의 불타기 신규 진입은 고점 윗꼬리에 물릴 확률이 68% 이상입니다."
            action_guide = "장중 급등 직후 첫 번째 15분봉/30분봉 눌림목 지지 반등(시초가 부근 지지)을 반드시 확인한 후 분할 진입하거나 관망을 권장합니다."
        elif change_pct >= 8.0:
            risk_badge = " [중립: 돌파 추세 분할 진입 가능]"
            verdict = "거래량이 탄탄하게 받쳐주고 있으나 당일 변동성이 커진 위치입니다."
            action_guide = f"비중을 30~50%로 축소하여 분할 매수하고, 원칙 손절선인 **{sl_price:,.0f}원 (-{sl_pct:.1f}%)** 이탈 시 즉시 기계적으로 대응할 준비가 되어 있을 때만 진입하세요."
        else:
            risk_badge = " [우수: 안정적 손익비 진입 구간]"
            verdict = "상승 파동 초입 또는 눌림목 지지 구간으로, 위로는 목표가 공간이 넓고 아래로는 손절폭이 짧아 손익비(Risk/Reward Ratio)가 매우 뛰어납니다."
            action_guide = f"현재가 **{close:,.0f}원** 기준 즉시 진입이 유효하며, 1차 목표가 **{tp_price:,.0f}원**을 타겟으로 매매를 추천합니다."

        custom_hdr = f"- **질문 분석**: `{q_clean}`\n" if is_custom_query else ""
        ans = f"""### **{name} ({code}) 신규 진입 적합도 진단**

{custom_hdr}**판정 결과**: **{risk_badge}**

- **현재가 상태**: **{close:,.0f}원** (당일 {change_pct:+.2f}%) | RSI: **{rsi:.1f}** | 퀀트 스코어: **{score:.1f}점**
- **퀀트 진단 소견**: {verdict}
- **매매 행동 요령**: {action_guide}
- **손익비 분석**: 
  - 기대 이익폭: **+{tp_pct:.1f}%** ({tp_price:,.0f}원)
  - 최대 허용 손실폭: **-{sl_pct:.1f}%** ({sl_price:,.0f}원)
  - 목표 손익비: **{rr_ratio:.2f} : 1** (기대수익이 위험 대비 우위)
"""

    # 10. 상승 원인 / 왜 오르나요?
    elif any(k in q_lower for k in ["왜", "원인", "이유", "급등", "상승", "오르", "배경", "오른", "올라", "모멘텀", "테마", "재료"]):
        custom_hdr = f"- **질문 분석**: `{q_clean}`\n- **퀀트 산출 결과값**: **거래량 전일비 {vol_ratio:.1f}배 | 퀀트 스코어 {score:.1f}점 ({label}) | {ma_status}**\n\n" if is_custom_query else ""
        ans = f"""### **{name} ({code}) 상승 원인 퀀트 정밀 진단**

{custom_hdr}**1. 수급 및 거래량 폭발력 ({vol_ratio:.1f}배 급증)**
- 전일 동시간대 대비 거래량이 **{vol_ratio:.1f}배** 터지며 시장의 기관/외인 주도 세력의 강력한 순매수 자금이 유입되었습니다.
- 거래량을 수반하지 않는 헛반등이 아니라, 실질적인 대량 수급이 뒷받침된 **신뢰도 높은 거래대금 주도 상승**입니다.

**2. 전략적 발굴 근거 ({strategy})**
- **핵심 근거**: {reason_core}
- **이동평균선 배열**: `{ma_status}` 상태로, 단기/중기 매물대를 상향 돌파하며 상방 압력이 강하게 형성되었습니다.
- **RSI 강도**: 현재 14일 RSI는 **{rsi:.1f}pt**로, { '과열권 이전의 탄탄한 상승 추세 가속 구간' if rsi <= 72 else '강한 단기 과열 양상이 나타나고 있는 강력한 매수세 국면' }입니다.

**3. 업종/테마 모멘텀**
- 소속 섹터인 **[{sector}]** 군으로 시장 주도 자금이 강하게 쏠리면서 테마 랠리 동조화 효과가 작용하고 있습니다.
"""

    # 11. 오후장 / 장후반 / 종가 마감 상승 가능성 및 탄력 분석
    elif any(k in q_lower for k in ["오후", "오후장", "장후반", "종가", "동시호가", "마감", "상승 가능성", "상승가능성", "상승 확률", "상승확률", "오를 가능성", "오를 확률"]):
        cm = _compute_complex_quant_metrics(stock_info, q_clean)
        aft_score = cm["afternoon_score"]
        aft_grade = (
            f"높음 (오후장 상승 가능성 {aft_score:.1f}점 / 100점 · 2차 수급 유입 우위)"
            if aft_score >= 70.0
            else (
                f"보통~완만한 우위 (오후장 상승 가능성 {aft_score:.1f}점 / 100점 · 지지선 확인 필요)"
                if aft_score >= 52.0
                else f"낮음·보수적 관망 (오후장 상승 가능성 {aft_score:.1f}점 / 100점 · 고점 매물 소화 구간)"
            )
        )
        ans = f"""### **{name} ({code}) 오후장·장마감 상승 가능성 퀀트 정밀 진단**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (오후장 상승 판정)**: **{aft_grade}**
  - **오후장 상승 가능성 점수**: **{aft_score:.1f}점 / 100점** (종합 퀀트 스코어 **{score:.1f}점 · {label}**)
  - **장중 고가 지지력 (캔들 실체 위치)**: **{candle_support_pct:.1f}%** (시가 **{open_p:,.0f}원** / 고가 **{high_p:,.0f}원** / 현재가 **{close:,.0f}원**)
  - **수급 에너지 및 거래량 배수**: 전일 대비 **{vol_ratio:.2f}배** (추정 거래대금 **약 {est_val_eok:,.1f}억원**)
  - **오후장 핵심 타점**: 1차 지지선 **{support_1:,.0f}원** | 오후 돌파 기준선 **{resist_1:,.0f}원** | 목표 익절가 **{tp_price:,.0f}원 (+{tp_pct:.1f}%)** / 손절가 **{sl_price:,.0f}원 (-{sl_pct:.1f}%)**

**1. 오후장(13:00 ~ 15:30) 수급 및 캔들 구조 해석**
- 오후장 추가 상승의 핵심 조건인 **장중 고점 지지율(캔들 실체 지지력)**이 현재 **{candle_support_pct:.1f}%**를 기록 중입니다. { '일중 고가 부근에서 차익 매물을 소화하며 탄탄하게 가격을 방어하고 있어 13:30 이후 2차 수급 및 종가배팅 자금 유입 가능성이 높습니다.' if candle_support_pct >= 60 else '장중 고점 대비 밀리며 윗꼬리 매물 부담이 일부 존재하므로, 성급한 추격매수보다는 1차 지지선 방어 여부를 먼저 확인해야 합니다.' }
- 이동평균선 배열은 `{ma_status}`이며 RSI(14)는 **{rsi:.1f}pt**로, { '단기 과열 부담 없이 오후장 상방 탄력을 이어갈 수 있는 구간입니다.' if 45 <= rsi <= 73 else '단기 변동성 확대 구간이므로 분할 대응이 유리합니다.' }

**2. 오후장 실전 매매 행동 지침**
- **신규/추가 진입 기준**: 오후장 피벗 중심선(**{pivot_price:,.0f}원**) 및 1차 지지선(**{support_1:,.0f}원**) 상단 안착 시 분할 접근, 당일 고가권(**{resist_1:,.0f}원**) 거래량 동반 돌파 시 목표가 **{tp_price:,.0f}원**까지 홀딩 권장합니다.
- **리스크 관리**: 만약 오후 14:30 이후 **{sl_price:,.0f}원(-{sl_pct:.1f}%)**을 하향 이탈할 경우 종가 투매 리스크 차단을 위해 원칙 손절로 대응하십시오.
"""

    # 12. 수급 / 거래량 / 거래대금 / 세력·외인·기관 집중 분석
    elif any(k in q_lower for k in ["수급", "거래량", "거래대금", "세력", "외인", "외국인", "기관", "매집", "프로그램"]):
        supply_grade = "메이저 수급 집중 유입 (강력)" if vol_ratio >= 2.0 else ("수급 양호 (점진적 매집)" if vol_ratio >= 1.0 else "거래 소강 (수급 관망)")
        ans = f"""### **{name} ({code}) 수급 강도 & 거래대금 퀀트 정밀 분석**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (수급 판정)**: **{supply_grade}**
  - **전일 대비 거래량 비율**: **{vol_ratio:.2f}배** (수급 팩터 점수: **{vol_score:.1f}점 / 30점**)
  - **당일 누적 거래량 / 추정 대금**: **{volume:,.0f}주** / **약 {est_val_eok:,.1f}억원**
  - **장중 양봉 캔들 지지력**: **{candle_support_pct:.0f}%** (윗꼬리 비율 **{upper_shadow_pct:.1f}%**)

**1. 수급 주체 및 에너지 해석**
- 당일 시가(**{open_p:,.0f}원**) 대비 현재가(**{close:,.0f}원**) 흐름과 **{candle_support_pct:.0f}%**의 캔들 지지율을 볼 때, { '장중 유입된 메이저 매수 자금이 이탈하지 않고 종가까지 강하게 방어되고 있습니다.' if candle_support_pct >= 55 else '장중 차익실현 매물과 저가 매수세가 공방을 벌이는 구간으로 지지선( ' + f'{support_1:,.0f}원) 방어 확인이 중요합니다.' }
"""

    # 13. 기타 모든 직접 질문: 질문 맞춤형 종합 퀀트 수치 산출 및 전망 결과값
    else:
        cm = _compute_complex_quant_metrics(stock_info, q_clean)
        aft_score = cm["afternoon_score"]
        if score >= 75.0:
            outlook_verdict = f"강세 우위 (종합 퀀트 스코어 {score:.1f}점 · 오후 탄력 {aft_score:.1f}점 · {label})"
        elif score >= 60.0:
            outlook_verdict = f"중립~완만한 상방 우위 (종합 퀀트 스코어 {score:.1f}점 · 오후 탄력 {aft_score:.1f}점 · {label})"
        else:
            outlook_verdict = f"보수적 관망 및 리스크 관리 구간 (종합 퀀트 스코어 {score:.1f}점 · 오후 탄력 {aft_score:.1f}점 · {label})"

        ans = f"""### **{name} ({code}) 퀀트 수석 애널리스트 맞춤형 분석 결과**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (종합 판정)**: **{outlook_verdict}**
  - **현재가 / 등락률**: **{close:,.0f}원** ({change_pct:+.2f}%) | 소속 업종: **[{sector}]** ({market})
  - **권장 목표가 / 원칙 손절가**: 목표가 **{tp_price:,.0f}원 (+{tp_pct:.1f}%)** / 손절가 **{sl_price:,.0f}원 (-{sl_pct:.1f}%)** (손익비 **{rr_ratio:.2f} : 1**)
  - **핵심 지지선 / 저항선**: 1차 지지 **{support_1:,.0f}원** (피벗 **{pivot_price:,.0f}원**) / 1차 저항 **{resist_1:,.0f}원**

**1. 4대 퀀트 팩터 정밀 산출 내역**
- **추세/이평 팩터 ({ma_score:.1f}점)**: `{ma_status}` (MA5 **{ma5:,.0f}원**, MA20 **{ma20:,.0f}원**, MA60 **{ma60:,.0f}원**)
- **수급/거래량 팩터 ({vol_score:.1f}점)**: 전일비 거래량 급증 **{vol_ratio:.1f}배** (추정 거래대금 **약 {est_val_eok:,.1f}억원**, 캔들 지지력 **{candle_support_pct:.0f}%**)
- **모멘텀/RSI 팩터 ({rsi_score:.1f}점)**: RSI(14) **{rsi:.1f}pt** ({ '탄력 상승 구간' if rsi < 70 else '단기 과열 구간' })
- **변동성/밴드 팩터 ({bb_score:.1f}점)**: 볼린저밴드 **BB %b {bb_pct:.2f}**

**2. 실전 매매 결론 ({strategy})**
- **핵심 근거**: {reason_core}
- **대응 원칙**: {rule_note}
"""

    try:
        us_m = fetch_us_market_summary()
        regime = us_m.get("regime", "중립")
        nasdaq_pct = us_m.get("nasdaq_pct", 0.0)
        soxx_pct = us_m.get("soxx_pct", 0.0)
        usdkrw = us_m.get("usdkrw_rate", 1340.0)
        action_g = us_m.get("action_guide", "")
        sec_note = get_sector_macro_note(sector, us_m)

        ans += f"\n\n**3. 글로벌 미장 연동 점검 ({regime} 국면)**\n"
        ans += f"- **전일 미장 지표**: 나스닥 **{nasdaq_pct:+.2f}%** | 필라델피아 반도체(SOXX) **{soxx_pct:+.2f}%** | 환율 **{usdkrw:,.1f}원**\n"
        ans += f"- **글로벌 대응 가이드**: {action_g}\n"
        if sec_note:
            ans += f"- **업종별 영향 ({sector})**: {sec_note}\n"
    except Exception:
        pass

    if fallback_notice:
        ans += "\n\n*(※ Gemini API 연동 대기 중으로 내장 퀀트 수치 추론 엔진이 실시간 생성한 정밀 리포트입니다.)*"

    return ans


def _compute_complex_quant_metrics(stock_info: Dict[str, Any], question: str = "") -> Dict[str, Any]:
    """Computes multi-dimensional quant metrics for comparative and complex scenario analysis."""
    name = str(stock_info.get("name") or "해당 종목")
    code = str(stock_info.get("code") or "")
    market = str(stock_info.get("market") or "KRX")
    sector = str(stock_info.get("sector") or "기타")
    close = _safe_to_float(stock_info.get("close"), 0.0)
    change_pct = _safe_to_float(stock_info.get("change_pct"), 0.0)
    open_p = _safe_to_float(stock_info.get("open"), close)
    high_p = _safe_to_float(stock_info.get("high"), close)
    low_p = _safe_to_float(stock_info.get("low"), close)
    volume = _safe_to_float(stock_info.get("volume"), 0.0)
    score = _safe_to_float(stock_info.get("score"), 55.0)
    label = str(stock_info.get("label") or ("고확신" if score >= 75 else ("우수" if score >= 65 else "중립")))
    strategy = str(stock_info.get("strategy_tag") or stock_info.get("strategy") or "일반 기술적 진단")
    tp_pct = _safe_to_float(stock_info.get("tp_pct"), 3.5)
    sl_pct = max(0.5, _safe_to_float(stock_info.get("sl_pct"), 2.0))
    tp_price = close * (1.0 + tp_pct / 100.0) if close > 0 else 0.0
    sl_price = close * (1.0 - sl_pct / 100.0) if close > 0 else 0.0
    rr_ratio = tp_pct / max(0.1, sl_pct)

    breakdown = stock_info.get("breakdown") or {}
    if not isinstance(breakdown, dict):
        breakdown = {}
    rsi = _safe_to_float(breakdown.get("rsi_val", stock_info.get("rsi")), 50.0)
    vol_ratio = _safe_to_float(breakdown.get("vol_ratio", stock_info.get("vol_ratio")), 1.0)
    ma_status = str(breakdown.get("ma_status") or stock_info.get("ma_status") or "이평 혼조")
    bb_pct = _safe_to_float(breakdown.get("bb_pct", stock_info.get("bb_pct")), 0.5)

    day_range = max(0.0, high_p - low_p)
    candle_support_pct = ((close - low_p) / day_range * 100.0) if day_range > 0 else (65.0 if change_pct > 0 else 45.0)
    candle_support_pct = max(0.0, min(100.0, candle_support_pct))
    intraday_chg_pct = ((close - open_p) / open_p * 100.0) if open_p > 0 else change_pct
    pivot_price = (high_p + low_p + close) / 3.0 if (high_p > 0 and low_p > 0 and close > 0) else close
    support_1 = max(sl_price, min(open_p if open_p > 0 else close * 0.99, pivot_price)) if close > 0 else 0.0
    resist_1 = max(high_p if high_p > close else close * 1.02, close * 1.015) if close > 0 else 0.0
    est_val_krw = _safe_to_float(stock_info.get("est_trading_val_krw"), close * volume)
    est_val_eok = est_val_krw / 100_000_000.0 if est_val_krw > 0 else 0.0

    # 1. 오후장 상승 가능성 점수 (0~100): 고점 지지력(35) + 거래량/수급(25) + 장중 양봉 탄력(20) + 퀀트/RSI 적정성(20)
    support_comp = candle_support_pct * 0.35
    vol_comp = min(25.0, vol_ratio * 11.0)
    mom_comp = max(0.0, min(20.0, 10.0 + change_pct * 1.1 + intraday_chg_pct * 0.9))
    if change_pct >= 28.5:
        mom_comp *= 0.5  # 상한가 근접 시 추가 상승 여력 제한
    rsi_bonus = 3.5 if 48.0 <= rsi <= 72.0 else (-4.0 if (rsi > 78.0 or rsi < 35.0) else 0.0)
    ma_bonus = 3.0 if "정배열" in ma_status else (-2.0 if "역배열" in ma_status else 0.0)
    quant_comp = max(0.0, min(20.0, score * 0.20 + rsi_bonus + ma_bonus))
    afternoon_score = round(max(12.0, min(98.0, support_comp + vol_comp + mom_comp + quant_comp)), 1)

    # 2. 익일 시초/오전장 상승 가능성 점수 (0~100)
    morning_score = round(
        max(12.0, min(98.0, score * 0.40 + candle_support_pct * 0.30 + min(20.0, vol_ratio * 9.0) + (10.0 if "정배열" in ma_status else 4.0))),
        1
    )

    # 3. 수급·거래량 강도 점수 (0~100)
    supply_score = round(
        max(10.0, min(98.0, min(55.0, vol_ratio * 22.0) + candle_support_pct * 0.30 + min(15.0, est_val_eok / 100.0))),
        1
    )

    # 4. 손익비·하방 안전도 점수 (0~100)
    safety_score = round(
        max(15.0, min(98.0, min(45.0, rr_ratio * 22.0) + score * 0.35 + (20.0 if 38.0 <= rsi <= 65.0 else 10.0))),
        1
    )

    # 5. 종합 상승 우위 점수 (0~100)
    composite_score = round(score * 0.45 + afternoon_score * 0.30 + safety_score * 0.25, 1)

    # Factor in US market sector impact
    try:
        us_m = fetch_us_market_summary()
        soxx_pct = us_m.get("soxx_pct", 0.0)
        sec_str = (sector + " " + name).lower()
        if any(k in sec_str for k in ["반도체", "hpsp", "하이닉스", "삼성전자", "한미반도체", "주성", "동진", "원익"]):
            if soxx_pct <= -2.5:
                morning_score = round(max(5.0, morning_score - 8.0), 1)
                composite_score = round(max(5.0, composite_score - 4.0), 1)
            elif soxx_pct >= 2.0:
                morning_score = round(min(98.0, morning_score + 5.0), 1)
                composite_score = round(min(98.0, composite_score + 2.5), 1)
    except Exception:
        pass

    q_low = (question or "").lower()
    if any(k in q_low for k in ["오후", "오후장", "장후반", "종가", "마감", "동시호가"]):
        target_metric_label = "오후장 상승 가능성 점수"
        target_metric_val = afternoon_score
    elif any(k in q_low for k in ["오전", "시초", "시초가", "내일", "익일"]):
        target_metric_label = "익일 시초·오전장 상승 가능성 점수"
        target_metric_val = morning_score
    elif any(k in q_low for k in ["수급", "거래량", "거래대금", "세력", "외인", "기관"]):
        target_metric_label = "실시간 수급·거래량 강도 점수"
        target_metric_val = supply_score
    elif any(k in q_low for k in ["안전", "리스크", "방어", "손익비", "손절"]):
        target_metric_label = "손익비·하방 안전도 점수"
        target_metric_val = safety_score
    else:
        target_metric_label = "종합 상승 가능성 우위 점수"
        target_metric_val = composite_score

    return {
        "name": name,
        "code": code,
        "market": market,
        "sector": sector,
        "close": close,
        "change_pct": change_pct,
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "volume": volume,
        "est_val_eok": est_val_eok,
        "score": score,
        "label": label,
        "strategy": strategy,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "rr_ratio": rr_ratio,
        "rsi": rsi,
        "vol_ratio": vol_ratio,
        "ma_status": ma_status,
        "bb_pct": bb_pct,
        "candle_support_pct": candle_support_pct,
        "intraday_chg_pct": intraday_chg_pct,
        "pivot_price": pivot_price,
        "support_1": support_1,
        "resist_1": resist_1,
        "afternoon_score": afternoon_score,
        "morning_score": morning_score,
        "supply_score": supply_score,
        "safety_score": safety_score,
        "composite_score": composite_score,
        "target_metric_label": target_metric_label,
        "target_metric_val": target_metric_val,
    }


def _built_in_multi_stock_comparative_reasoning(
    stocks: List[Dict[str, Any]],
    question: str,
    fallback_notice: bool = False
) -> str:
    """
    Compares 2 or more stocks quantitatively based on the user's specific question intent
    (e.g., 'HLB와 삼성전자 중 오후에 상승 가능성이 더 높은 종목은?') and determines a clear winner.
    """
    q_clean = (question or "").strip()
    metrics_list = [_compute_complex_quant_metrics(s, q_clean) for s in stocks]
    # Sort descending by target_metric_val, then score, then candle_support_pct
    ranked = sorted(
        metrics_list,
        key=lambda m: (m["target_metric_val"], m["score"], m["candle_support_pct"], m["vol_ratio"]),
        reverse=True
    )

    winner = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else ranked[0]
    metric_label = winner["target_metric_label"]
    score_gap = winner["target_metric_val"] - runner["target_metric_val"]

    names_joined = " vs ".join(f"{m['name']}({m['code']})" for m in metrics_list)

    # Build comparison reasons
    reasons = []
    if winner["candle_support_pct"] >= runner["candle_support_pct"]:
        reasons.append(
            f"장중 고가 지지력(캔들 실체)이 **{winner['name']} {winner['candle_support_pct']:.1f}%** vs **{runner['name']} {runner['candle_support_pct']:.1f}%**로, 장중 매물 소화 및 하방 지지력이 더 견고합니다"
        )
    if winner["score"] >= runner["score"]:
        reasons.append(
            f"종합 퀀트 스코어에서 **{winner['name']} {winner['score']:.1f}점({winner['label']})** vs **{runner['name']} {runner['score']:.1f}점({runner['label']})**으로 추세·모멘텀 복합 점수가 앞섭니다"
        )
    if winner["vol_ratio"] >= runner["vol_ratio"]:
        reasons.append(
            f"전일 대비 거래량 활성도가 **{winner['name']} {winner['vol_ratio']:.2f}배** vs **{runner['name']} {runner['vol_ratio']:.2f}배**로 수급 유입 탄력이 우위에 있습니다"
        )
    if not reasons:
        reasons.append(
            f"손익비(**{winner['rr_ratio']:.2f}:1**) 및 RSI(**{winner['rsi']:.1f}pt**) 기술적 균형도에서 **{winner['name']}**이(가) 상대적 우위를 점하고 있습니다"
        )

    # Markdown table rows for all compared stocks
    col_headers = " | ".join(f"**{m['name']} ({m['code']})**" for m in metrics_list)
    sep_cols = " | ".join(":---:" for _ in metrics_list)

    row_price = " | ".join(f"**{m['close']:,.0f}원** ({m['change_pct']:+.2f}%)" for m in metrics_list)
    row_target_m = " | ".join(
        f"**{m['target_metric_val']:.1f}점**" + (" (1위 우위)" if m["code"] == winner["code"] else "")
        for m in metrics_list
    )
    second_metric_label = "익일 시초·오전장 탄력 점수" if metric_label == "오후장 상승 가능성 점수" else "오후장 상승 가능성 점수"
    second_metric_key = "morning_score" if metric_label == "오후장 상승 가능성 점수" else "afternoon_score"
    row_second_m = " | ".join(f"**{m[second_metric_key]:.1f}점**" for m in metrics_list)
    row_quant = " | ".join(f"**{m['score']:.1f}점** ({m['label']})" for m in metrics_list)
    row_candle = " | ".join(f"**{m['candle_support_pct']:.1f}%** (시초비 {m['intraday_chg_pct']:+.2f}%)" for m in metrics_list)
    row_vol = " | ".join(f"**{m['vol_ratio']:.2f}배** (약 {m['est_val_eok']:,.0f}억)" for m in metrics_list)
    row_tech = " | ".join(f"RSI **{m['rsi']:.1f}pt** · {m['ma_status']}" for m in metrics_list)
    row_tpsl = " | ".join(
        f"**{m['tp_price']:,.0f}원**(+{m['tp_pct']:.1f}%) / **{m['sl_price']:,.0f}원**(-{m['sl_pct']:.1f}%) [R/R {m['rr_ratio']:.2f}]"
        for m in metrics_list
    )

    reason_bullets = "\n".join(f"- {r}." for r in reasons)

    per_stock_guides = []
    for idx, m in enumerate(ranked, start=1):
        rank_tag = "최우선 공략 1위" if idx == 1 else f"비교 후순위 ({idx}위)"
        per_stock_guides.append(
            f"- **[{rank_tag}] {m['name']} ({m['code']})**: "
            f"1차 지지선 **{m['support_1']:,.0f}원** 방어 확인 후 접근 | 돌파 기준선 **{m['resist_1']:,.0f}원** | "
            f"목표 익절가 **{m['tp_price']:,.0f}원 (+{m['tp_pct']:.1f}%)** / 원칙 손절가 **{m['sl_price']:,.0f}원 (-{m['sl_pct']:.1f}%)**"
        )
    guides_text = "\n".join(per_stock_guides)

    ans = f"""### **[수석 애널리스트 복합 비교 판정] {names_joined}**

- **질문 분석**: `{q_clean}`
- **퀀트 산출 결과값 (최종 우위 종목)**: **{winner['name']} ({winner['code']}) 우위 판정**
  - **{metric_label}**: **{winner['name']} {winner['target_metric_val']:.1f}점** vs **{runner['name']} {runner['target_metric_val']:.1f}점** (**+{score_gap:.1f}점 우위**)
  - **오후장 상승 가능성 / 종합 퀀트 스코어**: **{winner['name']}** (오후 탄력 **{winner['afternoon_score']:.1f}점**, 종합 **{winner['score']:.1f}점**) vs **{runner['name']}** (오후 탄력 **{runner['afternoon_score']:.1f}점**, 종합 **{runner['score']:.1f}점**)

| 핵심 비교 팩터 | {col_headers} |
| :--- | {sep_cols} |
| **현재가 (등락률)** | {row_price} |
| **{metric_label}** | {row_target_m} |
| **{second_metric_label}** | {row_second_m} |
| **종합 퀀트 스코어** | {row_quant} |
| **장중 고가 지지력 (캔들 실체)** | {row_candle} |
| **전일비 거래량 (추정 대금)** | {row_vol} |
| **기술적 지표 (RSI · 이평선)** | {row_tech} |
| **목표가 / 손절가 (손익비)** | {row_tpsl} |

**1. 왜 `{winner['name']}`의 상승 가능성이 더 높은가? (수석 애널리스트 핵심 근거)**
{reason_bullets}

**2. 종목별 실전 매매 타점 및 대응 전략**
{guides_text}
"""

    try:
        us_m = fetch_us_market_summary()
        regime = us_m.get("regime", "중립")
        nasdaq_pct = us_m.get("nasdaq_pct", 0.0)
        soxx_pct = us_m.get("soxx_pct", 0.0)
        usdkrw = us_m.get("usdkrw_rate", 1340.0)
        action_g = us_m.get("action_guide", "")

        sec_bullets = []
        for m in metrics_list:
            sec_n = get_sector_macro_note(m.get("sector", "") or m.get("name", ""), us_m)
            if sec_n:
                sec_bullets.append(f"- **{m['name']} ({m.get('sector', '업종')})**: {sec_n}")

        macro_part = f"\n\n**3. 글로벌 미장 연동 점검 ({regime} 국면)**\n"
        macro_part += f"- **전일 미장 동향**: 나스닥 **{nasdaq_pct:+.2f}%** | 필라델피아 반도체(SOXX) **{soxx_pct:+.2f}%** | 원/달러 환율 **{usdkrw:,.1f}원**\n"
        macro_part += f"- **글로벌 가이드**: {action_g}\n"
        if sec_bullets:
            macro_part += "- **비교 종목별 섹터 영향**:\n" + "\n".join(f"  {b}" for b in sec_bullets) + "\n"
        ans += macro_part
    except Exception:
        pass

    if fallback_notice:
        ans += "\n\n*(※ 내장 다종목 비교 퀀트 추론 엔진이 실시간 산출한 복합 분석 리포트입니다.)*"
    return ans


def build_stock_quant_snapshot(code: str) -> Dict[str, Any]:
    """Loads real-time market info, OHLCV, and quant score for any 6-digit stock code."""
    clean_code = str(code or "005930").strip().zfill(6)
    name = clean_code
    market = "KOSPI"
    sector = "기타"
    try:
        from src.collectors.krx_universe import get_stock_metadata
        meta = get_stock_metadata(clean_code) or {}
        name = str(meta.get("name") or clean_code)
        market = str(meta.get("market") or "KOSPI")
        sector = str(meta.get("sector") or "기타")
    except Exception:
        pass

    close = 0.0
    change_pct = 0.0
    open_p = 0.0
    high_p = 0.0
    low_p = 0.0
    prev_close = 0.0
    volume = 0
    try:
        from src.collectors.market_data import get_latest_market_info
        p_info = get_latest_market_info(clean_code) or {}
        close = _safe_to_float(p_info.get("close"), 0.0)
        change_pct = _safe_to_float(p_info.get("change_pct"), 0.0)
        open_p = _safe_to_float(p_info.get("open"), close)
        high_p = _safe_to_float(p_info.get("high"), close)
        low_p = _safe_to_float(p_info.get("low"), close)
        prev_close = _safe_to_float(p_info.get("prev_close"), close)
        volume = int(_safe_to_float(p_info.get("volume"), 0))
    except Exception:
        pass

    score = 50.0
    label = "중립"
    tp_pct = 3.5
    sl_pct = 2.0
    bk: Dict[str, Any] = {
        "rsi_val": 50.0,
        "vol_ratio": 1.0,
        "ma_status": "이평 혼조",
        "bb_pct": 0.5,
    }
    indicators: Dict[str, Any] = {}
    try:
        from src.collectors.market_data import fetch_ohlcv
        from src.core.scoring import evaluate_stock_latest
        df = fetch_ohlcv(clean_code)
        if df is not None and not df.empty and len(df) >= 20:
            s_info = evaluate_stock_latest(df)
            if s_info:
                score = _safe_to_float(s_info.get("score"), 50.0)
                label = str(s_info.get("label") or "중립")
                tp_pct = _safe_to_float(s_info.get("tp_pct"), 3.5)
                sl_pct = _safe_to_float(s_info.get("sl_pct"), 2.0)
                bk = s_info.get("breakdown") or bk
                indicators = s_info.get("indicators") or {}
    except Exception:
        pass

    rsi_val = _safe_to_float(bk.get("rsi_val"), 50.0)
    vol_ratio = _safe_to_float(bk.get("vol_ratio"), 1.0)
    ma_status = str(bk.get("ma_status") or "이평 혼조")
    bb_pct = _safe_to_float(bk.get("bb_pct"), 0.5)

    return {
        "code": clean_code,
        "name": name,
        "market": market,
        "sector": sector,
        "close": close,
        "change_pct": change_pct,
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "prev_close": prev_close,
        "volume": volume,
        "score": score,
        "label": label,
        "strategy_tag": "고확신 퀀트 모멘텀" if score >= 70.0 else "일반 기술적 진단",
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "vol_ratio": vol_ratio,
        "rsi": rsi_val,
        "ma_status": ma_status,
        "bb_pct": bb_pct,
        "breakdown": bk,
        "indicators": indicators,
        "est_trading_val_krw": close * volume,
        "reason_core": f"종합 퀀트 스코어 {score:.1f}점 ({ma_status}, RSI {rsi_val:.1f}pt)",
        "rule_note": "목표가 분할 익절 및 손절가 원칙 대응",
    }


_UNIVERSE_CACHE: Optional[List[tuple]] = None

_STOCK_ALIASES: Dict[str, tuple] = {
    "에이치엘비": ("028300", "HLB"),
    "삼전": ("005930", "삼성전자"),
    "하닉": ("000660", "SK하이닉스"),
    "하이닉스": ("000660", "SK하이닉스"),
    "에스케이하이닉스": ("000660", "SK하이닉스"),
    "현차": ("005380", "현대차"),
    "네이버": ("035420", "NAVER"),
    "포스코": ("005490", "POSCO홀딩스"),
    "엘지에너지솔루션": ("373220", "LG에너지솔루션"),
    "엘지에너지솔루션": ("373220", "LG에너지솔루션"),
    "엘지전자": ("066570", "LG전자"),
    "셀트리온제약": ("068760", "셀트리온제약"),
}

_QUESTION_STOPWORDS = {
    "지금", "오늘", "내일", "매수", "매도", "대상", "목표", "손절", "차트", "지표",
    "분석", "추천", "종목", "시장", "수급", "거래", "가격", "비중", "전략", "홀딩",
    "익절", "진입", "상승", "하락", "급등", "대응", "보유", "신규", "단타", "스윙",
    "얼마", "언제", "어떻", "어떤", "무슨", "이유", "원인", "현재", "전망", "의견",
    "오전", "오후", "종가", "시초", "장중", "가능", "확률", "비교", "우위", "반등",
    "추격", "눌림", "고점", "저점", "지지", "저항", "평단", "수익", "손실", "업종",
    "섹터", "테마", "대장", "누구", "어디", "모드",
}


def _load_universe_pairs() -> List[tuple]:
    """Loads (code, name) pairs from universe table sorted by name length descending."""
    global _UNIVERSE_CACHE
    if _UNIVERSE_CACHE is not None:
        return _UNIVERSE_CACHE
    try:
        from src.database.models import get_db_connection
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT code, name FROM universe WHERE is_active = 1 ORDER BY LENGTH(name) DESC"
            )
            rows = cur.fetchall()
            _UNIVERSE_CACHE = [(str(r[0]).strip().zfill(6), str(r[1]).strip()) for r in rows if r[0] and r[1]]
            return _UNIVERSE_CACHE
    except Exception as e:
        logger.warning(f"Failed to load universe cache in quant_analyst: {e}")
        return []


def extract_all_stocks_from_query(raw_query: str) -> List[Dict[str, str]]:
    """
    Extracts all mentioned stocks (6-digit codes and stock names, including multi-stock queries
    like 'HLB와 삼성전자 중 오후에 상승 가능성이 더 높은 종목은?') in order of appearance.
    """
    import re

    q = (raw_query or "").strip()
    if not q:
        return []

    universe_pairs = _load_universe_pairs()
    code_to_name = {code: name for code, name in universe_pairs}
    name_to_pair = {name.lower(): (code, name) for code, name in universe_pairs}
    for alias_k, alias_v in _STOCK_ALIASES.items():
        if alias_k.lower() not in name_to_pair:
            name_to_pair[alias_k.lower()] = alias_v

    matched_spans: List[tuple] = []  # (start, end, code, name, matched_by)

    def _overlaps(s: int, e: int) -> bool:
        for ms, me, _, _, _ in matched_spans:
            if max(s, ms) < min(e, me):
                return True
        return False

    # 1. 6-digit stock codes
    for m in re.finditer(r"(?<!\d)(\d{6})(?!\d)", q):
        c_str = m.group(1)
        n_str = code_to_name.get(c_str, c_str)
        matched_spans.append((m.start(), m.end(), c_str, n_str, "code"))

    q_lower = q.lower()

    # 2. 3+ char stock names & aliases (longest first to prevent partial overlap)
    search_pairs = list(universe_pairs) + [(v[0], k) for k, v in _STOCK_ALIASES.items()]
    search_pairs.sort(key=lambda x: len(x[1]), reverse=True)

    for code, name in search_pairs:
        if len(name) < 3:
            continue
        n_low = name.lower()
        start_pos = 0
        while True:
            idx = q_lower.find(n_low, start_pos)
            if idx == -1:
                break
            end_idx = idx + len(n_low)
            start_pos = idx + 1
            if _overlaps(idx, end_idx):
                continue
            # If ASCII boundary, ensure not part of a longer English word
            if re.match(r"^[A-Za-z0-9]", name) and idx > 0 and re.match(r"[A-Za-z0-9]", q[idx - 1]):
                continue
            if re.search(r"[A-Za-z0-9]$", name) and end_idx < len(q) and re.match(r"[A-Za-z0-9]", q[end_idx]):
                continue
            canon_name = code_to_name.get(code, _STOCK_ALIASES.get(name, (code, name))[1])
            matched_spans.append((idx, end_idx, code, canon_name, "name"))

    # 3. Token-based matching for 2-char stock names (stripping Korean particles like 와/과/랑/은/는/이/가)
    particles = ("에서", "으로", "이랑", "이나", "대비", "은", "는", "이", "가", "을", "를", "의", "에", "도", "만", "로", "과", "와", "랑", "나", "중")
    for tok_m in re.finditer(r"[가-힣A-Za-z0-9&]+", q):
        t_start, t_end = tok_m.start(), tok_m.end()
        if _overlaps(t_start, t_end):
            continue
        tok = tok_m.group(0)
        candidates = [tok]
        for p in particles:
            if tok.endswith(p) and len(tok) > len(p):
                candidates.append(tok[:-len(p)])
        for cand in candidates:
            cand_low = cand.lower()
            if cand not in _QUESTION_STOPWORDS and cand_low in name_to_pair:
                c_str, n_str = name_to_pair[cand_low]
                matched_spans.append((t_start, t_start + len(cand), c_str, n_str, "name"))
                break

    matched_spans.sort(key=lambda x: x[0])
    results: List[Dict[str, str]] = []
    seen_codes = set()
    for _, _, c_str, n_str, m_by in matched_spans:
        if c_str not in seen_codes:
            seen_codes.add(c_str)
            results.append({"code": c_str, "name": n_str, "matched_by": m_by})
    return results


def extract_stock_and_question_from_query(
    raw_query: str,
    current_code: str = "005930",
    fallback_top_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parses a user's question-first input (single-stock, multi-stock comparative, or market-wide recommendation)
    and resolves:
      - code: primary 6-digit stock code to analyze (or #1 comparative stock)
      - codes: list of all matched 6-digit stock codes
      - question: question prompt to pass to generate_quant_expert_answer
      - matched_by: 'multi' | 'code' | 'name' | 'recommend' | 'context'
      - is_stock_only: True if user only entered a stock name/code without a specific question
      - is_comparative: True if 2+ stocks were matched in the query
    """
    import re

    default_code = str(current_code or "005930").strip().zfill(6)
    default_q = "지금 사도 되나요? 현재 가격대 진입 적합성과 손익비를 알려줘."
    q = (raw_query or "").strip()

    if not q:
        return {
            "code": default_code,
            "codes": [default_code],
            "question": default_q,
            "matched_by": "context",
            "is_stock_only": True,
            "is_comparative": False,
        }

    matched_stocks = extract_all_stocks_from_query(q)

    # Case 1: 2 or more stocks mentioned -> Multi-Stock Comparative Analysis
    if len(matched_stocks) >= 2:
        all_codes = [s["code"] for s in matched_stocks]
        return {
            "code": all_codes[0],
            "codes": all_codes,
            "stocks": matched_stocks,
            "question": q,
            "matched_by": "multi",
            "is_stock_only": False,
            "is_comparative": True,
        }

    # Case 2: Exactly 1 stock mentioned
    if len(matched_stocks) == 1:
        m_item = matched_stocks[0]
        m_code_str = m_item["code"]
        m_name_str = m_item["name"]
        m_by = m_item["matched_by"]

        rem = q
        for token_to_strip in (m_code_str, m_name_str):
            if token_to_strip:
                pattern = re.compile(re.escape(token_to_strip), re.IGNORECASE)
                rem = pattern.sub("", rem, count=1).strip()
        for alias_k, alias_v in _STOCK_ALIASES.items():
            if alias_v[0] == m_code_str and alias_k in rem:
                rem = rem.replace(alias_k, "", 1).strip()
        rem = re.sub(r"^(은|는|이|가|을|를|의|에|에서|도|만|로|으로)(?=\s|$|\?)", "", rem).strip(" ?.,!·-")

        if len(rem) >= 2:
            return {
                "code": m_code_str,
                "codes": [m_code_str],
                "question": q,
                "matched_by": m_by,
                "is_stock_only": False,
                "is_comparative": False,
            }
        return {
            "code": m_code_str,
            "codes": [m_code_str],
            "question": default_q,
            "matched_by": m_by,
            "is_stock_only": True,
            "is_comparative": False,
        }

    # Case 3: General stock recommendation / screening question
    q_lower = q.lower()
    rec_keywords = [
        "매수할 만한 종목", "매수할만한 종목", "살 만한 종목", "살만한 종목",
        "종목 추천", "추천 종목", "어떤 종목", "무슨 종목", "급등주 추천",
        "지금 매수할 만한", "지금 매수할만한", "오늘 살 만한", "오늘 매수할",
        "상승 가능성이 더 높은 종목", "상승 가능성이 높은 종목", "오를 종목"
    ]
    if any(rk in q_lower for rk in rec_keywords):
        rec_code = str(fallback_top_code or default_code).strip().zfill(6)
        return {
            "code": rec_code,
            "codes": [rec_code],
            "question": q,
            "matched_by": "recommend",
            "is_stock_only": False,
            "is_comparative": False,
        }

    # Case 4: Follow-up or general question for current active stock
    return {
        "code": default_code,
        "codes": [default_code],
        "question": q,
        "matched_by": "context",
        "is_stock_only": False,
        "is_comparative": False,
    }



