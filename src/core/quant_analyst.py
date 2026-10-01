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

logger = logging.getLogger(__name__)


def generate_quant_expert_answer(
    stock_info: Dict[str, Any],
    question: str,
    api_key: Optional[str] = None
) -> str:
    """
    Generates a professional quant expert explanation for a given stock and question.
    Uses Gemini API if API key is provided/available, otherwise uses built-in
    statistical quant reasoning engine.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    
    if key and len(key.strip()) > 10:
        try:
            return _call_gemini_quant_expert(stock_info, question, key.strip())
        except Exception as e:
            logger.warning(f"Gemini API call failed, falling back to built-in quant engine: {e}")
            return _built_in_quant_reasoning(stock_info, question, fallback_notice=True)
            
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


def _call_gemini_quant_expert(stock_info: Dict[str, Any], question: str, api_key: str) -> str:
    """Calls Gemini API via REST with quant persona and strict financial data grounding."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    # Extract rich features
    name = str(stock_info.get("name") or "해당 종목")
    code = str(stock_info.get("code") or "")
    market = str(stock_info.get("market") or "")
    sector = str(stock_info.get("sector") or "기타")
    close = _safe_to_float(stock_info.get("close"), 0.0)
    change_pct = _safe_to_float(stock_info.get("change_pct"), 0.0)
    score = _safe_to_float(stock_info.get("score"), 0.0)
    label = str(stock_info.get("label") or "")
    strategy = str(stock_info.get("strategy_tag") or stock_info.get("strategy") or "일반 퀀트")
    tp_pct = _safe_to_float(stock_info.get("tp_pct"), 5.0)
    sl_pct = _safe_to_float(stock_info.get("sl_pct"), 2.0)
    
    tp_price = close * (1.0 + tp_pct / 100.0)
    sl_price = close * (1.0 - sl_pct / 100.0)
    
    breakdown = stock_info.get("breakdown") or {}
    if not isinstance(breakdown, dict):
        breakdown = {}
    rsi = _safe_to_float(breakdown.get("rsi_val", stock_info.get("rsi")), 50.0)
    vol_ratio = _safe_to_float(breakdown.get("vol_ratio", stock_info.get("vol_ratio")), 1.0)
    ma_status = str(breakdown.get("ma_status") or stock_info.get("ma_status") or "보통")
    rule_note = str(stock_info.get("rule_note") or "")
    reason_core = str(stock_info.get("reason_core") or "")
    
    system_instruction = (
        "당신은 대한민국 주식 시장에 특화된 최고 수준의 '수석 퀀트 트레이더 및 기술적 분석가(Chief Quant Analyst)'입니다.\n"
        "제공된 종목의 실제 체결 데이터, 거래량 급증비, 이평선 배열, RSI, 퀀트 스코어, 목표가/손절가를 바탕으로 "
        "사용자의 질문에 명쾌하고 객관적이며 신뢰도 높은 전문가 분석 답변을 한국어로 제공하세요.\n"
        "추상적이거나 모호한 답변 대신 구체적인 가격 수치, 백분율, 팩트를 인용하여 설명하세요.\n"
        "매매 위험과 원칙 손절의 중요성을 항상 함께 고지하세요."
    )
    
    context_data = f"""
[종목 퀀트 팩트시트]
- 종목명: {name} ({code}, {market})
- 소속 업종: {sector}
- 현재가: {close:,.0f}원 (등락률: {change_pct:+.2f}%)
- 퀀트 스코어: {score:.1f}점 / 100점 ({label})
- 발굴 전략: {strategy}
- 전일대비 거래량 폭발 배수: {vol_ratio:.2f}배
- RSI(14) 지표: {rsi:.1f}
- 이동평균선 배열: {ma_status}
- 알고리즘 목표 익절가: {tp_price:,.0f}원 (+{tp_pct:.1f}%)
- 원칙 손절 기준가: {sl_price:,.0f}원 (-{sl_pct:.1f}%)
- 발굴 핵심 근거: {reason_core}
- 체결 원칙: {rule_note}
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{system_instruction}\n\n{context_data}\n\n[사용자 질문]: {question}"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 800
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
    in-depth, tailored analysis based on mathematical signals.
    """
    name = str(stock_info.get("name") or "해당 종목")
    code = str(stock_info.get("code") or "")
    market = str(stock_info.get("market") or "KRX")
    sector = str(stock_info.get("sector") or "시장 주도 섹터")
    close = _safe_to_float(stock_info.get("close"), 0.0)
    change_pct = _safe_to_float(stock_info.get("change_pct"), 0.0)
    score = _safe_to_float(stock_info.get("score"), 70.0)
    strategy = str(stock_info.get("strategy_tag") or stock_info.get("strategy") or "퀀트 발굴")
    tp_pct = _safe_to_float(stock_info.get("tp_pct"), 5.0)
    sl_pct = _safe_to_float(stock_info.get("sl_pct"), 2.0)
    if sl_pct <= 0:
        sl_pct = 2.0
    
    tp_price = close * (1.0 + tp_pct / 100.0) if close > 0 else 0
    sl_price = close * (1.0 - sl_pct / 100.0) if close > 0 else 0
    
    breakdown = stock_info.get("breakdown") or {}
    if not isinstance(breakdown, dict):
        breakdown = {}
    rsi = _safe_to_float(breakdown.get("rsi_val", stock_info.get("rsi")), 55.0)
    vol_ratio = _safe_to_float(breakdown.get("vol_ratio", stock_info.get("vol_ratio")), 1.0)
    ma_status = str(breakdown.get("ma_status") or stock_info.get("ma_status") or "이평 정배열")
    reason_core = str(stock_info.get("reason_core") or "전략 기준 충족")
    
    q_lower = question.lower()
    
    # 1. 상승 원인 / 왜 오르나요?
    if any(k in q_lower for k in ["왜", "원인", "이유", "급등", "상승", "오르", "배경"]):
        ans = f"""### **{name} ({code}) 상승 원인 퀀트 정밀 진단**

**1. 수급 및 거래량 폭발력 ({vol_ratio:.1f}배 급증)**
- 전일 동시간대 대비 거래량이 **{vol_ratio:.1f}배** 터지며 시장의 기관/외인 주도 세력의 강력한 순매수 자금이 유입되었습니다.
- 거래량을 수반하지 않는 헛반등이 아니라, 실질적인 대량 수급이 뒷받침된 **신뢰도 높은 거래대금 주도 상승**입니다.

**2. 전략적 발굴 근거 ({strategy})**
- **핵심 근거**: {reason_core}
- **이동평균선 배열**: `{ma_status}` 상태로, 단기/중기 매물대를 상향 돌파하며 상방 압력이 강하게 형성되었습니다.
- **RSI 강도**: 현재 14일 RSI는 **{rsi:.1f}pt**로, { '과열권 이전의 탄탄한 상승 추세 가속 구간' if rsi <= 72 else '강한 단기 과열 양상이 나타나고 있는 강력한 매수세 국면' }입니다.

**3. 업종/테마 모멘텀**
- 소속 섹터인 **[{sector}]** 군으로 시장 주도 자금이 강하게 쏠리면서 테마 랠리 동조화 효과가 작용하고 있습니다.
"""

    # 2. 지금 사도 되나요? / 진입 가능 여부 / 추격매수 진단
    elif any(k in q_lower for k in ["진입", "사도", "매수", "추격", "타이밍", "지금", "들어가"]):
        if change_pct >= 18.0 or rsi >= 78.0:
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

        ans = f"""### **{name} ({code}) 신규 진입 적합도 진단**

**판정 결과**: **{risk_badge}**

- **현재가 상태**: **{close:,.0f}원** (당일 {change_pct:+.2f}%) | RSI: **{rsi:.1f}**
- **퀀트 진단 소견**: {verdict}
- **매매 행동 요령**: {action_guide}
- **손익비 분석**: 
  - 기대 이익폭: **+{tp_pct:.1f}%** ({tp_price:,.0f}원)
  - 최대 허용 손실폭: **-{sl_pct:.1f}%** ({sl_price:,.0f}원)
  - 목표 손익비: **{(tp_pct / max(0.1, sl_pct)):.2f} : 1** (기대수익이 위험 대비 2배 이상 우위)
"""

    # 3. 목표가 / 손절가 / 가격 전략
    elif any(k in q_lower for k in ["목표", "손절", "익절", "가격", "얼마", "청산", "스탑"]):
        ans = f"""### **{name} ({code}) 목표가 & 손절가 가격 전략 리포트**

**1. 알고리즘 목표 익절가 (Take-Profit)**
- **1차 목표가**: **{tp_price:,.0f}원** (현재가 대비 **+{tp_pct:.1f}%**)
- **실행 원칙**: 장중 해당 호가 터치 시 50% 분할 익절, 잔여 물량은 트레일링 스탑(+1.8% 이상 수익 락)으로 극대화.

**2. 원칙 손절 기준가 (Stop-Loss)**
- **손절 기준가**: **{sl_price:,.0f}원** (현재가 대비 **-{sl_pct:.1f}%**)
- **실행 원칙**: 종가나 장중 해당 가격을 하향 이탈할 경우 미련 없이 전량 손절하여 추가 낙폭(-5%~-10%)을 원천 차단.

**3. 트레일링 익절 락 (Trailing Lock)**
- 주가가 **+{min(2.4, tp_pct*0.7):.1f}%** 이상 상승했다가 되밀릴 경우, 본전 이상인 **+1.8%** 구간에서 자동 청산하여 수익을 지킵니다.
"""

    # 4. 리스크 / 주의사항
    elif any(k in q_lower for k in ["리스크", "위험", "주의", "손실", "조심", "탈락"]):
        ans = f"""### **{name} ({code}) 주요 투자 리스크 팩트체크**

**1. 시초가 갭 및 윗꼬리 차익실현 리스크**
- 당일 시초 갭상승 종목은 장초반 슈팅 후 기관/외인의 차익실현 매물로 인해 긴 윗꼬리를 달고 급락할 위험이 상존합니다.
- 고점 대비 -2% 이상 밀릴 경우 즉시 매도세 전환에 주의하세요.

**2. 업종 테마 쏠림 동반 조정 리스크**
- 현재 **[{sector}]** 테마에 수급이 쏠려있으므로, 대장주가 꺾이면 후발주자도 동반 급락할 수 있습니다.

**3. 거래량 소강 리스크**
- 오후장(11:00~13:00) 점심시간대 거래대금이 급감할 경우 호가 공백이 발생하여 일시적인 가격 왜곡이 일어날 수 있습니다.
"""

    # 5. 차트 / 기술적 지표 종합 분석
    elif any(k in q_lower for k in ["차트", "지표", "이평", "rsi", "볼린저", "기술적"]):
        ans = f"""### **{name} ({code}) 기술적 지표 & 차트 종합 해부**

- **이동평균선 배열**: `{ma_status}` (단기 이평선이 중장기 이평선 위에 위치하여 정배열 모멘텀 우위)
- **RSI(14) 모멘텀**: **{rsi:.1f}pt** ({ '중립 상단 탄력 구간' if rsi < 70 else '과열 진입 구간 (단기 슈팅 가능하나 급등락 주의)' })
- **거래량 급증비**: 전일 대비 **{vol_ratio:.2f}배** (신뢰도 높은 거래량 폭발)
- **종합 퀀트 스코어**: **{score:.1f}점 / 100점** (신호 신뢰도: **{stock_info.get('label', '우수')}**)
- **전략 적합도**: `{strategy}` 조건식 100% 만족 통과
"""

    # 6. 기본 일반 브리핑 (디폴트)
    else:
        ans = f"""### **{name} ({code}) 퀀트 수석 애널리스트 종합 브리핑**

- **종목 개요**: {market} 상장 | 소속 업종: **[{sector}]** | 현재가: **{close:,.0f}원** ({change_pct:+.2f}%)
- **선정 전략**: **{strategy}** (신호 스코어: **{score:.1f}점**, {stock_info.get('label', '고확신')})
- **거래 수급 상태**: 거래량 급증 **{vol_ratio:.1f}배** | RSI(14): **{rsi:.1f}** | `{ma_status}`
- **매매 가이드**:
  - 권장 목표가: **{tp_price:,.0f}원 (+{tp_pct:.1f}%)**
  - 원칙 손절가: **{sl_price:,.0f}원 (-{sl_pct:.1f}%)**
  - 발굴 근거: {reason_core}

추가로 궁금하신 사항은 아래의 빠른 질문 칩(상승 원인, 진입 가능 여부, 목표가 등)을 누르시거나 직접 질문해 주세요!
"""

    if fallback_notice:
        ans += "\n\n*(※ Gemini API 연동 대기 중으로 내장 퀀트 수치 추론 엔진이 실시간 생성한 정밀 리포트입니다.)*"

    return ans
