import unittest
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.models import get_db_connection
from src.collectors.market_data import get_latest_market_info, fetch_ohlcv
from src.core.scoring import evaluate_stock_latest
from src.core.quant_analyst import generate_quant_expert_answer


class TestAnalystTab(unittest.TestCase):
    def test_factsheet_payload_logic(self):
        code = "005930"
        info = get_latest_market_info(code)
        self.assertIsNotNone(info)
        self.assertIn("close", info)
        self.assertIn("change_pct", info)

        df = fetch_ohlcv(code)
        self.assertFalse(df.empty)

        score_info = evaluate_stock_latest(df)
        self.assertIsNotNone(score_info)
        self.assertIn("score", score_info)
        self.assertIn("label", score_info)
        self.assertIn("breakdown", score_info)

        close = float(info.get("close", 0.0))
        tp_pct = float(score_info.get("tp_pct", 5.0))
        sl_pct = float(score_info.get("sl_pct", 2.0))
        tp_price = close * (1.0 + tp_pct / 100.0) if close > 0 else 0.0
        sl_price = close * (1.0 - sl_pct / 100.0) if close > 0 else 0.0
        rr_ratio = tp_pct / max(0.1, sl_pct)

        self.assertGreater(tp_price, close)
        self.assertLess(sl_price, close)
        self.assertGreater(rr_ratio, 0.0)

    def test_analyst_qa_answers_for_stock(self):
        sample = {
            "name": "삼성전자",
            "code": "005930",
            "market": "KOSPI",
            "sector": "전기/전자",
            "close": 75000,
            "change_pct": 3.5,
            "score": 80.0,
            "label": "우수",
            "strategy_tag": "고확신 퀀트 모멘텀",
            "tp_pct": 5.0,
            "sl_pct": 2.5,
            "vol_ratio": 2.5,
            "rsi": 62.0,
            "ma_status": "이평 정배열",
            "reason_core": "거래량 2.5배 폭발 및 정배열 모멘텀",
            "rule_note": "목표가 도달 시 50% 분할 익절",
            "breakdown": {
                "rsi_val": 62.0,
                "vol_ratio": 2.5,
                "ma_status": "이평 정배열"
            }
        }
        ans = generate_quant_expert_answer(sample, "왜 급등하나요?")
        self.assertIn("삼성전자", ans)
        self.assertIn("상승 원인", ans)

        ans_buy = generate_quant_expert_answer(sample, "지금 사도 되나요?")
        self.assertIn("진입 적합도", ans_buy)

    def test_app_py_tab_configuration(self):
        # Inspect app.py text to verify main_tab_names definition
        app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn('"1. 시장 전체 익일 상승 후보 & 고확신 스크리너"', content)
        self.assertIn('"2. Q&A"', content)
        self.assertIn('"3. 관심종목 실시간 스코어보드"', content)
        self.assertIn('"4. 통합 성과 검증 & AI 자가최적화 센터"', content)
        self.assertIn("tab1, tab2, tab3, tab4 = st.tabs(main_tab_names", content)
        self.assertIn("render_analyst_factsheet_and_qa_tab()", content)
        self.assertIn("with tab1:", content)
        self.assertIn("with tab2:", content)
        self.assertIn("with tab3:", content)
        self.assertIn("with tab4:", content)

        # Tab 2 structure assertions:
        # 1. Direct search only (no quick chips, no dropdown)
        self.assertIn("qa_direct_search_kw", content)
        self.assertNotIn("빠른 종목 선택 칩", content)
        self.assertNotIn("qa_stock_search_selectbox", content)

        # 2. Compact metrics present
        self.assertIn("종합 퀀트 스코어", content)
        self.assertIn("가격 전략 & 손익비", content)
        self.assertIn("수급 및 거래량 폭발력", content)
        self.assertIn("기술적 지표 & 이평선", content)

        # 3. Console 1: 목표가&손절가, 리스크 팩트체크, 차트&지표진단 (3열 요약 카드 + 접이식 전문 보기, 선택 버튼 삭제)
        self.assertNotIn("btn_diag_tp_", content)
        self.assertNotIn("btn_diag_risk_", content)
        self.assertNotIn("btn_diag_chart_", content)
        self.assertIn("목표가 & 손절가", content)
        self.assertIn("리스크 팩트체크", content)
        self.assertIn("차트 & 지표 진단", content)
        self.assertIn("상세 진단 리포트 전문 보기", content)

        # 4. Console 2: 왜 급등하나요?, 지금 사도 되나요? + 질문칸
        self.assertIn("btn_qa_why_", content)
        self.assertIn("btn_qa_buy_", content)
        self.assertIn("qa_custom_form_", content)


if __name__ == "__main__":
    unittest.main()
