import unittest
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.models import get_db_connection
from src.collectors.market_data import get_latest_market_info, fetch_ohlcv
from src.core.scoring import evaluate_stock_latest
from src.core.quant_analyst import generate_quant_expert_answer, extract_stock_and_question_from_query


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

        self.assertIn('"1. 스크리너"', content)
        self.assertIn('"2. Q&A"', content)
        self.assertIn('"3. 관심종목"', content)
        self.assertIn('"4. 검증 및 최적화"', content)
        self.assertIn("tab1, tab2, tab3, tab4 = st.tabs(main_tab_names", content)
        self.assertIn("render_analyst_factsheet_and_qa_tab()", content)
        self.assertIn("with tab1:", content)
        self.assertIn("with tab2:", content)
        self.assertIn("with tab3:", content)
        self.assertIn("with tab4:", content)
        self.assertIn('st.session_state["sc_scope_sel"] = "전 상장사 전수조사 (약 2,700개)"', content)
        self.assertIn('[스크리너 필터 설정] 전 상장사 전수조사', content)

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

        # 4. Question-First Layout: 통합 질문 콘솔(qa_custom_form_, btn_qa_buy_, btn_qa_why_)이 종목 분석 카드(종합 퀀트 스코어)보다 먼저 배치
        tab2_fn_text = content[content.index("def render_analyst_factsheet_and_qa_tab():"):content.index("main_tab_names =")]
        self.assertIn("btn_qa_buy_", tab2_fn_text)
        self.assertIn("btn_qa_why_", tab2_fn_text)
        self.assertLess(tab2_fn_text.index("btn_qa_buy_"), tab2_fn_text.index("btn_qa_why_"))
        self.assertIn("qa_custom_form_", tab2_fn_text)
        self.assertLess(tab2_fn_text.index("qa_custom_form_"), tab2_fn_text.index("종합 퀀트 스코어"))
        self.assertLess(tab2_fn_text.index("btn_qa_buy_"), tab2_fn_text.index("종합 퀀트 스코어"))

    def test_question_first_stock_and_question_extraction(self):
        # 1. Stock name + question
        res1 = extract_stock_and_question_from_query("삼성전자 지금 사도 되나요?", current_code="000660")
        self.assertEqual(res1["code"], "005930")
        self.assertEqual(res1["matched_by"], "name")
        self.assertFalse(res1["is_stock_only"])
        self.assertIn("지금 사도 되나요", res1["question"])

        # 2. 6-digit stock code + question
        res2 = extract_stock_and_question_from_query("000660 내일 시초가에 어떻게 대응해야 하나요?", current_code="005930")
        self.assertEqual(res2["code"], "000660")
        self.assertEqual(res2["matched_by"], "code")
        self.assertFalse(res2["is_stock_only"])

        # 3. General market recommendation query
        res3 = extract_stock_and_question_from_query("지금 매수할 만한 종목은?", current_code="005930", fallback_top_code="035420")
        self.assertEqual(res3["code"], "035420")
        self.assertEqual(res3["matched_by"], "recommend")
        self.assertFalse(res3["is_stock_only"])

        # 4. Follow-up question without stock name keeps current_code
        res4 = extract_stock_and_question_from_query("목표가와 손절가는 얼마인가요?", current_code="005380")
        self.assertEqual(res4["code"], "005380")
        self.assertEqual(res4["matched_by"], "context")
        self.assertFalse(res4["is_stock_only"])

        # 5. Stock name only triggers default entry suitability question
        res5 = extract_stock_and_question_from_query("삼성전자", current_code="000660")
        self.assertEqual(res5["code"], "005930")
        self.assertTrue(res5["is_stock_only"])

    def test_custom_questions_produce_distinct_quant_results(self):
        sample = {
            "name": "삼성전자",
            "code": "005930",
            "market": "KOSPI",
            "sector": "전기/전자",
            "close": 75000,
            "open": 73500,
            "high": 76200,
            "low": 73100,
            "prev_close": 72500,
            "volume": 15000000,
            "change_pct": 3.45,
            "score": 80.0,
            "label": "우수",
            "strategy_tag": "고확신 퀀트 모멘텀",
            "tp_pct": 5.0,
            "sl_pct": 2.5,
            "vol_ratio": 2.5,
            "rsi": 62.0,
            "ma_status": "이평 정배열",
            "bb_pct": 0.72,
            "reason_core": "거래량 2.5배 폭발 및 정배열 모멘텀",
            "rule_note": "목표가 도달 시 50% 분할 익절",
            "breakdown": {
                "rsi_val": 62.0,
                "vol_ratio": 2.5,
                "ma_status": "이평 정배열",
                "bb_pct": 0.72
            }
        }
        custom_questions = [
            "내일 시초가에 어떻게 대응해야 하나요?",
            "보유 중인데 언제 매도할까요?",
            "지지선과 저항선이 어디인가요?",
            "수급이랑 거래량 상태는 어때?",
            "비중은 얼마나 들어가면 될까요?",
            "단타가 좋나요 스윙이 좋나요?",
            "앞으로 전반적인 주가 전망이 어때요?"
        ]
        answers = []
        for q in custom_questions:
            ans = generate_quant_expert_answer(sample, q)
            self.assertIn(q, ans)
            self.assertIn("퀀트 산출 결과값", ans)
            answers.append(ans)
        self.assertEqual(len(set(answers)), len(custom_questions))


if __name__ == "__main__":
    unittest.main()
