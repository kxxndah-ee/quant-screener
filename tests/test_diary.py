import unittest
import os
import sqlite3
from src.database.models import init_db, get_db_connection
from src.database.diary_manager import (
    record_daily_prediction,
    get_diary_history,
    evaluate_model_decay
)


class TestDiary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def test_record_and_query_prediction(self):
        pred_id = record_daily_prediction(
            date_str="2024-01-15",
            code="005930",
            name="삼성전자",
            score=77.5,
            label="강세",
            target_tp=2.0,
            target_sl=2.0,
            rsi=58.2,
            ma_align=1,
            vol_surge=1.8,
            bb_pct=0.65
        )
        self.assertIsNotNone(pred_id)
        df = get_diary_history(10)
        self.assertFalse(df.empty)
        matched = df[df["id"] == pred_id]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched.iloc[0]["code"], "005930")

    def test_model_decay_evaluation(self):
        decay = evaluate_model_decay(expected_ci_lower=50.0)
        self.assertIn("decay_detected", decay)
        self.assertIn("live_win_rate", decay)


if __name__ == "__main__":
    unittest.main()
