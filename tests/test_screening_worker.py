import unittest
import time
import pandas as pd
from unittest.mock import patch, MagicMock

from src.core.screening_worker import ScreeningWorker, get_screening_worker


class TestScreeningWorker(unittest.TestCase):

    def setUp(self):
        self.worker = ScreeningWorker(session_id="test_session")

    def test_registry(self):
        w1 = get_screening_worker("session_a")
        w2 = get_screening_worker("session_a")
        w3 = get_screening_worker("session_b")
        self.assertIs(w1, w2)
        self.assertIsNot(w1, w3)

    @patch("src.core.screening_worker.get_universe")
    def test_empty_universe(self, mock_get_univ):
        mock_get_univ.return_value = pd.DataFrame()
        started = self.worker.start_screening({"target_market": "전체", "sc_scope": "관심종목 138선"})
        self.assertTrue(started)

        # Wait for thread to finish
        for _ in range(20):
            if not self.worker.is_running():
                break
            time.sleep(0.05)

        status = self.worker.get_status()
        self.assertEqual(status["status"], "COMPLETED")
        self.assertTrue(status["results_df"].empty)

    @patch("src.core.screening_worker.get_universe")
    def test_cancel_flow(self, mock_get_univ):
        # Create a mock universe with 100 items
        mock_get_univ.return_value = pd.DataFrame({
            "code": [f"{i:06d}" for i in range(100)],
            "name": [f"Stock{i}" for i in range(100)],
            "market": ["KOSPI"] * 100,
            "sector": ["IT"] * 100
        })

        with patch("src.core.screening_worker.fetch_ohlcv") as mock_ohlcv:
            # Slow down fetch to simulate long task
            def slow_fetch(code):
                time.sleep(0.05)
                return pd.DataFrame()

            mock_ohlcv.side_effect = slow_fetch

            started = self.worker.start_screening({
                "sc_mode": "일반 퀀트",
                "sc_scope": "전수조사",
                "target_market": "전체"
            })
            self.assertTrue(started)
            time.sleep(0.1)
            self.assertTrue(self.worker.is_running())

            # Request cancellation
            self.worker.cancel()

            # Wait for thread termination
            for _ in range(30):
                if not self.worker.is_running():
                    break
                time.sleep(0.05)

            status = self.worker.get_status()
            self.assertEqual(status["status"], "CANCELLED")

    @patch("src.core.screening_worker.get_universe")
    @patch("src.core.screening_worker.get_benchmark_ohlcv")
    def test_pre_market_opening_strategy_worker(self, mock_bm, mock_univ):
        import numpy as np
        # 1. Setup mock universe
        mock_univ.return_value = pd.DataFrame({
            "code": ["005930"],
            "name": ["대장주"],
            "market": ["KOSPI"],
            "sector": ["반도체"]
        })

        # 2. Setup mock benchmark
        dates = pd.date_range("2024-01-01", periods=50, freq="B")
        bm_prices = np.linspace(300, 360, 50)
        mock_bm.return_value = pd.DataFrame({
            "Open": bm_prices * 0.99,
            "High": bm_prices * 1.01,
            "Low": bm_prices * 0.98,
            "Close": bm_prices,
            "Volume": 1000000
        }, index=dates)

        # 3. Setup mock stock data (market leader surge with 1,700억 trading value)
        prices = np.linspace(50000, 55000, 50)
        df_lead = pd.DataFrame({
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": [100000] * 49 + [3000000]  # 300만주 * 5.8만 = ~1,740억 원
        }, index=dates)
        df_lead.iloc[-1, df_lead.columns.get_loc("Open")] = 55000
        df_lead.iloc[-1, df_lead.columns.get_loc("High")] = 58200
        df_lead.iloc[-1, df_lead.columns.get_loc("Low")] = 54800
        df_lead.iloc[-1, df_lead.columns.get_loc("Close")] = 58000

        with patch("src.core.screening_worker.fetch_ohlcv", return_value=df_lead):
            started = self.worker.start_screening({
                "sc_mode": "장전 시초가 공략 (눌림 반등 + 주도 갭상승)",
                "sc_scope": "전수조사",
                "target_market": "전체",
                "min_val_krw_b": 100,
                "max_disparity_val": 103.5,
                "ignore_market_filter": False
            })
            self.assertTrue(started)

            for _ in range(40):
                if not self.worker.is_running():
                    break
                time.sleep(0.05)

            status = self.worker.get_status()
            self.assertEqual(status["status"], "COMPLETED")
            df_res = status["results_df"]
            self.assertFalse(df_res.empty)
            self.assertEqual(df_res.iloc[0]["code"], "005930")
            self.assertEqual(df_res.iloc[0]["strategy_mode"], "PRE_MARKET_OPEN")
            self.assertIn("장전시초가", df_res.iloc[0]["strategy_tag"])
            self.assertIn("장전", df_res.iloc[0]["timing_label"])
            self.assertIn("open_pct", df_res.columns)
            self.assertIn("high_pct", df_res.columns)
            self.assertIn("low_pct", df_res.columns)
            self.assertIn("change_pct", df_res.columns)
            self.assertAlmostEqual(df_res.iloc[0]["open_pct"], ((55000 - prices[-2]) / prices[-2]) * 100.0, places=2)
            self.assertAlmostEqual(df_res.iloc[0]["high_pct"], ((58200 - prices[-2]) / prices[-2]) * 100.0, places=2)
            self.assertAlmostEqual(df_res.iloc[0]["low_pct"], ((54800 - prices[-2]) / prices[-2]) * 100.0, places=2)


if __name__ == "__main__":
    unittest.main()
