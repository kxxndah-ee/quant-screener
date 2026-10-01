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


if __name__ == "__main__":
    unittest.main()
