import unittest
import pandas as pd
from src.backtest.statistical import run_bootstrap_analysis


class TestStatistical(unittest.TestCase):
    def test_verified_badge_conditions(self):
        # Highly profitable sample where excess return CI > 0 and win rate CI > 50%
        # 18 wins out of 20 trades with high excess return
        trades = pd.DataFrame({
            "net_return_pct": [2.5, 2.0, 1.8, 2.2, 3.0, 1.5, 2.1, 2.4, -1.5, 2.0,
                               1.9, 2.3, 2.8, -1.8, 1.7, 2.5, 2.2, 1.6, 2.0, 2.1],
            "excess_return_pct": [1.8, 1.5, 1.2, 1.7, 2.4, 1.0, 1.6, 1.9, -0.8, 1.4,
                                  1.3, 1.8, 2.2, -1.0, 1.1, 2.0, 1.7, 1.2, 1.5, 1.6],
        })
        res = run_bootstrap_analysis(trades, n_iterations=1000)
        self.assertTrue(res["is_verified"])
        self.assertEqual(res["status_badge"], "검증됨")
        self.assertGreater(res["win_rate_ci"][0], 50.0)
        self.assertGreater(res["excess_return_ci"][0], 0.0)

    def test_experimental_badge_conditions(self):
        # Mixed/mediocre strategy: win rate around 45%
        trades = pd.DataFrame({
            "net_return_pct": [-2.0, -1.9, 1.5, -2.1, 2.0, -1.8, 1.4, -2.0, 1.2, -1.5],
            "excess_return_pct": [-1.5, -1.4, 0.5, -1.6, 1.0, -1.2, 0.4, -1.5, 0.2, -1.0],
        })
        res = run_bootstrap_analysis(trades, n_iterations=1000)
        self.assertFalse(res["is_verified"])
        self.assertEqual(res["status_badge"], "실험적")


if __name__ == "__main__":
    unittest.main()
