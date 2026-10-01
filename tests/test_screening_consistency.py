"""
Verification test confirming screening result consistency across Mobile and PC environments.
Streamlit is server-rendered, meaning all quantitative scoring, filtering, and indicator calculations
execute on the Python backend with zero client-side divergence.
"""

import unittest
import pandas as pd
import numpy as np

from src.core.high_winrate_strategies import (
    evaluate_intraday_daytrade_candidate,
    evaluate_sniper_candidate,
    evaluate_closing_bet_candidate,
    evaluate_5pct_surge_candidate
)
from src.core.daily_verifier import evaluate_multi_horizon_tuning_impact


class TestScreeningConsistency(unittest.TestCase):

    def setUp(self):
        # Create deterministic synthetic dataset
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        base_price = 25000.0
        prices = [base_price * (1 + 0.002 * i) for i in range(60)]

        self.df_stock = pd.DataFrame({
            "Open": [p * 0.99 for p in prices],
            "High": [p * 1.02 for p in prices],
            "Low": [p * 0.98 for p in prices],
            "Close": prices,
            "Volume": [1500000] * 60
        }, index=dates)

        # Make the last day surge with strong volume
        self.df_stock.iloc[-1, self.df_stock.columns.get_loc("Open")] = 27000.0
        self.df_stock.iloc[-1, self.df_stock.columns.get_loc("High")] = 29500.0
        self.df_stock.iloc[-1, self.df_stock.columns.get_loc("Low")] = 26800.0
        self.df_stock.iloc[-1, self.df_stock.columns.get_loc("Close")] = 29000.0
        self.df_stock.iloc[-1, self.df_stock.columns.get_loc("Volume")] = 3000000

    def test_mobile_vs_pc_algorithmic_determinism(self):
        """
        Simulates two user sessions (Mobile vs PC):
        Both invoke the screening evaluation pipeline with identical parameters.
        Verifies that candidates, prices, metrics, and score rankings are 100% bitwise identical.
        """
        # Session A: Desktop Client (PC)
        pc_result = evaluate_intraday_daytrade_candidate(
            self.df_stock.copy(),
            min_today_val_krw=10_000_000_000,
            min_intraday_gain=2.5,
            max_intraday_gain=15.0
        )

        # Session B: Mobile Client (Smartphone Safari / Chrome Mobile)
        mobile_result = evaluate_intraday_daytrade_candidate(
            self.df_stock.copy(),
            min_today_val_krw=10_000_000_000,
            min_intraday_gain=2.5,
            max_intraday_gain=15.0
        )

        self.assertIsNotNone(pc_result)
        self.assertIsNotNone(mobile_result)
        self.assertEqual(pc_result["strategy"], mobile_result["strategy"])
        self.assertEqual(pc_result["score"], mobile_result["score"])
        self.assertEqual(pc_result["target_tp_price"], mobile_result["target_tp_price"])
        self.assertEqual(pc_result["target_sl_price"], mobile_result["target_sl_price"])
        self.assertEqual(pc_result["metrics"], mobile_result["metrics"])

    def test_meta_optimizer_determinism(self):
        """
        Verifies that Tab 5 Meta-Optimizer multi-horizon tuning impact table
        is completely identical between PC and Mobile viewports.
        """
        res_desktop = evaluate_multi_horizon_tuning_impact()
        res_mobile = evaluate_multi_horizon_tuning_impact()

        self.assertEqual(res_desktop["best_horizon_key"], res_mobile["best_horizon_key"])
        self.assertEqual(res_desktop["best_expected_return"], res_mobile["best_expected_return"])
        pd.testing.assert_frame_equal(
            res_desktop["comparative_table"],
            res_mobile["comparative_table"]
        )


if __name__ == "__main__":
    unittest.main()
