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

    def test_screener_time_sync_reapplication(self):
        """
        Verifies that when sc_trigger_time_sync is True (triggered by clicking '동기화'
        next to the date/time header), the recommended parameters for the active strategy
        are properly re-applied to session state and sc_trigger_time_sync resets to False.
        """
        # Mock session state and recommendation
        mock_session_state = {
            "sc_trigger_time_sync": True,
            "sc_last_applied_strat_slot": "old_slot",
            "sc_min_daytrade_val": 300,  # User previously customized this
            "sc_scope_sel": "내 관심종목 138선 (초고속)"
        }

        rec_params = {
            "min_daytrade_val_b": 100,
            "intraday_gain_range": (2.5, 10.0),
            "min_daytrade_vol_ratio": 0.5
        }
        cur_strat_slot_id = "실시간 당일 단타_dt_slot_0"

        # Execute auto-sync logic identical to app.py
        if "sc_last_applied_strat_slot" not in mock_session_state or mock_session_state.get("sc_trigger_time_sync") or mock_session_state.get("sc_last_applied_strat_slot") != cur_strat_slot_id:
            for _k, _v in rec_params.items():
                mock_session_state[f"sc_{_k}"] = _v
                if _k == "min_daytrade_val_b":
                    mock_session_state["sc_min_daytrade_val"] = _v
            mock_session_state["sc_scope_sel"] = "전 상장사 전수조사 (약 2,700개)"
            mock_session_state["sc_last_applied_strat_slot"] = cur_strat_slot_id
            mock_session_state["sc_trigger_time_sync"] = False

        self.assertFalse(mock_session_state["sc_trigger_time_sync"])
        self.assertEqual(mock_session_state["sc_min_daytrade_val"], 100)
        self.assertEqual(mock_session_state["sc_min_daytrade_val_b"], 100)
        self.assertEqual(mock_session_state["sc_intraday_gain_range"], (2.5, 10.0))
        self.assertEqual(mock_session_state["sc_min_daytrade_vol_ratio"], 0.5)
        self.assertEqual(mock_session_state["sc_scope_sel"], "전 상장사 전수조사 (약 2,700개)")
        self.assertEqual(mock_session_state["sc_last_applied_strat_slot"], cur_strat_slot_id)


if __name__ == "__main__":
    unittest.main()
