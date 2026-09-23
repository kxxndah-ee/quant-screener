"""
Unit tests for Daily Point-in-Time Verifier, Failure Diagnosis, and Auto-Tuning Engine.
"""

import unittest
import pandas as pd
import numpy as np

from src.core.daily_verifier import (
    get_available_trading_dates,
    diagnose_failure_reasons,
    generate_auto_tuning_recommendations
)


class TestDailyVerifier(unittest.TestCase):

    def setUp(self):
        # Sample synthetic verification results
        self.sample_results = pd.DataFrame([
            {
                "code": "005930",
                "name": "삼성전자",
                "t1_score": 75.0,
                "t1_val_krw": 25_000_000_000,
                "rsi": 58.0,
                "open_gap_pct": 1.2,
                "max_gain_pct": 5.4,
                "net_return_pct": 4.8,
                "target_tp": 5.0,
                "target_sl": 2.5,
                "is_hit": True,
                "exit_code": "TP",
                "excess_return": 4.2
            },
            {
                "code": "000660",
                "name": "SK하이닉스",
                "t1_score": 72.0,
                "t1_val_krw": 20_000_000_000,
                "rsi": 62.0,
                "open_gap_pct": 1.8,
                "max_gain_pct": 3.2,
                "net_return_pct": 2.1,
                "target_tp": 5.0,
                "target_sl": 2.5,
                "is_hit": True,
                "exit_code": "CLOSE",
                "excess_return": 1.5
            },
            {
                "code": "035420",
                "name": "NAVER",
                "t1_score": 68.0,
                "t1_val_krw": 8_000_000_000,  # Low liquidity
                "rsi": 74.0,                 # Overbought
                "open_gap_pct": 3.8,         # High gap
                "max_gain_pct": 0.8,
                "net_return_pct": -2.5,
                "target_tp": 5.0,
                "target_sl": 2.5,
                "is_hit": False,
                "exit_code": "SL",
                "excess_return": -3.1
            },
            {
                "code": "035720",
                "name": "카카오",
                "t1_score": 66.0,
                "t1_val_krw": 9_000_000_000,  # Low liquidity
                "rsi": 71.0,                 # Overbought
                "open_gap_pct": 4.2,         # High gap
                "max_gain_pct": 1.0,
                "net_return_pct": -2.5,
                "target_tp": 5.0,
                "target_sl": 2.5,
                "is_hit": False,
                "exit_code": "SL",
                "excess_return": -3.1
            }
        ])

    def test_get_available_trading_dates(self):
        dates = get_available_trading_dates(limit=10)
        self.assertIsInstance(dates, list)
        self.assertGreater(len(dates), 0)
        for d in dates:
            self.assertRegex(d, r"^\d{4}-\d{2}-\d{2}$")

    def test_diagnose_failure_clean_all_hits(self):
        all_hits = self.sample_results[self.sample_results["is_hit"]].copy()
        diag = diagnose_failure_reasons(all_hits, bm_change_pct=0.5)
        self.assertEqual(len(diag["issues"]), 0)
        self.assertIn("🎉", diag["summary"])

    def test_diagnose_failure_reasons_detection(self):
        diag = diagnose_failure_reasons(self.sample_results, bm_change_pct=-1.5)
        issues = diag["issues"]
        issue_types = [issue["type"] for issue in issues]

        # Should detect High Open Gap
        self.assertIn("HIGH_OPEN_GAP", issue_types)
        # Should detect Low Liquidity
        self.assertIn("LOW_LIQUIDITY", issue_types)
        # Should detect Overbought RSI
        self.assertIn("OVERBOUGHT_RSI", issue_types)
        # Should detect Market Crash
        self.assertIn("MARKET_CRASH", issue_types)

        # Check stats comparison structure
        self.assertIn("stats_comparison", diag)
        self.assertIn("hits", diag["stats_comparison"])
        self.assertIn("misses", diag["stats_comparison"])

    def test_auto_tuning_recommendations_and_simulation(self):
        current_params = {
            "min_val_krw": 10_000_000_000,
            "score_cutoff": 65.0
        }
        tuning = generate_auto_tuning_recommendations(self.sample_results, current_params)
        proposals = tuning["proposals"]
        sim = tuning["simulation"]

        self.assertGreater(len(proposals), 0)
        param_keys = [p["param_key"] for p in proposals]
        self.assertIn("max_open_gain", param_keys)
        self.assertIn("min_trading_val", param_keys)

        # Simulation metrics
        self.assertEqual(sim["before_total"], 4)
        self.assertEqual(sim["before_win_rate"], 50.0)
        # After filtering out low liquidity and high gap, only Samsung & Hynix remain
        self.assertEqual(sim["after_total"], 2)
        self.assertEqual(sim["after_win_rate"], 100.0)
        self.assertEqual(sim["filtered_losses"], 2)
        self.assertEqual(sim["win_rate_boost"], 50.0)


if __name__ == "__main__":
    unittest.main()
