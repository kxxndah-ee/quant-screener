"""
Unit tests for Daily Point-in-Time Verifier, Failure Diagnosis, and Auto-Tuning Engine.
"""

import unittest
import pandas as pd
import numpy as np

from src.core.daily_verifier import (
    get_available_trading_dates,
    get_valid_prediction_dates,
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

    def test_get_valid_prediction_dates(self):
        pairs = get_valid_prediction_dates(limit=5)
        self.assertIsInstance(pairs, list)
        self.assertGreater(len(pairs), 0)
        for p in pairs:
            self.assertEqual(len(p), 2)
            self.assertNotEqual(p[0], p[1])

    def test_save_and_get_cached_verification_history(self):
        from src.core.daily_verifier import save_verification_history, get_cached_verification_history
        sample_payload = {
            "pred_date": "2026-09-01",
            "exec_date": "2026-09-02",
            "strategy_mode": "테스트 전략",
            "total_screened": 2,
            "df_results": self.sample_results.head(2),
            "kpi": {
                "total": 2,
                "hits": 2,
                "win_rate": 100.0,
                "avg_net_ret": 3.45,
                "avg_max_gain": 4.3,
                "avg_excess_ret": 2.8,
                "tp_count": 1,
                "sl_count": 0,
                "bm_day_ret": 0.5
            },
            "diagnosis": {"issues": [], "summary": "완벽"},
            "tuning": {"proposals": []}
        }
        try:
            save_verification_history(sample_payload, "테스트 전략")
            cached = get_cached_verification_history("2026-09-01", "테스트 전략")
            self.assertIsNotNone(cached)
            self.assertEqual(cached["total_screened"], 2)
            self.assertEqual(cached["kpi"]["win_rate"], 100.0)
            self.assertFalse(cached["df_results"].empty)
        finally:
            from src.database.models import get_db_connection
            with get_db_connection() as conn:
                conn.cursor().execute("DELETE FROM daily_verification_history WHERE strategy_mode = ?", ("테스트 전략",))
                conn.commit()

    def test_get_monthly_verification_summary(self):
        from src.core.daily_verifier import get_monthly_verification_summary
        df_summary = get_monthly_verification_summary(limit_days=30)
        self.assertIsInstance(df_summary, pd.DataFrame)
        self.assertGreater(len(df_summary), 0)
        self.assertIn("pred_date", df_summary.columns)
        self.assertIn("strategy_mode", df_summary.columns)
        self.assertIn("win_rate", df_summary.columns)

    def test_get_weekly_verification_summary(self):
        from src.core.daily_verifier import get_weekly_verification_summary
        res = get_weekly_verification_summary(limit_days=5)
        self.assertIsInstance(res, dict)
        self.assertIn("overall_win_rate", res)
        self.assertIn("avg_net_ret", res)
        self.assertIn("best_strategy", res)
        self.assertIn("top_winners", res)
        self.assertIn("top_losers", res)
        self.assertIsInstance(res["df_history"], pd.DataFrame)
        self.assertIsInstance(res["strategy_summary"], pd.DataFrame)
        self.assertIsInstance(res["daily_trend"], pd.DataFrame)
        self.assertGreaterEqual(len(res["dates"]), 1)
        self.assertGreaterEqual(len(res["top_winners"]), 1)

    def test_evaluate_multi_horizon_tuning_impact(self):
        from src.core.daily_verifier import evaluate_multi_horizon_tuning_impact
        meta = evaluate_multi_horizon_tuning_impact()
        self.assertIsInstance(meta, dict)
        self.assertIn("comparative_table", meta)
        self.assertIsInstance(meta["comparative_table"], pd.DataFrame)
        self.assertEqual(len(meta["comparative_table"]), 3)
        self.assertEqual(meta["best_horizon_key"], "weekly")
        self.assertGreaterEqual(meta["best_expected_return"], 3.0)
        self.assertIn("best_proposals", meta)
        self.assertIn("horizon_details", meta)
        self.assertIn("daily", meta["horizon_details"])
        self.assertIn("weekly", meta["horizon_details"])
        self.assertIn("monthly", meta["horizon_details"])
        # Verify 3%+ return across all horizons
        for h_key in ["daily", "weekly", "monthly"]:
            self.assertGreaterEqual(meta["horizon_details"][h_key]["expected_return"], 3.0)

    def test_batch_verification_runners(self):
        from src.core.daily_verifier import run_weekly_batch_verification, run_monthly_batch_verification
        # Test with limit_days=1 to verify execution flow and return contract
        w_res = run_weekly_batch_verification(limit_days=1, force_refresh=False)
        self.assertIsInstance(w_res, dict)
        self.assertIn("overall_win_rate", w_res)
        self.assertIn("avg_net_ret", w_res)

        m_res = run_monthly_batch_verification(limit_days=1, force_refresh=False)
        self.assertIsInstance(m_res, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()

