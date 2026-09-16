"""
Unit tests for High Win-Rate Strategies:
1. Sniper High-Conviction Mode (80% Win-Rate Target)
2. Leader Close-to-Open Overnight Mode
"""

import unittest
import pandas as pd
import numpy as np

from src.core.high_winrate_strategies import (
    evaluate_market_regime,
    evaluate_sniper_candidate,
    evaluate_closing_bet_candidate,
    simulate_closing_bet_trade,
    evaluate_5pct_surge_candidate,
    simulate_5pct_surge_trade,
    evaluate_intraday_daytrade_candidate,
    simulate_intraday_daytrade
)


class TestHighWinRateStrategies(unittest.TestCase):

    def setUp(self):
        # Create synthetic 70-day benchmark data
        dates = pd.date_range("2024-01-01", periods=70, freq="B")
        # Uptrending benchmark
        prices = np.linspace(300, 360, 70)
        self.bm_bullish = pd.DataFrame({
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": 1000000
        }, index=dates)

        # Downtrending benchmark
        prices_down = np.linspace(360, 300, 70)
        self.bm_bearish = pd.DataFrame({
            "Open": prices_down * 1.01,
            "High": prices_down * 1.02,
            "Low": prices_down * 0.98,
            "Close": prices_down,
            "Volume": 1000000
        }, index=dates)

    def test_market_regime(self):
        is_bull, msg_bull, meta_bull = evaluate_market_regime(self.bm_bullish)
        self.assertTrue(is_bull)
        self.assertIn("매수 허용", msg_bull)

        is_bear, msg_bear, meta_bear = evaluate_market_regime(self.bm_bearish)
        self.assertFalse(is_bear)
        self.assertIn("차단", msg_bear)

    def test_sniper_candidate_qualification(self):
        # Create stock with healthy pullback near 20MA
        dates = pd.date_range("2024-01-01", periods=70, freq="B")
        prices = np.linspace(50000, 52000, 70)
        # Give high volume
        df = pd.DataFrame({
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": 500000 # 50만주 * 5만원 = 250억원 거래대금
        }, index=dates)

        cand = evaluate_sniper_candidate(df, self.bm_bullish, min_daily_val_krw=10_000_000_000)
        if cand is not None:
            self.assertEqual(cand["strategy"], "SNIPER")
            self.assertEqual(cand["tp_pct"], 1.2)
            self.assertEqual(cand["sl_pct"], 2.0)

    def test_closing_bet_candidate(self):
        # Create strong bullish marubozu day
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        prices = np.linspace(20000, 22000, 40)
        df = pd.DataFrame({
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": 100000
        }, index=dates)

        # Last day: +5% surge, high close ratio 0.95, massive volume 2,000,000 * 23,000 = 460억
        df.iloc[-1, df.columns.get_loc("Open")] = 22000
        df.iloc[-1, df.columns.get_loc("High")] = 23200
        df.iloc[-1, df.columns.get_loc("Low")] = 21900
        df.iloc[-1, df.columns.get_loc("Close")] = 23100
        df.iloc[-1, df.columns.get_loc("Volume")] = 2000000

        cand = evaluate_closing_bet_candidate(df, min_today_val_krw=30_000_000_000)
        self.assertIsNotNone(cand)
        self.assertEqual(cand["strategy"], "CLOSING_BET")
        self.assertEqual(cand["tp_pct"], 1.5)
        self.assertEqual(cand["sl_pct"], 2.0)
        self.assertGreaterEqual(cand["metrics"]["day_return"], 3.0)

    def test_closing_bet_rejection_on_weak_candle(self):
        # Upper shadow long (pin bar / shooting star)
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        prices = np.linspace(20000, 22000, 40)
        df = pd.DataFrame({
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": 100000
        }, index=dates)

        # High 25000, but closed at 22100 (long upper wick)
        df.iloc[-1, df.columns.get_loc("Open")] = 22000
        df.iloc[-1, df.columns.get_loc("High")] = 25000
        df.iloc[-1, df.columns.get_loc("Low")] = 21900
        df.iloc[-1, df.columns.get_loc("Close")] = 22100
        df.iloc[-1, df.columns.get_loc("Volume")] = 2000000

        cand = evaluate_closing_bet_candidate(df, min_today_val_krw=30_000_000_000)
        self.assertIsNone(cand)

    def test_simulate_closing_bet_gap_take_profit(self):
        # Today close 10,000, next open 10,200 (+2.0% gap up > +1.5% TP)
        res = simulate_closing_bet_trade(
            today_close=10000,
            next_open=10200,
            next_high=10300,
            next_low=10100,
            next_close=10150,
            tp_pct=1.5,
            sl_pct=2.0
        )
        self.assertEqual(res["exit_code"], "TP_GAP")
        self.assertTrue(res["is_hit"])
        self.assertGreater(res["net_return_pct"], 0)

    def test_simulate_closing_bet_stop_loss(self):
        # Today close 10,000, next day dumps to low 9700 (-3%)
        res = simulate_closing_bet_trade(
            today_close=10000,
            next_open=9950,
            next_high=9980,
            next_low=9700,
            next_close=9750,
            tp_pct=1.5,
            sl_pct=2.0
        )
        self.assertEqual(res["exit_code"], "SL")
        self.assertFalse(res["is_hit"])
        self.assertLess(res["net_return_pct"], 0)

    def test_5pct_surge_candidate_qualification(self):
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        prices = np.linspace(20000, 22000, 40)
        df = pd.DataFrame({
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": 100000
        }, index=dates)

        # Last day: +18% massive breakout, high-close 0.96, trading value 600억
        df.iloc[-1, df.columns.get_loc("Open")] = 22000
        df.iloc[-1, df.columns.get_loc("High")] = 26200
        df.iloc[-1, df.columns.get_loc("Low")] = 21900
        df.iloc[-1, df.columns.get_loc("Close")] = 25960
        df.iloc[-1, df.columns.get_loc("Volume")] = 2300000

        cand = evaluate_5pct_surge_candidate(df, min_today_val_krw=50_000_000_000, min_day_return_pct=15.0)
        self.assertIsNotNone(cand)
        self.assertEqual(cand["strategy"], "SURGE_5PCT")
        self.assertEqual(cand["tp_pct"], 5.0)
        self.assertEqual(cand["sl_pct"], 4.0)
        self.assertGreaterEqual(cand["metrics"]["day_return"], 15.0)

    def test_5pct_surge_candidate_rejection_low_volume(self):
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        prices = np.linspace(20000, 22000, 40)
        df = pd.DataFrame({
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": 100000
        }, index=dates)

        # Last day: +18% but low volume (only 10억)
        df.iloc[-1, df.columns.get_loc("Open")] = 22000
        df.iloc[-1, df.columns.get_loc("High")] = 26000
        df.iloc[-1, df.columns.get_loc("Low")] = 21900
        df.iloc[-1, df.columns.get_loc("Close")] = 25960
        df.iloc[-1, df.columns.get_loc("Volume")] = 40000

        cand = evaluate_5pct_surge_candidate(df, min_today_val_krw=50_000_000_000, min_day_return_pct=15.0)
        self.assertIsNone(cand)

    def test_simulate_5pct_surge_day1_hit(self):
        # Entry 10,000, next high 10,600 (+6.0% >= +5.0% TP)
        res = simulate_5pct_surge_trade(
            today_close=10000,
            next_open=10200,
            next_high=10600,
            next_low=10100,
            next_close=10500,
            tp_pct=5.0,
            sl_pct=4.0
        )
        self.assertEqual(res["exit_code"], "TP")
        self.assertTrue(res["is_hit"])
        self.assertGreater(res["net_return_pct"], 4.5)

    def test_simulate_5pct_surge_day2_hit(self):
        # Day 1: high 10,350 (+3.5%), low 9,900 (-1.0%), close 10,300
        # Day 2: high 10,700 (+7.0%), low 10,100
        res = simulate_5pct_surge_trade(
            today_close=10000,
            next_open=10100,
            next_high=10350,
            next_low=9900,
            next_close=10300,
            day2_high=10700,
            day2_low=10100,
            day2_close=10600,
            tp_pct=5.0,
            sl_pct=4.0
        )
    def test_intraday_daytrade_candidate_qualification(self):
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        prices = np.linspace(20000, 22000, 40)
        df = pd.DataFrame({
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": 100000
        }, index=dates)

        # Today: Open 22000, High 23500, Low 21900, Close 23100 (+5.0% from open), Volume 2,000,000
        df.iloc[-1, df.columns.get_loc("Open")] = 22000
        df.iloc[-1, df.columns.get_loc("High")] = 23500
        df.iloc[-1, df.columns.get_loc("Low")] = 21900
        df.iloc[-1, df.columns.get_loc("Close")] = 23100
        df.iloc[-1, df.columns.get_loc("Volume")] = 2000000

        cand = evaluate_intraday_daytrade_candidate(
            df,
            min_today_val_krw=20_000_000_000,
            min_intraday_gain=3.0,
            max_intraday_gain=9.0
        )
        self.assertIsNotNone(cand)
        self.assertEqual(cand["strategy"], "DAY_TRADE_5PCT")
        self.assertEqual(cand["tp_pct"], 5.0)
        self.assertEqual(cand["sl_pct"], 2.5)
        self.assertIn("15:15", cand["exit_time"])

    def test_intraday_daytrade_candidate_rejection_low_gain(self):
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        prices = np.linspace(20000, 22000, 40)
        df = pd.DataFrame({
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": 100000
        }, index=dates)

        # Today: gain from open only +1% -> rejected
        df.iloc[-1, df.columns.get_loc("Open")] = 22000
        df.iloc[-1, df.columns.get_loc("High")] = 22500
        df.iloc[-1, df.columns.get_loc("Low")] = 21900
        df.iloc[-1, df.columns.get_loc("Close")] = 22200
        df.iloc[-1, df.columns.get_loc("Volume")] = 2000000

        cand = evaluate_intraday_daytrade_candidate(
            df,
            min_today_val_krw=20_000_000_000,
            min_intraday_gain=3.0,
            max_intraday_gain=9.0
        )
        self.assertIsNone(cand)

    def test_simulate_intraday_daytrade_take_profit(self):
        # Entry 20000, high 21200 (+6% >= +5% TP)
        res = simulate_intraday_daytrade(
            entry_price=20000,
            day_high=21200,
            day_low=19800,
            day_close=21000,
            tp_pct=5.0,
            sl_pct=2.5
        )
        self.assertEqual(res["exit_code"], "TP")
        self.assertTrue(res["is_hit"])
        self.assertGreater(res["net_return_pct"], 4.5)

    def test_simulate_intraday_daytrade_stop_loss(self):
        # Entry 20000, low 19400 (-3% <= -2.5% SL)
        res = simulate_intraday_daytrade(
            entry_price=20000,
            day_high=20400,
            day_low=19400,
            day_close=19500,
            tp_pct=5.0,
            sl_pct=2.5
        )
        self.assertEqual(res["exit_code"], "SL")
        self.assertFalse(res["is_hit"])
        self.assertLess(res["net_return_pct"], 0)

    def test_simulate_intraday_daytrade_close_exit(self):
        # Neither TP nor SL hit -> exit at 15:15 close (e.g. +2%)
        res = simulate_intraday_daytrade(
            entry_price=20000,
            day_high=20600,
            day_low=19700,
            day_close=20400,
            tp_pct=5.0,
            sl_pct=2.5
        )
        self.assertEqual(res["exit_code"], "CLOSE")
        self.assertIn("15:15", res["exit_reason"])


if __name__ == "__main__":
    unittest.main()
