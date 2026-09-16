import unittest
from src.backtest.exit_rules import simulate_intraday_exit, get_target_percentages


class TestExitRules(unittest.TestCase):
    def test_dynamic_targets(self):
        tp85, sl85 = get_target_percentages(85.0)
        tp75, sl75 = get_target_percentages(75.0)
        tp65, sl65 = get_target_percentages(65.0)

        self.assertEqual(tp85, 3.0)
        self.assertEqual(tp75, 2.0)
        self.assertEqual(tp65, 1.5)
        self.assertEqual(sl85, 2.0)

    def test_take_profit_hit(self):
        # Entry 10000, TP = 10300 (Score 80 -> +3%), High reaches 10350, Low only drops to 9900
        res = simulate_intraday_exit(
            entry_open=10000,
            day_high=10350,
            day_low=9900,
            day_close=10200,
            score=82.0
        )
        self.assertEqual(res["exit_code"], "TP")
        self.assertEqual(res["raw_exit_price"], 10300.0)

    def test_stop_loss_hit(self):
        # Entry 10000, SL = 9800 (-2%), High 10100, Low drops to 9750
        res = simulate_intraday_exit(
            entry_open=10000,
            day_high=10100,
            day_low=9750,
            day_close=9850,
            score=75.0
        )
        self.assertEqual(res["exit_code"], "SL")
        self.assertEqual(res["raw_exit_price"], 9800.0)

    def test_conservative_tie_breaker(self):
        # High touches 10350 (>= TP 10300) AND Low touches 9700 (<= SL 9800) on the same bar
        # Conservative principle: Stop-Loss priority!
        res = simulate_intraday_exit(
            entry_open=10000,
            day_high=10350,
            day_low=9700,
            day_close=10000,
            score=85.0
        )
        self.assertEqual(res["exit_code"], "SL_TIE")
        self.assertEqual(res["raw_exit_price"], 9800.0)
        self.assertIn("손절우선", res["exit_reason"])


if __name__ == "__main__":
    unittest.main()
