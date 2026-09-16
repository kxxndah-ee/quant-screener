import unittest
from src.core.risk_manager import (
    calculate_half_kelly_weight,
    PortfolioRiskManager
)


class TestRiskManager(unittest.TestCase):
    def test_half_kelly_weight(self):
        # Win rate 60%, R = 1.5 -> Kelly K = 0.6 - 0.4/1.5 = 0.6 - 0.2667 = 0.3333
        # Half-Kelly = 0.1667
        # Score = 75 -> Multiplier = (75-50)/50 = 0.5
        # Expected weight = 0.1667 * 0.5 = 0.0833
        weight = calculate_half_kelly_weight(win_rate=0.60, win_loss_ratio=1.5, score=75.0)
        self.assertAlmostEqual(weight, 0.0833, places=3)

    def test_max_single_weight_cap(self):
        # Very high win rate and score -> should be capped at 20%
        weight = calculate_half_kelly_weight(win_rate=0.90, win_loss_ratio=3.0, score=100.0)
        self.assertEqual(weight, 0.20)

    def test_max_positions_limit(self):
        rm = PortfolioRiskManager(max_positions=2)
        rm.open_positions["005930"] = {}
        rm.open_positions["000660"] = {}

        can_open, reason = rm.can_open_position("035420")
        self.assertFalse(can_open)
        self.assertIn("MAX_POSITIONS", reason)

    def test_daily_circuit_breaker(self):
        rm = PortfolioRiskManager(initial_equity=10_000_000.0, daily_loss_limit_pct=-3.0)
        # Record loss of -350,000 KRW (-3.5%)
        rm.record_trade_exit("005930", net_pnl_amt=-350_000, net_return_pct=-3.5, exit_code="SL")

        self.assertTrue(rm.is_circuit_breaker_active)
        can_open, reason = rm.can_open_position("000660")
        self.assertFalse(can_open)
        self.assertIn("CIRCUIT_BREAKER", reason)

    def test_consecutive_stop_loss_cooldown(self):
        rm = PortfolioRiskManager(consecutive_loss_limit=3)
        rm.record_trade_exit("001", -20000, -2.0, "SL")
        self.assertFalse(rm.is_cooldown_active)
        rm.record_trade_exit("002", -20000, -2.0, "SL")
        self.assertFalse(rm.is_cooldown_active)
        rm.record_trade_exit("003", -20000, -2.0, "SL")
        self.assertTrue(rm.is_cooldown_active)

        can_open, reason = rm.can_open_position("004")
        self.assertFalse(can_open)
        self.assertIn("COOLDOWN", reason)


if __name__ == "__main__":
    unittest.main()
