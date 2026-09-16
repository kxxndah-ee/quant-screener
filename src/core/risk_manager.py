"""
Portfolio Risk Management and Position Sizing Engine.
Implements:
- Half-Kelly sizing with score multiplier ((Score - 50) / 50)
- Single stock maximum 20% allocation ceiling
- Maximum 5 concurrent open positions
- Daily -3% loss circuit breaker
- 3 consecutive stop-loss cooldown mechanism
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def calculate_half_kelly_weight(
    win_rate: float,
    win_loss_ratio: float,
    score: float,
    max_cap: float = 0.20
) -> float:
    """
    Computes position sizing weight:
    Kelly formula: K = W - (1 - W) / R
    Half-Kelly: K_half = 0.5 * K
    Score Multiplier: M_score = max(0.0, (score - 50.0) / 50.0)
    Weight = min(max_cap, max(0.0, K_half * M_score))
    """
    if win_rate <= 0 or win_loss_ratio <= 0:
        return 0.0

    # Avoid division by zero
    k = win_rate - ((1.0 - win_rate) / win_loss_ratio)
    if k <= 0:
        return 0.0

    k_half = 0.5 * k

    # Score multiplier: Score 50 -> 0.0, Score 75 -> 0.5, Score 100 -> 1.0
    score_mult = max(0.0, (score - 50.0) / 50.0)
    target_weight = k_half * score_mult

    # Strict cap at max_cap (default 20%)
    return min(max_cap, max(0.0, target_weight))


class PortfolioRiskManager:
    """
    Manages live/simulated portfolio risk states:
    - Equity tracking
    - Daily circuit breaker (-3% limit)
    - 5 concurrent positions limit
    - 3 consecutive stop losses cooldown
    """

    def __init__(
        self,
        initial_equity: float = 10_000_000.0,
        max_positions: int = 5,
        max_single_weight: float = 0.20,
        daily_loss_limit_pct: float = -3.0,
        consecutive_loss_limit: int = 3
    ):
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.max_positions = max_positions
        self.max_single_weight = max_single_weight
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.consecutive_loss_limit = consecutive_loss_limit

        self.daily_realized_loss_pct = 0.0
        self.daily_realized_loss_amt = 0.0
        self.consecutive_stop_losses = 0
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.is_circuit_breaker_active = False
        self.is_cooldown_active = False

    def can_open_position(self, code: str) -> Tuple[bool, str]:
        """
        Evaluates whether a new position can be entered given current risk parameters.
        """
        # 1. Check Circuit Breaker
        if self.is_circuit_breaker_active:
            return False, "CIRCUIT_BREAKER_ACTIVE (-3% Daily Loss Reached)"

        # 2. Check Cooldown
        if self.is_cooldown_active:
            return False, f"COOLDOWN_ACTIVE ({self.consecutive_loss_limit} Consecutive Stop Losses)"

        # 3. Check Max Concurrent Positions
        if len(self.open_positions) >= self.max_positions:
            return False, f"MAX_POSITIONS_REACHED ({self.max_positions} Already Open)"

        # 4. Check already holding
        if code in self.open_positions:
            return False, f"ALREADY_HOLDING ({code} is currently held)"

        return True, "APPROVED"

    def calculate_order_shares(
        self,
        code: str,
        price: float,
        score: float,
        win_rate: float = 0.55,
        win_loss_ratio: float = 1.3
    ) -> Tuple[int, float]:
        """
        Calculates number of shares and allocated capital for order.
        Returns (shares, capital).
        """
        can_open, _ = self.can_open_position(code)
        if not can_open or price <= 0:
            return 0, 0.0

        weight = calculate_half_kelly_weight(
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            score=score,
            max_cap=self.max_single_weight
        )
        if weight <= 0:
            return 0, 0.0

        capital = self.current_equity * weight
        shares = int(capital // price)
        actual_capital = shares * price
        return shares, actual_capital

    def record_trade_exit(
        self,
        code: str,
        net_pnl_amt: float,
        net_return_pct: float,
        exit_code: str
    ) -> None:
        """
        Updates risk state when a trade closes.
        """
        if code in self.open_positions:
            del self.open_positions[code]

        self.current_equity += net_pnl_amt

        # Update daily loss
        if net_pnl_amt < 0:
            self.daily_realized_loss_amt += net_pnl_amt
            daily_pct = (self.daily_realized_loss_amt / self.initial_equity) * 100.0
            self.daily_realized_loss_pct = daily_pct

            # Check daily circuit breaker (-3%)
            if self.daily_realized_loss_pct <= self.daily_loss_limit_pct:
                self.is_circuit_breaker_active = True

        # Track consecutive stop losses
        if "SL" in exit_code:
            self.consecutive_stop_losses += 1
            if self.consecutive_stop_losses >= self.consecutive_loss_limit:
                self.is_cooldown_active = True
        else:
            self.consecutive_stop_losses = 0
            self.is_cooldown_active = False

    def reset_daily_state(self) -> None:
        """Resets daily circuit breaker and daily realized loss at start of new trading day."""
        self.daily_realized_loss_pct = 0.0
        self.daily_realized_loss_amt = 0.0
        self.is_circuit_breaker_active = False
        # Cooldown remains until cleared or manual reset


if __name__ == "__main__":
    rm = PortfolioRiskManager()
    weight = calculate_half_kelly_weight(0.58, 1.35, 82.0)
    print(f"Half-Kelly weight for WR 58%, R 1.35, Score 82: {weight:.2%}")
    shares, cap = rm.calculate_order_shares("005930", 70000, 82.0)
    print(f"Shares: {shares}, Capital: {cap:,.0f} KRW")
