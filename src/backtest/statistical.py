"""
Statistical Significance Engine with 1,000-iteration Bootstrap Resampling.
Calculates 95% confidence intervals for win rate and excess return over KODEX 200.
Assigns automated verification badge: [● 검증됨] vs [● 실험적].
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


def calculate_profit_factor_and_mdd(returns_pct: List[float]) -> Tuple[float, float]:
    """Calculates Profit Factor and Maximum Drawdown % from sequence of trade return %."""
    if not returns_pct:
        return 0.0, 0.0

    gains = [r for r in returns_pct if r > 0]
    losses = [abs(r) for r in returns_pct if r < 0]

    sum_gain = sum(gains)
    sum_loss = sum(losses)

    if sum_loss > 0:
        profit_factor = round(sum_gain / sum_loss, 2)
    elif sum_gain > 0:
        profit_factor = 99.99
    else:
        profit_factor = 0.0

    # Max Drawdown based on compounding returns
    cum = np.cumprod(1.0 + np.array(returns_pct) / 100.0)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak * 100.0
    mdd = float(np.min(dd)) if len(dd) > 0 else 0.0

    return profit_factor, round(abs(mdd), 2)


def run_bootstrap_analysis(
    trades_df: pd.DataFrame,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Executes 1,000-iteration bootstrap resampling on trade returns and benchmark excess returns.
    Computes:
    - 95% CI for Win Rate
    - 95% CI for Excess Return vs KODEX 200
    - Verification Badge: "검증됨" if ER_CI_lower > 0 and WR_CI_lower > 50%, else "실험적"
    """
    n_trades = len(trades_df)
    if n_trades < 5:
        return {
            "status_badge": "실험적 (표본부족)",
            "is_verified": False,
            "badge_color": "gray",
            "total_trades": n_trades,
            "win_rate": 0.0,
            "win_rate_ci": (0.0, 0.0),
            "mean_excess_return": 0.0,
            "excess_return_ci": (0.0, 0.0),
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "mean_net_return": 0.0,
            "win_loss_ratio": 0.0,
        }

    rng = np.random.default_rng(random_seed)
    net_returns = trades_df["net_return_pct"].to_numpy()
    excess_returns = trades_df.get("excess_return_pct", trades_df["net_return_pct"]).to_numpy()

    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    # Sample-level stats
    actual_wins = np.sum(net_returns > 0)
    actual_win_rate = (actual_wins / n_trades) * 100.0
    actual_mean_excess = float(np.mean(excess_returns))
    actual_mean_net = float(np.mean(net_returns))

    pos_returns = net_returns[net_returns > 0]
    neg_returns = net_returns[net_returns < 0]
    avg_win = float(np.mean(pos_returns)) if len(pos_returns) > 0 else 0.0
    avg_loss = float(abs(np.mean(neg_returns))) if len(neg_returns) > 0 else 1.0
    win_loss_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0

    profit_factor, mdd = calculate_profit_factor_and_mdd(net_returns.tolist())

    # Bootstrap resampling
    bootstrap_win_rates = np.empty(n_iterations)
    bootstrap_excess_means = np.empty(n_iterations)

    for i in range(n_iterations):
        sample_indices = rng.integers(0, n_trades, size=n_trades)
        sample_net = net_returns[sample_indices]
        sample_excess = excess_returns[sample_indices]

        bootstrap_win_rates[i] = (np.sum(sample_net > 0) / n_trades) * 100.0
        bootstrap_excess_means[i] = np.mean(sample_excess)

    wr_lower = float(np.percentile(bootstrap_win_rates, lower_pct))
    wr_upper = float(np.percentile(bootstrap_win_rates, upper_pct))

    er_lower = float(np.percentile(bootstrap_excess_means, lower_pct))
    er_upper = float(np.percentile(bootstrap_excess_means, upper_pct))

    # Badge decision logic:
    # 1. Excess return 95% CI lower bound > 0
    # 2. Win rate 95% CI lower bound > 50%
    is_verified = (er_lower > 0.0) and (wr_lower > 50.0)

    if is_verified:
        badge_text = "검증됨"
        badge_color = "green"
    else:
        badge_text = "실험적"
        badge_color = "amber"

    return {
        "status_badge": badge_text,
        "is_verified": is_verified,
        "badge_color": badge_color,
        "total_trades": n_trades,
        "win_rate": round(actual_win_rate, 2),
        "win_rate_ci": (round(wr_lower, 2), round(wr_upper, 2)),
        "mean_excess_return": round(actual_mean_excess, 2),
        "excess_return_ci": (round(er_lower, 2), round(er_upper, 2)),
        "mean_net_return": round(actual_mean_net, 2),
        "profit_factor": profit_factor,
        "max_drawdown_pct": mdd,
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "win_loss_ratio": win_loss_ratio,
        "bootstrap_win_rates": bootstrap_win_rates.tolist()[:100],  # Sample for distribution display
    }


if __name__ == "__main__":
    dummy = pd.DataFrame({
        "net_return_pct": [2.5, -1.8, 1.5, 2.0, -1.9, 3.0, 1.8, -1.5, 2.2, 1.9, -2.0, 2.1],
        "excess_return_pct": [1.5, -0.8, 0.9, 1.2, -0.9, 2.1, 1.0, -0.5, 1.4, 1.1, -1.0, 1.3],
    })
    res = run_bootstrap_analysis(dummy, n_iterations=1000)
    print("Bootstrap Result:")
    for k, v in res.items():
        if k != "bootstrap_win_rates":
            print(f"  {k}: {v}")
