import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.backtest.walk_forward import run_walk_forward_backtest

print("=== 삼성전자 (005930) 스코어 기준별 4개년 워크포워드 승률 비교 ===")
for s in [60.0, 70.0, 75.0, 80.0, 85.0]:
    res = run_walk_forward_backtest('005930', start_year=2021, min_entry_score=s)
    stats = res.get('stats', {})
    if stats and stats.get('total_trades', 0) > 0:
        trades = stats.get('total_trades')
        wr = stats.get('win_rate')
        ci = stats.get('win_rate_ci')
        alpha = stats.get('mean_excess_return')
        pf = stats.get('profit_factor')
        print(f"기준 {s:.0f}점 이상 -> 거래 {trades:3d}건 | 승률 {wr:5.1f}% (CI: {ci[0]:.1f}~{ci[1]:.1f}%) | 알파 {alpha:+5.2f}% | PF {pf:.2f}")
