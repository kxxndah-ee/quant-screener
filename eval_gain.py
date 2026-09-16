import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import numpy as np
from src.collectors.market_data import fetch_ohlcv
from src.collectors.krx_universe import get_universe
from src.core.scoring import compute_point_in_time_indicators, calculate_score_for_row

df_univ = get_universe(active_only=True)
pool = df_univ.head(100)

stats = []

for _, r in pool.iterrows():
    code = r['code']
    name = r['name']
    try:
        df = fetch_ohlcv(code, start='2024-01-01')
        if '2026-09-09' not in df.index or '2026-09-10' not in df.index:
            continue
        df_pit = df.loc[:'2026-09-09']
        df_ind = compute_point_in_time_indicators(df_pit)
        if df_ind.empty:
            continue
        score_res = calculate_score_for_row(df_ind.iloc[-1])
        score = score_res['score']
        
        today = df.loc['2026-09-10']
        o = float(today['Open'])
        h = float(today['High'])
        l = float(today['Low'])
        c = float(today['Close'])
        if o <= 0:
            continue
        
        max_gain = ((h - o) / o) * 100.0
        close_gain = ((c - o) / o) * 100.0
        
        stats.append({
            'code': code,
            'name': name,
            'score': score,
            'max_gain': round(max_gain, 2),
            'close_gain': round(close_gain, 2),
            'open': o,
            'high': h,
            'low': l,
            'close': c
        })
    except Exception:
        pass

df = pd.DataFrame(stats)
df70 = df[df['score'] >= 70.0]

print(f"=== 고확신 상승 후보 (스코어 70점 이상) 총 {len(df70)}개 종목 기준 ===")
print(f"1. 장중 시가 대비 상승 (High > Open, +1.0% 이상): {sum(df70['max_gain'] >= 1.0)}개 / {len(df70)}개 -> {sum(df70['max_gain'] >= 1.0)/len(df70)*100:.1f}%")
print(f"2. 장중 +1.5% 이상 도달 (시스템 목표 익절선): {sum(df70['max_gain'] >= 1.5)}개 / {len(df70)}개 -> {sum(df70['max_gain'] >= 1.5)/len(df70)*100:.1f}%")
print(f"3. 장중 +2.0% 이상 도달: {sum(df70['max_gain'] >= 2.0)}개 / {len(df70)}개 -> {sum(df70['max_gain'] >= 2.0)/len(df70)*100:.1f}%")
print(f"4. 장중 +3.0% 이상 급등 도달: {sum(df70['max_gain'] >= 3.0)}개 / {len(df70)}개 -> {sum(df70['max_gain'] >= 3.0)/len(df70)*100:.1f}%")
print(f"5. 장중 +5.0% 이상 폭등 도달: {sum(df70['max_gain'] >= 5.0)}개 / {len(df70)}개 -> {sum(df70['max_gain'] >= 5.0)/len(df70)*100:.1f}%")

df_under = df[df['score'] < 70.0]
n_u = len(df_under)
print(f"\n=== 비교군: 스코어 70점 미만 종목 (총 {n_u}개) 기준 ===")
print(f"1. 장중 +1.0% 이상 상승 도달: {sum(df_under['max_gain'] >= 1.0)}개 / {n_u}개 -> {sum(df_under['max_gain'] >= 1.0)/n_u*100:.1f}%")
print(f"2. 장중 +2.0% 이상 도달: {sum(df_under['max_gain'] >= 2.0)}개 / {n_u}개 -> {sum(df_under['max_gain'] >= 2.0)/n_u*100:.1f}%")
print(f"3. 장중 +3.0% 이상 도달: {sum(df_under['max_gain'] >= 3.0)}개 / {n_u}개 -> {sum(df_under['max_gain'] >= 3.0)/n_u*100:.1f}%")

print("\n--- 장중 상승폭 상위 종목 ---")
print(df70[df70['max_gain'] > 0].sort_values('max_gain', ascending=False)[['name', 'code', 'score', 'max_gain', 'close_gain']].to_string(index=False))
