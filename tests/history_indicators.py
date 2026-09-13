import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from scripts.data_pipeline.indicators import compute_all
from scripts.data_pipeline.screener.conditions import (
    golden_cross_series,
    hammer_series,
    kdj_golden_cross_series,
)
from scripts.data_pipeline.signals import trend_snapshot

df = pd.read_parquet('data/daily/ts_code=601088.SH/data.parquet')
df = df.sort_values('trade_date').reset_index(drop=True)

# 先在全量上算指标（MACD/KDJ 是递推 EMA，切片后再算会缺 warmup），
# 最后再按日期筛出最近 12 个月的交叉点。
ind = compute_all(df, timeframe='daily')

cutoff = (
    pd.to_datetime(ind['trade_date'].max(), format='%Y%m%d') - pd.DateOffset(months=12)
).strftime('%Y%m%d')
recent = ind['trade_date'] >= cutoff  # 'YYYYMMDD' 字符串按字典序比较即时间序

# 复用 conditions.py 里的金叉定义（向量化版本），避免重复写交叉判定
macd_cross = golden_cross_series(ind)
kdj_cross  = kdj_golden_cross_series(ind)
hammer     = hammer_series(ind)

print('趋势判断：')
print(trend_snapshot(df).to_string())
print()
print('MACD 金叉（截止', ind['trade_date'].max(), '）')
print(ind[macd_cross & recent][['trade_date', 'close', 'DIF', 'DEA']])
print('KDJ 金叉')
print(ind[kdj_cross & recent][['trade_date', 'close', 'K', 'D']])
print('锤头线（下锤线）')
print(ind[hammer & recent][['trade_date', 'open', 'high', 'low', 'close']])
