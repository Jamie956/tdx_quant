import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from scripts.data_pipeline.adjust import forward_adjust
from scripts.data_pipeline.backtest import run_backtest

daily = pd.read_parquet('data/daily/ts_code=000001.SZ/data.parquet')
xdxr  = pd.read_parquet('data/xdxr/ts_code=000001.SZ/data.parquet')
adj   = forward_adjust(daily, xdxr)   # 前复权
r     = run_backtest(adj)             # 用复权价回测
print(r.summary())