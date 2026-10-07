# VNQuant V0

## Chiến lược version 0:

`initial_trading_method.ipynb`

## New Workflow

1. Store data
2. Lọc
3. WebSocket with Rust/C++
4. Clean/normalize
5. Calculate signal
6. Ranking
7. Carry out transaction
8. Portfolio Management
9. Backtesting
10. Báo cáo performance

### Workflow

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── src/
│   ├── data/
│   │   ├── provider.py
│   │   └── cleaner.py
│   ├── features/
│   │   ├── momentum.py
│   │   ├── volatility.py
│   │   └── liquidity.py
│   ├── signals/
│   │   ├── ranking.py
│   │   └── transforms.py
│   ├── strategies/
│   │   └── long_only.py
│   ├── portfolio/
│   │   └── weighting.py
│   ├── market/
│   │   └── vietnam.py
│   ├── backtest/
│   │   └── engine.py
│   └── metrics/
│       └── performance.py
├── notebooks/
│   ├── 00_data_check.ipynb
│   └── 01_momentum.ipynb
├── tests/
├── config.py
├── main.py
└── README.md
```
