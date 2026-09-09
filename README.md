# VNQuant V0

## Chiến lược version 0:

`QuanHuuNhatMinh_11230998.ipynb`

## Cấu trúc mới:

Quant research workflow for Vietnamese equities with a modular pipeline:

`DATA -> FEATURE -> SIGNAL -> STRATEGY -> PORTFOLIO -> BACKTEST -> METRICS`

## Trading workflow

1. Store data
2. Lọc cổ phiếu
3. WebSocket with Rust/C++
4. Clean/normalize
5. Calculate signal (Tính momentum 20 ngày, Tính MACD)
6. Ranking
7. Chọn Mua/Ban
8. Portfolio Management
9. Backtest bằng VectorBT
10. Báo cáo performance in dashboard

## Cấu trúc mới

- Backtest: Use strategies on real trading
- Data:
  Data Storage (Parquet) static history price, firms' characteristics
  Websocket (Rust/C++/TypeScript) to fetch live trading prices

- Features: Calculate **Technical Analysis**
- Market: Tình hình thị trường nhóm ngành, tin tức doanh nghiệp
- Metrics: **Strategies valuation** - Sharpe ratio, volatility
- Portfolio: **Portfolio Management**
- Signals: Xem xét **điểm mua/bán** từ features
- Strategies(\*): Portfolio Weights, Risk calculation, Lọc cổ phiếu, Phân tích đầu tư

### Workflow

Data -> Market

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

## Các module chính

### Data

- `src/data/provider.py`: abstraction `VNDataProvider` với `get_history(tickers, start, end)`.
- `src/data/cleaner.py`: chuẩn hóa ngày, sort, dedupe, lọc giá/khối lượng không hợp lệ, gom về contract ma trận (`index=date`, `columns=ticker`).

### Features

- `momentum(close, window=20)`
- `volatility(close, window=20)`
- `volume_ratio(volume, window=20)`

### Signal

- `cross_sectional_rank(feature)`

### Strategy

- `top_quantile(signal, q=0.2)`

### Portfolio

- `equal_weight(selection)`
- `rebalance_weights(weights, frequency='daily'|'weekly')`

### Backtest

- `run_backtest(...)` trong `src/backtest/engine.py` bọc VectorBT.

### Metrics

- `compute_basic_report(...)` trả về các chỉ số: total return, CAGR, annualized vol, Sharpe, max drawdown, win rate, số trade, turnover, exposure, ending value.
