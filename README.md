# VNQuant V0

Quant research workflow for Vietnamese equities with a modular pipeline:

`DATA -> FEATURE -> SIGNAL -> STRATEGY -> PORTFOLIO -> BACKTEST -> METRICS`

## Mục tiêu

Mục tiêu V0 là chạy được vertical slice cơ bản:

1. Download OHLCV
2. Clean/normalize
3. Tính momentum 20 ngày
4. Cross-sectional rank
5. Chọn top 20% cổ phiếu
6. Equal weight
7. Backtest bằng VectorBT
8. Báo cáo performance

## Cấu trúc mới

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

## Cài đặt

```bash
pip install pandas scipy statsmodels matplotlib vectorbt vnstock backtesting
```

Hoặc dùng môi trường đã có `pyproject.toml`:

```bash
pip install -e .
```

## Chạy workflow

1. Cấu hình ở `config.py`
2. Chạy notebook kiểm tra: `notebooks/00_data_check.ipynb`
3. Chạy notebook momentum: `notebooks/01_momentum.ipynb`
4. Chạy test:

```bash
pytest
```

## Lưu ý cho V0

- `VNDataProvider` giữ abstraction; mặc định thử gọi `vnstock` nhưng có thể truyền `fetcher` custom cho ổn định môi trường.
- V0 chưa triển khai benchmark VNINDEX đầy đủ.
- V0 chưa có optimization nâng cao, execution rules chi tiết (fees/tax/lot size), intraday, live trading.
