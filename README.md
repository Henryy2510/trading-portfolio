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

## Chạy workflow với vnstock

`python main.py` tải OHLCV cho 10 mã trong `config.py` trong khoảng ngày
`START_DATE`–`END_DATE`, chuẩn hóa dữ liệu, rồi chạy dynamic workflow. Mã nguồn
thô được lưu tại `data/raw/`; các ma trận OHLCV đã clean được lưu tại
`data/processed/`. Kết quả gồm thứ hạng các mô hình trên tập train và báo cáo
backtest out-of-sample trên tập test.

Để dùng ma trận giá đóng cửa từ CSV thay cho tải dữ liệu:

```bash
python main.py --csv path/to/close_prices.csv
```

vnstock cần được cài trong môi trường Python đang chạy. Nếu tải một mã thất bại,
workflow dừng và báo lỗi để tránh âm thầm tạo danh mục từ thiếu mã.
