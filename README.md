# VNQuant — Quant Research Workflow cho thị trường Việt Nam

## 1. Mục tiêu

Xây dựng một **quant research framework bằng Python cho cổ phiếu Việt Nam**, phục vụ nghiên cứu alpha trên:

* HOSE
* HNX
* UPCoM

Framework dùng:

* `vnstock` để lấy dữ liệu
* `pandas`, `numpy` để xử lý dữ liệu
* `scipy`, `statsmodels` cho thống kê
* `matplotlib` cho visualization
* `vectorbt` làm backtesting engine chính
* `backtesting.py` đã cài nhưng **chưa sử dụng trong V0**

Không sửa source code của `vectorbt`.

Các rule riêng của thị trường Việt Nam sẽ được xây dựng thành layer riêng sau.

---

# 2. Nguyên tắc kiến trúc

Pipeline phải tách rõ:

```text
DATA
↓
FEATURE
↓
SIGNAL
↓
STRATEGY
↓
PORTFOLIO
↓
BACKTEST
↓
METRICS / REPORT
```

Không viết toàn bộ logic vào notebook.

Notebook chỉ dùng để:

* nghiên cứu hypothesis;
* gọi các function/module trong `src/`;
* visualize kết quả.

Code có khả năng tái sử dụng phải nằm trong `src/`.

---

# 3. Repository structure

Tạo project theo cấu trúc:

```text
quant/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── src/
│   ├── data/
│   │   ├── provider.py
│   │   └── cleaner.py
│   │
│   ├── features/
│   │   ├── momentum.py
│   │   ├── volatility.py
│   │   └── liquidity.py
│   │
│   ├── signals/
│   │   ├── ranking.py
│   │   └── transforms.py
│   │
│   ├── strategies/
│   │   └── long_only.py
│   │
│   ├── portfolio/
│   │   └── weighting.py
│   │
│   ├── market/
│   │   └── vietnam.py
│   │
│   ├── backtest/
│   │   └── engine.py
│   │
│   └── metrics/
│       └── performance.py
│
├── notebooks/
│   ├── 00_data_check.ipynb
│   └── 01_momentum.ipynb
│
├── reports/
├── tests/
├── config.py
├── requirements.txt
└── README.md
```

Có thể điều chỉnh nhẹ nếu có lý do kiến trúc rõ ràng.

---

# 4. Việc đầu tiên: lập kế hoạch

Trước khi code:

1. Inspect environment và package versions.
2. Kiểm tra API hiện tại của `vnstock`.
3. Kiểm tra API hiện tại của `vectorbt`.
4. Đề xuất implementation plan.
5. Xác định dependency giữa các module.
6. Sau đó mới bắt đầu implement.

Không hỏi lại nếu có thể tự kiểm tra từ package/API hiện có.

---

# 5. Workflow V0 cần hoàn thành

Mục tiêu V0 là chạy được pipeline:

```text
VNStock
↓
Download OHLCV
↓
Clean / normalize
↓
Close matrix
↓
20-day Momentum
↓
Cross-sectional Rank
↓
Select Top 20–30%
↓
Equal Weight
↓
VectorBT
↓
Performance Report
```

Chỉ cần khoảng 10–20 cổ phiếu thanh khoản tốt để kiểm tra hệ thống.

Ví dụ universe ban đầu:

```text
FPT
HPG
VCB
VNM
MWG
SSI
MBB
TCB
VIC
VHM
```

Có thể thay ticker không khả dụng bằng ticker phù hợp.

---

# 6. Data layer

## 6.1 VNStock provider

Xây abstraction:

```python
class VNDataProvider:
    ...
```

Tối thiểu hỗ trợ:

```python
get_history(
    tickers,
    start,
    end
)
```

và nếu API cho phép:

```python
get_universe(exchange)
```

Không để notebook gọi trực tiếp `vnstock` khắp nơi.

Mục tiêu là sau này có thể thay data provider mà không cần sửa feature/strategy.

---

# 7. Data contract

Toàn framework sử dụng các matrix:

```python
open_
high
low
close
volume
```

Format:

```text
index   = trading date
columns = ticker
values  = market data
```

Ví dụ:

```text
Date         FPT      VNM      HPG
2026-01-02   100.2    62.1     27.5
2026-01-05   101.5    61.8     28.0
2026-01-06   103.0    62.5     27.8
```

Các feature phải giữ cùng:

* index
* columns

với input data.

---

# 8. Data cleaning

Tạo logic tối thiểu cho:

* datetime normalization;
* sorting;
* duplicated records;
* missing values;
* invalid prices;
* invalid/non-positive volume;
* ticker alignment;
* trading-date alignment.

Không tự forward-fill price qua những khoảng dài một cách tùy tiện.

Ghi rõ assumptions.

---

# 9. Feature layer

Implement tối thiểu:

## Momentum

```python
def momentum(close, window=20):
    return close.pct_change(window)
```

## Volatility

```python
def volatility(close, window=20):
    returns = close.pct_change()
    return returns.rolling(window).std()
```

## Volume ratio / volume surprise

```python
def volume_ratio(volume, window=20):
    return volume / volume.rolling(window).mean()
```

Feature chỉ đo lường.

Feature **không được quyết định BUY/SELL**.

---

# 10. Signal layer

Implement:

## Cross-sectional rank

Mỗi ngày rank cổ phiếu trong universe:

```python
def cross_sectional_rank(feature):
    return feature.rank(
        axis=1,
        pct=True
    )
```

Ví dụ:

```text
raw momentum:

FPT   +12%
HPG    +8%
VCB    +4%
VNM    -2%

↓

rank:

FPT   1.00
HPG   0.75
VCB   0.50
VNM   0.25
```

---

# 11. Strategy V0

Chỉ xây strategy:

```text
Long-only
```

Không short cổ phiếu cơ sở.

Ví dụ:

```python
def top_quantile(signal, q=0.2):
    ...
```

Mua các cổ phiếu nằm trong top `q` của signal.

Các cổ phiếu còn lại:

```text
weight = 0
```

---

# 12. Portfolio construction

Ban đầu dùng:

```text
Equal Weight
```

Ví dụ nếu chọn 5 cổ phiếu:

```text
20% mỗi cổ phiếu
```

Implement reusable function:

```python
def equal_weight(selection):
    ...
```

Không sử dụng Markowitz, mean-variance optimization hoặc ML trong V0.

---

# 13. Rebalancing

Thiết kế để có thể cấu hình:

```text
daily
weekly
```

Nhưng V0 chỉ cần một implementation đơn giản hoạt động ổn định.

Tránh unnecessary turnover nếu signal không thay đổi.

---

# 14. VectorBT backtest wrapper

Tạo một abstraction trong:

```text
src/backtest/engine.py
```

Không để notebook gọi API VectorBT phức tạp trực tiếp.

Wrapper phải nhận được:

* prices;
* signals hoặc weights;
* fees;
* slippage;
* initial cash;
* frequency.

Trả về:

* VectorBT portfolio;
* standardized performance results.

VectorBT là backtest engine chính.

Không sử dụng `backtesting.py` trong V0.

---

# 15. Metrics

Tạo standardized report tối thiểu gồm:

* Total Return
* CAGR / Annualized Return
* Annualized Volatility
* Sharpe Ratio
* Maximum Drawdown
* Win Rate nếu phù hợp
* Number of Trades
* Turnover nếu có thể tính chính xác
* Exposure
* Ending Portfolio Value

Nếu metric không trực tiếp có từ VectorBT, implement cẩn thận và có test.

---

# 16. Benchmark

Thiết kế interface để strategy có thể so sánh với benchmark.

Mục tiêu cuối:

```text
Strategy
vs
VNINDEX
```

Nếu VNINDEX chưa lấy được ổn định trong V0:

* tạo benchmark interface;
* document limitation;
* không block việc hoàn thành V0.

---

# 17. Notebook 00 — Data Check

`notebooks/00_data_check.ipynb`

Phải:

1. download universe mẫu;
2. inspect OHLCV;
3. kiểm tra missing data;
4. kiểm tra date range;
5. visualize 1–3 cổ phiếu;
6. xác minh standardized matrices;
7. kiểm tra data quality cơ bản.

---

# 18. Notebook 01 — Momentum experiment

`notebooks/01_momentum.ipynb`

Phải thực hiện:

```text
Close
↓
20-day return
↓
Cross-sectional rank
↓
Top 20–30%
↓
Equal weight
↓
Backtest
↓
Report
```

Visualize:

* equity curve;
* drawdown;
* basic performance statistics.

Notebook phải ngắn và chủ yếu gọi reusable code từ `src/`.

---

# 19. Configuration

Tạo configuration tập trung, ví dụ:

```python
START_DATE
END_DATE

INITIAL_CASH

FEES
SLIPPAGE

MOMENTUM_WINDOW

TOP_QUANTILE

REBALANCE_FREQUENCY
```

Không hard-code parameter rải rác khắp project.

---

# 20. Tests

Viết unit tests tối thiểu cho:

### Features

```text
momentum
volatility
volume_ratio
```

### Signals

```text
cross-sectional rank
```

### Portfolio

```text
equal weighting
weights sum correctly
no negative weight
```

### Data

```text
date sorting
duplicate handling
column alignment
```

Tests không cần exhaustive nhưng phải bảo vệ các logic cốt lõi.

---

# 21. README

README phải giải thích:

## Project purpose

Quant research framework cho cổ phiếu Việt Nam.

## Architecture

```text
Data → Feature → Signal → Strategy → Portfolio → Backtest
```

## Installation

Dùng environment hiện tại hoặc:

```bash
pip install pandas scipy statsmodels matplotlib vectorbt vnstock backtesting
```

## Running

Cách:

* lấy data;
* chạy notebook;
* chạy tests.

## Current scope

V0 chưa phải live trading system.

---

# 22. Không làm trong V0

Không implement:

* Machine Learning;
* Deep Learning;
* LSTM;
* Reinforcement Learning;
* optimization phức tạp;
* automated alpha mining;
* live trading;
* broker API;
* intraday/HFT;
* options;
* futures;
* short selling engine;
* full point-in-time fundamental database;
* toàn bộ market microstructure của Việt Nam.

Không over-engineer.

Mục tiêu là có một **vertical slice chạy hoàn chỉnh**.

---

# 23. Vietnam Market Layer

Tạo file/interface:

```text
src/market/vietnam.py
```

Nhưng trong V0 chỉ cần thiết kế architecture/placeholder phù hợp.

Sau V0 sẽ lần lượt bổ sung:

```text
V1
fees
tax
slippage
liquidity constraints

V2
HOSE/HNX/UPCoM market rules
price ceiling/floor
suspension
lot size
corporate actions

V3
historical universe
delisting
exchange migration
point-in-time fundamentals

V4
IC
Rank IC
quintile analysis
walk-forward validation
alpha correlation
parameter sensitivity
sector robustness
HOSE/HNX/UPCoM robustness
```

Không cần implement V1–V4 ngay.

---

# 24. Design constraints

Ưu tiên:

```text
simple
modular
testable
replaceable
research-friendly
```

Tránh:

```text
huge classes
deep inheritance
complex dependency injection
premature optimization
duplicate code
notebook-based architecture
```

Functions thuần túy được ưu tiên nếu phù hợp.

---

# 25. Definition of Done — V0

V0 được coi là hoàn thành khi có thể chạy:

```text
10–20 Vietnamese stocks
↓
historical OHLCV
↓
clean standardized matrices
↓
momentum_20 feature
↓
cross-sectional rank
↓
top quantile selection
↓
equal-weight portfolio
↓
VectorBT backtest
↓
performance report
```

và:

```text
pytest
```

pass các test cốt lõi.

Một người mới clone project phải hiểu được:

```text
where data comes from
how feature is created
how signal is generated
how portfolio is constructed
how backtest is executed
```

---

# 26. Cách thực hiện

Hãy:

1. phân tích yêu cầu;
2. inspect packages/APIs hiện tại;
3. đưa ra implementation plan ngắn gọn;
4. tạo repository structure;
5. implement từng layer theo dependency order;
6. test sau mỗi layer;
7. chạy thử pipeline end-to-end;
8. sửa lỗi;
9. hoàn thành notebook demo;
10. cập nhật README;
11. báo cáo những gì đã hoàn thành và những limitation còn lại.

Ưu tiên có **working V0** trước khi bổ sung bất kỳ tính năng nâng cao nào.
