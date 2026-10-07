import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Tạo dữ liệu giá mẫu để thử DataFrame và biểu đồ, không cần gọi API.
rng = np.random.default_rng(42)
dates = pd.bdate_range(start="2025-01-01", periods=100)
daily_returns = rng.normal(loc=0.0005, scale=0.015, size=len(dates))
close = 100 * (1 + daily_returns).cumprod()

frame = pd.DataFrame(
    {
        "close": close,
        "volume": rng.integers(500_000, 2_000_000, size=len(dates)),
    },
    index=dates,
)
frame.index.name = "date"
print(frame.head())

fig, (price_ax, volume_ax) = plt.subplots(
    2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
price_ax.plot(frame.index, frame["close"], color="royalblue", label="Close")
price_ax.set_title("Demo stock price")
price_ax.set_ylabel("Price")
price_ax.legend()
price_ax.grid(alpha=0.3)

volume_ax.bar(frame.index, frame["volume"], color="darkorange", width=0.7)
volume_ax.set_ylabel("Volume")
volume_ax.set_xlabel("Date")
volume_ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
plt.show()
