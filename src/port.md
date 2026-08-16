```
VectorBT
   ↑
Vietnam execution layer
   ↑
Vietnam market rules
   ↑
HOSE / HNX / UPCoM data
```

# Strategy: Momentum-based trading strategy
signal = momentum > threshold


# Corporate actions
✓ Adjusted prices
✓ Cổ tức
✓ Chia tách
✓ Phát hành thêm
✓ Hủy niêm yết
✓ Chuyển sàn
✓ Ngày ngừng giao dịch


# Lọc cổ phiếu
Rank toàn thị trường
→ chọn Top 10 / Top 20
```
universe(date)
```


# Workflow
```
QUANT MODEL
"What do I want to buy?"
       ↓

EXECUTION MODEL
"Can I actually buy it?"
       ↓

PORTFOLIO
"How much should I buy?"
```

# Research question
```
Hypothesis:
Cổ phiếu Việt Nam tăng/giảm mạnh trong ngắn hạn
có hiện tượng đảo chiều không?

                    ↓

Universe:
HOSE/HNX/UPCoM

                    ↓

Liquidity filter:
loại cổ phiếu thanh khoản quá thấp

                    ↓

Alpha:
-return_N

                    ↓

Cross-sectional rank

                    ↓

N ∈ {3, 5, 10, 20}

                    ↓

Future return:
1d / 3d / 5d / 10d

                    ↓

Test IC
quintile return
turnover
cost
drawdown

                    ↓

Nếu robust
→ xây strategy
```