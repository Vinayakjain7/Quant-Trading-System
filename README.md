# 📈 Quant Trading System (Multi-Asset Portfolio)

## 🚀 Overview

This project implements a multi-asset quantitative trading system combining:

* Mean Reversion Strategy
* Trend Filtering
* Stop-Loss Risk Management
* Portfolio Allocation (Volatility-Based)
* Hybrid Strategy (Aggressive + Safe)
* Periodic Rebalancing

---

## 🧠 Strategy Logic

### 1. Mean Reversion

* Buy when price falls below lower band
* Sell when price exceeds upper band

### 2. Trend Filter

* Trades only when price is near or above long-term moving average

### 3. Stop Loss

* Exit position if loss exceeds 5%

---

## 📊 Portfolio Construction

* Multi-asset approach (Tech, Banking, Energy, Healthcare, Consumer)
* Inverse volatility weighting
* Hybrid allocation:

  * 60% Aggressive assets
  * 40% Defensive assets

---

## 🔁 Rebalancing

* Portfolio weights updated quarterly
* Adapts to changing market volatility

---

## 📈 Performance

* Return: ~26%
* Sharpe Ratio: ~1.0
* Max Drawdown: ~-11%

---

## 📊 Dashboard

The system includes an interactive dashboard with:

* Equity curve
* Drawdown analysis
* Daily returns

---

## 🛠 Tech Stack

* Python
* Pandas, NumPy
* yFinance (data)
* Plotly (visualization)

---

## 📂 How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## ⚠️ Disclaimer

This project is for educational purposes only.
Backtested results do not guarantee future performance.

---

## 👤 Author

Built by a first-year AI student exploring quantitative finance and algorithmic trading.
