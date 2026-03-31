import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── CONFIG ─────────────────────────────
AGGRESSIVE = ["AAPL","MSFT","TSLA","GOOGL","NVDA","AMZN"]
SAFE = ["JPM","BAC","XOM","JNJ","PFE","WMT","KO"]

START = "2020-01-01"
END = "2024-01-01"
WINDOW = 20
LONG_MA = 100
STOP_LOSS = 0.05
CAPITAL = 10000
# ─────────────────────────────────────


# ── METRICS ───────────────────────────
def sharpe(returns):
    returns = returns.dropna()
    if returns.std() == 0:
        return 0
    return (returns.mean() / returns.std()) * np.sqrt(252)

def max_drawdown(equity):
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return drawdown.min() * 100
# ─────────────────────────────────────


# ── STRATEGY FUNCTION ─────────────────
def get_strategy_returns(tickers):
    results = {}

    for ticker in tickers:
        df = yf.download(ticker, start=START, end=END)
        df = df[['Close']].copy()

        # Indicators
        df['mean'] = df['Close'].rolling(WINDOW).mean()
        df['std'] = df['Close'].rolling(WINDOW).std()
        df['upper'] = df['mean'] + 2 * df['std']
        df['lower'] = df['mean'] - 2 * df['std']
        df['long_ma'] = df['Close'].rolling(LONG_MA).mean()

        df = df.dropna()

        pos = 0
        entry_price = 0
        positions = []

        for i in range(len(df)):
            price = float(df['Close'].iloc[i])
            lower = float(df['lower'].iloc[i])
            upper = float(df['upper'].iloc[i])
            long_ma = float(df['long_ma'].iloc[i])

            # Stop loss
            if pos == 1 and price < entry_price * (1 - STOP_LOSS):
                pos = 0

            # Trading logic
            if price > 0.90 * long_ma:
                if price < lower and pos == 0:
                    pos = 1
                    entry_price = price
                elif price > upper:
                    pos = 0
            else:
                pos = 0

            positions.append(pos)

        df['position'] = positions
        df['returns'] = df['Close'].pct_change().fillna(0)

        df['strategy'] = (
            df['returns'] *
            pd.Series(positions, index=df.index).shift(1).fillna(0)
        )

        results[ticker] = df['strategy']

    return pd.DataFrame(results)
# ─────────────────────────────────────


# ── MAIN PIPELINE ─────────────────────

# Get returns
agg_df = get_strategy_returns(AGGRESSIVE)
safe_df = get_strategy_returns(SAFE)

# Align data
combined = pd.concat([agg_df, safe_df], axis=1).dropna()
agg_df = combined[agg_df.columns]
safe_df = combined[safe_df.columns]

# Portfolio construction
portfolio = []
index = combined.index

agg_weights = None
safe_weights = None

for i in range(len(index)):

    if i < 30:
        portfolio.append(0)
        continue

    # Quarterly rebalance
    if i % 63 == 0:
        agg_vol = agg_df.iloc[:i].std()
        safe_vol = safe_df.iloc[:i].std()

        agg_vol = agg_vol.replace(0, np.nan).dropna()
        safe_vol = safe_vol.replace(0, np.nan).dropna()

        agg_weights = (1 / agg_vol) / (1 / agg_vol).sum()
        safe_weights = (1 / safe_vol) / (1 / safe_vol).sum()

    if agg_weights is None or safe_weights is None:
        portfolio.append(0)
        continue

    agg_ret = (agg_df.iloc[i] * agg_weights).sum()
    safe_ret = (safe_df.iloc[i] * safe_weights).sum()

    total_ret = 0.6 * agg_ret + 0.4 * safe_ret
    portfolio.append(total_ret)

portfolio = pd.Series(portfolio, index=index)

# Equity
equity = CAPITAL * (1 + portfolio).cumprod()

# Metrics
ret = (equity.iloc[-1] / CAPITAL - 1) * 100
sh = sharpe(portfolio)
dd = max_drawdown(equity)

print("\n=== FINAL HYBRID SYSTEM ===")
print(f"Return: {ret:.2f}%")
print(f"Sharpe: {sh:.2f}")
print(f"Max Drawdown: {dd:.2f}%")

# ── DASHBOARD ─────────────────────────

roll_max = equity.cummax()
drawdown = (equity - roll_max) / roll_max * 100

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    subplot_titles=("Equity Curve", "Drawdown (%)", "Returns")
)

# Equity
fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Equity"), row=1, col=1)

# Drawdown
fig.add_trace(
    go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", name="Drawdown"),
    row=2, col=1
)

# Returns
fig.add_trace(
    go.Scatter(x=portfolio.index, y=portfolio, name="Returns"),
    row=3, col=1
)

fig.update_layout(height=800, title="Quant Trading Dashboard")
fig.show()
