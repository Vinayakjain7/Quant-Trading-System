import yfinance as yf
import pandas as pd
import numpy as np

# ── CONFIG ─────────────────────────────
TICKERS = [
    # 🏦 BANKING & FINANCE
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS",

    # 💻 IT
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",

    # 🛢️ ENERGY
    "RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS",

    # 🛍️ FMCG
    "ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",

    # 🚗 AUTO
    "TATAMOTORS.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",

    # 💊 PHARMA
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",

    # 🏗️ INFRA & CAPITAL GOODS
    "LT.NS", "ULTRACEMCO.NS", "GRASIM.NS", "ADANIPORTS.NS",

    # ⛏️ METALS
    "TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS",

    # 📡 TELECOM & OTHERS
    "BHARTIARTL.NS", "POWERGRID.NS", "NTPC.NS"
]

START = "2020-01-01"
END = "2024-01-01"

WINDOW = 20
LONG_MA = 100
STOP_LOSS = 0.05

CAPITAL = 100000  # ₹1L test
# ─────────────────────────────────────


def sharpe(r):
    r = r.dropna()
    if r.std() == 0:
        return 0
    return (r.mean() / r.std()) * np.sqrt(252)


def max_dd(eq):
    roll = eq.cummax()
    return ((eq - roll) / roll).min() * 100


def get_returns(ticker):
    df = yf.download(ticker, start=START, end=END)

    if df.empty:
        return pd.Series()

    df = df[['Close']].copy()

    df['mean'] = df['Close'].rolling(WINDOW).mean()
    df['std'] = df['Close'].rolling(WINDOW).std()
    df['upper'] = df['mean'] + 1.5 * df['std']
    df['lower'] = df['mean'] - 1.5 * df['std']
    df['long_ma'] = df['Close'].rolling(LONG_MA).mean()

    df = df.dropna()

    if df.empty:
        return pd.Series()

    pos = 0
    entry = 0
    positions = []

    for i in range(len(df)):
        price = float(df['Close'].iloc[i])
        lower = float(df['lower'].iloc[i])
        upper = float(df['upper'].iloc[i])
        long_ma = float(df['long_ma'].iloc[i])

        if pos == 1 and price < entry * (1 - STOP_LOSS):
            pos = 0

        if price > long_ma:
            if price < lower and pos == 0:
                pos = 1
                entry = price
            elif price > upper:
                pos = 0
        else:
            pos = 0

        positions.append(pos)

    df['returns'] = df['Close'].pct_change().fillna(0)

    strat = df['returns'] * pd.Series(positions, index=df.index).shift(1).fillna(0)

    return strat


# ── MAIN ──────────────────────────────
all_returns = []

for t in TICKERS:
    r = get_returns(t)
    if not r.empty:
        all_returns.append(r)

combined = pd.concat(all_returns, axis=1).dropna()

portfolio = combined.mean(axis=1)

equity = CAPITAL * (1 + portfolio).cumprod()

ret = (equity.iloc[-1] / CAPITAL - 1) * 100
sh = sharpe(portfolio)
dd = max_dd(equity)

print("\n=== INDIAN BACKTEST ===")
print(f"Return: {ret:.2f}%")
print(f"Sharpe: {sh:.2f}")
print(f"Max Drawdown: {dd:.2f}%")