from plyer import notification
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ── CONFIG (INDIA) ─────────────────────
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

WINDOW = 20
LONG_MA = 100
STOP_LOSS = 0.05
TOP_N = 5   # 🔥 only top 5 trades
# ─────────────────────────────────────
def send_alert(message):
    notification.notify(
        title="Trading Signal 🚀",
        message=message,
        timeout=10
    )

# ── SIGNAL + SCORE FUNCTION ───────────
def get_signal_and_score(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")

        if df.empty:
            return "NO DATA", 0, 0

        df = df[['Close']].copy()

        # Indicators
        df['mean'] = df['Close'].rolling(WINDOW).mean()
        df['std'] = df['Close'].rolling(WINDOW).std()

        df['upper'] = df['mean'] + 1.5 * df['std']
        df['lower'] = df['mean'] - 1.5 * df['std']

        df['long_ma'] = df['Close'].rolling(LONG_MA).mean()

        df = df.dropna()

        if df.empty:
            return "NO DATA", 0, 0

        pos = 0
        entry_price = 0

        for i in range(len(df)):
            price = float(df['Close'].iloc[i])
            lower = float(df['lower'].iloc[i])
            upper = float(df['upper'].iloc[i])
            long_ma = float(df['long_ma'].iloc[i])
            std = float(df['std'].iloc[i])

            # Stop loss
            if pos == 1 and price < entry_price * (1 - STOP_LOSS):
                pos = 0

            if price > long_ma and std > 0:

                if price < lower and pos == 0:
                    pos = 1
                    entry_price = price

                elif price > upper:
                    pos = 0

            else:
                pos = 0

        latest_price = float(df['Close'].iloc[-1])
        latest_lower = float(df['lower'].iloc[-1])
        latest_std = float(df['std'].iloc[-1])

        # 🔥 SCORING (alpha strength)
        score = (latest_lower - latest_price) / latest_std

        if pos == 1:
            return "BUY", latest_price, score
        else:
            return "SELL / HOLD", latest_price, score

    except Exception:
        return "NO DATA", 0, 0


# ── MAIN EXECUTION ────────────────────

signals_list = []
today = datetime.now().strftime("%Y-%m-%d")

print("\n=== TODAY'S SIGNALS (INDIA + RANKED) ===\n")

for ticker in TICKERS:
    signal, price, score = get_signal_and_score(ticker)

    if signal == "NO DATA":
        print(f"{ticker}: No data")
    else:
        print(f"{ticker}: {signal} at {price:.2f}")

    signals_list.append({
        "Date": today,
        "Stock": ticker,
        "Signal": signal,
        "Price": round(price, 2),
        "Score": score
    })

# Convert to DataFrame
df_signals = pd.DataFrame(signals_list)

# 🔥 FILTER BUY SIGNALS ONLY
buy_df = df_signals[df_signals['Signal'] == "BUY"]

# 🔥 RANK BEST STOCKS
buy_df = buy_df.sort_values(by="Score", ascending=False).head(TOP_N)

print("\n=== TOP TRADE OPPORTUNITIES ===\n")

if buy_df.empty:
    print("No BUY signals today")
else:
    print(buy_df[['Stock', 'Price', 'Score']])

    # 🔥 Send notification
    message = "BUY Signals:\n"
    for _, row in buy_df.iterrows():
        message += f"{row['Stock']} @ {row['Price']}\n"

    send_alert(message)

# Save daily file
daily_filename = f"signals_{today}.csv"
df_signals.to_csv(daily_filename, index=False)

# Save history
history_file = "signals_history.csv"
file_exists = os.path.isfile(history_file)

df_signals.to_csv(
    history_file,
    mode='a',
    header=not file_exists,
    index=False
)

print(f"\nSaved: {daily_filename}")
print("Updated: signals_history.csv")