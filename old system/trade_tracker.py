import pandas as pd
import os

# ── CONFIG ─────────────────────────────
CAPITAL = 100000
STOP_LOSS = 0.05
TOP_N = 5

RISK_PER_TRADE = 0.02      # 2% risk per trade
MAX_ALLOCATION = 0.2       # max 20% capital per stock
MAX_POSITIONS = 5          # max open trades
MAX_DRAWDOWN = -0.1        # stop trading if -10%

SIGNALS_FILE = "signals_history.csv"
TRADES_FILE = "trades.csv"
PORTFOLIO_FILE = "portfolio.csv"
# ─────────────────────────────────────


def load_data():
    if not os.path.exists(SIGNALS_FILE):
        print("❌ Run signals.py first")
        return None, None, None

    signals = pd.read_csv(SIGNALS_FILE)

    if signals.empty:
        print("❌ No signals available")
        return None, None, None

    portfolio = pd.read_csv(PORTFOLIO_FILE) if os.path.exists(PORTFOLIO_FILE) else pd.DataFrame(columns=["Stock","Entry Price","Quantity"])
    trades = pd.read_csv(TRADES_FILE) if os.path.exists(TRADES_FILE) else pd.DataFrame(columns=["Date","Stock","Action","Price","Quantity","PnL"])

    return signals, portfolio, trades


def process(signals, portfolio, trades):
    today = signals['Date'].iloc[-1]
    today_data = signals[signals['Date'] == today]

    buy_signals = today_data[today_data['Signal'] == "BUY"]

    if buy_signals.empty:
        print("No BUY signals today")
        return portfolio, trades

    # 🔥 Rank signals
    buy_signals = buy_signals.sort_values(by="Score", ascending=False).head(TOP_N)

    print("\n=== TOP TRADES ===")
    print(buy_signals[['Stock','Price','Score']])

    # ── BUY LOGIC ─────────────────────
    for _, row in buy_signals.iterrows():

        # 🔥 Max positions check
        if len(portfolio) >= MAX_POSITIONS:
            print("⚠️ Max positions reached")
            break

        stock = row['Stock']
        price = float(row['Price'])
        score = float(row['Score'])

        if stock in portfolio['Stock'].values:
            continue

        # 🔥 Risk-based sizing
        risk_amount = CAPITAL * RISK_PER_TRADE
        stop_loss_amount = price * STOP_LOSS
        qty_risk = int(risk_amount // stop_loss_amount)

        # 🔥 Allocation-based sizing
        allocation = CAPITAL * (score / buy_signals['Score'].sum())
        allocation = min(allocation, CAPITAL * MAX_ALLOCATION)
        qty_alloc = int(allocation // price)

        # 🔥 Final qty = minimum of both
        qty = min(qty_risk, qty_alloc)

        if qty <= 0:
            continue

        portfolio = pd.concat([portfolio, pd.DataFrame([{
            "Stock": stock,
            "Entry Price": price,
            "Quantity": qty
        }])], ignore_index=True)

        trades = pd.concat([trades, pd.DataFrame([{
            "Date": today,
            "Stock": stock,
            "Action": "BUY",
            "Price": price,
            "Quantity": qty,
            "PnL": 0
        }])], ignore_index=True)

        print(f"BUY {stock} | Qty: {qty}")

    # ── SELL LOGIC ────────────────────
    for _, row in portfolio.copy().iterrows():
        stock = row['Stock']
        entry = float(row['Entry Price'])
        qty = float(row['Quantity'])

        signal_row = today_data[today_data['Stock'] == stock]

        if signal_row.empty:
            continue

        signal = signal_row.iloc[0]['Signal']
        price = float(signal_row.iloc[0]['Price'])

        if signal == "SELL / HOLD" or price < entry * (1 - STOP_LOSS):

            pnl = (price - entry) * qty

            trades = pd.concat([trades, pd.DataFrame([{
                "Date": today,
                "Stock": stock,
                "Action": "SELL",
                "Price": price,
                "Quantity": qty,
                "PnL": pnl
            }])], ignore_index=True)

            portfolio = portfolio[portfolio['Stock'] != stock]

            print(f"SELL {stock} | PnL: {pnl:.2f}")

    return portfolio, trades


def main():
    signals, portfolio, trades = load_data()

    if signals is None:
        return

    # 🔥 Drawdown check
    if not trades.empty:
        total_pnl = trades['PnL'].sum()
        drawdown = total_pnl / CAPITAL

        if drawdown < MAX_DRAWDOWN:
            print("⚠️ Drawdown limit hit. Trading stopped.")
            return

    portfolio, trades = process(signals, portfolio, trades)

    portfolio.to_csv(PORTFOLIO_FILE, index=False)
    trades.to_csv(TRADES_FILE, index=False)

    print("\n=== PORTFOLIO ===")
    print(portfolio)

    print("\n=== TOTAL PnL ===")
    print(trades['PnL'].sum())


if __name__ == "__main__":
    main()