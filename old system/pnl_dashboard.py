import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load trades
df = pd.read_csv("trades.csv")

# Convert PnL to numeric
df['PnL'] = pd.to_numeric(df['PnL'], errors='coerce').fillna(0)

# Convert date
df['Date'] = pd.to_datetime(df['Date'])

# Sort
df = df.sort_values(by='Date')

# 📈 Cumulative PnL
df['Cumulative_PnL'] = df['PnL'].cumsum()

# 📉 Drawdown
df['Peak'] = df['Cumulative_PnL'].cummax()
df['Drawdown'] = df['Cumulative_PnL'] - df['Peak']

# ─────────────────────────────────────

fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=("Cumulative PnL", "Drawdown", "Trade PnL"),
    shared_xaxes=True
)

# 📈 Cumulative PnL
fig.add_trace(
    go.Scatter(x=df['Date'], y=df['Cumulative_PnL'], name="PnL"),
    row=1, col=1
)

# 📉 Drawdown
fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['Drawdown'],
        fill='tozeroy',
        name="Drawdown"
    ),
    row=2, col=1
)

# 📊 Trade-by-trade PnL
fig.add_trace(
    go.Bar(x=df['Date'], y=df['PnL'], name="Trade PnL"),
    row=3, col=1
)

fig.update_layout(height=800, title="PnL Dashboard")

fig.show()