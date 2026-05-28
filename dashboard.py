"""
dashboard.py
Unified trading performance dashboard for Shares + Crypto bots.
Run locally:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="HMM Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp {
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f1629;
    border-right: 1px solid #1e2d4a;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    color: #f1f5f9;
}
[data-testid="stMetricDelta"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Space Mono', monospace;
    color: #f1f5f9;
}
h1 { font-size: 1.6rem !important; letter-spacing: -0.02em; }
h2 { font-size: 1.1rem !important; color: #94a3b8; border-bottom: 1px solid #1e2d4a; padding-bottom: 8px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1629;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #64748b;
    background: transparent;
    border-radius: 6px;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #38bdf8 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2d4a;
    border-radius: 8px;
}

/* Regime badge colours */
.regime-bull  { color: #22c55e; font-weight: 600; font-family: 'Space Mono', monospace; }
.regime-bear  { color: #ef4444; font-weight: 600; font-family: 'Space Mono', monospace; }
.regime-neutral { color: #f59e0b; font-weight: 600; font-family: 'Space Mono', monospace; }

/* Divider */
hr { border-color: #1e2d4a; }

/* Select box + number input */
.stSelectbox > div, .stNumberInput > div { background: #111827; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────

DATA_DIR = Path(os.getenv("DATA_DIR", "."))

FILES = {
    "shares": {
        "equity":  DATA_DIR / "live_equity_shares.csv",
        "trades":  DATA_DIR / "live_trades_shares.csv",
        "bt_equity": DATA_DIR / "backtest_equity.csv",
        "bt_trades": DATA_DIR / "backtest_trades.csv",
    },
    "crypto": {
        "equity":  DATA_DIR / "live_equity_crypto.csv",
        "trades":  DATA_DIR / "live_trades_crypto.csv",
        "bt_equity": DATA_DIR / "backtest_equity_crypto.csv",
        "bt_trades": DATA_DIR / "backtest_trades_crypto.csv",
    },
}


@st.cache_data(ttl=30)   # refresh every 30 seconds
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()


def load_bot(bot: str) -> dict:
    return {k: load_csv(v) for k, v in FILES[bot].items()}


# ── Metric helpers ────────────────────────────────────────────────────

def pct(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"


def compute_metrics(equity_df: pd.DataFrame, initial: float = 100_000) -> dict:
    if equity_df.empty or "equity" not in equity_df.columns:
        return {}
    eq = equity_df["equity"].dropna()
    if len(eq) < 2:
        return {}
    total_ret = (eq.iloc[-1] / initial - 1) * 100
    returns = eq.pct_change().dropna()
    ann_vol = returns.std() * np.sqrt(252 * 78) * 100   # ticks per day ≈ 78 for 5-min
    sharpe = (total_ret / ann_vol) if ann_vol > 0 else 0
    roll_max = eq.cummax()
    max_dd = ((eq - roll_max) / roll_max).min() * 100
    return {
        "current_equity": eq.iloc[-1],
        "total_return": total_ret,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "n_ticks": len(eq),
    }


def regime_color(r: str) -> str:
    r = str(r).upper()
    if r == "BULL":    return "#22c55e"
    if r == "BEAR":    return "#ef4444"
    return "#f59e0b"


# ── Chart helpers (using st.line_chart / plotly) ──────────────────────

def equity_chart(equity_df: pd.DataFrame, label: str, initial: float = 100_000):
    import plotly.graph_objects as go

    if equity_df.empty or "equity" not in equity_df.columns:
        st.info(f"No equity data yet for {label}.")
        return

    time_col = "timestamp" if "timestamp" in equity_df.columns else equity_df.index
    eq = equity_df.set_index("timestamp")["equity"] if "timestamp" in equity_df.columns else equity_df["equity"]

    color = "#38bdf8"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values,
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.07)",
        name="Equity",
    ))
    fig.add_hline(y=initial, line_dash="dot", line_color="#334155", line_width=1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#94a3b8", size=12),
        xaxis=dict(gridcolor="#1e2d4a", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#1e2d4a", showgrid=True, zeroline=False, tickprefix="$"),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)


def trades_bar_chart(trades_df: pd.DataFrame):
    import plotly.graph_objects as go

    if trades_df.empty:
        st.info("No trades yet.")
        return

    df = trades_df.copy()
    time_col = "timestamp" if "timestamp" in df.columns else "date"
    if time_col not in df.columns:
        st.info("No timestamp column in trades.")
        return

    df["date"] = pd.to_datetime(df[time_col]).dt.date
    daily = df.groupby(["date", "side"])["value"].sum().unstack(fill_value=0).reset_index()

    fig = go.Figure()
    if "buy" in daily.columns:
        fig.add_trace(go.Bar(x=daily["date"], y=daily["buy"],
                             name="Buy", marker_color="#22c55e", opacity=0.85))
    if "sell" in daily.columns:
        fig.add_trace(go.Bar(x=daily["date"], y=daily["sell"],
                             name="Sell", marker_color="#ef4444", opacity=0.85))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#94a3b8", size=12),
        xaxis=dict(gridcolor="#1e2d4a"),
        yaxis=dict(gridcolor="#1e2d4a", tickprefix="$"),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=220,
    )
    st.plotly_chart(fig, use_container_width=True)


def regime_pie(equity_df: pd.DataFrame):
    import plotly.graph_objects as go

    if equity_df.empty or "regime" not in equity_df.columns:
        return

    counts = equity_df["regime"].value_counts()
    colors = [regime_color(r) for r in counts.index]
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color="#0a0e1a", width=2)),
        textfont=dict(family="Space Mono", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#94a3b8"),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        height=200,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Bot section renderer ──────────────────────────────────────────────

def render_bot(bot: str, label: str, emoji: str, initial: float = 100_000):
    data = load_bot(bot)
    eq_df = data["equity"]
    tr_df = data["trades"]
    bt_eq = data["bt_equity"]
    bt_tr = data["bt_trades"]

    metrics = compute_metrics(eq_df, initial)

    # ── KPI row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        val = f"${metrics.get('current_equity', initial):,.0f}" if metrics else f"${initial:,.0f}"
        delta = pct(metrics["total_return"]) if metrics else None
        st.metric("Portfolio Value", val, delta)
    with c2:
        st.metric("Total Return", pct(metrics.get("total_return", 0)) if metrics else "—")
    with c3:
        st.metric("Max Drawdown", pct(metrics.get("max_drawdown", 0)) if metrics else "—")
    with c4:
        st.metric("Sharpe (approx)", f"{metrics.get('sharpe', 0):.2f}" if metrics else "—")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Equity curve + regime ──
    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.markdown("## Equity Curve")
        equity_chart(eq_df, label, initial)
    with col_right:
        st.markdown("## Regime Mix")
        regime_pie(eq_df)

    # ── Latest regime badge ──
    if not eq_df.empty and "regime" in eq_df.columns:
        latest_regime = eq_df["regime"].iloc[-1]
        col = regime_color(latest_regime)
        st.markdown(
            f"**Current Regime:** <span style='color:{col}; font-family:Space Mono; font-size:1rem'>▶ {latest_regime}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs: Live trades | Backtest ──
    live_tab, bt_tab = st.tabs(["📋 Live Trades", "🔬 Backtest"])

    with live_tab:
        st.markdown("## Daily Trade Volume")
        trades_bar_chart(tr_df)

        st.markdown("## Transaction Log")
        if tr_df.empty:
            st.info("No live trades recorded yet.")
        else:
            display = tr_df.sort_values(
                "timestamp" if "timestamp" in tr_df.columns else tr_df.columns[0],
                ascending=False
            ).head(200)
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "value": st.column_config.NumberColumn("Value ($)", format="$%.2f"),
                    "price": st.column_config.NumberColumn("Price", format="%.4f"),
                    "qty":   st.column_config.NumberColumn("Qty", format="%.6f"),
                }
            )

    with bt_tab:
        st.markdown("## Backtest Equity Curve")
        if not bt_eq.empty:
            # Rename index col if needed
            if bt_eq.columns[0] != "timestamp":
                bt_eq = bt_eq.rename(columns={bt_eq.columns[0]: "timestamp"})
            if "equity" not in bt_eq.columns and len(bt_eq.columns) >= 2:
                bt_eq = bt_eq.rename(columns={bt_eq.columns[1]: "equity"})
            equity_chart(bt_eq, f"{label} Backtest", initial)
        else:
            st.info("Run `python main.py --backtest` to generate backtest data.")

        st.markdown("## Backtest Trades")
        if not bt_tr.empty:
            st.dataframe(bt_tr.head(500), use_container_width=True, hide_index=True)
        else:
            st.info("No backtest trade data found.")


# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    initial_capital = st.number_input(
        "Initial Capital ($)", value=100_000, step=1000, min_value=1000
    )
    st.markdown("---")
    st.markdown("### 📁 Data Directory")
    data_dir_input = st.text_input(
        "Path to CSV folder",
        value=str(DATA_DIR),
        help="Folder containing live_equity_*.csv and live_trades_*.csv files",
    )
    if data_dir_input:
        DATA_DIR = Path(data_dir_input)
        for bot_key in FILES:
            for fkey, fpath in FILES[bot_key].items():
                FILES[bot_key][fkey] = DATA_DIR / fpath.name

    st.markdown("---")
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#334155'>Auto-refreshes every 30s</small>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<small style='color:#334155'>Last load: {datetime.now().strftime('%H:%M:%S')}</small>",
        unsafe_allow_html=True,
    )


# ── Main layout ───────────────────────────────────────────────────────

st.markdown("# 📈 HMM Trading Dashboard")
st.markdown(
    "<p style='color:#64748b; font-size:0.9rem; margin-top:-12px'>Live performance · Shares & Crypto · Alpaca Paper Trading</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

shares_tab, crypto_tab, combined_tab = st.tabs(["🏦 Shares Bot", "₿ Crypto Bot", "📊 Combined"])

with shares_tab:
    render_bot("shares", "Shares", "🏦", initial_capital)

with crypto_tab:
    render_bot("crypto", "Crypto", "₿", initial_capital)

with combined_tab:
    st.markdown("## Combined Portfolio Overview")

    s_data = load_bot("shares")
    c_data = load_bot("crypto")
    s_eq = s_data["equity"]
    c_eq = c_data["equity"]

    # Combined equity = sum of both
    if not s_eq.empty and not c_eq.empty and "equity" in s_eq.columns and "equity" in c_eq.columns:
        import plotly.graph_objects as go

        s = s_eq.set_index("timestamp")["equity"] if "timestamp" in s_eq.columns else s_eq["equity"]
        c = c_eq.set_index("timestamp")["equity"] if "timestamp" in c_eq.columns else c_eq["equity"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Shares Equity", f"${s.iloc[-1]:,.0f}")
        with col2:
            st.metric("Crypto Equity", f"${c.iloc[-1]:,.0f}")
        with col3:
            st.metric("Combined", f"${s.iloc[-1] + c.iloc[-1]:,.0f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Shares",
                                  line=dict(color="#38bdf8", width=2)))
        fig.add_trace(go.Scatter(x=c.index, y=c.values, name="Crypto",
                                  line=dict(color="#f59e0b", width=2)))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#94a3b8"),
            xaxis=dict(gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a", tickprefix="$"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Combined trades
        s_tr = s_data["trades"]
        c_tr = c_data["trades"]
        if not s_tr.empty or not c_tr.empty:
            s_tr["bot"] = "shares"
            c_tr["bot"] = "crypto"
            combined_trades = pd.concat([s_tr, c_tr]).sort_values(
                "timestamp", ascending=False
            ).head(300)
            st.markdown("## All Transactions")
            st.dataframe(combined_trades, use_container_width=True, hide_index=True)
    else:
        st.info("Start both bots to see the combined view.")
