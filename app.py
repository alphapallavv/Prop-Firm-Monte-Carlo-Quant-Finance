import streamlit as st
import pandas as pd
import numpy as np
import time

from monte_carlo import returns_from_csv, simulate_bootstrap_paths
from utils import plot_equity_cloud, plot_histogram, compute_metrics

# --- Streamlit setup ---
st.set_page_config(
    page_title="Prop Firm Monte Carlo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Apply CSS styling (safe loader) ---
try:
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("⚠️ Custom CSS not found. Using default Streamlit theme.")

# --- App title & description ---
st.title("Prop Firm Monte Carlo — Quant Dashboard")
st.write("Upload your TradingView CSV to simulate pass probability under prop firm rules.")

# --- Sidebar inputs ---
st.sidebar.header("Simulation Inputs")
uploaded_file = st.sidebar.file_uploader("Upload TradingView CSV (Close column)", type=['csv'])

default_start = 50000
starting_equity = st.sidebar.number_input("Starting Equity ($)", value=float(default_start), step=1000.0)
target_profit = st.sidebar.number_input("Profit Target ($)", value=3000.0, step=100.0)
max_days = st.sidebar.number_input("Max Days", value=180, min_value=1, step=1)
daily_loss = st.sidebar.number_input("Daily Loss Limit ($)", value=1000.0, step=50.0)
static_dd = st.sidebar.number_input("Static Drawdown Limit ($)", value=2000.0, step=50.0)
trailing_dd = st.sidebar.number_input("Trailing Drawdown Limit ($)", value=2000.0, step=50.0)
n_sims = st.sidebar.number_input("Number of Simulations", value=5000, min_value=100, step=100)
seed = st.sidebar.number_input("Random Seed (optional)", value=42, step=1)

run_button = st.sidebar.button("Run Monte Carlo")

# --- Main layout ---
col1, col2 = st.columns((1, 1))

# --- Load CSV or fallback to demo ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV uploaded successfully!")
    except Exception:
        st.warning("⚠️ Uploaded file unreadable. Using sample_trades.csv instead.")
        try:
            df = pd.read_csv("sample_trades.csv")
            st.info("Loaded sample_trades.csv successfully.")
        except FileNotFoundError:
            st.error("❌ No valid CSV file found. Please upload a CSV or add sample_trades.csv.")
            df = None
else:
    st.warning("⚠️ No file uploaded. Using sample_trades.csv as fallback.")
    try:
        df = pd.read_csv("sample_trades.csv")
        st.info("Loaded sample_trades.csv successfully.")
    except FileNotFoundError:
        st.error("❌ sample_trades.csv not found! Please upload a CSV or add a sample_trades.csv file.")
        df = None

# --- Simulation Execution ---
if df is not None and run_button:
    try:
        returns = returns_from_csv(df)
    except Exception as e:
        st.error("Error parsing CSV: " + str(e))
        st.stop()

    params = {
        "starting_equity": starting_equity,
        "target_profit": target_profit,
        "max_days": max_days,
        "daily_loss_limit": daily_loss,
        "static_drawdown_limit": static_dd,
        "trailing_drawdown_limit": trailing_dd,
        "n_sims": n_sims,
        "seed": seed
    }

    t0 = time.time()
    with st.spinner("🚀 Running simulations... please wait."):
        results = simulate_bootstrap_paths(returns, params)
    t1 = time.time()

    metrics = compute_metrics(results, starting_equity)

    # --- Display metrics ---
    col1.metric("Pass Rate", f"{metrics['pass_rate']:.2f}%")
    col1.metric("Simulations", metrics['n_sims'])
    col1.metric("Passes", metrics['passes'])

    fail_counts = metrics['fail_counts']
    col2.write("### Fail Reason Breakdown:")
    col2.json(fail_counts)

    # --- Equity path cloud ---
    st.subheader("📈 Equity Path Cloud")
    fig = plot_equity_cloud(results['equity_matrix'], sample_n=300, title="Monte Carlo Equity Cloud")
    st.plotly_chart(fig, use_container_width=True)

    # --- Final P&L Histogram ---
    st.subheader("💰 Final P&L Distribution")
    hist = plot_histogram(results['final_equity'], starting_equity, bins=60)
    st.plotly_chart(hist, use_container_width=True)

    # --- Show simulation outcomes ---
    st.subheader("🧾 Sample Simulation Outcomes")
    outcomes = pd.DataFrame({
        "pass": results['pass_mask'],
        "pass_day": results['pass_day'],
        "final_equity": results['final_equity'],
        "fail_reason": results['fail_reason']
    })
    st.dataframe(outcomes.sample(min(50, len(outcomes))).reset_index(drop=True))

    st.success(f"✅ Simulations completed in {t1 - t0:.2f} seconds.")

    with st.expander("ℹ️ Understanding Your Metrics"):
        st.write("""
        **Pass Criteria:**
        - A simulation *passes* if equity reaches `starting_equity + target_profit`  
          before hitting any rule breach (daily loss, static drawdown, trailing drawdown).  

        **Failure Conditions:**
        - **Daily Loss:** Daily P&L ≤ -daily_loss_limit → fail  
        - **Static Drawdown:** Equity < starting_equity - static_drawdown_limit → fail  
        - **Trailing Drawdown:** Equity falls > trailing_drawdown_limit from a running high → fail
        """)
else:
    st.write("⚙️ Configure inputs on the left and upload CSV or use sample_trades.csv, then press **Run Monte Carlo**.")
