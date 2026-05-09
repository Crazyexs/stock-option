import streamlit as st
import pandas as pd
import requests
import builtins
import io
import sys
import os
import warnings

warnings.filterwarnings('ignore')

import en_option_v3 as opt

st.set_page_config(page_title="Quantitative Options Engine", layout="wide")

# --- Helper to capture CLI output and smart-mock input ---
def run_cli_function(func, prompt_map, *args, **kwargs):
    """
    Runs a function that normally uses input() and print(), capturing output.
    Uses prompt_map dictionary to intelligently answer input() prompts based on substrings.
    """
    old_input = builtins.input
    old_stdout = sys.stdout
    
    def mocked_input(prompt=""):
        prompt_lower = prompt.lower()
        print(prompt, end="") # echo prompt
        
        for key, val in prompt_map.items():
            if key in prompt_lower:
                print(str(val)) # echo answer
                return str(val)
                
        # Fallback if no match found
        print("") 
        return ""
            
    builtins.input = mocked_input
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    try:
        result = func(*args, **kwargs)
        output_text = captured_output.getvalue()
    except Exception as e:
        output_text = captured_output.getvalue() + f"\n\nERROR: {str(e)}"
        result = None
    finally:
        builtins.input = old_input
        sys.stdout = old_stdout
        
    return result, output_text

# --- Sidebar for Settings ---
st.sidebar.header("Settings")
st.sidebar.markdown("Configure your AI model using [MegaLLM](https://docs.megallm.io/en/releases/overview)")
api_key = st.sidebar.text_input("MegaLLM API Key", type="password")
api_base_url = "https://api.megallm.io/v1"

if "models" not in st.session_state:
    st.session_state.models = []

if st.sidebar.button("Fetch Models"):
    if api_key:
        try:
            url = f"{api_base_url.rstrip('/')}/models"
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            data = response.json()
            st.session_state.models = [m["id"] for m in data.get("data", [])]
            st.sidebar.success("Models fetched successfully!")
        except Exception as e:
            st.sidebar.error(f"Failed to fetch models: {e}")
    else:
        st.sidebar.warning("Please enter an API Key first.")

selected_model = st.sidebar.selectbox(
    "Select Model", 
    options=st.session_state.models if st.session_state.models else ["gpt-4o"]
)

# --- Main App ---
st.title("Quantitative Options Engine")
st.markdown("Configure the parameters below and execute. The system automatically navigates the underlying analytical models.")

mode = st.selectbox("Select Mode", [
    "1. Full Analysis",
    "2. Trade Finder",
    "3. Backtest Model",
    "4. Market Scanner",
    "5. Scanner Backtest"
])

st.divider()

if "cli_output" not in st.session_state:
    st.session_state.cli_output = ""
if "df_result" not in st.session_state:
    st.session_state.df_result = None

# --- Mode UIs ---

if mode == "1. Full Analysis":
    st.subheader("Mode 1: Full Analysis")
    symbol = st.text_input("Stock Symbol (e.g., AAPL)").strip().upper()
    run_strat = st.checkbox("Run strategy analysis?", value=False)
    run_tf = st.checkbox("Run Trade Finder after?", value=False)
    
    if st.button("Run Full Analysis", type="primary"):
        if symbol:
            with st.spinner(f"Running Full Analysis for {symbol}..."):
                pm = {
                    "symbol": symbol,
                    "run strategy analysis": "y" if run_strat else "n",
                    "run trade finder": "y" if run_tf else "n"
                }
                res, out = run_cli_function(opt.main, pm)
                st.session_state.cli_output = out
                st.session_state.df_result = None
        else:
            st.warning("Please enter a symbol.")

elif mode == "2. Trade Finder":
    st.subheader("Mode 2: Trade Finder")
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Stock Symbol (e.g., TSLA)").strip().upper()
        action = st.selectbox("Buy or Sell?", ["b", "s"])
        opt_type = st.selectbox("Call, Put, or Auto (GEX)?", ["auto", "c", "p"])
    with col2:
        dte_min = st.number_input("Min Days to Expiry (DTE)", value=20)
        dte_max = st.number_input("Max Days to Expiry (DTE)", value=60)
        account_size = st.number_input("Account Size ($) for Kelly Sizing", value=10000)
    with col3:
        budget = st.number_input("Max Premium Budget / Min Credit ($)", value=5.0)
        target_delta = st.text_input("Target Delta (optional, e.g. 0.30)", value="")
        
    if st.button("Run Trade Finder", type="primary"):
        if symbol:
            with st.spinner("Finding optimal trades..."):
                gex_override = "" if opt_type == "auto" else opt_type
                pm = {
                    "symbol": symbol,
                    "buy or sell": action,
                    "gex suggests": gex_override,
                    "call or put": opt_type if opt_type != "auto" else "c", # fallback
                    "min days": str(int(dte_min)),
                    "max days": str(int(dte_max)),
                    "max premium": str(budget),
                    "min credit": str(budget),
                    "target delta": target_delta,
                    "account size": str(account_size)
                }
                res, out = run_cli_function(opt.find_trade, pm)
                st.session_state.cli_output = out
                st.session_state.df_result = None
        else:
            st.warning("Please enter a symbol.")

elif mode == "3. Backtest Model":
    st.subheader("Mode 3: Backtest Model")
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Stock Symbol (e.g., AAPL)").strip().upper()
        action = st.selectbox("Buy or Sell?", ["b", "s"])
        direction = st.selectbox("Direction", ["a", "c", "p"], format_func=lambda x: {"a":"Auto-momentum", "c":"Call", "p":"Put"}[x])
        budget = st.number_input("Max Premium Budget / Min Credit ($)", value=5.0)
    with col2:
        target_dte = st.number_input("Target DTE at entry", value=30)
        lookback = st.number_input("Lookback Days", value=252)
        exit_dte = st.number_input("Exit at X DTE remaining (0=hold to expiry)", value=0)
    with col3:
        tp = st.number_input("Take profit % (0=none)", value=100)
        sl = st.number_input("Stop loss % (0=none)", value=50)
        be = st.number_input("Break-even trigger % (0=none)", value=0)
        
    if st.button("Run Backtest", type="primary"):
        if symbol:
            with st.spinner("Running historical backtest..."):
                pm = {
                    "symbol": symbol,
                    "buy or sell": action,
                    "direction": direction,
                    "target dte": str(int(target_dte)),
                    "lookback": str(int(lookback)),
                    "premium": str(budget),
                    "credit": str(budget),
                    "take profit": str(int(tp)),
                    "break-even": str(int(be)),
                    "stop loss": str(int(sl)),
                    "exit at": str(int(exit_dte))
                }
                res, out = run_cli_function(opt.backtest_model, pm)
                st.session_state.cli_output = out
                st.session_state.df_result = res if isinstance(res, pd.DataFrame) else None
        else:
            st.warning("Please enter a symbol.")

elif mode == "4. Market Scanner":
    st.subheader("Mode 4: Market Scanner")
    col1, col2 = st.columns(2)
    with col1:
        budget = st.number_input("Max premium per contract ($)", value=5.00)
    with col2:
        max_stocks = st.number_input("Max stocks to scan", value=50)
        
    if st.button("Run Scanner", type="primary"):
        with st.spinner("Scanning the market... this might take a few minutes!"):
            res, out = run_cli_function(opt.market_scanner, {}, budget=budget, max_stocks=max_stocks)
            st.session_state.cli_output = out
            st.session_state.df_result = res if isinstance(res, pd.DataFrame) else None

elif mode == "5. Scanner Backtest":
    st.subheader("Mode 5: Scanner Backtest")
    st.info("Uses a preset universe (e.g., Tech Giants) or standard list if none provided.")
    col1, col2 = st.columns(2)
    with col1:
        lookback = st.number_input("Lookback Days", value=252)
    with col2:
        holding = st.number_input("Holding Days", value=14)
        
    if st.button("Run Scanner Backtest", type="primary"):
        with st.spinner("Backtesting scanner... this might take a while!"):
            res, out = run_cli_function(opt.backtest_scanner, {}, lookback_days=lookback, holding_days=holding)
            st.session_state.cli_output = out
            st.session_state.df_result = res if isinstance(res, pd.DataFrame) else None


# --- Display Results ---
if st.session_state.cli_output:
    st.divider()
    st.subheader("Console Output")
    st.text(st.session_state.cli_output)
    
if st.session_state.df_result is not None:
    st.subheader("Data Result")
    st.dataframe(st.session_state.df_result)

# --- Optional Synthesis ---
if st.session_state.cli_output:
    st.divider()
    st.subheader("Algorithmic Synthesis & Strategy Recommendation")
    st.markdown("Execute natural language synthesis of quantitative outputs.")
    
    if st.button("Synthesize Output"):
        if not api_key:
            st.error("Please enter your API Key in the sidebar first.")
        else:
            with st.spinner(f"Generating insights using {selected_model}..."):
                text_to_analyze = st.session_state.cli_output[-4000:]
                
                prompt = f"""
As a senior quantitative options analyst, please analyze the following CLI output from our Python stock options engine.
The user just ran a quantitative options analysis tool. Here is the raw output (truncated if too long):

```
{text_to_analyze}
```

Please provide:
1. A brief summary of what the data is telling us.
2. The most actionable trade or takeaway from this output.
3. Key risks associated with these findings.
"""
                payload = {
                    "model": selected_model,
                    "messages": [
                        {"role": "system", "content": "You are a senior quantitative options analyst providing actionable insights based on options data output."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1500
                }
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                url = f"{api_base_url.rstrip('/')}/chat/completions"
                try:
                    ai_res = requests.post(url, headers=headers, json=payload)
                    if ai_res.status_code == 200:
                        ai_text = ai_res.json()["choices"][0]["message"]["content"]
                        st.markdown(ai_text)
                    else:
                        st.error(f"API Error ({ai_res.status_code}): {ai_res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to LLM API: {e}")
