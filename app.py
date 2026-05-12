import streamlit as st
import pandas as pd
import requests
import builtins
import io
import sys
import os
import time
import warnings

warnings.filterwarnings('ignore')

import en_option_v3 as opt

st.set_page_config(
    page_title="Quantitative Options Engine",
    layout="wide",
    page_icon="📈",
)

# ─── CLI capture helper ────────────────────────────────────────────────────────

def run_cli_function(func, prompt_map, *args, **kwargs):
    """
    Runs a function that uses input()/print(), capturing stdout.
    prompt_map: {substring_of_prompt_lowercase: answer_string}
    Handles rate-limit errors gracefully — surfaces them in the output
    instead of crashing the app.
    """
    old_input  = builtins.input
    old_stdout = sys.stdout

    def mocked_input(prompt=""):
        prompt_lower = prompt.lower()
        print(prompt, end="")
        for key, val in prompt_map.items():
            if key and key in prompt_lower:
                print(str(val))
                return str(val)
        print("")
        return ""

    builtins.input = mocked_input
    captured = io.StringIO()
    sys.stdout = captured

    try:
        result = func(*args, **kwargs)
        output = captured.getvalue()
    except Exception as e:
        output = captured.getvalue()
        err    = str(e)
        # Surface rate-limit errors with friendly guidance
        if "429" in err or "rate limit" in err.lower() or "too many requests" in err.lower():
            output += (
                "\n\n⚠  RATE LIMIT HIT (Yahoo Finance / yfinance)\n"
                "─────────────────────────────────────────────\n"
                "Yahoo Finance is temporarily blocking requests from this server.\n"
                "Fixes:\n"
                "  1. Wait 60–120 seconds and click Run again.\n"
                "  2. Use a smaller universe (fewer stocks) for Scanner modes.\n"
                "  3. If on Streamlit Cloud, multiple users may share the same IP.\n"
                "     Consider running locally for heavy scans.\n"
            )
        elif "401" in err or "unauthorized" in err.lower():
            output += (
                "\n\n⚠  DATA ACCESS ERROR (Yahoo Finance HTTP 401)\n"
                "──────────────────────────────────────────────\n"
                "Yahoo Finance blocked this server's IP (cloud IPs are sometimes banned).\n"
                "The engine has automatically switched to the standard yfinance fallback.\n"
                "Fixes:\n"
                "  1. Click Run again — the fallback mode is now active and should work.\n"
                "  2. If it persists, wait 30 seconds and retry.\n"
                "  3. For heavy scans, run the tool locally to avoid cloud IP blocks.\n"
            )
        else:
            output += f"\n\nERROR: {err}"
        result = None
    finally:
        builtins.input = old_input
        sys.stdout     = old_stdout

    return result, output


# ─── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Settings")
st.sidebar.markdown("**AI Synthesis** — connect an LLM API to get natural language analysis of results.")
api_key      = st.sidebar.text_input("API Key", type="password", help="Optional. Used only for AI Synthesis button.")
api_base_url = st.sidebar.text_input("API Base URL", value="https://api.megallm.io/v1")
selected_model = st.sidebar.text_input("Model", value="gpt-4o")

st.sidebar.divider()
st.sidebar.markdown(
    "**Rate limit tip:** Yahoo Finance limits ~100 req/min per IP. "
    "If you see a rate-limit error, wait 60–120 s and retry."
)

# ─── Session state ─────────────────────────────────────────────────────────────

for key in ("cli_output", "df_result"):
    if key not in st.session_state:
        st.session_state[key] = None

# ─── Header ───────────────────────────────────────────────────────────────────

st.title("📈 Quantitative Options Engine")
st.markdown(
    "Select a mode, configure parameters, and click **Run**. "
    "Results appear below. Use **AI Synthesis** to get a plain-English interpretation."
)

mode = st.selectbox("Select Mode", [
    "1. Full Analysis",
    "2. Trade Finder",
    "3. Backtest Model",
    "4. Market Scanner",
    "5. Scanner Backtest",
])

st.divider()

# ─── Mode UIs ─────────────────────────────────────────────────────────────────

if mode == "1. Full Analysis":
    st.subheader("Mode 1: Full Analysis")
    st.markdown(
        "Fetches the full option chain for all expiries, computes Greeks, "
        "GARCH vol forecast, SABR smile, pin risk, and income screeners (CSP / CC / Iron Condor)."
    )
    col1, col2 = st.columns(2)
    with col1:
        symbol     = st.text_input("Stock Symbol (e.g. AAPL)", "AAPL").strip().upper()
        run_strat  = st.checkbox("Run optionlab strategy analysis?", value=False)
    with col2:
        run_tf     = st.checkbox("Auto-run Trade Finder after?", value=False)

    if st.button("🚀 Run Full Analysis", type="primary"):
        if symbol:
            with st.spinner(f"Running Full Analysis for {symbol}… (may take 30–60 s)"):
                pm = {
                    "enter stock symbol":        symbol,
                    "run strategy analysis":     "y" if run_strat else "n",
                    "run trade finder":          "y" if run_tf else "n",
                }
                res, out = run_cli_function(opt.main, pm)
                st.session_state.cli_output = out
                st.session_state.df_result  = None
        else:
            st.warning("Please enter a stock symbol.")

elif mode == "2. Trade Finder":
    st.subheader("Mode 2: Trade Finder")
    st.markdown(
        "Ranks every call/put in your DTE window by an 8-signal institutional score "
        "including **IV Rank**, **real-world EV** (P-measure), **RSI alignment**, and **Kelly sizing**."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol       = st.text_input("Stock Symbol (e.g. TSLA)", "TSLA").strip().upper()
        action       = st.selectbox("Buy or Sell?", ["b", "s"],
                                    format_func=lambda x: "Buy" if x == "b" else "Sell")
        opt_type     = st.selectbox("Direction (auto = GEX decides)", ["auto", "c", "p"],
                                    format_func=lambda x: {"auto":"Auto (GEX)","c":"Call","p":"Put"}[x])
    with col2:
        dte_min      = st.number_input("Min DTE", value=20, min_value=1)
        dte_max      = st.number_input("Max DTE", value=60, min_value=2)
    with col3:
        budget       = st.number_input("Max premium / Min credit ($)", value=5.00, min_value=0.01, step=0.50)
        target_delta = st.text_input("Target Delta (optional, e.g. 0.30)", value="")

    if st.button("🔍 Run Trade Finder", type="primary"):
        if symbol:
            with st.spinner(f"Finding best {opt_type.upper() if opt_type != 'auto' else 'Call/Put'} trades for {symbol}…"):
                # GEX override answer: "" means accept GEX suggestion, "c"/"p" overrides
                gex_ans = "" if opt_type == "auto" else opt_type
                pm = {
                    "symbol":               symbol,
                    "buy or sell":          action,
                    # GEX prompt: "GEX suggests CALL — press Enter to accept or type [c/p]"
                    "gex suggests":         gex_ans,
                    # Fallback if GEX unavailable: "Call or Put?  [c/p]"
                    "call or put":          opt_type if opt_type != "auto" else "c",
                    "min days to expiry":   str(int(dte_min)),
                    "max days to expiry":   str(int(dte_max)),
                    "max premium":          str(budget),
                    "min credit":           str(budget),
                    "target delta":         target_delta,
                }
                res, out = run_cli_function(opt.find_trade, pm)
                st.session_state.cli_output = out
                st.session_state.df_result  = None
        else:
            st.warning("Please enter a symbol.")

elif mode == "3. Backtest Model":
    st.subheader("Mode 3: Backtest Model (single stock)")
    st.markdown(
        "Replays the option-buying model on one stock historically using BS-priced synthetic options."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol    = st.text_input("Stock Symbol (e.g. AAPL)", "AAPL").strip().upper()
        action    = st.selectbox("Buy or Sell?", ["b", "s"],
                                 format_func=lambda x: "Buy" if x == "b" else "Sell")
        direction = st.selectbox("Direction", ["a", "c", "p"],
                                 format_func=lambda x: {"a":"Auto-momentum","c":"Call","p":"Put"}[x])
        budget    = st.number_input("Max Premium / Min Credit ($)", value=5.00, min_value=0.01, step=0.50)
    with col2:
        target_dte = st.number_input("Target DTE at entry", value=30, min_value=5)
        lookback   = st.number_input("Lookback Days", value=252, min_value=60)
        exit_dte   = st.number_input("Exit at X DTE remaining (0=hold to expiry)", value=0, min_value=0)
    with col3:
        tp = st.number_input("Take profit % (0=none)", value=100, min_value=0)
        sl = st.number_input("Stop loss %  (0=none)", value=50,  min_value=0)
        be = st.number_input("Break-even trigger % (0=none)", value=0, min_value=0)

    if st.button("📊 Run Backtest", type="primary"):
        if symbol:
            with st.spinner(f"Running historical backtest for {symbol}…"):
                pm = {
                    "symbol":       symbol,
                    "buy or sell":  action,
                    "direction":    direction,
                    "target dte":   str(int(target_dte)),
                    "lookback":     str(int(lookback)),
                    "premium":      str(budget),
                    "credit":       str(budget),
                    "take profit":  str(int(tp)),
                    "break-even":   str(int(be)),
                    "stop loss":    str(int(sl)),
                    "exit at":      str(int(exit_dte)),
                }
                res, out = run_cli_function(opt.backtest_model, pm)
                st.session_state.cli_output = out
                st.session_state.df_result  = res if isinstance(res, pd.DataFrame) else None
        else:
            st.warning("Please enter a symbol.")

elif mode == "4. Market Scanner":
    st.subheader("Mode 4: Market Scanner")
    st.markdown(
        "Scans up to 600 stocks (S&P 500 + Nasdaq-100 + CBOE most-active) "
        "and ranks them by an IV-HV / GEX / GARCH composite score."
    )
    st.warning(
        "⏱ Large scans hit Yahoo Finance's rate limit quickly on cloud deployments. "
        "Keep **Max stocks** ≤ 50 on Streamlit Cloud, or use a custom watchlist."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        budget     = st.number_input("Max premium per contract ($)", value=5.00, min_value=0.01, step=0.50)
    with col2:
        max_stocks = st.number_input("Max stocks to scan", value=30, min_value=5, max_value=300)
    with col3:
        watchlist_raw = st.text_input("Custom watchlist (comma-separated, or leave blank for auto)", value="")

    watchlist = [t.strip().upper() for t in watchlist_raw.split(',') if t.strip()] if watchlist_raw else None

    if st.button("🔭 Run Scanner", type="primary"):
        with st.spinner(f"Scanning {max_stocks} stocks… this takes 2–5 minutes…"):
            res, out = run_cli_function(
                opt.market_scanner, {},
                budget=float(budget), max_stocks=int(max_stocks),
                watchlist=watchlist,
            )
            st.session_state.cli_output = out
            st.session_state.df_result  = res if isinstance(res, pd.DataFrame) else None

elif mode == "5. Scanner Backtest":
    st.subheader("Mode 5: Scanner Backtest (v2 — improved model)")
    st.markdown(
        "Replays the market scanner historically with improved position sizing, "
        "vol-cheap gate, IV percentile filter, and SPY regime filter."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        watchlist_raw = st.text_input("Watchlist (e.g. AAPL,TSLA,SOFI — blank for default 70)", value="")
        lookback      = st.number_input("Lookback Days", value=252, min_value=60)
        holding       = st.number_input("Max Holding Days per trade", value=14, min_value=3)
    with col2:
        scan_freq = st.number_input("Scan every N trading days", value=14, min_value=1)
        top_n     = st.number_input("Top N trades per scan", value=3, min_value=1, max_value=10)
        acct      = st.number_input("Starting Account ($)", value=190.0, min_value=10.0)
    with col3:
        budget_bt = st.number_input("Max cost per contract ($)", value=5.00, min_value=0.01, step=0.50)
        tp_bt     = st.number_input("Take profit %", value=50, min_value=5)
        sl_bt     = st.number_input("Stop loss %",   value=60, min_value=5)

    watchlist = [t.strip().upper() for t in watchlist_raw.split(',') if t.strip()] if watchlist_raw else None

    if st.button("⏪ Run Scanner Backtest", type="primary"):
        with st.spinner("Backtesting scanner signals historically… (~2–5 min for default universe)"):
            res, out = run_cli_function(
                opt.backtest_scanner, {},
                watchlist=watchlist,
                lookback_days=int(lookback),
                holding_days=int(holding),
                scan_freq=int(scan_freq),
                top_n=int(top_n),
                account=float(acct),
                budget=float(budget_bt),
                take_profit=tp_bt / 100,
                stop_loss=sl_bt / 100,
            )
            st.session_state.cli_output = out
            st.session_state.df_result  = res if isinstance(res, pd.DataFrame) else None


# ─── Results display ───────────────────────────────────────────────────────────

if st.session_state.cli_output:
    st.divider()

    output_text = st.session_state.cli_output

    # Detect and surface rate-limit errors prominently
    if "RATE LIMIT HIT" in output_text or "Too Many Requests" in output_text:
        st.error(
            "**Rate Limit Hit (Yahoo Finance)**\n\n"
            "Yahoo Finance is blocking requests from this server. "
            "Wait 60–120 seconds and click Run again, or reduce the number of stocks scanned."
        )

    st.subheader("Console Output")
    st.text(output_text)

if st.session_state.df_result is not None:
    st.divider()
    st.subheader("Trade Results Table")
    st.dataframe(st.session_state.df_result, use_container_width=True)

    # Download button for CSV
    csv = st.session_state.df_result.to_csv(index=False)
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv,
        file_name="options_results.csv",
        mime="text/csv",
    )


# ─── AI Synthesis ──────────────────────────────────────────────────────────────

if st.session_state.cli_output:
    st.divider()
    st.subheader("🤖 AI Synthesis")
    st.markdown(
        "Click below to send the console output to an LLM API for plain-English analysis. "
        "Requires an API key in the sidebar."
    )

    if st.button("Synthesize Output"):
        if not api_key:
            st.error("Enter your API Key in the sidebar first.")
        else:
            with st.spinner(f"Generating insights using {selected_model}…"):
                text_to_analyze = st.session_state.cli_output[-4000:]
                prompt = f"""
You are a senior quantitative options analyst. Analyze the following output from a Python options analysis engine.

```
{text_to_analyze}
```

Provide:
1. A concise summary (2–3 sentences) of what the data shows.
2. The single most actionable trade or takeaway from this output.
3. The top 2 risks associated with acting on this data.
Keep your response under 300 words. Be direct and specific — reference actual numbers from the output.
"""
                payload = {
                    "model":    selected_model,
                    "messages": [
                        {"role": "system",
                         "content": "You are a senior quantitative options analyst providing actionable insights."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.5,
                    "max_tokens":  600,
                }
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                }
                try:
                    ai_res = requests.post(
                        f"{api_base_url.rstrip('/')}/chat/completions",
                        headers=headers, json=payload, timeout=30,
                    )
                    if ai_res.status_code == 200:
                        ai_text = ai_res.json()["choices"][0]["message"]["content"]
                        st.markdown(ai_text)
                    elif ai_res.status_code == 429:
                        st.error("LLM API rate limit hit. Try again in a moment.")
                    else:
                        st.error(f"API Error {ai_res.status_code}: {ai_res.text[:200]}")
                except requests.Timeout:
                    st.error("LLM API timed out. Try again.")
                except Exception as e:
                    st.error(f"Failed to connect to LLM API: {e}")
