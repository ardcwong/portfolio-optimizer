# app_v2.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from datetime import timedelta
import math
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Smart Framework v2", layout="wide")
sns.set_style("whitegrid")

# -------------------------
# Constants
# -------------------------
TRADING_DAYS = 252

# -------------------------
# Utility functions
# -------------------------
@st.cache_data
def fetch_index_series(ticker, start, end):
    """Use .history() for index tickers to avoid yfinance quirks. Fallback to ticker 'SPY' if empty."""
    try:
        t = yf.Ticker(ticker)
        s = t.history(start=start, end=end)['Close'].dropna()
        s.index = s.index.tz_localize(None)
        if s.empty and ticker.upper() == "^GSPC":
            # fallback to SPY
            t2 = yf.Ticker("SPY")
            s = t2.history(start=start, end=end)['Close'].dropna()
        return s
    except Exception:
        # final fallback: try yf.download
        data = yf.download(ticker, start=start, end=end, progress=False)
        if 'Close' in data:
            return data['Close'].dropna()
        return pd.Series(dtype=float)

@st.cache_data
def fetch_prices(tickers, start, end, auto_adjust=True):
    """Fetch adjusted close prices for multiple tickers."""
    if isinstance(tickers, str):
        tickers = [tickers]
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=auto_adjust)
    # Convert tz-aware index to tz-naive (remove timezone)
    df.index = df.index.tz_localize(None)

    if isinstance(df, pd.DataFrame) and 'Close' in df.columns:
        df = df['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame()
    # forward-fill then drop leading NaNs per column
    df = df.ffill().dropna(how='all')
    return df

def compute_log_returns(prices):
    returns = np.log(prices / prices.shift(1))
    return returns.dropna(how='all')

# -------------------------
# Step 0: GMM (2D features μ, σ)
# -------------------------
def run_step_0_gmm(index_ticker, gmm_start, gmm_end, vol_window=21, k=3):
    spx = fetch_index_series(index_ticker, gmm_start, gmm_end)
    if spx.empty:
        st.error(f"Could not fetch index {index_ticker}. Try 'SPY'.")
        return None

    spx_ret = compute_log_returns(spx)
    # rolling features (use simple rolling on daily log returns)
    mu_roll = spx_ret.rolling(vol_window).mean()
    sigma_roll = spx_ret.rolling(vol_window).std() * math.sqrt(TRADING_DAYS)
    features = pd.concat([mu_roll, sigma_roll], axis=1).dropna()
    features.columns = ['mu', 'sigma']

    # scale features, fit GMM
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    gmm.fit(X)
    comp_idx = gmm.predict(X)
    probs = gmm.predict_proba(X)  # rows aligned to features.index

    features = features.assign(component=comp_idx)
    # compute component mean (in original mu space) for labeling
    comp_means = pd.DataFrame(gmm.means_, columns=['mu_scaled', 'sigma_scaled'])
    # To label by mu, we need mapping based on decoded means in original scale:
    # inverse transform means
    comp_means_orig = pd.DataFrame(scaler.inverse_transform(gmm.means_), columns=['mu', 'sigma'])
    order = np.argsort(comp_means_orig['mu'].values)  # ascending mu
    # map indices to names
    label_map = {}
    labels_sorted = ["Bear", "Calm", "Bull"]
    for i, comp in enumerate(order):
        label_map[comp] = labels_sorted[i] if i < len(labels_sorted) else f"Regime_{i+1}"

    # historical regime names (aligned to features.index)
    hist_labels = pd.Series(comp_idx, index=features.index).map(label_map)

    # predict current features (most recent rolling mu & sigma)
    last_feat = features.iloc[[-1]][['mu', 'sigma']]
    last_scaled = scaler.transform(last_feat)
    cur_comp = int(gmm.predict(last_scaled)[0])
    cur_regime = label_map.get(cur_comp, "Unknown")
    # get probabilities for the last row and map to regime names
    last_probs = pd.Series(probs[-1], index=[label_map[i] for i in range(len(probs[-1]))])
    # ensure ordering Bear/Calm/Bull
    probs_map = {name: float(last_probs.get(name, 0.0)) for name in ["Bear", "Calm", "Bull"]}

    # return: current regime name, probs, features df, historical regime names, label_map, gmm/scaler for debug if needed
    return cur_regime, probs_map, features, hist_labels, label_map, gmm, scaler

# -------------------------
# Step 1: get stock inputs (regime-filtered mu and Sigma)
# -------------------------
def run_step_1_get_inputs(tickers, start_date, end_date, use_gmm=True, current_regime=None, hist_labels=None):
    prices = fetch_prices(tickers, start_date, end_date, auto_adjust=True)
    if prices.shape[1] == 0:
        st.error("No valid price columns returned for tickers.")
        return None
    returns = compute_log_returns(prices)
    if returns.empty:
        st.error("No returns available for selected tickers.")
        return None

    if use_gmm and current_regime is not None and hist_labels is not None:
        # align labels to returns index (no dropna)
        aligned_labels = hist_labels.reindex(returns.index, method='ffill')
        mask = aligned_labels.eq(current_regime).fillna(False)
        filtered = returns.loc[mask]
        used_fallback = False
        if filtered.shape[0] < 30:
            # fallback
            filtered = returns.copy()
            used_fallback = True
    else:
        filtered = returns.copy()
        used_fallback = False

    mu_daily = filtered.mean()
    mu_annual = mu_daily * TRADING_DAYS
    Sigma_noisy_daily = filtered.cov()
    Sigma_noisy = Sigma_noisy_daily * TRADING_DAYS  # annualize

    sigma_vec = pd.Series(np.sqrt(np.diag(Sigma_noisy)), index=Sigma_noisy.index)
    sigma_vec = sigma_vec.replace(0, 1e-8)

    return mu_annual, Sigma_noisy, sigma_vec, filtered, prices, used_fallback

# -------------------------
# Step 2: PCA denoising (hybrid RMT / 90% var)
# -------------------------
def run_step_2_pca_denoise(Sigma_noisy, sigma_vector, filtered_returns, var_threshold=0.90):
    T = filtered_returns.shape[0]
    N = filtered_returns.shape[1]
    if T <= 1 or N == 0:
        return Sigma_noisy, 0, 0.0

    # build correlation-like matrix P
    D_inv = np.diag(1 / sigma_vector)
    P = D_inv @ Sigma_noisy.values @ D_inv
    # eigen
    vals, vecs = np.linalg.eigh(P)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    # RMT threshold (Marchenko-Pastur) using sigma2 = 1 here since P standardized
    q = N / T
    lambda_plus = (1 + math.sqrt(q))**2 if q > 0 else 0.0
    k_rmt = int(np.sum(vals > lambda_plus))
    if k_rmt == 0:
        k_rmt = 1
    # 90% var
    cumvar = np.cumsum(vals) / np.sum(vals)
    k_var = int(np.searchsorted(cumvar, var_threshold) + 1)
    k_keep = max(k_rmt, k_var)
    # reconstruct
    vals_top = vals[:k_keep]
    vecs_top = vecs[:, :k_keep]
    P_cleaned = vecs_top @ np.diag(vals_top) @ vecs_top.T
    diag_pc = np.diag(P_cleaned).copy()
    diag_pc[diag_pc <= 0] = 1e-8
    Dp_inv = np.diag(1 / np.sqrt(diag_pc))
    P_cleaned = Dp_inv @ P_cleaned @ Dp_inv

    # unscale
    D = np.diag(sigma_vector)
    Sigma_cleaned = D @ P_cleaned @ D
    Sigma_cleaned = pd.DataFrame(Sigma_cleaned, index=Sigma_noisy.index, columns=Sigma_noisy.columns)
    return Sigma_cleaned, k_keep, lambda_plus

# -------------------------
# Step 3: optimizer
# -------------------------
def run_step_3_optimize(mu, Sigma, lambd):
    mu_arr = mu.values
    Sigma_arr = Sigma.values
    n = len(mu_arr)
    def obj(w):
        return - (w @ mu_arr - 0.5 * lambd * (w @ (Sigma_arr @ w)))
    cons = ({'type':'eq', 'fun': lambda w: np.sum(w) - 1},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    x0 = np.repeat(1.0 / n, n)
    res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol':1e-9, 'maxiter':1000})
    if not res.success:
        w = np.clip(res.x, 0, None)
        if w.sum() <= 0:
            w = np.repeat(1.0/n, n)
        else:
            w = w / w.sum()
    else:
        w = res.x
    w_series = pd.Series(w, index=mu.index)
    port_mu = float(w_series @ mu)
    port_var = float(w_series @ (Sigma @ w_series))
    port_vol = math.sqrt(max(port_var, 0.0))
    return w_series, port_mu, port_vol

# -------------------------
# Portfolio helpers
# -------------------------
def compute_portfolio_daily_returns(weights, daily_returns):
    # align weights to daily_returns columns
    w = weights.reindex(daily_returns.columns).fillna(0.0)
    port = daily_returns.dot(w)
    return port

def compute_cumulative_from_daily(daily_ret):
    # daily_ret is log returns; convert to cumulative simple returns series
    # sum logs to get log cumulative, then expm1
    cum = np.exp(daily_ret.cumsum()) - 1.0
    return cum

# -------------------------
# UI & Session state init
# -------------------------
st.title("Smart Framework — GMM (μ,σ) + PCA Denoised Covariance (Full Simulation)")

with st.sidebar:
    st.header("Controls")
    index_ticker = st.text_input("Index ticker (GMM)", value="^GSPC")
    default_date = pd.to_datetime("2024-01-01")
    current_date_input = st.date_input("Current analysis date", value=default_date.date())
    risk_appetite = st.slider("Risk appetite (1 Aggressive → 10 Conservative)", 1, 10, 5)
    budget_input = st.number_input("Budget", min_value=100.0, value=100000.0, step=100.0, format="%.2f")
    # You can customize this list with any universe you want
    default_universe = [
        "AAPL","MSFT","AMZN","GOOG","META","TSLA","NVDA","JPM","V","XOM",
        "NFLX","ADBE","CRM","ORCL","INTC","AMD","BAC","WMT","DIS","CSCO"
    ]
    
    tickers = st.multiselect(
        "Select up to 10 stocks",
        options=default_universe,
        default=["AAPL", "MSFT", "AMZN"],
        max_selections=10
    )
    
    # Safety: If user selects nothing, block workflow
    if len(tickers) == 0:
        st.warning("Please select at least 1 stock.")
    
    gmm_lookback_years = st.number_input("GMM lookback (years)", min_value=3, max_value=20, value=10)
    stocks_lookback_years = st.number_input("Stock history lookback (years)", min_value=1, max_value=10, value=5)
    rolling_window_days = st.number_input("Rolling window (days) for μ,σ", min_value=20, max_value=252, value=60)
    advance_period = st.selectbox("Advance time by", ["1 month", "1 quarter (3 months)"])
    run_button = st.button("Run Smart Framework (Rebalance now)")

# initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []  # list of dicts recording each rebalance
if 'current_date' not in st.session_state:
    st.session_state.current_date = pd.to_datetime(current_date_input)
if 'budget' not in st.session_state:
    st.session_state.budget = float(budget_input)
if 'tickers' not in st.session_state:
    st.session_state.tickers = tickers

# update session tickers if changed
if tickers != st.session_state.tickers:
    st.session_state.tickers = tickers

# -------------------------
# Run pipeline (initial rebalance)
# -------------------------
def run_pipeline_and_record(analysis_date, tickers, budget, risk_appetite):
    gmm_start = (pd.to_datetime(analysis_date) - pd.DateOffset(years=int(gmm_lookback_years))).strftime("%Y-%m-%d")
    gmm_end = pd.to_datetime(analysis_date).strftime("%Y-%m-%d")
    cur_regime, probs_map, features, hist_labels, label_map, gmm_model, scaler = run_step_0_gmm(index_ticker, gmm_start, gmm_end, vol_window=int(rolling_window_days), k=3)
    if cur_regime is None:
        st.error("GMM failed. Aborting pipeline.")
        return None

    # stock dates
    stocks_start = (pd.to_datetime(analysis_date) - pd.DateOffset(years=int(stocks_lookback_years))).strftime("%Y-%m-%d")
    stocks_end = pd.to_datetime(analysis_date).strftime("%Y-%m-%d")
    mu, Sigma_noisy, sigma_vec, filtered_returns, price_df, used_fallback = run_step_1_get_inputs(st_session_tickers := st.session_state.tickers, stocks_start, stocks_end, use_gmm=True, current_regime=cur_regime, hist_labels=hist_labels)
    if mu is None:
        st.error("Failed to compute stock inputs.")
        return None

    Sigma_cleaned, k_keep, lambda_plus = run_step_2_pca_denoise(Sigma_noisy, sigma_vec, filtered_returns, var_threshold=0.90)

    # optimize
    w_star, forecast_mu_port, forecast_vol_port = run_step_3_optimize(mu, Sigma_cleaned, lambd=float(risk_appetite))
    # naive
    assets = list(mu.index)
    w_naive = pd.Series(np.repeat(1.0/len(assets), len(assets)), index=assets)

    # allocations
    allocation = (w_star * budget).round(2)
    allocation_naive = (w_naive * budget).round(2)

    # prepare record
    record = {
        'date': pd.to_datetime(analysis_date),
        'predicted_regime': cur_regime,
        'regime_probs': probs_map,
        'used_fallback': used_fallback,
        'assets': assets,
        'weights_smart': w_star,
        'weights_naive': w_naive,
        'alloc_smart': allocation,
        'alloc_naive': allocation_naive,
        'mu_annual': mu,
        'Sigma_annual': Sigma_cleaned,
        'k_keep': k_keep,
        'lambda_plus': lambda_plus,
        'forecast_port_mu': forecast_mu_port,
        'forecast_port_vol': forecast_vol_port,
        'price_df': price_df,
        'filtered_returns': filtered_returns,
        # realized placeholders
        'realized_return_smart': None,
        'realized_vol_smart': None,
        'realized_return_naive': None,
        'realized_vol_naive': None,
        'sharpe_smart': None,
        'sharpe_naive': None
    }
    st.session_state.history.append(record)
    return record

# Bind run_button
if run_button:
    # reset history on fresh run
    st.session_state.history = []
    st.session_state.current_date = pd.to_datetime(current_date_input)
    st.session_state.budget = float(budget_input)
    st.success("Executing Smart Framework...")
    rec = run_pipeline_and_record(st.session_state.current_date, st.session_state.tickers, st.session_state.budget, risk_appetite)
    if rec:
        st.success("Rebalance completed and recorded.")

# -------------------------
# Advance time action
# -------------------------
col1, col2 = st.columns([1,1])
with col1:
    if st.button("Advance time"):
        if len(st.session_state.history) == 0:
            st.warning("No rebalance history to realize. Run Smart Framework first.")
        else:
            # compute new date
            delta = pd.DateOffset(months=1) if advance_period == "1 month" else pd.DateOffset(months=3)
            prev = st.session_state.current_date
            new_dt = prev + delta
            st.session_state.current_date = new_dt
            st.info(f"Advanced date to {new_dt.date()}")

            # compute realized for last record
            last = st.session_state.history[-1]
            price_df = last['price_df']
            reb_date = last['date']
            # realized window: >reb_date and <= new_dt
            mask = (price_df.index > reb_date) & (price_df.index <= new_dt)
            window_prices = price_df.loc[mask]
            if window_prices.shape[0] == 0:
                st.warning("No price data in advanced window; cannot compute realized returns.")
            else:
                returns_all = compute_log_returns(price_df)
                realized_window = returns_all.loc[(returns_all.index > reb_date) & (returns_all.index <= new_dt)]
                if realized_window.shape[0] == 0:
                    st.warning("No return rows in realized window.")
                else:
                    w_smart = last['weights_smart'].reindex(realized_window.columns).fillna(0.0)
                    w_naive = last['weights_naive'].reindex(realized_window.columns).fillna(0.0)
                    smart_daily = realized_window.dot(w_smart)
                    naive_daily = realized_window.dot(w_naive)
                    # sum log returns -> log cumulative
                    realized_log_smart = float(smart_daily.sum())
                    realized_return_smart = math.expm1(realized_log_smart)
                    realized_vol_smart = float(smart_daily.std() * math.sqrt(TRADING_DAYS))
                    realized_log_naive = float(naive_daily.sum())
                    realized_return_naive = math.expm1(realized_log_naive)
                    realized_vol_naive = float(naive_daily.std() * math.sqrt(TRADING_DAYS))
                    sharpe_smart = realized_return_smart / (realized_vol_smart + 1e-12) if realized_vol_smart > 0 else np.nan
                    sharpe_naive = realized_return_naive / (realized_vol_naive + 1e-12) if realized_vol_naive > 0 else np.nan

                    # update record
                    last['realized_return_smart'] = realized_return_smart
                    last['realized_vol_smart'] = realized_vol_smart
                    last['realized_return_naive'] = realized_return_naive
                    last['realized_vol_naive'] = realized_vol_naive
                    last['sharpe_smart'] = sharpe_smart
                    last['sharpe_naive'] = sharpe_naive

                    # update budget
                    prev_budget = st.session_state.budget
                    new_budget = prev_budget * (1.0 + realized_return_smart)
                    st.session_state.budget = float(max(new_budget, 0.0))
                    st.success(f"Realized Smart return: {realized_return_smart:.2%}. New budget = {st.session_state.budget:,.2f}")

with col2:
    if st.button("Rebalance now (use current budget)"):
        # allow rebalancing using new budget & (optionally) new tickers in sidebar
        rec = run_pipeline_and_record(st.session_state.current_date, st.session_state.tickers, st.session_state.budget, risk_appetite)
        if rec:
            st.success("Rebalance computed and appended to history.")

# -------------------------
# Visualizations & Tables
# -------------------------
st.markdown("## Rebalance History")
if len(st.session_state.history) == 0:
    st.info("No rebalances yet. Click 'Run Smart Framework' to start.")
else:
    # summary
    rows = []
    for i, e in enumerate(st.session_state.history):
        rows.append({
            'idx': i+1,
            'date': e['date'].date(),
            'predicted_regime': e['predicted_regime'],
            'used_fallback': e['used_fallback'],
            'k_keep': e['k_keep'],
            'forecast_return': e['forecast_port_mu'],
            'forecast_vol': e['forecast_port_vol'],
            'realized_return_smart': e.get('realized_return_smart'),
            'realized_vol_smart': e.get('realized_vol_smart'),
            'realized_return_naive': e.get('realized_return_naive'),
            'realized_vol_naive': e.get('realized_vol_naive'),
            'sharpe_smart': e.get('sharpe_smart'),
            'sharpe_naive': e.get('sharpe_naive'),
        })
    summary_df = pd.DataFrame(rows).set_index('idx')
    st.dataframe(summary_df.style.format({
        'forecast_return': "{:.2%}",
        'forecast_vol': "{:.2%}",
        'realized_return_smart': lambda x: f"{x:.2%}" if pd.notnull(x) else "",
        'realized_vol_smart': lambda x: f"{x:.2%}" if pd.notnull(x) else "",
        'realized_return_naive': lambda x: f"{x:.2%}" if pd.notnull(x) else "",
        'realized_vol_naive': lambda x: f"{x:.2%}" if pd.notnull(x) else "",
        'sharpe_smart': "{:.3f}",
        'sharpe_naive': "{:.3f}"
    }))

    # display last regime probs
    last = st.session_state.history[-1]
    st.markdown("### Last predicted regime & posterior probabilities")
    probs = last['regime_probs']
    probs_df = pd.DataFrame.from_dict(probs, orient='index', columns=['prob'])
    probs_df = probs_df.reindex(["Bear","Calm","Bull"]).fillna(0.0)
    st.table(probs_df.style.format({'prob': "{:.2%}"}))
    st.bar_chart(probs_df['prob'])

    # performance chart: combine realized segments
    st.markdown("### Portfolio performance per realized segment (Smart vs 1/N)")
    perf_df = pd.DataFrame()
    for i, e in enumerate(st.session_state.history):
        prices = e['price_df']
        if prices is None or prices.shape[0] == 0:
            continue
        start_dt = e['date']
        if i+1 < len(st.session_state.history):
            end_dt = st.session_state.history[i+1]['date']
        else:
            end_dt = prices.index[-1]
        returns_all = compute_log_returns(prices)
        interval = returns_all.loc[(returns_all.index > start_dt) & (returns_all.index <= end_dt)]
        if interval.shape[0] == 0:
            continue
        w_smart = e['weights_smart'].reindex(interval.columns).fillna(0.0)
        w_naive = e['weights_naive'].reindex(interval.columns).fillna(0.0)
        smart_daily = interval.dot(w_smart)
        naive_daily = interval.dot(w_naive)
        smart_cum = compute_cumulative_from_daily(smart_daily)
        naive_cum = compute_cumulative_from_daily(naive_daily)
        perf_df = pd.concat([perf_df, pd.DataFrame({
            f"Smart_{i+1}": smart_cum,
            f"Naive_{i+1}": naive_cum
        })], axis=1)
    if perf_df.shape[1] > 0:
        st.line_chart(perf_df.fillna(method='ffill').fillna(0))
    else:
        st.info("No realized periods yet to plot (advance time after a rebalance).")

    # Pearson correlations
    st.markdown("### Pearson correlation: predicted μ vs actual asset realized returns")
    pearson_rows = []
    for i, e in enumerate(st.session_state.history):
        if e.get('realized_return_smart') is None:
            continue
        prices = e['price_df']
        reb_date = e['date']
        returns_all = compute_log_returns(prices)
        realized_window = returns_all.loc[(returns_all.index > reb_date)]
        if realized_window.shape[0] == 0:
            continue
        actual_asset_log = realized_window.sum()
        actual_asset_simple = np.expm1(actual_asset_log)
        mu_series = e['mu_annual']
        common = actual_asset_simple.index.intersection(mu_series.index)
        if len(common) < 2:
            continue
        try:
            from scipy.stats import pearsonr
            r, p = pearsonr(mu_series.reindex(common).values, actual_asset_simple.reindex(common).values)
        except Exception:
            r = np.nan
        pearson_rows.append({'rebalance': i+1, 'date': e['date'], 'pearson_mu_vs_actual': r})
    if len(pearson_rows) > 0:
        pearson_df = pd.DataFrame(pearson_rows).set_index('rebalance')
        st.dataframe(pearson_df.style.format({'pearson_mu_vs_actual': '{:.3f}'}))
    else:
        st.info("No realized data yet for Pearson correlations.")

    # last allocation table
    st.markdown("### Last allocation (Smart vs 1/N)")
    last = st.session_state.history[-1]
    alloc_df = pd.DataFrame({
        'Smart_weight': last['weights_smart'],
        'Smart_alloc': last['alloc_smart'],
        'Naive_weight': last['weights_naive'],
        'Naive_alloc': last['alloc_naive']
    }).fillna(0.0)
    st.dataframe(alloc_df.style.format({
        'Smart_weight': '{:.4f}',
        'Smart_alloc': '{:,.2f}',
        'Naive_weight': '{:.4f}',
        'Naive_alloc': '{:,.2f}'
    }))

# -------------------------
# Footer notes
# -------------------------
st.markdown("---")
st.markdown("**Notes:** GMM uses rolling μ and σ (daily returns). If filtered regime data < 30 observations, we fallback to use all data. PCA denoising uses hybrid RMT / 90% variance rule. Optimization maximizes utility with λ=risk_appetite and constraints sum(w)=1, w>=0.")






# import streamlit as st
# import numpy as np
# import pandas as pd
# import yfinance as yf
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.mixture import GaussianMixture
# from sklearn.preprocessing import StandardScaler
# from scipy.optimize import minimize

# # --- Constants ---
# TRADING_DAYS = 252

# # --- Step 0: GMM Regime Detection ---
# def run_step_0_gmm(index_ticker, start_date, end_date, vol_window, n_components):
#     prices = yf.download(index_ticker, start=start_date, end=end_date)['Close'].dropna()
#     log_returns = np.log(prices / prices.shift(1))
#     rolling_vol = log_returns.rolling(window=vol_window).std() * np.sqrt(TRADING_DAYS)

#     features = pd.concat([log_returns, rolling_vol], axis=1).dropna()
#     features.columns = ['log_return', 'volatility']

#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(features)

#     gmm = GaussianMixture(n_components=n_components, covariance_type = 'full', random_state=42)
#     gmm.fit(scaled_features)

#     features['regime'] = gmm.predict(scaled_features)
#     probs = gmm.predict_proba(scaled_features)

#     regime_stats_return = features.groupby('regime')['log_return'].mean() * TRADING_DAYS
#     bear_label = regime_stats_return.idxmin()
#     bull_label = regime_stats_return.idxmax()
#     calm_label = [x for x in regime_stats_return.index if x not in [bear_label, bull_label]][0]

#     regime_map = {bear_label: "Bear", bull_label: "Bull", calm_label: "Calm"}
#     historical_regime_labels = features['regime'].map(regime_map)
#     features['Regime Name'] = features['regime'].map(regime_map)

#     current_features = features[['log_return', 'volatility']].iloc[[-1]] 
#     scaled_current_features = scaler.transform(current_features)
    
#     current_regime_label = gmm.predict(scaled_current_features)[0]
#     current_regime_name = regime_map.get(current_regime_label, "Unknown")
#     current_regime_probs = dict(zip(["Bear", "Calm", "Bull"], probs[-1]))

#     return current_regime_name, current_regime_probs, features, historical_regime_labels, regime_map

# # --- Step 1: Stock Data and Inputs ---
# def run_step_1_get_inputs(stock_tickers, start_date, end_date, use_gmm=True, current_regime=None, hist_labels=None):
#     stock_prices = yf.download(stock_tickers, start=start_date, end=end_date, auto_adjust=True)['Close'].dropna()
#     log_returns = np.log(stock_prices / stock_prices.shift(1)).dropna()

#     if use_gmm and current_regime is not None and hist_labels is not None:
#         # aligned_labels = hist_labels.reindex(log_returns.index, method='ffill').dropna()
#         # filtered_returns = log_returns[aligned_labels == current_regime]

#         aligned_labels = hist_labels.reindex(log_returns.index, method='ffill')

#         mask = aligned_labels.eq(current_regime)
#         mask = mask.fillna(False)

#         filtered_returns = log_returns.loc[mask]



        
#         if len(filtered_returns) < 30:
#             filtered_returns = log_returns
#     else:
#         filtered_returns = log_returns

#     mu = filtered_returns.mean() * TRADING_DAYS
#     Sigma_noisy = filtered_returns.cov() * TRADING_DAYS
#     sigma_vector = pd.Series(np.sqrt(np.diag(Sigma_noisy)), index=Sigma_noisy.index)
#     sigma_vector[sigma_vector == 0] = 1e-8

#     return mu, Sigma_noisy, sigma_vector, filtered_returns

# # --- Step 2: PCA Denoising ---
# def run_step_2_pca_denoise(Sigma_noisy, sigma_vector, filtered_returns):
#     T, N = filtered_returns.shape
#     if T < N:
#         return Sigma_noisy

#     inv_sigma = np.diag(1 / sigma_vector)
#     P = inv_sigma @ Sigma_noisy @ inv_sigma
#     eigenvalues, eigenvectors = np.linalg.eigh(P)
#     idx = np.argsort(eigenvalues)[::-1]
#     eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

#     c = N / T
#     lambda_max = (1 + np.sqrt(c))**2
#     k_rmt = np.sum(eigenvalues > lambda_max)
#     k_var = np.searchsorted(np.cumsum(eigenvalues / np.sum(eigenvalues)), 0.9) + 1
#     k = max(k_rmt, k_var)

#     P_cleaned = eigenvectors[:, :k] @ np.diag(eigenvalues[:k]) @ eigenvectors[:, :k].T
#     diag_P_cleaned = np.diag(P_cleaned)
#     D_inv = np.diag(1 / np.sqrt(np.maximum(diag_P_cleaned, 1e-8)))
#     P_cleaned = D_inv @ P_cleaned @ D_inv

#     Sigma_cleaned = np.diag(sigma_vector) @ P_cleaned @ np.diag(sigma_vector)
#     return pd.DataFrame(Sigma_cleaned, index=Sigma_noisy.index, columns=Sigma_noisy.columns)

# # --- Step 3: Optimization ---
# def run_step_3_optimize(mu, Sigma_final, lambda_value):
#     n = len(mu)
#     R_f = 0.02

#     def objective(w):
#         port_ret = np.dot(w, mu)
#         port_var = np.dot(w.T, np.dot(Sigma_final, w))
#         return -(port_ret - 0.5 * lambda_value * port_var)

#     constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
#     bounds = tuple((0, 1) for _ in range(n))
#     w0 = np.ones(n) / n
#     res = minimize(objective, w0, bounds=bounds, constraints=constraints)

#     if not res.success:
#         return pd.Series(w0, index=mu.index), 0, 0, 0

#     w = pd.Series(res.x / np.sum(res.x), index=mu.index)
#     port_ret = np.dot(w, mu)
#     port_vol = np.sqrt(np.dot(w.T, np.dot(Sigma_final, w)))
#     sharpe = (port_ret - R_f) / port_vol if port_vol != 0 else 0

#     return w, port_ret, port_vol, sharpe

# # --- Step 4: Allocation ---
# def run_step_4_allocate(w, budget):
#     allocation = w * budget
#     return pd.DataFrame({
#         'Ticker': w.index,
#         'Weight (%)': w.values * 100,
#         'Allocation ($)': allocation.values
#     })

# # --- Streamlit UI ---
# st.set_page_config(page_title="Smart Framework Optimizer", layout="wide")
# st.title("📊 Smart Framework for Risk & Budget Allocation")

# # Sidebar Inputs
# st.sidebar.header("Simulation Controls")
# index_ticker = st.sidebar.text_input("Market Index (for GMM)", "^GSPC")
# start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2013-01-01"))
# end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
# risk_appetite = st.sidebar.slider("Risk Appetite (1=Aggressive, 10=Conservative)", 1, 10, 5)
# budget = st.sidebar.number_input("Budget ($)", 1000, 1000000, 10000, 1000)
# selected_stocks = st.sidebar.multiselect("Select up to 10 Stocks", 
#                                          ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NVDA", "JPM", "V", "XOM"], 
#                                          default=["AAPL", "MSFT", "GOOG"])

# if st.sidebar.button("🚀 Run Smart Framework"):
#     with st.spinner("Running GMM regime detection..."):
#         current_regime, regime_probs, gmm_features, historical_regime_labels, regime_map = run_step_0_gmm(index_ticker, start_date, end_date, 21, 3)
#         st.success(f"Predicted Current Regime: **{current_regime}**")
#         st.write("Regime Probabilities:", regime_probs)

#     with st.spinner("Fetching stock data and computing inputs..."):
#         mu, Sigma_noisy, sigma_vector, filtered_returns = run_step_1_get_inputs(
#             selected_stocks, start_date, end_date, use_gmm=True, current_regime=current_regime, hist_labels=historical_regime_labels
#         )

#     with st.spinner("Applying PCA denoising..."):
#         Sigma_cleaned = run_step_2_pca_denoise(Sigma_noisy, sigma_vector, filtered_returns)

#     lambda_value = risk_appetite
#     with st.spinner("Optimizing portfolio..."):
#         w_star, exp_ret, exp_vol, sharpe = run_step_3_optimize(mu, Sigma_cleaned, lambda_value)

#     with st.spinner("Allocating budget..."):
#         final_alloc = run_step_4_allocate(w_star, budget)
#         st.subheader("💰 Optimal Portfolio Allocation")
#         st.dataframe(final_alloc.style.format({'Weight (%)': "{:.2f}", 'Allocation ($)': "${:,.2f}"}))

#         st.metric("Expected Annual Return", f"{exp_ret*100:.2f}%")
#         st.metric("Expected Annual Volatility", f"{exp_vol*100:.2f}%")
#         st.metric("Sharpe Ratio", f"{sharpe:.2f}")

#     st.subheader("📈 GMM Clusters (Return vs Volatility)")
#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.scatterplot(data=gmm_features, x='log_return', y='volatility', hue='Regime Name',
#                     palette={'Bear': 'red', 'Calm': 'blue', 'Bull': 'green'}, alpha=0.4, s=15, ax=ax)
#     ax.set_title("GMM Clusters: Market Regimes")
#     st.pyplot(fig)




## smart_framework_app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# from sklearn.mixture import GaussianMixture
# from sklearn.decomposition import PCA
# from scipy.optimize import minimize
# from scipy.stats import pearsonr
# from datetime import datetime
# import math

# st.set_page_config(page_title="Smart Framework — GMM(μ,σ) + PCA Denoised Covariance", layout="wide")

# # ------------------------
# # Utilities & caching
# # ------------------------
# @st.cache_data(show_spinner=False)
# def download_prices(tickers, start, end):
#     if isinstance(tickers, (list, tuple)):
#         tickers_list = tickers
#     else:
#         tickers_list = [tickers]
#     df = yf.download(tickers_list, start=start, end=end, progress=False, threads=True)
#     # prefer 'Adj Close' if available
#     if isinstance(df, pd.DataFrame) and 'Adj Close' in df.columns:
#         df = df['Adj Close']
#     # if single ticker, ensure DataFrame
#     if isinstance(df, pd.Series):
#         df = df.to_frame()
#     # normalize column names (remove spaces)
#     df.columns = [c.replace("^", "").strip() for c in df.columns]
#     return df

# def compute_log_returns(price_df):
#     return np.log(price_df / price_df.shift(1)).dropna(how='all')

# def annualize_rets(mu_daily, days=252):
#     return mu_daily * days

# def annualize_cov(cov_daily, days=252):
#     return cov_daily * days

# def marchenko_pastur_lambda_plus(T, N, sigma2=1.0):
#     # Marchenko-Pastur upper bound
#     q = N / T
#     if q <= 0:
#         return 0.0
#     return sigma2 * (1 + math.sqrt(q))**2

# def pca_denoise_cov(returns_df, var_threshold=0.90):
#     """
#     Returns:
#       cov_sample (daily), cov_denoised (daily), n_components_kept, lambda_plus
#     """
#     R = returns_df.dropna(axis=1, how='all')
#     T, N = R.shape[0], R.shape[1]
#     if N == 0 or T <= 1:
#         raise ValueError("Not enough data to compute covariance.")
#     X = (R - R.mean()).values  # demeaned
#     cov = np.cov(X.T, bias=False)
#     # eigen decomp
#     vals, vecs = np.linalg.eigh(cov)
#     idx = np.argsort(vals)[::-1]
#     vals = vals[idx]
#     vecs = vecs[:, idx]
#     # estimate sigma^2 as average variance (diagonal)
#     avg_var = np.mean(np.diag(cov))
#     lambda_plus = marchenko_pastur_lambda_plus(T, N, sigma2=avg_var)
#     n_rmt = int(np.sum(vals > lambda_plus))
#     cumvar = np.cumsum(vals) / np.sum(vals)
#     n_var = int(np.searchsorted(cumvar, var_threshold) + 1)
#     k_keep = max(1, max(n_rmt, n_var))
#     # reconstruct
#     vals_top = vals[:k_keep]
#     vecs_top = vecs[:, :k_keep]
#     cov_denoised = (vecs_top * vals_top) @ vecs_top.T
#     cov_denoised = (cov_denoised + cov_denoised.T) / 2
#     return cov, cov_denoised, k_keep, lambda_plus

# def optimize_weights(mu, cov, gamma, allow_short=False):
#     """
#     maximize w^T mu - (gamma/2) w^T cov w
#     subject sum(w)=1 and w >= 0 if not allow_short
#     """
#     mu = np.array(mu).flatten()
#     cov = np.array(cov)
#     N = len(mu)
#     def obj(w):
#         return - (np.dot(w, mu) - 0.5 * gamma * np.dot(w, cov.dot(w)))
#     cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
#     bounds = tuple((0.0, 1.0) for _ in range(N)) if not allow_short else None
#     x0 = np.repeat(1.0/N, N)
#     res = minimize(obj, x0, bounds=bounds, constraints=cons, method='SLSQP', options={'ftol':1e-9, 'maxiter':1000})
#     if not res.success:
#         # fallback: normalized clipped weights
#         w = np.clip(res.x, 0, None)
#         if w.sum() <= 0:
#             w = np.repeat(1.0/N, N)
#         else:
#             w = w / w.sum()
#     else:
#         w = res.x
#     return np.array(w)

# # ------------------------
# # GMM with 2D features (rolling mean, rolling std)
# # ------------------------
# def fit_gmm_2d(spx_returns, rolling_window=60, k=3, random_state=42):
#     """
#     spx_returns: pd.Series of daily log returns (index: dates)
#     rolling_window: window for computing rolling mean & std
#     returns: fitted gmm, feature_df (aligned), feature_used_df (rows used to fit)
#     """
#     # rolling mean & std (use center=False default)
#     mu_roll = spx_returns.rolling(window=rolling_window).mean()
#     sigma_roll = spx_returns.rolling(window=rolling_window).std()
#     features = pd.DataFrame({"mu": mu_roll, "sigma": sigma_roll}).dropna()
#     if features.shape[0] < k:
#         raise ValueError("Not enough rolling observations to fit GMM. Increase dataset or reduce rolling window.")
#     X = features.values
#     gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=random_state)
#     gmm.fit(X)
#     # component means in 2D
#     comp_means = gmm.means_  # shape (k, 2) -> [mu_mean, sigma_mean]
#     # We'll label components later by sorting comp_means[:,0] (mu)
#     return gmm, features, comp_means

# def label_components_by_mu(comp_means):
#     """
#     Map component index -> regime name based on mu (ascending)
#     returns: dict: comp_idx -> regime_name
#     """
#     order = np.argsort(comp_means[:, 0])  # ascending by mu
#     label_map = {}
#     # lowest -> Bear, mid -> Calm, highest -> Bull
#     mapping = ["Bear", "Calm", "Bull"]
#     # If k != 3, we'll still use names but adapt length
#     for i, comp_idx in enumerate(order):
#         name = mapping[i] if i < len(mapping) else f"Regime_{i+1}"
#         label_map[comp_idx] = name
#     return label_map

# def predict_regime_and_probs(gmm, current_mu, current_sigma, comp_label_map):
#     Xcur = np.array([[current_mu, current_sigma]])
#     probs = gmm.predict_proba(Xcur).flatten()  # length k
#     comp_idx = int(np.argmax(probs))
#     regime_name = comp_label_map.get(comp_idx, f"Comp_{comp_idx}")
#     # build dict of regime name -> prob (map every component via label map)
#     probs_map = {}
#     for idx, p in enumerate(probs):
#         name = comp_label_map.get(idx, f"Comp_{idx}")
#         probs_map[name] = float(p)
#     return regime_name, probs_map, probs, comp_idx

# # ------------------------
# # Streamlit UI
# # ------------------------
# st.title("Smart Framework — GMM(μ,σ) + PCA Denoised Covariance")

# with st.sidebar:
#     st.header("Settings")
#     risk_appetite = st.slider("Risk appetite (1 = aggressive, 10 = conservative)", min_value=1, max_value=10, value=5)
#     gamma = float(risk_appetite)  # risk aversion equals slider (1 aggressive ... 10 conservative)
#     st.markdown(f"**Risk-aversion γ = {gamma:.2f}**")
#     budget_input = st.number_input("Budget (currency units)", min_value=100.0, value=100000.0, step=100.0, format="%.2f")
#     tickers = st.text_input("Tickers (comma-separated, up to 10)", value="AAPL,MSFT,AMZN,GOOG,TSLA,NVDA,JPM,V,PG,MA")
#     tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
#     if len(tickers_list) > 10:
#         st.warning("Only the first 10 tickers will be used.")
#         tickers_list = tickers_list[:10]
#     current_date = st.date_input("Analysis date (current date)", value=pd.to_datetime("2024-01-01").date())
#     gmm_lookback_years = st.number_input("GMM lookback (years) for S&P500", min_value=3, max_value=20, value=10)
#     stocks_lookback_years = st.number_input("Stock lookback (years)", min_value=1, max_value=10, value=5)
#     rolling_window_days = st.number_input("Rolling window (days) for μ,σ (SPX)", min_value=20, max_value=252, value=60)
#     advance_period = st.selectbox("Advance time by", options=["1 month", "1 quarter (3 months)"])
#     btn_run = st.button("Run Smart Framework (Rebalance now)")

# # session state initializations
# if 'history' not in st.session_state:
#     st.session_state.history = []  # list of rebalance dicts
# if 'current_date' not in st.session_state:
#     st.session_state.current_date = pd.to_datetime(current_date)
# if 'budget' not in st.session_state:
#     st.session_state.budget = float(budget_input)
# if 'tickers' not in st.session_state:
#     st.session_state.tickers = tickers_list

# # update tickers if changed
# if tickers_list != st.session_state.tickers:
#     st.session_state.tickers = tickers_list

# # ------------------------
# # Core pipeline function
# # ------------------------
# def run_rebalance(analysis_date, tickers, budget, gamma,
#                   gmm_lookback_years=10, stocks_lookback_years=5, rolling_window=60):
#     """
#     Performs one rebalance:
#       - Fit GMM on SPX rolling mu/sigma
#       - Predict regime & probs
#       - Fetch tickers returns, filter by regime or fallback
#       - Compute mu (annual) and PCA-denoised cov (annual)
#       - Optimize weights (no short)
#       - Return dict with results
#     """
#     analysis_date = pd.to_datetime(analysis_date)
#     # 1) SPX data for GMM
#     spx_end = analysis_date
#     spx_start = spx_end - pd.DateOffset(years=int(gmm_lookback_years))
#     spx_df = download_prices("^GSPC", start=spx_start.strftime("%Y-%m-%d"), end=(spx_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
#     if spx_df.shape[1] == 0:
#         st.error("Could not download S&P500 data.")
#         return None
#     spx_series = spx_df.iloc[:, 0].dropna()
#     spx_returns = compute_log_returns(spx_series)
#     if spx_returns.shape[0] < rolling_window + 5:
#         st.error("Not enough SPX returns for chosen rolling window and lookback.")
#         return None

#     # Fit GMM on rolling features
#     try:
#         gmm, features_df, comp_means = fit_gmm_2d(spx_returns.squeeze(), rolling_window=rolling_window, k=3)
#     except Exception as ex:
#         st.error(f"GMM fit failed: {ex}")
#         return None

#     comp_label_map = label_components_by_mu(comp_means)

#     # compute most recent rolling mu/sigma to predict current regime
#     recent_mu = float(spx_returns.rolling(window=rolling_window).mean().iloc[-1])
#     recent_sigma = float(spx_returns.rolling(window=rolling_window).std().iloc[-1])

#     predicted_regime_name, probs_map, probs_array, comp_idx = predict_regime_and_probs(gmm, recent_mu, recent_sigma, comp_label_map)

#     # 2) Stock data (lookback)
#     stocks_end = analysis_date
#     stocks_start = stocks_end - pd.DateOffset(years=int(stocks_lookback_years))
#     prices = download_prices(tickers, start=stocks_start.strftime("%Y-%m-%d"), end=(stocks_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
#     prices = prices.dropna(how='all').ffill().dropna(how='all')
#     if prices.shape[1] == 0:
#         st.error("No valid stock prices for the selected tickers.")
#         return None
#     stock_returns = compute_log_returns(prices)

#     # 3) Filter stock_returns to regime dates
#     # Need mapping of SPX dates -> component label for each date
#     # Use features_df (dates where rolling mu/sigma exist) and gmm.predict to assign component labels to each date with features
#     features_all = features_df  # index aligned to spx_returns dates where rolling available
#     labels = gmm.predict(features_all.values)  # component index per date (aligned to features_all.index)
#     # build series mapping date -> component_idx
#     regime_series = pd.Series(labels, index=features_all.index)

#     # Filter stock_returns where regime == comp_idx
#     regime_dates = regime_series[regime_series == comp_idx].index
#     filtered_returns = stock_returns.loc[stock_returns.index.intersection(regime_dates)]
#     used_fallback = False
#     if filtered_returns.shape[0] < 30:
#         # fallback to use all regimes
#         used_fallback = True
#         filtered_returns = stock_returns.copy()

#     if filtered_returns.shape[1] == 0 or filtered_returns.shape[0] < 2:
#         st.error("Not enough stock return data after filtering/fallback.")
#         return None

#     # 4) compute mu (annual) and cov denoising (daily -> annualize later)
#     mu_daily = filtered_returns.mean()
#     mu_annual = annualize_rets(mu_daily)
#     cov_sample_daily, cov_denoised_daily, k_keep, lambda_plus = pca_denoise_cov(filtered_returns, var_threshold=0.90)
#     cov_annual = annualize_cov(cov_denoised_daily)

#     # Keep only assets that have mu and exist in cov matrix
#     assets = list(filtered_returns.columns)
#     mu_vector = mu_annual.reindex(assets).fillna(0.0).values
#     cov_matrix = pd.DataFrame(cov_annual, index=assets, columns=assets).reindex(index=assets, columns=assets).fillna(0.0).values

#     # 5) Optimize
#     weights = optimize_weights(mu_vector, cov_matrix, gamma=gamma, allow_short=False)
#     weights_series = pd.Series(weights, index=assets)

#     # Naive 1/N for comparison
#     w_naive = np.repeat(1.0 / len(assets), len(assets))
#     weights_naive_series = pd.Series(w_naive, index=assets)

#     # Forecasted port stats (annual)
#     port_mu = float(np.dot(weights, mu_vector))
#     port_var = float(weights.dot(cov_matrix.dot(weights)))
#     port_vol = math.sqrt(max(port_var, 0.0))
#     port_mu_naive = float(np.dot(w_naive, mu_vector))
#     port_var_naive = float(w_naive.dot(cov_matrix.dot(w_naive)))
#     port_vol_naive = math.sqrt(max(port_var_naive, 0.0))

#     # allocation
#     allocation = (weights_series * budget).round(2)
#     allocation_naive = (weights_naive_series * budget).round(2)

#     result = {
#         "analysis_date": analysis_date,
#         "gmm": gmm,
#         "comp_means": comp_means,
#         "comp_label_map": comp_label_map,
#         "predicted_regime_name": predicted_regime_name,
#         "probs_map": probs_map,
#         "probs_array": probs_array,
#         "regime_series": regime_series,
#         "used_fallback": used_fallback,
#         "assets": assets,
#         "weights_smart": weights_series,
#         "allocation_smart": allocation,
#         "weights_naive": weights_naive_series,
#         "allocation_naive": allocation_naive,
#         "mu_annual": pd.Series(mu_vector, index=assets),
#         "cov_annual": pd.DataFrame(cov_matrix, index=assets, columns=assets),
#         "k_keep": k_keep,
#         "lambda_plus": lambda_plus,
#         "forecast_port_mu": port_mu,
#         "forecast_port_vol": port_vol,
#         "forecast_port_mu_naive": port_mu_naive,
#         "forecast_port_vol_naive": port_vol_naive,
#         "prices": prices,
#         "filtered_returns": filtered_returns,
#     }
#     return result

# # ------------------------
# # Run (Rebalance now)
# # ------------------------
# if btn_run:
#     st.session_state.current_date = pd.to_datetime(current_date)
#     st.session_state.budget = float(budget_input)
#     st.session_state.tickers = tickers_list
#     st.session_state.history = []  # reset on fresh run
#     st.info("Running Smart Framework...")

#     out = run_rebalance(st.session_state.current_date, st.session_state.tickers, st.session_state.budget,
#                         gamma, gmm_lookback_years=int(gmm_lookback_years),
#                         stocks_lookback_years=int(stocks_lookback_years),
#                         rolling_window=int(rolling_window_days))
#     if out is not None:
#         entry = {
#             "date": out["analysis_date"],
#             "predicted_regime": out["predicted_regime_name"],
#             "probs_map": out["probs_map"],
#             "used_fallback": out["used_fallback"],
#             "weights_smart": out["weights_smart"],
#             "weights_naive": out["weights_naive"],
#             "allocation_smart": out["allocation_smart"],
#             "allocation_naive": out["allocation_naive"],
#             "mu_annual": out["mu_annual"],
#             "cov_annual": out["cov_annual"],
#             "k_keep": out["k_keep"],
#             "lambda_plus": out["lambda_plus"],
#             "forecast_port_mu": out["forecast_port_mu"],
#             "forecast_port_vol": out["forecast_port_vol"],
#             "forecast_port_mu_naive": out["forecast_port_mu_naive"],
#             "forecast_port_vol_naive": out["forecast_port_vol_naive"],
#             "prices": out["prices"],
#             "filtered_returns": out["filtered_returns"],
#             # realized (to be filled after advancing time)
#             "realized_return_smart": None,
#             "realized_vol_smart": None,
#             "realized_return_naive": None,
#             "realized_vol_naive": None,
#             "sharpe_smart": None,
#             "sharpe_naive": None
#         }
#         st.session_state.history.append(entry)
#         st.success("Rebalance computed and added to history.")

# # ------------------------
# # Advance time & realize returns
# # ------------------------
# col1, col2 = st.columns([1,1])
# with col1:
#     if st.button("Advance time"):
#         # advance current date
#         delta = pd.DateOffset(months=1) if advance_period == "1 month" else pd.DateOffset(months=3)
#         prev_date = st.session_state.current_date
#         new_date = prev_date + delta
#         st.session_state.current_date = new_date
#         st.info(f"Advanced from {prev_date.date()} to {new_date.date()}")

#         if len(st.session_state.history) == 0:
#             st.warning("No prior rebalance to realize returns for.")
#         else:
#             last = st.session_state.history[-1]
#             prices = last.get("prices")
#             if prices is None or prices.shape[0] == 0:
#                 st.warning("No price data to compute realized returns.")
#             else:
#                 reb_date = pd.to_datetime(last["date"])
#                 # realized window: >reb_date and <= new_date
#                 mask = (prices.index > reb_date) & (prices.index <= new_date)
#                 window_prices = prices.loc[mask]
#                 if window_prices.shape[0] == 0:
#                     st.warning("No price data in advanced period; cannot compute realized P&L.")
#                 else:
#                     returns_window = compute_log_returns(prices.loc[prices.index <= new_date])
#                     realized_window = returns_window.loc[(returns_window.index > reb_date) & (returns_window.index <= new_date)]
#                     if realized_window.shape[0] == 0:
#                         st.warning("Not enough return rows in realized window.")
#                     else:
#                         # compute realized daily portfolio returns
#                         assets = last["weights_smart"].index.intersection(realized_window.columns)
#                         w_smart = last["weights_smart"].reindex(realized_window.columns).fillna(0.0)
#                         w_naive = last["weights_naive"].reindex(realized_window.columns).fillna(0.0)
#                         smart_daily = realized_window.dot(w_smart)
#                         naive_daily = realized_window.dot(w_naive)
#                         # sum of log returns is log cumulative return (approx)
#                         realized_log_smart = float(smart_daily.sum())
#                         realized_return_smart = math.expm1(realized_log_smart)
#                         realized_vol_smart = float(smart_daily.std() * math.sqrt(252))
#                         realized_log_naive = float(naive_daily.sum())
#                         realized_return_naive = math.expm1(realized_log_naive)
#                         realized_vol_naive = float(naive_daily.std() * math.sqrt(252))
#                         sharpe_smart = realized_return_smart / (realized_vol_smart + 1e-12) if realized_vol_smart > 0 else np.nan
#                         sharpe_naive = realized_return_naive / (realized_vol_naive + 1e-12) if realized_vol_naive > 0 else np.nan
#                         last["realized_return_smart"] = float(realized_return_smart)
#                         last["realized_vol_smart"] = float(realized_vol_smart)
#                         last["realized_return_naive"] = float(realized_return_naive)
#                         last["realized_vol_naive"] = float(realized_vol_naive)
#                         last["sharpe_smart"] = float(sharpe_smart)
#                         last["sharpe_naive"] = float(sharpe_naive)
#                         # Update budget to realized portfolio value (apply P/L)
#                         prev_budget = st.session_state.budget
#                         new_budget = prev_budget * (1.0 + realized_return_smart)
#                         st.session_state.budget = float(max(new_budget, 0.0))
#                         st.success(f"Realized Smart return: {realized_return_smart:.2%}. New budget = {st.session_state.budget:,.2f}")

# with col2:
#     if st.button("Rebalance now (use current budget)"):
#         # allow user to change tickers (from sidebar) but budget is session budget
#         out = run_rebalance(st.session_state.current_date, st.session_state.tickers, st.session_state.budget,
#                             gamma, gmm_lookback_years=int(gmm_lookback_years),
#                             stocks_lookback_years=int(stocks_lookback_years),
#                             rolling_window=int(rolling_window_days))
#         if out is not None:
#             entry = {
#                 "date": out["analysis_date"],
#                 "predicted_regime": out["predicted_regime_name"],
#                 "probs_map": out["probs_map"],
#                 "used_fallback": out["used_fallback"],
#                 "weights_smart": out["weights_smart"],
#                 "weights_naive": out["weights_naive"],
#                 "allocation_smart": out["allocation_smart"],
#                 "allocation_naive": out["allocation_naive"],
#                 "mu_annual": out["mu_annual"],
#                 "cov_annual": out["cov_annual"],
#                 "k_keep": out["k_keep"],
#                 "lambda_plus": out["lambda_plus"],
#                 "forecast_port_mu": out["forecast_port_mu"],
#                 "forecast_port_vol": out["forecast_port_vol"],
#                 "forecast_port_mu_naive": out["forecast_port_mu_naive"],
#                 "forecast_port_vol_naive": out["forecast_port_vol_naive"],
#                 "prices": out["prices"],
#                 "filtered_returns": out["filtered_returns"],
#                 "realized_return_smart": None,
#                 "realized_vol_smart": None,
#                 "realized_return_naive": None,
#                 "realized_vol_naive": None,
#                 "sharpe_smart": None,
#                 "sharpe_naive": None
#             }
#             st.session_state.history.append(entry)
#             st.success("Rebalance computed and appended to history.")

# # ------------------------
# # Visualizations & tables
# # ------------------------
# st.markdown("## Rebalance history & analytics")
# if len(st.session_state.history) == 0:
#     st.info("No rebalances yet. Click 'Run Smart Framework' to create the first rebalance.")
# else:
#     # Summary table
#     rows = []
#     for i, e in enumerate(st.session_state.history):
#         rows.append({
#             "idx": i+1,
#             "date": e["date"].date(),
#             "predicted_regime": e["predicted_regime"],
#             "used_fallback": e["used_fallback"],
#             "k_keep": e["k_keep"],
#             "forecast_return": e["forecast_port_mu"],
#             "forecast_vol": e["forecast_port_vol"],
#             "forecast_return_naive": e["forecast_port_mu_naive"],
#             "forecast_vol_naive": e["forecast_port_vol_naive"],
#             "realized_return_smart": e.get("realized_return_smart"),
#             "realized_vol_smart": e.get("realized_vol_smart"),
#             "realized_return_naive": e.get("realized_return_naive"),
#             "realized_vol_naive": e.get("realized_vol_naive"),
#             "sharpe_smart": e.get("sharpe_smart"),
#             "sharpe_naive": e.get("sharpe_naive")
#         })
#     summary_df = pd.DataFrame(rows).set_index("idx")
#     st.dataframe(summary_df.style.format({
#         "forecast_return": "{:.2%}",
#         "forecast_vol": "{:.2%}",
#         "forecast_return_naive": "{:.2%}",
#         "forecast_vol_naive": "{:.2%}",
#         "realized_return_smart": lambda x: f"{x:.2%}" if pd.notnull(x) else "",
#         "realized_vol_smart": lambda x: f"{x:.2%}" if pd.notnull(x) else "",
#         "realized_return_naive": lambda x: f"{x:.2%}" if pd.notnull(x) else "",
#         "realized_vol_naive": lambda x: f"{x:.2%}" if pd.notnull(x) else "",
#         "sharpe_smart": "{:.3f}",
#         "sharpe_naive": "{:.3f}"
#     }))

#     # Regime probabilities display for last rebalance
#     st.markdown("### Last predicted regime & posterior probabilities")
#     last = st.session_state.history[-1]
#     probs_map = last.get("probs_map", {})
#     probs_df = pd.DataFrame.from_dict(probs_map, orient="index", columns=["probability"])
#     probs_df = probs_df.sort_values("probability", ascending=False)
#     st.table(probs_df.style.format({"probability": "{:.2%}"}))

#     # bar chart of probs
#     st.bar_chart(probs_df["probability"])

#     # Portfolio performance line chart (realized segments per rebalance) vs 1/N
#     st.markdown("### Portfolio performance per rebalance (realized periods) — Smart vs 1/N")
#     perf_df = pd.DataFrame()
#     for i, e in enumerate(st.session_state.history):
#         prices = e.get("prices")
#         if prices is None or prices.shape[0] == 0:
#             continue
#         start_dt = pd.to_datetime(e["date"])
#         # end date = next rebalance date if exists else last available price
#         if i + 1 < len(st.session_state.history):
#             end_dt = pd.to_datetime(st.session_state.history[i+1]["date"])
#         else:
#             end_dt = prices.index[-1]
#         returns_df = compute_log_returns(prices)
#         interval = returns_df.loc[(returns_df.index > start_dt) & (returns_df.index <= end_dt)]
#         if interval.shape[0] == 0:
#             continue
#         w_smart = e["weights_smart"].reindex(interval.columns).fillna(0.0)
#         w_naive = e["weights_naive"].reindex(interval.columns).fillna(0.0)
#         smart_daily = interval.dot(w_smart)
#         naive_daily = interval.dot(w_naive)
#         smart_cum = (np.exp(smart_daily).cumprod() - 1)  # cumulative simple return
#         naive_cum = (np.exp(naive_daily).cumprod() - 1)
#         perf_df = pd.concat([perf_df, pd.DataFrame({
#             f"Smart_{i+1}": smart_cum,
#             f"Naive_{i+1}": naive_cum
#         })], axis=1)
#     if perf_df.shape[1] > 0:
#         st.line_chart(perf_df.fillna(method='ffill').fillna(0))
#     else:
#         st.info("No realized performance periods yet. Advance time after a rebalance to get realized P&L.")

#     # Pearson correlations between mu_regime (predicted mu) vs actual asset realized returns in the realized window
#     st.markdown("### Pearson correlation: predicted μ (per-asset) vs actual realized asset returns (per rebalance)")
#     pearson_rows = []
#     for i, e in enumerate(st.session_state.history):
#         if e.get("realized_return_smart") is None:
#             continue
#         prices = e.get("prices")
#         if prices is None:
#             continue
#         reb_date = pd.to_datetime(e["date"])
#         returns_all = compute_log_returns(prices)
#         realized_window = returns_all.loc[(returns_all.index > reb_date)]
#         if realized_window.shape[0] == 0:
#             continue
#         actual_asset_log = realized_window.sum()
#         actual_asset_simple = np.expm1(actual_asset_log)
#         mu_series = e.get("mu_annual")
#         if mu_series is None:
#             continue
#         common = actual_asset_simple.index.intersection(mu_series.index)
#         if len(common) < 2:
#             continue
#         try:
#             r, p = pearsonr(mu_series.reindex(common).values, actual_asset_simple.reindex(common).values)
#         except Exception:
#             r, p = np.nan, np.nan
#         pearson_rows.append({"rebalance": i+1, "date": e["date"], "pearson_mu_vs_actual": r})
#     if len(pearson_rows) > 0:
#         pearson_df = pd.DataFrame(pearson_rows).set_index("rebalance")
#         st.dataframe(pearson_df.style.format({"pearson_mu_vs_actual": "{:.3f}"}))
#     else:
#         st.info("No realized data available to compute Pearson correlations yet.")

#     # Show last allocation table
#     st.markdown("### Last allocation (Smart vs 1/N)")
#     last = st.session_state.history[-1]
#     alloc_df = pd.DataFrame({
#         "Smart_weight": last["weights_smart"],
#         "Smart_alloc": last["allocation_smart"],
#         "Naive_weight": last["weights_naive"],
#         "Naive_alloc": last["allocation_naive"]
#     }).fillna(0.0)
#     st.dataframe(alloc_df.style.format({
#         "Smart_weight": "{:.4f}",
#         "Smart_alloc": "{:,.2f}",
#         "Naive_weight": "{:.4f}",
#         "Naive_alloc": "{:,.2f}"
#     }))

# # ------------------------
# # Notes & requirements
# # ------------------------
# st.markdown("---")
# st.markdown("### Notes & assumptions")
# st.markdown("""
# - **GMM features:** rolling μ and σ of daily S&P 500 log returns (rolling window configurable).  
# - **GMM k=3:** components labeled by sorting the component means of μ: lowest→Bear, middle→Calm, highest→Bull.  
# - **Regime selection:** filter N-stock returns to dates where SPX rolling (μ,σ) belonged to predicted component. If filtered sample < 30 observations, we **use all stock data** as fallback.  
# - **Covariance denoising:** PCA with hybrid rule — keep `max(#eigenvalues > λ+ (MP), #components for 90% variance)`. λ+ is computed with Marchenko-Pastur using average variance.  
# - **Optimization:** maximize `w^T μ - (γ/2) w^T Σ w` with `sum(w)=1` and `w>=0`. `γ` is set equal to the Risk Appetite slider (1 aggressive ... 10 conservative).  
# - **Simulation:** Advance time by month/quarter uses realized returns from yfinance price history to update the budget.  
# - All returns are computed using **log returns** internally; displayed cumulative returns are converted to simple returns for interpretability.
# """)

# st.markdown("### Requirements")
# st.code("""
# pip install streamlit pandas numpy yfinance scikit-learn scipy
# """)
