# smart_framework_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy.stats import pearsonr
from datetime import datetime
import math

st.set_page_config(page_title="Smart Framework — GMM(μ,σ) + PCA Denoised Covariance", layout="wide")

# ------------------------
# Utilities & caching
# ------------------------
@st.cache_data(show_spinner=False)
def download_prices(tickers, start, end):
    if isinstance(tickers, (list, tuple)):
        tickers_list = tickers
    else:
        tickers_list = [tickers]
    df = yf.download(tickers_list, start=start, end=end, progress=False, threads=True)
    # prefer 'Adj Close' if available
    if isinstance(df, pd.DataFrame) and 'Adj Close' in df.columns:
        df = df['Adj Close']
    # if single ticker, ensure DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame()
    # normalize column names (remove spaces)
    df.columns = [c.replace("^", "").strip() for c in df.columns]
    return df

def compute_log_returns(price_df):
    return np.log(price_df / price_df.shift(1)).dropna(how='all')

def annualize_rets(mu_daily, days=252):
    return mu_daily * days

def annualize_cov(cov_daily, days=252):
    return cov_daily * days

def marchenko_pastur_lambda_plus(T, N, sigma2=1.0):
    # Marchenko-Pastur upper bound
    q = N / T
    if q <= 0:
        return 0.0
    return sigma2 * (1 + math.sqrt(q))**2

def pca_denoise_cov(returns_df, var_threshold=0.90):
    """
    Returns:
      cov_sample (daily), cov_denoised (daily), n_components_kept, lambda_plus
    """
    R = returns_df.dropna(axis=1, how='all')
    T, N = R.shape[0], R.shape[1]
    if N == 0 or T <= 1:
        raise ValueError("Not enough data to compute covariance.")
    X = (R - R.mean()).values  # demeaned
    cov = np.cov(X.T, bias=False)
    # eigen decomp
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    # estimate sigma^2 as average variance (diagonal)
    avg_var = np.mean(np.diag(cov))
    lambda_plus = marchenko_pastur_lambda_plus(T, N, sigma2=avg_var)
    n_rmt = int(np.sum(vals > lambda_plus))
    cumvar = np.cumsum(vals) / np.sum(vals)
    n_var = int(np.searchsorted(cumvar, var_threshold) + 1)
    k_keep = max(1, max(n_rmt, n_var))
    # reconstruct
    vals_top = vals[:k_keep]
    vecs_top = vecs[:, :k_keep]
    cov_denoised = (vecs_top * vals_top) @ vecs_top.T
    cov_denoised = (cov_denoised + cov_denoised.T) / 2
    return cov, cov_denoised, k_keep, lambda_plus

def optimize_weights(mu, cov, gamma, allow_short=False):
    """
    maximize w^T mu - (gamma/2) w^T cov w
    subject sum(w)=1 and w >= 0 if not allow_short
    """
    mu = np.array(mu).flatten()
    cov = np.array(cov)
    N = len(mu)
    def obj(w):
        return - (np.dot(w, mu) - 0.5 * gamma * np.dot(w, cov.dot(w)))
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(N)) if not allow_short else None
    x0 = np.repeat(1.0/N, N)
    res = minimize(obj, x0, bounds=bounds, constraints=cons, method='SLSQP', options={'ftol':1e-9, 'maxiter':1000})
    if not res.success:
        # fallback: normalized clipped weights
        w = np.clip(res.x, 0, None)
        if w.sum() <= 0:
            w = np.repeat(1.0/N, N)
        else:
            w = w / w.sum()
    else:
        w = res.x
    return np.array(w)

# ------------------------
# GMM with 2D features (rolling mean, rolling std)
# ------------------------
def fit_gmm_2d(spx_returns, rolling_window=60, k=3, random_state=42):
    """
    spx_returns: pd.Series of daily log returns (index: dates)
    rolling_window: window for computing rolling mean & std
    returns: fitted gmm, feature_df (aligned), feature_used_df (rows used to fit)
    """
    # rolling mean & std (use center=False default)
    mu_roll = spx_returns.rolling(window=rolling_window).mean()
    sigma_roll = spx_returns.rolling(window=rolling_window).std()
    features = pd.DataFrame({"mu": mu_roll, "sigma": sigma_roll}).dropna()
    if features.shape[0] < k:
        raise ValueError("Not enough rolling observations to fit GMM. Increase dataset or reduce rolling window.")
    X = features.values
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=random_state)
    gmm.fit(X)
    # component means in 2D
    comp_means = gmm.means_  # shape (k, 2) -> [mu_mean, sigma_mean]
    # We'll label components later by sorting comp_means[:,0] (mu)
    return gmm, features, comp_means

def label_components_by_mu(comp_means):
    """
    Map component index -> regime name based on mu (ascending)
    returns: dict: comp_idx -> regime_name
    """
    order = np.argsort(comp_means[:, 0])  # ascending by mu
    label_map = {}
    # lowest -> Bear, mid -> Calm, highest -> Bull
    mapping = ["Bear", "Calm", "Bull"]
    # If k != 3, we'll still use names but adapt length
    for i, comp_idx in enumerate(order):
        name = mapping[i] if i < len(mapping) else f"Regime_{i+1}"
        label_map[comp_idx] = name
    return label_map

def predict_regime_and_probs(gmm, current_mu, current_sigma, comp_label_map):
    Xcur = np.array([[current_mu, current_sigma]])
    probs = gmm.predict_proba(Xcur).flatten()  # length k
    comp_idx = int(np.argmax(probs))
    regime_name = comp_label_map.get(comp_idx, f"Comp_{comp_idx}")
    # build dict of regime name -> prob (map every component via label map)
    probs_map = {}
    for idx, p in enumerate(probs):
        name = comp_label_map.get(idx, f"Comp_{idx}")
        probs_map[name] = float(p)
    return regime_name, probs_map, probs, comp_idx

# ------------------------
# Streamlit UI
# ------------------------
st.title("Smart Framework — GMM(μ,σ) + PCA Denoised Covariance")

with st.sidebar:
    st.header("Settings")
    risk_appetite = st.slider("Risk appetite (1 = aggressive, 10 = conservative)", min_value=1, max_value=10, value=5)
    gamma = float(risk_appetite)  # risk aversion equals slider (1 aggressive ... 10 conservative)
    st.markdown(f"**Risk-aversion γ = {gamma:.2f}**")
    budget_input = st.number_input("Budget (currency units)", min_value=100.0, value=100000.0, step=100.0, format="%.2f")
    tickers = st.text_input("Tickers (comma-separated, up to 10)", value="AAPL,MSFT,AMZN,GOOG,TSLA,NVDA,JPM,V,PG,MA")
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(tickers_list) > 10:
        st.warning("Only the first 10 tickers will be used.")
        tickers_list = tickers_list[:10]
    current_date = st.date_input("Analysis date (current date)", value=pd.to_datetime("2024-01-01").date())
    gmm_lookback_years = st.number_input("GMM lookback (years) for S&P500", min_value=3, max_value=20, value=10)
    stocks_lookback_years = st.number_input("Stock lookback (years)", min_value=1, max_value=10, value=5)
    rolling_window_days = st.number_input("Rolling window (days) for μ,σ (SPX)", min_value=20, max_value=252, value=60)
    advance_period = st.selectbox("Advance time by", options=["1 month", "1 quarter (3 months)"])
    btn_run = st.button("Run Smart Framework (Rebalance now)")

# session state initializations
if 'history' not in st.session_state:
    st.session_state.history = []  # list of rebalance dicts
if 'current_date' not in st.session_state:
    st.session_state.current_date = pd.to_datetime(current_date)
if 'budget' not in st.session_state:
    st.session_state.budget = float(budget_input)
if 'tickers' not in st.session_state:
    st.session_state.tickers = tickers_list

# update tickers if changed
if tickers_list != st.session_state.tickers:
    st.session_state.tickers = tickers_list

# ------------------------
# Core pipeline function
# ------------------------
def run_rebalance(analysis_date, tickers, budget, gamma,
                  gmm_lookback_years=10, stocks_lookback_years=5, rolling_window=60):
    """
    Performs one rebalance:
      - Fit GMM on SPX rolling mu/sigma
      - Predict regime & probs
      - Fetch tickers returns, filter by regime or fallback
      - Compute mu (annual) and PCA-denoised cov (annual)
      - Optimize weights (no short)
      - Return dict with results
    """
    analysis_date = pd.to_datetime(analysis_date)
    # 1) SPX data for GMM
    spx_end = analysis_date
    spx_start = spx_end - pd.DateOffset(years=int(gmm_lookback_years))
    spx_df = download_prices("^GSPC", start=spx_start.strftime("%Y-%m-%d"), end=(spx_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    if spx_df.shape[1] == 0:
        st.error("Could not download S&P500 data.")
        return None
    spx_series = spx_df.iloc[:, 0].dropna()
    spx_returns = compute_log_returns(spx_series)
    if spx_returns.shape[0] < rolling_window + 5:
        st.error("Not enough SPX returns for chosen rolling window and lookback.")
        return None

    # Fit GMM on rolling features
    try:
        gmm, features_df, comp_means = fit_gmm_2d(spx_returns.squeeze(), rolling_window=rolling_window, k=3)
    except Exception as ex:
        st.error(f"GMM fit failed: {ex}")
        return None

    comp_label_map = label_components_by_mu(comp_means)

    # compute most recent rolling mu/sigma to predict current regime
    recent_mu = float(spx_returns.rolling(window=rolling_window).mean().iloc[-1])
    recent_sigma = float(spx_returns.rolling(window=rolling_window).std().iloc[-1])

    predicted_regime_name, probs_map, probs_array, comp_idx = predict_regime_and_probs(gmm, recent_mu, recent_sigma, comp_label_map)

    # 2) Stock data (lookback)
    stocks_end = analysis_date
    stocks_start = stocks_end - pd.DateOffset(years=int(stocks_lookback_years))
    prices = download_prices(tickers, start=stocks_start.strftime("%Y-%m-%d"), end=(stocks_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    prices = prices.dropna(how='all').ffill().dropna(how='all')
    if prices.shape[1] == 0:
        st.error("No valid stock prices for the selected tickers.")
        return None
    stock_returns = compute_log_returns(prices)

    # 3) Filter stock_returns to regime dates
    # Need mapping of SPX dates -> component label for each date
    # Use features_df (dates where rolling mu/sigma exist) and gmm.predict to assign component labels to each date with features
    features_all = features_df  # index aligned to spx_returns dates where rolling available
    labels = gmm.predict(features_all.values)  # component index per date (aligned to features_all.index)
    # build series mapping date -> component_idx
    regime_series = pd.Series(labels, index=features_all.index)

    # Filter stock_returns where regime == comp_idx
    regime_dates = regime_series[regime_series == comp_idx].index
    filtered_returns = stock_returns.loc[stock_returns.index.intersection(regime_dates)]
    used_fallback = False
    if filtered_returns.shape[0] < 30:
        # fallback to use all regimes
        used_fallback = True
        filtered_returns = stock_returns.copy()

    if filtered_returns.shape[1] == 0 or filtered_returns.shape[0] < 2:
        st.error("Not enough stock return data after filtering/fallback.")
        return None

    # 4) compute mu (annual) and cov denoising (daily -> annualize later)
    mu_daily = filtered_returns.mean()
    mu_annual = annualize_rets(mu_daily)
    cov_sample_daily, cov_denoised_daily, k_keep, lambda_plus = pca_denoise_cov(filtered_returns, var_threshold=0.90)
    cov_annual = annualize_cov(cov_denoised_daily)

    # Keep only assets that have mu and exist in cov matrix
    assets = list(filtered_returns.columns)
    mu_vector = mu_annual.reindex(assets).fillna(0.0).values
    cov_matrix = pd.DataFrame(cov_annual, index=assets, columns=assets).reindex(index=assets, columns=assets).fillna(0.0).values

    # 5) Optimize
    weights = optimize_weights(mu_vector, cov_matrix, gamma=gamma, allow_short=False)
    weights_series = pd.Series(weights, index=assets)

    # Naive 1/N for comparison
    w_naive = np.repeat(1.0 / len(assets), len(assets))
    weights_naive_series = pd.Series(w_naive, index=assets)

    # Forecasted port stats (annual)
    port_mu = float(np.dot(weights, mu_vector))
    port_var = float(weights.dot(cov_matrix.dot(weights)))
    port_vol = math.sqrt(max(port_var, 0.0))
    port_mu_naive = float(np.dot(w_naive, mu_vector))
    port_var_naive = float(w_naive.dot(cov_matrix.dot(w_naive)))
    port_vol_naive = math.sqrt(max(port_var_naive, 0.0))

    # allocation
    allocation = (weights_series * budget).round(2)
    allocation_naive = (weights_naive_series * budget).round(2)

    result = {
        "analysis_date": analysis_date,
        "gmm": gmm,
        "comp_means": comp_means,
        "comp_label_map": comp_label_map,
        "predicted_regime_name": predicted_regime_name,
        "probs_map": probs_map,
        "probs_array": probs_array,
        "regime_series": regime_series,
        "used_fallback": used_fallback,
        "assets": assets,
        "weights_smart": weights_series,
        "allocation_smart": allocation,
        "weights_naive": weights_naive_series,
        "allocation_naive": allocation_naive,
        "mu_annual": pd.Series(mu_vector, index=assets),
        "cov_annual": pd.DataFrame(cov_matrix, index=assets, columns=assets),
        "k_keep": k_keep,
        "lambda_plus": lambda_plus,
        "forecast_port_mu": port_mu,
        "forecast_port_vol": port_vol,
        "forecast_port_mu_naive": port_mu_naive,
        "forecast_port_vol_naive": port_vol_naive,
        "prices": prices,
        "filtered_returns": filtered_returns,
    }
    return result

# ------------------------
# Run (Rebalance now)
# ------------------------
if btn_run:
    st.session_state.current_date = pd.to_datetime(current_date)
    st.session_state.budget = float(budget_input)
    st.session_state.tickers = tickers_list
    st.session_state.history = []  # reset on fresh run
    st.info("Running Smart Framework...")

    out = run_rebalance(st.session_state.current_date, st.session_state.tickers, st.session_state.budget,
                        gamma, gmm_lookback_years=int(gmm_lookback_years),
                        stocks_lookback_years=int(stocks_lookback_years),
                        rolling_window=int(rolling_window_days))
    if out is not None:
        entry = {
            "date": out["analysis_date"],
            "predicted_regime": out["predicted_regime_name"],
            "probs_map": out["probs_map"],
            "used_fallback": out["used_fallback"],
            "weights_smart": out["weights_smart"],
            "weights_naive": out["weights_naive"],
            "allocation_smart": out["allocation_smart"],
            "allocation_naive": out["allocation_naive"],
            "mu_annual": out["mu_annual"],
            "cov_annual": out["cov_annual"],
            "k_keep": out["k_keep"],
            "lambda_plus": out["lambda_plus"],
            "forecast_port_mu": out["forecast_port_mu"],
            "forecast_port_vol": out["forecast_port_vol"],
            "forecast_port_mu_naive": out["forecast_port_mu_naive"],
            "forecast_port_vol_naive": out["forecast_port_vol_naive"],
            "prices": out["prices"],
            "filtered_returns": out["filtered_returns"],
            # realized (to be filled after advancing time)
            "realized_return_smart": None,
            "realized_vol_smart": None,
            "realized_return_naive": None,
            "realized_vol_naive": None,
            "sharpe_smart": None,
            "sharpe_naive": None
        }
        st.session_state.history.append(entry)
        st.success("Rebalance computed and added to history.")

# ------------------------
# Advance time & realize returns
# ------------------------
col1, col2 = st.columns([1,1])
with col1:
    if st.button("Advance time"):
        # advance current date
        delta = pd.DateOffset(months=1) if advance_period == "1 month" else pd.DateOffset(months=3)
        prev_date = st.session_state.current_date
        new_date = prev_date + delta
        st.session_state.current_date = new_date
        st.info(f"Advanced from {prev_date.date()} to {new_date.date()}")

        if len(st.session_state.history) == 0:
            st.warning("No prior rebalance to realize returns for.")
        else:
            last = st.session_state.history[-1]
            prices = last.get("prices")
            if prices is None or prices.shape[0] == 0:
                st.warning("No price data to compute realized returns.")
            else:
                reb_date = pd.to_datetime(last["date"])
                # realized window: >reb_date and <= new_date
                mask = (prices.index > reb_date) & (prices.index <= new_date)
                window_prices = prices.loc[mask]
                if window_prices.shape[0] == 0:
                    st.warning("No price data in advanced period; cannot compute realized P&L.")
                else:
                    returns_window = compute_log_returns(prices.loc[prices.index <= new_date])
                    realized_window = returns_window.loc[(returns_window.index > reb_date) & (returns_window.index <= new_date)]
                    if realized_window.shape[0] == 0:
                        st.warning("Not enough return rows in realized window.")
                    else:
                        # compute realized daily portfolio returns
                        assets = last["weights_smart"].index.intersection(realized_window.columns)
                        w_smart = last["weights_smart"].reindex(realized_window.columns).fillna(0.0)
                        w_naive = last["weights_naive"].reindex(realized_window.columns).fillna(0.0)
                        smart_daily = realized_window.dot(w_smart)
                        naive_daily = realized_window.dot(w_naive)
                        # sum of log returns is log cumulative return (approx)
                        realized_log_smart = float(smart_daily.sum())
                        realized_return_smart = math.expm1(realized_log_smart)
                        realized_vol_smart = float(smart_daily.std() * math.sqrt(252))
                        realized_log_naive = float(naive_daily.sum())
                        realized_return_naive = math.expm1(realized_log_naive)
                        realized_vol_naive = float(naive_daily.std() * math.sqrt(252))
                        sharpe_smart = realized_return_smart / (realized_vol_smart + 1e-12) if realized_vol_smart > 0 else np.nan
                        sharpe_naive = realized_return_naive / (realized_vol_naive + 1e-12) if realized_vol_naive > 0 else np.nan
                        last["realized_return_smart"] = float(realized_return_smart)
                        last["realized_vol_smart"] = float(realized_vol_smart)
                        last["realized_return_naive"] = float(realized_return_naive)
                        last["realized_vol_naive"] = float(realized_vol_naive)
                        last["sharpe_smart"] = float(sharpe_smart)
                        last["sharpe_naive"] = float(sharpe_naive)
                        # Update budget to realized portfolio value (apply P/L)
                        prev_budget = st.session_state.budget
                        new_budget = prev_budget * (1.0 + realized_return_smart)
                        st.session_state.budget = float(max(new_budget, 0.0))
                        st.success(f"Realized Smart return: {realized_return_smart:.2%}. New budget = {st.session_state.budget:,.2f}")

with col2:
    if st.button("Rebalance now (use current budget)"):
        # allow user to change tickers (from sidebar) but budget is session budget
        out = run_rebalance(st.session_state.current_date, st.session_state.tickers, st.session_state.budget,
                            gamma, gmm_lookback_years=int(gmm_lookback_years),
                            stocks_lookback_years=int(stocks_lookback_years),
                            rolling_window=int(rolling_window_days))
        if out is not None:
            entry = {
                "date": out["analysis_date"],
                "predicted_regime": out["predicted_regime_name"],
                "probs_map": out["probs_map"],
                "used_fallback": out["used_fallback"],
                "weights_smart": out["weights_smart"],
                "weights_naive": out["weights_naive"],
                "allocation_smart": out["allocation_smart"],
                "allocation_naive": out["allocation_naive"],
                "mu_annual": out["mu_annual"],
                "cov_annual": out["cov_annual"],
                "k_keep": out["k_keep"],
                "lambda_plus": out["lambda_plus"],
                "forecast_port_mu": out["forecast_port_mu"],
                "forecast_port_vol": out["forecast_port_vol"],
                "forecast_port_mu_naive": out["forecast_port_mu_naive"],
                "forecast_port_vol_naive": out["forecast_port_vol_naive"],
                "prices": out["prices"],
                "filtered_returns": out["filtered_returns"],
                "realized_return_smart": None,
                "realized_vol_smart": None,
                "realized_return_naive": None,
                "realized_vol_naive": None,
                "sharpe_smart": None,
                "sharpe_naive": None
            }
            st.session_state.history.append(entry)
            st.success("Rebalance computed and appended to history.")

# ------------------------
# Visualizations & tables
# ------------------------
st.markdown("## Rebalance history & analytics")
if len(st.session_state.history) == 0:
    st.info("No rebalances yet. Click 'Run Smart Framework' to create the first rebalance.")
else:
    # Summary table
    rows = []
    for i, e in enumerate(st.session_state.history):
        rows.append({
            "idx": i+1,
            "date": e["date"].date(),
            "predicted_regime": e["predicted_regime"],
            "used_fallback": e["used_fallback"],
            "k_keep": e["k_keep"],
            "forecast_return": e["forecast_port_mu"],
            "forecast_vol": e["forecast_port_vol"],
            "forecast_return_naive": e["forecast_port_mu_naive"],
            "forecast_vol_naive": e["forecast_port_vol_naive"],
            "realized_return_smart": e.get("realized_return_smart"),
            "realized_vol_smart": e.get("realized_vol_smart"),
            "realized_return_naive": e.get("realized_return_naive"),
            "realized_vol_naive": e.get("realized_vol_naive"),
            "sharpe_smart": e.get("sharpe_smart"),
            "sharpe_naive": e.get("sharpe_naive")
        })
    summary_df = pd.DataFrame(rows).set_index("idx")
    st.dataframe(summary_df.style.format({
        "forecast_return": "{:.2%}",
        "forecast_vol": "{:.2%}",
        "forecast_return_naive": "{:.2%}",
        "forecast_vol_naive": "{:.2%}",
        "realized_return_smart": lambda x: f"{x:.2%}" if pd.notnull(x) else "",
        "realized_vol_smart": lambda x: f"{x:.2%}" if pd.notnull(x) else "",
        "realized_return_naive": lambda x: f"{x:.2%}" if pd.notnull(x) else "",
        "realized_vol_naive": lambda x: f"{x:.2%}" if pd.notnull(x) else "",
        "sharpe_smart": "{:.3f}",
        "sharpe_naive": "{:.3f}"
    }))

    # Regime probabilities display for last rebalance
    st.markdown("### Last predicted regime & posterior probabilities")
    last = st.session_state.history[-1]
    probs_map = last.get("probs_map", {})
    probs_df = pd.DataFrame.from_dict(probs_map, orient="index", columns=["probability"])
    probs_df = probs_df.sort_values("probability", ascending=False)
    st.table(probs_df.style.format({"probability": "{:.2%}"}))

    # bar chart of probs
    st.bar_chart(probs_df["probability"])

    # Portfolio performance line chart (realized segments per rebalance) vs 1/N
    st.markdown("### Portfolio performance per rebalance (realized periods) — Smart vs 1/N")
    perf_df = pd.DataFrame()
    for i, e in enumerate(st.session_state.history):
        prices = e.get("prices")
        if prices is None or prices.shape[0] == 0:
            continue
        start_dt = pd.to_datetime(e["date"])
        # end date = next rebalance date if exists else last available price
        if i + 1 < len(st.session_state.history):
            end_dt = pd.to_datetime(st.session_state.history[i+1]["date"])
        else:
            end_dt = prices.index[-1]
        returns_df = compute_log_returns(prices)
        interval = returns_df.loc[(returns_df.index > start_dt) & (returns_df.index <= end_dt)]
        if interval.shape[0] == 0:
            continue
        w_smart = e["weights_smart"].reindex(interval.columns).fillna(0.0)
        w_naive = e["weights_naive"].reindex(interval.columns).fillna(0.0)
        smart_daily = interval.dot(w_smart)
        naive_daily = interval.dot(w_naive)
        smart_cum = (np.exp(smart_daily).cumprod() - 1)  # cumulative simple return
        naive_cum = (np.exp(naive_daily).cumprod() - 1)
        perf_df = pd.concat([perf_df, pd.DataFrame({
            f"Smart_{i+1}": smart_cum,
            f"Naive_{i+1}": naive_cum
        })], axis=1)
    if perf_df.shape[1] > 0:
        st.line_chart(perf_df.fillna(method='ffill').fillna(0))
    else:
        st.info("No realized performance periods yet. Advance time after a rebalance to get realized P&L.")

    # Pearson correlations between mu_regime (predicted mu) vs actual asset realized returns in the realized window
    st.markdown("### Pearson correlation: predicted μ (per-asset) vs actual realized asset returns (per rebalance)")
    pearson_rows = []
    for i, e in enumerate(st.session_state.history):
        if e.get("realized_return_smart") is None:
            continue
        prices = e.get("prices")
        if prices is None:
            continue
        reb_date = pd.to_datetime(e["date"])
        returns_all = compute_log_returns(prices)
        realized_window = returns_all.loc[(returns_all.index > reb_date)]
        if realized_window.shape[0] == 0:
            continue
        actual_asset_log = realized_window.sum()
        actual_asset_simple = np.expm1(actual_asset_log)
        mu_series = e.get("mu_annual")
        if mu_series is None:
            continue
        common = actual_asset_simple.index.intersection(mu_series.index)
        if len(common) < 2:
            continue
        try:
            r, p = pearsonr(mu_series.reindex(common).values, actual_asset_simple.reindex(common).values)
        except Exception:
            r, p = np.nan, np.nan
        pearson_rows.append({"rebalance": i+1, "date": e["date"], "pearson_mu_vs_actual": r})
    if len(pearson_rows) > 0:
        pearson_df = pd.DataFrame(pearson_rows).set_index("rebalance")
        st.dataframe(pearson_df.style.format({"pearson_mu_vs_actual": "{:.3f}"}))
    else:
        st.info("No realized data available to compute Pearson correlations yet.")

    # Show last allocation table
    st.markdown("### Last allocation (Smart vs 1/N)")
    last = st.session_state.history[-1]
    alloc_df = pd.DataFrame({
        "Smart_weight": last["weights_smart"],
        "Smart_alloc": last["allocation_smart"],
        "Naive_weight": last["weights_naive"],
        "Naive_alloc": last["allocation_naive"]
    }).fillna(0.0)
    st.dataframe(alloc_df.style.format({
        "Smart_weight": "{:.4f}",
        "Smart_alloc": "{:,.2f}",
        "Naive_weight": "{:.4f}",
        "Naive_alloc": "{:,.2f}"
    }))

# ------------------------
# Notes & requirements
# ------------------------
st.markdown("---")
st.markdown("### Notes & assumptions")
st.markdown("""
- **GMM features:** rolling μ and σ of daily S&P 500 log returns (rolling window configurable).  
- **GMM k=3:** components labeled by sorting the component means of μ: lowest→Bear, middle→Calm, highest→Bull.  
- **Regime selection:** filter N-stock returns to dates where SPX rolling (μ,σ) belonged to predicted component. If filtered sample < 30 observations, we **use all stock data** as fallback.  
- **Covariance denoising:** PCA with hybrid rule — keep `max(#eigenvalues > λ+ (MP), #components for 90% variance)`. λ+ is computed with Marchenko-Pastur using average variance.  
- **Optimization:** maximize `w^T μ - (γ/2) w^T Σ w` with `sum(w)=1` and `w>=0`. `γ` is set equal to the Risk Appetite slider (1 aggressive ... 10 conservative).  
- **Simulation:** Advance time by month/quarter uses realized returns from yfinance price history to update the budget.  
- All returns are computed using **log returns** internally; displayed cumulative returns are converted to simple returns for interpretability.
""")

st.markdown("### Requirements")
st.code("""
pip install streamlit pandas numpy yfinance scikit-learn scipy
""")
