"""
Livetest: simulate the combined stock+crypto live trading schedule on historical data.
Mimics run_live.py behavior — stock strategy during market hours, crypto outside.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from backtest import fetch_history, _compute_rsi
from crypto.strategies.beta_reversion import _compute_hurst


def _is_market_hours(ts):
    """Check if timestamp falls within US stock market hours (9:30-16:00 ET)."""
    # Convert to ET if timezone-aware, otherwise assume ET
    if ts.tzinfo is not None:
        import pytz
        et = pytz.timezone("US/Eastern")
        ts_et = ts.astimezone(et)
    else:
        ts_et = ts

    if ts_et.weekday() >= 5:  # Saturday/Sunday
        return False

    market_open = ts_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = ts_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= ts_et < market_close


def _load_strategy_config(market, name):
    """Load optimized params for a strategy from its params file."""
    from main import STOCK_STRATEGIES, CRYPTO_STRATEGIES

    strategies = CRYPTO_STRATEGIES if market == "crypto" else STOCK_STRATEGIES
    cfg = strategies[name]
    kwargs = dict(cfg["kwargs"])
    rebalance_every = 30

    params_file = Path(__file__).parent / market / "params" / f"{name}.json"
    try:
        with open(params_file) as f:
            params = json.load(f)
        rebalance_every = params.pop("rebalance_every", 30)
        params.pop("updated_at", None)
        kwargs.update(params)
    except FileNotFoundError:
        pass

    return cfg["class"], kwargs, rebalance_every


def _score_mean_reversion(closes, volumes, ts, kwargs):
    """Score symbols using mean reversion logic at a single timestamp."""
    vwap_window = kwargs.get("vwap_window", 60)
    min_dip = kwargs.get("min_dip", 0.003)
    max_dip = kwargs.get("max_dip", 0.015)
    top_n = kwargs.get("top_n", 3)
    trend_window = kwargs.get("trend_window", 120)

    # Get data window ending at ts
    loc = closes.index.get_loc(ts)
    window = max(vwap_window, trend_window) + 10
    if loc < window:
        return []

    c = closes.iloc[loc - window:loc + 1]
    v = volumes.iloc[loc - window:loc + 1] if volumes is not None else None

    if v is not None:
        roll_cv = (c * v).rolling(vwap_window, min_periods=10).sum()
        roll_v = v.rolling(vwap_window, min_periods=10).sum()
        vwap = roll_cv / roll_v.replace(0, np.nan)
        vwap = vwap.fillna(c.rolling(vwap_window, min_periods=10).mean())
    else:
        vwap = c.rolling(vwap_window, min_periods=10).mean()

    dips = (c - vwap) / vwap
    trend_sma = c.rolling(trend_window, min_periods=20).mean()

    winners = []
    for sym in c.columns:
        t = trend_sma.iloc[-1][sym]
        if np.isnan(t) or c.iloc[-1][sym] < t:
            continue
        d = dips.iloc[-1][sym]
        if np.isnan(d):
            continue
        if -max_dip <= d <= -min_dip:
            winners.append((sym, d))

    winners.sort(key=lambda x: x[1])
    return [sym for sym, _ in winners[:top_n]]


def _score_bayesian(closes, volumes, ts, kwargs):
    """Score symbols using Bayesian logic at a single timestamp."""
    sma_period = kwargs.get("sma_period", 20)
    rsi_period = kwargs.get("rsi_period", 14)
    momentum_weight = kwargs.get("momentum_weight", 1.0)
    volume_weight = kwargs.get("volume_weight", 0.5)
    rsi_weight = kwargs.get("rsi_weight", 0.5)
    threshold = kwargs.get("threshold", 0.6)
    trend_window = kwargs.get("trend_window", 120)
    top_n = kwargs.get("top_n", 2)

    loc = closes.index.get_loc(ts)
    window = max(sma_period, rsi_period, trend_window) + 20
    if loc < window:
        return []

    c = closes.iloc[loc - window:loc + 1]
    v = volumes.iloc[loc - window:loc + 1] if volumes is not None else None

    sma = c.rolling(sma_period, min_periods=10).mean()
    rsi = c.apply(lambda col: _compute_rsi(col, rsi_period))
    trend_sma = c.rolling(trend_window, min_periods=20).mean()
    vol_avg = v.rolling(20, min_periods=10).mean() if v is not None else None

    posteriors = {}
    for sym in c.columns:
        t = trend_sma.iloc[-1][sym]
        if np.isnan(t) or c.iloc[-1][sym] < t:
            continue
        sma_val = sma.iloc[-1][sym]
        price = c.iloc[-1][sym]
        if np.isnan(sma_val) or sma_val <= 0:
            continue
        log_odds = 0.0
        momentum_signal = (price - sma_val) / sma_val
        log_odds += momentum_weight * momentum_signal * 10
        if vol_avg is not None and v is not None:
            vv = v.iloc[-1][sym]
            va = vol_avg.iloc[-1][sym]
            if va > 0 and not np.isnan(va):
                log_odds += volume_weight * ((vv - va) / va)
        rsi_val = rsi.iloc[-1][sym]
        if np.isnan(rsi_val):
            continue
        log_odds += rsi_weight * ((rsi_val - 50) / 20.0)
        posterior = 1.0 / (1.0 + np.exp(-log_odds))
        if posterior >= threshold:
            posteriors[sym] = posterior

    winners = sorted(posteriors, key=posteriors.get, reverse=True)[:top_n]
    return winners


def _score_momentum(closes, volumes, ts, kwargs):
    """Score symbols using stock momentum logic."""
    lookback = kwargs.get("lookback_minutes", 30)
    top_n = kwargs.get("top_n", 5)
    trend_window = kwargs.get("trend_window", 120)

    loc = closes.index.get_loc(ts)
    window = max(lookback, trend_window) + 20
    if loc < window:
        return []

    c = closes.iloc[loc - window:loc + 1]
    trend_sma = c.rolling(trend_window, min_periods=20).mean()

    scores = {}
    for sym in c.columns:
        t = trend_sma.iloc[-1][sym]
        price = c.iloc[-1][sym]
        if np.isnan(t) or price < t:
            continue
        past = c.iloc[-1 - lookback][sym]
        if np.isnan(past) or past <= 0:
            continue
        ret = (price - past) / past
        scores[sym] = ret

    winners = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [w for w in winners if scores[w] > 0]


def _score_crypto_momentum(closes, volumes, ts, kwargs):
    """Score symbols using crypto momentum logic."""
    rsi_period = kwargs.get("rsi_period", 14)
    rsi_threshold = kwargs.get("rsi_threshold", 53)
    bb_period = kwargs.get("bb_period", 20)
    bb_std_val = kwargs.get("bb_std", 2.0)
    volume_mult = kwargs.get("volume_mult", 1.3)
    top_n = kwargs.get("top_n", 3)
    trend_window = kwargs.get("trend_window", 120)

    loc = closes.index.get_loc(ts)
    window = max(rsi_period, bb_period, trend_window) + 20
    if loc < window:
        return []

    c = closes.iloc[loc - window:loc + 1]
    v = volumes.iloc[loc - window:loc + 1] if volumes is not None else None

    sma = c.rolling(bb_period, min_periods=10).mean()
    std = c.rolling(bb_period, min_periods=10).std()
    width = ((sma + bb_std_val * std) - (sma - bb_std_val * std)) / sma
    width_expanding = width.iloc[-1] > width.rolling(5, min_periods=3).mean().iloc[-1]
    in_upper_half = c.iloc[-1] > sma.iloc[-1]
    rsi = c.apply(lambda col: _compute_rsi(col, rsi_period))
    trend_sma = c.rolling(trend_window, min_periods=20).mean()
    vol_avg = v.rolling(20, min_periods=10).mean() if v is not None else None

    scores = {}
    for sym in c.columns:
        t = trend_sma.iloc[-1][sym]
        if np.isnan(t) or c.iloc[-1][sym] < t:
            continue
        r = rsi.iloc[-1][sym]
        if np.isnan(r) or r < rsi_threshold:
            continue
        if not width_expanding[sym]:
            continue
        if not in_upper_half[sym]:
            continue
        if vol_avg is not None and v is not None:
            vv = v.iloc[-1][sym]
            va = vol_avg.iloc[-1][sym]
            if va > 0 and not np.isnan(va) and vv < volume_mult * va:
                continue
        scores[sym] = r

    winners = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return winners


def _score_rsi_mean_revert(closes, volumes, ts, kwargs):
    """Score symbols using RSI mean reversion logic."""
    rsi_period = kwargs.get("rsi_period", 14)
    rsi_oversold = kwargs.get("rsi_oversold", 30)
    bb_period = kwargs.get("bb_period", 20)
    bb_std_val = kwargs.get("bb_std", 2.0)
    trend_window = kwargs.get("trend_window", 120)
    top_n = kwargs.get("top_n", 2)
    use_bb = kwargs.get("use_bb", 1)
    adaptive_rsi = kwargs.get("adaptive_rsi", 0)

    loc = closes.index.get_loc(ts)
    window = max(rsi_period, bb_period, trend_window) + 20
    if loc < window:
        return []

    c = closes.iloc[loc - window:loc + 1]

    rsi = c.apply(lambda col: _compute_rsi(col, rsi_period))
    trend_sma = c.rolling(trend_window, min_periods=20).mean()
    sma = c.rolling(bb_period, min_periods=10).mean()
    std = c.rolling(bb_period, min_periods=10).std()
    lower_bb = sma - bb_std_val * std

    scores = {}
    for sym in c.columns:
        t = trend_sma.iloc[-1][sym]
        if np.isnan(t) or c.iloc[-1][sym] < t:
            continue
        r = rsi.iloc[-1][sym]
        if np.isnan(r):
            continue

        threshold = rsi_oversold
        if adaptive_rsi:
            atr = c[sym].diff().abs().rolling(14, min_periods=7).mean().iloc[-1]
            if not np.isnan(atr) and c.iloc[-1][sym] > 0:
                vol_ratio = atr / c.iloc[-1][sym]
                threshold = rsi_oversold + vol_ratio * 500

        if r > threshold:
            continue

        if use_bb and c.iloc[-1][sym] > lower_bb.iloc[-1][sym]:
            continue

        scores[sym] = -r  # lower RSI = better score

    winners = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return winners


def _score_breakout(closes, volumes, ts, kwargs):
    """Score symbols using breakout logic."""
    channel_period = kwargs.get("channel_period", 40)
    volume_mult = kwargs.get("volume_mult", 1.2)
    trend_window = kwargs.get("trend_window", 120)
    top_n = kwargs.get("top_n", 2)

    loc = closes.index.get_loc(ts)
    window = max(channel_period, trend_window) + 20
    if loc < window:
        return []

    c = closes.iloc[loc - window:loc + 1]
    v = volumes.iloc[loc - window:loc + 1] if volumes is not None else None

    trend_sma = c.rolling(trend_window, min_periods=20).mean()
    channel_high = c.rolling(channel_period, min_periods=10).max()
    vol_avg = v.rolling(20, min_periods=10).mean() if v is not None else None

    scores = {}
    for sym in c.columns:
        t = trend_sma.iloc[-1][sym]
        if np.isnan(t) or c.iloc[-1][sym] < t:
            continue
        ch = channel_high.iloc[-2][sym] if len(c) > 1 else np.nan
        if np.isnan(ch) or c.iloc[-1][sym] <= ch:
            continue
        if vol_avg is not None and v is not None:
            vv = v.iloc[-1][sym]
            va = vol_avg.iloc[-1][sym]
            if va > 0 and not np.isnan(va) and vv < volume_mult * va:
                continue
        scores[sym] = c.iloc[-1][sym] / ch

    winners = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return winners


def _score_ema_crossover(closes, volumes, ts, kwargs):
    """Score symbols using EMA crossover logic."""
    fast_period = kwargs.get("fast_period", 10)
    slow_period = kwargs.get("slow_period", 60)
    trend_window = kwargs.get("trend_window", 240)
    top_n = kwargs.get("top_n", 2)

    loc = closes.index.get_loc(ts)
    window = max(fast_period, slow_period, trend_window) + 20
    if loc < window:
        return []

    c = closes.iloc[loc - window:loc + 1]
    fast_ema = c.ewm(span=fast_period, min_periods=fast_period).mean()
    slow_ema = c.ewm(span=slow_period, min_periods=slow_period).mean()
    trend_sma = c.rolling(trend_window, min_periods=20).mean()

    scores = {}
    for sym in c.columns:
        t = trend_sma.iloc[-1][sym]
        if np.isnan(t) or c.iloc[-1][sym] < t:
            continue
        fe = fast_ema.iloc[-1][sym]
        se = slow_ema.iloc[-1][sym]
        if np.isnan(fe) or np.isnan(se) or se <= 0:
            continue
        if fe > se:
            scores[sym] = (fe - se) / se

    winners = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return winners


def _score_bandit(closes, ts, kwargs):
    """Score symbols using bandit logic."""
    reward_window = kwargs.get("reward_window", 60)
    exploration_factor = kwargs.get("exploration_factor", 1.0)
    return_threshold = kwargs.get("return_threshold", 0.001)
    trend_window = kwargs.get("trend_window", 120)
    top_n = kwargs.get("top_n", 2)

    loc = closes.index.get_loc(ts)
    window = max(reward_window, trend_window) + 20
    if loc < window:
        return []

    c = closes.iloc[loc - window:loc + 1]
    trend_sma = c.rolling(trend_window, min_periods=20).mean()

    scores = {}
    for sym in c.columns:
        t = trend_sma.iloc[-1][sym]
        if np.isnan(t) or c.iloc[-1][sym] < t:
            continue
        rets = c[sym].pct_change(reward_window).iloc[-1]
        if np.isnan(rets) or rets < return_threshold:
            continue
        vol = c[sym].pct_change().rolling(reward_window).std().iloc[-1]
        if np.isnan(vol) or vol <= 0:
            continue
        scores[sym] = rets + exploration_factor * vol

    winners = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return winners


def _score_monte_carlo(closes, volumes, ts, kwargs):
    """Score symbols using Monte Carlo logic."""
    lookback = kwargs.get("lookback", 60)
    n_simulations = kwargs.get("n_simulations", 100)
    horizon = kwargs.get("horizon", 10)
    volume_weight = kwargs.get("volume_weight", 0.5)
    trend_window = kwargs.get("trend_window", 120)
    top_n = kwargs.get("top_n", 2)

    loc = closes.index.get_loc(ts)
    window = max(lookback, trend_window) + 20
    if loc < window:
        return []

    c = closes.iloc[loc - window:loc + 1]
    v = volumes.iloc[loc - window:loc + 1] if volumes is not None else None
    trend_sma = c.rolling(trend_window, min_periods=20).mean()

    scores = {}
    for sym in c.columns:
        t = trend_sma.iloc[-1][sym]
        if np.isnan(t) or c.iloc[-1][sym] < t:
            continue
        rets = c[sym].pct_change().dropna().iloc[-lookback:]
        if len(rets) < lookback // 2:
            continue
        mu = rets.mean()
        sigma = rets.std()
        if sigma <= 0:
            continue
        sims = np.random.normal(mu, sigma, (n_simulations, horizon))
        final_returns = np.exp(sims.sum(axis=1)) - 1
        expected = final_returns.mean()
        if expected <= 0:
            continue
        vol_score = 0
        if v is not None and volume_weight > 0:
            vv = v.iloc[-1][sym]
            va = v[sym].rolling(20, min_periods=10).mean().iloc[-1]
            if va > 0 and not np.isnan(va):
                vol_score = volume_weight * ((vv - va) / va)
        scores[sym] = expected + vol_score

    winners = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return winners


def _score_regime_mr(closes, volumes, ts, kwargs):
    """Score symbols using regime mean reversion logic."""
    vwap_window = kwargs.get("vwap_window", 60)
    min_dip = kwargs.get("min_dip", 0.005)
    max_dip = kwargs.get("max_dip", 0.025)
    top_n = kwargs.get("top_n", 1)
    trend_window = kwargs.get("trend_window", 240)
    regime_window = kwargs.get("regime_window", 720)

    loc = closes.index.get_loc(ts)
    window = max(vwap_window, trend_window, regime_window) + 20
    if loc < window:
        return []

    c = closes.iloc[loc - window:loc + 1]
    v = volumes.iloc[loc - window:loc + 1] if volumes is not None else None

    if v is not None:
        roll_cv = (c * v).rolling(vwap_window, min_periods=10).sum()
        roll_v = v.rolling(vwap_window, min_periods=10).sum()
        vwap = roll_cv / roll_v.replace(0, np.nan)
        vwap = vwap.fillna(c.rolling(vwap_window, min_periods=10).mean())
    else:
        vwap = c.rolling(vwap_window, min_periods=10).mean()

    dips = (c - vwap) / vwap
    trend_sma = c.rolling(trend_window, min_periods=20).mean()
    regime_sma = c.rolling(regime_window, min_periods=100).mean()

    winners = []
    for sym in c.columns:
        t = trend_sma.iloc[-1][sym]
        if np.isnan(t) or c.iloc[-1][sym] < t:
            continue
        rs = regime_sma.iloc[-1][sym]
        if np.isnan(rs) or c.iloc[-1][sym] < rs:
            continue
        d = dips.iloc[-1][sym]
        if np.isnan(d):
            continue
        if -max_dip <= d <= -min_dip:
            winners.append((sym, d))

    winners.sort(key=lambda x: x[1])
    return [sym for sym, _ in winners[:top_n]]


def _score_beta_reversion(closes, volumes, ts, kwargs):
    """Score symbols using beta-adjusted z-score mean reversion."""
    lookback = kwargs.get("lookback", 90)
    beta_window = kwargs.get("beta_window", 240)
    z_window = kwargs.get("z_window", 120)
    z_entry = kwargs.get("z_entry", 1.5)
    hurst_window = kwargs.get("hurst_window", 240)
    hurst_threshold = kwargs.get("hurst_threshold", 0.50)
    trend_window = kwargs.get("trend_window", 240)
    top_n = kwargs.get("top_n", 1)

    loc = closes.index.get_loc(ts)
    window = max(beta_window, z_window + lookback, hurst_window, trend_window) + 20
    if loc < window:
        return []

    c = closes.iloc[loc - window:loc + 1]
    rets = c.pct_change().fillna(0)

    # Find BTC column
    btc_sym = None
    for col in c.columns:
        if "BTC" in col:
            btc_sym = col
            break
    if btc_sym is None:
        return []

    # Hurst gate on BTC
    btc_prices = c[btc_sym].dropna().iloc[-hurst_window:]
    if len(btc_prices) < 64:
        return []
    hurst = _compute_hurst(btc_prices)
    if hurst > hurst_threshold:
        return []

    btc_rets = rets[btc_sym]
    btc_var = btc_rets.iloc[-beta_window:].var()
    if btc_var <= 0:
        return []
    btc_cum = btc_rets.iloc[-lookback:].sum()
    trend_sma = c.rolling(trend_window, min_periods=20).mean()

    scores = {}
    for sym in c.columns:
        if sym == btc_sym:
            continue
        if c[sym].iloc[-1] < trend_sma[sym].iloc[-1]:
            continue
        # Beta = Cov(r_i, r_BTC) / Var(r_BTC)
        sym_rets = rets[sym].iloc[-beta_window:]
        btc_w = btc_rets.iloc[-beta_window:]
        cov = ((sym_rets - sym_rets.mean()) * (btc_w - btc_w.mean())).mean()
        beta = cov / btc_var
        # Residual = R_i - beta * R_BTC
        cum_ret = rets[sym].iloc[-lookback:].sum()
        residual = cum_ret - beta * btc_cum
        # Z-score over z_window
        residuals = []
        for j in range(z_window):
            idx = -(z_window - j)
            if abs(idx) > len(rets) - lookback:
                continue
            r = rets[sym].iloc[idx-lookback:idx].sum()
            b = btc_rets.iloc[idx-lookback:idx].sum()
            residuals.append(r - beta * b)
        if len(residuals) < 20:
            continue
        mu = np.mean(residuals)
        sigma = np.std(residuals)
        if sigma <= 0:
            continue
        z = (residual - mu) / sigma
        if z < -z_entry:
            scores[sym] = z

    winners = sorted(scores, key=scores.get)[:top_n]
    return winners


# Map strategy class names to scoring functions
SCORERS = {
    "MeanReversionStrategy": _score_mean_reversion,
    "CryptoMomentumStrategy": _score_crypto_momentum,
    "CryptoBreakoutStrategy": _score_breakout,
    "RsiMeanRevertStrategy": _score_rsi_mean_revert,
    "BayesianStrategy": _score_bayesian,
    "BanditStrategy": lambda c, v, ts, kw: _score_bandit(c, ts, kw),
    "MonteCarloStrategy": _score_monte_carlo,
    "RegimeMeanReversionStrategy": _score_regime_mr,
    "MomentumStrategy": _score_momentum,
    "BreakoutStrategy": _score_breakout,
    "EmaCrossoverStrategy": _score_ema_crossover,
    "BetaReversionStrategy": _score_beta_reversion,
}


def cmd_livetest(api, days=365):
    """Simulate the combined stock+crypto live trading schedule on historical data."""
    from main import STOCK_SYMBOLS, CRYPTO_SYMBOLS

    # Load best strategies
    best_file = Path(__file__).parent / "best_strategy.json"
    with open(best_file) as f:
        best = json.load(f)

    stock_name = best.get("stock", {}).get("strategy")
    crypto_name = best.get("crypto", {}).get("strategy")

    if not stock_name and not crypto_name:
        print("No strategies in best_strategy.json. Run 'python main.py refresh' first.")
        return

    # Load strategy configs
    stock_cls, stock_kwargs, stock_interval = (None, None, None)
    crypto_cls, crypto_kwargs, crypto_interval = (None, None, None)

    if stock_name:
        stock_cls, stock_kwargs, stock_interval = _load_strategy_config("stock", stock_name)
        print(f"Stock strategy:  {stock_name} (every {stock_interval}min)")
    if crypto_name:
        crypto_cls, crypto_kwargs, crypto_interval = _load_strategy_config("crypto", crypto_name)
        print(f"Crypto strategy: {crypto_name} (every {crypto_interval}min)")

    # Fetch historical data for both markets
    print(f"\nFetching {days} days of historical data...")

    stock_history = {}
    crypto_history = {}
    if stock_name:
        print("\nStock symbols:")
        stock_history = fetch_history(api, STOCK_SYMBOLS, days)
    if crypto_name:
        print("\nCrypto symbols:")
        crypto_history = fetch_history(api, CRYPTO_SYMBOLS, days)

    if not stock_history and not crypto_history:
        print("No data fetched.")
        return

    # Build closes/volumes DataFrames
    stock_closes = stock_volumes = None
    crypto_closes = crypto_volumes = None

    if stock_history:
        stock_closes = pd.DataFrame({sym: df["close"] for sym, df in stock_history.items()})
        stock_closes = stock_closes.dropna(how="all").ffill()
        stock_volumes = pd.DataFrame({sym: df["volume"] for sym, df in stock_history.items() if "volume" in df.columns})
        stock_volumes = stock_volumes.reindex(stock_closes.index).fillna(0) if not stock_volumes.empty else None

    if crypto_history:
        crypto_closes = pd.DataFrame({sym: df["close"] for sym, df in crypto_history.items()})
        crypto_closes = crypto_closes.dropna(how="all").ffill()
        crypto_volumes = pd.DataFrame({sym: df["volume"] for sym, df in crypto_history.items() if "volume" in df.columns})
        crypto_volumes = crypto_volumes.reindex(crypto_closes.index).fillna(0) if not crypto_volumes.empty else None

    # Build unified timeline from all available data
    all_indices = []
    if stock_closes is not None:
        all_indices.append(stock_closes.index)
    if crypto_closes is not None:
        all_indices.append(crypto_closes.index)
    unified_index = all_indices[0]
    for idx in all_indices[1:]:
        unified_index = unified_index.union(idx)
    unified_index = unified_index.sort_values()

    # Simulation state
    initial_cash = 100_000.0
    cash = initial_cash
    holdings = {}  # {symbol: qty}
    current_market = None

    # Tracking
    equity_curve = []
    timestamps = []
    market_switches = []
    stock_minutes = 0
    crypto_minutes = 0
    idle_periods = []
    daily_equity = {}  # date -> (start, end)
    last_rebalance_idx = -9999

    stock_scorer = SCORERS.get(stock_cls.__name__) if stock_cls else None
    crypto_scorer = SCORERS.get(crypto_cls.__name__) if crypto_cls else None

    # Pre-build index sets for O(1) membership checks
    stock_index_set = set(stock_closes.index) if stock_closes is not None else set()
    crypto_index_set = set(crypto_closes.index) if crypto_closes is not None else set()

    def _port_value(cash, holdings, closes_df, index_set, ts):
        val = cash
        for sym, qty in holdings.items():
            if sym in closes_df.columns and ts in index_set:
                p = closes_df.loc[ts, sym]
                if not np.isnan(p):
                    val += qty * p
        return val

    print(f"\nSimulating {len(unified_index)} minutes over {days} days...")
    print(f"{'='*60}")

    for i, ts in enumerate(unified_index):
        market_open = _is_market_hours(ts)

        # Determine which market/strategy to use
        if market_open and stock_name and stock_closes is not None:
            active_market = "stock"
            active_name = stock_name
            active_closes = stock_closes
            active_volumes = stock_volumes
            active_kwargs = stock_kwargs
            active_interval = stock_interval
            active_scorer = stock_scorer
        elif crypto_name and crypto_closes is not None:
            active_market = "crypto"
            active_name = crypto_name
            active_closes = crypto_closes
            active_volumes = crypto_volumes
            active_kwargs = crypto_kwargs
            active_interval = crypto_interval
            active_scorer = crypto_scorer
        elif stock_name and stock_closes is not None:
            # Fallback to stock even outside hours if no crypto
            active_market = "stock"
            active_name = stock_name
            active_closes = stock_closes
            active_volumes = stock_volumes
            active_kwargs = stock_kwargs
            active_interval = stock_interval
            active_scorer = stock_scorer
        else:
            continue

        # Track market time
        if active_market == "stock":
            stock_minutes += 1
        else:
            crypto_minutes += 1

        # Detect market switch — liquidate and switch
        if active_market != current_market and current_market is not None:
            # Liquidate current holdings
            for sym, qty in holdings.items():
                # Find price in the appropriate closes DataFrame
                if current_market == "stock" and stock_closes is not None and sym in stock_closes.columns:
                    if ts in stock_index_set:
                        p = stock_closes.loc[ts, sym]
                    else:
                        valid = stock_closes.index[stock_closes.index <= ts]
                        p = stock_closes.loc[valid[-1], sym] if len(valid) > 0 else np.nan
                elif current_market == "crypto" and crypto_closes is not None and sym in crypto_closes.columns:
                    if ts in crypto_index_set:
                        p = crypto_closes.loc[ts, sym]
                    else:
                        valid = crypto_closes.index[crypto_closes.index <= ts]
                        p = crypto_closes.loc[valid[-1], sym] if len(valid) > 0 else np.nan
                else:
                    p = np.nan
                if not np.isnan(p):
                    cash += qty * p
            holdings = {}
            market_switches.append((ts, current_market, active_market))
            last_rebalance_idx = -9999  # Force rebalance on switch

        current_market = active_market

        # Check if it's time to rebalance
        active_index_set = stock_index_set if active_market == "stock" else crypto_index_set
        if ts not in active_index_set:
            continue

        bars_since = i - last_rebalance_idx
        if bars_since < active_interval:
            if i % 60 == 0:
                port_value = _port_value(cash, holdings, active_closes, active_index_set, ts)
                equity_curve.append(port_value)
                timestamps.append(ts)
                day = ts.date() if hasattr(ts, 'date') else pd.Timestamp(ts).date()
                if day not in daily_equity:
                    daily_equity[day] = [port_value, port_value]
                daily_equity[day][1] = port_value
            continue

        last_rebalance_idx = i

        port_value = _port_value(cash, holdings, active_closes, active_index_set, ts)
        equity_curve.append(port_value)
        timestamps.append(ts)

        day = ts.date() if hasattr(ts, 'date') else pd.Timestamp(ts).date()
        if day not in daily_equity:
            daily_equity[day] = [port_value, port_value]
        daily_equity[day][1] = port_value

        # Score and pick winners
        if active_scorer is None:
            continue

        winners = active_scorer(active_closes, active_volumes, ts, active_kwargs)

        # Liquidate current holdings
        for sym, qty in holdings.items():
            if sym in active_closes.columns and ts in active_index_set:
                p = active_closes.loc[ts, sym]
                if not np.isnan(p):
                    cash += qty * p
        holdings = {}

        if not winners:
            idle_periods.append(ts)
            continue

        # Buy winners equally weighted
        per_stock = cash / len(winners)
        for sym in winners:
            if sym in active_closes.columns and ts in active_index_set:
                p = active_closes.loc[ts, sym]
                if not np.isnan(p) and p > 0:
                    qty = per_stock / p
                    holdings[sym] = qty
                    cash -= qty * p

    # Final liquidation
    if holdings:
        last_ts = unified_index[-1]
        for sym, qty in holdings.items():
            p = np.nan
            if stock_closes is not None and sym in stock_closes.columns:
                valid = stock_closes.index[stock_closes.index <= last_ts]
                if len(valid) > 0:
                    p = stock_closes.loc[valid[-1], sym]
            if np.isnan(p) and crypto_closes is not None and sym in crypto_closes.columns:
                valid = crypto_closes.index[crypto_closes.index <= last_ts]
                if len(valid) > 0:
                    p = crypto_closes.loc[valid[-1], sym]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}

    final_equity = cash
    equity_curve.append(final_equity)

    # Results
    pv = np.array(equity_curve)
    total_return = (final_equity - initial_cash) / initial_cash
    max_dd = ((pv - np.maximum.accumulate(pv)) / np.maximum.accumulate(pv)).min()

    print(f"\n{'='*60}")
    print(f"  LIVETEST RESULTS ({days} days)")
    print(f"{'='*60}")
    print(f"  Initial equity:  ${initial_cash:,.2f}")
    print(f"  Final equity:    ${final_equity:,.2f}")
    print(f"  Total return:    {total_return:+.2%}")
    print(f"  Max drawdown:    {max_dd:.2%}")
    print(f"{'='*60}")

    print(f"\n  Strategy allocation:")
    total_minutes = stock_minutes + crypto_minutes
    if total_minutes > 0:
        if stock_name:
            print(f"    Stock ({stock_name}):  {stock_minutes:,} min ({stock_minutes/total_minutes:.1%})")
        if crypto_name:
            print(f"    Crypto ({crypto_name}): {crypto_minutes:,} min ({crypto_minutes/total_minutes:.1%})")

    print(f"\n  Market switches: {len(market_switches)}")
    print(f"  Idle rebalances (no picks): {len(idle_periods)}")

    # Daily P&L breakdown
    if daily_equity:
        sorted_days = sorted(daily_equity.keys())
        print(f"\n  Daily P&L (showing first/last 10 days):")
        print(f"  {'Date':<12} {'Start':>12} {'End':>12} {'P&L':>10} {'Return':>8}")
        print(f"  {'-'*54}")

        show_days = sorted_days[:10] + sorted_days[-10:] if len(sorted_days) > 20 else sorted_days
        prev_end = initial_cash
        shown_ellipsis = False
        for day in sorted_days:
            start_eq, end_eq = daily_equity[day]
            day_pnl = end_eq - prev_end
            day_ret = day_pnl / prev_end if prev_end > 0 else 0
            if day in show_days:
                sign = "+" if day_pnl >= 0 else ""
                print(f"  {day}  ${start_eq:>10,.2f}  ${end_eq:>10,.2f}  {sign}${day_pnl:>8,.2f} {sign}{day_ret:>6.2%}")
            elif not shown_ellipsis:
                print(f"  {'... ':>12}({len(sorted_days) - 20} more days)")
                shown_ellipsis = True
            prev_end = end_eq

    print(f"\n{'='*60}")
