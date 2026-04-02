import os
import hashlib
import numpy as np
import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime, timedelta

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def fetch_history(api: tradeapi.REST, symbols: list[str], days: int, end_days_ago: int = 1) -> dict[str, pd.DataFrame]:
    """Fetch 1-minute bars, cached to disk per symbol per day."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    end = datetime.now() - timedelta(days=end_days_ago)
    start = end - timedelta(days=days + 5)

    history = {}
    for symbol in symbols:
        is_crypto = "/" in symbol or (symbol.endswith("USD") and len(symbol) <= 10)
        safe_sym = symbol.replace("/", "")

        # Fetch per-day, check cache for each day
        all_bars = []
        current = start
        while current <= end:
            day_str = current.strftime("%Y-%m-%d")
            cache_file = os.path.join(CACHE_DIR, f"{safe_sym}_{day_str}_1min.parquet")

            if os.path.exists(cache_file):
                bars = pd.read_parquet(cache_file)
            else:
                try:
                    next_day = (current + timedelta(days=1)).strftime("%Y-%m-%d")
                    if is_crypto:
                        bars = api.get_crypto_bars(
                            symbol, tradeapi.TimeFrame.Minute,
                            start=day_str, end=next_day,
                        ).df
                    else:
                        bars = api.get_bars(
                            symbol, tradeapi.TimeFrame.Minute,
                            start=day_str, end=next_day,
                        ).df
                    if not bars.empty:
                        bars.to_parquet(cache_file)
                except Exception as e:
                    current += timedelta(days=1)
                    continue

            if not bars.empty:
                all_bars.append(bars)
            current += timedelta(days=1)

        if all_bars:
            combined = pd.concat(all_bars)
            combined = combined[~combined.index.duplicated(keep='first')]
            history[symbol] = combined
            print(f"  {symbol}: {len(combined)} bars")
        else:
            print(f"  Skipping {symbol}: no data")

    return history


def run_backtest(
    strategy_cls,
    strategy_kwargs: dict,
    api: tradeapi.REST,
    symbols: list[str],
    days: int = 7,
    rebalance_every: int = 30,
    initial_cash: float = 100_000.0,
    end_days_ago: int = 1,
):
    """
    Simulate a strategy over historical 1-min bars using vectorized ops.
    """
    cls_name = strategy_cls.__name__
    print(f"Fetching {days} days of 1-min bars for {len(symbols)} symbols...")
    history = fetch_history(api, symbols, days, end_days_ago=end_days_ago)

    if not history:
        print("No data fetched.")
        return

    closes = pd.DataFrame({sym: df["close"] for sym, df in history.items()})
    closes = closes.dropna(how="all").ffill()

    highs = pd.DataFrame({sym: df["high"] for sym, df in history.items() if "high" in df.columns})
    highs = highs.reindex(closes.index).ffill() if not highs.empty else closes

    volumes = pd.DataFrame({sym: df["volume"] for sym, df in history.items() if "volume" in df.columns})
    volumes = volumes.reindex(closes.index).fillna(0) if not volumes.empty else None

    if cls_name == "MeanReversionStrategy":
        pv = _backtest_mean_reversion(closes, volumes, strategy_kwargs, rebalance_every, initial_cash)
    elif cls_name == "CryptoMomentumStrategy":
        pv = _backtest_crypto_momentum(closes, volumes, strategy_kwargs, rebalance_every, initial_cash)
    elif cls_name == "CryptoBreakoutStrategy":
        pv = _backtest_breakout(closes, highs, volumes, strategy_kwargs, rebalance_every, initial_cash)
    elif cls_name == "RsiMeanRevertStrategy":
        pv = _backtest_rsi_mean_revert(closes, volumes, strategy_kwargs, rebalance_every, initial_cash)
    elif cls_name == "BayesianStrategy":
        pv = _backtest_bayesian(closes, volumes, strategy_kwargs, rebalance_every, initial_cash)
    elif cls_name == "RegimeMeanReversionStrategy":
        pv = _backtest_regime_mr(closes, volumes, strategy_kwargs, rebalance_every, initial_cash)
    elif cls_name == "BanditStrategy":
        pv = _backtest_bandit(closes, strategy_kwargs, rebalance_every, initial_cash)
    elif cls_name == "MonteCarloStrategy":
        pv = _backtest_monte_carlo(closes, volumes, strategy_kwargs, rebalance_every, initial_cash)
    elif cls_name == "MomentumStrategy":
        pv = _backtest_stock_momentum(closes, volumes, strategy_kwargs, rebalance_every, initial_cash)
    elif cls_name == "BreakoutStrategy":
        pv = _backtest_breakout(closes, highs, volumes, strategy_kwargs, rebalance_every, initial_cash)
    elif cls_name == "EmaCrossoverStrategy":
        pv = _backtest_ema_crossover(closes, volumes, strategy_kwargs, rebalance_every, initial_cash)
    else:
        pv = _backtest_momentum(closes, strategy_kwargs, rebalance_every, initial_cash)

    if pv is None or len(pv) < 2:
        print("No trades simulated.")
        return None

    is_crypto = any("/" in s for s in symbols)
    return _print_results(pv, initial_cash, rebalance_every, cls_name, is_crypto)


def _compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _backtest_crypto_momentum(closes, volumes, kwargs, rebalance_every, initial_cash):
    rsi_period = kwargs.get("rsi_period", 14)
    rsi_threshold = kwargs.get("rsi_threshold", 55)
    bb_period = kwargs.get("bb_period", 20)
    bb_std = kwargs.get("bb_std", 1.5)
    volume_mult = kwargs.get("volume_mult", 1.0)
    top_n = kwargs.get("top_n", 3)
    trend_window = kwargs.get("trend_window", 120)

    # Precompute indicators
    rsi = closes.apply(lambda col: _compute_rsi(col, rsi_period))
    sma = closes.rolling(bb_period, min_periods=10).mean()
    std = closes.rolling(bb_period, min_periods=10).std()
    width = ((sma + bb_std * std) - (sma - bb_std * std)) / sma
    width_expanding = width > width.rolling(5, min_periods=3).mean()
    in_upper_half = closes > sma
    trend_sma = closes.rolling(trend_window, min_periods=20).mean()

    vol_avg = volumes.rolling(20, min_periods=10).mean() if volumes is not None else None

    warmup = max(rsi_period, bb_period, trend_window) + 10
    idx = closes.index[warmup:]
    rebal_points = idx[::rebalance_every]

    cash = initial_cash
    holdings = {}
    values = []

    for ts in rebal_points:
        port_value = cash
        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                port_value += qty * p
        values.append(port_value)

        scores = {}
        for sym in closes.columns:
            # Trend filter
            t = trend_sma.loc[ts, sym]
            if np.isnan(t) or closes.loc[ts, sym] < t:
                continue
            r = rsi.loc[ts, sym]
            if np.isnan(r) or r < rsi_threshold:
                continue
            if not in_upper_half.loc[ts, sym]:
                continue
            if not width_expanding.loc[ts, sym]:
                continue
            if vol_avg is not None and volumes is not None:
                v = volumes.loc[ts, sym]
                va = vol_avg.loc[ts, sym]
                if va > 0 and v < volume_mult * va:
                    continue
                vol_ratio = v / va if va > 0 else 1.0
            else:
                vol_ratio = 1.0
            scores[sym] = (r - 50) * vol_ratio

        winners = sorted(scores, key=scores.get, reverse=True)[:top_n]

        # Always liquidate
        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}

        if not winners:
            continue

        per_stock = cash / len(winners)
        for sym in winners:
            p = closes.loc[ts, sym]
            if np.isnan(p) or p <= 0:
                continue
            qty = per_stock / p
            holdings[sym] = qty
            cash -= qty * p

    return np.array(values) if values else None


def _backtest_breakout(closes, highs, volumes, kwargs, rebalance_every, initial_cash):
    channel_period = kwargs.get("channel_period", 40)
    volume_mult = kwargs.get("volume_mult", 1.2)
    atr_period = kwargs.get("atr_period", 14)
    trend_window = kwargs.get("trend_window", 120)
    top_n = kwargs.get("top_n", 2)

    # Precompute
    channel_high = highs.rolling(channel_period, min_periods=10).max().shift(1)
    trend_sma = closes.rolling(trend_window, min_periods=20).mean()
    vol_avg = volumes.rolling(20, min_periods=10).mean() if volumes is not None else None

    # ATR
    high_low = highs - closes.shift(1).combine_first(closes)  # simplified ATR
    atr = high_low.abs().rolling(atr_period, min_periods=5).mean()

    warmup = max(channel_period, trend_window, atr_period) + 10
    idx = closes.index[warmup:]
    rebal_points = idx[::rebalance_every]

    cash = initial_cash
    holdings = {}
    values = []

    for ts in rebal_points:
        port_value = cash
        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                port_value += qty * p
        values.append(port_value)

        scores = {}
        atr_vals = {}
        for sym in closes.columns:
            t = trend_sma.loc[ts, sym]
            if np.isnan(t) or closes.loc[ts, sym] < t:
                continue
            ch = channel_high.loc[ts, sym]
            if np.isnan(ch) or closes.loc[ts, sym] <= ch:
                continue
            if vol_avg is not None and volumes is not None:
                v = volumes.loc[ts, sym]
                va = vol_avg.loc[ts, sym]
                if va > 0 and v < volume_mult * va:
                    continue
            a = atr.loc[ts, sym]
            if np.isnan(a) or a <= 0:
                continue
            breakout_strength = (closes.loc[ts, sym] - ch) / ch
            scores[sym] = breakout_strength
            atr_vals[sym] = a

        winners = sorted(scores, key=scores.get, reverse=True)[:top_n]

        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}

        if not winners:
            continue

        # ATR-based risk parity sizing
        inv_atrs = {s: 1.0 / atr_vals[s] for s in winners}
        total_inv = sum(inv_atrs.values())
        for sym in winners:
            alloc = cash * (inv_atrs[sym] / total_inv)
            p = closes.loc[ts, sym]
            if np.isnan(p) or p <= 0:
                continue
            qty = alloc / p
            holdings[sym] = qty
            cash -= qty * p

    return np.array(values) if values else None


def _backtest_rsi_mean_revert(closes, volumes, kwargs, rebalance_every, initial_cash):
    rsi_period = kwargs.get("rsi_period", 14)
    rsi_oversold = kwargs.get("rsi_oversold", 30)
    bb_period = kwargs.get("bb_period", 20)
    bb_std = kwargs.get("bb_std", 2.0)
    volume_mult = kwargs.get("volume_mult", 1.2)
    trend_window = kwargs.get("trend_window", 120)
    top_n = kwargs.get("top_n", 2)
    bounce_window = kwargs.get("bounce_window", 3)
    use_bb = kwargs.get("use_bb", 1)

    # Precompute
    rsi = closes.apply(lambda col: _compute_rsi(col, rsi_period))
    # Rolling min RSI over bounce_window (was RSI recently oversold?)
    rsi_rolling_min = rsi.rolling(bounce_window, min_periods=1).min().shift(1)
    sma = closes.rolling(bb_period, min_periods=10).mean()
    std = closes.rolling(bb_period, min_periods=10).std()
    lower_bb = sma - bb_std * std
    near_lower = closes <= lower_bb * 1.01
    trend_sma = closes.rolling(trend_window, min_periods=20).mean()
    vol_avg = volumes.rolling(20, min_periods=10).mean() if volumes is not None else None

    warmup = max(rsi_period, bb_period, trend_window, bounce_window) + 10
    idx = closes.index[warmup:]
    rebal_points = idx[::rebalance_every]

    cash = initial_cash
    holdings = {}
    values = []

    for ts in rebal_points:
        port_value = cash
        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                port_value += qty * p
        values.append(port_value)

        scores = {}
        for sym in closes.columns:
            t = trend_sma.loc[ts, sym]
            if np.isnan(t) or closes.loc[ts, sym] < t:
                continue
            r = rsi.loc[ts, sym]
            min_r = rsi_rolling_min.loc[ts, sym]
            if np.isnan(r) or np.isnan(min_r):
                continue
            # Current RSI must be above oversold, and recently was below
            if r < rsi_oversold or min_r >= rsi_oversold:
                continue
            if use_bb and not near_lower.loc[ts, sym]:
                continue
            if vol_avg is not None and volumes is not None:
                v = volumes.loc[ts, sym]
                va = vol_avg.loc[ts, sym]
                if va > 0 and v < volume_mult * va:
                    continue
            scores[sym] = rsi_oversold - min_r  # deeper dip = higher score

        winners = sorted(scores, key=scores.get, reverse=True)[:top_n]

        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}

        if not winners:
            continue

        per_stock = cash / len(winners)
        for sym in winners:
            p = closes.loc[ts, sym]
            if np.isnan(p) or p <= 0:
                continue
            qty = per_stock / p
            holdings[sym] = qty
            cash -= qty * p

    return np.array(values) if values else None


def _backtest_ema_crossover(closes, volumes, kwargs, rebalance_every, initial_cash):
    """EMA crossover backtest with trend filter + volume confirmation."""
    fast_period = kwargs.get("fast_period", 10)
    slow_period = kwargs.get("slow_period", 60)
    volume_mult = kwargs.get("volume_mult", 1.0)
    trend_window = kwargs.get("trend_window", 240)
    top_n = kwargs.get("top_n", 2)

    fast_ema = closes.ewm(span=fast_period, min_periods=fast_period).mean()
    slow_ema = closes.ewm(span=slow_period, min_periods=slow_period).mean()
    trend_sma = closes.rolling(trend_window, min_periods=20).mean()
    vol_avg = volumes.rolling(20, min_periods=10).mean() if volumes is not None else None

    warmup = max(slow_period, trend_window) + 10
    idx = closes.index[warmup:]
    rebal_points = idx[::rebalance_every]

    cash = initial_cash
    holdings = {}
    values = []

    for ts in rebal_points:
        port_value = cash
        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                port_value += qty * p
        values.append(port_value)

        scores = {}
        for sym in closes.columns:
            t = trend_sma.loc[ts, sym]
            if np.isnan(t) or closes.loc[ts, sym] < t:
                continue
            f = fast_ema.loc[ts, sym]
            s = slow_ema.loc[ts, sym]
            if np.isnan(f) or np.isnan(s) or f <= s:
                continue
            if vol_avg is not None and volumes is not None:
                v = volumes.loc[ts, sym]
                va = vol_avg.loc[ts, sym]
                if va > 0 and v < volume_mult * va:
                    continue
                vol_ratio = v / va if va > 0 else 1.0
            else:
                vol_ratio = 1.0
            spread = (f - s) / s
            scores[sym] = spread * vol_ratio

        winners = sorted(scores, key=scores.get, reverse=True)[:top_n]

        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}

        if not winners:
            continue

        per_stock = cash / len(winners)
        for sym in winners:
            p = closes.loc[ts, sym]
            if np.isnan(p) or p <= 0:
                continue
            qty = per_stock / p
            holdings[sym] = qty
            cash -= qty * p

    return np.array(values) if values else None


def _backtest_stock_momentum(closes, volumes, kwargs, rebalance_every, initial_cash):
    """Stock momentum backtest with RSI + trend filter + volume confirmation."""
    rsi_period = kwargs.get("rsi_period", 14)
    rsi_threshold = kwargs.get("rsi_threshold", 55)
    volume_mult = kwargs.get("volume_mult", 1.2)
    top_n = kwargs.get("top_n", 3)
    trend_window = kwargs.get("trend_window", 120)

    rsi = closes.apply(lambda col: _compute_rsi(col, rsi_period))
    trend_sma = closes.rolling(trend_window, min_periods=20).mean()
    vol_avg = volumes.rolling(20, min_periods=10).mean() if volumes is not None else None

    warmup = max(rsi_period, trend_window) + 10
    idx = closes.index[warmup:]
    rebal_points = idx[::rebalance_every]

    cash = initial_cash
    holdings = {}
    values = []

    for ts in rebal_points:
        port_value = cash
        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                port_value += qty * p
        values.append(port_value)

        scores = {}
        for sym in closes.columns:
            t = trend_sma.loc[ts, sym]
            if np.isnan(t) or closes.loc[ts, sym] < t:
                continue
            r = rsi.loc[ts, sym]
            if np.isnan(r) or r < rsi_threshold:
                continue
            if vol_avg is not None and volumes is not None:
                v = volumes.loc[ts, sym]
                va = vol_avg.loc[ts, sym]
                if va > 0 and v < volume_mult * va:
                    continue
                vol_ratio = v / va if va > 0 else 1.0
            else:
                vol_ratio = 1.0
            scores[sym] = (r - 50) * vol_ratio

        winners = sorted(scores, key=scores.get, reverse=True)[:top_n]

        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}

        if not winners:
            continue

        per_stock = cash / len(winners)
        for sym in winners:
            p = closes.loc[ts, sym]
            if np.isnan(p) or p <= 0:
                continue
            qty = per_stock / p
            holdings[sym] = qty
            cash -= qty * p

    return np.array(values) if values else None


def _backtest_momentum(closes, kwargs, rebalance_every, initial_cash):
    lookback = kwargs.get("lookback_minutes", 30)
    top_n = kwargs.get("top_n", 5)

    idx = closes.index[lookback:]
    rebal_points = idx[::rebalance_every]

    cash = initial_cash
    holdings = {}
    values = []

    for ts in rebal_points:
        # Value portfolio
        port_value = cash
        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                port_value += qty * p
        values.append(port_value)

        # Score: return over lookback window
        window = closes.loc[:ts].tail(lookback)
        if len(window) < 2:
            continue
        returns = (window.iloc[-1] - window.iloc[0]) / window.iloc[0]
        returns = returns.dropna().sort_values(ascending=False)
        winners = list(returns[returns > 0].head(top_n).index)

        if not winners:
            continue

        # Liquidate
        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}

        # Buy
        per_stock = cash / len(winners)
        for sym in winners:
            p = closes.loc[ts, sym]
            if np.isnan(p) or p <= 0:
                continue
            qty = int(per_stock // p)
            if qty > 0:
                holdings[sym] = qty
                cash -= qty * p

    return np.array(values) if values else None


def _backtest_mean_reversion(closes, volumes, kwargs, rebalance_every, initial_cash):
    vwap_window = kwargs.get("vwap_window", 60)
    min_dip = kwargs.get("min_dip", 0.003)
    max_dip = kwargs.get("max_dip", 0.015)
    top_n = kwargs.get("top_n", 3)
    trend_window = kwargs.get("trend_window", 120)
    take_profit = kwargs.get("take_profit", 0.0)

    # Precompute VWAP and dip
    if volumes is not None:
        roll_cv = (closes * volumes).rolling(vwap_window, min_periods=10).sum()
        roll_v = volumes.rolling(vwap_window, min_periods=10).sum()
        vwap = roll_cv / roll_v.replace(0, np.nan)
        vwap = vwap.fillna(closes.rolling(vwap_window, min_periods=10).mean())
    else:
        vwap = closes.rolling(vwap_window, min_periods=10).mean()

    dips = (closes - vwap) / vwap
    trend_sma = closes.rolling(trend_window, min_periods=20).mean()

    warmup = max(vwap_window, trend_window) + 10
    idx = closes.index[warmup:]
    rebal_points = list(idx[::rebalance_every])

    cash = initial_cash
    holdings = {}
    values = []

    for ri, ts in enumerate(rebal_points):
        # Check take-profit between rebalances
        if take_profit > 0 and holdings:
            prev_ts = rebal_points[ri - 1] if ri > 0 else idx[0]
            between = idx[(idx > prev_ts) & (idx < ts)]
            for bar_ts in between:
                to_close = []
                for sym in list(holdings.keys()):
                    d = dips.loc[bar_ts, sym]
                    if not np.isnan(d) and d >= -take_profit:
                        to_close.append(sym)
                for sym in to_close:
                    p = closes.loc[bar_ts, sym]
                    if not np.isnan(p):
                        cash += holdings[sym] * p
                    del holdings[sym]

        port_value = cash
        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                port_value += qty * p
        values.append(port_value)

        # Find dip candidates with trend filter
        winners = []
        for sym in closes.columns:
            t = trend_sma.loc[ts, sym]
            if np.isnan(t) or closes.loc[ts, sym] < t:
                continue
            d = dips.loc[ts, sym]
            if np.isnan(d):
                continue
            if -max_dip <= d <= -min_dip:
                winners.append((sym, d))

        winners.sort(key=lambda x: x[1])
        winners = [sym for sym, _ in winners[:top_n]]

        # Always liquidate first
        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}

        if not winners:
            continue

        per_stock = cash / len(winners)
        for sym in winners:
            p = closes.loc[ts, sym]
            if np.isnan(p) or p <= 0:
                continue
            qty = per_stock / p
            if qty > 0:
                holdings[sym] = qty
                cash -= qty * p

    return np.array(values) if values else None


def _backtest_regime_mr(closes, volumes, kwargs, rebalance_every, initial_cash):
    vwap_window = kwargs.get("vwap_window", 60)
    min_dip = kwargs.get("min_dip", 0.005)
    max_dip = kwargs.get("max_dip", 0.025)
    top_n = kwargs.get("top_n", 1)
    trend_window = kwargs.get("trend_window", 240)
    take_profit = kwargs.get("take_profit", 0.003)
    stop_loss = kwargs.get("stop_loss", 0.015)
    max_hold = kwargs.get("max_hold", 30)
    regime_window = kwargs.get("regime_window", 720)
    volume_mult = kwargs.get("volume_mult", 0.0)

    # Precompute VWAP and dips
    if volumes is not None:
        roll_cv = (closes * volumes).rolling(vwap_window, min_periods=10).sum()
        roll_v = volumes.rolling(vwap_window, min_periods=10).sum()
        vwap = roll_cv / roll_v.replace(0, np.nan)
        vwap = vwap.fillna(closes.rolling(vwap_window, min_periods=10).mean())
    else:
        vwap = closes.rolling(vwap_window, min_periods=10).mean()

    dips = (closes - vwap) / vwap
    trend_sma = closes.rolling(trend_window, min_periods=20).mean()

    # BTC regime filter (first column)
    btc = closes.iloc[:, 0]
    btc_sma = btc.rolling(regime_window, min_periods=50).mean()
    regime_ok = btc > btc_sma
    vol_avg = volumes.rolling(20, min_periods=10).mean() if volumes is not None else None

    warmup = max(vwap_window, trend_window, regime_window) + 10
    idx = closes.index[warmup:]
    rebal_points = list(idx[::rebalance_every])

    cash = initial_cash
    holdings = {}
    entry_prices = {}
    entry_indices = {}
    values = []

    for ri, ts in enumerate(rebal_points):
        # Between rebalances: check take-profit, stop-loss, max-hold per bar
        if holdings:
            prev_ts = rebal_points[ri - 1] if ri > 0 else idx[0]
            between = idx[(idx > prev_ts) & (idx < ts)]
            for bar_idx, bar_ts in enumerate(between):
                to_close = []
                for sym in list(holdings.keys()):
                    p = closes.loc[bar_ts, sym]
                    if np.isnan(p):
                        continue
                    # Stop-loss
                    if stop_loss > 0 and sym in entry_prices:
                        if p <= entry_prices[sym] * (1 - stop_loss):
                            to_close.append(sym)
                            continue
                    # Take-profit: dip reverted
                    if take_profit > 0:
                        d = dips.loc[bar_ts, sym]
                        if not np.isnan(d) and d >= -take_profit:
                            to_close.append(sym)
                            continue
                    # Max hold
                    if max_hold > 0 and sym in entry_indices:
                        bars_held = closes.index.get_loc(bar_ts) - entry_indices[sym]
                        if bars_held >= max_hold:
                            to_close.append(sym)
                            continue

                for sym in to_close:
                    p = closes.loc[bar_ts, sym]
                    if not np.isnan(p):
                        cash += holdings[sym] * p
                    del holdings[sym]
                    entry_prices.pop(sym, None)
                    entry_indices.pop(sym, None)

        port_value = cash
        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                port_value += qty * p
        values.append(port_value)

        # Regime gate
        if not regime_ok.loc[ts]:
            for sym, qty in holdings.items():
                p = closes.loc[ts, sym]
                if not np.isnan(p):
                    cash += qty * p
            holdings = {}
            entry_prices = {}
            entry_indices = {}
            continue

        # Find dip candidates with trend filter
        winners = []
        for sym in closes.columns:
            t = trend_sma.loc[ts, sym]
            if np.isnan(t) or closes.loc[ts, sym] < t:
                continue
            d = dips.loc[ts, sym]
            if np.isnan(d):
                continue
            if -max_dip <= d <= -min_dip:
                if volume_mult > 0 and vol_avg is not None and volumes is not None:
                    v = volumes.loc[ts, sym]
                    va = vol_avg.loc[ts, sym]
                    if va > 0 and v < volume_mult * va:
                        continue
                winners.append((sym, d))

        winners.sort(key=lambda x: x[1])
        winners = [sym for sym, _ in winners[:top_n]]

        # Liquidate
        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}
        entry_prices = {}
        entry_indices = {}

        if not winners:
            continue

        per_stock = cash / len(winners)
        for sym in winners:
            p = closes.loc[ts, sym]
            if np.isnan(p) or p <= 0:
                continue
            qty = per_stock / p
            if qty > 0:
                holdings[sym] = qty
                entry_prices[sym] = p
                entry_indices[sym] = closes.index.get_loc(ts)
                cash -= qty * p

    return np.array(values) if values else None


def _backtest_bayesian(closes, volumes, kwargs, rebalance_every, initial_cash):
    sma_period = kwargs.get("sma_period", 20)
    rsi_period = kwargs.get("rsi_period", 14)
    momentum_weight = kwargs.get("momentum_weight", 1.0)
    volume_weight = kwargs.get("volume_weight", 0.5)
    rsi_weight = kwargs.get("rsi_weight", 0.5)
    threshold = kwargs.get("threshold", 0.6)
    trend_window = kwargs.get("trend_window", 120)
    top_n = kwargs.get("top_n", 2)

    sma = closes.rolling(sma_period, min_periods=10).mean()
    rsi = closes.apply(lambda col: _compute_rsi(col, rsi_period))
    trend_sma = closes.rolling(trend_window, min_periods=20).mean()
    vol_avg = volumes.rolling(20, min_periods=10).mean() if volumes is not None else None

    warmup = max(sma_period, rsi_period, trend_window) + 10
    idx = closes.index[warmup:]
    rebal_points = idx[::rebalance_every]

    cash = initial_cash
    holdings = {}
    values = []

    for ts in rebal_points:
        port_value = cash
        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                port_value += qty * p
        values.append(port_value)

        posteriors = {}
        for sym in closes.columns:
            t = trend_sma.loc[ts, sym]
            if np.isnan(t) or closes.loc[ts, sym] < t:
                continue
            sma_val = sma.loc[ts, sym]
            price = closes.loc[ts, sym]
            if np.isnan(sma_val) or sma_val <= 0:
                continue
            log_odds = 0.0
            momentum_signal = (price - sma_val) / sma_val
            log_odds += momentum_weight * momentum_signal * 10
            if vol_avg is not None and volumes is not None:
                v = volumes.loc[ts, sym]
                va = vol_avg.loc[ts, sym]
                if va > 0 and not np.isnan(va):
                    log_odds += volume_weight * ((v - va) / va)
            rsi_val = rsi.loc[ts, sym]
            if np.isnan(rsi_val):
                continue
            log_odds += rsi_weight * ((rsi_val - 50) / 20.0)
            posterior = 1.0 / (1.0 + np.exp(-log_odds))
            if posterior >= threshold:
                posteriors[sym] = posterior

        winners = sorted(posteriors, key=posteriors.get, reverse=True)[:top_n]

        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}

        if not winners:
            continue
        per_stock = cash / len(winners)
        for sym in winners:
            p = closes.loc[ts, sym]
            if np.isnan(p) or p <= 0:
                continue
            qty = per_stock / p
            holdings[sym] = qty
            cash -= qty * p

    return np.array(values) if values else None


def _backtest_bandit(closes, kwargs, rebalance_every, initial_cash):
    reward_window = kwargs.get("reward_window", 60)
    exploration_factor = kwargs.get("exploration_factor", 1.0)
    return_threshold = kwargs.get("return_threshold", 0.001)
    trend_window = kwargs.get("trend_window", 120)
    top_n = kwargs.get("top_n", 2)

    trend_sma = closes.rolling(trend_window, min_periods=20).mean()

    warmup = max(reward_window, trend_window) + 10
    idx = closes.index[warmup:]
    rebal_points = idx[::rebalance_every]

    cash = initial_cash
    holdings = {}
    values = []

    for ts in rebal_points:
        port_value = cash
        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                port_value += qty * p
        values.append(port_value)

        ts_idx = closes.index.get_loc(ts)
        window_start = max(0, ts_idx - reward_window)
        scores = {}

        for sym in closes.columns:
            t = trend_sma.loc[ts, sym]
            if np.isnan(t) or closes.loc[ts, sym] < t:
                continue
            window = closes.iloc[window_start:ts_idx + 1][sym]
            bar_returns = window.pct_change().dropna()
            if len(bar_returns) == 0:
                continue
            avg_reward = bar_returns.mean()
            if avg_reward <= 0:
                continue
            total_periods = len(bar_returns)
            played = (bar_returns > return_threshold).sum()
            if played == 0:
                ucb_score = avg_reward + exploration_factor * 10.0
            else:
                ucb_score = avg_reward + exploration_factor * np.sqrt(np.log(total_periods) / played)
            scores[sym] = ucb_score

        winners = sorted(scores, key=scores.get, reverse=True)[:top_n]

        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}

        if not winners:
            continue
        per_stock = cash / len(winners)
        for sym in winners:
            p = closes.loc[ts, sym]
            if np.isnan(p) or p <= 0:
                continue
            qty = per_stock / p
            holdings[sym] = qty
            cash -= qty * p

    return np.array(values) if values else None


def _backtest_monte_carlo(closes, volumes, kwargs, rebalance_every, initial_cash):
    np.random.seed(42)
    lookback = kwargs.get("lookback", 60)
    n_simulations = kwargs.get("n_simulations", 100)
    horizon = kwargs.get("horizon", 10)
    volume_weight = kwargs.get("volume_weight", 0.5)
    trend_window = kwargs.get("trend_window", 120)
    top_n = kwargs.get("top_n", 2)

    trend_sma = closes.rolling(trend_window, min_periods=20).mean()

    warmup = max(lookback, trend_window) + 10
    idx = closes.index[warmup:]
    rebal_points = idx[::rebalance_every]

    cash = initial_cash
    holdings = {}
    values = []

    for ts in rebal_points:
        port_value = cash
        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                port_value += qty * p
        values.append(port_value)

        ts_idx = closes.index.get_loc(ts)
        scores = {}

        for sym in closes.columns:
            t = trend_sma.loc[ts, sym]
            if np.isnan(t) or closes.loc[ts, sym] < t:
                continue
            if ts_idx < lookback:
                continue
            price_window = closes.iloc[ts_idx - lookback:ts_idx + 1][sym].values
            if np.any(np.isnan(price_window)):
                continue
            current_price = price_window[-1]
            if current_price <= 0:
                continue
            returns = np.diff(price_window) / price_window[:-1]
            if len(returns) < 10 or np.std(returns) == 0:
                continue

            # Bootstrap resampling of actual returns
            indices = np.random.randint(0, len(returns), size=(n_simulations, horizon))
            sim_returns = returns[indices]
            cumulative = np.cumprod(1.0 + sim_returns, axis=1)
            final_returns = cumulative[:, -1] - 1.0

            p_profit = np.mean(final_returns > 0)
            median_return = np.median(final_returns)
            downside = final_returns[final_returns < 0]
            downside_std = np.std(downside) if len(downside) > 1 else 1.0

            if p_profit < 0.55 or median_return <= 0:
                continue

            score = median_return / downside_std if downside_std > 0 else median_return

            if volume_weight > 0 and volumes is not None:
                vol_start = max(0, ts_idx - 20)
                vol_window = volumes.iloc[vol_start:ts_idx + 1][sym].values
                if len(vol_window) > 1 and not np.all(np.isnan(vol_window)):
                    current_vol = vol_window[-1]
                    avg_vol = np.nanmean(vol_window[:-1])
                    if avg_vol > 0 and not np.isnan(current_vol):
                        volume_ratio = current_vol / avg_vol
                        price_momentum = 1.0 if returns[-1] > 0 else -1.0
                        score *= (1.0 + volume_weight * volume_ratio * price_momentum)

            if score > 0:
                scores[sym] = score

        winners = sorted(scores, key=scores.get, reverse=True)[:top_n]

        for sym, qty in holdings.items():
            p = closes.loc[ts, sym]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}

        if not winners:
            continue
        per_stock = cash / len(winners)
        for sym in winners:
            p = closes.loc[ts, sym]
            if np.isnan(p) or p <= 0:
                continue
            qty = per_stock / p
            holdings[sym] = qty
            cash -= qty * p

    return np.array(values) if values else None


def _print_results(pv, initial_cash, rebalance_every, cls_name, is_crypto=False):
    total_return = (pv[-1] - initial_cash) / initial_cash
    returns = np.diff(pv) / pv[:-1]
    std = returns.std()
    periods_per_year = (1440 * 365) / rebalance_every if is_crypto else (390 * 252) / rebalance_every
    sharpe = (returns.mean() / std) * np.sqrt(periods_per_year) if std > 0 else 0
    max_dd = ((pv - np.maximum.accumulate(pv)) / np.maximum.accumulate(pv)).min()

    print(f"\n{'='*50}")
    print(f"  Strategy:       {cls_name}")
    print(f"  Initial:        ${initial_cash:,.2f}")
    print(f"  Final:          ${pv[-1]:,.2f}")
    print(f"  Total Return:   {total_return:.2%}")
    print(f"  Sharpe Ratio:   {sharpe:.2f}")
    print(f"  Max Drawdown:   {max_dd:.2%}")
    print(f"{'='*50}")

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final": pv[-1],
    }
