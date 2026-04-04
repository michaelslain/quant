import itertools
import json
import os
import numpy as np
import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime, timedelta

PARAMS_DIR = os.path.join(os.path.dirname(__file__), "..", "params")


def is_crypto(symbol):
    return "/" in symbol or (symbol.endswith("USD") and len(symbol) <= 10)


def _params_file():
    return os.path.join(PARAMS_DIR, "ou_reversion.json")


def _estimate_ou_params(series):
    """
    Estimate Ornstein-Uhlenbeck parameters via OLS on discrete observations.

    Model: S(t+1) = a * S(t) + b + epsilon
    where a = exp(-theta), b = mu * (1 - a)

    Returns: (theta, mu, sigma_eq, half_life) or None if not mean-reverting
    """
    x = series.values
    if len(x) < 30:
        return None

    y = x[1:]
    x_lag = x[:-1]

    # OLS: y = a * x_lag + b
    n = len(y)
    x_mean = x_lag.mean()
    y_mean = y.mean()
    cov_xy = ((x_lag - x_mean) * (y - y_mean)).sum()
    var_x = ((x_lag - x_mean) ** 2).sum()

    if var_x <= 0:
        return None

    a = cov_xy / var_x
    b = y_mean - a * x_mean

    # a must be in (0, 1) for mean reversion
    if a <= 0 or a >= 1:
        return None

    theta = -np.log(a)  # mean-reversion speed
    mu = b / (1 - a)    # long-run mean
    half_life = np.log(2) / theta  # bars to half-revert

    # Residual volatility
    residuals = y - (a * x_lag + b)
    sigma_res = residuals.std()
    # Equilibrium volatility: sigma_eq = sigma_res / sqrt((1 - a^2) / (2*theta))
    denom = (1 - a**2) / (2 * theta)
    if denom <= 0:
        return None
    sigma_eq = sigma_res / np.sqrt(denom)

    return theta, mu, sigma_eq, half_life


def _ou_reversion_backtest_worker(closes_vals, btc_col, n_cols, kwargs,
                                   rebalance_every, initial_cash=100_000.0,
                                   return_equity=False):
    """
    Ornstein-Uhlenbeck mean reversion backtest worker.

    For each asset, estimates OU parameters on log-prices over a rolling window.
    Only trades assets with half-life in [min_hl, max_hl] (tradeable at this
    rebalance frequency). BTC regime gate: only trades when BTC > regime SMA.

    Math:
        dS = theta * (mu - S) * dt + sigma * dW
        half_life = ln(2) / theta
        deviation = (S - mu) / sigma_eq
        Entry: deviation < -entry_dev AND half_life in [min_hl, max_hl] AND BTC bullish
        Exit: deviation > -exit_dev OR stop-loss OR max_hold OR BTC bearish
    """
    ou_window = kwargs["ou_window"]
    min_hl = kwargs["min_hl"]
    max_hl = kwargs["max_hl"]
    entry_dev = kwargs["entry_dev"]
    exit_dev = kwargs["exit_dev"]
    trend_window = kwargs["trend_window"]
    top_n = kwargs["top_n"]
    stop_loss = kwargs["stop_loss"]
    max_hold = kwargs["max_hold"]
    regime_window = kwargs.get("regime_window", 720)
    take_profit = kwargs.get("take_profit", 0.0)

    n_rows = closes_vals.shape[0]
    warmup = max(ou_window, trend_window, regime_window) + 20

    if n_rows <= warmup:
        return None

    # Precompute log prices
    log_prices = np.log(np.where(closes_vals > 0, closes_vals, np.nan))

    # Precompute trend SMA using cumsum
    trend_cs = np.nancumsum(closes_vals, axis=0)
    trend_sma = np.full_like(closes_vals, np.nan)
    for i in range(trend_window, n_rows):
        s = i - trend_window
        trend_sma[i] = (trend_cs[i] - trend_cs[s]) / trend_window

    # Precompute returns for volatility sizing
    returns = np.empty_like(closes_vals)
    returns[0, :] = 0
    returns[1:, :] = (closes_vals[1:] - closes_vals[:-1]) / np.where(
        closes_vals[:-1] > 0, closes_vals[:-1], np.nan
    )

    # Precompute rolling OU params (deviation, half_life) every `rebalance_every` bars
    # This avoids re-estimating OLS at every rebalance (expensive)
    ou_dev = np.full((n_rows, n_cols), np.nan)  # deviation from mu in sigma_eq units
    ou_hl = np.full((n_rows, n_cols), np.nan)   # half-life
    ou_theta = np.full((n_rows, n_cols), np.nan)

    # Compute OU at every rebalance point
    rebal_indices = list(range(warmup, n_rows, rebalance_every))
    for i in rebal_indices:
        for ci in range(n_cols):
            log_slice = log_prices[i-ou_window:i, ci]
            valid = log_slice[~np.isnan(log_slice)]
            if len(valid) < 30:
                continue
            ou = _estimate_ou_params(pd.Series(valid))
            if ou is None:
                continue
            theta, mu, sigma_eq, half_life = ou
            if sigma_eq <= 0:
                continue
            current_log = log_prices[i, ci]
            if np.isnan(current_log):
                continue
            ou_dev[i, ci] = (current_log - mu) / sigma_eq
            ou_hl[i, ci] = half_life
            ou_theta[i, ci] = theta

    cash = initial_cash
    holdings = {}  # col_idx -> qty
    entry_prices = {}
    entry_bars = {}
    values = []

    for ri, i in enumerate(rebal_indices):
        # Per-bar stop-loss and max-hold checks between rebalances
        if holdings:
            prev_i = rebal_indices[ri - 1] if ri > 0 else warmup
            for bar in range(prev_i + 1, i):
                to_close = []
                for ci in list(holdings.keys()):
                    p = closes_vals[bar, ci]
                    if np.isnan(p):
                        continue
                    # Stop-loss
                    if stop_loss > 0 and ci in entry_prices:
                        if p <= entry_prices[ci] * (1 - stop_loss):
                            to_close.append(ci)
                            continue
                    # Take-profit: exit when price rose by take_profit from entry
                    if take_profit > 0 and ci in entry_prices:
                        if p >= entry_prices[ci] * (1 + take_profit):
                            to_close.append(ci)
                            continue
                    # Max hold
                    if max_hold > 0 and ci in entry_bars:
                        if bar - entry_bars[ci] >= max_hold:
                            to_close.append(ci)
                            continue

                for ci in to_close:
                    p = closes_vals[bar, ci]
                    if not np.isnan(p):
                        cash += holdings[ci] * p
                    del holdings[ci]
                    entry_prices.pop(ci, None)
                    entry_bars.pop(ci, None)

        # Portfolio value
        port_value = cash
        for ci, qty in holdings.items():
            p = closes_vals[i, ci]
            if not np.isnan(p):
                port_value += qty * p
        values.append(port_value)

        # Drawdown control
        peak = max(values) if values else initial_cash
        dd = (port_value - peak) / peak if peak > 0 else 0
        dd_scale = 1.0
        if dd < -0.15:
            for ci, qty in holdings.items():
                p = closes_vals[i, ci]
                if not np.isnan(p):
                    cash += qty * p
            holdings = {}
            entry_prices = {}
            entry_bars = {}
            continue
        elif dd < -0.08:
            dd_scale = 0.5

        # BTC regime gate: only trade when BTC > regime SMA (bullish)
        if btc_col >= 0 and regime_window > 0:
            btc_price = closes_vals[i, btc_col]
            btc_regime_sma = np.nanmean(closes_vals[max(0, i-regime_window):i, btc_col])
            if not np.isnan(btc_price) and not np.isnan(btc_regime_sma):
                if btc_price < btc_regime_sma:
                    # Bearish regime — liquidate and skip
                    for ci, qty in holdings.items():
                        p = closes_vals[i, ci]
                        if not np.isnan(p):
                            cash += qty * p
                    holdings = {}
                    entry_prices = {}
                    entry_bars = {}
                    continue

        # Use precomputed OU params for entry scoring
        ou_scores = {}
        for ci in range(n_cols):
            # Trend filter
            t = trend_sma[i, ci]
            if np.isnan(t) or closes_vals[i, ci] < t:
                continue

            deviation = ou_dev[i, ci]
            half_life = ou_hl[i, ci]
            if np.isnan(deviation) or np.isnan(half_life):
                continue

            # Only trade assets with half-life in tradeable range
            if half_life < min_hl or half_life > max_hl:
                continue

            # Entry: oversold (deviation < -entry_dev)
            if deviation < -entry_dev:
                ou_scores[ci] = (deviation, ou_theta[i, ci])

        # Check exits for current holdings using precomputed deviations
        to_close = []
        for ci in list(holdings.keys()):
            deviation = ou_dev[i, ci]
            if np.isnan(deviation):
                to_close.append(ci)
                continue
            if deviation > -exit_dev:
                to_close.append(ci)

        for ci in to_close:
            p = closes_vals[i, ci]
            if not np.isnan(p):
                cash += holdings[ci] * p
            holdings.pop(ci, None)
            entry_prices.pop(ci, None)
            entry_bars.pop(ci, None)

        # Entry: pick most oversold by deviation
        candidates = [(ci, dev) for ci, (dev, theta) in ou_scores.items()
                       if ci not in holdings]
        # Sort by deviation (most negative first)
        candidates.sort(key=lambda x: x[1])
        winners = [ci for ci, _ in candidates[:top_n]]

        if not winners:
            continue

        # Liquidate holdings not in winners
        for ci in list(holdings.keys()):
            if ci not in winners:
                p = closes_vals[i, ci]
                if not np.isnan(p):
                    cash += holdings[ci] * p
                del holdings[ci]
                entry_prices.pop(ci, None)
                entry_bars.pop(ci, None)

        # Volatility-scaled sizing (target 15% annual vol)
        target_vol = 0.15
        weights = {}
        for ci in winners:
            ret_slice = returns[max(0, i-120):i, ci]
            sigma_real = np.nanstd(ret_slice)
            if sigma_real <= 0:
                continue
            sigma_annual = sigma_real * np.sqrt(1440 * 365)
            w = target_vol / sigma_annual
            w = min(w, 0.5)
            weights[ci] = w * dd_scale

        if not weights:
            continue

        total_w = sum(weights.values())
        if total_w > 1:
            for ci in weights:
                weights[ci] /= total_w

        for ci, w in weights.items():
            if ci in holdings:
                continue
            p = closes_vals[i, ci]
            if np.isnan(p) or p <= 0:
                continue
            alloc = port_value * w
            qty = alloc / p
            if qty > 0 and cash >= alloc:
                holdings[ci] = qty
                entry_prices[ci] = p
                entry_bars[ci] = i
                cash -= qty * p

    if len(values) < 2:
        return None

    pv = np.array(values)
    total_return = (pv[-1] - initial_cash) / initial_cash
    rets = np.diff(pv) / pv[:-1]
    std = rets.std()
    max_dd = ((pv - np.maximum.accumulate(pv)) / np.maximum.accumulate(pv)).min()
    if std > 0:
        periods_per_year = (1440 * 365) / rebalance_every
        sharpe = (rets.mean() / std) * np.sqrt(periods_per_year)
    else:
        sharpe = 0

    result = {"total_return": total_return, "sharpe": sharpe, "max_dd": max_dd}
    if return_equity:
        result["equity_curve"] = values
    return result


class OUReversionStrategy:
    """
    Ornstein-Uhlenbeck mean reversion for crypto.

    Estimates OU process parameters (theta, mu, sigma) on log-prices via OLS.
    Only trades assets whose half-life falls within a tradeable range.
    Entry when price deviates beyond optimal threshold from equilibrium.
    Exit when price reverts or stop-loss/max-hold triggers.

    Math:
        dS = theta * (mu - S) * dt + sigma * dW
        half_life = ln(2) / theta
        deviation = (log_price - mu) / sigma_eq
    """

    GRID = {
        "ou_window": [120, 240, 480],
        "min_hl": [5, 10],
        "max_hl": [60, 120],
        "entry_dev": [1.5, 2.0, 2.5],
        "exit_dev": [0.0, 0.5],
        "trend_window": [120, 240],
        "top_n": [1, 2],
        "stop_loss": [0.0, 0.02],
        "take_profit": [0.0, 0.002, 0.004],
        "max_hold": [30, 60, 120],
        "regime_window": [0, 540],
    }
    REBALANCE_OPTIONS = [15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 ou_window: int = 240, min_hl: int = 5, max_hl: int = 120,
                 entry_dev: float = 2.0, exit_dev: float = 0.0,
                 trend_window: int = 240, top_n: int = 1,
                 stop_loss: float = 0.02, take_profit: float = 0.0,
                 max_hold: int = 60, regime_window: int = 720):
        self.api = api
        self.symbols = symbols
        self.ou_window = ou_window
        self.min_hl = min_hl
        self.max_hl = max_hl
        self.entry_dev = entry_dev
        self.exit_dev = exit_dev
        self.trend_window = trend_window
        self.top_n = top_n
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_hold = max_hold
        self.regime_window = regime_window
        self._load_params()

    def _load_params(self):
        pf = _params_file()
        if os.path.exists(pf):
            with open(pf) as f:
                p = json.load(f)
            for attr in ["ou_window", "min_hl", "max_hl", "entry_dev",
                         "exit_dev", "trend_window", "top_n", "stop_loss",
                         "take_profit", "max_hold", "regime_window"]:
                if attr in p:
                    setattr(self, attr, p[attr])
            print(f"Loaded params from {pf}")

    def _save_params(self, rebalance_every=30, params_suffix=None):
        os.makedirs(PARAMS_DIR, exist_ok=True)
        pf = _params_file()
        if params_suffix:
            pf = pf.replace(".json", f"_{params_suffix}.json")
        params = {
            "ou_window": self.ou_window,
            "min_hl": self.min_hl,
            "max_hl": self.max_hl,
            "entry_dev": self.entry_dev,
            "exit_dev": self.exit_dev,
            "trend_window": self.trend_window,
            "top_n": self.top_n,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "max_hold": self.max_hold,
            "regime_window": self.regime_window,
            "rebalance_every": rebalance_every,
            "updated_at": datetime.now().isoformat(),
        }
        with open(pf, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Saved params to {pf}")

    def optimize(self, days: int = 7, fixed_interval: int = None, params_suffix: str = None):
        print(f"Optimizing OU-reversion over {days} days of data...")
        history = self._fetch_history(days)
        if not history:
            print("No data — keeping current params.")
            return

        closes = pd.DataFrame({sym: df["close"] for sym, df in history.items()})
        closes = closes.dropna(how="all").ffill()

        grid = self.GRID

        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from optimize import bayesian_search

        closes_vals = closes.values
        n_cols = closes_vals.shape[1]

        # Find BTC column for regime gate
        btc_syms = [s for s in closes.columns if "BTC" in s]
        btc_col = list(closes.columns).index(btc_syms[0]) if btc_syms else -1

        rebal_options = [fixed_interval] if fixed_interval else self.REBALANCE_OPTIONS
        fixed_args = (closes_vals, btc_col, n_cols)

        best_params, best_rebal, best_sharpe, best_return = bayesian_search(
            _ou_reversion_backtest_worker, grid, rebal_options, fixed_args,
            n_trials=300,
        )
        keys = list(grid.keys())

        if best_params:
            for attr in keys:
                setattr(self, attr, best_params[attr])
            label = "Optimal" if best_return > 0 else "Best (still negative)"
            print(f"\n{label} params found:")
            for k in keys:
                print(f"  {k}={best_params[k]}")
            print(f"  rebalance_every={best_rebal}min")
            print(f"  Backtest return: {best_return:.2%} | Sharpe: {best_sharpe:.2f}")
            self._save_params(best_rebal, params_suffix=params_suffix)
            return best_rebal
        else:
            print("No valid params found — keeping defaults.")
            return 30

    def _fetch_history(self, days: int, end_days_ago: int = 1) -> dict[str, pd.DataFrame]:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from backtest import fetch_history
        return fetch_history(self.api, self.symbols, days, end_days_ago=end_days_ago)

    # --- Live trading methods ---

    def get_momentum_scores(self) -> pd.Series:
        """Score each symbol by OU deviation (for compare command)."""
        window_needed = max(self.ou_window, self.trend_window) + 30
        end = datetime.now()
        start = end - timedelta(minutes=window_needed)

        all_data = {}
        for symbol in self.symbols:
            try:
                bars = self.api.get_crypto_bars(
                    symbol, tradeapi.TimeFrame.Minute,
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=window_needed,
                ).df
                if len(bars) > 60:
                    all_data[symbol] = bars["close"]
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")

        if not all_data:
            return pd.Series(dtype=float)

        closes = pd.DataFrame(all_data).dropna(how="all").ffill()
        trend_sma = closes.rolling(self.trend_window, min_periods=20).mean()

        scores = {}
        for sym in closes.columns:
            # Trend filter
            if closes[sym].iloc[-1] < trend_sma[sym].iloc[-1]:
                continue

            log_prices = np.log(closes[sym].dropna().iloc[-self.ou_window:])
            if len(log_prices) < 30:
                continue

            ou = _estimate_ou_params(log_prices)
            if ou is None:
                continue
            theta, mu, sigma_eq, half_life = ou

            if half_life < self.min_hl or half_life > self.max_hl:
                continue
            if sigma_eq <= 0:
                continue

            deviation = (log_prices.iloc[-1] - mu) / sigma_eq
            if deviation < -self.entry_dev:
                scores[sym] = deviation  # more negative = more oversold

        return pd.Series(scores).sort_values(ascending=True)

    def get_target_positions(self) -> dict[str, float]:
        scores = self.get_momentum_scores()
        picks = scores.head(self.top_n)

        if picks.empty:
            return {}

        account = self.api.get_account()
        cash = float(account.cash) * 0.95

        targets = {}
        for symbol in picks.index:
            try:
                price = self.api.get_latest_crypto_trades([symbol])[symbol].price
                qty = round(cash / len(picks) / price, 4)
                if qty > 0:
                    targets[symbol] = qty
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")

        return targets

    def rebalance(self):
        import time as _time
        print(f"\n[{datetime.now()}] Running OU-reversion strategy...")

        scores = self.get_momentum_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"OU-reversion picks: {picks}")

        current = {}
        for pos in self.api.list_positions():
            current[pos.symbol] = float(pos.qty)

        if not picks:
            print("No candidates — liquidating to cash.")
            for symbol, qty in current.items():
                if is_crypto(symbol):
                    print(f"  Closing {symbol} ({qty})")
                    try:
                        self.api.submit_order(
                            symbol=symbol, qty=abs(qty), side="sell",
                            type="market", time_in_force="gtc",
                        )
                    except Exception as e:
                        print(f"  Error closing {symbol}: {e}")
            print("Rebalance complete (all cash).")
            return

        # Close positions not in picks
        for symbol, qty in current.items():
            normalized = symbol
            for pick in picks:
                if pick.replace("/", "") == symbol:
                    normalized = pick
                    break
            if normalized not in picks:
                print(f"  Closing {symbol} ({qty})")
                try:
                    self.api.submit_order(
                        symbol=symbol, qty=abs(qty), side="sell",
                        type="market", time_in_force="gtc" if is_crypto(symbol) else "day",
                    )
                except Exception as e:
                    print(f"  Error closing {symbol}: {e}")

        _time.sleep(2)

        account = self.api.get_account()
        cash = float(account.cash) * 0.95
        per_stock = cash / len(picks)

        for symbol in picks:
            held_sym = symbol.replace("/", "")
            current_qty = current.get(held_sym, current.get(symbol, 0))
            try:
                price = self.api.get_latest_crypto_trades([symbol])[symbol].price
                target_qty = round(per_stock / price, 4)
                diff = target_qty - current_qty

                if diff > 0:
                    print(f"  Buying {diff} of {symbol}")
                    self.api.submit_order(
                        symbol=symbol, qty=diff, side="buy",
                        type="market", time_in_force="gtc",
                    )
                elif diff < 0:
                    print(f"  Selling {abs(diff)} of {symbol}")
                    self.api.submit_order(
                        symbol=symbol, qty=abs(diff), side="sell",
                        type="market", time_in_force="gtc",
                    )
                else:
                    print(f"  Holding {symbol} ({current_qty})")
            except Exception as e:
                print(f"  Error trading {symbol}: {e}")

        print("Rebalance complete.")
