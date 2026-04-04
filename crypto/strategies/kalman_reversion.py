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
    return os.path.join(PARAMS_DIR, "kalman_reversion.json")


def _kalman_filter_fast(y, x, delta, R):
    """
    Fast Kalman filter for dynamic hedge ratio: y_t = beta_t * x_t + alpha_t + noise.
    State [beta, alpha] follows a random walk. Uses Joseph form for stability.

    All scalar math — no numpy overhead per iteration.

    Returns: (beta, alpha, spread, zscore) arrays of shape (T,)
    """
    T = len(y)
    beta_out = np.empty(T)
    alpha_out = np.empty(T)
    spread_out = np.empty(T)
    zscore_out = np.empty(T)

    # State
    b, a = 0.0, 0.0

    # P matrix (symmetric 2x2): [[p00, p01], [p01, p11]]
    p00, p01, p11 = 1.0, 0.0, 1.0

    for t in range(T):
        xt = x[t]
        yt = y[t]

        # Predict: P = P + Q (add delta to diagonal)
        p00 += delta
        p11 += delta

        # Innovation: e = y - (b*x + a)
        e = yt - (b * xt + a)

        # Innovation variance: S = H @ P @ H^T + R where H = [xt, 1]
        S = xt * xt * p00 + 2.0 * xt * p01 + p11 + R

        # Kalman gain
        S_inv = 1.0 / S
        k0 = (p00 * xt + p01) * S_inv
        k1 = (p01 * xt + p11) * S_inv

        # Update state
        b += k0 * e
        a += k1 * e

        # Update P (Joseph form)
        m00 = 1.0 - k0 * xt
        m01 = -k0
        m10 = -k1 * xt
        m11_j = 1.0 - k1

        t00 = m00 * p00 + m01 * p01
        t01 = m00 * p01 + m01 * p11
        t10 = m10 * p00 + m11_j * p01
        t11 = m10 * p01 + m11_j * p11

        p00 = t00 * m00 + t01 * m01 + k0 * k0 * R
        p01 = t00 * m10 + t01 * m11_j + k0 * k1 * R
        p11 = t10 * m10 + t11 * m11_j + k1 * k1 * R

        beta_out[t] = b
        alpha_out[t] = a
        spread_out[t] = e
        zscore_out[t] = e / (S ** 0.5) if S > 0 else 0.0

    return beta_out, alpha_out, spread_out, zscore_out


def _kalman_reversion_backtest_worker(closes_vals, btc_col, n_cols, kwargs,
                                       rebalance_every, initial_cash=100_000.0,
                                       return_equity=False):
    """
    Kalman filter cross-sectional mean reversion backtest.

    Runs one Kalman filter per altcoin vs BTC on log-prices. At each rebalance,
    ranks altcoins by z-score magnitude and buys the most oversold (z < -z_entry).

    Math:
        log(P_alt) = beta * log(P_BTC) + alpha + noise
        beta, alpha estimated online via Kalman filter
        z = innovation / sqrt(innovation_variance)
        Entry: z < -z_entry AND price > trend SMA
        Exit: z > -z_exit OR stop-loss OR max_hold
    """
    delta = kwargs["delta"]
    R = kwargs["R"]
    z_entry = kwargs["z_entry"]
    z_exit = kwargs["z_exit"]
    warmup = kwargs["warmup"]
    z_window = kwargs.get("z_window", 120)
    trend_window = kwargs["trend_window"]
    top_n = kwargs["top_n"]
    stop_loss = kwargs["stop_loss"]
    take_profit = kwargs.get("take_profit", 0.0)
    max_hold = kwargs["max_hold"]
    regime_window = kwargs.get("regime_window", 0)

    n_rows = closes_vals.shape[0]
    actual_warmup = max(warmup, trend_window, z_window, regime_window) + 20

    if n_rows <= actual_warmup:
        return None

    # Log prices
    log_prices = np.log(np.where(closes_vals > 0, closes_vals, np.nan))

    # Run Kalman filter for each altcoin vs BTC
    # Use raw spread (y - beta*x - alpha) with rolling z-score
    # instead of innovation z-score (which is auto-compressed)
    btc_log = log_prices[:, btc_col]
    zscores = np.zeros((n_rows, n_cols))

    for ci in range(n_cols):
        if ci == btc_col:
            continue
        alt_log = log_prices[:, ci]

        valid_mask = ~(np.isnan(alt_log) | np.isnan(btc_log))
        if valid_mask.sum() < warmup:
            continue

        alt_clean = np.where(np.isnan(alt_log), 0, alt_log)
        btc_clean = np.where(np.isnan(btc_log), 0, btc_log)

        beta, alpha, spread, _ = _kalman_filter_fast(alt_clean, btc_clean, delta, R)

        # Fully vectorized rolling z-score using pandas
        s = pd.Series(spread)
        roll_mean = s.rolling(z_window, min_periods=2).mean().values
        roll_std = s.rolling(z_window, min_periods=2).std().values
        mask = roll_std > 0
        zscores[mask, ci] = (spread[mask] - roll_mean[mask]) / roll_std[mask]
        # Zero out warmup and NaN regions
        zscores[:actual_warmup, ci] = 0
        zscores[~valid_mask, ci] = 0

    # Precompute trend SMA
    trend_sma = np.empty_like(closes_vals)
    trend_sma[:] = np.nan
    for i in range(trend_window, n_rows):
        trend_sma[i] = np.nanmean(closes_vals[i-trend_window:i], axis=0)

    # Precompute returns for vol sizing
    returns = np.empty_like(closes_vals)
    returns[0, :] = 0
    returns[1:, :] = (closes_vals[1:] - closes_vals[:-1]) / np.where(
        closes_vals[:-1] > 0, closes_vals[:-1], np.nan
    )

    # BTC regime gate
    if regime_window > 0:
        btc = closes_vals[:, btc_col]
        btc_cs = np.nancumsum(btc)
        regime_ok = np.zeros(n_rows, dtype=bool)
        for ii in range(regime_window, n_rows):
            s = ii - regime_window
            btc_sma = (btc_cs[ii] - btc_cs[s]) / regime_window
            regime_ok[ii] = btc[ii] > btc_sma
    else:
        regime_ok = np.ones(n_rows, dtype=bool)

    rebal_indices = list(range(actual_warmup, n_rows, rebalance_every))

    cash = initial_cash
    holdings = {}
    entry_prices = {}
    entry_bars = {}
    values = []

    for ri, i in enumerate(rebal_indices):
        # Per-bar stop-loss and max-hold
        if holdings:
            prev_i = rebal_indices[ri - 1] if ri > 0 else actual_warmup
            for bar in range(prev_i + 1, i):
                to_close = []
                for ci in list(holdings.keys()):
                    p = closes_vals[bar, ci]
                    if np.isnan(p):
                        continue
                    if stop_loss > 0 and ci in entry_prices:
                        if p <= entry_prices[ci] * (1 - stop_loss):
                            to_close.append(ci)
                            continue
                    if take_profit > 0 and ci in entry_prices:
                        if p >= entry_prices[ci] * (1 + take_profit):
                            to_close.append(ci)
                            continue
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

        # Regime gate
        if not regime_ok[i]:
            for ci, qty in holdings.items():
                p = closes_vals[i, ci]
                if not np.isnan(p):
                    cash += qty * p
            holdings = {}
            entry_prices = {}
            entry_bars = {}
            continue

        # Check exits: z reverted
        to_close = []
        for ci in list(holdings.keys()):
            z = zscores[i, ci]
            if z > -z_exit:
                to_close.append(ci)
        for ci in to_close:
            p = closes_vals[i, ci]
            if not np.isnan(p):
                cash += holdings[ci] * p
            holdings.pop(ci, None)
            entry_prices.pop(ci, None)
            entry_bars.pop(ci, None)

        # Entry: most oversold altcoins (most negative z-score)
        candidates = []
        for ci in range(n_cols):
            if ci == btc_col or ci in holdings:
                continue
            z = zscores[i, ci]
            if z >= -z_entry:
                continue
            # Trend filter
            t = trend_sma[i, ci]
            if np.isnan(t) or closes_vals[i, ci] < t:
                continue
            candidates.append((ci, z))

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

        # Volatility-scaled sizing
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


class KalmanReversionStrategy:
    """
    Kalman filter cross-sectional mean reversion for crypto.

    Runs one Kalman filter per altcoin vs BTC on log-prices to estimate
    dynamic hedge ratios. Trades standardized innovations (spreads).

    Math:
        log(P_alt) = beta_t * log(P_BTC) + alpha_t + noise
        z_t = innovation / sqrt(innovation_variance)
        Entry: z < -z_entry (altcoin oversold vs BTC-predicted value)
        Exit: z > -z_exit OR stop-loss OR max_hold
    """

    GRID = {
        "delta": [1e-5, 1e-4, 1e-3],
        "R": [1e-3, 1e-2],
        "z_entry": [1.5, 2.0, 2.5],
        "z_exit": [0.0, 0.5],
        "z_window": [60, 120, 240],
        "warmup": [500],
        "trend_window": [120, 240],
        "top_n": [1, 2],
        "stop_loss": [0.0, 0.02],
        "take_profit": [0.0, 0.002, 0.004],
        "max_hold": [30, 60, 120],
        "regime_window": [0, 540, 720],
    }
    REBALANCE_OPTIONS = [15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 delta: float = 1e-4, R: float = 1e-3,
                 z_entry: float = 2.0, z_exit: float = 0.0,
                 z_window: int = 120, warmup: int = 500,
                 trend_window: int = 240, top_n: int = 1,
                 stop_loss: float = 0.02, take_profit: float = 0.0,
                 max_hold: int = 120, regime_window: int = 0):
        self.api = api
        self.symbols = symbols
        self.delta = delta
        self.R = R
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_window = z_window
        self.warmup = warmup
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
            for attr in ["delta", "R", "z_entry", "z_exit", "z_window",
                         "warmup", "trend_window", "top_n", "stop_loss",
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
            "delta": self.delta,
            "R": self.R,
            "z_entry": self.z_entry,
            "z_exit": self.z_exit,
            "z_window": self.z_window,
            "warmup": self.warmup,
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
        print(f"Optimizing kalman-reversion over {days} days of data...")
        history = self._fetch_history(days)
        if not history:
            print("No data — keeping current params.")
            return

        closes = pd.DataFrame({sym: df["close"] for sym, df in history.items()})
        closes = closes.dropna(how="all").ffill()

        btc_syms = [s for s in closes.columns if "BTC" in s]
        if not btc_syms:
            print("BTC not in symbols — cannot compute Kalman hedge ratios.")
            return
        btc_col = list(closes.columns).index(btc_syms[0])

        grid = self.GRID

        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from optimize import bayesian_search

        closes_vals = closes.values
        n_cols = closes_vals.shape[1]

        rebal_options = [fixed_interval] if fixed_interval else self.REBALANCE_OPTIONS
        fixed_args = (closes_vals, btc_col, n_cols)

        best_params, best_rebal, best_sharpe, best_return = bayesian_search(
            _kalman_reversion_backtest_worker, grid, rebal_options, fixed_args,
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
        """Score each symbol by Kalman z-score (for compare command)."""
        window_needed = max(self.warmup, self.trend_window) + 100
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

        if not all_data or not any("BTC" in s for s in all_data):
            return pd.Series(dtype=float)

        closes = pd.DataFrame(all_data).dropna(how="all").ffill()
        btc_sym = [s for s in closes.columns if "BTC" in s][0]
        trend_sma = closes.rolling(self.trend_window, min_periods=20).mean()

        btc_log = np.log(closes[btc_sym].values)

        scores = {}
        for sym in closes.columns:
            if sym == btc_sym:
                continue
            if closes[sym].iloc[-1] < trend_sma[sym].iloc[-1]:
                continue

            alt_log = np.log(closes[sym].values)
            _, _, spread, _ = _kalman_filter_fast(alt_log, btc_log, self.delta, self.R)

            if len(spread) < self.warmup:
                continue

            # Rolling z-score of raw spread
            window = spread[-self.z_window:]
            mu = window.mean()
            sigma = window.std()
            if sigma <= 0:
                continue
            current_z = (spread[-1] - mu) / sigma
            if current_z < -self.z_entry:
                scores[sym] = current_z

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
        print(f"\n[{datetime.now()}] Running kalman-reversion strategy...")

        scores = self.get_momentum_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"Kalman-reversion picks: {picks}")

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
