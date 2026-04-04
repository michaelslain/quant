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
    return os.path.join(PARAMS_DIR, "fracdiff_mr.json")


def _fracdiff_weights(d, threshold=1e-5, max_k=500):
    """
    Compute fractional differencing weights w_k = prod((k - 1 - d) / k, k=1..K).
    Truncate when |w_k| < threshold.
    """
    w = [1.0]
    for k in range(1, max_k):
        w_k = w[-1] * (k - 1 - d) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w)


def _fracdiff_series(series, d, threshold=1e-5):
    """
    Apply fractional differencing of order d to a 1D array.
    Returns array of same length with NaN for warmup period.
    """
    w = _fracdiff_weights(d, threshold)
    n_w = len(w)
    n = len(series)
    out = np.full(n, np.nan)
    for t in range(n_w - 1, n):
        out[t] = np.dot(w, series[t - n_w + 1:t + 1][::-1])
    return out


def _fracdiff_mr_backtest_worker(closes_vals, n_cols, kwargs, rebalance_every,
                                  initial_cash=100_000.0, return_equity=False):
    """
    Fractional differentiation mean-reversion backtest worker.

    Key idea: fractional differencing (d in 0-1) achieves stationarity while
    preserving long-memory structure. Z-score of the frac-diff series gives
    a mathematically grounded mean-reversion signal.

    Math:
        fd_t = sum(w_k * log_price_{t-k})   where w_k = prod((k-1-d)/k)
        z_t = (fd_t - rolling_mean(fd, z_window)) / rolling_std(fd, z_window)
        Entry: z < -z_entry AND trend filter AND regime gate
        Exit: z > -z_exit OR take-profit OR stop-loss OR max-hold
    """
    frac_d = kwargs["frac_d"]
    z_entry = kwargs["z_entry"]
    z_exit = kwargs["z_exit"]
    z_window = kwargs["z_window"]
    trend_window = kwargs["trend_window"]
    top_n = kwargs["top_n"]
    take_profit = kwargs["take_profit"]
    stop_loss = kwargs["stop_loss"]
    max_hold = kwargs["max_hold"]
    regime_window = kwargs["regime_window"]

    n_rows = closes_vals.shape[0]

    # Compute frac-diff weights once
    weights = _fracdiff_weights(frac_d)
    n_w = len(weights)

    warmup = max(n_w + z_window, trend_window, regime_window) + 20

    if warmup >= n_rows:
        return None

    # Precompute log prices and frac-diff series for all coins
    log_prices = np.log(np.maximum(closes_vals, 1e-10))
    fd = np.full((n_rows, n_cols), np.nan)
    for ci in range(n_cols):
        fd[:, ci] = _fracdiff_series(log_prices[:, ci], frac_d)

    # Rolling z-score of frac-diff using pandas for vectorization
    zscores = np.full((n_rows, n_cols), np.nan)
    for ci in range(n_cols):
        s = pd.Series(fd[:, ci])
        roll_mean = s.rolling(z_window, min_periods=max(2, z_window // 4)).mean().values
        roll_std = s.rolling(z_window, min_periods=max(2, z_window // 4)).std().values
        mask = roll_std > 0
        zscores[mask, ci] = (fd[mask, ci] - roll_mean[mask]) / roll_std[mask]

    # Trend SMA via cumsum
    cs = np.nancumsum(closes_vals, axis=0)
    trend_sma = np.full((n_rows, n_cols), np.nan)
    for t in range(trend_window, n_rows):
        trend_sma[t, :] = (cs[t, :] - cs[t - trend_window, :]) / trend_window

    # BTC regime gate
    if regime_window > 0:
        btc = closes_vals[:, 0]
        btc_cs = np.nancumsum(btc)
        regime_ok = np.zeros(n_rows, dtype=bool)
        for ii in range(regime_window, n_rows):
            s = ii - regime_window
            btc_sma = (btc_cs[ii] - btc_cs[s]) / regime_window
            regime_ok[ii] = btc[ii] > btc_sma
    else:
        regime_ok = np.ones(n_rows, dtype=bool)

    rebal_indices = list(range(warmup, n_rows, rebalance_every))

    cash = initial_cash
    holdings = {}
    entry_prices = {}
    entry_bars = {}
    values = []

    for ri, i in enumerate(rebal_indices):
        # Between rebalances: per-bar take-profit, stop-loss, max-hold
        if holdings:
            prev_i = rebal_indices[ri - 1] if ri > 0 else warmup
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

        # Find mean-reversion candidates: z < -z_entry AND above trend
        candidates = []
        for ci in range(n_cols):
            z = zscores[i, ci]
            if np.isnan(z):
                continue
            # Already holding: check exit
            if ci in holdings:
                if z > -z_exit:
                    # Exit signal
                    p = closes_vals[i, ci]
                    if not np.isnan(p):
                        cash += holdings[ci] * p
                    del holdings[ci]
                    entry_prices.pop(ci, None)
                    entry_bars.pop(ci, None)
                continue

            # Entry: z below -z_entry AND trend filter
            if z >= -z_entry:
                continue
            t = trend_sma[i, ci]
            if np.isnan(t) or closes_vals[i, ci] < t:
                continue
            candidates.append((ci, z))

        # Sort by most negative z (deepest reversion signal)
        candidates.sort(key=lambda x: x[1])
        winners = [ci for ci, _ in candidates[:top_n]]

        # Liquidate holdings not in winners and not still held
        for ci in list(holdings.keys()):
            if ci not in winners:
                p = closes_vals[i, ci]
                if not np.isnan(p):
                    cash += holdings[ci] * p
                del holdings[ci]
                entry_prices.pop(ci, None)
                entry_bars.pop(ci, None)

        if not winners:
            continue

        # Equal-weight allocation
        alloc = cash * dd_scale
        per_stock = alloc / len(winners)
        for ci in winners:
            if ci in holdings:
                continue  # already holding
            p = closes_vals[i, ci]
            if np.isnan(p) or p <= 0:
                continue
            qty = per_stock / p
            if qty > 0:
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
    if std > 0:
        periods_per_year = (1440 * 365) / rebalance_every
        sharpe = (rets.mean() / std) * np.sqrt(periods_per_year)
    else:
        sharpe = 0

    result = {"total_return": total_return, "sharpe": sharpe}
    if return_equity:
        result["equity_curve"] = values
    return result


class FracDiffMRStrategy:
    """
    Fractional differentiation mean-reversion for crypto.

    Uses fractionally differenced log-prices (d in 0-1) to achieve stationarity
    while preserving predictive long-memory structure. Z-score of the frac-diff
    series identifies mean-reversion opportunities.

    Features: BTC regime gate, per-bar take-profit/stop-loss/max-hold, drawdown control.
    """

    GRID = {
        "frac_d": [0.3, 0.4, 0.5],
        "z_entry": [1.5, 2.0, 2.5],
        "z_exit": [0.0, 0.5],
        "z_window": [120, 240, 480],
        "trend_window": [120, 240, 480],
        "top_n": [1, 2],
        "take_profit": [0.0, 0.002, 0.004],
        "stop_loss": [0.0, 0.015],
        "max_hold": [0, 30, 60],
        "regime_window": [0, 540, 720],
    }
    REBALANCE_OPTIONS = [15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 frac_d: float = 0.4, z_entry: float = 2.0,
                 z_exit: float = 0.5, z_window: int = 240,
                 trend_window: int = 240, top_n: int = 1,
                 take_profit: float = 0.002, stop_loss: float = 0.0,
                 max_hold: int = 0, regime_window: int = 0):
        self.api = api
        self.symbols = symbols
        self.frac_d = frac_d
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.z_window = z_window
        self.trend_window = trend_window
        self.top_n = top_n
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.max_hold = max_hold
        self.regime_window = regime_window
        self._load_params()

    def _load_params(self):
        pf = _params_file()
        if os.path.exists(pf):
            with open(pf) as f:
                p = json.load(f)
            for attr in ["frac_d", "z_entry", "z_exit", "z_window",
                         "trend_window", "top_n", "take_profit", "stop_loss",
                         "max_hold", "regime_window"]:
                if attr in p:
                    setattr(self, attr, p[attr])
            print(f"Loaded params from {pf}")

    def _save_params(self, rebalance_every=30, params_suffix=None):
        os.makedirs(PARAMS_DIR, exist_ok=True)
        pf = _params_file()
        if params_suffix:
            pf = pf.replace(".json", f"_{params_suffix}.json")
        params = {
            "frac_d": self.frac_d,
            "z_entry": self.z_entry,
            "z_exit": self.z_exit,
            "z_window": self.z_window,
            "trend_window": self.trend_window,
            "top_n": self.top_n,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "max_hold": self.max_hold,
            "regime_window": self.regime_window,
            "rebalance_every": rebalance_every,
            "updated_at": datetime.now().isoformat(),
        }
        with open(pf, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Saved params to {pf}")

    def optimize(self, days: int = 7, fixed_interval: int = None, params_suffix: str = None):
        print(f"Optimizing frac-diff mean-reversion over {days} days of data...")
        history = self._fetch_history(days)
        if not history:
            print("No data — keeping current params.")
            return

        closes = pd.DataFrame({sym: df["close"] for sym, df in history.items()})
        closes = closes.dropna(how="all").ffill()

        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from optimize import bayesian_search

        closes_vals = closes.values
        n_cols = closes_vals.shape[1]

        rebal_options = [fixed_interval] if fixed_interval else self.REBALANCE_OPTIONS
        fixed_args = (closes_vals, n_cols)

        print("Running Bayesian optimization...")
        best_params, best_rebal, best_sharpe, best_return = bayesian_search(
            _fracdiff_mr_backtest_worker, self.GRID, rebal_options, fixed_args,
            n_trials=300)

        if best_params:
            for attr in self.GRID.keys():
                setattr(self, attr, best_params[attr])
            label = "Optimal" if best_return > 0 else "Best (still negative)"
            print(f"\n{label} params found:")
            for k in self.GRID.keys():
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

    def _check_regime(self) -> bool:
        if self.regime_window <= 0:
            return True
        end = datetime.now()
        start = end - timedelta(minutes=self.regime_window + 30)
        try:
            bars = self.api.get_crypto_bars(
                "BTC/USD", tradeapi.TimeFrame.Minute,
                start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                limit=self.regime_window,
            ).df
            if len(bars) < self.regime_window // 2:
                return False
            sma = bars["close"].rolling(self.regime_window, min_periods=50).mean()
            return bars["close"].iloc[-1] > sma.iloc[-1]
        except Exception as e:
            print(f"Error checking BTC regime: {e}")
            return False

    def get_momentum_scores(self) -> pd.Series:
        if not self._check_regime():
            print("  BTC regime: BEARISH — staying in cash")
            return pd.Series(dtype=float)

        # Need enough history for frac-diff weights + z-score window
        weights = _fracdiff_weights(self.frac_d)
        n_w = len(weights)
        window = max(n_w + self.z_window, self.trend_window) + 50
        end = datetime.now()
        start = end - timedelta(minutes=window)

        all_closes = {}
        for symbol in self.symbols:
            try:
                bars = self.api.get_crypto_bars(
                    symbol, tradeapi.TimeFrame.Minute,
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=window,
                ).df
                if len(bars) > 60:
                    all_closes[symbol] = bars["close"]
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")

        if not all_closes:
            return pd.Series(dtype=float)

        closes = pd.DataFrame(all_closes).dropna(how="all").ffill()
        log_prices = np.log(closes.values.clip(min=1e-10))

        # Compute frac-diff and z-score for each coin
        trend_sma = closes.rolling(self.trend_window, min_periods=20).mean()

        scores = {}
        for ci, sym in enumerate(closes.columns):
            fd = _fracdiff_series(log_prices[:, ci], self.frac_d)
            s = pd.Series(fd)
            roll_mean = s.rolling(self.z_window, min_periods=max(2, self.z_window // 4)).mean()
            roll_std = s.rolling(self.z_window, min_periods=max(2, self.z_window // 4)).std()

            z = np.nan
            if roll_std.iloc[-1] > 0:
                z = (fd[-1] - roll_mean.iloc[-1]) / roll_std.iloc[-1]

            if np.isnan(z) or z >= -self.z_entry:
                continue

            # Trend filter
            if closes[sym].iloc[-1] < trend_sma[sym].iloc[-1]:
                continue

            scores[sym] = z  # more negative = stronger signal

        return pd.Series(scores).sort_values(ascending=True)

    def get_target_positions(self) -> dict[str, float]:
        scores = self.get_momentum_scores()
        picks = scores.head(self.top_n)

        if picks.empty:
            return {}

        account = self.api.get_account()
        cash = float(account.cash) * 0.95
        per_stock = cash / len(picks)

        targets = {}
        for symbol in picks.index:
            try:
                price = self.api.get_latest_crypto_trades([symbol])[symbol].price
                qty = round(per_stock / price, 4)
                if qty > 0:
                    targets[symbol] = qty
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")

        return targets

    def rebalance(self):
        import time as _time
        print(f"\n[{datetime.now()}] Running frac-diff mean-reversion strategy...")

        if not self._check_regime():
            print("BTC regime: BEARISH — liquidating to cash.")
            for pos in self.api.list_positions():
                if is_crypto(pos.symbol):
                    print(f"  Closing {pos.symbol} ({pos.qty})")
                    try:
                        self.api.submit_order(
                            symbol=pos.symbol, qty=abs(float(pos.qty)),
                            side="sell", type="market", time_in_force="gtc",
                        )
                    except Exception as e:
                        print(f"  Error closing {pos.symbol}: {e}")
            print("Rebalance complete (all cash — bearish regime).")
            return

        scores = self.get_momentum_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"Frac-diff MR picks: {picks}")

        current = {}
        for pos in self.api.list_positions():
            current[pos.symbol] = float(pos.qty)

        if not picks:
            print("No reversion candidates — liquidating to cash.")
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
