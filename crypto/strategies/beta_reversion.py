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
    return os.path.join(PARAMS_DIR, "beta_reversion.json")


def _compute_hurst(series, max_chunk=256):
    """
    Estimate Hurst exponent via rescaled range (R/S) method.

    For sub-windows of sizes [32, 64, 128, 256]:
        Y(t) = X(t) - mean(X)
        Z(t) = cumsum(Y)
        R = max(Z) - min(Z)
        S = std(X)
        R/S = R / S

    H = slope of log(R/S) vs log(chunk_size)

    H < 0.45 => mean-reverting
    H > 0.55 => trending
    """
    x = series.values
    n = len(x)
    chunk_sizes = [s for s in [32, 64, 128, 256] if s <= min(max_chunk, n // 2)]
    if len(chunk_sizes) < 2:
        return 0.5  # not enough data, assume random walk

    log_sizes = []
    log_rs = []

    for size in chunk_sizes:
        n_chunks = n // size
        if n_chunks < 1:
            continue
        rs_values = []
        for i in range(n_chunks):
            chunk = x[i * size:(i + 1) * size]
            mean_c = chunk.mean()
            y = chunk - mean_c
            z = np.cumsum(y)
            r = z.max() - z.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_values.append(r / s)
        if rs_values:
            log_sizes.append(np.log(size))
            log_rs.append(np.log(np.mean(rs_values)))

    if len(log_sizes) < 2:
        return 0.5

    # Linear regression: H = slope of log(R/S) vs log(n)
    coeffs = np.polyfit(log_sizes, log_rs, 1)
    return coeffs[0]


def _beta_reversion_backtest_worker(closes_vals, btc_col, n_cols, kwargs,
                                     rebalance_every, initial_cash=100_000.0,
                                     return_equity=False):
    """
    Beta-adjusted cross-sectional mean reversion backtest worker.

    Math:
        r_i(t) = (P_i(t) - P_i(t-1)) / P_i(t-1)
        beta_i = Cov(r_i, r_BTC) / Var(r_BTC)  over beta_window
        residual_i = R_i(t,L) - beta_i * R_BTC(t,L)
        z_i = (residual_i - mu) / sigma  over z_window
        Entry: z_i < -z_entry AND hurst(BTC) < hurst_threshold
        Exit: z_i > -z_exit OR stop-loss OR max_hold
    """
    lookback = kwargs["lookback"]
    beta_window = kwargs["beta_window"]
    z_window = kwargs["z_window"]
    z_entry = kwargs["z_entry"]
    z_exit = kwargs["z_exit"]
    hurst_window = kwargs["hurst_window"]
    hurst_threshold = kwargs["hurst_threshold"]
    trend_window = kwargs["trend_window"]
    top_n = kwargs["top_n"]
    stop_atr_mult = kwargs["stop_atr_mult"]
    max_hold = kwargs["max_hold"]

    n_rows = closes_vals.shape[0]
    warmup = max(beta_window, z_window + lookback, hurst_window, trend_window) + 20

    if n_rows <= warmup:
        return None

    # Precompute returns
    returns = np.empty_like(closes_vals)
    returns[0, :] = 0
    returns[1:, :] = (closes_vals[1:] - closes_vals[:-1]) / np.where(
        closes_vals[:-1] > 0, closes_vals[:-1], np.nan
    )

    btc_returns = returns[:, btc_col]

    # Precompute ATR (using absolute returns as proxy since we only have closes)
    atr = np.empty(n_rows)
    atr[:] = np.nan
    for i in range(60, n_rows):
        atr[i] = np.nanmean(np.abs(returns[i-60:i, :].mean(axis=1)))

    # Precompute trend SMA for all columns
    trend_sma = np.empty_like(closes_vals)
    trend_sma[:] = np.nan
    for i in range(trend_window, n_rows):
        trend_sma[i] = np.nanmean(closes_vals[i-trend_window:i], axis=0)

    rebal_indices = list(range(warmup, n_rows, rebalance_every))

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
                    # ATR stop-loss: entry - stop_atr_mult * ATR
                    if stop_atr_mult > 0 and ci in entry_prices:
                        bar_atr = atr[bar] if not np.isnan(atr[bar]) else 0
                        stop_price = entry_prices[ci] * (1 - stop_atr_mult * bar_atr * 100)
                        if p <= stop_price:
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

        # Drawdown control:
        #   dd < -15% => go to cash
        #   dd < -8% => half position size
        peak = max(values) if values else initial_cash
        dd = (port_value - peak) / peak if peak > 0 else 0
        dd_scale = 1.0
        if dd < -0.15:
            # Liquidate everything
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

        # Hurst regime gate on BTC
        btc_slice = closes_vals[i-hurst_window:i, btc_col]
        btc_slice_clean = btc_slice[~np.isnan(btc_slice)]
        if len(btc_slice_clean) < 64:
            continue

        # Compute Hurst
        chunk_sizes = [s for s in [32, 64, 128, 256] if s <= len(btc_slice_clean) // 2]
        if len(chunk_sizes) < 2:
            continue
        log_sizes = []
        log_rs = []
        for size in chunk_sizes:
            n_chunks = len(btc_slice_clean) // size
            rs_vals = []
            for c in range(n_chunks):
                chunk = btc_slice_clean[c*size:(c+1)*size]
                mean_c = chunk.mean()
                y = chunk - mean_c
                z = np.cumsum(y)
                r = z.max() - z.min()
                s = chunk.std(ddof=1)
                if s > 0:
                    rs_vals.append(r / s)
            if rs_vals:
                log_sizes.append(np.log(size))
                log_rs.append(np.log(np.mean(rs_vals)))

        if len(log_sizes) < 2:
            continue
        hurst = np.polyfit(log_sizes, log_rs, 1)[0]

        if hurst > hurst_threshold:
            # Trending regime — liquidate and skip
            for ci, qty in holdings.items():
                p = closes_vals[i, ci]
                if not np.isnan(p):
                    cash += qty * p
            holdings = {}
            entry_prices = {}
            entry_bars = {}
            continue

        # Hurst ambiguous zone: reduce sizing
        hurst_scale = 0.5 if hurst > 0.45 else 1.0

        # Compute beta-adjusted residuals for each non-BTC column
        #   beta_i = Cov(r_i, r_BTC) / Var(r_BTC) over beta_window
        #   residual_i = R_i(lookback) - beta_i * R_BTC(lookback)
        #   z_i = (residual - mu) / sigma over z_window
        btc_ret_window = btc_returns[i-beta_window:i]
        btc_var = np.nanvar(btc_ret_window)
        if btc_var <= 0:
            continue

        # Cumulative returns over lookback
        btc_cum = np.nansum(btc_returns[i-lookback:i])

        z_scores = {}
        for ci in range(n_cols):
            if ci == btc_col:
                continue
            # Trend filter
            t = trend_sma[i, ci]
            if np.isnan(t) or closes_vals[i, ci] < t:
                continue

            # Beta
            ret_window = returns[i-beta_window:i, ci]
            cov = np.nanmean((ret_window - np.nanmean(ret_window)) *
                             (btc_ret_window - np.nanmean(btc_ret_window)))
            beta = cov / btc_var

            # Residual over lookback
            cum_ret = np.nansum(returns[i-lookback:i, ci])
            residual = cum_ret - beta * btc_cum

            # Z-score: need rolling mean/std of residuals over z_window
            # Compute residuals for the past z_window rebalance points
            residuals = []
            for j in range(max(warmup, i - z_window), i):
                btc_cum_j = np.nansum(btc_returns[j-lookback:j]) if j >= lookback else 0
                ret_j = np.nansum(returns[j-lookback:j, ci]) if j >= lookback else 0
                residuals.append(ret_j - beta * btc_cum_j)

            if len(residuals) < 20:
                continue
            residuals = np.array(residuals)
            mu = residuals.mean()
            sigma = residuals.std()
            if sigma <= 0:
                continue

            z = (residual - mu) / sigma
            z_scores[ci] = z

        # Check exits for current holdings (z reverted)
        to_close = []
        for ci in list(holdings.keys()):
            if ci in z_scores and z_scores[ci] > -z_exit:
                to_close.append(ci)
        for ci in to_close:
            p = closes_vals[i, ci]
            if not np.isnan(p):
                cash += holdings[ci] * p
            del holdings[ci]
            entry_prices.pop(ci, None)
            entry_bars.pop(ci, None)

        # Entry: pick most oversold (most negative z-score)
        candidates = [(ci, z) for ci, z in z_scores.items()
                       if z < -z_entry and ci not in holdings]
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

        # Volatility-scaled sizing:
        #   weight_i = target_vol / realized_vol_annual
        #   target_vol = 15% annualized
        target_vol = 0.15
        weights = {}
        for ci in winners:
            ret_slice = returns[i-120:i, ci]
            sigma_real = np.nanstd(ret_slice)
            if sigma_real <= 0:
                continue
            sigma_annual = sigma_real * np.sqrt(1440 * 365)
            w = target_vol / sigma_annual
            w = min(w, 0.5)  # cap at 50% per position
            weights[ci] = w * dd_scale * hurst_scale

        if not weights:
            continue

        # Normalize weights to sum <= 1
        total_w = sum(weights.values())
        if total_w > 1:
            for ci in weights:
                weights[ci] /= total_w

        # Buy
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


class BetaReversionStrategy:
    """
    Cross-sectional beta-adjusted mean reversion for crypto.

    Core math:
        beta_i = Cov(r_i, r_BTC) / Var(r_BTC)
        residual_i = R_i(lookback) - beta_i * R_BTC(lookback)
        z_i = (residual - mu) / sigma

    Entry: z < -z_entry AND Hurst(BTC) < threshold (mean-reverting regime)
    Exit: z > -z_exit OR ATR stop-loss OR max hold
    Sizing: target_vol / realized_vol, scaled by drawdown and Hurst confidence
    """

    GRID = {
        "lookback": [60, 90, 120],
        "beta_window": [180, 240, 360],
        "z_window": [90, 120, 180],
        "z_entry": [1.2, 1.5, 2.0],
        "z_exit": [0.0, 0.3],
        "hurst_window": [180, 240],
        "hurst_threshold": [0.45, 0.50],
        "trend_window": [120, 240],
        "top_n": [1, 2],
        "stop_atr_mult": [2.0, 2.5, 3.0],
        "max_hold": [30, 60, 120],
    }
    REBALANCE_OPTIONS = [15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 lookback: int = 90, beta_window: int = 240,
                 z_window: int = 120, z_entry: float = 1.5,
                 z_exit: float = 0.0, hurst_window: int = 240,
                 hurst_threshold: float = 0.50, trend_window: int = 240,
                 top_n: int = 1, stop_atr_mult: float = 2.5,
                 max_hold: int = 60):
        self.api = api
        self.symbols = symbols
        self.lookback = lookback
        self.beta_window = beta_window
        self.z_window = z_window
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.hurst_window = hurst_window
        self.hurst_threshold = hurst_threshold
        self.trend_window = trend_window
        self.top_n = top_n
        self.stop_atr_mult = stop_atr_mult
        self.max_hold = max_hold
        self._load_params()

    def _load_params(self):
        pf = _params_file()
        if os.path.exists(pf):
            with open(pf) as f:
                p = json.load(f)
            for attr in ["lookback", "beta_window", "z_window", "z_entry",
                         "z_exit", "hurst_window", "hurst_threshold",
                         "trend_window", "top_n", "stop_atr_mult", "max_hold"]:
                if attr in p:
                    setattr(self, attr, p[attr])
            print(f"Loaded params from {pf}")

    def _save_params(self, rebalance_every=30, params_suffix=None):
        os.makedirs(PARAMS_DIR, exist_ok=True)
        pf = _params_file()
        if params_suffix:
            pf = pf.replace(".json", f"_{params_suffix}.json")
        params = {
            "lookback": self.lookback,
            "beta_window": self.beta_window,
            "z_window": self.z_window,
            "z_entry": self.z_entry,
            "z_exit": self.z_exit,
            "hurst_window": self.hurst_window,
            "hurst_threshold": self.hurst_threshold,
            "trend_window": self.trend_window,
            "top_n": self.top_n,
            "stop_atr_mult": self.stop_atr_mult,
            "max_hold": self.max_hold,
            "rebalance_every": rebalance_every,
            "updated_at": datetime.now().isoformat(),
        }
        with open(pf, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Saved params to {pf}")

    def optimize(self, days: int = 7, fixed_interval: int = None, params_suffix: str = None):
        print(f"Optimizing beta-reversion over {days} days of data...")
        history = self._fetch_history(days)
        if not history:
            print("No data — keeping current params.")
            return

        closes = pd.DataFrame({sym: df["close"] for sym, df in history.items()})
        closes = closes.dropna(how="all").ffill()

        # BTC must be first column
        btc_syms = [s for s in closes.columns if "BTC" in s]
        if not btc_syms:
            print("BTC not in symbols — cannot compute beta.")
            return
        btc_col = list(closes.columns).index(btc_syms[0])

        grid = self.GRID

        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from optimize import grid_search, find_best

        keys = list(grid.keys())
        combos = list(itertools.product(*grid.values()))
        closes_vals = closes.values
        n_cols = closes_vals.shape[1]

        jobs = []
        jobs_meta = []
        rebal_options = [fixed_interval] if fixed_interval else self.REBALANCE_OPTIONS
        for rebal in rebal_options:
            for vals in combos:
                kwargs = dict(zip(keys, vals))
                jobs.append((closes_vals, btc_col, n_cols, kwargs, rebal))
                jobs_meta.append((kwargs, rebal))

        print(f"Running {len(jobs)} parameter combinations...")
        results = grid_search(_beta_reversion_backtest_worker, jobs)
        best_params, best_rebal, best_sharpe, best_return = find_best(jobs_meta, results)

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
        """Score each symbol by beta-adjusted z-score (for compare command)."""
        window_needed = max(self.beta_window, self.z_window + self.lookback,
                            self.hurst_window, self.trend_window) + 30
        end = datetime.now()
        start = end - timedelta(minutes=window_needed)

        # Fetch BTC + all symbols
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

        # Hurst gate
        btc_prices = closes[btc_sym].dropna()
        hurst = _compute_hurst(btc_prices.iloc[-self.hurst_window:])
        if hurst > self.hurst_threshold:
            print(f"  Hurst={hurst:.3f} > {self.hurst_threshold} — trending regime, no trades")
            return pd.Series(dtype=float)

        # Compute returns
        rets = closes.pct_change().fillna(0)
        btc_rets = rets[btc_sym]

        # Beta + residual + z-score
        btc_var = btc_rets.iloc[-self.beta_window:].var()
        if btc_var <= 0:
            return pd.Series(dtype=float)

        btc_cum = btc_rets.iloc[-self.lookback:].sum()
        trend_sma = closes.rolling(self.trend_window, min_periods=20).mean()

        scores = {}
        for sym in closes.columns:
            if sym == btc_sym:
                continue
            # Trend filter
            if closes[sym].iloc[-1] < trend_sma[sym].iloc[-1]:
                continue

            # Beta
            sym_rets = rets[sym].iloc[-self.beta_window:]
            cov = ((sym_rets - sym_rets.mean()) * (btc_rets.iloc[-self.beta_window:] - btc_rets.iloc[-self.beta_window:].mean())).mean()
            beta = cov / btc_var

            # Residual
            cum_ret = rets[sym].iloc[-self.lookback:].sum()
            residual = cum_ret - beta * btc_cum

            # Z-score
            residuals = []
            for j in range(self.z_window):
                idx = -(self.z_window - j)
                if abs(idx) > len(rets) - self.lookback:
                    continue
                r = rets[sym].iloc[idx-self.lookback:idx].sum()
                b = btc_rets.iloc[idx-self.lookback:idx].sum()
                residuals.append(r - beta * b)

            if len(residuals) < 20:
                continue
            mu = np.mean(residuals)
            sigma = np.std(residuals)
            if sigma <= 0:
                continue
            z = (residual - mu) / sigma

            if z < -self.z_entry:
                scores[sym] = z  # more negative = more oversold

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
        print(f"\n[{datetime.now()}] Running beta-reversion strategy...")

        scores = self.get_momentum_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"Beta-adjusted picks: {picks}")

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
