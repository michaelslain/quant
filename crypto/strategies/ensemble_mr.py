import json
import os
import numpy as np
import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime, timedelta

from crypto.strategies.kalman_reversion import _kalman_filter_fast

PARAMS_DIR = os.path.join(os.path.dirname(__file__), "..", "params")


def is_crypto(symbol):
    return "/" in symbol or (symbol.endswith("USD") and len(symbol) <= 10)


def _params_file():
    return os.path.join(PARAMS_DIR, "ensemble_mr.json")


def _ensemble_mr_backtest_worker(closes_vals, volumes_vals, vol_avg_np,
                                  btc_col, n_cols, kwargs, rebalance_every,
                                  initial_cash=100_000.0, return_equity=False):
    """
    Ensemble mean-reversion backtest worker.

    Combines two proven signals:
    1. VWAP dip (price below volume-weighted average)
    2. Kalman spread z-score (altcoin cheap vs BTC-predicted value)

    Only enters when BOTH signals agree: coin has a VWAP dip AND negative
    Kalman z-score. This dual confirmation reduces false signals.

    Math:
        dip_i = (price_i - VWAP_i) / VWAP_i
        spread_i = log(P_alt) - beta * log(P_BTC) - alpha  (Kalman filter)
        z_kalman = rolling_zscore(spread_i, z_window)
        Entry: dip in [-max_dip, -min_dip] AND z_kalman < -kalman_entry AND trend filter
        Score: dip * |z_kalman|  (deeper dip + more negative Kalman = stronger)
    """
    vwap_window = kwargs["vwap_window"]
    min_dip = kwargs["min_dip"]
    max_dip = kwargs["max_dip"]
    delta = kwargs["delta"]
    R = kwargs["R"]
    kalman_entry = kwargs["kalman_entry"]
    z_window = kwargs["z_window"]
    trend_window = kwargs["trend_window"]
    top_n = kwargs["top_n"]
    take_profit = kwargs["take_profit"]
    stop_loss = kwargs["stop_loss"]
    max_hold = kwargs["max_hold"]
    regime_window = kwargs["regime_window"]

    if min_dip >= max_dip:
        return None

    n_rows = closes_vals.shape[0]
    warmup = max(vwap_window, trend_window, regime_window, z_window, 500) + 20

    if n_rows <= warmup:
        return None

    # --- Signal 1: VWAP dip ---
    if volumes_vals is not None:
        cv = closes_vals * volumes_vals
        cv_cs = np.nancumsum(cv, axis=0)
        v_cs = np.nancumsum(volumes_vals, axis=0)
        dips_vals = np.full((n_rows, n_cols), np.nan)
        for t in range(vwap_window, n_rows):
            s = t - vwap_window
            num = cv_cs[t, :] - cv_cs[s, :]
            den = v_cs[t, :] - v_cs[s, :]
            vwap = np.where(den > 0, num / den, closes_vals[t, :])
            dips_vals[t, :] = (closes_vals[t, :] - vwap) / np.where(vwap > 0, vwap, 1)
    else:
        # Fallback: use SMA as VWAP proxy
        cs = np.nancumsum(closes_vals, axis=0)
        dips_vals = np.full((n_rows, n_cols), np.nan)
        for t in range(vwap_window, n_rows):
            sma = (cs[t, :] - cs[t - vwap_window, :]) / vwap_window
            dips_vals[t, :] = (closes_vals[t, :] - sma) / np.where(sma > 0, sma, 1)

    # --- Signal 2: Kalman spread z-score vs BTC ---
    log_prices = np.log(np.maximum(closes_vals, 1e-10))
    btc_log = log_prices[:, btc_col]
    kalman_zscores = np.zeros((n_rows, n_cols))

    for ci in range(n_cols):
        if ci == btc_col:
            continue
        alt_log = log_prices[:, ci]
        valid_mask = ~(np.isnan(alt_log) | np.isnan(btc_log))
        if valid_mask.sum() < 500:
            continue

        alt_clean = np.where(np.isnan(alt_log), 0, alt_log)
        btc_clean = np.where(np.isnan(btc_log), 0, btc_log)

        _, _, spread, _ = _kalman_filter_fast(alt_clean, btc_clean, delta, R)

        # Rolling z-score of spread
        s = pd.Series(spread)
        roll_mean = s.rolling(z_window, min_periods=2).mean().values
        roll_std = s.rolling(z_window, min_periods=2).std().values
        mask = roll_std > 0
        kalman_zscores[mask, ci] = (spread[mask] - roll_mean[mask]) / roll_std[mask]
        kalman_zscores[:warmup, ci] = 0
        kalman_zscores[~valid_mask, ci] = 0

    # --- Trend SMA ---
    cs = np.nancumsum(closes_vals, axis=0)
    trend_sma = np.full((n_rows, n_cols), np.nan)
    for t in range(trend_window, n_rows):
        trend_sma[t, :] = (cs[t, :] - cs[t - trend_window, :]) / trend_window

    # --- BTC regime gate ---
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

        # ENSEMBLE: find candidates where BOTH signals agree
        candidates = []
        for ci in range(n_cols):
            if ci == btc_col:
                continue

            # Trend filter
            t = trend_sma[i, ci]
            if np.isnan(t) or closes_vals[i, ci] < t:
                continue

            # Signal 1: VWAP dip
            d = dips_vals[i, ci]
            if np.isnan(d) or not (-max_dip <= d <= -min_dip):
                continue

            # Signal 2: Kalman z-score negative
            kz = kalman_zscores[i, ci]
            if kz >= -kalman_entry:
                continue

            # Combined score: deeper dip * more negative Kalman = stronger
            score = d * abs(kz)
            candidates.append((ci, score))

        candidates.sort(key=lambda x: x[1])
        winners = [ci for ci, _ in candidates[:top_n]]

        # Liquidate current positions
        for ci, qty in holdings.items():
            p = closes_vals[i, ci]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}
        entry_prices = {}
        entry_bars = {}

        if not winners:
            continue

        alloc = cash * dd_scale
        per_stock = alloc / len(winners)
        for ci in winners:
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


class EnsembleMRStrategy:
    """
    Ensemble mean-reversion for crypto: VWAP dip + Kalman spread confirmation.

    Only enters when both signals agree — the coin must have a VWAP dip AND
    a negative Kalman z-score vs BTC. This dual confirmation filters false signals.
    """

    GRID = {
        "vwap_window": [15, 25, 40],
        "min_dip": [0.0005, 0.001, 0.002],
        "max_dip": [0.008, 0.012, 0.020],
        "delta": [1e-5, 1e-4],
        "R": [1e-3, 1e-2],
        "kalman_entry": [0.5, 1.0, 1.5],
        "z_window": [60, 120, 240],
        "trend_window": [60, 120, 240],
        "top_n": [1],
        "take_profit": [0.002, 0.003, 0.004],
        "stop_loss": [0.0, 0.015],
        "max_hold": [0, 30, 60],
        "regime_window": [0, 540, 720],
    }
    REBALANCE_OPTIONS = [15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 vwap_window: int = 25, min_dip: float = 0.001,
                 max_dip: float = 0.012, delta: float = 1e-4,
                 R: float = 1e-3, kalman_entry: float = 1.0,
                 z_window: int = 120, trend_window: int = 120,
                 top_n: int = 1, take_profit: float = 0.002,
                 stop_loss: float = 0.0, max_hold: int = 0,
                 regime_window: int = 0):
        self.api = api
        self.symbols = symbols
        self.vwap_window = vwap_window
        self.min_dip = min_dip
        self.max_dip = max_dip
        self.delta = delta
        self.R = R
        self.kalman_entry = kalman_entry
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
            for attr in ["vwap_window", "min_dip", "max_dip", "delta", "R",
                         "kalman_entry", "z_window", "trend_window", "top_n",
                         "take_profit", "stop_loss", "max_hold", "regime_window"]:
                if attr in p:
                    setattr(self, attr, p[attr])
            print(f"Loaded params from {pf}")

    def _save_params(self, rebalance_every=30, params_suffix=None):
        os.makedirs(PARAMS_DIR, exist_ok=True)
        pf = _params_file()
        if params_suffix:
            pf = pf.replace(".json", f"_{params_suffix}.json")
        params = {
            "vwap_window": self.vwap_window,
            "min_dip": self.min_dip,
            "max_dip": self.max_dip,
            "delta": self.delta,
            "R": self.R,
            "kalman_entry": self.kalman_entry,
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
        print(f"Optimizing ensemble MR over {days} days of data...")
        history = self._fetch_history(days)
        if not history:
            print("No data — keeping current params.")
            return

        closes = pd.DataFrame({sym: df["close"] for sym, df in history.items()})
        closes = closes.dropna(how="all").ffill()
        volumes = pd.DataFrame({sym: df["volume"] for sym, df in history.items() if "volume" in df.columns})
        volumes = volumes.reindex(closes.index).fillna(0) if not volumes.empty else None

        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from optimize import bayesian_search

        closes_vals = closes.values
        volumes_vals = volumes.values if volumes is not None else None
        vol_avg = volumes.rolling(20, min_periods=10).mean() if volumes is not None else None
        vol_avg_np = vol_avg.values if vol_avg is not None else None

        # Find BTC column
        cols = closes.columns.tolist()
        btc_col = 0
        for idx, c in enumerate(cols):
            if "BTC" in c:
                btc_col = idx
                break

        n_cols = closes_vals.shape[1]
        rebal_options = [fixed_interval] if fixed_interval else self.REBALANCE_OPTIONS
        fixed_args = (closes_vals, volumes_vals, vol_avg_np, btc_col, n_cols)

        print("Running Bayesian optimization...")
        best_params, best_rebal, best_sharpe, best_return = bayesian_search(
            _ensemble_mr_backtest_worker, self.GRID, rebal_options, fixed_args,
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

        window = max(self.vwap_window, self.trend_window, self.z_window, 500) + 50
        end = datetime.now()
        start = end - timedelta(minutes=window)

        all_closes = {}
        all_volumes = {}
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
                    all_volumes[symbol] = bars["volume"]
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")

        if not all_closes:
            return pd.Series(dtype=float)

        closes = pd.DataFrame(all_closes).dropna(how="all").ffill()
        volumes = pd.DataFrame(all_volumes).reindex(closes.index).fillna(0)

        # VWAP dip
        roll_cv = (closes * volumes).rolling(self.vwap_window, min_periods=10).sum()
        roll_v = volumes.rolling(self.vwap_window, min_periods=10).sum()
        vwap = roll_cv / roll_v.replace(0, np.nan)
        vwap = vwap.fillna(closes.rolling(self.vwap_window, min_periods=10).mean())
        dips = (closes - vwap) / vwap

        trend_sma = closes.rolling(self.trend_window, min_periods=20).mean()

        # Kalman z-score vs BTC
        log_prices = np.log(closes.values.clip(min=1e-10))
        btc_sym = "BTC/USD"
        btc_idx = list(closes.columns).index(btc_sym) if btc_sym in closes.columns else 0
        btc_log = log_prices[:, btc_idx]

        scores = {}
        for ci, sym in enumerate(closes.columns):
            if ci == btc_idx:
                continue
            if closes[sym].iloc[-1] < trend_sma[sym].iloc[-1]:
                continue

            d = dips[sym].iloc[-1]
            if np.isnan(d) or not (-self.max_dip <= d <= -self.min_dip):
                continue

            alt_log = log_prices[:, ci]
            _, _, spread, _ = _kalman_filter_fast(alt_log, btc_log, self.delta, self.R)
            s = pd.Series(spread)
            roll_mean = s.rolling(self.z_window, min_periods=2).mean()
            roll_std = s.rolling(self.z_window, min_periods=2).std()
            kz = np.nan
            if roll_std.iloc[-1] > 0:
                kz = (spread[-1] - roll_mean.iloc[-1]) / roll_std.iloc[-1]

            if np.isnan(kz) or kz >= -self.kalman_entry:
                continue

            scores[sym] = d * abs(kz)

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
        print(f"\n[{datetime.now()}] Running ensemble MR strategy...")

        if not self._check_regime():
            print("BTC regime: BEARISH — liquidating to cash.")
            for pos in self.api.list_positions():
                if is_crypto(pos.symbol):
                    try:
                        self.api.submit_order(
                            symbol=pos.symbol, qty=abs(float(pos.qty)),
                            side="sell", type="market", time_in_force="gtc",
                        )
                    except Exception as e:
                        print(f"  Error closing {pos.symbol}: {e}")
            return

        scores = self.get_momentum_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"Ensemble MR picks: {picks}")

        current = {}
        for pos in self.api.list_positions():
            current[pos.symbol] = float(pos.qty)

        if not picks:
            print("No dual-confirmed candidates — liquidating to cash.")
            for symbol, qty in current.items():
                if is_crypto(symbol):
                    try:
                        self.api.submit_order(
                            symbol=symbol, qty=abs(qty), side="sell",
                            type="market", time_in_force="gtc",
                        )
                    except Exception as e:
                        print(f"  Error closing {symbol}: {e}")
            return

        for symbol, qty in current.items():
            normalized = symbol
            for pick in picks:
                if pick.replace("/", "") == symbol:
                    normalized = pick
                    break
            if normalized not in picks:
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
                    self.api.submit_order(symbol=symbol, qty=diff, side="buy",
                                          type="market", time_in_force="gtc")
                    print(f"  Bought {diff} of {symbol}")
                elif diff < 0:
                    self.api.submit_order(symbol=symbol, qty=abs(diff), side="sell",
                                          type="market", time_in_force="gtc")
                    print(f"  Sold {abs(diff)} of {symbol}")
            except Exception as e:
                print(f"  Error trading {symbol}: {e}")

        print("Rebalance complete.")
