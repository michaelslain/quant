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
    return os.path.join(PARAMS_DIR, "velocity_mr.json")


def _velocity_mr_backtest_worker(closes_vals, n_cols, kwargs, rebalance_every,
                                  initial_cash=100_000.0, return_equity=False):
    """
    Price Velocity Mean Reversion backtest worker.

    Inspired by the Relativistic Black-Scholes model (Trzetrzelewski 2013):
    price changes have a finite maximal speed. When price velocity approaches
    this observed speed limit (from below), the move is exhausting and likely
    to revert.

    Math:
        v_i(t) = log(P_i(t)) - log(P_i(t - vel_window))   # log-return velocity
        v_max_i = rolling_max(|v_i|, lookback)              # observed speed limit
        normalized_v = v_i / v_max_i                        # fraction of speed limit
        Entry: normalized_v < -entry_frac (sharp drop near speed limit)
               AND price > trend_sma AND regime gate
        Score: more negative normalized_v = deeper into speed limit = stronger signal
    """
    vel_window = kwargs["vel_window"]
    lookback = kwargs["lookback"]
    entry_frac = kwargs["entry_frac"]
    top_n = kwargs["top_n"]
    trend_window = kwargs["trend_window"]
    take_profit = kwargs.get("take_profit", 0.0)
    stop_loss = kwargs.get("stop_loss", 0.0)
    max_hold = kwargs.get("max_hold", 0)
    regime_window = kwargs.get("regime_window", 0)

    n_rows = closes_vals.shape[0]
    warmup = max(vel_window + lookback, trend_window, regime_window) + 20

    if n_rows <= warmup:
        return None

    # Compute log-return velocity
    log_prices = np.log(np.where(closes_vals > 0, closes_vals, np.nan))
    velocity = np.empty_like(log_prices)
    velocity[:] = np.nan
    velocity[vel_window:] = log_prices[vel_window:] - log_prices[:-vel_window]

    # Compute rolling max of |velocity| (observed speed limit per coin)
    abs_vel = np.abs(velocity)
    v_max = np.empty_like(abs_vel)
    v_max[:] = np.nan
    for i in range(vel_window + lookback, n_rows):
        s = i - lookback
        v_max[i] = np.nanmax(abs_vel[s:i + 1], axis=0)

    # Normalized velocity: fraction of speed limit
    norm_vel = np.where(v_max > 0, velocity / v_max, 0)

    # VWAP for take-profit (reuse rolling mean)
    vwap_w = max(vel_window, 25)
    cs = np.nancumsum(closes_vals, axis=0)
    vwap = np.empty_like(closes_vals)
    vwap[:] = np.nan
    for i in range(vwap_w, n_rows):
        s = i - vwap_w
        vwap[i] = (cs[i] - cs[s]) / vwap_w
    dips = np.where(vwap > 0, (closes_vals - vwap) / vwap, np.nan)

    # Trend SMA
    trend_cs = np.nancumsum(closes_vals, axis=0)
    trend_sma = np.empty_like(closes_vals)
    trend_sma[:] = np.nan
    for i in range(trend_window, n_rows):
        s = i - trend_window
        trend_sma[i] = (trend_cs[i] - trend_cs[s]) / trend_window

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
        # Per-bar take-profit, stop-loss, max-hold
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
                    if take_profit > 0:
                        d = dips[bar, ci]
                        if not np.isnan(d) and d >= -take_profit:
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

        # Find velocity reversion candidates
        candidates = []
        for ci in range(n_cols):
            t = trend_sma[i, ci]
            if np.isnan(t) or closes_vals[i, ci] < t:
                continue
            nv = norm_vel[i, ci]
            if np.isnan(nv):
                continue
            # Looking for sharp drops approaching negative speed limit
            if nv < -entry_frac:
                candidates.append((ci, nv))

        candidates.sort(key=lambda x: x[1])  # most negative first
        winners = [ci for ci, _ in candidates[:top_n]]

        # Liquidate current holdings
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
    returns_pv = np.diff(pv) / pv[:-1]
    std = returns_pv.std()
    if std > 0:
        periods_per_year = (1440 * 365) / rebalance_every
        sharpe = (returns_pv.mean() / std) * np.sqrt(periods_per_year)
    else:
        sharpe = 0

    if return_equity:
        return {"total_return": total_return, "sharpe": sharpe, "equity_curve": values}
    return {"total_return": total_return, "sharpe": sharpe}


class VelocityMRStrategy:
    """
    Price Velocity Mean Reversion for crypto.

    Based on the Relativistic Black-Scholes insight: price changes have a finite
    maximal speed. Buy coins whose price velocity is approaching the negative
    speed limit (sharp drop near historical max) — these moves are exhausting
    and likely to revert.
    """

    GRID = {
        "vel_window": [5, 15, 30],
        "lookback": [120, 240, 480],
        "entry_frac": [0.5, 0.6, 0.7, 0.8],
        "top_n": [1, 2],
        "trend_window": [120, 240],
        "take_profit": [0.0, 0.002],
        "stop_loss": [0.0, 0.025],
        "max_hold": [0, 25, 60],
        "regime_window": [0, 540],
    }
    REBALANCE_OPTIONS = [15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 vel_window: int = 15, lookback: int = 240,
                 entry_frac: float = 0.6, top_n: int = 1,
                 trend_window: int = 120, take_profit: float = 0.002,
                 stop_loss: float = 0.0, max_hold: int = 0,
                 regime_window: int = 0):
        self.api = api
        self.symbols = symbols
        self.vel_window = vel_window
        self.lookback = lookback
        self.entry_frac = entry_frac
        self.top_n = top_n
        self.trend_window = trend_window
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
            for attr in ["vel_window", "lookback", "entry_frac", "top_n",
                         "trend_window", "take_profit", "stop_loss",
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
            "vel_window": self.vel_window,
            "lookback": self.lookback,
            "entry_frac": self.entry_frac,
            "top_n": self.top_n,
            "trend_window": self.trend_window,
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
        print(f"Optimizing velocity MR strategy over {days} days of data...")
        history = self._fetch_history(days)
        if not history:
            print("No data -- keeping current params.")
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

        best_params, best_rebal, best_sharpe, best_return = bayesian_search(
            _velocity_mr_backtest_worker, self.GRID, rebal_options, fixed_args,
            n_trials=300)

        if best_params:
            for k, v in best_params.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            label = "Optimal" if best_return > 0 else "Best (still negative)"
            print(f"\n{label} params found:")
            for k, v in best_params.items():
                print(f"  {k}={v}")
            print(f"  rebalance_every={best_rebal}min")
            print(f"  Backtest return: {best_return:.2%} | Sharpe: {best_sharpe:.2f}")
            self._save_params(best_rebal, params_suffix=params_suffix)
            return best_rebal
        else:
            print("No valid params found -- keeping defaults.")
            return 30

    def _fetch_history(self, days: int, end_days_ago: int = 1) -> dict[str, pd.DataFrame]:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from backtest import fetch_history
        return fetch_history(self.api, self.symbols, days, end_days_ago=end_days_ago)

    def get_momentum_scores(self) -> pd.Series:
        end = datetime.now()
        lookback_bars = max(self.vel_window + self.lookback, self.trend_window) + 100
        start = end - timedelta(minutes=lookback_bars)

        all_bars = {}
        for symbol in self.symbols:
            try:
                bars = self.api.get_crypto_bars(
                    symbol, tradeapi.TimeFrame.Minute,
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=lookback_bars,
                ).df
                if len(bars) > 0:
                    all_bars[symbol] = bars
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")

        if not all_bars:
            return pd.Series(dtype=float)

        closes = pd.DataFrame({s: b["close"] for s, b in all_bars.items()})
        closes = closes.dropna(how="all").ffill()

        log_p = np.log(closes)
        velocity = log_p - log_p.shift(self.vel_window)
        abs_vel = velocity.abs()
        v_max = abs_vel.rolling(self.lookback).max()
        norm_vel = velocity / v_max

        trend_sma = closes.rolling(self.trend_window, min_periods=20).mean()

        scores = {}
        for symbol in closes.columns:
            price = closes[symbol].iloc[-1]
            t = trend_sma[symbol].iloc[-1]
            if np.isnan(t) or price < t:
                continue
            nv = norm_vel[symbol].iloc[-1]
            if np.isnan(nv):
                continue
            if nv < -self.entry_frac:
                scores[symbol] = nv

        return pd.Series(scores).sort_values()

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
        print(f"\n[{datetime.now()}] Running velocity MR strategy...")
        scores = self.get_momentum_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"Picks: {picks}")
        if not picks:
            print("No velocity MR candidates -- going to cash.")
            for pos in self.api.list_positions():
                sym = pos.symbol
                qty = float(pos.qty)
                if qty > 0:
                    try:
                        self.api.submit_order(
                            symbol=sym, qty=abs(qty), side="sell",
                            type="market",
                            time_in_force="gtc" if is_crypto(sym) else "day",
                        )
                    except Exception as e:
                        print(f"  Error closing {sym}: {e}")
            return

        current = {}
        for pos in self.api.list_positions():
            current[pos.symbol] = float(pos.qty)
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
                        type="market",
                        time_in_force="gtc" if is_crypto(symbol) else "day",
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
                elif diff < 0:
                    self.api.submit_order(symbol=symbol, qty=abs(diff), side="sell",
                                          type="market", time_in_force="gtc")
            except Exception as e:
                print(f"  Error trading {symbol}: {e}")
        print("Rebalance complete.")
