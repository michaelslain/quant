import itertools
import json
import os
import numpy as np
import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime, timedelta

PARAMS_DIR = os.path.join(os.path.dirname(__file__), "..", "params")


def _params_file():
    return os.path.join(PARAMS_DIR, "ema_crossover.json")


def _ema_crossover_worker(closes_vals, volumes_vals, fast_ema_np, slow_ema_np,
                          vol_avg_np, trend_np, kwargs, rebalance_every,
                          initial_cash=100_000.0):
    """Standalone backtest for EMA crossover strategy."""
    fast_period = kwargs["fast_period"]
    slow_period = kwargs["slow_period"]
    volume_mult = kwargs["volume_mult"]
    trend_window = kwargs["trend_window"]
    top_n = kwargs["top_n"]

    fast_ema = fast_ema_np[fast_period]
    slow_ema = slow_ema_np[slow_period]
    vol_avg = vol_avg_np.get(20)
    trend_sma = trend_np[trend_window]

    warmup = max(slow_period, trend_window) + 10
    n_rows = closes_vals.shape[0]
    n_cols = closes_vals.shape[1]
    rebal_indices = list(range(warmup, n_rows, rebalance_every))

    cash = initial_cash
    holdings = {}
    values = []

    for i in rebal_indices:
        port_value = cash
        for ci, qty in holdings.items():
            p = closes_vals[i, ci]
            if not np.isnan(p):
                port_value += qty * p
        values.append(port_value)

        scores = {}
        for ci in range(n_cols):
            # Trend filter
            t = trend_sma[i, ci]
            if np.isnan(t) or closes_vals[i, ci] < t:
                continue

            f = fast_ema[i, ci]
            s = slow_ema[i, ci]
            if np.isnan(f) or np.isnan(s):
                continue

            # Fast EMA must be above slow EMA (bullish crossover)
            if f <= s:
                continue

            # Volume confirmation
            if vol_avg is not None and volumes_vals is not None:
                v = volumes_vals[i, ci]
                va = vol_avg[i, ci]
                if va > 0 and v < volume_mult * va:
                    continue
                vol_ratio = v / va if va > 0 else 1.0
            else:
                vol_ratio = 1.0

            # Score: EMA spread (how strong the crossover is) * volume
            spread = (f - s) / s
            scores[ci] = spread * vol_ratio

        winners = sorted(scores, key=scores.get, reverse=True)[:top_n]

        for ci, qty in holdings.items():
            p = closes_vals[i, ci]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}

        if not winners:
            continue

        per_stock = cash / len(winners)
        for ci in winners:
            p = closes_vals[i, ci]
            if np.isnan(p) or p <= 0:
                continue
            qty = per_stock / p
            holdings[ci] = qty
            cash -= qty * p

    if len(values) < 2:
        return None

    pv = np.array(values)
    total_return = (pv[-1] - initial_cash) / initial_cash
    returns = np.diff(pv) / pv[:-1]
    std = returns.std()
    if std > 0:
        periods_per_year = (390 * 252) / rebalance_every
        sharpe = (returns.mean() / std) * np.sqrt(periods_per_year)
    else:
        sharpe = 0

    return {"total_return": total_return, "sharpe": sharpe}


class EmaCrossoverStrategy:
    """
    EMA crossover trend strategy for stocks.
    Buys when fast EMA crosses above slow EMA, confirmed by volume
    and a longer-term trend filter (price above SMA).
    Goes to cash when no crossover signals are active.
    """

    GRID = {
        "fast_period": [5, 10, 15, 20],
        "slow_period": [30, 60, 90, 120],
        "volume_mult": [1.0, 1.2],
        "trend_window": [120, 240, 480],
        "top_n": [1, 2, 3],
    }
    REBALANCE_OPTIONS = [5, 15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 fast_period: int = 10, slow_period: int = 60,
                 volume_mult: float = 1.0, trend_window: int = 240,
                 top_n: int = 2):
        self.api = api
        self.symbols = symbols
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.volume_mult = volume_mult
        self.trend_window = trend_window
        self.top_n = top_n
        self._load_params()

    def _load_params(self):
        pf = _params_file()
        if os.path.exists(pf):
            with open(pf) as f:
                p = json.load(f)
            self.fast_period = p.get("fast_period", self.fast_period)
            self.slow_period = p.get("slow_period", self.slow_period)
            self.volume_mult = p.get("volume_mult", self.volume_mult)
            self.trend_window = p.get("trend_window", self.trend_window)
            self.top_n = p.get("top_n", self.top_n)
            print(f"Loaded params from {pf}")

    def _save_params(self, rebalance_every=5, params_suffix=None):
        os.makedirs(PARAMS_DIR, exist_ok=True)
        pf = _params_file()
        if params_suffix:
            pf = pf.replace(".json", f"_{params_suffix}.json")
        params = {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "volume_mult": self.volume_mult,
            "trend_window": self.trend_window,
            "top_n": self.top_n,
            "rebalance_every": rebalance_every,
            "updated_at": datetime.now().isoformat(),
        }
        with open(pf, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Saved params to {pf}")

    def optimize(self, days: int = 7, fixed_interval: int = None,
                 params_suffix: str = None):
        print(f"Optimizing stock EMA crossover over {days} days of data...")
        history = self._fetch_history(days)
        if not history:
            print("No data -- keeping current params.")
            return

        closes = pd.DataFrame({sym: df["close"] for sym, df in history.items()})
        closes = closes.dropna(how="all").ffill()
        volumes = pd.DataFrame({sym: df["volume"] for sym, df in history.items() if "volume" in df.columns})
        volumes = volumes.reindex(closes.index).fillna(0) if not volumes.empty else None

        print("Precomputing indicators...")

        fast_ema_cache = {}
        for fp in self.GRID["fast_period"]:
            fast_ema_cache[fp] = closes.ewm(span=fp, min_periods=fp).mean()

        slow_ema_cache = {}
        for sp in self.GRID["slow_period"]:
            slow_ema_cache[sp] = closes.ewm(span=sp, min_periods=sp).mean()

        vol_cache = {}
        if volumes is not None:
            vol_cache[20] = volumes.rolling(20, min_periods=10).mean()

        trend_cache = {}
        for tw in self.GRID["trend_window"]:
            trend_cache[tw] = closes.rolling(tw, min_periods=20).mean()

        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from optimize import grid_search, find_best

        keys = list(self.GRID.keys())
        combos = list(itertools.product(*self.GRID.values()))

        closes_vals = closes.values
        volumes_vals = volumes.values if volumes is not None else None
        fast_ema_np = {k: v.values for k, v in fast_ema_cache.items()}
        slow_ema_np = {k: v.values for k, v in slow_ema_cache.items()}
        vol_avg_np = {k: v.values for k, v in vol_cache.items()}
        trend_np = {k: v.values for k, v in trend_cache.items()}

        jobs = []
        jobs_meta = []
        rebal_options = [fixed_interval] if fixed_interval else self.REBALANCE_OPTIONS
        for rebal in rebal_options:
            for vals in combos:
                kwargs = dict(zip(keys, vals))
                # Skip invalid combos where fast >= slow
                if kwargs["fast_period"] >= kwargs["slow_period"]:
                    continue
                jobs.append((closes_vals, volumes_vals, fast_ema_np, slow_ema_np,
                             vol_avg_np, trend_np, kwargs, rebal))
                jobs_meta.append((kwargs, rebal))

        results = grid_search(_ema_crossover_worker, jobs)
        best_params, best_rebal, best_sharpe, best_return = find_best(jobs_meta, results)

        if best_params:
            self.fast_period = best_params["fast_period"]
            self.slow_period = best_params["slow_period"]
            self.volume_mult = best_params["volume_mult"]
            self.trend_window = best_params["trend_window"]
            self.top_n = best_params["top_n"]
            label = "Optimal" if best_return > 0 else "Best (still negative)"
            print(f"\n{label} params found:")
            print(f"  fast_period={self.fast_period}, slow_period={self.slow_period}")
            print(f"  volume_mult={self.volume_mult}, trend_window={self.trend_window}")
            print(f"  top_n={self.top_n}")
            print(f"  rebalance_every={best_rebal}min")
            print(f"  Backtest return: {best_return:.2%} | Sharpe: {best_sharpe:.2f}")
            self._save_params(best_rebal, params_suffix=params_suffix)
            return best_rebal
        else:
            print("No valid params found -- keeping defaults.")
            return 5

    def _fetch_history(self, days: int, end_days_ago: int = 1) -> dict[str, pd.DataFrame]:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from backtest import fetch_history
        return fetch_history(self.api, self.symbols, days, end_days_ago=end_days_ago)

    def get_momentum_scores(self) -> pd.Series:
        """Return EMA crossover scores (matches interface name)."""
        end = datetime.now()
        lookback = max(self.slow_period, self.trend_window) + 30
        start = end - timedelta(minutes=lookback)

        scores = {}
        for symbol in self.symbols:
            try:
                bars = self.api.get_bars(
                    symbol,
                    tradeapi.TimeFrame.Minute,
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=lookback,
                ).df

                if len(bars) < self.slow_period:
                    continue

                closes = bars["close"]
                volumes = bars["volume"]

                # Trend filter
                trend_sma = closes.rolling(self.trend_window, min_periods=20).mean()
                if closes.iloc[-1] <= trend_sma.iloc[-1]:
                    continue

                fast_ema = closes.ewm(span=self.fast_period).mean()
                slow_ema = closes.ewm(span=self.slow_period).mean()

                if fast_ema.iloc[-1] <= slow_ema.iloc[-1]:
                    continue

                # Volume
                vol_avg = volumes.rolling(20).mean()
                if vol_avg.iloc[-1] > 0 and volumes.iloc[-1] < self.volume_mult * vol_avg.iloc[-1]:
                    continue

                spread = (fast_ema.iloc[-1] - slow_ema.iloc[-1]) / slow_ema.iloc[-1]
                vol_ratio = volumes.iloc[-1] / vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 1.0
                scores[symbol] = spread * vol_ratio

            except Exception as e:
                print(f"Error fetching {symbol}: {e}")

        return pd.Series(scores).sort_values(ascending=False)

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
                price = self.api.get_latest_trade(symbol).price
                qty = int(per_stock // price)
                if qty > 0:
                    targets[symbol] = qty
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")

        return targets

    def rebalance(self):
        import time as _time
        print(f"\n[{datetime.now()}] Running stock EMA-crossover strategy...")

        scores = self.get_momentum_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"Picks: {picks}")

        current = {}
        for pos in self.api.list_positions():
            current[pos.symbol] = int(float(pos.qty))

        if not picks:
            print("No EMA crossover candidates -- going to cash.")
            for symbol, qty in current.items():
                try:
                    self.api.submit_order(
                        symbol=symbol, qty=abs(qty), side="sell",
                        type="market", time_in_force="day",
                    )
                    print(f"  Closed {symbol} ({qty})")
                except Exception as e:
                    print(f"  Error closing {symbol}: {e}")
            return

        for symbol, qty in current.items():
            if symbol not in picks:
                print(f"  Closing {symbol} ({qty} shares)")
                try:
                    self.api.submit_order(
                        symbol=symbol, qty=abs(qty), side="sell",
                        type="market", time_in_force="day",
                    )
                except Exception as e:
                    print(f"  Error closing {symbol}: {e}")

        _time.sleep(2)

        account = self.api.get_account()
        cash = float(account.cash) * 0.95
        per_stock = cash / len(picks)

        for symbol in picks:
            current_qty = current.get(symbol, 0)
            try:
                price = self.api.get_latest_trade(symbol).price
                target_qty = int(per_stock // price)
                diff = target_qty - current_qty

                if diff > 0:
                    print(f"  Buying {diff} of {symbol}")
                    self.api.submit_order(
                        symbol=symbol, qty=diff, side="buy",
                        type="market", time_in_force="day",
                    )
                elif diff < 0:
                    print(f"  Selling {abs(diff)} of {symbol}")
                    self.api.submit_order(
                        symbol=symbol, qty=abs(diff), side="sell",
                        type="market", time_in_force="day",
                    )
                else:
                    print(f"  Holding {symbol} ({current_qty})")
            except Exception as e:
                print(f"  Error trading {symbol}: {e}")

        print("Rebalance complete.")
