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
    return os.path.join(PARAMS_DIR, "regime_mr.json")


def _regime_mr_backtest_worker(closes_vals, dips_vals, trend_np, regime_np,
                               n_cols, kwargs, rebalance_every,
                               volumes_vals=None, vol_avg_np=None,
                               initial_cash=100_000.0):
    """Regime-aware mean-reversion backtest worker for multiprocessing."""
    min_dip = kwargs["min_dip"]
    max_dip = kwargs["max_dip"]
    top_n = kwargs["top_n"]
    trend_window = kwargs["trend_window"]
    take_profit = kwargs["take_profit"]
    stop_loss = kwargs["stop_loss"]
    max_hold = kwargs["max_hold"]
    regime_window = kwargs["regime_window"]
    volume_mult = kwargs.get("volume_mult", 0.0)

    trend_sma = trend_np[trend_window]
    regime = regime_np[regime_window]
    n_rows = closes_vals.shape[0]
    warmup = max(kwargs["vwap_window"], trend_window, regime_window) + 10
    rebal_indices = list(range(warmup, n_rows, rebalance_every))

    cash = initial_cash
    holdings = {}
    entry_prices = {}
    entry_bars = {}
    values = []

    for ri, i in enumerate(rebal_indices):
        # Between rebalances: check take-profit, stop-loss, max-hold per bar
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
                    # Take-profit: dip reverted toward VWAP
                    if take_profit > 0:
                        d = dips_vals[bar, ci]
                        if not np.isnan(d) and d >= -take_profit:
                            to_close.append(ci)
                            continue
                    # Max hold time
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

        # Regime gate: if BTC is bearish, liquidate and stay in cash
        if not regime[i]:
            for ci, qty in holdings.items():
                p = closes_vals[i, ci]
                if not np.isnan(p):
                    cash += qty * p
            holdings = {}
            entry_prices = {}
            entry_bars = {}
            continue

        # Find dip candidates with per-coin trend + volume filter
        winners = []
        for ci in range(n_cols):
            t = trend_sma[i, ci]
            if np.isnan(t) or closes_vals[i, ci] < t:
                continue
            d = dips_vals[i, ci]
            if np.isnan(d):
                continue
            if -max_dip <= d <= -min_dip:
                # Volume confirmation
                if volume_mult > 0 and vol_avg_np is not None and volumes_vals is not None:
                    v = volumes_vals[i, ci]
                    va = vol_avg_np[i, ci]
                    if va > 0 and v < volume_mult * va:
                        continue
                winners.append((ci, d))

        winners.sort(key=lambda x: x[1])
        winners = [ci for ci, _ in winners[:top_n]]

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

        per_stock = cash / len(winners)
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
    returns = np.diff(pv) / pv[:-1]
    std = returns.std()
    if std > 0:
        periods_per_year = (1440 * 365) / rebalance_every
        sharpe = (returns.mean() / std) * np.sqrt(periods_per_year)
    else:
        sharpe = 0

    return {"total_return": total_return, "sharpe": sharpe}


class RegimeMeanReversionStrategy:
    """
    Regime-aware mean-reversion for crypto:
    - BTC regime gate: only trades when BTC is above its long SMA
    - VWAP dip-buying with per-bar take-profit and stop-loss
    - Max hold time to prevent holding through extended drops
    - Stays in cash during bear markets
    """

    GRID = {
        "vwap_window": [15, 25, 40],
        "min_dip": [0.0005, 0.001, 0.002, 0.003],
        "max_dip": [0.008, 0.012, 0.015, 0.025],
        "take_profit": [0.002, 0.003, 0.004, 0.006],
        "stop_loss": [0.015, 0.025],
        "max_hold": [25, 45],
        "regime_window": [540, 720, 1080],
        "trend_window": [60, 120],
        "volume_mult": [0.0],
        "top_n": [1],
    }
    REBALANCE_OPTIONS = [15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 vwap_window: int = 60, min_dip: float = 0.005,
                 max_dip: float = 0.025, top_n: int = 1,
                 trend_window: int = 240, take_profit: float = 0.003,
                 stop_loss: float = 0.015, max_hold: int = 30,
                 regime_window: int = 720, volume_mult: float = 0.0):
        self.api = api
        self.symbols = symbols
        self.vwap_window = vwap_window
        self.min_dip = min_dip
        self.max_dip = max_dip
        self.top_n = top_n
        self.trend_window = trend_window
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.max_hold = max_hold
        self.regime_window = regime_window
        self.volume_mult = volume_mult
        self._load_params()

    def _load_params(self):
        pf = _params_file()
        if os.path.exists(pf):
            with open(pf) as f:
                p = json.load(f)
            for attr in ["vwap_window", "min_dip", "max_dip", "top_n",
                         "trend_window", "take_profit", "stop_loss",
                         "max_hold", "regime_window", "volume_mult"]:
                if attr in p:
                    setattr(self, attr, p[attr])
            print(f"Loaded params from {pf}")

    def _save_params(self, rebalance_every=5, params_suffix=None):
        os.makedirs(PARAMS_DIR, exist_ok=True)
        pf = _params_file()
        if params_suffix:
            pf = pf.replace(".json", f"_{params_suffix}.json")
        params = {
            "vwap_window": self.vwap_window,
            "min_dip": self.min_dip,
            "max_dip": self.max_dip,
            "top_n": self.top_n,
            "trend_window": self.trend_window,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "max_hold": self.max_hold,
            "regime_window": self.regime_window,
            "volume_mult": self.volume_mult,
            "rebalance_every": rebalance_every,
            "updated_at": datetime.now().isoformat(),
        }
        with open(pf, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Saved params to {pf}")

    def optimize(self, days: int = 7, fixed_interval: int = None, params_suffix: str = None):
        print(f"Optimizing regime mean-reversion over {days} days of data...")
        history = self._fetch_history(days)
        if not history:
            print("No data — keeping current params.")
            return

        closes = pd.DataFrame({sym: df["close"] for sym, df in history.items()})
        closes = closes.dropna(how="all").ffill()
        volumes = pd.DataFrame({sym: df["volume"] for sym, df in history.items() if "volume" in df.columns})
        volumes = volumes.reindex(closes.index).fillna(0) if not volumes.empty else None

        grid = self.GRID

        # Precompute VWAP and dips for each window
        print("Precomputing signals...")
        dip_cache = {}
        for w in grid["vwap_window"]:
            if volumes is not None:
                roll_cv = (closes * volumes).rolling(w, min_periods=10).sum()
                roll_v = volumes.rolling(w, min_periods=10).sum()
                vwap = roll_cv / roll_v.replace(0, np.nan)
                vwap = vwap.fillna(closes.rolling(w, min_periods=10).mean())
            else:
                vwap = closes.rolling(w, min_periods=10).mean()
            dip_cache[w] = (closes - vwap) / vwap

        # Precompute trend SMAs
        trend_cache = {}
        for tw in grid["trend_window"]:
            trend_cache[tw] = closes.rolling(tw, min_periods=20).mean()

        # Precompute BTC regime filter (BTC = first column)
        btc_closes = closes.iloc[:, 0]
        regime_cache = {}
        for rw in grid["regime_window"]:
            btc_sma = btc_closes.rolling(rw, min_periods=50).mean()
            regime_cache[rw] = (btc_closes > btc_sma).values

        # Precompute volume averages
        vol_avg = None
        if volumes is not None:
            vol_avg = volumes.rolling(20, min_periods=10).mean()

        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from optimize import grid_search, find_best

        keys = list(grid.keys())
        combos = list(itertools.product(*grid.values()))

        closes_vals = closes.values
        volumes_vals = volumes.values if volumes is not None else None
        vol_avg_np = vol_avg.values if vol_avg is not None else None
        n_cols = closes_vals.shape[1]
        dip_np = {w: v.values for w, v in dip_cache.items()}
        trend_np = {w: v.values for w, v in trend_cache.items()}

        jobs = []
        jobs_meta = []
        rebal_options = [fixed_interval] if fixed_interval else self.REBALANCE_OPTIONS
        for rebal in rebal_options:
            for vals in combos:
                kwargs = dict(zip(keys, vals))
                if kwargs["min_dip"] >= kwargs["max_dip"]:
                    continue
                jobs.append((closes_vals, dip_np[kwargs["vwap_window"]],
                             trend_np, regime_cache, n_cols, kwargs, rebal,
                             volumes_vals, vol_avg_np))
                jobs_meta.append((kwargs, rebal))

        print(f"Running {len(jobs)} parameter combinations...")
        results = grid_search(_regime_mr_backtest_worker, jobs)
        best_params, best_rebal, best_sharpe, best_return = find_best(jobs_meta, results)

        if best_params:
            for attr in ["vwap_window", "min_dip", "max_dip", "top_n",
                         "trend_window", "take_profit", "stop_loss",
                         "max_hold", "regime_window", "volume_mult"]:
                setattr(self, attr, best_params[attr])
            label = "Optimal" if best_return > 0 else "Best (still negative)"
            print(f"\n{label} params found:")
            print(f"  vwap_window={self.vwap_window}, min_dip={self.min_dip}, max_dip={self.max_dip}")
            print(f"  take_profit={self.take_profit}, stop_loss={self.stop_loss}, max_hold={self.max_hold}")
            print(f"  regime_window={self.regime_window}, trend_window={self.trend_window}, top_n={self.top_n}, volume_mult={self.volume_mult}")
            print(f"  rebalance_every={best_rebal}min")
            print(f"  Backtest return: {best_return:.2%} | Sharpe: {best_sharpe:.2f}")
            self._save_params(best_rebal, params_suffix=params_suffix)
            return best_rebal
        else:
            print("No valid params found — keeping defaults.")
            return 5

    def _fetch_history(self, days: int, end_days_ago: int = 1) -> dict[str, pd.DataFrame]:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from backtest import fetch_history
        return fetch_history(self.api, self.symbols, days, end_days_ago=end_days_ago)

    # --- Live trading methods ---

    def _check_regime(self) -> bool:
        """Check if BTC is above its regime SMA (market not in downtrend)."""
        end = datetime.now()
        start = end - timedelta(minutes=self.regime_window + 30)
        try:
            bars = self.api.get_crypto_bars(
                "BTC/USD",
                tradeapi.TimeFrame.Minute,
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
        """Score each symbol by VWAP dip depth (for compare command)."""
        if not self._check_regime():
            print("  BTC regime: BEARISH — staying in cash")
            return pd.Series(dtype=float)

        end = datetime.now()
        start = end - timedelta(minutes=self.vwap_window + self.trend_window + 10)

        scores = {}
        for symbol in self.symbols:
            try:
                bars = self.api.get_crypto_bars(
                    symbol,
                    tradeapi.TimeFrame.Minute,
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=self.vwap_window + self.trend_window,
                ).df

                if len(bars) < self.vwap_window:
                    continue

                closes = bars["close"]
                volumes = bars["volume"]

                # Trend filter
                trend_sma = closes.rolling(self.trend_window, min_periods=20).mean()
                if closes.iloc[-1] < trend_sma.iloc[-1]:
                    continue

                # VWAP dip
                if volumes.sum() > 0:
                    vwap = (closes * volumes).rolling(self.vwap_window, min_periods=10).sum() / \
                           volumes.rolling(self.vwap_window, min_periods=10).sum()
                else:
                    vwap = closes.rolling(self.vwap_window, min_periods=10).mean()

                current_price = closes.iloc[-1]
                dip = (current_price - vwap.iloc[-1]) / vwap.iloc[-1]

                if -self.max_dip <= dip <= -self.min_dip:
                    # Volume confirmation
                    if self.volume_mult > 0:
                        vol_avg = volumes.rolling(20).mean()
                        if vol_avg.iloc[-1] > 0 and volumes.iloc[-1] < self.volume_mult * vol_avg.iloc[-1]:
                            continue
                    scores[symbol] = dip

            except Exception as e:
                print(f"Error fetching {symbol}: {e}")

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
        print(f"\n[{datetime.now()}] Running regime mean-reversion strategy...")

        # Regime gate
        if not self._check_regime():
            print("BTC regime: BEARISH — liquidating to cash.")
            current = {}
            for pos in self.api.list_positions():
                current[pos.symbol] = float(pos.qty)
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
            print("Rebalance complete (all cash — bearish regime).")
            return

        scores = self.get_momentum_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"Dip picks: {picks}")

        current = {}
        for pos in self.api.list_positions():
            current[pos.symbol] = float(pos.qty)

        if not picks:
            print("No dip candidates — liquidating to cash.")
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
