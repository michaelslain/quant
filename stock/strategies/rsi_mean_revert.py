import itertools
import json
import os
import numpy as np
import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime, timedelta

PARAMS_DIR = os.path.join(os.path.dirname(__file__), "..", "params")


def _params_file():
    return os.path.join(PARAMS_DIR, "rsi_mean_revert.json")


def _stock_rsi_mr_worker(closes_vals, volumes_vals, cols, rsi_np, bb_np,
                         vol_avg_np, trend_np, kwargs, rebalance_every,
                         initial_cash=100_000.0):
    """Standalone backtest function for multiprocessing (stock RSI mean revert)."""
    rsi_period = kwargs["rsi_period"]
    rsi_oversold = kwargs["rsi_oversold"]
    bb_period = kwargs["bb_period"]
    bb_std = kwargs["bb_std"]
    volume_mult = kwargs["volume_mult"]
    top_n = kwargs["top_n"]
    trend_window = kwargs["trend_window"]

    rsi = rsi_np[rsi_period]
    bb = bb_np[(bb_period, bb_std)]
    vol_avg = vol_avg_np.get(20)
    trend_sma = trend_np[trend_window]

    warmup = max(rsi_period, bb_period, trend_window) + 10
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
            if np.isnan(trend_sma[i, ci]) or closes_vals[i, ci] < trend_sma[i, ci]:
                continue

            r = rsi[i, ci]
            if np.isnan(r) or i < 1:
                continue
            prev_r = rsi[i - 1, ci]
            if np.isnan(prev_r):
                continue

            if not (prev_r < rsi_oversold and r >= rsi_oversold):
                continue

            if not bb["near_lower"][i, ci]:
                continue

            if vol_avg is not None and volumes_vals is not None:
                v = volumes_vals[i, ci]
                va = vol_avg[i, ci]
                if va > 0 and v < volume_mult * va:
                    continue
                vol_ratio = v / va if va > 0 else 1.0
            else:
                vol_ratio = 1.0

            dip_depth = rsi_oversold - prev_r
            scores[ci] = dip_depth * vol_ratio

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

    final_value = cash
    if rebal_indices:
        last_i = min(closes_vals.shape[0] - 1, rebal_indices[-1] + rebalance_every)
        for ci, qty in holdings.items():
            p = closes_vals[last_i, ci]
            if not np.isnan(p):
                final_value += qty * p

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


class RsiMeanRevertStrategy:
    """
    RSI-based mean reversion strategy for stocks.
    Buys when RSI bounces from oversold territory with Bollinger Band
    and volume confirmation. Only takes bounces in an uptrend context.
    Goes to cash when no signals.
    """

    GRID = {
        "rsi_period": [10, 14, 20],
        "rsi_oversold": [25, 30, 35, 40],
        "bb_period": [15, 20, 30],
        "bb_std": [1.5, 2.0, 2.5],
        "volume_mult": [1.0, 1.2, 1.5],
        "trend_window": [120, 240, 480],
        "top_n": [1, 2, 3],
    }
    REBALANCE_OPTIONS = [5, 15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 rsi_period: int = 14, rsi_oversold: float = 30,
                 bb_period: int = 20, bb_std: float = 2.0,
                 volume_mult: float = 1.2, top_n: int = 2,
                 trend_window: int = 120):
        self.api = api
        self.symbols = symbols
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.volume_mult = volume_mult
        self.top_n = top_n
        self.trend_window = trend_window
        self._load_params()

    def _load_params(self):
        pf = _params_file()
        if os.path.exists(pf):
            with open(pf) as f:
                p = json.load(f)
            self.rsi_period = p.get("rsi_period", self.rsi_period)
            self.rsi_oversold = p.get("rsi_oversold", self.rsi_oversold)
            self.bb_period = p.get("bb_period", self.bb_period)
            self.bb_std = p.get("bb_std", self.bb_std)
            self.volume_mult = p.get("volume_mult", self.volume_mult)
            self.top_n = p.get("top_n", self.top_n)
            self.trend_window = p.get("trend_window", self.trend_window)
            print(f"Loaded params from {pf}")

    def _save_params(self, rebalance_every=5, params_suffix=None):
        os.makedirs(PARAMS_DIR, exist_ok=True)
        pf = _params_file()
        if params_suffix:
            pf = pf.replace(".json", f"_{params_suffix}.json")
        params = {
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "volume_mult": self.volume_mult,
            "top_n": self.top_n,
            "trend_window": self.trend_window,
            "rebalance_every": rebalance_every,
            "updated_at": datetime.now().isoformat(),
        }
        with open(pf, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Saved params to {pf}")

    @staticmethod
    def _compute_rsi(closes: pd.Series, period: int) -> pd.Series:
        delta = closes.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
        loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def optimize(self, days: int = 7, fixed_interval: int = None,
                 params_suffix: str = None):
        print(f"Optimizing stock RSI mean-revert over {days} days of data...")
        history = self._fetch_history(days)
        if not history:
            print("No data -- keeping current params.")
            return

        closes = pd.DataFrame({sym: df["close"] for sym, df in history.items()})
        closes = closes.dropna(how="all").ffill()
        volumes = pd.DataFrame({sym: df["volume"] for sym, df in history.items() if "volume" in df.columns})
        volumes = volumes.reindex(closes.index).fillna(0) if not volumes.empty else None

        print("Precomputing indicators...")

        rsi_cache = {}
        for rp in self.GRID["rsi_period"]:
            rsi_cache[rp] = closes.apply(lambda col: self._compute_rsi(col, rp))

        bb_cache = {}
        for bp in self.GRID["bb_period"]:
            sma = closes.rolling(bp, min_periods=10).mean()
            std = closes.rolling(bp, min_periods=10).std()
            for bs in self.GRID["bb_std"]:
                lower_bb = sma - bs * std
                near_lower = closes <= lower_bb * 1.01
                bb_cache[(bp, bs)] = {"near_lower": near_lower}

        vol_cache = {}
        if volumes is not None:
            vol_avg = volumes.rolling(20, min_periods=10).mean()
            vol_cache[20] = vol_avg

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
        rsi_np = {k: v.values for k, v in rsi_cache.items()}
        bb_np = {k: {"near_lower": v["near_lower"].values} for k, v in bb_cache.items()}
        vol_avg_np = {k: v.values for k, v in vol_cache.items()}
        trend_np = {k: v.values for k, v in trend_cache.items()}

        jobs = []
        jobs_meta = []
        rebal_options = [fixed_interval] if fixed_interval else self.REBALANCE_OPTIONS
        for rebal in rebal_options:
            for vals in combos:
                kwargs = dict(zip(keys, vals))
                jobs.append((closes_vals, volumes_vals, closes.columns.tolist(),
                             rsi_np, bb_np, vol_avg_np, trend_np, kwargs, rebal))
                jobs_meta.append((kwargs, rebal))

        results = grid_search(_stock_rsi_mr_worker, jobs)
        best_params, best_rebal, best_sharpe, best_return = find_best(jobs_meta, results)

        if best_params:
            self.rsi_period = best_params["rsi_period"]
            self.rsi_oversold = best_params["rsi_oversold"]
            self.bb_period = best_params["bb_period"]
            self.bb_std = best_params["bb_std"]
            self.volume_mult = best_params["volume_mult"]
            self.top_n = best_params["top_n"]
            self.trend_window = best_params["trend_window"]
            label = "Optimal" if best_return > 0 else "Best (still negative)"
            print(f"\n{label} params found:")
            print(f"  rsi_period={self.rsi_period}, rsi_oversold={self.rsi_oversold}")
            print(f"  bb_period={self.bb_period}, bb_std={self.bb_std}")
            print(f"  volume_mult={self.volume_mult}, top_n={self.top_n}")
            print(f"  trend_window={self.trend_window}")
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
        """Return RSI bounce scores (matches interface name)."""
        end = datetime.now()
        lookback = max(self.rsi_period, self.bb_period, self.trend_window) + 30
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

                if len(bars) < lookback - 10:
                    continue

                closes = bars["close"]
                volumes = bars["volume"]

                # Trend filter
                trend_sma = closes.rolling(self.trend_window, min_periods=20).mean()
                if closes.iloc[-1] <= trend_sma.iloc[-1]:
                    continue

                # RSI
                rsi = self._compute_rsi(closes, self.rsi_period)
                current_rsi = rsi.iloc[-1]
                prev_rsi = rsi.iloc[-2]
                if np.isnan(current_rsi) or np.isnan(prev_rsi):
                    continue

                # RSI bounce: was oversold, now crossing back up
                if not (prev_rsi < self.rsi_oversold and current_rsi >= self.rsi_oversold):
                    continue

                # Bollinger Band
                sma = closes.rolling(self.bb_period, min_periods=10).mean()
                std = closes.rolling(self.bb_period, min_periods=10).std()
                lower_bb = sma - self.bb_std * std
                if closes.iloc[-1] > lower_bb.iloc[-1] * 1.01:
                    continue

                # Volume
                vol_avg = volumes.rolling(20).mean()
                if vol_avg.iloc[-1] > 0 and volumes.iloc[-1] < self.volume_mult * vol_avg.iloc[-1]:
                    continue

                dip_depth = self.rsi_oversold - prev_rsi
                vol_ratio = volumes.iloc[-1] / vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 1.0
                scores[symbol] = dip_depth * vol_ratio

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
        print(f"\n[{datetime.now()}] Running stock RSI mean-revert strategy...")

        scores = self.get_momentum_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"Picks: {picks}")

        if not picks:
            print("No RSI bounce candidates found.")
            return

        current = {}
        for pos in self.api.list_positions():
            current[pos.symbol] = int(float(pos.qty))

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
