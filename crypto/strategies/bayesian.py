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
    return os.path.join(PARAMS_DIR, "bayesian.json")


def _bayesian_backtest_worker(closes_vals, volumes_vals, cols, sma_np, rsi_np,
                               vol_avg_np, trend_np, kwargs, rebalance_every,
                               initial_cash=100_000.0):
    """Standalone Bayesian backtest function for multiprocessing."""
    sma_period = kwargs["sma_period"]
    rsi_period = kwargs["rsi_period"]
    momentum_weight = kwargs["momentum_weight"]
    volume_weight = kwargs["volume_weight"]
    rsi_weight = kwargs["rsi_weight"]
    threshold = kwargs["threshold"]
    trend_window = kwargs["trend_window"]
    top_n = kwargs["top_n"]

    sma = sma_np[sma_period]
    rsi = rsi_np[rsi_period]
    vol_avg = vol_avg_np.get(20)
    trend_sma = trend_np[trend_window]

    warmup = max(sma_period, rsi_period, trend_window) + 10
    n_rows = closes_vals.shape[0]
    n_cols = closes_vals.shape[1]
    rebal_indices = list(range(warmup, n_rows, rebalance_every))

    cash = initial_cash
    holdings = {}  # col_idx -> qty
    values = []

    for i in rebal_indices:
        port_value = cash
        for ci, qty in holdings.items():
            p = closes_vals[i, ci]
            if not np.isnan(p):
                port_value += qty * p
        values.append(port_value)

        posteriors = {}
        for ci in range(n_cols):
            # Trend filter: only trade if price is above longer-term SMA
            if np.isnan(trend_sma[i, ci]) or closes_vals[i, ci] < trend_sma[i, ci]:
                continue

            # Start with prior of 0.5 (log-odds = 0)
            log_odds = 0.0

            # Evidence 1: Momentum signal (price vs SMA)
            sma_val = sma[i, ci]
            price = closes_vals[i, ci]
            if np.isnan(sma_val) or np.isnan(price) or sma_val <= 0:
                continue
            momentum_signal = (price - sma_val) / sma_val
            log_odds += momentum_weight * momentum_signal * 10  # scale for meaningful update

            # Evidence 2: Volume signal
            if vol_avg is not None and volumes_vals is not None:
                v = volumes_vals[i, ci]
                va = vol_avg[i, ci]
                if va > 0 and not np.isnan(va):
                    volume_signal = (v - va) / va  # positive if above avg
                    log_odds += volume_weight * volume_signal
                else:
                    volume_signal = 0.0
            else:
                volume_signal = 0.0

            # Evidence 3: RSI signal
            rsi_val = rsi[i, ci]
            if np.isnan(rsi_val):
                continue
            # Map RSI to signal: RSI 50 -> 0, RSI 70 -> +1, RSI 30 -> -1
            rsi_signal = (rsi_val - 50) / 20.0
            log_odds += rsi_weight * rsi_signal

            # Convert log-odds back to posterior probability
            posterior = 1.0 / (1.0 + np.exp(-log_odds))
            if posterior >= threshold:
                posteriors[ci] = posterior

        winners = sorted(posteriors, key=posteriors.get, reverse=True)[:top_n]

        # Always liquidate current holdings first
        for ci, qty in holdings.items():
            p = closes_vals[i, ci]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}

        # If no winners, stay in cash
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

    # Final portfolio value
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
        periods_per_year = (1440 * 365) / rebalance_every
        sharpe = (returns.mean() / std) * np.sqrt(periods_per_year)
    else:
        sharpe = 0

    return {"total_return": total_return, "sharpe": sharpe}


class BayesianStrategy:
    """
    Bayesian probability strategy for crypto trading.
    Maintains a posterior probability that each coin will go up, updated via
    Bayes' theorem in log-odds form using momentum, volume, and RSI evidence.
    Buys coins with the highest posterior probability above a threshold.
    """

    GRID = {
        "sma_period": [10, 20],
        "rsi_period": [14, 20],
        "momentum_weight": [1.0, 2.0],
        "volume_weight": [0.0, 0.5],
        "rsi_weight": [0.5, 1.0],
        "threshold": [0.65, 0.70, 0.75],
        "trend_window": [120, 240],
        "top_n": [1, 2],
    }
    REBALANCE_OPTIONS = [5, 15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 sma_period: int = 20, rsi_period: int = 14,
                 momentum_weight: float = 1.0, volume_weight: float = 0.5,
                 rsi_weight: float = 0.5, threshold: float = 0.6,
                 trend_window: int = 120, top_n: int = 2):
        self.api = api
        self.symbols = symbols
        self.sma_period = sma_period
        self.rsi_period = rsi_period
        self.momentum_weight = momentum_weight
        self.volume_weight = volume_weight
        self.rsi_weight = rsi_weight
        self.threshold = threshold
        self.trend_window = trend_window
        self.top_n = top_n
        self._load_params()

    def _load_params(self):
        pf = _params_file()
        if os.path.exists(pf):
            with open(pf) as f:
                p = json.load(f)
            self.sma_period = p.get("sma_period", self.sma_period)
            self.rsi_period = p.get("rsi_period", self.rsi_period)
            self.momentum_weight = p.get("momentum_weight", self.momentum_weight)
            self.volume_weight = p.get("volume_weight", self.volume_weight)
            self.rsi_weight = p.get("rsi_weight", self.rsi_weight)
            self.threshold = p.get("threshold", self.threshold)
            self.trend_window = p.get("trend_window", self.trend_window)
            self.top_n = p.get("top_n", self.top_n)
            print(f"Loaded params from {pf}")

    def _save_params(self, rebalance_every=5, params_suffix=None):
        os.makedirs(PARAMS_DIR, exist_ok=True)
        pf = _params_file()
        if params_suffix:
            pf = pf.replace(".json", f"_{params_suffix}.json")
        params = {
            "sma_period": self.sma_period,
            "rsi_period": self.rsi_period,
            "momentum_weight": self.momentum_weight,
            "volume_weight": self.volume_weight,
            "rsi_weight": self.rsi_weight,
            "threshold": self.threshold,
            "trend_window": self.trend_window,
            "top_n": self.top_n,
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

    def optimize(self, days: int = 7, fixed_interval: int = None, params_suffix: str = None):
        print(f"Optimizing Bayesian strategy over {days} days of data...")
        history = self._fetch_history(days)
        if not history:
            print("No data — keeping current params.")
            return

        closes = pd.DataFrame({sym: df["close"] for sym, df in history.items()})
        closes = closes.dropna(how="all").ffill()
        volumes = pd.DataFrame({sym: df["volume"] for sym, df in history.items() if "volume" in df.columns})
        volumes = volumes.reindex(closes.index).fillna(0) if not volumes.empty else None

        # Precompute indicators for each parameter combo
        print("Precomputing indicators...")

        # SMA cache for momentum signal
        sma_cache = {}
        for sp in self.GRID["sma_period"]:
            sma_cache[sp] = closes.rolling(sp, min_periods=5).mean()

        # RSI cache
        rsi_cache = {}
        for rp in self.GRID["rsi_period"]:
            rsi_cache[rp] = closes.apply(lambda col: self._compute_rsi(col, rp))

        # Volume average cache
        vol_cache = {}
        if volumes is not None:
            vol_avg = volumes.rolling(20, min_periods=10).mean()
            vol_cache[20] = vol_avg

        # Trend SMA cache
        trend_cache = {}
        for tw in self.GRID["trend_window"]:
            trend_cache[tw] = closes.rolling(tw, min_periods=20).mean()

        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from optimize import grid_search, find_best

        keys = list(self.GRID.keys())
        combos = list(itertools.product(*self.GRID.values()))

        # Convert caches to numpy for pickling
        closes_vals = closes.values
        volumes_vals = volumes.values if volumes is not None else None
        sma_np = {k: v.values for k, v in sma_cache.items()}
        rsi_np = {k: v.values for k, v in rsi_cache.items()}
        vol_avg_np = {k: v.values for k, v in vol_cache.items()}
        trend_np = {k: v.values for k, v in trend_cache.items()}

        jobs = []
        jobs_meta = []
        rebal_options = [fixed_interval] if fixed_interval else self.REBALANCE_OPTIONS
        for rebal in rebal_options:
            for vals in combos:
                kwargs = dict(zip(keys, vals))
                jobs.append((closes_vals, volumes_vals, closes.columns.tolist(),
                             sma_np, rsi_np, vol_avg_np, trend_np, kwargs, rebal))
                jobs_meta.append((kwargs, rebal))

        results = grid_search(_bayesian_backtest_worker, jobs)
        best_params, best_rebal, best_sharpe, best_return = find_best(jobs_meta, results)

        if best_params:
            self.sma_period = best_params["sma_period"]
            self.rsi_period = best_params["rsi_period"]
            self.momentum_weight = best_params["momentum_weight"]
            self.volume_weight = best_params["volume_weight"]
            self.rsi_weight = best_params["rsi_weight"]
            self.threshold = best_params["threshold"]
            self.trend_window = best_params["trend_window"]
            self.top_n = best_params["top_n"]
            label = "Optimal" if best_return > 0 else "Best (still negative)"
            print(f"\n{label} params found:")
            print(f"  sma_period={self.sma_period}, rsi_period={self.rsi_period}")
            print(f"  momentum_weight={self.momentum_weight}, volume_weight={self.volume_weight}")
            print(f"  rsi_weight={self.rsi_weight}, threshold={self.threshold}")
            print(f"  trend_window={self.trend_window}, top_n={self.top_n}")
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

    def get_momentum_scores(self) -> pd.Series:
        """Compute Bayesian posterior for each coin. Named get_momentum_scores for interface compatibility."""
        end = datetime.now()
        lookback = max(self.sma_period, self.rsi_period, self.trend_window) + 30
        start = end - timedelta(minutes=lookback)

        scores = {}
        for symbol in self.symbols:
            try:
                bars = self.api.get_crypto_bars(
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
                if closes.iloc[-1] < trend_sma.iloc[-1]:
                    continue

                # Start with prior of 0.5 (log-odds = 0)
                log_odds = 0.0

                # Evidence 1: Momentum signal (price vs SMA)
                sma = closes.rolling(self.sma_period, min_periods=5).mean()
                sma_val = sma.iloc[-1]
                price = closes.iloc[-1]
                if np.isnan(sma_val) or sma_val <= 0:
                    continue
                momentum_signal = (price - sma_val) / sma_val
                log_odds += self.momentum_weight * momentum_signal * 10

                # Evidence 2: Volume signal
                vol_avg = volumes.rolling(20, min_periods=10).mean()
                va = vol_avg.iloc[-1]
                v = volumes.iloc[-1]
                if va > 0 and not np.isnan(va):
                    volume_signal = (v - va) / va
                    log_odds += self.volume_weight * volume_signal
                else:
                    volume_signal = 0.0

                # Evidence 3: RSI signal
                rsi = self._compute_rsi(closes, self.rsi_period)
                rsi_val = rsi.iloc[-1]
                if np.isnan(rsi_val):
                    continue
                rsi_signal = (rsi_val - 50) / 20.0
                log_odds += self.rsi_weight * rsi_signal

                # Convert log-odds to posterior
                posterior = 1.0 / (1.0 + np.exp(-log_odds))

                if posterior >= self.threshold:
                    scores[symbol] = posterior

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
                price = self.api.get_latest_crypto_trades([symbol])[symbol].price
                qty = round(per_stock / price, 4)
                if qty > 0:
                    targets[symbol] = qty
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")

        return targets

    def rebalance(self):
        import time as _time
        print(f"\n[{datetime.now()}] Running Bayesian strategy...")

        scores = self.get_momentum_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"Picks: {picks}")

        if not picks:
            print("No Bayesian candidates found — going to cash.")

        # Close positions not in picks
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
                print(f"  Closing {symbol} ({qty} shares)")
                try:
                    self.api.submit_order(
                        symbol=symbol, qty=abs(qty), side="sell",
                        type="market", time_in_force="gtc" if is_crypto(symbol) else "day",
                    )
                except Exception as e:
                    print(f"  Error closing {symbol}: {e}")

        if not picks:
            return

        _time.sleep(2)

        # Size buys with fresh cash
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


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

    CRYPTO_SYMBOLS = [
        "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD",
        "XRP/USD", "ADA/USD", "LINK/USD", "LTC/USD",
    ]

    args = [a for a in sys.argv[1:]]
    api = tradeapi.REST(
        key_id=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        base_url="https://paper-api.alpaca.markets",
    )

    days = int(args[0]) if args else 7
    print(f"Optimizing Bayesian strategy...\n")
    strategy = BayesianStrategy(api=api, symbols=CRYPTO_SYMBOLS)
    strategy.optimize(days=days)
