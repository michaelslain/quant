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


def _params_file(symbols=None):
    return os.path.join(PARAMS_DIR, "mean_reversion.json")


def _mr_backtest_worker(closes_vals, dips_vals, trend_np, n_cols, kwargs, rebalance_every, initial_cash=100_000.0):
    """Standalone mean-reversion backtest for multiprocessing."""
    vwap_window = kwargs["vwap_window"]
    min_dip = kwargs["min_dip"]
    max_dip = kwargs["max_dip"]
    top_n = kwargs["top_n"]
    trend_window = kwargs["trend_window"]

    trend_sma = trend_np[trend_window]
    n_rows = closes_vals.shape[0]
    warmup = max(vwap_window, trend_window) + 10
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

        # Find symbols in the dip zone WITH trend filter
        winners = []
        for ci in range(n_cols):
            # Trend filter: only buy dips in an uptrend
            t = trend_sma[i, ci]
            if np.isnan(t) or closes_vals[i, ci] < t:
                continue
            d = dips_vals[i, ci]
            if np.isnan(d):
                continue
            if -max_dip <= d <= -min_dip:
                winners.append((ci, d))

        winners.sort(key=lambda x: x[1])
        winners = [ci for ci, _ in winners[:top_n]]

        # Always liquidate first
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
            if qty > 0:
                holdings[ci] = qty
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


class MeanReversionStrategy:
    """
    Self-tuning intraday mean-reversion strategy:
    - Find stocks that have dipped below their recent VWAP
    - Buy the most oversold ones (expecting a bounce back to VWAP)
    - Runs grid search to find optimal parameters on recent data
    """

    STOCK_GRID = {
        "vwap_window": [30, 60, 90],
        "min_dip": [0.002, 0.003, 0.005],
        "max_dip": [0.010, 0.015, 0.020],
        "top_n": [2, 3, 5],
    }
    CRYPTO_GRID = {
        "vwap_window": [30, 60, 120],
        "min_dip": [0.003, 0.005, 0.008, 0.010],
        "max_dip": [0.015, 0.020, 0.030, 0.040, 0.050],
        "top_n": [1, 2, 3],
        "trend_window": [60, 120, 240],
    }
    REBALANCE_OPTIONS = [5, 15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 vwap_window: int = 60, min_dip: float = 0.003,
                 max_dip: float = 0.015, top_n: int = 3,
                 trend_window: int = 120):
        self.api = api
        self.symbols = symbols
        self.vwap_window = vwap_window
        self.min_dip = min_dip
        self.max_dip = max_dip
        self.top_n = top_n
        self.trend_window = trend_window
        self._load_params()

    def _load_params(self):
        pf = _params_file(self.symbols)
        if os.path.exists(pf):
            with open(pf) as f:
                p = json.load(f)
            self.vwap_window = p.get("vwap_window", self.vwap_window)
            self.min_dip = p.get("min_dip", self.min_dip)
            self.max_dip = p.get("max_dip", self.max_dip)
            self.top_n = p.get("top_n", self.top_n)
            self.trend_window = p.get("trend_window", self.trend_window)
            print(f"Loaded params from {pf}")

    def _save_params(self, rebalance_every=30, params_suffix=None):
        os.makedirs(PARAMS_DIR, exist_ok=True)
        pf = _params_file(self.symbols)
        if params_suffix:
            pf = pf.replace(".json", f"_{params_suffix}.json")
        params = {
            "vwap_window": self.vwap_window,
            "min_dip": self.min_dip,
            "max_dip": self.max_dip,
            "top_n": self.top_n,
            "trend_window": self.trend_window,
            "rebalance_every": rebalance_every,
            "updated_at": datetime.now().isoformat(),
        }
        with open(pf, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Saved params to {pf}")

    def optimize(self, days: int = 7, fixed_interval: int = None, params_suffix: str = None):
        """Grid search over params on recent data."""
        print(f"Optimizing over {days} days of data...")
        history = self._fetch_history(days)
        if not history:
            print("No data — keeping current params.")
            return

        closes = pd.DataFrame({sym: df["close"] for sym, df in history.items()})
        closes = closes.dropna(how="all").ffill()
        volumes = pd.DataFrame({sym: df["volume"] for sym, df in history.items() if "volume" in df.columns})
        volumes = volumes.reindex(closes.index).fillna(0) if not volumes.empty else None

        is_crypto = any("/" in s for s in self.symbols)
        grid = self.CRYPTO_GRID if is_crypto else self.STOCK_GRID

        # Precompute rolling VWAPs for each window size
        print("Precomputing VWAP signals...")
        vwap_cache = {}
        dip_cache = {}
        for w in grid["vwap_window"]:
            if volumes is not None:
                roll_cv = (closes * volumes).rolling(w, min_periods=10).sum()
                roll_v = volumes.rolling(w, min_periods=10).sum()
                vwap = roll_cv / roll_v.replace(0, np.nan)
                vwap = vwap.fillna(closes.rolling(w, min_periods=10).mean())
            else:
                vwap = closes.rolling(w, min_periods=10).mean()
            vwap_cache[w] = vwap
            dip_cache[w] = (closes - vwap) / vwap

        # Precompute trend SMAs
        trend_cache = {}
        for tw in grid.get("trend_window", [120]):
            trend_cache[tw] = closes.rolling(tw, min_periods=20).mean()

        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from optimize import grid_search, find_best

        keys = list(grid.keys())
        combos = list(itertools.product(*grid.values()))

        # Convert to numpy for pickling
        closes_vals = closes.values
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
                jobs.append((closes_vals, dip_np[kwargs["vwap_window"]], trend_np, n_cols, kwargs, rebal))
                jobs_meta.append((kwargs, rebal))

        results = grid_search(_mr_backtest_worker, jobs)
        best_params, best_rebal, best_sharpe, best_return = find_best(jobs_meta, results)

        if best_params:
            self.vwap_window = best_params["vwap_window"]
            self.min_dip = best_params["min_dip"]
            self.max_dip = best_params["max_dip"]
            self.top_n = best_params["top_n"]
            self.trend_window = best_params["trend_window"]
            profitable = best_return > 0
            label = "Optimal" if profitable else "Best (still negative)"
            print(f"\n{label} params found:")
            print(f"  vwap_window={self.vwap_window}, min_dip={self.min_dip}, max_dip={self.max_dip}, top_n={self.top_n}")
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
        end = datetime.now()
        start = end - timedelta(minutes=self.vwap_window + 10)

        scores = {}
        for symbol in self.symbols:
            try:
                fetch = self.api.get_crypto_bars if is_crypto(symbol) else self.api.get_bars
                bars = fetch(
                    symbol,
                    tradeapi.TimeFrame.Minute,
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=self.vwap_window,
                ).df

                if len(bars) < 10:
                    continue

                if "volume" in bars.columns and bars["volume"].sum() > 0:
                    vwap = (bars["close"] * bars["volume"]).sum() / bars["volume"].sum()
                else:
                    vwap = bars["close"].mean()

                current_price = bars["close"].iloc[-1]
                dip = (current_price - vwap) / vwap

                if -self.max_dip <= dip <= -self.min_dip:
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
        cash = float(account.cash)
        per_stock = (cash * 0.95) / len(picks)

        targets = {}
        for symbol in picks.index:
            try:
                if is_crypto(symbol):
                    price = self.api.get_latest_crypto_trades([symbol])[symbol].price
                else:
                    price = self.api.get_latest_trade(symbol).price
                if is_crypto(symbol):
                    qty = round(per_stock / price, 4)
                else:
                    qty = int(per_stock // price)
                if qty > 0:
                    targets[symbol] = qty
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")

        return targets

    def rebalance(self):
        import time as _time
        print(f"\n[{datetime.now()}] Running mean-reversion strategy...")

        # Score symbols first
        scores = self.get_momentum_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"Picks: {picks}")

        if not picks:
            print("No dip candidates found.")
            return

        # Close positions not in picks
        current = {}
        for pos in self.api.list_positions():
            current[pos.symbol] = float(pos.qty)

        for symbol, qty in current.items():
            # Normalize: DOGEUSD -> DOGE/USD for comparison
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

        # Wait for sells to settle
        _time.sleep(2)

        # Now size buys based on fresh cash
        account = self.api.get_account()
        cash = float(account.cash) * 0.95
        per_stock = cash / len(picks)

        for symbol in picks:
            # Check if we already hold this
            held_sym = symbol.replace("/", "")
            current_qty = current.get(held_sym, current.get(symbol, 0))
            try:
                if is_crypto(symbol):
                    price = self.api.get_latest_crypto_trades([symbol])[symbol].price
                else:
                    price = self.api.get_latest_trade(symbol).price

                if is_crypto(symbol):
                    target_qty = round(per_stock / price, 4)
                else:
                    target_qty = int(per_stock // price)

                diff = target_qty - current_qty
                if diff > 0:
                    print(f"  Buying {diff} of {symbol}")
                    self.api.submit_order(
                        symbol=symbol, qty=diff, side="buy",
                        type="market", time_in_force="gtc" if is_crypto(symbol) else "day",
                    )
                elif diff < 0:
                    print(f"  Selling {abs(diff)} of {symbol}")
                    self.api.submit_order(
                        symbol=symbol, qty=abs(diff), side="sell",
                        type="market", time_in_force="gtc" if is_crypto(symbol) else "day",
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

    STOCK_SYMBOLS = [
        "TQQQ", "SOXL", "UPRO", "TNA", "LABU",
        "COIN", "HOOD", "SOFI", "MARA", "RIOT",
        "PLTR", "IONQ", "SMCI", "AFRM", "UPST",
    ]

    CRYPTO_SYMBOLS = [
        "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD",
        "XRP/USD", "ADA/USD", "LINK/USD", "LTC/USD",
    ]

    crypto = "--crypto" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--crypto"]
    symbols = CRYPTO_SYMBOLS if crypto else STOCK_SYMBOLS

    api = tradeapi.REST(
        key_id=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        base_url="https://paper-api.alpaca.markets",
    )

    days = int(args[0]) if args else 7
    mode = "crypto" if crypto else "stocks"
    print(f"Optimizing for {mode}...\n")
    strategy = MeanReversionStrategy(api=api, symbols=symbols)
    strategy.optimize(days=days)
