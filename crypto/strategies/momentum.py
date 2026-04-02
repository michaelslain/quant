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
    return os.path.join(PARAMS_DIR, "momentum.json")


def _backtest_worker(closes_vals, volumes_vals, cols, rsi_np, bb_np, vol_avg_np,
                      trend_np, kwargs, rebalance_every, initial_cash=100_000.0):
    """Standalone backtest function for multiprocessing."""
    rsi_period = kwargs["rsi_period"]
    rsi_threshold = kwargs["rsi_threshold"]
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
    holdings = {}  # col_idx -> qty
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
            # Trend filter: only trade if price is above longer-term SMA
            if np.isnan(trend_sma[i, ci]) or closes_vals[i, ci] < trend_sma[i, ci]:
                continue

            r = rsi[i, ci]
            if np.isnan(r) or r < rsi_threshold:
                continue
            if not bb["in_upper_half"][i, ci]:
                continue
            if not bb["width_expanding"][i, ci]:
                continue
            if vol_avg is not None and volumes_vals is not None:
                v = volumes_vals[i, ci]
                va = vol_avg[i, ci]
                if va > 0 and v < volume_mult * va:
                    continue
                vol_ratio = v / va if va > 0 else 1.0
            else:
                vol_ratio = 1.0
            scores[ci] = (r - 50) * vol_ratio

        winners = sorted(scores, key=scores.get, reverse=True)[:top_n]

        # Always liquidate current holdings first
        for ci, qty in holdings.items():
            p = closes_vals[i, ci]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}

        # If no winners, stay in cash (don't re-enter)
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

    # Add final value (including any remaining holdings)
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


class CryptoMomentumStrategy:
    """
    Crypto-specific momentum-volatility breakout strategy.
    Buys coins showing strong upward momentum with volume confirmation
    and expanding volatility (Bollinger Band breakouts).
    Crypto trends more than it mean-reverts — this trades WITH the trend.
    """

    GRID = {
        "rsi_period": [10, 14, 20],
        "rsi_threshold": [45, 50, 55, 60],
        "bb_period": [15, 20, 30],
        "bb_std": [1.5, 2.0, 2.5],
        "volume_mult": [1.0, 1.2, 1.5],
        "top_n": [1, 2, 3],
        "trend_window": [60, 120, 240],
    }
    REBALANCE_OPTIONS = [5, 15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 rsi_period: int = 14, rsi_threshold: float = 53,
                 bb_period: int = 20, bb_std: float = 2.0,
                 volume_mult: float = 1.3, top_n: int = 3,
                 trend_window: int = 120):
        self.api = api
        self.symbols = symbols
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
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
            self.rsi_threshold = p.get("rsi_threshold", self.rsi_threshold)
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
            "rsi_threshold": self.rsi_threshold,
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

    def optimize(self, days: int = 7, fixed_interval: int = None, params_suffix: str = None):
        print(f"Optimizing crypto momentum over {days} days of data...")
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
        rsi_cache = {}
        for rp in self.GRID["rsi_period"]:
            rsi_cache[rp] = closes.apply(lambda col: self._compute_rsi(col, rp))

        bb_cache = {}
        for bp in self.GRID["bb_period"]:
            sma = closes.rolling(bp, min_periods=10).mean()
            std = closes.rolling(bp, min_periods=10).std()
            for bs in self.GRID["bb_std"]:
                upper = sma + bs * std
                lower = sma - bs * std
                width = (upper - lower) / sma
                width_expanding = width > width.rolling(5, min_periods=3).mean()
                in_upper_half = closes > sma
                bb_cache[(bp, bs)] = {
                    "upper": upper,
                    "width_expanding": width_expanding,
                    "in_upper_half": in_upper_half,
                }

        vol_cache = {}
        if volumes is not None:
            vol_avg = volumes.rolling(20, min_periods=10).mean()
            vol_cache[20] = vol_avg

        # Precompute trend SMAs
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
        rsi_np = {k: v.values for k, v in rsi_cache.items()}
        bb_np = {k: {
            "in_upper_half": v["in_upper_half"].values,
            "width_expanding": v["width_expanding"].values,
        } for k, v in bb_cache.items()}
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

        results = grid_search(_backtest_worker, jobs)
        best_params, best_rebal, best_sharpe, best_return = find_best(jobs_meta, results)

        if best_params:
            self.rsi_period = best_params["rsi_period"]
            self.rsi_threshold = best_params["rsi_threshold"]
            self.bb_period = best_params["bb_period"]
            self.bb_std = best_params["bb_std"]
            self.volume_mult = best_params["volume_mult"]
            self.top_n = best_params["top_n"]
            self.trend_window = best_params["trend_window"]
            label = "Optimal" if best_return > 0 else "Best (still negative)"
            print(f"\n{label} params found:")
            print(f"  rsi_period={self.rsi_period}, rsi_threshold={self.rsi_threshold}")
            print(f"  bb_period={self.bb_period}, bb_std={self.bb_std}")
            print(f"  volume_mult={self.volume_mult}, top_n={self.top_n}, trend_window={self.trend_window}")
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
        end = datetime.now()
        lookback = max(self.rsi_period, self.bb_period) + 30
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

                # RSI
                rsi = self._compute_rsi(closes, self.rsi_period)
                current_rsi = rsi.iloc[-1]
                if np.isnan(current_rsi) or current_rsi < self.rsi_threshold:
                    continue

                # Bollinger Bands
                sma = closes.rolling(self.bb_period).mean()
                std = closes.rolling(self.bb_period).std()
                upper = sma + self.bb_std * std
                lower = sma - self.bb_std * std
                width = (upper - lower) / sma

                # In upper half of BB
                if closes.iloc[-1] <= sma.iloc[-1]:
                    continue

                # Volatility expanding
                width_avg = width.rolling(5).mean()
                if width.iloc[-1] <= width_avg.iloc[-1]:
                    continue

                # Volume confirmation
                vol_avg = volumes.rolling(20).mean()
                if vol_avg.iloc[-1] > 0 and volumes.iloc[-1] < self.volume_mult * vol_avg.iloc[-1]:
                    continue

                # Score: RSI strength * volume ratio
                vol_ratio = volumes.iloc[-1] / vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 1.0
                scores[symbol] = (current_rsi - 50) * vol_ratio

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
        print(f"\n[{datetime.now()}] Running crypto-momentum strategy...")

        scores = self.get_momentum_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"Picks: {picks}")

        if not picks:
            print("No momentum candidates found.")
            return

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
    print(f"Optimizing crypto momentum...\n")
    strategy = CryptoMomentumStrategy(api=api, symbols=CRYPTO_SYMBOLS)
    strategy.optimize(days=days)
