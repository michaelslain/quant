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
    return os.path.join(PARAMS_DIR, "bandit.json")


def _bandit_backtest_worker(closes_vals, cols, trend_np, kwargs,
                            rebalance_every, initial_cash=100_000.0):
    """Standalone backtest function for multiprocessing (top-level for pickling).

    Approximates Multi-Armed Bandit / UCB behaviour in a vectorised backtest:
      - At each rebalance point, compute each coin's average return over the
        last `reward_window` bars.
      - Count how many of those bars each coin was "selected" (return above
        `return_threshold`).
      - Apply UCB formula: avg_reward + exploration_factor * sqrt(ln(total) / played)
      - Pick top_n coins by UCB score, subject to a trend filter (price > SMA).
    """
    reward_window = kwargs["reward_window"]
    exploration_factor = kwargs["exploration_factor"]
    return_threshold = kwargs["return_threshold"]
    trend_window = kwargs["trend_window"]
    top_n = kwargs["top_n"]

    trend_sma = trend_np[trend_window]

    warmup = max(reward_window, trend_window) + 10
    n_rows = closes_vals.shape[0]
    n_cols = closes_vals.shape[1]
    rebal_indices = list(range(warmup, n_rows, rebalance_every))

    cash = initial_cash
    holdings = {}  # col_idx -> qty
    values = []

    for i in rebal_indices:
        # Portfolio value at this rebalance point
        port_value = cash
        for ci, qty in holdings.items():
            p = closes_vals[i, ci]
            if not np.isnan(p):
                port_value += qty * p
        values.append(port_value)

        # ----- UCB scoring -----
        window_start = max(0, i - reward_window)
        scores = {}

        for ci in range(n_cols):
            # Trend filter: price must be above longer-term SMA
            if np.isnan(trend_sma[i, ci]) or closes_vals[i, ci] < trend_sma[i, ci]:
                continue

            # Compute per-bar returns inside the reward window
            bar_returns = []
            for t in range(window_start + 1, i + 1):
                prev = closes_vals[t - 1, ci]
                cur = closes_vals[t, ci]
                if np.isnan(prev) or np.isnan(cur) or prev <= 0:
                    continue
                bar_returns.append((cur - prev) / prev)

            if len(bar_returns) == 0:
                continue

            avg_reward = np.mean(bar_returns)
            total_periods = len(bar_returns)
            played = sum(1 for r in bar_returns if r > return_threshold)

            # Only consider coins with positive recent returns
            if avg_reward <= 0:
                continue

            if played == 0:
                ucb_score = avg_reward + exploration_factor * 10.0
            else:
                ucb_score = avg_reward + exploration_factor * np.sqrt(
                    np.log(total_periods) / played
                )

            scores[ci] = ucb_score

        winners = sorted(scores, key=scores.get, reverse=True)[:top_n]

        # Liquidate current holdings
        for ci, qty in holdings.items():
            p = closes_vals[i, ci]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}

        # If no winners pass the trend filter, stay in cash
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
        last_i = min(n_rows - 1, rebal_indices[-1] + rebalance_every)
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


class BanditStrategy:
    """
    Multi-Armed Bandit / Upper Confidence Bound (UCB) strategy for crypto.

    Treats each coin as a slot-machine arm and balances exploitation (trade
    profitable coins) with exploration (try under-traded coins).

    UCB score = avg_reward + exploration_factor * sqrt(ln(total_periods) / periods_held)

    A trend filter (price > SMA) gates entry; if no coin passes the filter the
    strategy liquidates and holds cash.
    """

    GRID = {
        "reward_window": [15, 30, 60, 120],
        "exploration_factor": [0.01, 0.05, 0.1, 0.5],
        "return_threshold": [0.001, 0.002, 0.005, 0.01],
        "trend_window": [120, 240, 360],
        "top_n": [1, 2],
    }
    REBALANCE_OPTIONS = [5, 15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 reward_window: int = 60,
                 exploration_factor: float = 1.0,
                 return_threshold: float = 0.001,
                 trend_window: int = 120,
                 top_n: int = 2):
        self.api = api
        self.symbols = symbols
        self.reward_window = reward_window
        self.exploration_factor = exploration_factor
        self.return_threshold = return_threshold
        self.trend_window = trend_window
        self.top_n = top_n
        self._load_params()

    def _load_params(self):
        pf = _params_file()
        if os.path.exists(pf):
            with open(pf) as f:
                p = json.load(f)
            self.reward_window = p.get("reward_window", self.reward_window)
            self.exploration_factor = p.get("exploration_factor", self.exploration_factor)
            self.return_threshold = p.get("return_threshold", self.return_threshold)
            self.trend_window = p.get("trend_window", self.trend_window)
            self.top_n = p.get("top_n", self.top_n)
            print(f"Loaded params from {pf}")

    def _save_params(self, rebalance_every=5, params_suffix=None):
        os.makedirs(PARAMS_DIR, exist_ok=True)
        pf = _params_file()
        if params_suffix:
            pf = pf.replace(".json", f"_{params_suffix}.json")
        params = {
            "reward_window": self.reward_window,
            "exploration_factor": self.exploration_factor,
            "return_threshold": self.return_threshold,
            "trend_window": self.trend_window,
            "top_n": self.top_n,
            "rebalance_every": rebalance_every,
            "updated_at": datetime.now().isoformat(),
        }
        with open(pf, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Saved params to {pf}")

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize(self, days: int = 7, fixed_interval: int = None, params_suffix: str = None):
        print(f"Optimizing bandit strategy over {days} days of data...")
        history = self._fetch_history(days)
        if not history:
            print("No data -- keeping current params.")
            return

        closes = pd.DataFrame({sym: df["close"] for sym, df in history.items()})
        closes = closes.dropna(how="all").ffill()

        # Precompute trend SMAs
        print("Precomputing indicators...")
        trend_cache = {}
        for tw in self.GRID["trend_window"]:
            trend_cache[tw] = closes.rolling(tw, min_periods=20).mean()

        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from optimize import grid_search, find_best

        keys = list(self.GRID.keys())
        combos = list(itertools.product(*self.GRID.values()))

        # Convert to numpy for pickling
        closes_vals = closes.values
        trend_np = {k: v.values for k, v in trend_cache.items()}

        jobs = []
        jobs_meta = []
        rebal_options = [fixed_interval] if fixed_interval else self.REBALANCE_OPTIONS
        for rebal in rebal_options:
            for vals in combos:
                kwargs = dict(zip(keys, vals))
                jobs.append((closes_vals, closes.columns.tolist(),
                             trend_np, kwargs, rebal))
                jobs_meta.append((kwargs, rebal))

        results = grid_search(_bandit_backtest_worker, jobs)
        best_params, best_rebal, best_sharpe, best_return = find_best(jobs_meta, results)

        if best_params:
            self.reward_window = best_params["reward_window"]
            self.exploration_factor = best_params["exploration_factor"]
            self.return_threshold = best_params["return_threshold"]
            self.trend_window = best_params["trend_window"]
            self.top_n = best_params["top_n"]
            label = "Optimal" if best_return > 0 else "Best (still negative)"
            print(f"\n{label} params found:")
            print(f"  reward_window={self.reward_window}, "
                  f"exploration_factor={self.exploration_factor}")
            print(f"  return_threshold={self.return_threshold}, "
                  f"trend_window={self.trend_window}, top_n={self.top_n}")
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

    # ------------------------------------------------------------------
    # Live trading
    # ------------------------------------------------------------------

    def get_momentum_scores(self) -> pd.Series:
        """Return UCB scores for each symbol (matches interface name)."""
        end = datetime.now()
        lookback = self.reward_window + self.trend_window + 30
        start = end - timedelta(minutes=lookback)

        scores = {}
        total_symbols = len(self.symbols)

        for symbol in self.symbols:
            try:
                bars = self.api.get_crypto_bars(
                    symbol,
                    tradeapi.TimeFrame.Minute,
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=lookback,
                ).df

                if len(bars) < self.reward_window:
                    continue

                closes = bars["close"]

                # Trend filter: current price must be above SMA
                trend_sma = closes.rolling(self.trend_window, min_periods=20).mean()
                if closes.iloc[-1] <= trend_sma.iloc[-1]:
                    continue

                # Compute per-bar returns over the reward window
                recent = closes.iloc[-self.reward_window:]
                bar_returns = recent.pct_change().dropna()
                if len(bar_returns) == 0:
                    continue

                avg_reward = bar_returns.mean()
                total_periods = len(bar_returns)
                played = (bar_returns > self.return_threshold).sum()

                if played == 0:
                    ucb_score = avg_reward + self.exploration_factor * 10.0
                else:
                    ucb_score = avg_reward + self.exploration_factor * np.sqrt(
                        np.log(total_periods) / played
                    )

                scores[symbol] = ucb_score

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
        print(f"\n[{datetime.now()}] Running bandit strategy...")

        scores = self.get_momentum_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"Picks: {picks}")

        if not picks:
            print("No bandit candidates found -- going to cash.")
            # Liquidate everything when no coins pass the trend filter
            for pos in self.api.list_positions():
                sym = pos.symbol
                qty = float(pos.qty)
                if qty > 0:
                    print(f"  Closing {sym} ({qty} shares)")
                    try:
                        self.api.submit_order(
                            symbol=sym, qty=abs(qty), side="sell",
                            type="market",
                            time_in_force="gtc" if is_crypto(sym) else "day",
                        )
                    except Exception as e:
                        print(f"  Error closing {sym}: {e}")
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
                        type="market",
                        time_in_force="gtc" if is_crypto(symbol) else "day",
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
    print(f"Optimizing bandit strategy...\n")
    strategy = BanditStrategy(api=api, symbols=CRYPTO_SYMBOLS)
    strategy.optimize(days=days)
