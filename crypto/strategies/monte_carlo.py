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
    return os.path.join(PARAMS_DIR, "monte_carlo.json")


def _mc_backtest_worker(closes_vals, volumes_vals, cols, trend_np, kwargs,
                        rebalance_every, initial_cash=100_000.0):
    """Bootstrap Monte Carlo backtest: resample actual returns, score by risk-adjusted metrics."""
    np.random.seed(42)

    lookback = kwargs["lookback"]
    n_simulations = kwargs["n_simulations"]
    horizon = kwargs["horizon"]
    volume_weight = kwargs["volume_weight"]
    trend_window = kwargs["trend_window"]
    top_n = kwargs["top_n"]

    trend_sma = trend_np[trend_window]

    warmup = max(lookback, trend_window) + 10
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

            if i < lookback:
                continue

            price_window = closes_vals[i - lookback:i + 1, ci]
            if np.any(np.isnan(price_window)):
                continue

            current_price = price_window[-1]
            if current_price <= 0:
                continue

            returns = np.diff(price_window) / price_window[:-1]
            if len(returns) < 10 or np.std(returns) == 0:
                continue

            # Bootstrap resampling: draw actual historical returns with replacement
            # This preserves fat tails, skew, and non-normality
            indices = np.random.randint(0, len(returns), size=(n_simulations, horizon))
            sim_returns = returns[indices]  # (n_simulations, horizon)
            cumulative = np.cumprod(1.0 + sim_returns, axis=1)
            final_returns = cumulative[:, -1] - 1.0  # (n_simulations,)

            # Risk-adjusted scoring:
            # - P(profit): fraction of paths that are profitable
            # - median return: robust central tendency (not skewed by outliers)
            # - downside deviation: only penalize negative outcomes
            p_profit = np.mean(final_returns > 0)
            median_return = np.median(final_returns)
            downside = final_returns[final_returns < 0]
            downside_std = np.std(downside) if len(downside) > 1 else 1.0

            # Need majority of paths profitable AND positive median
            if p_profit < 0.55 or median_return <= 0:
                continue

            # Sortino-like score: median / downside risk
            score = median_return / downside_std if downside_std > 0 else median_return

            # Volume confirmation
            if volume_weight > 0 and volumes_vals is not None:
                vol_window = volumes_vals[max(0, i - 20):i + 1, ci]
                if len(vol_window) > 1 and not np.all(np.isnan(vol_window)):
                    current_vol = vol_window[-1]
                    avg_vol = np.nanmean(vol_window[:-1])
                    if avg_vol > 0 and not np.isnan(current_vol):
                        volume_ratio = current_vol / avg_vol
                        price_momentum = 1.0 if returns[-1] > 0 else -1.0
                        score *= (1.0 + volume_weight * volume_ratio * price_momentum)

            if score > 0:
                scores[ci] = score

        winners = sorted(scores, key=scores.get, reverse=True)[:top_n]

        # Liquidate all current holdings first
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


class MonteCarloStrategy:
    """
    Monte Carlo simulation + Expectimax strategy for crypto.
    Simulates future price paths using Monte Carlo, builds a probability tree,
    and uses expectimax to evaluate expected value of buying each coin.
    The eval function uses supply/demand imbalance approximated by volume analysis.
    """

    GRID = {
        "lookback": [30, 60, 120],
        "n_simulations": [50, 100, 200],
        "horizon": [5, 10, 20],
        "volume_weight": [0.0, 0.3, 0.5],
        "trend_window": [60, 120, 240],
        "top_n": [1, 2],
    }
    REBALANCE_OPTIONS = [5, 15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 lookback: int = 60, n_simulations: int = 100,
                 horizon: int = 10, volume_weight: float = 0.5,
                 trend_window: int = 120, top_n: int = 2):
        self.api = api
        self.symbols = symbols
        self.lookback = lookback
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.volume_weight = volume_weight
        self.trend_window = trend_window
        self.top_n = top_n
        self._load_params()

    def _load_params(self):
        pf = _params_file()
        if os.path.exists(pf):
            with open(pf) as f:
                p = json.load(f)
            self.lookback = p.get("lookback", self.lookback)
            self.n_simulations = p.get("n_simulations", self.n_simulations)
            self.horizon = p.get("horizon", self.horizon)
            self.volume_weight = p.get("volume_weight", self.volume_weight)
            self.trend_window = p.get("trend_window", self.trend_window)
            self.top_n = p.get("top_n", self.top_n)
            print(f"Loaded params from {pf}")

    def _save_params(self, rebalance_every=5, params_suffix=None):
        os.makedirs(PARAMS_DIR, exist_ok=True)
        pf = _params_file()
        if params_suffix:
            pf = pf.replace(".json", f"_{params_suffix}.json")
        params = {
            "lookback": self.lookback,
            "n_simulations": self.n_simulations,
            "horizon": self.horizon,
            "volume_weight": self.volume_weight,
            "trend_window": self.trend_window,
            "top_n": self.top_n,
            "rebalance_every": rebalance_every,
            "updated_at": datetime.now().isoformat(),
        }
        with open(pf, "w") as f:
            json.dump(params, f, indent=2)
        print(f"Saved params to {pf}")

    def optimize(self, days: int = 7, fixed_interval: int = None, params_suffix: str = None):
        print(f"Optimizing Monte Carlo strategy over {days} days of data...")
        history = self._fetch_history(days)
        if not history:
            print("No data — keeping current params.")
            return

        closes = pd.DataFrame({sym: df["close"] for sym, df in history.items()})
        closes = closes.dropna(how="all").ffill()
        volumes = pd.DataFrame({sym: df["volume"] for sym, df in history.items() if "volume" in df.columns})
        volumes = volumes.reindex(closes.index).fillna(0) if not volumes.empty else None

        # Precompute trend SMAs
        print("Precomputing trend SMAs...")
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
        volumes_vals = volumes.values if volumes is not None else None
        trend_np = {k: v.values for k, v in trend_cache.items()}

        jobs = []
        jobs_meta = []
        rebal_options = [fixed_interval] if fixed_interval else self.REBALANCE_OPTIONS
        for rebal in rebal_options:
            for vals in combos:
                kwargs = dict(zip(keys, vals))
                jobs.append((closes_vals, volumes_vals, closes.columns.tolist(),
                             trend_np, kwargs, rebal))
                jobs_meta.append((kwargs, rebal))

        results = grid_search(_mc_backtest_worker, jobs)
        best_params, best_rebal, best_sharpe, best_return = find_best(jobs_meta, results)

        if best_params:
            self.lookback = best_params["lookback"]
            self.n_simulations = best_params["n_simulations"]
            self.horizon = best_params["horizon"]
            self.volume_weight = best_params["volume_weight"]
            self.trend_window = best_params["trend_window"]
            self.top_n = best_params["top_n"]
            label = "Optimal" if best_return > 0 else "Best (still negative)"
            print(f"\n{label} params found:")
            print(f"  lookback={self.lookback}, n_simulations={self.n_simulations}")
            print(f"  horizon={self.horizon}, volume_weight={self.volume_weight}")
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
        """Score each coin using bootstrap Monte Carlo with risk-adjusted metrics."""
        np.random.seed(42)

        end = datetime.now()
        bars_needed = max(self.lookback, self.trend_window) + 30
        start = end - timedelta(minutes=bars_needed)

        scores = {}
        for symbol in self.symbols:
            try:
                bars = self.api.get_crypto_bars(
                    symbol,
                    tradeapi.TimeFrame.Minute,
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=bars_needed,
                ).df

                if len(bars) < self.lookback:
                    continue

                closes = bars["close"]
                volumes = bars["volume"]

                current_price = closes.iloc[-1]
                if current_price <= 0:
                    continue

                trend_sma = closes.rolling(self.trend_window, min_periods=20).mean()
                if np.isnan(trend_sma.iloc[-1]) or current_price < trend_sma.iloc[-1]:
                    continue

                price_window = closes.iloc[-self.lookback:].values
                rets = np.diff(price_window) / price_window[:-1]
                if len(rets) < 10 or np.std(rets) == 0:
                    continue

                # Bootstrap resampling of actual returns
                indices = np.random.randint(0, len(rets), size=(self.n_simulations, self.horizon))
                sim_returns = rets[indices]
                cumulative = np.cumprod(1.0 + sim_returns, axis=1)
                final_returns = cumulative[:, -1] - 1.0

                p_profit = np.mean(final_returns > 0)
                median_return = np.median(final_returns)
                downside = final_returns[final_returns < 0]
                downside_std = np.std(downside) if len(downside) > 1 else 1.0

                if p_profit < 0.55 or median_return <= 0:
                    continue

                score = median_return / downside_std if downside_std > 0 else median_return

                if self.volume_weight > 0:
                    vol_avg = volumes.rolling(20).mean()
                    if vol_avg.iloc[-1] > 0:
                        volume_ratio = volumes.iloc[-1] / vol_avg.iloc[-1]
                        price_momentum = 1.0 if rets[-1] > 0 else -1.0
                        score *= (1.0 + self.volume_weight * volume_ratio * price_momentum)

                if score > 0:
                    scores[symbol] = score

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
        print(f"\n[{datetime.now()}] Running Monte Carlo strategy...")

        scores = self.get_momentum_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"Picks: {picks}")

        if not picks:
            print("No Monte Carlo candidates found.")
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
    print(f"Optimizing Monte Carlo strategy...\n")
    strategy = MonteCarloStrategy(api=api, symbols=CRYPTO_SYMBOLS)
    strategy.optimize(days=days)
