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
                               initial_cash=100_000.0, return_equity=False):
    """Standalone Bayesian backtest function for multiprocessing."""
    sma_period = kwargs["sma_period"]
    rsi_period = kwargs["rsi_period"]
    momentum_weight = kwargs["momentum_weight"]
    volume_weight = kwargs["volume_weight"]
    rsi_weight = kwargs["rsi_weight"]
    threshold = kwargs["threshold"]
    trend_window = kwargs["trend_window"]
    top_n = kwargs["top_n"]
    take_profit = kwargs.get("take_profit", 0.0)
    stop_loss = kwargs.get("stop_loss", 0.0)
    max_hold = kwargs.get("max_hold", 0)
    regime_window = kwargs.get("regime_window", 0)

    sma = sma_np[sma_period]
    rsi = rsi_np[rsi_period]
    vol_avg = vol_avg_np.get(20)
    trend_sma = trend_np[trend_window]

    warmup = max(sma_period, rsi_period, trend_window, regime_window) + 10
    n_rows = closes_vals.shape[0]
    n_cols = closes_vals.shape[1]
    rebal_indices = list(range(warmup, n_rows, rebalance_every))

    # BTC regime (column 0) if enabled
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

    cash = initial_cash
    holdings = {}  # col_idx -> qty
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
        entry_prices = {}
        entry_bars = {}

        # If no winners, stay in cash
        if not winners:
            continue

        alloc = cash * dd_scale
        per_stock = alloc / len(winners)
        for ci in winners:
            p = closes_vals[i, ci]
            if np.isnan(p) or p <= 0:
                continue
            qty = per_stock / p
            holdings[ci] = qty
            entry_prices[ci] = p
            entry_bars[ci] = i
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

    result = {"total_return": total_return, "sharpe": sharpe}
    if return_equity:
        result["equity_curve"] = values
    return result


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
        "take_profit": [0.0, 0.002, 0.004],
        "stop_loss": [0.0, 0.015],
        "max_hold": [0, 30, 60],
        "regime_window": [0, 540],
    }
    REBALANCE_OPTIONS = [15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 sma_period: int = 20, rsi_period: int = 14,
                 momentum_weight: float = 1.0, volume_weight: float = 0.5,
                 rsi_weight: float = 0.5, threshold: float = 0.6,
                 trend_window: int = 120, top_n: int = 2,
                 take_profit: float = 0.0, stop_loss: float = 0.0,
                 max_hold: int = 0, regime_window: int = 0):
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
            for attr in ["sma_period", "rsi_period", "momentum_weight",
                         "volume_weight", "rsi_weight", "threshold",
                         "trend_window", "top_n", "take_profit",
                         "stop_loss", "max_hold", "regime_window"]:
                if attr in p:
                    setattr(self, attr, p[attr])
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
        from optimize import bayesian_search

        # Convert caches to numpy for pickling
        closes_vals = closes.values
        volumes_vals = volumes.values if volumes is not None else None
        sma_np = {k: v.values for k, v in sma_cache.items()}
        rsi_np = {k: v.values for k, v in rsi_cache.items()}
        vol_avg_np = {k: v.values for k, v in vol_cache.items()}
        trend_np = {k: v.values for k, v in trend_cache.items()}

        rebal_options = [fixed_interval] if fixed_interval else self.REBALANCE_OPTIONS
        fixed_args = (closes_vals, volumes_vals, closes.columns.tolist(),
                      sma_np, rsi_np, vol_avg_np, trend_np)

        best_params, best_rebal, best_sharpe, best_return = bayesian_search(
            _bayesian_backtest_worker, self.GRID, rebal_options, fixed_args,
            n_trials=300)

        if best_params:
            for attr in best_params:
                if hasattr(self, attr):
                    setattr(self, attr, best_params[attr])
            label = "Optimal" if best_return > 0 else "Best (still negative)"
            print(f"\n{label} params found:")
            for k, v in best_params.items():
                print(f"  {k}={v}")
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
