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
    return os.path.join(PARAMS_DIR, "breakout.json")


def _breakout_backtest_worker(closes_vals, highs_vals, volumes_vals, cols,
                               channel_np, vol_avg_np, atr_np, trend_np,
                               kwargs, rebalance_every, initial_cash=100_000.0):
    """Standalone backtest function for multiprocessing (must be top-level)."""
    channel_period = kwargs["channel_period"]
    volume_mult = kwargs["volume_mult"]
    atr_period = kwargs["atr_period"]
    trend_window = kwargs["trend_window"]
    top_n = kwargs["top_n"]
    take_profit = kwargs.get("take_profit", 0.0)
    stop_loss = kwargs.get("stop_loss", 0.0)
    max_hold = kwargs.get("max_hold", 0)
    regime_window = kwargs.get("regime_window", 0)

    highest_high = channel_np[channel_period]
    vol_avg = vol_avg_np.get(20)
    atr = atr_np[atr_period]
    trend_sma = trend_np[trend_window]

    warmup = max(channel_period, atr_period, trend_window, regime_window) + 10
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

        # Calculate portfolio value
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

        # Score breakout candidates
        scores = {}
        for ci in range(n_cols):
            price = closes_vals[i, ci]
            if np.isnan(price) or price <= 0:
                continue

            # Trend filter: price must be above longer-term SMA
            sma_val = trend_sma[i, ci]
            if np.isnan(sma_val) or price < sma_val:
                continue

            # Breakout detection: price breaks above highest high of last N bars
            hh = highest_high[i, ci]
            if np.isnan(hh) or price <= hh:
                continue

            # Volume confirmation: volume must be above average
            if vol_avg is not None and volumes_vals is not None:
                v = volumes_vals[i, ci]
                va = vol_avg[i, ci]
                if va > 0 and v < volume_mult * va:
                    continue
                vol_ratio = v / va if va > 0 else 1.0
            else:
                vol_ratio = 1.0

            # ATR-based inverse volatility score (lower ATR = higher score)
            atr_val = atr[i, ci]
            if np.isnan(atr_val) or atr_val <= 0:
                continue
            inv_vol = 1.0 / atr_val
            scores[ci] = inv_vol * vol_ratio

        winners = sorted(scores, key=scores.get, reverse=True)[:top_n]

        # Liquidate all current holdings
        for ci, qty in holdings.items():
            p = closes_vals[i, ci]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}
        entry_prices = {}
        entry_bars = {}

        # If no breakout detected, stay in cash
        if not winners:
            continue

        # ATR-based risk parity sizing
        total_inv_atr = 0.0
        winner_atrs = {}
        for ci in winners:
            atr_val = atr[i, ci]
            if not np.isnan(atr_val) and atr_val > 0:
                inv = 1.0 / atr_val
                winner_atrs[ci] = inv
                total_inv_atr += inv

        if total_inv_atr <= 0:
            continue

        alloc_cash = cash * dd_scale
        for ci in winners:
            p = closes_vals[i, ci]
            if np.isnan(p) or p <= 0:
                continue
            weight = winner_atrs.get(ci, 0) / total_inv_atr
            alloc = alloc_cash * weight
            qty = alloc / p
            holdings[ci] = qty
            entry_prices[ci] = p
            entry_bars[ci] = i

        cash_used = 0.0
        for ci, qty in holdings.items():
            cash_used += qty * closes_vals[i, ci]
        cash -= cash_used

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


class CryptoBreakoutStrategy:
    """
    Crypto channel breakout strategy.
    Buys when price breaks above the highest high of the last N bars,
    confirmed by above-average volume. Uses ATR for risk-parity sizing
    and a trend filter (SMA) to only trade in the direction of the trend.
    Goes to cash when no breakouts are detected.
    """

    GRID = {
        "channel_period": [20, 40, 60, 90],
        "volume_mult": [1.0, 1.2, 1.5],
        "atr_period": [14, 20, 30],
        "trend_window": [60, 120, 240],
        "top_n": [1, 2],
        "take_profit": [0.0, 0.002, 0.004],
        "stop_loss": [0.0, 0.015],
        "max_hold": [0, 30, 60],
        "regime_window": [0, 540],
    }
    REBALANCE_OPTIONS = [15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 channel_period: int = 40, volume_mult: float = 1.2,
                 atr_period: int = 14, trend_window: int = 120,
                 top_n: int = 2, take_profit: float = 0.0,
                 stop_loss: float = 0.0, max_hold: int = 0,
                 regime_window: int = 0):
        self.api = api
        self.symbols = symbols
        self.channel_period = channel_period
        self.volume_mult = volume_mult
        self.atr_period = atr_period
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
            self.channel_period = p.get("channel_period", self.channel_period)
            self.volume_mult = p.get("volume_mult", self.volume_mult)
            self.atr_period = p.get("atr_period", self.atr_period)
            self.trend_window = p.get("trend_window", self.trend_window)
            self.top_n = p.get("top_n", self.top_n)
            self.take_profit = p.get("take_profit", self.take_profit)
            self.stop_loss = p.get("stop_loss", self.stop_loss)
            self.max_hold = p.get("max_hold", self.max_hold)
            self.regime_window = p.get("regime_window", self.regime_window)
            print(f"Loaded params from {pf}")

    def _save_params(self, rebalance_every=5, params_suffix=None):
        os.makedirs(PARAMS_DIR, exist_ok=True)
        pf = _params_file()
        if params_suffix:
            pf = pf.replace(".json", f"_{params_suffix}.json")
        params = {
            "channel_period": self.channel_period,
            "volume_mult": self.volume_mult,
            "atr_period": self.atr_period,
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
    def _compute_atr(highs: pd.Series, lows: pd.Series, closes: pd.Series,
                     period: int) -> pd.Series:
        prev_close = closes.shift(1)
        tr = pd.concat([
            highs - lows,
            (highs - prev_close).abs(),
            (lows - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=period).mean()

    def optimize(self, days: int = 7, fixed_interval: int = None, params_suffix: str = None):
        print(f"Optimizing crypto breakout over {days} days of data...")
        history = self._fetch_history(days)
        if not history:
            print("No data -- keeping current params.")
            return

        closes = pd.DataFrame({sym: df["close"] for sym, df in history.items()})
        closes = closes.dropna(how="all").ffill()
        highs = pd.DataFrame({sym: df["high"] for sym, df in history.items()})
        highs = highs.reindex(closes.index).ffill()
        lows = pd.DataFrame({sym: df["low"] for sym, df in history.items()})
        lows = lows.reindex(closes.index).ffill()
        volumes = pd.DataFrame({sym: df["volume"] for sym, df in history.items() if "volume" in df.columns})
        volumes = volumes.reindex(closes.index).fillna(0) if not volumes.empty else None

        # Precompute indicators
        print("Precomputing indicators...")

        # Channel (highest high of previous N bars -- exclude current bar)
        channel_cache = {}
        for cp in self.GRID["channel_period"]:
            channel_cache[cp] = highs.shift(1).rolling(cp, min_periods=cp).max()

        # Volume average
        vol_cache = {}
        if volumes is not None:
            vol_avg = volumes.rolling(20, min_periods=10).mean()
            vol_cache[20] = vol_avg

        # ATR
        atr_cache = {}
        for ap in self.GRID["atr_period"]:
            atr_df = pd.DataFrame(index=closes.index)
            for col in closes.columns:
                atr_df[col] = self._compute_atr(highs[col], lows[col], closes[col], ap)
            atr_cache[ap] = atr_df

        # Trend SMA
        trend_cache = {}
        for tw in self.GRID["trend_window"]:
            trend_cache[tw] = closes.rolling(tw, min_periods=20).mean()

        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from optimize import bayesian_search

        # Convert caches to numpy for pickling
        closes_vals = closes.values
        highs_vals = highs.values
        volumes_vals = volumes.values if volumes is not None else None
        channel_np = {k: v.values for k, v in channel_cache.items()}
        vol_avg_np = {k: v.values for k, v in vol_cache.items()}
        atr_np = {k: v.values for k, v in atr_cache.items()}
        trend_np = {k: v.values for k, v in trend_cache.items()}

        fixed_args = (closes_vals, highs_vals, volumes_vals,
                      closes.columns.tolist(),
                      channel_np, vol_avg_np, atr_np, trend_np)
        rebal_options = [fixed_interval] if fixed_interval else self.REBALANCE_OPTIONS

        def _breakout_worker_dict(closes_vals, highs_vals, volumes_vals, cols,
                                  channel_np, vol_avg_np, atr_np, trend_np,
                                  kwargs, rebalance_every):
            return _breakout_backtest_worker(closes_vals, highs_vals, volumes_vals, cols,
                                             channel_np, vol_avg_np, atr_np, trend_np,
                                             kwargs, rebalance_every)

        best_params, best_rebal, best_sharpe, best_return = bayesian_search(
            _breakout_worker_dict, self.GRID, rebal_options, fixed_args,
            n_trials=300)

        if best_params:
            self.channel_period = best_params["channel_period"]
            self.volume_mult = best_params["volume_mult"]
            self.atr_period = best_params["atr_period"]
            self.trend_window = best_params["trend_window"]
            self.top_n = best_params["top_n"]
            self.take_profit = best_params.get("take_profit", 0.0)
            self.stop_loss = best_params.get("stop_loss", 0.0)
            self.max_hold = best_params.get("max_hold", 0)
            self.regime_window = best_params.get("regime_window", 0)
            label = "Optimal" if best_return > 0 else "Best (still negative)"
            print(f"\n{label} params found:")
            print(f"  channel_period={self.channel_period}, volume_mult={self.volume_mult}")
            print(f"  atr_period={self.atr_period}, trend_window={self.trend_window}")
            print(f"  top_n={self.top_n}, take_profit={self.take_profit}")
            print(f"  stop_loss={self.stop_loss}, max_hold={self.max_hold}")
            print(f"  regime_window={self.regime_window}")
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

    # --- Live trading methods ---

    def get_breakout_scores(self) -> pd.Series:
        end = datetime.now()
        lookback = max(self.channel_period, self.atr_period, self.trend_window) + 30
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
                highs = bars["high"]
                lows = bars["low"]
                volumes = bars["volume"]

                current_price = closes.iloc[-1]

                # Trend filter: price above SMA
                sma = closes.rolling(self.trend_window).mean()
                if np.isnan(sma.iloc[-1]) or current_price < sma.iloc[-1]:
                    continue

                # Channel breakout: price above highest high of previous N bars
                highest_high = highs.shift(1).rolling(self.channel_period).max()
                if np.isnan(highest_high.iloc[-1]) or current_price <= highest_high.iloc[-1]:
                    continue

                # Volume confirmation
                vol_avg = volumes.rolling(20).mean()
                if vol_avg.iloc[-1] > 0 and volumes.iloc[-1] < self.volume_mult * vol_avg.iloc[-1]:
                    continue

                # ATR for inverse-volatility scoring
                atr = self._compute_atr(highs, lows, closes, self.atr_period)
                atr_val = atr.iloc[-1]
                if np.isnan(atr_val) or atr_val <= 0:
                    continue

                vol_ratio = volumes.iloc[-1] / vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 1.0
                scores[symbol] = (1.0 / atr_val) * vol_ratio

            except Exception as e:
                print(f"Error fetching {symbol}: {e}")

        return pd.Series(scores).sort_values(ascending=False)

    def get_target_positions(self) -> dict[str, float]:
        scores = self.get_breakout_scores()
        picks = scores.head(self.top_n)

        if picks.empty:
            return {}

        account = self.api.get_account()
        cash = float(account.cash) * 0.95

        # ATR-based risk parity sizing
        atr_values = {}
        for symbol in picks.index:
            try:
                end = datetime.now()
                lookback = self.atr_period + 30
                start = end - timedelta(minutes=lookback)
                bars = self.api.get_crypto_bars(
                    symbol,
                    tradeapi.TimeFrame.Minute,
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=lookback,
                ).df
                atr_val = self._compute_atr(
                    bars["high"], bars["low"], bars["close"], self.atr_period
                ).iloc[-1]
                if not np.isnan(atr_val) and atr_val > 0:
                    atr_values[symbol] = atr_val
            except Exception as e:
                print(f"Error computing ATR for {symbol}: {e}")

        if not atr_values:
            return {}

        total_inv_atr = sum(1.0 / v for v in atr_values.values())

        targets = {}
        for symbol, atr_val in atr_values.items():
            weight = (1.0 / atr_val) / total_inv_atr
            alloc = cash * weight
            try:
                price = self.api.get_latest_crypto_trades([symbol])[symbol].price
                qty = round(alloc / price, 4)
                if qty > 0:
                    targets[symbol] = qty
            except Exception as e:
                print(f"Error getting price for {symbol}: {e}")

        return targets

    def rebalance(self):
        import time as _time
        print(f"\n[{datetime.now()}] Running crypto-breakout strategy...")

        scores = self.get_breakout_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"Breakout picks: {picks}")

        # Get current positions
        current = {}
        for pos in self.api.list_positions():
            current[pos.symbol] = float(pos.qty)

        if not picks:
            print("No breakout candidates -- liquidating to cash.")
            for symbol, qty in current.items():
                try:
                    self.api.submit_order(
                        symbol=symbol, qty=abs(qty), side="sell",
                        type="market", time_in_force="gtc" if is_crypto(symbol) else "day",
                    )
                    print(f"  Closed {symbol} ({qty})")
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
                print(f"  Closing {symbol} ({qty} shares)")
                try:
                    self.api.submit_order(
                        symbol=symbol, qty=abs(qty), side="sell",
                        type="market", time_in_force="gtc" if is_crypto(symbol) else "day",
                    )
                except Exception as e:
                    print(f"  Error closing {symbol}: {e}")

        _time.sleep(2)

        # Re-fetch cash and compute ATR-based sizing
        account = self.api.get_account()
        cash = float(account.cash) * 0.95

        atr_values = {}
        for symbol in picks:
            try:
                end = datetime.now()
                lookback = self.atr_period + 30
                start = end - timedelta(minutes=lookback)
                bars = self.api.get_crypto_bars(
                    symbol,
                    tradeapi.TimeFrame.Minute,
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=lookback,
                ).df
                atr_val = self._compute_atr(
                    bars["high"], bars["low"], bars["close"], self.atr_period
                ).iloc[-1]
                if not np.isnan(atr_val) and atr_val > 0:
                    atr_values[symbol] = atr_val
            except Exception as e:
                print(f"Error computing ATR for {symbol}: {e}")

        if not atr_values:
            print("Could not compute ATR for any pick -- skipping buys.")
            print("Rebalance complete.")
            return

        total_inv_atr = sum(1.0 / v for v in atr_values.values())

        for symbol in picks:
            if symbol not in atr_values:
                continue
            held_sym = symbol.replace("/", "")
            current_qty = current.get(held_sym, current.get(symbol, 0))
            weight = (1.0 / atr_values[symbol]) / total_inv_atr
            alloc = cash * weight
            try:
                price = self.api.get_latest_crypto_trades([symbol])[symbol].price
                target_qty = round(alloc / price, 4)
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
    print(f"Optimizing crypto breakout...\n")
    strategy = CryptoBreakoutStrategy(api=api, symbols=CRYPTO_SYMBOLS)
    strategy.optimize(days=days)
