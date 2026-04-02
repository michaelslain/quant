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
    return os.path.join(PARAMS_DIR, "rsi_mean_revert.json")


def _rsi_mr_backtest_worker(closes_vals, volumes_vals, cols, rsi_np, bb_np,
                            vol_avg_np, trend_np, kwargs, rebalance_every,
                            atr_ratio_np=None, initial_cash=100_000.0):
    """Standalone backtest function for multiprocessing (must be top-level)."""
    rsi_period = kwargs["rsi_period"]
    rsi_oversold = kwargs["rsi_oversold"]
    bb_period = kwargs["bb_period"]
    bb_std = kwargs["bb_std"]
    volume_mult = kwargs["volume_mult"]
    top_n = kwargs["top_n"]
    trend_window = kwargs["trend_window"]
    bounce_window = kwargs["bounce_window"]
    use_bb = kwargs.get("use_bb", 1)

    rsi = rsi_np[rsi_period]
    bb = bb_np[(bb_period, bb_std)]
    vol_avg = vol_avg_np.get(20)
    trend_sma = trend_np[trend_window]

    stop_loss = kwargs.get("stop_loss", 0.0)
    adaptive_rsi = kwargs.get("adaptive_rsi", 0)

    warmup = max(rsi_period, bb_period, trend_window, 240) + 10
    n_rows = closes_vals.shape[0]
    n_cols = closes_vals.shape[1]
    rebal_indices = list(range(warmup, n_rows, rebalance_every))

    cash = initial_cash
    holdings = {}  # col_idx -> qty
    entry_prices = {}  # col_idx -> entry price
    values = []

    for ri, i in enumerate(rebal_indices):
        # Check stop-loss on bars between previous and current rebalance
        if stop_loss > 0 and holdings:
            prev_i = rebal_indices[ri - 1] if ri > 0 else warmup
            for bar in range(prev_i + 1, i):
                to_close = []
                for ci in list(holdings.keys()):
                    if ci in entry_prices:
                        p = closes_vals[bar, ci]
                        if not np.isnan(p) and p <= entry_prices[ci] * (1 - stop_loss):
                            to_close.append(ci)
                for ci in to_close:
                    p = closes_vals[bar, ci]
                    if not np.isnan(p):
                        cash += holdings[ci] * p
                    del holdings[ci]
                    del entry_prices[ci]

        port_value = cash
        for ci, qty in holdings.items():
            p = closes_vals[i, ci]
            if not np.isnan(p):
                port_value += qty * p
        values.append(port_value)

        scores = {}
        for ci in range(n_cols):
            # Trend filter: only take bounces when longer-term trend is up
            if np.isnan(trend_sma[i, ci]) or closes_vals[i, ci] < trend_sma[i, ci]:
                continue

            r = rsi[i, ci]
            if np.isnan(r):
                continue

            # Adaptive RSI threshold: widen in high vol, tighten in low vol
            if adaptive_rsi and atr_ratio_np is not None:
                ar = atr_ratio_np[i, ci]
                if not np.isnan(ar):
                    # High vol (ar>1): raise threshold (more permissive, e.g. 35->42)
                    # Low vol (ar<1): lower threshold (more selective, e.g. 35->28)
                    effective_oversold = rsi_oversold * ar
                    effective_oversold = max(15, min(50, effective_oversold))
                else:
                    effective_oversold = rsi_oversold
            else:
                effective_oversold = rsi_oversold

            # Current RSI must be above oversold (recovering)
            if r < effective_oversold:
                continue

            # Check if RSI was oversold within the last bounce_window bars
            if i < bounce_window:
                continue
            min_rsi = r
            for j in range(1, bounce_window + 1):
                prev = rsi[i - j, ci]
                if not np.isnan(prev) and prev < min_rsi:
                    min_rsi = prev
            if min_rsi >= effective_oversold:
                continue

            # Bollinger Band support: price near or below lower band (optional)
            if use_bb and not bb["near_lower"][i, ci]:
                continue

            # Volume surge on bounce (skip filter if volume_mult=0)
            vol_ratio = 1.0
            if volume_mult > 0 and vol_avg is not None and volumes_vals is not None:
                v = volumes_vals[i, ci]
                va = vol_avg[i, ci]
                if va > 0 and v < volume_mult * va:
                    continue
                vol_ratio = v / va if va > 0 else 1.0

            # Score: how deep the dip was * volume
            dip_depth = effective_oversold - min_rsi
            scores[ci] = dip_depth * vol_ratio

        winners = sorted(scores, key=scores.get, reverse=True)[:top_n]

        # Always liquidate current holdings first (go to cash)
        for ci, qty in holdings.items():
            p = closes_vals[i, ci]
            if not np.isnan(p):
                cash += qty * p
        holdings = {}
        entry_prices = {}

        # If no bounce signals, stay in cash
        if not winners:
            continue

        per_stock = cash / len(winners)
        for ci in winners:
            p = closes_vals[i, ci]
            if np.isnan(p) or p <= 0:
                continue
            qty = per_stock / p
            holdings[ci] = qty
            entry_prices[ci] = p
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


class RsiMeanRevertStrategy:
    """
    RSI-based mean reversion strategy for crypto.
    Buys when RSI bounces from oversold territory with Bollinger Band
    and volume confirmation. Only takes bounces in an uptrend context
    to avoid catching falling knives. Goes to cash when no signals.
    """

    GRID = {
        "rsi_period": [14, 20],
        "rsi_oversold": [30, 35, 40],
        "bb_period": [15, 20],
        "bb_std": [1.5, 2.0],
        "volume_mult": [0.0, 1.0],
        "trend_window": [60, 120, 240],
        "top_n": [1, 2],
        "bounce_window": [1, 3, 5],
        "use_bb": [0, 1],
        "stop_loss": [0.0, 0.01],
        "adaptive_rsi": [0, 1],
    }
    REBALANCE_OPTIONS = [5, 15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 rsi_period: int = 14, rsi_oversold: float = 30,
                 bb_period: int = 20, bb_std: float = 2.0,
                 volume_mult: float = 1.2, top_n: int = 2,
                 trend_window: int = 120, bounce_window: int = 3,
                 use_bb: int = 1, stop_loss: float = 0.0,
                 adaptive_rsi: int = 0):
        self.api = api
        self.symbols = symbols
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.volume_mult = volume_mult
        self.top_n = top_n
        self.trend_window = trend_window
        self.bounce_window = bounce_window
        self.use_bb = use_bb
        self.stop_loss = stop_loss
        self.adaptive_rsi = adaptive_rsi
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
            self.bounce_window = p.get("bounce_window", self.bounce_window)
            self.use_bb = p.get("use_bb", self.use_bb)
            self.stop_loss = p.get("stop_loss", self.stop_loss)
            self.adaptive_rsi = p.get("adaptive_rsi", self.adaptive_rsi)
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
            "bounce_window": self.bounce_window,
            "use_bb": self.use_bb,
            "stop_loss": self.stop_loss,
            "adaptive_rsi": self.adaptive_rsi,
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
        print(f"Optimizing RSI mean-reversion over {days} days of data...")
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
                lower = sma - bs * std
                # near_lower: price is within 1% above or below the lower band
                near_lower = closes <= lower * 1.01
                bb_cache[(bp, bs)] = {
                    "lower": lower,
                    "near_lower": near_lower,
                }

        vol_cache = {}
        if volumes is not None:
            vol_avg = volumes.rolling(20, min_periods=10).mean()
            vol_cache[20] = vol_avg

        # Precompute trend SMAs
        trend_cache = {}
        for tw in self.GRID["trend_window"]:
            trend_cache[tw] = closes.rolling(tw, min_periods=20).mean()

        # Precompute ATR for adaptive thresholds
        atr_raw = closes.diff().abs().rolling(20, min_periods=10).mean()
        atr_median = atr_raw.rolling(240, min_periods=60).median()
        # atr_ratio: >1 means higher vol than usual, <1 means calmer
        atr_ratio = (atr_raw / atr_median.replace(0, np.nan)).fillna(1.0)

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
            "near_lower": v["near_lower"].values,
        } for k, v in bb_cache.items()}
        vol_avg_np = {k: v.values for k, v in vol_cache.items()}
        trend_np = {k: v.values for k, v in trend_cache.items()}
        atr_ratio_np = atr_ratio.values

        jobs = []
        jobs_meta = []
        rebal_options = [fixed_interval] if fixed_interval else self.REBALANCE_OPTIONS
        for rebal in rebal_options:
            for vals in combos:
                kwargs = dict(zip(keys, vals))
                jobs.append((closes_vals, volumes_vals, closes.columns.tolist(),
                             rsi_np, bb_np, vol_avg_np, trend_np, kwargs, rebal,
                             atr_ratio_np))
                jobs_meta.append((kwargs, rebal))

        results = grid_search(_rsi_mr_backtest_worker, jobs)
        best_params, best_rebal, best_sharpe, best_return = find_best(jobs_meta, results)

        if best_params:
            self.rsi_period = best_params["rsi_period"]
            self.rsi_oversold = best_params["rsi_oversold"]
            self.bb_period = best_params["bb_period"]
            self.bb_std = best_params["bb_std"]
            self.volume_mult = best_params["volume_mult"]
            self.top_n = best_params["top_n"]
            self.trend_window = best_params["trend_window"]
            self.bounce_window = best_params["bounce_window"]
            self.use_bb = best_params["use_bb"]
            self.stop_loss = best_params["stop_loss"]
            self.adaptive_rsi = best_params["adaptive_rsi"]
            label = "Optimal" if best_return > 0 else "Best (still negative)"
            print(f"\n{label} params found:")
            print(f"  rsi_period={self.rsi_period}, rsi_oversold={self.rsi_oversold}")
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

    def get_bounce_scores(self) -> pd.Series:
        """Score each symbol by RSI oversold bounce strength."""
        end = datetime.now()
        lookback = max(self.rsi_period, self.bb_period, self.trend_window) + 30
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

                # Trend filter: price must be above long SMA
                trend_sma = closes.rolling(self.trend_window, min_periods=20).mean()
                if closes.iloc[-1] < trend_sma.iloc[-1]:
                    continue

                # RSI
                rsi = self._compute_rsi(closes, self.rsi_period)
                current_rsi = rsi.iloc[-1]
                if np.isnan(current_rsi) or current_rsi < self.rsi_oversold:
                    continue

                # Check if RSI was oversold within the last bounce_window bars
                recent_rsi = rsi.iloc[-(self.bounce_window + 1):-1]
                min_recent_rsi = recent_rsi.min()
                if np.isnan(min_recent_rsi) or min_recent_rsi >= self.rsi_oversold:
                    continue

                # Bollinger Band support: price near or below lower band (optional)
                if self.use_bb:
                    sma = closes.rolling(self.bb_period).mean()
                    std = closes.rolling(self.bb_period).std()
                    lower = sma - self.bb_std * std
                    if closes.iloc[-1] > lower.iloc[-1] * 1.01:
                        continue

                # Volume surge on bounce
                vol_avg = volumes.rolling(20).mean()
                if vol_avg.iloc[-1] > 0 and volumes.iloc[-1] < self.volume_mult * vol_avg.iloc[-1]:
                    continue

                # Score: deeper dip = stronger bounce signal * volume ratio
                vol_ratio = volumes.iloc[-1] / vol_avg.iloc[-1] if vol_avg.iloc[-1] > 0 else 1.0
                dip_depth = self.rsi_oversold - min_recent_rsi
                scores[symbol] = dip_depth * vol_ratio

            except Exception as e:
                print(f"Error fetching {symbol}: {e}")

        return pd.Series(scores).sort_values(ascending=False)

    def get_target_positions(self) -> dict[str, float]:
        scores = self.get_bounce_scores()
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
        print(f"\n[{datetime.now()}] Running RSI mean-reversion strategy...")

        scores = self.get_bounce_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"Bounce picks: {picks}")

        # Get current positions
        current = {}
        for pos in self.api.list_positions():
            current[pos.symbol] = float(pos.qty)

        if not picks:
            print("No bounce signals — liquidating to cash.")
            for symbol, qty in current.items():
                if is_crypto(symbol):
                    print(f"  Closing {symbol} ({qty} shares)")
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
    print(f"Optimizing RSI mean-reversion...\n")
    strategy = RsiMeanRevertStrategy(api=api, symbols=CRYPTO_SYMBOLS)
    strategy.optimize(days=days)
