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
    return os.path.join(PARAMS_DIR, "adaptive_mr.json")


def _adaptive_mr_backtest_worker(closes_vals, volumes_vals, vol_avg_np, n_cols,
                                  kwargs, rebalance_every, initial_cash=100_000.0,
                                  return_equity=False):
    """
    Adaptive mean-reversion backtest worker.

    Key innovation: ATR-adaptive dip thresholds. When volatility is high,
    require deeper dips (avoid noise). When vol is low, trade shallower dips.

    Math:
        atr_i = rolling_std(returns_i, atr_window) * sqrt(1440)  # annualized
        adaptive_min_dip = base_min_dip * (atr_i / median_atr)
        adaptive_max_dip = base_max_dip * (atr_i / median_atr)
        dip = (price - VWAP) / VWAP
        Entry: adaptive_min_dip <= -dip <= adaptive_max_dip AND trend filter
    """
    vwap_window = kwargs["vwap_window"]
    base_min_dip = kwargs["base_min_dip"]
    base_max_dip = kwargs["base_max_dip"]
    atr_window = kwargs["atr_window"]
    top_n = kwargs["top_n"]
    trend_window = kwargs["trend_window"]
    take_profit = kwargs["take_profit"]
    stop_loss = kwargs["stop_loss"]
    max_hold = kwargs["max_hold"]
    regime_window = kwargs["regime_window"]

    if base_min_dip >= base_max_dip:
        return None

    n_rows = closes_vals.shape[0]
    warmup = max(vwap_window, trend_window, regime_window, atr_window) + 20

    if n_rows <= warmup:
        return None

    # Precompute returns
    returns = np.empty_like(closes_vals)
    returns[0, :] = 0
    returns[1:, :] = (closes_vals[1:] - closes_vals[:-1]) / np.where(
        closes_vals[:-1] > 0, closes_vals[:-1], np.nan
    )

    # Precompute rolling ATR (volatility) per coin using cumsum trick
    # Use rolling std of returns as ATR proxy
    ret_sq = returns ** 2
    cs_ret = np.nancumsum(returns, axis=0)
    cs_ret_sq = np.nancumsum(ret_sq, axis=0)

    atr = np.full_like(closes_vals, np.nan)
    for i in range(atr_window, n_rows):
        s = i - atr_window
        n_w = atr_window
        r_sum = cs_ret[i] - cs_ret[s]
        r_sum2 = cs_ret_sq[i] - cs_ret_sq[s]
        mu = r_sum / n_w
        var = r_sum2 / n_w - mu * mu
        var = np.maximum(var, 0)
        atr[i] = np.sqrt(var)

    # Median ATR across coins at each bar (for normalization)
    median_atr = np.nanmedian(atr, axis=1, keepdims=True)
    # Avoid division by zero
    median_atr = np.where(median_atr > 0, median_atr, 1e-6)

    # ATR ratio per coin: how volatile is this coin relative to the cross-section
    atr_ratio = atr / median_atr

    # Precompute VWAP dips
    if volumes_vals is not None:
        cv_cumsum = np.nancumsum(closes_vals * volumes_vals, axis=0)
        v_cumsum = np.nancumsum(volumes_vals, axis=0)
        dips_vals = np.full_like(closes_vals, np.nan)
        for i in range(vwap_window, n_rows):
            s = i - vwap_window
            cv = cv_cumsum[i] - cv_cumsum[s]
            v = v_cumsum[i] - v_cumsum[s]
            vwap = np.where(v > 0, cv / v, np.nan)
            dips_vals[i] = (closes_vals[i] - vwap) / np.where(vwap > 0, vwap, np.nan)
    else:
        # Fallback: SMA-based dip
        cs = np.nancumsum(closes_vals, axis=0)
        dips_vals = np.full_like(closes_vals, np.nan)
        for i in range(vwap_window, n_rows):
            s = i - vwap_window
            sma = (cs[i] - cs[s]) / vwap_window
            dips_vals[i] = (closes_vals[i] - sma) / np.where(sma > 0, sma, np.nan)

    # Precompute trend SMA
    trend_cs = np.nancumsum(closes_vals, axis=0)
    trend_sma = np.full_like(closes_vals, np.nan)
    for i in range(trend_window, n_rows):
        s = i - trend_window
        trend_sma[i] = (trend_cs[i] - trend_cs[s]) / trend_window

    # BTC regime (column 0)
    btc_closes = closes_vals[:, 0]
    btc_cs = np.nancumsum(btc_closes)
    regime_ok = np.zeros(n_rows, dtype=bool)
    if regime_window > 0:
        for i in range(regime_window, n_rows):
            s = i - regime_window
            btc_sma = (btc_cs[i] - btc_cs[s]) / regime_window
            regime_ok[i] = btc_closes[i] > btc_sma
    else:
        regime_ok[warmup:] = True

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

        # Portfolio value
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

        # Find adaptive dip candidates
        winners = []
        for ci in range(n_cols):
            # Trend filter
            t = trend_sma[i, ci]
            if np.isnan(t) or closes_vals[i, ci] < t:
                continue

            d = dips_vals[i, ci]
            if np.isnan(d):
                continue

            # Adaptive thresholds based on per-coin ATR ratio
            ar = atr_ratio[i, ci]
            if np.isnan(ar):
                continue
            ar = max(0.3, min(ar, 3.0))  # clamp to avoid extreme scaling

            adaptive_min = base_min_dip * ar
            adaptive_max = base_max_dip * ar

            if -adaptive_max <= d <= -adaptive_min:
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

        alloc = cash * dd_scale
        per_stock = alloc / len(winners)
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
    rets = np.diff(pv) / pv[:-1]
    std = rets.std()
    if std > 0:
        periods_per_year = (1440 * 365) / rebalance_every
        sharpe = (rets.mean() / std) * np.sqrt(periods_per_year)
    else:
        sharpe = 0

    result = {"total_return": total_return, "sharpe": sharpe}
    if return_equity:
        result["equity_curve"] = values
    return result


class AdaptiveMRStrategy:
    """
    Adaptive mean-reversion for crypto:
    - ATR-adaptive dip thresholds (wider in high vol, tighter in low vol)
    - BTC regime gate
    - Per-bar take-profit, stop-loss, max-hold
    - Drawdown control
    """

    GRID = {
        "vwap_window": [15, 25, 40],
        "base_min_dip": [0.0005, 0.001, 0.002],
        "base_max_dip": [0.008, 0.012, 0.020],
        "atr_window": [60, 120, 240],
        "take_profit": [0.002, 0.003, 0.004, 0.006],
        "stop_loss": [0.0, 0.015, 0.025],
        "max_hold": [25, 45, 90],
        "regime_window": [0, 540, 720, 1080],
        "trend_window": [60, 120, 240],
        "top_n": [1],
    }
    REBALANCE_OPTIONS = [15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 vwap_window: int = 25, base_min_dip: float = 0.001,
                 base_max_dip: float = 0.012, atr_window: int = 120,
                 top_n: int = 1, trend_window: int = 120,
                 take_profit: float = 0.003, stop_loss: float = 0.015,
                 max_hold: int = 30, regime_window: int = 540):
        self.api = api
        self.symbols = symbols
        self.vwap_window = vwap_window
        self.base_min_dip = base_min_dip
        self.base_max_dip = base_max_dip
        self.atr_window = atr_window
        self.top_n = top_n
        self.trend_window = trend_window
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
            for attr in ["vwap_window", "base_min_dip", "base_max_dip",
                         "atr_window", "top_n", "trend_window",
                         "take_profit", "stop_loss", "max_hold",
                         "regime_window"]:
                if attr in p:
                    setattr(self, attr, p[attr])
            print(f"Loaded params from {pf}")

    def _save_params(self, rebalance_every=15, params_suffix=None):
        os.makedirs(PARAMS_DIR, exist_ok=True)
        pf = _params_file()
        if params_suffix:
            pf = pf.replace(".json", f"_{params_suffix}.json")
        params = {
            "vwap_window": self.vwap_window,
            "base_min_dip": self.base_min_dip,
            "base_max_dip": self.base_max_dip,
            "atr_window": self.atr_window,
            "top_n": self.top_n,
            "trend_window": self.trend_window,
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

    def optimize(self, days: int = 7, fixed_interval: int = None, params_suffix: str = None):
        print(f"Optimizing adaptive mean-reversion over {days} days of data...")
        history = self._fetch_history(days)
        if not history:
            print("No data — keeping current params.")
            return

        closes = pd.DataFrame({sym: df["close"] for sym, df in history.items()})
        closes = closes.dropna(how="all").ffill()
        volumes = pd.DataFrame({sym: df["volume"] for sym, df in history.items() if "volume" in df.columns})
        volumes = volumes.reindex(closes.index).fillna(0) if not volumes.empty else None

        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from optimize import bayesian_search

        closes_vals = closes.values
        volumes_vals = volumes.values if volumes is not None else None
        vol_avg = volumes.rolling(20, min_periods=10).mean() if volumes is not None else None
        vol_avg_np = vol_avg.values if vol_avg is not None else None
        n_cols = closes_vals.shape[1]

        rebal_options = [fixed_interval] if fixed_interval else self.REBALANCE_OPTIONS
        fixed_args = (closes_vals, volumes_vals, vol_avg_np, n_cols)

        grid = self.GRID
        print(f"Running Bayesian optimization...")
        best_params, best_rebal, best_sharpe, best_return = bayesian_search(
            _adaptive_mr_backtest_worker, grid, rebal_options, fixed_args,
            n_trials=300)

        if best_params:
            for attr in grid.keys():
                setattr(self, attr, best_params[attr])
            label = "Optimal" if best_return > 0 else "Best (still negative)"
            print(f"\n{label} params found:")
            for k in grid.keys():
                print(f"  {k}={best_params[k]}")
            print(f"  rebalance_every={best_rebal}min")
            print(f"  Backtest return: {best_return:.2%} | Sharpe: {best_sharpe:.2f}")
            self._save_params(best_rebal, params_suffix=params_suffix)
            return best_rebal
        else:
            print("No valid params found — keeping defaults.")
            return 15

    def _fetch_history(self, days: int, end_days_ago: int = 1) -> dict[str, pd.DataFrame]:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from backtest import fetch_history
        return fetch_history(self.api, self.symbols, days, end_days_ago=end_days_ago)

    # --- Live trading methods ---

    def _check_regime(self) -> bool:
        if self.regime_window <= 0:
            return True
        end = datetime.now()
        start = end - timedelta(minutes=self.regime_window + 30)
        try:
            bars = self.api.get_crypto_bars(
                "BTC/USD", tradeapi.TimeFrame.Minute,
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
        if not self._check_regime():
            print("  BTC regime: BEARISH — staying in cash")
            return pd.Series(dtype=float)

        window = max(self.vwap_window, self.trend_window, self.atr_window) + 50
        end = datetime.now()
        start = end - timedelta(minutes=window)

        all_closes = {}
        all_volumes = {}
        for symbol in self.symbols:
            try:
                bars = self.api.get_crypto_bars(
                    symbol, tradeapi.TimeFrame.Minute,
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=window,
                ).df
                if len(bars) > 60:
                    all_closes[symbol] = bars["close"]
                    all_volumes[symbol] = bars["volume"]
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")

        if not all_closes:
            return pd.Series(dtype=float)

        closes = pd.DataFrame(all_closes).dropna(how="all").ffill()
        volumes = pd.DataFrame(all_volumes).reindex(closes.index).fillna(0)

        # Compute ATR ratio
        rets = closes.pct_change()
        atr = rets.rolling(self.atr_window, min_periods=20).std()
        median_atr = atr.median(axis=1)

        # VWAP
        roll_cv = (closes * volumes).rolling(self.vwap_window, min_periods=10).sum()
        roll_v = volumes.rolling(self.vwap_window, min_periods=10).sum()
        vwap = roll_cv / roll_v.replace(0, np.nan)
        vwap = vwap.fillna(closes.rolling(self.vwap_window, min_periods=10).mean())
        dips = (closes - vwap) / vwap

        trend_sma = closes.rolling(self.trend_window, min_periods=20).mean()

        scores = {}
        for sym in closes.columns:
            if closes[sym].iloc[-1] < trend_sma[sym].iloc[-1]:
                continue

            d = dips[sym].iloc[-1]
            if np.isnan(d):
                continue

            ar = atr[sym].iloc[-1] / median_atr.iloc[-1] if median_atr.iloc[-1] > 0 else 1.0
            ar = max(0.3, min(ar, 3.0))

            adaptive_min = self.base_min_dip * ar
            adaptive_max = self.base_max_dip * ar

            if -adaptive_max <= d <= -adaptive_min:
                scores[sym] = d

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
        print(f"\n[{datetime.now()}] Running adaptive mean-reversion strategy...")

        if not self._check_regime():
            print("BTC regime: BEARISH — liquidating to cash.")
            for pos in self.api.list_positions():
                if is_crypto(pos.symbol):
                    print(f"  Closing {pos.symbol} ({pos.qty})")
                    try:
                        self.api.submit_order(
                            symbol=pos.symbol, qty=abs(float(pos.qty)),
                            side="sell", type="market", time_in_force="gtc",
                        )
                    except Exception as e:
                        print(f"  Error closing {pos.symbol}: {e}")
            print("Rebalance complete (all cash — bearish regime).")
            return

        scores = self.get_momentum_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"Adaptive dip picks: {picks}")

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
