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
    return os.path.join(PARAMS_DIR, "dispersion_mr.json")


def _dispersion_mr_backtest_worker(closes_vals, n_cols, kwargs, rebalance_every,
                                    initial_cash=100_000.0, return_equity=False):
    """
    Return Dispersion Mean Reversion backtest worker.

    Key insight: When cross-sectional return dispersion is low, individual coin
    movements are noise (not information). VWAP dips in low-dispersion regimes
    revert more reliably than in high-dispersion regimes.

    Math:
        ret_i = close_i / close_i_lag - 1  (per coin)
        dispersion = std(ret_all_coins)      (cross-sectional)
        rolling_disp = EMA(dispersion, disp_window)
        disp_percentile = percentile_rank(rolling_disp, disp_lookback)

        VWAP dip = (price - rolling_mean) / rolling_mean
        Entry: VWAP dip in [-max_dip, -min_dip]
               AND disp_percentile < disp_thresh (low dispersion)
               AND price > trend_sma (uptrend)
               AND BTC regime gate
    """
    vwap_window = kwargs["vwap_window"]
    min_dip = kwargs["min_dip"]
    max_dip = kwargs["max_dip"]
    if min_dip >= max_dip:
        return None
    disp_window = kwargs["disp_window"]
    disp_lookback = kwargs["disp_lookback"]
    disp_thresh = kwargs["disp_thresh"]
    top_n = kwargs["top_n"]
    trend_window = kwargs["trend_window"]
    take_profit = kwargs.get("take_profit", 0.0)
    stop_loss = kwargs.get("stop_loss", 0.0)
    max_hold = kwargs.get("max_hold", 0)
    regime_window = kwargs.get("regime_window", 0)

    n_rows = closes_vals.shape[0]
    warmup = max(vwap_window, trend_window, regime_window, disp_lookback + disp_window) + 20

    if n_rows <= warmup:
        return None

    # Compute VWAP (rolling mean as proxy)
    vwap = np.empty_like(closes_vals)
    vwap[:] = np.nan
    cs = np.nancumsum(closes_vals, axis=0)
    for i in range(vwap_window, n_rows):
        s = i - vwap_window
        vwap[i] = (cs[i] - cs[s]) / vwap_window

    dips = np.where(vwap > 0, (closes_vals - vwap) / vwap, np.nan)

    # Compute cross-sectional return dispersion
    returns = np.empty_like(closes_vals)
    returns[0, :] = 0
    returns[1:, :] = np.where(
        closes_vals[:-1] > 0,
        (closes_vals[1:] - closes_vals[:-1]) / closes_vals[:-1],
        np.nan
    )

    # Cross-sectional std at each bar
    cross_disp = np.nanstd(returns, axis=1)

    # EMA of dispersion
    ema_disp = np.empty(n_rows)
    ema_disp[:] = np.nan
    alpha = 2.0 / (disp_window + 1)
    ema_disp[0] = cross_disp[0]
    for i in range(1, n_rows):
        if np.isnan(ema_disp[i - 1]):
            ema_disp[i] = cross_disp[i]
        else:
            ema_disp[i] = alpha * cross_disp[i] + (1 - alpha) * ema_disp[i - 1]

    # Percentile rank of current dispersion vs lookback
    disp_pctile = np.full(n_rows, 50.0)
    for i in range(disp_lookback, n_rows):
        window = ema_disp[i - disp_lookback:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 1:
            disp_pctile[i] = (np.sum(valid < ema_disp[i]) / len(valid)) * 100

    # Trend SMA
    trend_sma = np.empty_like(closes_vals)
    trend_sma[:] = np.nan
    trend_cs = np.nancumsum(closes_vals, axis=0)
    for i in range(trend_window, n_rows):
        s = i - trend_window
        trend_sma[i] = (trend_cs[i] - trend_cs[s]) / trend_window

    # BTC regime gate
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

    rebal_indices = list(range(warmup, n_rows, rebalance_every))

    cash = initial_cash
    holdings = {}
    entry_prices = {}
    entry_bars = {}
    values = []

    for ri, i in enumerate(rebal_indices):
        # Per-bar take-profit, stop-loss, max-hold
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
                    if take_profit > 0:
                        d = dips[bar, ci]
                        if not np.isnan(d) and d >= -take_profit:
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

        # Dispersion filter: only trade in low-dispersion regimes
        if disp_pctile[i] > disp_thresh:
            for ci, qty in holdings.items():
                p = closes_vals[i, ci]
                if not np.isnan(p):
                    cash += qty * p
            holdings = {}
            entry_prices = {}
            entry_bars = {}
            continue

        # Find dip candidates
        winners = []
        for ci in range(n_cols):
            t = trend_sma[i, ci]
            if np.isnan(t) or closes_vals[i, ci] < t:
                continue
            d = dips[i, ci]
            if np.isnan(d):
                continue
            if -max_dip <= d <= -min_dip:
                winners.append((ci, d))

        winners.sort(key=lambda x: x[1])
        winners = [ci for ci, _ in winners[:top_n]]

        # Liquidate current holdings
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
    returns_pv = np.diff(pv) / pv[:-1]
    std = returns_pv.std()
    if std > 0:
        periods_per_year = (1440 * 365) / rebalance_every
        sharpe = (returns_pv.mean() / std) * np.sqrt(periods_per_year)
    else:
        sharpe = 0

    if return_equity:
        return {"total_return": total_return, "sharpe": sharpe, "equity_curve": values}
    return {"total_return": total_return, "sharpe": sharpe}


class DispersionMRStrategy:
    """
    Return Dispersion Mean Reversion for crypto.

    Buys VWAP dips only when cross-sectional return dispersion is low,
    indicating noise-driven moves rather than information-driven divergence.
    """

    GRID = {
        "vwap_window": [15, 25, 40],
        "min_dip": [0.0005, 0.001, 0.002],
        "max_dip": [0.008, 0.012, 0.020],
        "disp_window": [30, 60, 120],
        "disp_lookback": [240, 480, 720],
        "disp_thresh": [30, 50, 70],
        "top_n": [1, 2],
        "trend_window": [120, 240],
        "take_profit": [0.0, 0.002],
        "stop_loss": [0.0, 0.025],
        "max_hold": [0, 25],
        "regime_window": [0, 540],
    }
    REBALANCE_OPTIONS = [15, 30]

    def __init__(self, api: tradeapi.REST, symbols: list[str],
                 vwap_window: int = 25, min_dip: float = 0.001,
                 max_dip: float = 0.012, disp_window: int = 60,
                 disp_lookback: int = 480, disp_thresh: int = 50,
                 top_n: int = 1, trend_window: int = 120,
                 take_profit: float = 0.002, stop_loss: float = 0.0,
                 max_hold: int = 0, regime_window: int = 0):
        self.api = api
        self.symbols = symbols
        self.vwap_window = vwap_window
        self.min_dip = min_dip
        self.max_dip = max_dip
        self.disp_window = disp_window
        self.disp_lookback = disp_lookback
        self.disp_thresh = disp_thresh
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
            for attr in ["vwap_window", "min_dip", "max_dip", "disp_window",
                         "disp_lookback", "disp_thresh", "top_n", "trend_window",
                         "take_profit", "stop_loss", "max_hold", "regime_window"]:
                if attr in p:
                    setattr(self, attr, p[attr])
            print(f"Loaded params from {pf}")

    def _save_params(self, rebalance_every=30, params_suffix=None):
        os.makedirs(PARAMS_DIR, exist_ok=True)
        pf = _params_file()
        if params_suffix:
            pf = pf.replace(".json", f"_{params_suffix}.json")
        params = {
            "vwap_window": self.vwap_window,
            "min_dip": self.min_dip,
            "max_dip": self.max_dip,
            "disp_window": self.disp_window,
            "disp_lookback": self.disp_lookback,
            "disp_thresh": self.disp_thresh,
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
        print(f"Optimizing dispersion MR strategy over {days} days of data...")
        history = self._fetch_history(days)
        if not history:
            print("No data -- keeping current params.")
            return

        closes = pd.DataFrame({sym: df["close"] for sym, df in history.items()})
        closes = closes.dropna(how="all").ffill()

        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from optimize import bayesian_search

        closes_vals = closes.values
        n_cols = closes_vals.shape[1]

        rebal_options = [fixed_interval] if fixed_interval else self.REBALANCE_OPTIONS
        fixed_args = (closes_vals, n_cols)

        best_params, best_rebal, best_sharpe, best_return = bayesian_search(
            _dispersion_mr_backtest_worker, self.GRID, rebal_options, fixed_args,
            n_trials=300)

        if best_params:
            for k, v in best_params.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            label = "Optimal" if best_return > 0 else "Best (still negative)"
            print(f"\n{label} params found:")
            for k, v in best_params.items():
                print(f"  {k}={v}")
            print(f"  rebalance_every={best_rebal}min")
            print(f"  Backtest return: {best_return:.2%} | Sharpe: {best_sharpe:.2f}")
            self._save_params(best_rebal, params_suffix=params_suffix)
            return best_rebal
        else:
            print("No valid params found -- keeping defaults.")
            return 30

    def _fetch_history(self, days: int, end_days_ago: int = 1) -> dict[str, pd.DataFrame]:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from backtest import fetch_history
        return fetch_history(self.api, self.symbols, days, end_days_ago=end_days_ago)

    def get_momentum_scores(self) -> pd.Series:
        """Return dip scores for live trading."""
        end = datetime.now()
        lookback = max(self.vwap_window, self.trend_window, self.disp_lookback) + 100
        start = end - timedelta(minutes=lookback)

        all_bars = {}
        for symbol in self.symbols:
            try:
                bars = self.api.get_crypto_bars(
                    symbol, tradeapi.TimeFrame.Minute,
                    start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=lookback,
                ).df
                if len(bars) > 0:
                    all_bars[symbol] = bars
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")

        if not all_bars:
            return pd.Series(dtype=float)

        closes = pd.DataFrame({s: b["close"] for s, b in all_bars.items()})
        closes = closes.dropna(how="all").ffill()

        # Check dispersion
        returns = closes.pct_change()
        cross_disp = returns.std(axis=1)
        ema_disp = cross_disp.ewm(span=self.disp_window).mean()
        lookback_disp = ema_disp.iloc[-self.disp_lookback:]
        current_pctile = (lookback_disp < ema_disp.iloc[-1]).sum() / len(lookback_disp) * 100

        if current_pctile > self.disp_thresh:
            return pd.Series(dtype=float)

        # VWAP dip scores
        vwap = closes.rolling(self.vwap_window).mean()
        dips = (closes - vwap) / vwap
        trend_sma = closes.rolling(self.trend_window, min_periods=20).mean()

        scores = {}
        for symbol in closes.columns:
            price = closes[symbol].iloc[-1]
            t = trend_sma[symbol].iloc[-1]
            if np.isnan(t) or price < t:
                continue
            d = dips[symbol].iloc[-1]
            if np.isnan(d):
                continue
            if -self.max_dip <= d <= -self.min_dip:
                scores[symbol] = d

        return pd.Series(scores).sort_values()

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
        print(f"\n[{datetime.now()}] Running dispersion MR strategy...")

        scores = self.get_momentum_scores()
        picks = list(scores.head(self.top_n).index)
        print(f"Picks: {picks}")

        if not picks:
            print("No dispersion MR candidates -- going to cash.")
            for pos in self.api.list_positions():
                sym = pos.symbol
                qty = float(pos.qty)
                if qty > 0:
                    print(f"  Closing {sym} ({qty})")
                    try:
                        self.api.submit_order(
                            symbol=sym, qty=abs(qty), side="sell",
                            type="market",
                            time_in_force="gtc" if is_crypto(sym) else "day",
                        )
                    except Exception as e:
                        print(f"  Error closing {sym}: {e}")
            return

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
                print(f"  Closing {symbol} ({qty})")
                try:
                    self.api.submit_order(
                        symbol=symbol, qty=abs(qty), side="sell",
                        type="market",
                        time_in_force="gtc" if is_crypto(symbol) else "day",
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
