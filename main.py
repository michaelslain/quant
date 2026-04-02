import os
import sys
import json
import hashlib
import inspect
from datetime import date
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from stock.strategies.momentum import MomentumStrategy
from stock.strategies.mean_reversion import MeanReversionStrategy
from stock.strategies.breakout import BreakoutStrategy
from stock.strategies.rsi_mean_revert import RsiMeanRevertStrategy as StockRsiMeanRevertStrategy
from stock.strategies.ema_crossover import EmaCrossoverStrategy
from crypto.strategies.momentum import CryptoMomentumStrategy
from crypto.strategies.mean_reversion import MeanReversionStrategy as CryptoMeanReversionStrategy
from crypto.strategies.breakout import CryptoBreakoutStrategy
from crypto.strategies.rsi_mean_revert import RsiMeanRevertStrategy
from crypto.strategies.bayesian import BayesianStrategy
from crypto.strategies.bandit import BanditStrategy
from crypto.strategies.monte_carlo import MonteCarloStrategy
from crypto.strategies.regime_mr import RegimeMeanReversionStrategy
from backtest import run_backtest

load_dotenv()

STOCK_SYMBOLS = [
    "TQQQ", "SOXL", "UPRO", "TNA", "LABU",
    "COIN", "HOOD", "SOFI", "MARA", "RIOT",
    "PLTR", "IONQ", "SMCI", "AFRM", "UPST",
]

CRYPTO_SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD",
    "XRP/USD", "ADA/USD", "LINK/USD", "LTC/USD",
]

STOCK_STRATEGIES = {
    "momentum": {
        "class": MomentumStrategy,
        "kwargs": {"lookback_minutes": 30, "top_n": 5},
    },
    "mean_reversion": {
        "class": MeanReversionStrategy,
        "kwargs": {"vwap_window": 60, "min_dip": 0.003, "max_dip": 0.015, "top_n": 3, "trend_window": 120, "volume_mult": 0.0, "velocity_window": 0, "recovery_bars": 0, "take_profit": 0.0},
    },
    "breakout": {
        "class": BreakoutStrategy,
        "kwargs": {"channel_period": 40, "volume_mult": 1.2, "atr_period": 14, "trend_window": 120, "top_n": 2},
    },
    "rsi_mean_revert": {
        "class": StockRsiMeanRevertStrategy,
        "kwargs": {"rsi_period": 14, "rsi_oversold": 30, "bb_period": 20, "bb_std": 2.0, "volume_mult": 1.2, "trend_window": 120, "top_n": 2},
    },
    "ema_crossover": {
        "class": EmaCrossoverStrategy,
        "kwargs": {"fast_period": 10, "slow_period": 60, "volume_mult": 1.0, "trend_window": 240, "top_n": 2},
    },
}

CRYPTO_STRATEGIES = {
    "momentum": {
        "class": CryptoMomentumStrategy,
        "kwargs": {"rsi_period": 14, "rsi_threshold": 53, "bb_period": 20, "bb_std": 2.0, "volume_mult": 1.3, "top_n": 3, "trend_window": 120},
    },
    "mean_reversion": {
        "class": CryptoMeanReversionStrategy,
        "kwargs": {"vwap_window": 60, "min_dip": 0.003, "max_dip": 0.015, "top_n": 3},
    },
    "breakout": {
        "class": CryptoBreakoutStrategy,
        "kwargs": {"channel_period": 40, "volume_mult": 1.2, "atr_period": 14, "trend_window": 120, "top_n": 2},
    },
    "rsi_mean_revert": {
        "class": RsiMeanRevertStrategy,
        "kwargs": {"rsi_period": 14, "rsi_oversold": 30, "bb_period": 20, "bb_std": 2.0, "volume_mult": 1.2, "trend_window": 120, "top_n": 2, "bounce_window": 3, "use_bb": 1, "stop_loss": 0.0, "adaptive_rsi": 0},
    },
    "bayesian": {
        "class": BayesianStrategy,
        "kwargs": {"sma_period": 20, "rsi_period": 14, "momentum_weight": 1.0, "volume_weight": 0.5, "rsi_weight": 0.5, "threshold": 0.6, "trend_window": 120, "top_n": 2},
    },
    "bandit": {
        "class": BanditStrategy,
        "kwargs": {"reward_window": 60, "exploration_factor": 1.0, "return_threshold": 0.001, "trend_window": 120, "top_n": 2},
    },
    "monte_carlo": {
        "class": MonteCarloStrategy,
        "kwargs": {"lookback": 60, "n_simulations": 100, "horizon": 10, "volume_weight": 0.5, "trend_window": 120, "top_n": 2},
    },
    "regime_mr": {
        "class": RegimeMeanReversionStrategy,
        "kwargs": {"vwap_window": 60, "min_dip": 0.005, "max_dip": 0.025, "top_n": 1, "trend_window": 240, "take_profit": 0.003, "stop_loss": 0.015, "max_hold": 30, "regime_window": 720, "volume_mult": 0.0},
    },
}


def get_api() -> tradeapi.REST:
    return tradeapi.REST(
        key_id=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        base_url="https://paper-api.alpaca.markets",
    )


def get_strategies(market):
    return CRYPTO_STRATEGIES if market == "crypto" else STOCK_STRATEGIES


def get_symbols(market):
    return CRYPTO_SYMBOLS if market == "crypto" else STOCK_SYMBOLS


RESULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache", "results")


def _result_cache_key(name, market, days, interval, strategy_cls):
    """Build a cache key from strategy source + params + date + settings."""
    # Hash the entire strategy .py file so cache busts when any code changes
    # (including top-level _worker functions, not just the class)
    src_file = inspect.getfile(strategy_cls)
    with open(src_file, "rb") as f:
        code_hash = hashlib.md5(f.read()).hexdigest()[:12]

    # Hash the params file content (if it exists)
    params_file = f"{market}/params/{name}.json"
    params_hash = "noparams"
    if os.path.exists(params_file):
        with open(params_file, "rb") as f:
            params_hash = hashlib.md5(f.read()).hexdigest()[:12]

    interval_str = f"{interval}min" if interval else "opt"
    today = date.today().isoformat()
    return f"{name}_{market}_{days}d_{interval_str}_{today}_{code_hash}_{params_hash}"


def _get_cached_result(cache_key, kind="backtest"):
    """Return cached result dict or None."""
    os.makedirs(RESULT_CACHE_DIR, exist_ok=True)
    path = os.path.join(RESULT_CACHE_DIR, f"{kind}_{cache_key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _save_cached_result(cache_key, result, kind="backtest"):
    """Save result dict to cache."""
    os.makedirs(RESULT_CACHE_DIR, exist_ok=True)
    path = os.path.join(RESULT_CACHE_DIR, f"{kind}_{cache_key}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def cmd_compare(api, market):
    strategies = get_strategies(market)
    symbols = get_symbols(market)

    account = api.get_account()
    print(f"Account equity: ${account.equity}")
    print(f"Buying power:   ${account.buying_power}\n")

    for name, cfg in strategies.items():
        strategy = cfg["class"](api=api, symbols=symbols, **cfg["kwargs"])
        scores = strategy.get_momentum_scores()
        targets = strategy.get_target_positions()

        print(f"=== {name} ===")
        print(f"\n  All scores:")
        for sym, score in scores.items():
            marker = " <-- BUY" if sym in targets else ""
            print(f"    {sym:<10} {score:>8.2%}{marker}")

        print(f"\n  Would buy {len(targets)} positions:")
        for sym, qty in targets.items():
            print(f"    {sym:<10} {qty} shares")
        print()


def cmd_trade(api, market, strategy_name):
    strategies = get_strategies(market)
    symbols = get_symbols(market)
    cfg = strategies[strategy_name]
    strategy = cfg["class"](api=api, symbols=symbols, **cfg["kwargs"])
    strategy.rebalance()


def cmd_run(api, market, strategy_name, interval_min=None):
    import time
    import json
    from datetime import datetime

    strategies = get_strategies(market)
    symbols = get_symbols(market)
    is_crypto = market == "crypto"

    cfg = strategies[strategy_name]
    kwargs = dict(cfg["kwargs"])

    # Load interval-specific params if available, else default
    if interval_min is not None:
        interval_params = f"{market}/params/{strategy_name}_{interval_min}min.json"
        default_params = f"{market}/params/{strategy_name}.json"
        try:
            with open(interval_params) as f:
                params = json.load(f)
            params.pop("rebalance_every", None)
            params.pop("updated_at", None)
            kwargs.update(params)
            print(f"Using {interval_min}min-optimized params from {interval_params}")
        except FileNotFoundError:
            try:
                with open(default_params) as f:
                    params = json.load(f)
                params.pop("rebalance_every", None)
                params.pop("updated_at", None)
                kwargs.update(params)
                print(f"No {interval_min}min params found, using {default_params}")
            except FileNotFoundError:
                pass
    else:
        params_file = f"{market}/params/{strategy_name}.json"
        try:
            with open(params_file) as f:
                params = json.load(f)
            interval_min = params.pop("rebalance_every", 5)
            params.pop("updated_at", None)
            kwargs.update(params)
            print(f"Using optimized params from {params_file} (interval: {interval_min}min)")
        except FileNotFoundError:
            interval_min = 5

    strategy = cfg["class"](api=api, symbols=symbols, **kwargs)

    # Track starting equity for P&L
    account = api.get_account()
    start_equity = float(account.equity)
    print(f"Starting equity: ${start_equity:,.2f}")
    print(f"Running '{strategy_name}' ({market}) every {interval_min} min. Ctrl+C to stop.\n")

    run_count = 0
    while True:
        if not is_crypto:
            clock = api.get_clock()
            if not clock.is_open:
                next_open = clock.next_open.strftime("%Y-%m-%d %H:%M")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Market closed. Opens at {next_open}. Waiting...")
                time.sleep(60)
                continue

        run_count += 1
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Rebalance #{run_count}...")
        try:
            strategy.rebalance()
        except Exception as e:
            print(f"  Error: {e}")

        fresh_api = get_api()
        account = fresh_api.get_account()
        equity = float(account.equity)
        pnl = equity - start_equity
        pnl_pct = (pnl / start_equity) * 100
        sign = "+" if pnl >= 0 else ""

        # Show positions
        positions = fresh_api.list_positions()
        if positions:
            print(f"  Positions:")
            for pos in positions:
                pos_pnl = float(pos.unrealized_pl)
                pos_sign = "+" if pos_pnl >= 0 else ""
                print(f"    {pos.symbol:<10} qty={float(pos.qty):<12} P&L: {pos_sign}${pos_pnl:,.2f}")

        print(f"  Equity: ${equity:,.2f} | P&L: {sign}${pnl:,.2f} ({sign}{pnl_pct:.2f}%)")
        print(f"  Next run in {interval_min} min...\n")
        time.sleep(interval_min * 60)


def cmd_optimize(api, market, strategy_name, days=7, interval=None):
    strategies = get_strategies(market)
    symbols = get_symbols(market)

    names = [strategy_name] if strategy_name else list(strategies.keys())
    for name in names:
        if name not in strategies:
            print(f"Unknown strategy '{name}'. Available: {', '.join(strategies.keys())}")
            continue

        cfg = strategies[name]

        # Check optimize cache (keyed by strategy code + date + days + interval)
        src_file = inspect.getfile(cfg["class"])
        with open(src_file, "rb") as f:
            code_hash = hashlib.md5(f.read()).hexdigest()[:12]
        interval_str = f"{interval}min" if interval else "opt"
        today = date.today().isoformat()
        opt_cache_key = f"{name}_{market}_{days}d_{interval_str}_{today}_{code_hash}"
        cached = _get_cached_result(opt_cache_key, "optimize")
        if cached:
            print(f"--- {name} ({market}) --- optimization already done today (cached)")
            if "sharpe" in cached:
                print(f"  Sharpe: {cached['sharpe']:.2f} | Return: {cached['return']:.2%}")
            print()
            continue

        strategy = cfg["class"](api=api, symbols=symbols, **cfg["kwargs"])

        if not hasattr(strategy, "optimize"):
            print(f"Strategy '{name}' does not support optimization.")
            continue

        suffix = f"{interval}min" if interval else None
        if interval:
            print(f"Optimizing '{name}' at fixed {interval}min interval (saving to {name}_{suffix}.json)")
        strategy.optimize(days=days, fixed_interval=interval, params_suffix=suffix)

        # Read the just-saved params to store results in cache
        params_file = f"{market}/params/{name}.json"
        if suffix:
            params_file = f"{market}/params/{name}_{suffix}.json"
        opt_result = {"done": True}
        try:
            with open(params_file) as f:
                saved = json.load(f)
            opt_result["params"] = {k: v for k, v in saved.items() if k != "updated_at"}
        except FileNotFoundError:
            pass
        _save_cached_result(opt_cache_key, opt_result, "optimize")
        print()


def cmd_backtest(api, market, strategy_name=None, days=7, interval=None, end_days_ago=1):
    strategies = get_strategies(market)
    symbols = get_symbols(market)
    names = [strategy_name] if strategy_name else list(strategies.keys())

    if interval:
        print(f"Forcing rebalance interval: {interval} min for all strategies\n")

    results = {}
    for name in names:
        cfg = strategies[name]
        kwargs = dict(cfg["kwargs"])
        rebalance_every = 30

        # If forced interval, try interval-specific params first, then fall back to default
        if interval:
            interval_params_file = f"{market}/params/{name}_{interval}min.json"
            default_params_file = f"{market}/params/{name}.json"
            try:
                with open(interval_params_file) as f:
                    params = json.load(f)
                params.pop("rebalance_every", None)
                params.pop("updated_at", None)
                kwargs.update(params)
                print(f"Using interval-optimized params from {interval_params_file}")
            except FileNotFoundError:
                try:
                    with open(default_params_file) as f:
                        params = json.load(f)
                    params.pop("rebalance_every", None)
                    params.pop("updated_at", None)
                    kwargs.update(params)
                    print(f"No {interval}min params found, using {default_params_file}")
                except FileNotFoundError:
                    pass
        else:
            params_file = f"{market}/params/{name}.json"
            try:
                with open(params_file) as f:
                    params = json.load(f)
                rebalance_every = params.pop("rebalance_every", 30)
                params.pop("updated_at", None)
                kwargs.update(params)
                print(f"Using optimized params from {params_file}")
            except FileNotFoundError:
                pass

        rebal = interval if interval else rebalance_every

        # Check cache
        cache_key = _result_cache_key(name, market, days, rebal, cfg["class"])
        cached = _get_cached_result(cache_key, "backtest")
        if cached:
            print(f"\n--- {name} ({market}) [every {rebal}min] --- (cached)")
            print(f"  Return: {cached['total_return']:.2%} | Sharpe: {cached['sharpe']:.2f} | Max DD: {cached['max_drawdown']:.2%}")
            results[name] = cached
            continue

        print(f"\n--- Backtesting: {name} ({market}) [every {rebal}min] ---")
        result = run_backtest(
            strategy_cls=cfg["class"],
            strategy_kwargs=kwargs,
            api=api,
            symbols=symbols,
            days=days,
            rebalance_every=rebal,
            end_days_ago=end_days_ago,
        )
        if result:
            results[name] = result
            _save_cached_result(cache_key, result, "backtest")

    # Print summary table if we backtested multiple strategies
    if len(results) > 1:
        print(f"\n{'='*65}")
        print(f"  SUMMARY ({market.upper()}, {days}d"
              f"{f', {interval}min interval' if interval else ', optimal intervals'})")
        print(f"{'='*65}")
        print(f"  {'Strategy':<20} {'Return':>10} {'Sharpe':>10} {'Max DD':>10}")
        print(f"  {'-'*50}")
        for name in sorted(results, key=lambda n: results[n]["total_return"], reverse=True):
            r = results[name]
            print(f"  {name:<20} {r['total_return']:>9.2%} {r['sharpe']:>10.2f} {r['max_drawdown']:>9.2%}")
        print(f"{'='*65}")


BEST_STRATEGY_FILE = os.path.join(os.path.dirname(__file__), "best_strategy.json")


def cmd_refresh(api):
    """Optimize all strategies (60d), backtest out-of-sample (365d), save best to best_strategy.json."""
    from datetime import datetime

    opt_days = 60
    bt_days = 365
    end_days_ago = opt_days + bt_days + 5  # no overlap

    best = {}

    for market in ("crypto", "stock"):
        strategies = get_strategies(market)
        symbols = get_symbols(market)

        # Step 1: Optimize all strategies
        print(f"\n{'='*60}")
        print(f"  OPTIMIZING {market.upper()} ({opt_days} days)")
        print(f"{'='*60}")
        for name, cfg in strategies.items():
            strategy = cfg["class"](api=api, symbols=symbols, **cfg["kwargs"])
            if hasattr(strategy, "optimize"):
                print(f"\n--- {name} ---")
                strategy.optimize(days=opt_days)
            else:
                print(f"\n--- {name} --- (no optimize method, skipping)")

        # Step 2: Backtest all strategies on out-of-sample data
        print(f"\n{'='*60}")
        print(f"  BACKTESTING {market.upper()} ({bt_days} days, ending {end_days_ago} days ago)")
        print(f"{'='*60}")

        results = {}
        for name, cfg in strategies.items():
            kwargs = dict(cfg["kwargs"])
            rebalance_every = 30

            params_file = f"{market}/params/{name}.json"
            try:
                with open(params_file) as f:
                    params = json.load(f)
                rebalance_every = params.pop("rebalance_every", 30)
                params.pop("updated_at", None)
                kwargs.update(params)
            except FileNotFoundError:
                pass

            print(f"\n--- {name} [every {rebalance_every}min] ---")
            result = run_backtest(
                strategy_cls=cfg["class"],
                strategy_kwargs=kwargs,
                api=api,
                symbols=symbols,
                days=bt_days,
                rebalance_every=rebalance_every,
                end_days_ago=end_days_ago,
            )
            if result:
                results[name] = result

        # Pick best by return
        best_name = None
        best_ret = -999
        for name, r in results.items():
            if r["total_return"] > best_ret:
                best_ret = r["total_return"]
                best_name = name

        if best_name and best_ret < 0:
            print(f"\n  All {market} strategies negative — no pick.")
            best_name = None
            best_ret = 0

        best[market] = {
            "strategy": best_name,
            "backtest_return": best_ret,
        }

        # Summary
        if results:
            print(f"\n{'='*60}")
            print(f"  {market.upper()} RESULTS")
            print(f"{'='*60}")
            print(f"  {'Strategy':<20} {'Return':>10} {'Sharpe':>10} {'Max DD':>10}")
            print(f"  {'-'*50}")
            for name in sorted(results, key=lambda n: results[n]["total_return"], reverse=True):
                r = results[name]
                marker = " <--" if name == best_name else ""
                print(f"  {name:<20} {r['total_return']:>9.2%} {r['sharpe']:>10.2f} {r['max_drawdown']:>9.2%}{marker}")

    # Write best_strategy.json
    output = {
        "updated_at": datetime.now().isoformat(),
        "optimize_days": opt_days,
        "backtest_days": bt_days,
    }
    output.update(best)
    with open(BEST_STRATEGY_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  SAVED: best_strategy.json")
    print(f"{'='*60}")
    print(f"  Crypto: {best['crypto']['strategy'] or 'none'} ({best['crypto']['backtest_return']:+.2%})")
    print(f"  Stock:  {best['stock']['strategy'] or 'none'} ({best['stock']['backtest_return']:+.2%})")
    print(f"\nCommit & push to deploy to Pi.")


def main():
    api = get_api()
    args = sys.argv[1:]

    if not args:
        print("Usage: python main.py <command> <stock|crypto> [strategy] [options]")
        print("Commands: compare, trade, run, backtest, optimize, refresh")
        print("Examples:")
        print("  python main.py compare stock")
        print("  python main.py backtest crypto momentum 7")
        print("  python main.py backtest crypto momentum 365 --end-days-ago 430  # validate on out-of-sample data")
        print("  python main.py run crypto momentum")
        print("  python main.py run stock mean_reversion 15")
        print("  python main.py optimize crypto momentum 60")
        print("  python main.py backtest crypto --interval 1      # compare all at 1min")
        print("  python main.py refresh                           # optimize + backtest + save best")
        return

    cmd = args[0]

    if cmd == "refresh":
        cmd_refresh(api)
        return

    market = args[1] if len(args) > 1 else "stock"

    if market not in ("stock", "crypto"):
        print(f"Unknown market '{market}'. Use 'stock' or 'crypto'.")
        return

    print(f"Mode: {market.upper()}{' (24/7)' if market == 'crypto' else ''}\n")

    if cmd == "compare":
        cmd_compare(api, market)

    elif cmd == "trade":
        strategy_name = args[2] if len(args) > 2 else "momentum"
        cmd_trade(api, market, strategy_name)

    elif cmd == "run":
        # Parse --interval N from args
        interval = None
        filtered = []
        i = 2
        while i < len(args):
            if args[i] == "--interval" and i + 1 < len(args):
                interval = int(args[i + 1])
                i += 2
            else:
                filtered.append(args[i])
                i += 1
        strategy_name = filtered[0] if filtered else "momentum"
        # Also support old positional syntax: run crypto momentum 15
        if interval is None and len(filtered) > 1:
            interval = int(filtered[1])
        cmd_run(api, market, strategy_name, interval)

    elif cmd == "backtest":
        # Parse --interval N and --end-days-ago N from args
        interval = None
        end_days_ago = 1
        filtered = []
        i = 2
        while i < len(args):
            if args[i] == "--interval" and i + 1 < len(args):
                interval = int(args[i + 1])
                i += 2
            elif args[i] == "--end-days-ago" and i + 1 < len(args):
                end_days_ago = int(args[i + 1])
                i += 2
            else:
                filtered.append(args[i])
                i += 1
        strategy_name = None
        days = 7
        for f in filtered:
            try:
                days = int(f)
            except ValueError:
                strategy_name = f
        cmd_backtest(api, market, strategy_name, days, interval, end_days_ago)

    elif cmd == "optimize":
        # Parse --interval N from args
        interval = None
        filtered = []
        i = 2
        while i < len(args):
            if args[i] == "--interval" and i + 1 < len(args):
                interval = int(args[i + 1])
                i += 2
            else:
                filtered.append(args[i])
                i += 1
        strategy_name = None
        days = 7
        for f in filtered:
            try:
                days = int(f)
            except ValueError:
                strategy_name = f
        cmd_optimize(api, market, strategy_name, days, interval)

    else:
        print(f"Unknown command '{cmd}'. Use: compare, trade, run, backtest, optimize, refresh")


if __name__ == "__main__":
    main()
