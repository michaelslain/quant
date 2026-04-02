#!/usr/bin/env python3
"""
Long-running quant trader that automatically picks and runs the best strategy.

- Backtests all strategies on 30 days
- Picks the strategy with the best backtest return
- Runs the best stock strategy when market is open, crypto when closed
- Writes status to ~/.claude/daemon/quant_status.json for monitoring
- Switches strategies at market open/close boundaries
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

QUANT_DIR = Path(__file__).parent
sys.path.insert(0, str(QUANT_DIR))

PYTHON = "/Users/michaelslain/miniconda3/bin/python"
MAIN_PY = QUANT_DIR / "main.py"
STATUS_FILE = Path.home() / ".claude" / "daemon" / "quant_status.json"
BEST_STRATEGY_FILE = QUANT_DIR / "best_strategy.json"
LOG_DIR = Path.home() / ".claude" / "daemon" / "logs"

# Load env and imports
from dotenv import load_dotenv
load_dotenv(QUANT_DIR / ".env")
import alpaca_trade_api as tradeapi

from backtest import run_backtest
from main import STOCK_STRATEGIES, CRYPTO_STRATEGIES, STOCK_SYMBOLS, CRYPTO_SYMBOLS


def get_api():
    return tradeapi.REST(
        key_id=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        base_url="https://paper-api.alpaca.markets",
    )


def write_status(data):
    """Write status JSON for the monitor to read."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now().isoformat()
    data["pid"] = os.getpid()
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def write_best_strategy(best_crypto, crypto_return, best_stock, stock_return):
    """Write best_strategy.json so the Pi runner knows what to trade."""
    data = {
        "updated_at": datetime.now().isoformat(),
        "crypto": {
            "strategy": best_crypto,
            "backtest_return": crypto_return,
        },
        "stock": {
            "strategy": best_stock,
            "backtest_return": stock_return,
        },
    }
    with open(BEST_STRATEGY_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {BEST_STRATEGY_FILE}")


def pick_best_strategy(market, days=365, end_days_ago=430):
    """
    Validate all strategies on out-of-sample data. Pick the best by return.
    Uses 365 days of data ending 430 days ago to avoid overlap with the
    60-day optimization window.
    """
    strategies = CRYPTO_STRATEGIES if market == "crypto" else STOCK_STRATEGIES
    symbols = CRYPTO_SYMBOLS if market == "crypto" else STOCK_SYMBOLS
    api = get_api()

    write_status({
        "state": "backtesting",
        "market": market,
        "message": f"Validating all {market} strategies over {days}d out-of-sample data...",
    })

    results = {}
    for name, cfg in strategies.items():
        print(f"\n--- {name} ({market}) ---")

        # Load optimized params
        kwargs = dict(cfg["kwargs"])
        params_file = QUANT_DIR / market / "params" / f"{name}.json"
        rebalance_every = 30
        try:
            with open(params_file) as f:
                params = json.load(f)
            rebalance_every = params.pop("rebalance_every", 30)
            params.pop("updated_at", None)
            kwargs.update(params)
            print(f"  Using optimized params from {params_file}")
        except FileNotFoundError:
            print(f"  No optimized params, using defaults")

        result = run_backtest(
            strategy_cls=cfg["class"],
            strategy_kwargs=kwargs,
            api=api,
            symbols=symbols,
            days=days,
            rebalance_every=rebalance_every,
            end_days_ago=end_days_ago,
        )

        ret = result["total_return"] if result else None
        results[name] = {"return": ret, "rebalance_every": rebalance_every}
        print(f"  Return: {ret:+.2%}" if ret is not None else "  Return: N/A")

    # Pick best by return (skip if all negative)
    best_name = None
    best_ret = -999
    for name, r in results.items():
        if r["return"] is not None and r["return"] > best_ret:
            best_ret = r["return"]
            best_name = name

    if best_name is not None and best_ret < 0:
        print(f"  All {market} strategies have negative returns — skipping.")
        best_name = None
        best_ret = 0

    # Log summary
    log_file = LOG_DIR / f"quant_backtest_{market}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as f:
        f.write(f"=== {market.upper()} Strategy Selection ({days}d backtest) ===\n\n")
        f.write(f"{'Strategy':<20} {'Return':>10}\n")
        f.write(f"{'-'*30}\n")
        for name, r in sorted(results.items(), key=lambda x: x[1]["return"] or -999, reverse=True):
            ret_str = f"{r['return']:+.2%}" if r["return"] is not None else "N/A"
            marker = " <-- BEST" if name == best_name else ""
            f.write(f"{name:<20} {ret_str:>10}{marker}\n")

    return best_name, best_ret


def is_market_open():
    """Check if US stock market is currently open via Alpaca."""
    for attempt in range(3):
        try:
            api = get_api()
            clock = api.get_clock()
            return clock.is_open
        except Exception as e:
            print(f"Error checking market status (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(5)
    print("All retries failed for market status check, defaulting to closed")
    return False


def run_strategy(market, strategy):
    """Start main.py run as a subprocess. Returns the Popen object."""
    cmd = [PYTHON, str(MAIN_PY), "run", market, strategy]
    log_file = LOG_DIR / f"quant_run_{market}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_file, "a")

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=str(QUANT_DIR),
    )
    return proc, log_fh


def get_account_equity():
    """Get current account equity from Alpaca."""
    try:
        api = get_api()
        account = api.get_account()
        return float(account.equity)
    except Exception:
        return None


def main():
    print("=== Quant Runner Starting ===")
    print(f"PID: {os.getpid()}")

    # Fixed baseline: paper trading starting amount
    start_equity = 100000
    # Daily P&L tracking: snapshot equity at start of each day
    day_start_equity = get_account_equity() or start_equity
    day_start_date = datetime.now().date()
    print(f"Baseline equity: ${start_equity:,.2f}")
    print(f"Day start equity: ${day_start_equity:,.2f}")

    # Pick best strategies by 30-day backtest
    print("\nSelecting best crypto strategy...")
    best_crypto, crypto_return = pick_best_strategy("crypto")
    print(f"Best crypto: {best_crypto} ({crypto_return:+.2%})")

    print("\nSelecting best stock strategy...")
    best_stock, stock_return = pick_best_strategy("stock")
    print(f"Best stock: {best_stock} ({stock_return:+.2%})")

    write_best_strategy(best_crypto, crypto_return, best_stock, stock_return)

    if not best_crypto and not best_stock:
        write_status({
            "state": "error",
            "message": "No viable strategies found in backtests",
            "start_equity": start_equity,
        })
        print("ERROR: No viable strategies found. Exiting.")
        sys.exit(1)

    current_proc = None
    current_log_fh = None
    current_market = None

    def cleanup(signum=None, frame=None):
        nonlocal current_proc, current_log_fh
        if current_proc and current_proc.poll() is None:
            print("Stopping trading process...")
            current_proc.terminate()
            try:
                current_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                current_proc.kill()
        if current_log_fh:
            current_log_fh.close()
        write_status({
            "state": "stopped",
            "message": "Runner stopped cleanly",
            "best_crypto": best_crypto,
            "best_stock": best_stock,
            "start_equity": start_equity,
            "current_equity": get_account_equity(),
        })
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    print("\nStarting trading loop...")

    last_optimization_time = datetime.now()

    while True:
        market_open = is_market_open()
        target_market = "stock" if market_open else "crypto"
        target_strategy = best_stock if market_open else best_crypto

        # If no viable strategy for this market, idle until market switches
        if target_strategy is None:
            if current_proc and current_proc.poll() is None:
                print(f"No viable {target_market} strategy. Stopping trading...")
                current_proc.terminate()
                try:
                    current_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    current_proc.kill()
                if current_log_fh:
                    current_log_fh.close()
                current_proc = None
                current_log_fh = None
                current_market = None

            write_status({
                "state": "idle",
                "market": target_market,
                "best_crypto": best_crypto,
                "best_stock": best_stock,
                "start_equity": start_equity,
                "current_equity": get_account_equity(),
                "message": f"All {target_market} strategies negative — waiting for market to switch",
            })
            time.sleep(120)
            continue

        # Switch strategy if market changed
        if target_market != current_market:
            # Kill current process
            if current_proc and current_proc.poll() is None:
                print(f"Market {'opened' if market_open else 'closed'}. Switching to {target_market}/{target_strategy}...")
                current_proc.terminate()
                try:
                    current_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    current_proc.kill()
                if current_log_fh:
                    current_log_fh.close()

            # Start new strategy
            print(f"Starting {target_market}/{target_strategy}...")
            current_proc, current_log_fh = run_strategy(target_market, target_strategy)
            current_market = target_market

            current_equity = get_account_equity()
            pnl = (current_equity - day_start_equity) if day_start_equity and current_equity else None

            write_status({
                "state": "running",
                "market": current_market,
                "strategy": target_strategy,
                "best_crypto": best_crypto,
                "best_stock": best_stock,
                "crypto_return": crypto_return,
                "stock_return": stock_return,
                "start_equity": start_equity,
                "current_equity": current_equity,
                "day_start_equity": day_start_equity,
                "pnl": pnl,
                "message": f"Running {target_strategy} on {current_market}",
            })

        # Check if process is still alive
        if current_proc and current_proc.poll() is not None:
            exit_code = current_proc.returncode
            print(f"Trading process exited with code {exit_code}. Restarting...")
            write_status({
                "state": "restarting",
                "market": current_market,
                "strategy": target_strategy,
                "message": f"Process exited ({exit_code}), restarting...",
                "start_equity": start_equity,
                "current_equity": get_account_equity(),
            })
            if current_log_fh:
                current_log_fh.close()
            current_proc, current_log_fh = run_strategy(current_market, target_strategy)

        # Update equity periodically
        current_equity = get_account_equity()
        # Reset daily P&L at day boundary
        today = datetime.now().date()
        if today != day_start_date:
            day_start_equity = current_equity or day_start_equity
            day_start_date = today
            print(f"New day — resetting day start equity to ${day_start_equity:,.2f}")
        if day_start_equity and current_equity:
            pnl = current_equity - day_start_equity
            write_status({
                "state": "running",
                "market": current_market,
                "strategy": target_strategy,
                "best_crypto": best_crypto,
                "best_stock": best_stock,
                "start_equity": start_equity,
                "current_equity": current_equity,
                "day_start_equity": day_start_equity,
                "pnl": pnl,
                "pnl_pct": (pnl / day_start_equity) * 100,
                "message": f"Running {target_strategy} on {current_market} | Daily P&L: {'+'if pnl>=0 else ''}${pnl:,.2f}",
            })

        # Check if it's time to reoptimize (every 24 hours)
        current_time = datetime.now()
        time_since_optimization = current_time - last_optimization_time
        if time_since_optimization.total_seconds() >= 86400:  # 24 hours
            print(f"\n=== Daily Reoptimization at {current_time.strftime('%Y-%m-%d %H:%M:%S')} ===")

            old_best_crypto = best_crypto
            old_best_stock = best_stock

            # Reoptimize both markets
            print("\nReselecting best crypto strategy...")
            best_crypto, crypto_return = pick_best_strategy("crypto")
            print(f"Best crypto: {best_crypto} ({crypto_return:+.2%})" if best_crypto else "Best crypto: None (all negative)")

            print("\nReselecting best stock strategy...")
            best_stock, stock_return = pick_best_strategy("stock")
            print(f"Best stock: {best_stock} ({stock_return:+.2%})" if best_stock else "Best stock: None (all negative)")

            write_best_strategy(best_crypto, crypto_return, best_stock, stock_return)

            # If strategy changed for current market, restart trading process
            strategy_changed = (best_crypto != old_best_crypto and target_market == "crypto") or \
                              (best_stock != old_best_stock and target_market == "stock")

            if strategy_changed:
                print(f"Strategy changed! Restarting with new selection...")
                if current_proc and current_proc.poll() is None:
                    current_proc.terminate()
                    try:
                        current_proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        current_proc.kill()
                if current_log_fh:
                    current_log_fh.close()
                current_market = None  # Force restart in next iteration

            last_optimization_time = current_time

        # Check every 2 minutes
        time.sleep(120)


if __name__ == "__main__":
    main()
