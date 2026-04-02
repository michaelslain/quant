#!/usr/bin/env python3
"""
Lightweight live trader for Raspberry Pi.
Reads best_strategy.json, loads params, and trades. No backtesting or optimization.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import alpaca_trade_api as tradeapi

# Import strategy classes and symbols from main
sys.path.insert(0, str(Path(__file__).parent))
from main import (
    STOCK_STRATEGIES, CRYPTO_STRATEGIES,
    STOCK_SYMBOLS, CRYPTO_SYMBOLS,
    get_api,
)

BEST_STRATEGY_FILE = Path(__file__).parent / "best_strategy.json"
STATUS_FILE = Path(__file__).parent / "quant_status.json"


BASELINE_EQUITY = 100_000.00


def write_status(state, market=None, strategy=None, equity=None, day_start_equity=None, error=None):
    """Write current status to quant_status.json for the monitor."""
    # Read best strategies from best_strategy.json
    best = {}
    try:
        with open(BEST_STRATEGY_FILE) as f:
            data = json.load(f)
        for m in ("crypto", "stock"):
            if data.get(m, {}).get("strategy"):
                best[m] = data[m]["strategy"]
    except Exception:
        pass

    status = {
        "state": state,
        "market": market,
        "strategy": strategy,
        "best": best,
        "equity": equity,
        "day_start_equity": day_start_equity,
        "baseline_equity": BASELINE_EQUITY,
        "error": str(error) if error else None,
        "updated_at": datetime.now().isoformat(),
        "pid": os.getpid(),
    }
    tmp = STATUS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    tmp.rename(STATUS_FILE)


def load_best_strategy():
    """Read best_strategy.json and return {market: {strategy, cfg, kwargs, rebalance_every}}."""
    with open(BEST_STRATEGY_FILE) as f:
        data = json.load(f)

    result = {}
    for market in ("crypto", "stock"):
        name = data[market]["strategy"]
        if not name:
            continue

        strategies = CRYPTO_STRATEGIES if market == "crypto" else STOCK_STRATEGIES
        symbols = CRYPTO_SYMBOLS if market == "crypto" else STOCK_SYMBOLS
        cfg = strategies[name]
        kwargs = dict(cfg["kwargs"])
        rebalance_every = 30

        # Load optimized params
        params_file = Path(__file__).parent / market / "params" / f"{name}.json"
        try:
            with open(params_file) as f:
                params = json.load(f)
            rebalance_every = params.pop("rebalance_every", 30)
            params.pop("updated_at", None)
            kwargs.update(params)
            print(f"  {market}/{name}: loaded params (every {rebalance_every}min)")
        except FileNotFoundError:
            print(f"  {market}/{name}: using defaults (every {rebalance_every}min)")

        result[market] = {
            "name": name,
            "strategy": cfg["class"](api=get_api(), symbols=symbols, **kwargs),
            "rebalance_every": rebalance_every,
        }

    return result


def is_market_open():
    try:
        clock = get_api().get_clock()
        return clock.is_open
    except Exception as e:
        print(f"Error checking market: {e}")
        return False


def main():
    print("=== Quant Live Trader (Pi) ===")
    print(f"Reading {BEST_STRATEGY_FILE}...")

    strats = load_best_strategy()
    if not strats:
        print("No strategies in best_strategy.json. Run 'python main.py refresh' on Mac first.")
        sys.exit(1)

    for market, info in strats.items():
        print(f"  {market}: {info['name']} (every {info['rebalance_every']}min)")

    account = get_api().get_account()
    equity = float(account.equity)
    day_start_equity = float(account.last_equity)
    print(f"\nEquity: ${equity:,.2f} (start of day: ${day_start_equity:,.2f})")
    print("Trading loop started. Ctrl+C to stop.\n")
    write_status("starting", equity=equity, day_start_equity=day_start_equity)

    run_count = 0
    while True:
        market_open = is_market_open()
        market = "stock" if market_open else "crypto"

        if market not in strats:
            other = "crypto" if market == "stock" else "stock"
            if other in strats:
                market = other
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No strategy for current market. Waiting...")
                write_status("waiting", error="No strategy for current market")
                time.sleep(60)
                continue

        info = strats[market]
        run_count += 1
        print(f"[{datetime.now().strftime('%H:%M:%S')}] #{run_count} {market}/{info['name']}...")
        write_status("rebalancing", market=market, strategy=info["name"])

        try:
            info["strategy"].rebalance()
        except Exception as e:
            print(f"  Error: {e}")
            write_status("error", market=market, strategy=info["name"], error=e)

        # Show P&L
        try:
            account = get_api().get_account()
            equity = float(account.equity)
            day_start_equity = float(account.last_equity)
            pnl = equity - day_start_equity
            sign = "+" if pnl >= 0 else ""
            print(f"  Equity: ${equity:,.2f} | Today: {sign}${pnl:,.2f} ({sign}{(pnl/day_start_equity)*100:.2f}%)")
            write_status("running", market=market, strategy=info["name"], equity=equity, day_start_equity=day_start_equity)
        except Exception:
            write_status("running", market=market, strategy=info["name"])

        print(f"  Next in {info['rebalance_every']}min...\n")
        time.sleep(info["rebalance_every"] * 60)


if __name__ == "__main__":
    main()
