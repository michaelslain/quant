#!/usr/bin/env python3
"""Pretty-print quant trader status."""

import json
import subprocess
import sys
from pathlib import Path

STATUS_FILE = Path(__file__).parent / "quant_status.json"


def main():
    # Check systemd service
    result = subprocess.run(
        ["systemctl", "is-active", "quant-trader"],
        capture_output=True, text=True,
    )
    service_state = result.stdout.strip()

    # Read status file
    try:
        with open(STATUS_FILE) as f:
            s = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Quant Trading:")
        print("-" * 70)
        print(f"  Service:   {service_state}")
        print("  Status file not found or unreadable.")
        return

    state = s.get("state", "unknown")
    market = s.get("market", "?")
    strategy = s.get("strategy", "?")
    equity = s.get("equity")
    day_start = s.get("day_start_equity")
    baseline = s.get("baseline_equity", 100_000)
    best = s.get("best", {})
    updated = s.get("updated_at", "?")
    pid = s.get("pid", "?")
    error = s.get("error")

    # Format best strategies
    best_parts = [f"{m}={name}" for m, name in best.items()]
    best_str = ", ".join(best_parts) if best_parts else "?"

    # Format today's P&L
    if equity and day_start:
        daily_pnl = equity - day_start
        pct = (daily_pnl / day_start) * 100
        sign = "+" if daily_pnl >= 0 else ""
        pnl_str = f"{sign}${daily_pnl:,.2f} ({sign}{pct:.2f}%)"
    else:
        pnl_str = "?"

    # Format equity
    if equity:
        equity_str = f"${equity:,.2f}"
    else:
        equity_str = "?"

    print("Quant Trading:")
    print("-" * 70)
    print(f"  State:     {state}")
    print(f"  Market:    {market} | Strategy: {strategy}")
    print(f"  Best:      {best_str}")
    print(f"  Equity:    {equity_str}")
    print(f"  Today:     {pnl_str}")
    if error:
        print(f"  Error:     {error}")
    print(f"  Updated:   {updated}")
    print(f"  PID:       {pid}")


if __name__ == "__main__":
    main()
