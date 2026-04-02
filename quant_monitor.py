#!/usr/bin/env python3
"""
Monitor for quant-trader systemd service on Raspberry Pi.
Checks service health, status file freshness, and recent logs.
Restarts the service if issues are detected.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

STATUS_FILE = Path(__file__).parent / "quant_status.json"
LOG_FILE = Path(__file__).parent / "monitor.log"
SERVICE_NAME = "quant-trader"
STALE_MINUTES = 60  # status older than this = problem


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def is_service_active():
    result = subprocess.run(
        ["systemctl", "is-active", SERVICE_NAME],
        capture_output=True, text=True,
    )
    return result.stdout.strip() == "active"


def restart_service():
    log(f"Restarting {SERVICE_NAME}...")
    result = subprocess.run(
        ["sudo", "systemctl", "restart", SERVICE_NAME],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log(f"Restart failed: {result.stderr.strip()}")
        return False
    log("Restart succeeded.")
    return True


def get_recent_logs(lines=50):
    result = subprocess.run(
        ["journalctl", "-u", SERVICE_NAME, "-n", str(lines), "--no-pager"],
        capture_output=True, text=True,
    )
    return result.stdout


def check_status_file():
    """Check quant_status.json for freshness and errors. Returns (ok, message)."""
    if not STATUS_FILE.exists():
        return False, "Status file missing"

    try:
        with open(STATUS_FILE) as f:
            status = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return False, f"Status file unreadable: {e}"

    updated = datetime.fromisoformat(status["updated_at"])
    age = datetime.now() - updated
    if age > timedelta(minutes=STALE_MINUTES):
        return False, f"Status stale ({age.total_seconds()/60:.0f}min old)"

    if status.get("state") == "error":
        return False, f"Strategy error: {status.get('error', 'unknown')}"

    return True, f"OK: {status.get('state')} | {status.get('market')}/{status.get('strategy')} | equity=${status.get('equity', '?')}"


def count_recent_errors(log_text):
    """Count 'Error' lines in recent logs."""
    return sum(1 for line in log_text.splitlines() if "Error" in line)


def main():
    issues = []

    # Check systemd service
    if not is_service_active():
        issues.append("Service not active")
        log("Service is not active. Attempting restart...")
        if restart_service():
            time.sleep(5)
            if not is_service_active():
                issues.append("Service failed to restart")

    # Check status file
    ok, msg = check_status_file()
    if ok:
        log(msg)
    else:
        issues.append(msg)
        log(f"Status check failed: {msg}")

    # Check logs for repeated errors
    logs = get_recent_logs()
    error_count = count_recent_errors(logs)
    if error_count >= 5:
        issues.append(f"{error_count} errors in recent logs")
        log(f"High error count in logs: {error_count}")

    if issues:
        log(f"ISSUES FOUND: {'; '.join(issues)}")
        # If service is dead and restart didn't help, that's all we can do
        sys.exit(1)
    else:
        log("All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
