# Quant Trading Bot

Algorithmic paper trading system using Alpaca API. Supports stock and crypto markets.

## Structure

- `main.py` -- CLI entry point. Commands: compare, trade, run, backtest, optimize, refresh, livetest
- `backtest.py` -- vectorized backtester with per-day parquet caching in `.cache/`
- `optimize.py` -- multiprocessing grid search helper (`grid_search`, `find_best`)
- `quant_runner.py` -- long-running trader that auto-picks best strategy, writes status to ~/.claude/daemon/quant_status.json
- `run_live.py` -- lightweight Pi live trader, reads best_strategy.json and trades
- `livetest.py` -- simulates combined stock+crypto live trading schedule on historical data
- `crypto/strategies/` -- crypto-specific strategies (9 strategies)
- `stock/strategies/` -- stock-specific strategies (5 strategies)
- `<market>/params/` -- optimized parameters per strategy (JSON)
- `<market>/params/<strategy>_<interval>min.json` -- interval-specific params

## CLI Pattern

`python main.py <command> <stock|crypto> [strategy] [options]`

### Commands

- `compare <market>` -- view current scores for all strategies
- `trade <market> <strategy>` -- single rebalance
- `run <market> <strategy> [interval]` -- paper trade in a loop with live P&L tracking
- `backtest <market> [strategy] [days] [--interval N] [--end-days-ago N]` -- simulate on historical data
- `optimize <market> [strategy] [days] [--interval N]` -- grid search for best params
- `refresh` -- optimize + backtest all strategies + save best to best_strategy.json
- `livetest [days]` -- simulate combined stock+crypto live trading on historical data using best_strategy.json

### Flags

- `--interval N` -- force a specific rebalance interval (minutes); uses interval-specific params files

## Key Conventions

- Use `get_crypto_bars` (not `get_bars`) for crypto symbols
- Use `get_latest_crypto_trades([symbol])[symbol].price` for crypto prices
- Crypto orders need `time_in_force="gtc"` (not "day")
- Stock orders use `time_in_force="day"`
- Crypto positions are fractional: use `float(pos.qty)` not `int`
- Stock positions are integer: use `int(per_stock // price)`
- Alpaca returns crypto positions without slash (DOGEUSD not DOGE/USD) -- use `is_crypto()` helper for detection
- Sharpe calculation: crypto uses `(1440 * 365) / rebalance_every`, stocks use `(390 * 252) / rebalance_every`
- Python (Mac): `/Users/michaelslain/miniconda3/bin/python`
- Python (Pi): `/home/m/dev/quant/.venv/bin/python`

## Caching

Two-layer caching speeds up iteration:

1. **Data caching** (`.cache/*.parquet`): Per-symbol, per-day 1-minute bars cached to disk. On subsequent backtests, only new days are fetched from Alpaca.
2. **Result caching** (`.cache/results/*.json`): Full backtest results cached. Cache is keyed by strategy code hash + params hash + date, so automatically busts when code or params change.

Cache only speeds up data fetch & backtest computation—the strategy optimization always runs fresh.

## Optimization & Validation Workflow

Optimize on 60 days, then validate on 365 days of separate out-of-sample data:
```
python main.py optimize crypto momentum 60
python main.py backtest crypto momentum 365 --end-days-ago 430
```
The `--end-days-ago 430` ensures the backtest period (365 days) doesn't overlap with the 60-day optimization window (365 + 60 + 5 buffer = 430 days back).

Additional notes:
- All strategies use multiprocessing grid search (half of available cores) -- run ONE optimization at a time to avoid CPU overload
- Mean reversion strategies generalize best; trend-following (momentum, breakout) tends to overfit on these assets
- Longer trend filters (240-480 bars) and concentrated picks (top_n=1-2) improve results
- Keep grid sizes under ~20K combos to avoid very long optimization runs

## Strategy Pattern

Each strategy file follows the same pattern:
- Top-level `_worker()` function for multiprocessing (must be picklable)
- Strategy class with: `GRID`, `REBALANCE_OPTIONS`, `__init__`, `_load_params`, `_save_params`, `optimize`, `_fetch_history`, `get_momentum_scores`, `get_target_positions`, `rebalance`
- `optimize()` accepts: `days`, `fixed_interval`, `params_suffix`
- `_fetch_history()` accepts: `days`, `end_days_ago`
- `_save_params()` accepts: `rebalance_every`, `params_suffix`

## Notable Strategy Features

- Stock mean reversion has intra-rebalance **take-profit**: exits early when dip reverts to VWAP (controlled by `take_profit` param)
- Crypto RSI mean revert has **ATR-adaptive RSI thresholds**: scales oversold level by current volatility (`adaptive_rsi` param)
- Both strategies support per-bar stop-loss checks between rebalances
- `backtest.py` mirrors these features (take-profit for mean reversion, bounce_window/use_bb/adaptive for RSI)
- Crypto beta reversion uses **Hurst exponent regime gate**: only trades when BTC Hurst < threshold (mean-reverting regime), with beta-adjusted z-score entry/exit and volatility-scaled sizing
