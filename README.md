# Quant Trading

Algorithmic trading system with backtesting, optimization, and paper trading via Alpaca.

## Setup

```bash
pip install -r requirements.txt
```

Add your Alpaca paper trading keys to `.env`:

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

## Commands

All commands follow: `python main.py <command> <stock|crypto> [strategy] [options]`

### Compare (view current scores)

```bash
python main.py compare stock
python main.py compare crypto
```

### Backtest

```bash
python main.py backtest stock                          # all stock strategies, 7 days
python main.py backtest crypto                         # all crypto strategies, 7 days
python main.py backtest crypto velocity_mr             # just velocity_mr, crypto
python main.py backtest stock mean_reversion 14        # mean_reversion, stocks, 14 days
python main.py backtest crypto --interval 1            # all crypto at 1-min interval
python main.py backtest crypto velocity_mr 365 --end-days-ago 430  # out-of-sample validation
```

### Optimize (find best params)

```bash
python main.py optimize crypto velocity_mr 60                # 60 days, Bayesian search
python main.py optimize stock mean_reversion 30              # 30 days of data
python main.py optimize stock --interval 5                   # optimize all stock strategies at 5-min interval
```

Saves params to `<market>/params/<strategy>.json` (or `<strategy>_<interval>min.json` with `--interval`).

Strategies with large grids use **Bayesian search** (Optuna TPE, 300 trials). Smaller grids use exhaustive grid search with multiprocessing.

### Trade (single rebalance)

```bash
python main.py trade crypto velocity_mr
python main.py trade stock mean_reversion
```

### Run (paper trade in a loop)

```bash
python main.py run crypto velocity_mr               # uses optimized interval from params
python main.py run crypto mean_reversion --interval 15  # override interval to 15 min
```

Shows live P&L tracking per position and running total.

### Refresh (optimize + backtest all strategies)

```bash
python main.py refresh
```

### Livetest (simulate combined live trading)

```bash
python main.py livetest 30    # 30 days of simulated live trading
```

## Strategy Performance (365-day Out-of-Sample)

Optimized on 60 days, validated on 365 separate days (`--end-days-ago 430`).

| Strategy | Signal | Return | Sharpe | Max DD |
|---|---|---|---|---|
| **velocity_mr** | Price velocity vs "speed limit" (Relativistic BS) | 29.89% | **3.87** | -3.53% |
| **kalman_reversion** | Kalman filter dynamic hedge ratio z-score | 16.61% | **3.06** | -2.10% |
| **regime_mr** | VWAP dip + BTC regime gate | 47.38% | 2.23 | -8.69% |
| **dispersion_mr** | Return dispersion filter + VWAP dip | 41.63% | 2.01 | -8.40% |
| **adaptive_mr** | ATR-adaptive dip thresholds | 27.21% | 1.85 | -8.84% |
| **breakout** | Channel breakout + ATR risk parity | 62.19% | 1.69 | -20.37% |
| **hurst_mr** | Per-coin Hurst exponent selection | 4.36% | 1.64 | -1.69% |
| **mean_reversion** | VWAP dip buying with trend filter | 17.55% | 1.54 | -4.03% |

## Project Structure

```
quant/
  main.py                 # CLI entry point
  backtest.py             # vectorized backtester with per-day caching
  optimize.py             # grid search + Bayesian search (Optuna TPE)
  quant_runner.py         # long-running auto-pick trader
  run_live.py             # lightweight Pi live trader
  livetest.py             # simulated combined live trading
  crypto/
    strategies/           # 24 crypto strategies
      velocity_mr.py      # price velocity "speed limit" mean reversion
      kalman_reversion.py # Kalman filter hedge ratio z-score
      regime_mr.py        # VWAP dip + BTC regime gate + risk management
      dispersion_mr.py    # cross-sectional return dispersion filter
      adaptive_mr.py      # ATR-adaptive dip thresholds
      hurst_mr.py         # per-coin Hurst exponent ranking
      mean_reversion.py   # VWAP dip buying with trend filter
      breakout.py         # channel breakout with ATR risk parity
      bayesian.py         # Bayesian log-odds posterior scoring
      beta_reversion.py   # beta-adjusted cross-sectional + Hurst regime
      ou_reversion.py     # Ornstein-Uhlenbeck process parameters
      fracdiff_mr.py      # fractional differentiation z-score
      ...                 # + more experimental strategies
    params/               # optimized params (JSON)
  stock/
    strategies/           # 5 stock strategies
    params/
  .cache/                 # per-day parquet bar cache + result cache
```

## Traded Assets

### Crypto
BTC/USD, ETH/USD, SOL/USD, DOGE/USD, XRP/USD, ADA/USD, LINK/USD, LTC/USD

### Stocks
TQQQ, SOXL, UPRO, TNA, LABU, COIN, HOOD, SOFI, MARA, RIOT, PLTR, IONQ, SMCI, AFRM, UPST

## Key Findings

- **Per-bar take-profit** is the single most impactful feature -- `take_profit=0.002` is the universal sweet spot across all mean reversion strategies
- **Simpler signals win**: adding filters or complexity to a working strategy consistently hurts OOS performance
- **Mean reversion dominates crypto**: the top 8 strategies are all mean reversion variants; trend-following tends to overfit
- **BTC regime gate** (only trade when BTC > 540-bar SMA) improves most strategies
- **Drawdown control** (50% at -8% DD, liquidate at -15% DD) helps strategies designed for it but hurts momentum-based ones
- **Concentrated picks** (top_n=1-2) and **longer trend filters** (240-480 bars) improve results
- Strategy params are stable across re-optimizations -- Bayesian search converges to the same optimal

## Optimization & Validation Workflow

```bash
# 1. Optimize on 60 days of recent data
python main.py optimize crypto velocity_mr 60

# 2. Validate on 365 separate out-of-sample days
python main.py backtest crypto velocity_mr 365 --end-days-ago 430
```

The `--end-days-ago 430` ensures no overlap (365 + 60 + 5 buffer = 430).

Run **one optimization at a time** -- they use half of available CPU cores.

## Data Caching

Two-layer caching:
1. **Data cache** (`.cache/*.parquet`): per-symbol per-day 1-min bars. Only new days fetched from Alpaca.
2. **Result cache** (`.cache/results/*.json`): keyed by strategy code hash + params hash + date range. Auto-busts when code or params change.
