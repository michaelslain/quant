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
python main.py backtest crypto momentum                # just momentum, crypto
python main.py backtest stock mean_reversion 14        # mean_reversion, stocks, 14 days
python main.py backtest crypto --interval 1            # all crypto at 1-min interval
```

### Optimize (find best params via grid search)

```bash
python main.py optimize stock mean_reversion 30              # 30 days of data
python main.py optimize stock --interval 5                   # optimize all stock strategies at 5-min interval
python main.py optimize crypto breakout 14 --interval 1      # interval-specific optimization
```

Saves params to `<market>/params/<strategy>.json` (or `<strategy>_<interval>min.json` with `--interval`).

### Trade (single rebalance)

```bash
python main.py trade stock mean_reversion
python main.py trade crypto momentum
```

### Run (paper trade in a loop)

```bash
python main.py run stock mean_reversion              # uses optimized interval from params
python main.py run crypto mean_reversion             # uses optimized interval from params
python main.py run crypto mean_reversion --interval 15  # override interval to 15 min
```

Shows live P&L tracking per position and running total.

## Project Structure

```
quant/
  main.py                 # CLI entry point
  backtest.py             # vectorized backtester with per-day caching
  optimize.py             # multiprocessing grid search helper
  crypto/
    strategies/
      momentum.py         # RSI + Bollinger Bands + volume breakout
      mean_reversion.py   # VWAP dip buying with trend filter
      breakout.py         # channel breakout with ATR risk parity
      rsi_mean_revert.py  # RSI oversold bounce with BB support
      bayesian.py         # Bayesian log-odds posterior scoring
      bandit.py           # UCB multi-armed bandit
      monte_carlo.py      # Monte Carlo price simulation
    params/               # optimized params (JSON)
  stock/
    strategies/
      momentum.py         # RSI + trend filter + volume confirmation
      mean_reversion.py   # VWAP dip buying with trend filter
      breakout.py         # channel breakout with ATR risk parity
      rsi_mean_revert.py  # RSI oversold bounce with BB support
      ema_crossover.py    # EMA crossover trend following
    params/               # optimized params (JSON)
  .cache/                 # per-day parquet bar cache
```

## Traded Assets

### Crypto
| Symbol | Name |
|--------|------|
| BTC/USD | Bitcoin |
| ETH/USD | Ethereum |
| SOL/USD | Solana |
| DOGE/USD | Dogecoin |
| XRP/USD | Ripple |
| ADA/USD | Cardano |
| LINK/USD | Chainlink |
| LTC/USD | Litecoin |

### Stocks
| Symbol | Name |
|--------|------|
| TQQQ | ProShares UltraPro QQQ (3x Nasdaq) |
| SOXL | Direxion Semiconductor Bull 3x |
| UPRO | ProShares UltraPro S&P 500 (3x S&P) |
| TNA | Direxion Small Cap Bull 3x |
| LABU | Direxion Biotech Bull 3x |
| COIN | Coinbase |
| HOOD | Robinhood |
| SOFI | SoFi Technologies |
| MARA | Marathon Digital |
| RIOT | Riot Platforms |
| PLTR | Palantir Technologies |
| IONQ | IonQ (Quantum Computing) |
| SMCI | Super Micro Computer |
| AFRM | Affirm |
| UPST | Upstart |

## Strategies

### Stock (5 strategies)
- **mean_reversion** -- buys stocks that dipped below VWAP in an uptrend, expecting a bounce. Self-tuning via grid search. Best performer (+23% on unseen data).
- **momentum** -- RSI momentum with trend filter and volume confirmation. Self-tuning.
- **breakout** -- channel breakout with ATR-based risk parity sizing and trend filter. Self-tuning.
- **rsi_mean_revert** -- RSI oversold bounce with Bollinger Band support and trend filter. Self-tuning.
- **ema_crossover** -- EMA crossover trend following with volume confirmation. Self-tuning.

### Crypto (7 strategies)
- **mean_reversion** -- buys crypto that dipped below VWAP in an uptrend, expecting a bounce. Self-tuning. Best performer (+3.6% on unseen data).
- **momentum** -- trend-following with RSI, Bollinger Bands, and volume confirmation. Self-tuning.
- **breakout** -- channel breakout with ATR-based risk parity sizing. Self-tuning.
- **rsi_mean_revert** -- RSI oversold bounce with Bollinger Band support. Self-tuning. Second best (+0.9% on unseen data).
- **bayesian** -- Bayes' theorem posterior probability updates via momentum, volume, and RSI evidence. Self-tuning.
- **bandit** -- multi-armed bandit with Upper Confidence Bound (UCB) for explore/exploit balance. Self-tuning.
- **monte_carlo** -- Monte Carlo price path simulation with expectimax evaluation. Self-tuning.

## Optimization Tips

- Use **60 days** of training data for best generalization (reduces overfitting vs 7 or 14 days)
- Run **one optimization at a time** -- they're all multicore and will overload CPU if run in parallel
- Keep grid sizes under ~20K combos to avoid very long optimization runs
- Mean reversion strategies work best for leveraged ETFs and volatile crypto -- these assets are inherently mean-reverting intraday

## Data Caching

Historical data is cached per-symbol per-day in `.cache/`. First run fetches from Alpaca, subsequent runs load instantly.
