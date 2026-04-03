# Crypto Strategy Research: Cross-Sectional Beta-Adjusted Mean Reversion

## Core Thesis

At 15-60 min rebalancing, crypto shows strong **short-term mean reversion** (Makarov & Schoar 2020, Bianchi & Babiak 2022). Altcoins are highly correlated with BTC — temporary divergences from their expected BTC-relative return revert. The strategy exploits this by:

1. Measuring each coin's **beta-adjusted deviation** from expected return
2. Buying the most oversold coins (relative to their beta-predicted move)
3. Sizing by inverse volatility
4. Gating entry with a regime filter (Hurst exponent)

---

## Math

### 1. Beta Estimation

For each asset i, estimate rolling beta to BTC over W_beta bars:

```
r_i(t) = (P_i(t) - P_i(t-1)) / P_i(t-1)       # 1-bar return
r_BTC(t) = (P_BTC(t) - P_BTC(t-1)) / P_BTC(t-1)

beta_i = Cov(r_i, r_BTC) / Var(r_BTC)            # over trailing W_beta bars
```

W_beta = 240 bars (4 hours at 1-min resolution).

### 2. Beta-Adjusted Return (Residual)

The "alpha" return — how much asset i moved beyond what BTC predicted:

```
R_i(t, L) = sum_{k=0}^{L-1} r_i(t-k)            # cumulative return over lookback L
R_BTC(t, L) = sum_{k=0}^{L-1} r_BTC(t-k)

residual_i(t) = R_i(t, L) - beta_i * R_BTC(t, L)
```

L = lookback window (60-120 bars).

### 3. Z-Score of Residual

Normalize the residual by its rolling standard deviation:

```
mu_i = mean(residual_i) over trailing W_z bars
sigma_i = std(residual_i) over trailing W_z bars

z_i(t) = (residual_i(t) - mu_i) / sigma_i
```

W_z = 120 bars. Entry when z_i < -z_entry (e.g., -1.5). Exit when z_i > -z_exit (e.g., 0).

### 4. Hurst Exponent Regime Gate

Estimate rolling Hurst exponent H over W_hurst bars using rescaled range (R/S) method:

```
For a series X of length n:
  Y(t) = X(t) - mean(X)                          # mean-adjusted series
  Z(t) = cumsum(Y)                                # cumulative deviation
  R = max(Z) - min(Z)                             # range
  S = std(X)                                      # standard deviation
  H = log(R/S) / log(n)                           # Hurst exponent
```

Simplified practical version using multiple sub-windows and regression:
- Split W_hurst bars into chunks of sizes [32, 64, 128, 256]
- Compute R/S for each chunk size
- H = slope of log(R/S) vs log(chunk_size)

**Interpretation:**
- H < 0.45: mean-reverting regime (TRADE)
- H > 0.55: trending regime (DON'T TRADE mean reversion)
- 0.45-0.55: random walk (trade with reduced size)

Use BTC's Hurst exponent as the regime gate for the whole portfolio.

### 5. Volatility-Scaled Position Sizing

Target a fixed annualized volatility per position:

```
sigma_realized_i = std(r_i) over trailing 120 bars
sigma_annual_i = sigma_realized_i * sqrt(1440 * 365)   # annualize from 1-min bars

weight_i = target_vol / sigma_annual_i
weight_i = min(weight_i, max_weight)                    # cap at max_weight per position
```

target_vol = 15% annualized. max_weight = 0.5 (50% of portfolio per position).

### 6. Portfolio Drawdown Control

Track portfolio equity E(t) and peak equity E_peak(t):

```
drawdown(t) = (E(t) - E_peak(t)) / E_peak(t)

if drawdown < -dd_reduce:   scale all positions by 0.5
if drawdown < -dd_flat:     go to cash entirely
```

dd_reduce = 0.08 (8%), dd_flat = 0.15 (15%).

### 7. ATR Stop-Loss

Per-position trailing stop using ATR:

```
ATR_i = rolling_mean(|high_i - low_i|, 60 bars)     # or use true range if available
stop_price_i = entry_price_i - stop_atr_mult * ATR_i
```

stop_atr_mult = 2.5

---

## Entry/Exit Rules Summary

**Entry:**
1. Hurst(BTC, 240) < hurst_threshold (regime is mean-reverting)
2. z_i(t) < -z_entry (asset is oversold relative to beta-predicted return)
3. Price above trend SMA (uptrend filter)
4. Portfolio drawdown < dd_flat

**Exit:**
1. z_i(t) > -z_exit (residual reverted)
2. Stop-loss hit (entry - stop_atr_mult * ATR)
3. Max hold time exceeded
4. Portfolio drawdown triggers reduce/flat

**Sizing:**
- weight = target_vol / realized_vol, capped at max_weight
- If Hurst is in ambiguous zone (0.45-0.55), halve the weight
- If drawdown > dd_reduce, halve all weights

---

## Parameter Grid (for optimization)

```
lookback: [60, 90, 120]           # residual lookback L
beta_window: [180, 240, 360]      # W_beta for beta estimation
z_window: [90, 120, 180]          # W_z for z-score normalization
z_entry: [1.2, 1.5, 2.0]         # z-score threshold for entry
z_exit: [0.0, 0.3]               # z-score threshold for exit
hurst_window: [180, 240]          # W_hurst
hurst_threshold: [0.45, 0.50]    # max Hurst for entry
trend_window: [120, 240]          # trend SMA filter
top_n: [1, 2]                     # max positions
stop_atr_mult: [2.0, 2.5, 3.0]   # ATR stop multiplier
max_hold: [30, 60, 120]           # max bars to hold
```

Grid size: 3*3*3*3*2*2*2*2*2*3*3 = 23,328 combos (within budget)
