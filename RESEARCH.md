# Quant Strategy Research

Research notes for both crypto and stock trading strategies. Covers mathematical foundations, academic references, and implementation insights.

---

## Mean Reversion Strategies

### VWAP Dip-Buying (Crypto & Stock)
Buy assets trading below VWAP in an uptrend. Score by dip depth `(price - VWAP) / VWAP`. Trend filter (SMA) prevents catching falling knives.

- **Stock variant** has intra-rebalance take-profit: exits early when dip reverts to VWAP
- **Regime variant** adds BTC regime gate (only trades in bullish market) + per-bar stop-loss/take-profit + max hold

### RSI Mean Revert (Crypto & Stock)
Buy oversold bounces (RSI recovers from below threshold within `bounce_window` bars).

- **ATR-adaptive RSI** (crypto): scales oversold threshold by current volatility — wider in high vol, tighter in low vol
- Bollinger Band support confirmation optional
- Per-bar stop-loss between rebalances

### Beta-Adjusted Cross-Sectional Reversion (Crypto)
Exploits altcoin-BTC correlation: temporary divergences from beta-predicted return revert.

**Math:**
```
beta_i = Cov(r_i, r_BTC) / Var(r_BTC)           # rolling OLS
residual_i = R_i(lookback) - beta_i * R_BTC(lookback)
z_i = (residual - mu) / sigma                     # over z_window
```
Entry: z < -z_entry AND Hurst(BTC) < threshold (mean-reverting regime)

**Hurst exponent** via rescaled range (R/S): H < 0.45 = mean-reverting, H > 0.55 = trending.

**References:** Makarov & Schoar 2020, Bianchi & Babiak 2022

### Ornstein-Uhlenbeck Optimal Reversion (Crypto) — NEW
Estimates OU process parameters directly from log-price series, giving a mathematically grounded entry/exit framework.

**Math:**
```
dS = theta * (mu - S) * dt + sigma * dW

# Discrete estimation via OLS:
S(t+1) = a * S(t) + b + epsilon
a = exp(-theta),  b = mu * (1 - a)
theta = -ln(a)                     # mean-reversion speed
mu = b / (1 - a)                   # long-run equilibrium
half_life = ln(2) / theta          # bars to half-revert
sigma_eq = sigma_res / sqrt((1 - a^2) / (2*theta))

# Entry/exit:
deviation = (log_price - mu) / sigma_eq
Entry: deviation < -entry_dev AND min_hl <= half_life <= max_hl
Exit: deviation > -exit_dev OR stop-loss OR max_hold
```

**Key insight:** Half-life tells you which assets are tradeable at your rebalance frequency. Only trade assets where half_life is between `min_hl` and `max_hl` minutes.

**Advantages over fixed z-score:** Calibrates actual reversion speed from data. No need for arbitrary lookback windows for z-score normalization.

**References:** Bertram 2010 (optimal entry/exit boundaries), ArbitrageLab OU model docs, PyQuantLab rolling OU backtest

### ATR-Adaptive Dip Reversion (Crypto) — NEW
VWAP dip-buying with volatility-adaptive entry thresholds. Instead of fixed min/max dip, scales by per-coin ATR ratio vs cross-sectional median.

**Math:**
```
atr_i = rolling_std(returns_i, atr_window)           # per-coin volatility
atr_ratio_i = atr_i / median(atr_all_coins)          # relative to peers
adaptive_min = base_min_dip * clamp(atr_ratio, 0.3, 3.0)
adaptive_max = base_max_dip * clamp(atr_ratio, 0.3, 3.0)
Entry: adaptive_min <= -dip <= adaptive_max AND trend filter AND regime gate
```

**Key insight:** High-vol coins need deeper dips to signal true reversion (not just noise). Low-vol coins revert from shallower dips. Cross-sectional normalization avoids market-wide vol shifts.

**OOS results:** 27.21% return / 1.85 Sharpe / -8.84% DD (365 days). take_profit=0.002 confirmed as universal sweet spot.

### Cross-Sectional Z-Score Mean Reversion (Crypto) — NEW
Buys coins that are most oversold relative to the crypto universe, using cross-sectional z-score of recent returns.

**Math:**
```
ret_i = (close_i / close_i_lag) - 1          for lookback period
z_i = (ret_i - mean(ret_all)) / std(ret_all) # cross-sectional
Entry: z < -z_entry AND close > trend_sma AND regime_ok
```

**Implementation:** `crypto/strategies/zscore_mr.py`. Optimal lookback=60, z_entry=2.0, trend_window=120, take_profit=0.002.

**OOS results:** 2.91% return / 0.46 Sharpe / -10.17% DD (365 days). Weak — cross-sectional return z-scores don't provide strong enough signal on 1-min data. The VWAP-based dip signal (used by regime_mr, adaptive_mr) is fundamentally better because it measures price-vs-volume-weighted-average rather than just price-vs-peers.

### Return Dispersion VWAP Reversion (Crypto) — NEW
Buys VWAP dips only when cross-sectional return dispersion is low. Low dispersion = individual coin movements are noise-driven (not information-driven) and revert more reliably.

**Math:**
```
ret_i = close_i / close_i_lag - 1             (per coin)
dispersion = std(ret_all_coins)                 (cross-sectional)
ema_disp = EMA(dispersion, disp_window)
disp_pctile = percentile_rank(ema_disp, disp_lookback)

dip = (price - rolling_mean) / rolling_mean
Entry: dip in [-max_dip, -min_dip]
       AND disp_pctile < disp_thresh (low dispersion)
       AND trend filter AND regime gate
```

**Key insight:** When all coins move together (low dispersion), individual dips are likely noise. When dispersion is high, some coins are moving on real information — dips may not revert.

**Implementation:** `crypto/strategies/dispersion_mr.py`. Optimal vwap_window=15, min_dip=0.002, max_dip=0.008, disp_window=60, disp_lookback=480, disp_thresh=50, regime_window=540.

**OOS results:** 41.63% return / 2.01 Sharpe / -8.40% DD (365 days). Strong — dispersion filter effectively separates noise-driven dips from information-driven ones.

### Hurst-Ranked VWAP Reversion (Crypto) — NEW
Per-coin Hurst exponent selects coins with strongest mean-reversion tendency, then buys VWAP dips. Position sized by (0.5 - H) / volatility.

**Math:**
```
H_i = Hurst(returns_i, hurst_window) via R/S analysis
    R/S = (max(cumdev) - min(cumdev)) / std(segment)
    H = slope of log(R/S) vs log(window_size)

Entry: H_i < h_max AND dip in [-max_dip, -min_dip] AND trend filter
Sizing: w_i = (0.5 - H_i) / sigma_i, normalized
```

**Key insight:** Per-coin Hurst works as a coin selector but is too restrictive at 1-min frequency — most coins hover around H=0.5, so the filter eliminates most candidates most of the time. Results in very conservative, low-activity trading.

**Implementation:** `crypto/strategies/hurst_mr.py`. Optimal hurst_window=360, h_max=0.5, vwap_window=40, regime_window=540.

**OOS results:** 4.36% return / 1.64 Sharpe / -1.69% DD (365 days). Ultra-conservative: tiny DD but weak returns. Hurst better used as regime gate (like beta_reversion's BTC Hurst gate) than as per-coin signal.

**References:** Mandelbrot & Wallis 1969 (R/S analysis), Lo 1991 (modified R/S), Kristoufek 2012 (time-varying H), Bianchi & Pianese 2018 (multifractal BTC)

### Low-Volume Dip Reversion (Crypto) — NEW
Buys VWAP dips occurring on below-average volume. The microstructure hypothesis: low-volume drops indicate thin order books, not genuine selling pressure, and revert faster.

**Math:**
```
dip = (price - VWAP) / VWAP
vol_ratio = volume / avg_volume
Entry: dip in [-max_dip, -min_dip] AND vol_ratio < vol_ceil AND trend filter
Score: dip * (1 / vol_ratio)  -- deeper dip on lower volume = stronger
```

**OOS results:** -0.23% return / 0.04 Sharpe / -15.45% DD (365 days). The low-volume hypothesis does NOT hold on 1-min crypto data. Optimizer chose vol_ceil=1.2 (barely filtering), suggesting volume isn't a useful discriminator for dip quality.

---

## Trend-Following Strategies

### Momentum (Crypto & Stock)
RSI + Bollinger Band breakout with volume confirmation. Buys strong upward momentum with expanding BB width.

```
Score = (RSI - 50) * volume_ratio
Requires: price in upper BB half, expanding BB width, above-average volume, above trend SMA
```

**Note:** Tends to overfit on crypto assets. Longer trend filters (240-480 bars) help.

### Breakout (Crypto & Stock)
Channel breakout (price > highest high over `channel_period`) with volume confirmation.

```
Score = (1/ATR) * volume_ratio     # inverse volatility = higher conviction
Sizing: risk parity via inverse ATR weights
```

**Note:** Risk management features (take-profit, stop-loss, regime gate) hurt breakout OOS (37.74%/1.06/-31.71% vs original 62.19%/1.69/-20.37%). Trend-following needs to let winners run — take-profit cuts off trends prematurely. Code supports all features but params keep them disabled.

### EMA Crossover (Stock)
Fast/slow EMA crossover with volume and trend confirmation.

---

## Probabilistic / ML Strategies

### Bayesian Posterior (Crypto)
Combines momentum, volume, RSI evidence via Bayes' theorem in log-odds form:
```
log_odds = 0  (neutral prior)
log_odds += momentum_weight * (price - SMA) / SMA * 10
log_odds += volume_weight * (volume - avg) / avg
log_odds += rsi_weight * (RSI - 50) / 20
P(up) = 1 / (1 + exp(-log_odds))
Buy if P(up) >= threshold
```

### Multi-Armed Bandit / UCB (Crypto)
Balances exploitation of profitable coins with exploration of under-traded ones:
```
UCB = avg_reward + exploration_factor * sqrt(ln(total_bars) / played)
```

### Monte Carlo Path Simulation (Crypto)
Bootstrap resampling of actual returns (preserves fat tails), scores by Sortino-like ratio:
```
Score = median_return / downside_std
Requires: P(profit) >= 55% AND median_return > 0
```

---

## Research Pipeline: Potential Future Strategies

### 1. Kalman Filter Dynamic Hedge Ratio
Replaces rolling OLS beta with continuously adapting Kalman filter. Produces uncertainty estimates for confidence-weighted sizing.

```
State: x(t) = [intercept, beta]^T
Observation: y(t) = H(t) * x(t) + v(t)
Innovation: e(t) = y(t) - H(t) * x_pred (= spread)
z(t) = e(t) / sqrt(S(t))  # auto-accounts for estimation uncertainty
```
Only 2 tunable params: Q (state noise), R (observation noise).

**References:** QuantStart Kalman pairs, Portfolio Optimization Book Ch. 15.6

### 2. Fractional Differentiation — IMPLEMENTED
Standard differencing (d=1) destroys long-memory structure. Fractional differencing (d in 0-1) preserves predictive memory while achieving stationarity.

```
(1 - B)^d * X(t) = sum w_k * X(t-k)
w_k = w_{k-1} * (k - 1 - d) / k    (truncate when |w_k| < 1e-5)
z_t = (fd_t - rolling_mean(fd)) / rolling_std(fd)
Entry: z < -z_entry AND trend filter AND regime gate
```

**Implementation:** `crypto/strategies/fracdiff_mr.py`. Optimal d=0.3, z_entry=2.5, trend_window=480, regime_window=720.

**OOS results:** 5.56% return / 0.62 Sharpe / -6.34% DD (365 days). Very conservative — low drawdown but weak returns. The frac-diff signal is too noisy on 1-min crypto data for a primary trading signal. Might work better as a filter/overlay on other strategies.

**References:** De Prado "Advances in Financial Machine Learning" Ch. 5

### 3. Hidden Markov Model Regime Gate + Entropy Sizing
3-state HMM (mean-reverting, trending, chaotic) with entropy-based position sizing:
```
H_t = -sum gamma_t(j) * ln(gamma_t(j))    # entropy of state distribution
f_adjusted = f_kelly * (1 - H_t / ln(K)) * 0.5
```
Low entropy = high confidence = larger positions.

**References:** Markov/HMM regime detection preprint 2026

### 4. Johansen Cointegration Multi-Leg
Finds optimal linear combinations across 3+ assets simultaneously via VECM. Richer mean-reverting structures than pairwise analysis.

**References:** Dynamic Cointegration Crypto Pairs (arXiv 2109.10662)

### 5. VPIN Flow Toxicity
Volume-synchronized probability of informed trading. Can be used as position-sizing overlay:
- VPIN > 0.55: reduce exposure (high adverse selection)
- VPIN < 0.35: increase exposure (low toxicity)

**References:** Easley et al., ScienceDirect 2025

### 6. Relativistic Black-Scholes Model
Extends Black-Scholes by imposing a finite maximal speed of price changes (analogous to the speed of light in relativity). The PDF of log-returns has compact support bounded by c_m * t, naturally producing a volatility frown/skew without requiring stochastic volatility.

```
dS/S = mu*dt + sigma*dW  (standard BS)
->  Relativistic extension via telegraphers equation (Euclidean Dirac)
    p(x,t) = 0 for |x| > c_m * t   (bounded log-returns)
```

**Key insight:** Market "speed of light" c_m is much smaller than theoretical max — observed max |log-return| ~ 10^0, not 10^12. Price movements face inherent friction from order book resistance. Could inform volatility modeling for options pricing or as a regime indicator (when price velocity approaches c_m, expect reversal).

**Implementation:** `crypto/strategies/velocity_mr.py`. Optimal vel_window=15, lookback=120, entry_frac=0.5, regime_window=540, take_profit=0.002.

**OOS results:** 29.89% return / 3.87 Sharpe / -3.53% DD (365 days). **Best Sharpe in portfolio.** The "speed limit" concept provides a fundamentally different and superior entry signal compared to VWAP dip — it identifies when a drop is exhausting its historical range.

**References:** Trzetrzelewski 2013 (arXiv:1307.5122)

### 7. Fractal-Chaotic Co-Driven Volatility Forecasting (FCOC)
Deep learning framework combining multifractal feature extraction with chaotic oscillation activation functions for volatility forecasting. Two innovations: Fractal Feature Corrector (FFC) using asymmetric MF-ADCCA for cross-asset fractal features, and Chaotic Oscillation Component (COC) replacing static ReLU activations.

**Key insight:** Financial time series have multifractal structure — volatility is non-uniform across time scales. Standard fractal analysis is unstable; the FFC stabilizes it. The COC addresses the "complexity mismatch" where static activation functions can't process chaotic financial signals. Validated on S&P 500 and DJI.

**Potential use:** Volatility forecasting overlay — predict next-period vol to dynamically adjust position sizing or take-profit/stop-loss thresholds.

**References:** Zeng et al. 2025 (arXiv:2511.10365)

### 8. Mean-Field Games with Differing Beliefs for Algorithmic Trading
Game-theoretic model where heterogeneous agents with different beliefs about asset dynamics trade the same asset. Agents optimize trading rates accounting for their permanent price impact. Nash equilibrium solved via forward-backward SDEs.

```
Optimal trading rate: v*_t = g_{1,t} + g_{2,t} * q_bar*_t
    g_{1,t} = function of beliefs, g_{2,t} = deterministic
    q_bar*_t = mean-field inventory
Individual: v^j_t = v_bar^k_t + (1/2a_k) * h^k_{2,t} * (q^j_t - q_bar^k_t)
```

**Key insight:** Increasing disagreement among agents increases both price volatility and trading volume. Agents deviate from mean-field inversely proportional to their trading cost. Could inform: (1) measuring market disagreement as a vol predictor, (2) optimal execution timing based on crowd inventory estimates.

**References:** Casgrain & Jaimungal 2019 (arXiv:1810.06101), Mathematical Finance

### 9. Testosterone and Asset Trading (Behavioral)
Experimental evidence that testosterone causally increases overbidding for financial assets, generating larger and longer-lasting price bubbles. Traders with elevated testosterone bid higher, leading to slower incorporation of fundamental value.

**Key insight:** Biological/behavioral factors create predictable market inefficiencies. Bubbles driven by overconfidence (testosterone-mediated) eventually revert — supporting mean-reversion strategies during bubble deflation periods. Not directly implementable as a signal, but supports the theoretical case for mean-reversion trading.

**References:** Nadler, Jiao, Johnson, Alexander & Zak 2017 (ResearchGate:316428328)

### 10. Microfish Multi-Agent Swarm
(Reference pending — no paper/link provided yet)

---

## Optimization Methodology

### Grid Search (Current Default)
Exhaustive evaluation of all parameter combinations. Reliable but scales poorly.
- Grid sizes under ~20K combos: fine
- Uses half of available CPU cores via multiprocessing

### Bayesian Search (Optuna TPE) — NEW
Tree-structured Parzen Estimator learns from prior evaluations. Typically finds equivalent results in 10-20% of evaluations.
- Used when grid > 500 combos
- Sequential evaluation (each trial informs the next)
- Falls back to exhaustive for small grids

### Validation Workflow
Optimize on 60 days, validate on 365 days of separate out-of-sample data:
```
python main.py optimize crypto <strategy> 60
python main.py backtest crypto <strategy> 365 --end-days-ago 430
```
The `--end-days-ago 430` ensures no overlap (365 + 60 + 5 buffer = 430).

---

## General Observations

- Mean reversion strategies generalize best on crypto
- Trend-following (momentum, breakout) tends to overfit
- Longer trend filters (240-480 bars) and concentrated picks (top_n=1-2) improve results
- Volatility-scaled sizing (target 15% annual vol) is used across strategies
- Drawdown control (50% reduction at -8%, liquidate at -15%) prevents catastrophic losses
- Crypto uses `(1440 * 365) / rebalance_every` for annualization; stocks use `(390 * 252) / rebalance_every`

### Signal Quality Hierarchy (1-min crypto)
Extensive testing shows clear signal quality ranking:
1. **Price velocity vs speed limit** (velocity_mr): normalized log-return velocity approaching historical max — best OOS Sharpe (3.87)
2. **Kalman spread z-score**: dynamic hedge ratio vs BTC provides statistically grounded entry/exit (3.06)
3. **VWAP dip** (regime_mr, adaptive_mr): most reliable reversion signal — price vs volume-weighted average (2.23)
4. **VWAP dip + dispersion filter** (dispersion_mr): filtering by cross-sectional return dispersion improves VWAP dip quality (2.01 Sharpe)
5. **Bayesian posterior**: combines momentum + volume + RSI via log-odds (2.16)
6. **Return z-scores** (cross-sectional, frac-diff): too noisy on 1-min data, poor OOS
7. **Volume-based filters** (low-vol dip, volume confirmation): not useful discriminators for crypto

### What Doesn't Work on 1-min Crypto
- Fractional differentiation as primary signal (0.62 Sharpe OOS)
- Cross-sectional return z-scores (0.46 Sharpe OOS)
- Low-volume dip filtering (0.04 Sharpe OOS)
- Ensemble of correlated signals (0.13 Sharpe OOS) — VWAP dip + Kalman z-score dual confirmation is too restrictive; both signals are correlated MR signals on the same assets, so requiring both just reduces trade frequency without improving quality
- Take-profit on trend-following strategies (cuts off trends prematurely)
- Momentum confirmation for dip-buying (no improvement over base signal)
- Partial take-profit (sell half at 50% of target) — cuts gains too early, worse Sharpe (1.94 vs 2.23 for regime_mr)
- Dual-velocity filtering (0.59 Sharpe OOS) — requiring slow velocity > 0 on top of fast velocity speed limit doesn't add value, just reduces trade frequency. Same pattern as momentum confirmation on regime_mr.
- UCB bandit signal with risk management (-0.52 Sharpe OOS) — exploration/exploitation scoring is fundamentally weak for crypto selection
