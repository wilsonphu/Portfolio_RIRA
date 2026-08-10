# Frozen TCN and graph-risk shadow protocol

Status: preregistered shadow research; no live allocation authority.

Frozen on: 2026-08-09, before any TCN or graph-hybrid result was observed.

## Objective and live boundary

The live mandate remains the permanent-QLD `residual_vol55` allocator. QLD is
the permanent core and SOXL is the only sleeve the shadow system may reduce.
The shadow system cannot increase SOXL above the live strategic weight, add a
short position, alter live holdings, or send a trade recommendation.

A separate cash-gated benchmark holds QLD only when QQQ closes strictly above
its 200-session simple moving average and otherwise holds zero-return cash.
This benchmark tests the distinct drawdown-constrained mandate; it is not a
reinterpretation of the permanent-core live strategy.

The confirmatory objective is annualized net log growth. CAGR, maximum
drawdown, Sharpe ratio, Calmar ratio, turnover, Brier score, log loss, expected
calibration error, and chronological stability are diagnostics. No historical
maximum drawdown is treated as a guaranteed future cap.

## Information set

All decisions use completed, adjusted daily bars. A close at session `t` may
first affect a fill at the next session's open. Missing, stale, nonpositive, or
nonfinite required data gives the shadow layer zero SOXL authority.

The fixed market universe is:

- strategic and alpha assets: QQQ, SMH, QLD, SOXL;
- independent context assets: SPY, HYG, TLT, GLD, UUP, and ^VIX.

The TCN receives 64 completed sessions and exactly these 12 causal channels:

1. QQQ one-session log return;
2. SMH one-session log return;
3. SPY one-session log return;
4. HYG one-session log return;
5. TLT one-session log return;
6. GLD one-session log return;
7. UUP one-session log return;
8. ^VIX one-session log change;
9. QQQ log distance from its 200-session SMA;
10. the frozen 126-estimation/63-scoring SMH-on-QQQ residual z-score;
11. log ratio of QQQ 21-session to 63-session realized volatility;
12. log ratio of SMH 21-session to 63-session realized volatility.

Each feature is standardized using the model-training partition only and
clipped to plus or minus five training standard deviations. Validation,
calibration, and future observations never enter feature normalization.

## Prediction target

The model does not predict a one-day up/down label. For a signal at the `t`
close, it predicts whether SOXL beats QLD in log return from the `t+1` open to
the `t+22` open, after a fixed 20-basis-point full-notional round-trip hurdle:

```text
y_t = 1[
  log(SOXL_open[t+22] / SOXL_open[t+1])
  - log(QLD_open[t+22] / QLD_open[t+1])
  - 0.002 > 0
]
```

The 21-session horizon matches the frozen alpha-review cadence. Every training
row must have a fully observed label endpoint strictly before the prediction
being produced.

## Small causal TCN

The model is fixed before testing:

- four causal residual Conv1d blocks;
- eight hidden channels;
- kernel size three;
- dilations 1, 2, 4, and 8;
- two convolutions per block;
- layer normalization, GELU activation, and 10% dropout;
- global average pooling and one binary logit;
- AdamW, learning rate 0.001, weight decay 0.001;
- batch size 64, at most 100 epochs, early-stopping patience 10;
- seeds 17, 29, and 43, averaged as an ensemble rather than selected;
- unweighted binary cross-entropy.

The effective receptive field is 61 sessions. Model capacity, parameter count,
training dates, early-stopping epochs, and seed-level results must be reported.

## Walk-forward fitting and calibration

Models are refit every 21 sessions using expanding history. A fit requires:

- at least 756 model-training labels;
- a 252-label validation partition;
- a separate 252-label probability-calibration partition;
- a 21-session purge between training/validation and validation/calibration;
- labels ending before the relevant later partition begins.

The validation partition chooses the early-stopping epoch. The three seed
logits are averaged. A two-parameter Platt map is then fitted only on the
calibration partition. No test or future observation is used for training,
early stopping, calibration, threshold selection, or normalization.

The TCN is unqualified for the next prediction block unless its calibration
partition has positive Brier skill versus the partition's constant base rate.
An unqualified or failed model has confidence zero.

## Deadbanded confidence map

For qualified calibrated probability `p_t`, the frozen confidence multiplier
is:

```text
c_t = clip((p_t - 0.55) / 0.10, 0, 1)
```

Thus `p_t <= 0.55` removes the shadow SOXL sleeve, `p_t >= 0.65` leaves the
live sleeve unchanged, and intermediate probabilities scale it linearly. The
shadow never shorts and never creates SOXL exposure that the live strategy did
not independently authorize.

## Graph shock haircut

The graph is a risk control, not a return signal. Its nodes are the unlevered
or independent context series QQQ, SMH, SPY, HYG, TLT, GLD, UUP, and ^VIX;
QLD and SOXL are excluded to avoid mechanically duplicated leverage nodes.

At every completed close:

1. calculate 63-session log-return correlations;
2. shrink the correlation matrix 25% toward the identity;
3. invert it and convert finite off-diagonal elements to absolute partial
   correlations;
4. row-normalize the nonnegative adjacency matrix;
5. define each node shock as `max(0, abs(r_t) / sigma_63 - 1)`;
6. diffuse shocks as `u + 0.5 A u + 0.25 A^2 u`;
7. take the maximum propagated QQQ/SMH shock as the portfolio shock score.

The score is ranked against the preceding 756 valid scores, excluding the
current session. The frozen haircut is one through the 90th percentile and
then declines linearly to zero at the 100th percentile:

```text
h_t = 1                              if percentile_t <= 0.90
h_t = (1 - percentile_t) / 0.10      otherwise
```

Invalid or insufficient graph data gives the shadow layer zero SOXL authority.

## Hybrid allocation

Let `w_live,t` be the frozen live strategic SOXL weight after its OLS residual,
QQQ trend, volatility budget, and state-transition rules. The shadow weights
are:

```text
w_graph,t = w_live,t * h_t
w_tcn,t   = w_live,t * c_t
w_hybrid,t = w_live,t * c_t * h_t
```

QLD is `1 - w_SOXL` for all permanent-core paths. The cash-gated benchmark is
100% QLD when QQQ is above its SMA200 and 100% cash otherwise.

All paths use next-open fills, fractional shares, actual daily holdings,
ordinary five-percentage-point drift control, exact risk reductions, and
primary 10-basis-point plus sensitivity 25-basis-point gross trading costs.

## Comparisons and promotion boundary

The declared post-selection shadow family is exactly:

1. frozen live `residual_vol55`;
2. permanent-core graph haircut;
3. permanent-core TCN confidence modifier;
4. permanent-core TCN-plus-graph hybrid;
5. QQQ-SMA200 cash-gated QLD benchmark.

The TCN and graph paths are new post-selection trials and must never be folded
into the earlier seven-trial confirmatory family. Historical results are
exploratory even with causal walk-forward construction.

No component can receive live authority from this historical study. Continued
shadow candidacy requires all of the following at 10 and 25 basis points:

- positive annualized log-growth difference versus frozen live in at least
  three of four chronological slices;
- positive median moving-block-bootstrap log-growth difference;
- no worse maximum drawdown than frozen live;
- no material calibration failure;
- stable direction across the three fixed seeds;
- an economically nontrivial improvement after honest expanded trial counts.

Live promotion additionally requires at least 252 prospectively recorded
sessions and three future live SOXL structural decisions. A prospective result
must improve net log growth without worsening drawdown; otherwise the live
strategy remains unchanged.
