# Alpha redesign research protocol

Frozen on 2026-08-06 before evaluating the volatility-sized or ridge
challengers described below.

This is an exploratory, contamination-aware study. The QLD/SOXL residual
momentum idea and portions of the available history have already been viewed.
No historical result is an untouched confirmation, and no model is allowed to
claim guaranteed alpha.

## Mandate and decision rule

The economic mandate is maximum long-horizon geometric growth for a
hyper-aggressive Roth IRA investor who is willing to hold and keep buying QLD
through bear markets. Accordingly:

- QLD is the permanent core and is never sold merely because a trend or
  volatility signal is bearish.
- SOXL is the only incremental alpha sleeve and is capped at 35% of net asset
  value.
- Maximum advertised daily equity exposure is 2.35x:
  `65% * 2 + 35% * 3`.
- No margin, options, inverse products, leveraged single stocks, synthetic
  pre-inception returns, or replacement tickers are permitted.
- Confirmed broker shares and cash remain the source of truth.

The primary promotion statistic is the net annualized log-growth difference
against both 100% QLD and an exposure-matched static 65% QLD / 35% SOXL
benchmark. Drawdown and tail metrics are reported as consequences, not
silently optimized objectives.

## Transparent econometric candidate

Signals are calculated after completed close `t`; any resulting order fills at
the adjusted open of the next validated XNYS session. A current-day bar is not
treated as final until 15 minutes after the official XNYS close.

### Trend and residual alpha

1. Broad trend is positive only when QQQ close is strictly above its
   200-session simple moving average.
2. Fit `log(SMH return) = intercept + beta * log(QQQ return) + residual`
   over 126 daily returns.
3. The beta-estimation window ends before a separate 63-return scoring window.
4. Residual momentum is the sum of those 63 out-of-estimation residuals.
5. Its z-score is the sum divided by the estimation residual standard
   deviation times `sqrt(63)`.
6. A zero/NaN regressor variance, residual variance, price, or score fails
   closed immediately and cannot authorize or retain SOXL.
7. The alpha sleeve is eligible only when trend is positive and residual
   momentum is strictly positive.

Alpha eligibility is reviewed on one fixed 21-session phase anchored to the
first validated model-history session on 2010-03-11; research and production
use that identical phase, including reviews that occur while trend is bearish.
QQQ trend failure removes SOXL immediately. Re-entry occurs at an alpha review
and requires two distinct eligible closes; duplicate same-date runs never
advance a count.

### Volatility sizing

The live volatility control applies only to incremental SOXL risk.

For QLD and SOXL, a daily-bar log-variance forecast uses a fixed ridge
log-linear HAR-style specification:

- trailing 5-session squared-return variance;
- trailing 21-session squared-return variance;
- trailing 63-session squared-return variance;
- trailing 21-session downside squared-return variance.

The dependent value is the next 21-session annualized squared-return variance.
A live or replayed fit uses the complete, contiguous adjusted history beginning
at SOXL's actual 2010-03-11 inception; research and production may not silently
use different rolling training windows.
A target enters training only after all 21 future returns are known. Features
are standardized using training observations only, the ridge penalty is fixed
at 10, and at least 756 labeled observations are required. Exponentiation uses
a train-only Duan smearing correction; the audit records both the last label
origin and the latest information cutoff. Because daily bars are used, this is
called a HAR-style daily variance forecast, not formal intraday realized
semivariance.

The forecast used for sizing is the maximum of the model forecast and trailing
21- and 63-session annualized volatility. Portfolio correlation is the maximum
finite trailing 21- and 63-session QLD/SOXL correlation, clipped to `[-1, 1]`.

Choose the largest SOXL weight in
`{0%, 5%, 10%, 15%, 20%, 25%, 30%, 35%}` for which forecast portfolio
volatility is no more than 55% annualized. If QLD alone exceeds that budget,
hold 100% QLD; the control never sells the core.

Volatility downshifts are immediate. An increase in SOXL weight requires five
distinct completed sessions with the same raw weight. Fresh volatility sizing
is applied immediately on a valid re-entry. Ordinary position drift uses a
5 percentage-point trigger and a 2.5 percentage-point destination.

## Frozen candidate and ablation registry

The following paths are the complete confirmatory family for this round:

1. `qld_buy_hold`: 100% QLD.
2. `static_65_35`: initial 65% QLD / 35% SOXL, held without tactical timing.
3. `residual_35`: transparent trend/residual eligibility with a fixed active
   35% SOXL sleeve and no volatility sizing.
4. `residual_vol55`: the full transparent candidate above.
5. `trend_vol55`: volatility-sized sleeve using only the QQQ trend gate.
6. `residual_vol55_no_trend`: volatility-sized sleeve using only positive
   residual momentum.
7. `ridge_shadow`: a ridge confirmation layer applied to `residual_vol55`.

The first six paths identify whether any apparent gain comes from exposure,
trend, residual semiconductor strength, or volatility sizing. `ridge_shadow`
is not eligible to control production money in this round.

## Ridge shadow challenger

At each 21-session review, ridge regression predicts the next 21-session
open-to-open log-return difference between a 65% QLD / 35% SOXL sleeve and
100% QLD.

Frozen features:

1. `log(QQQ / SMA200)`;
2. SMH-versus-QQQ residual-momentum z-score;
3. `log(QLD forecast volatility / QLD 63-session volatility)`;
4. QQQ drawdown from its trailing 252-session high;
5. trailing 21-session QQQ downside-to-total variance ratio.

Training is expanding and chronological. A row is usable only after its full
21-session open-to-open label is known. Scaling uses training data only.
The ridge penalty is fixed at 10 and at least 756 labeled daily observations
are required. The shadow permits the transparent sleeve only when its predicted
relative log return is positive. It may suppress leverage but cannot exceed
the transparent candidate's weight.

The implementation must record the training cutoff, sample count, feature
means and standard deviations, coefficients, current feature row, prediction,
data fingerprint, and challenger fingerprint.

## Data, execution, and costs

- Actual adjusted ETF history only, beginning no earlier than each required
  product's inception.
- Complete XNYS sessions; no forward fill, backfill, interpolation, row
  deletion, or synthetic substitution.
- Completed-close signal, next-session adjusted-open fill.
- Fractional shares and cash accounting.
- Trading costs charged on gross buy plus sell notional.
- Cost cases: 0, 10, 25, and 50 basis points.
- Initial deployment affects wealth but is excluded from ongoing turnover.
- Every result records the complete-data SHA-256 and software/configuration
  fingerprints.

Primary comparisons use a common scoring interval. Per-strategy inception
results may be shown only as labeled sensitivity analysis. The existing
Original strategy is compared over its own actual common universe interval and
over any later interval shared with the redesigned family.

## Required diagnostics

- CAGR and annualized log growth;
- maximum drawdown, recovery time, and time underwater;
- annualized volatility, downside deviation, worst month, and 95% expected
  shortfall;
- rolling three- and five-year win rates;
- gross and one-way turnover, event counts, and cost sensitivity;
- trade size versus trailing dollar volume;
- chronological slices and expanding walk-forward results;
- stationary or moving-block bootstrap intervals;
- White Reality Check or Hansen SPA across the frozen family;
- deflated-Sharpe and CSCV/PBO diagnostics with the honest trial count;
- rolling return-based style attribution to QLD, SOXL, and SPY, including
  intercept, exposures, residual volatility, and R-squared;
- return contribution by QLD core, SOXL sleeve, timing decisions, and costs.

Promotion requires positive 10bp net growth versus both QLD and static 65/35,
no reversal at 25bp, acceptable liquidity, no single chronological slice
carrying the conclusion, and a coherent mechanism. Historical evidence remains
provisional until at least 252 future sessions and three structural decisions
have been observed in a chained shadow ledger. Its count, final date, and tail
hash must be anchored in portfolio state; loss, truncation, or a mismatched
tail fails closed.

At the user's explicit direction, a candidate may run as an experimental live
allocator before research promotion. That is an operational override, not a
scientific promotion: the production report and notification must label it
experimental, preserve `promotion_complete=false`, cap SOXL at 35%, retain
confirmed holdings as truth, and continue the future shadow ledger. Every
unseen completed session must be replayed after an outage so the signal state
does not depend on workflow uptime.
