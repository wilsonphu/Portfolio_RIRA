# Leveraged core and satellite universe research protocol

Frozen on 2026-08-09 before calculating any candidate-path result.

This is a new, post-selection exploratory family. The existing
`residual_vol55` strategy remains the live benchmark and no historical result
can change production authority. The purpose is to test whether a broader core
or a less concentrated satellite improves long-run geometric efficiency.

## Mandate and selection rule

The investor is a long-horizon, high-risk-tolerance Roth IRA investor. The
primary objective is maximum net annualized log growth subject to a maximum
drawdown no worse than 70%. Sharpe, Calmar, downside deviation, recovery time,
and worst rolling returns are secondary diagnostics rather than substitutes
for compounding.

A challenger can displace the live benchmark historically only if, at both 10
and 25 basis points:

- its annualized log-growth difference is positive;
- its maximum drawdown is no worse than 70%;
- at least three of four chronological log-growth differences are positive;
- its moving-block-bootstrap median log-growth difference is positive;
- the conclusion does not reverse at 50 basis points; and
- no single crisis or final subperiod carries the result.

A lower-growth path may be labeled a risk-efficiency alternative, but not the
growth winner, only if it improves both Sharpe and Calmar, improves maximum
drawdown by at least five percentage points, and sacrifices no more than three
percentage points of CAGR at 25 basis points.

All conclusions remain provisional because this history has been viewed.

## Frozen universe and data

Signal series: QQQ, SPY, SMH, and XLK.

Traded funds: QLD, SSO, UPRO, SOXL, USD, TECL, GLD, and IEF.

Use actual adjusted Open, Close, and Volume observations only. The frozen
sample ends at the completed 2026-08-07 XNYS session. No synthetic extension,
substitution, filling, interpolation, or silent session deletion is allowed.
Every path is compared over one common executable interval after all required
signals and volatility estimates are available.

Signals use completed close `t`; orders fill at adjusted open `t+1` with
fractional shares and actual cash. Costs are charged on gross buys plus sells
at 10, 25, and 50 basis points. Initial deployment affects wealth but is
excluded from ongoing turnover.

## Shared residual signal

The semiconductor paths retain the frozen live signal:

1. QQQ must be strictly above its 200-session SMA.
2. Fit SMH daily log returns on QQQ daily log returns over 126 returns.
3. Score residual momentum on a separate subsequent 63-return window.
4. Positive, finite residual momentum permits the semiconductor satellite.
5. Invalid inputs fail closed.

The broad-technology path uses the same separated-window calculation with XLK
as the dependent series and SPY as the regressor. Its trend gate is SPY above
its 200-session SMA.

The QLD/SSO router uses QQQ-on-SPY residual momentum with the same 126/63
separation. At each fixed 21-session review it holds QLD when the residual is
positive and SSO otherwise. It never routes to cash.

Satellite eligibility is reviewed on the existing fixed 21-session phase.
Trend failure removes a satellite immediately; re-entry requires two distinct
eligible reviews. Volatility downshifts are immediate and upshifts require five
distinct completed sessions. Duplicate dates never advance state.

## Shared volatility budget

Each traded asset uses the existing fixed daily-bar HAR-style log-variance
ridge forecast, with the same features, 21-session label horizon, ridge penalty
10, 756-observation minimum, training-only scaling, and Duan smearing. Its
sizing volatility is the maximum of the model forecast and trailing 21- and
63-session annualized volatility.

For every candidate weight vector, calculate forecast portfolio volatility
with both the trailing 21- and 63-session correlation matrices and use the
larger value. Invalid covariance inputs fail closed. The satellite weight is
the largest value in `{0%, 5%, 10%, 15%, 20%, 25%, 30%, 35%}` whose portfolio
volatility is no more than 55% annualized. If the permanent core alone exceeds
55%, retain the core and authorize no satellite.

Satellite weight replaces the base basket pro rata. Ordinary drift uses a
five-percentage-point individual-position or aggregate risky-allocation
trigger and trades to a 2.5-percentage-point inner destination. Strategic risk
reductions bypass drift bands.

## Frozen candidate registry

The complete family is exactly:

1. `live_qld_soxl`: 100% QLD base with the frozen SOXL satellite.
2. `sso_soxl`: 100% SSO base with the same SOXL signal and sizing.
3. `qld_sso_equal_soxl`: 50% QLD / 50% SSO base with SOXL.
4. `qld_sso_router_soxl`: QLD-or-SSO residual router with SOXL.
5. `qld80_gld20_soxl`: 80% QLD / 20% GLD base with SOXL.
6. `upro60_gld20_ief20_soxl`: 60% UPRO / 20% GLD / 20% IEF base with SOXL.
7. `qld_usd`: 100% QLD base with USD as the semiconductor satellite.
8. `sso_tecl`: 100% SSO base with TECL as the broad-technology satellite.

There is no parameter search, ex-post combination of winners, or uncounted
candidate. Any later combination is a new trial and cannot reuse this result
as confirmation.

## Required diagnostics

- CAGR and annualized log growth at every cost;
- maximum drawdown, Calmar, Sharpe, downside deviation, recovery time, and
  worst rolling three- and five-year returns;
- average holdings, advertised daily exposure, turnover, costs, and rebalance
  count;
- four chronological comparisons versus live;
- 10,000 moving-block resamples with a 21-session block;
- familywise data-snooping adjustment using the honest eight-path count;
- Pareto frontier in log growth, drawdown, Sharpe, and Calmar;
- contribution by each traded sleeve; and
- exact data, strategy, and software fingerprints.

No historical winner receives live authority. Promotion still requires 252
prospectively recorded sessions and at least three future structural satellite
decisions with positive net log-growth improvement and no violation of the
drawdown boundary.
