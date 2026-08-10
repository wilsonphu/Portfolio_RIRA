# Alpha redesign findings — 2026-08-06

## Bottom line

On the exact shared 2016-08-12 through 2026-08-06 interval, the redesigned
volatility-sized strategy modeled a 41.21% net CAGR at 10bp versus 33.64% for
the causal Original strategy. Its maximum drawdown was worse
(-64.47% versus -39.09%), but it traded far less: 2.47x versus 32.36x annual
gross turnover.

That is credible historical evidence that the redesign beat the Original
under identical data, dates, next-open execution, and costs. It is not proof
that the redesign will win in the future. The paired annualized log-growth
edge was +5.51 percentage points, but its 95% moving-block interval was
-10.24 to +20.85 points, the estimated probability of a nonpositive edge was
25.98%, and only two of four chronological slices were positive.

The harder benchmark remains static 65% QLD / 35% SOXL. The live candidate
beat it by only 0.05 point of annualized log growth at 10bp, with a very wide
interval, only one positive slice out of four, and a reversal at 25bp. The
frozen seven-model family also failed its multiplicity-adjusted tests.
Consequently, `promotion_complete=false` remains the scientifically correct
conclusion.

Production implements the user-directed `residual_vol55` design as an
explicitly labeled experimental allocator. It was selected for its frozen
economic rationale and risk controls, not because it happened to be the
highest line in this backtest.

## Frozen data and provenance

- Union snapshot SHA-256:
  `69ed7ed215e73a4442f38f3fc3068c703b826f529f90b56a6c607c5b143617ab`
- Redesigned five-ticker view SHA-256:
  `548a5a3bb0c192930d8cbf9968ca4b6e5a2eaff424c8d00814f55b6f380a8f6b`
- Current-family fingerprint:
  `a6c368a679a05527a181870b5e5520d9a79b4a8ca899a43f7a13c408f2188dcf`
- Current software fingerprint:
  `367feea0f966b4f05bff8cf96d41f02d756eb921e3d7b11eeb0035f6272f327c`
- Original strategy fingerprint:
  `7cd1ac4e2eb950b1836c923eb218df9e1500319d4700a9a6a5caaf573366c872`
- Original source commit:
  `fa9b3245acfb3d6d17f4a1bc6c17cd96e27137e3`
- Redesigned model inception: 2010-03-11.
- Original common-universe inception: 2015-10-12.
- Paired execution interval: 2016-08-12 through 2026-08-06.
- Paired observations: 2,509 completed sessions.
- Starting value: $10,000; fractional shares; zero cash return.
- Primary costs: 10bp on gross security buys plus sells.
- Inference: 10,000 paired moving-block resamples.

The Original is an external historical comparator. It is deliberately not an
eighth trial in the redesigned family's White Reality Check, deflated-Sharpe,
or CSCV/PBO counts.

## Primary shared-interval results at 10bp

| Strategy | CAGR | Max drawdown | Annual volatility | Annual gross turnover | Ending $10,000 |
|---|---:|---:|---:|---:|---:|
| Original ROTH IRA Barbell | 33.64% | -39.09% | 37.09% | 32.36x | $179,376 |
| QLD buy and hold | 33.26% | -63.68% | 45.06% | 0.00x | $174,386 |
| Static 65% QLD / 35% SOXL | 41.14% | -80.51% | 71.19% | 0.00x | $308,909 |
| Residual gate, fixed 35% SOXL | 42.33% | -66.53% | 50.81% | 2.31x | $335,883 |
| Residual gate plus 55% vol cap | 41.21% | -64.47% | 47.97% | 2.47x | $310,453 |
| Trend gate plus 55% vol cap | 41.59% | -64.48% | 50.87% | 2.63x | $319,048 |
| Residual/vol without trend | 41.32% | -64.60% | 48.13% | 2.63x | $312,853 |
| Ridge-confirmed shadow | 38.58% | -64.47% | 47.13% | 2.28x | $257,556 |

The Original's own executable history begins on 2015-10-13. Over that longer
2,719-session path it modeled 31.60% CAGR, -39.09% maximum drawdown, and 33.20x
annual gross turnover at 10bp. Its previously reported headline cannot be
compared directly unless cost, ending date, and same-day versus next-open
execution are identical.

## Evidence strength

For `residual_vol55` at 10bp:

- versus Original: +5.51 points annualized log growth;
- paired 95% interval versus Original: -10.24 to +20.85 points;
- probability of nonpositive edge versus Original: 25.98%;
- positive chronological slices versus Original: 2 of 4;
- versus QLD: +5.79 points annualized log growth;
- paired 95% interval versus QLD: -0.02 to +11.28 points;
- probability of nonpositive edge versus QLD: 2.58%;
- versus static 65/35: +0.05 point annualized log growth;
- paired 95% interval versus static: -17.28 to +17.51 points;
- probability of nonpositive edge versus static: 50.94%;
- positive chronological slices versus static: 1 of 4.

At 25bp, `residual_vol55` retained a +10.00-point annualized log-growth edge
over the high-turnover Original, but fell behind static 65/35. This distinction
matters: lower turnover is a robust advantage over the Original, while the
tactical alpha claim over an exposure-matched buy-and-hold mix remains weak.

Across the complete frozen current family:

- White Reality Check p-value versus QLD: 0.2812;
- White Reality Check p-value versus static 65/35: 0.5687;
- CSCV/PBO selection-instability estimate: 1.00;
- honest current-family trial count: 7.

The raw winner was `residual_35`, not the live candidate. It modeled 42.33%
CAGR, but its edge over static was only 0.84 point of annualized log growth and
was positive in two of four slices. Chasing that line after viewing the result
would add another uncounted model-selection decision.

## Earlier-inception sensitivity

The transparent fixed residual candidate can begin after its 126+63-return
warm-up, before the HAR/ridge family has 756 complete labels. The separately
labeled 2010-10-05 sensitivity produced:

| Strategy | CAGR | Max drawdown |
|---|---:|---:|
| QLD | 32.85% | -63.68% |
| Static 65/35 | 37.81% | -80.58% |
| Residual 35% | 38.36% | -66.53% |

Residual timing added 4.06 points of annualized log growth over QLD
(95% interval -1.55 to +9.82; 3 of 4 positive slices), but only 0.40 point
over static (95% interval -10.37 to +11.12; 1 of 4 positive slices).

## Econometric and ML conclusions

- Separated-window SMH-on-QQQ residual momentum is causal, transparent, and
  economically interpretable. Its single-sector timing alpha is not proven.
- The HAR-style ridge variance forecast is useful only as a bounded leverage
  control. It is combined with trailing 21- and 63-session risk rather than
  trusted as an oracle.
- The directional ridge veto lowered shared-sample CAGR to 38.58% and has no
  live authority.
- The latest 756-session constrained style attribution for `residual_vol55`
  was approximately 88.4% QLD, 7.9% SOXL, and 3.6% SPY, with R-squared 0.954.
  These collinear return-based exposures are descriptive, not causal proof of
  a 5.38% annual intercept.
- Deep learning, boosted trees, HMMs, and reinforcement learning were rejected
  because one short ETF history offers too little independent information for
  their model capacity.
- VIX/IV rank was retained as a risk concept, not an equity-direction signal.
- Options writing was excluded because it caps the upside this mandate seeks
  and adds assignment, liquidity, and broker-automation failure modes.

## Why the live design remains experimental

The production candidate has a permanent QLD core, caps SOXL at 35%, limits
advertised daily exposure to 2.35x, removes the satellite immediately on QQQ
trend failure or invalid model data, and replays every missed market session.
Confirmed shares and cash—not a recalculated target—remain the source of
truth. Ordinary HOLD runs do not email.

Historical promotion is still blocked because:

- the hypothesis and sample are contaminated by prior viewing;
- the edge over static leverage is economically tiny and statistically weak;
- the 25bp static comparison reverses;
- chronological stability is poor;
- the seven-path selection diagnostic is unstable;
- 252 future sessions and three future structural decisions have not yet been
  recorded in the state-anchored shadow ledger.

At the completed 2026-08-06 close, QQQ trend was positive but SMH residual
momentum was negative. The strategic destination was **100% QLD / 0% SOXL**.

## Preregistered TCN/graph shadow result (2026-08-09)

This was a new post-selection shadow family, not an extension of the earlier
seven-trial confirmatory family. Its protocol and thresholds were committed
before the first completed model result. The exact extended-universe snapshot
fingerprint was
`2da0387bf9d1c50a380df24387931e4185c77ccd01099fdcb793bc83e7250679`.
The executable comparison ran from 2016-07-28 through 2026-08-07 for 2,521
sessions with next-open fractional fills.

### Net results

| Path | CAGR at 10bp | CAGR at 25bp | Max drawdown at 25bp | Log-growth difference vs live at 25bp |
|---|---:|---:|---:|---:|
| Frozen live residual/vol55 | 41.84% | 41.29% | -64.54% | baseline |
| Graph shock haircut | 40.88% | 39.75% | -64.47% | -1.10 points/year |
| TCN confidence modifier | 34.15% | 33.96% | -63.68% | -5.33 points/year |
| TCN plus graph | 33.91% | 33.70% | -63.68% | -5.52 points/year |
| SMA200 cash-gated QLD | 29.18% | 28.38% | -43.40% | -9.58 points/year |

The cash gate reduced drawdown substantially but imposed too large a geometric
growth penalty for the stated lifelong CAGR-first, high-drawdown-tolerance
mandate. The permanent QLD core therefore remains the selected live structure.

### Statistical decision

- The TCN's aggregate out-of-sample Brier score was 0.2894 versus a 0.2487
  base-rate score, for **-16.36% Brier skill** and 0.7844 log loss.
- The TCN-only and hybrid paths lost annualized log growth in all four
  chronological slices at both cost assumptions.
- Their 10bp moving-block median differences were -5.45 and -5.64 points per
  year; their 95% upper bounds remained negative.
- Graph-only was positive in one of four slices and had a negative bootstrap
  median at both costs. At 25bp, its probability of a nonpositive edge was
  97.47%.
- Cash-gated QLD was positive in one of four slices and also had a negative
  bootstrap median at both costs.

Every challenger fails the frozen continued-shadow gate. Seed-stability and
expanded-trial promotion tests are unnecessary after these earlier decisive
failures. No result receives prospective or live authority, no production
weight changes, and no additional notification is justified.

## Evidence base

- ProShares states that QLD targets 2x the **daily** Nasdaq-100 return and that
  longer-horizon returns can differ materially:
  https://www.proshares.com/globalassets/proshares/prospectuses/qld_summary_prospectus.pdf
- Direxion states that SOXL targets +300% for a day, not three times cumulative
  return beyond a day:
  https://www.direxion.com/product/daily-semiconductor-bull-bear-3x-etfs
- HAR volatility foundation: https://doi.org/10.1093/jjfinec/nbp001
- Machine-learning asset-pricing capacity and overfit context:
  https://academic.oup.com/rfs/article/33/5/2223/5758276
- Out-of-sample caution on volatility-managed portfolios:
  https://doi.org/10.1016/j.jfineco.2020.04.015
- White's data-snooping Reality Check: https://doi.org/10.1111/1468-0262.00152
- Residual-momentum evidence:
  https://repub.eur.nl/pub/22252
- Leveraged-ETF path dependence:
  https://epubs.siam.org/doi/pdf/10.1137/090760805

## Leveraged core and satellite universe result (2026-08-09)

### Strategy conclusion

The existing permanent-QLD core with the stateful SOXL overlay remains the
best maximum-compounding strategy in the frozen eight-path family. At the
primary 25bp cost assumption it produced 39.09% CAGR, versus 37.42% for the
closest growth challenger. None of the seven challengers had a positive
annualized log-growth difference from live.

The 80% QLD / 20% GLD base with the same SOXL overlay is the only candidate
that met the separately frozen risk-efficiency rule. It sacrificed 1.99
percentage points of CAGR while improving maximum drawdown by 7.63 points,
Sharpe from 0.960 to 1.016, and Calmar from 0.606 to 0.652. It is therefore a
credible balanced alternative for an investor who values a shallower loss and
better return per unit of risk, but it is not evidence of higher expected
growth.

### Frozen sample and provenance

- Common actual-fund-history start: 2010-03-11.
- Common executable interval: 2013-07-12 through 2026-08-07.
- Executable sessions: 3,288.
- Data snapshot SHA-256:
  `59c84803bf08c08db18c4683972844fa70a79b143cc361695818e347f866820d`.
- Protocol SHA-256:
  `610e1b7725bf6157a6294399fb75df373331ff8f31a7ae801f70b391618fa4cb`.
- Strategy fingerprint:
  `264f356035800fbb7531ac186dc8711b4971a67ed7965e49ac62e4237ed0fd8b`.
- Software fingerprint:
  `3a697fd54e5b1f462181343dbe401b3324389bf22695f8e553ee4766861f68ee`.
- Starting value: $10,000; fractional shares; zero cash return.
- Primary costs: 25bp on gross buys plus sells.
- Inference: 10,000 paired 21-session moving-block resamples.

Only actual adjusted observations were used. No synthetic pre-inception data,
substitution, interpolation, or silent row deletion was permitted. Six actual
zero-volume observations were retained rather than rewritten as missing.

### Net results at 25bp

| Strategy | CAGR | Max drawdown | Sharpe | Calmar | Gross turnover/year |
|---|---:|---:|---:|---:|---:|
| Live QLD + SOXL overlay | 39.09% | -64.54% | 0.960 | 0.606 | 2.57x |
| QLD + USD overlay | 37.42% | -64.11% | 0.948 | 0.584 | 2.60x |
| 80% QLD / 20% GLD + SOXL | 37.10% | -56.91% | 1.016 | 0.652 | 2.72x |
| 50% QLD / 50% SSO + SOXL | 35.25% | -55.78% | 0.941 | 0.632 | 2.62x |
| QLD-or-SSO router + SOXL | 34.51% | -53.57% | 0.920 | 0.644 | 7.92x |
| 60% UPRO / 20% GLD / 20% IEF + SOXL | 29.50% | -54.98% | 0.905 | 0.537 | 2.92x |
| SSO + SOXL overlay | 29.45% | -59.34% | 0.865 | 0.496 | 2.57x |
| SSO + TECL overlay | 23.14% | -63.11% | 0.749 | 0.367 | 2.66x |

The live path's largest drawdown ran from its 2021-11-19 peak to the
2022-12-28 trough and recovered on 2024-02-22. The QLD/GLD alternative's
largest drawdown ran from 2021-12-27 to 2022-11-03 and recovered on
2024-02-08. Its maximum underwater spell was 531 sessions versus 564 for
live. This is a useful but modest recovery improvement, not elimination of
leveraged-equity risk.

### Robustness and rejection decisions

For the QLD/GLD alternative, annualized log growth trailed live by 1.42, 1.44,
and 1.48 percentage points at 10bp, 25bp, and 50bp. At 25bp only two of four
chronological slices were positive. Its paired moving-block median difference
was -1.44 points, with a 95% interval from -5.35 to +2.84 points and a 75.66%
estimated probability of a nonpositive edge. That is why it is labeled a
risk-efficiency alternative rather than a growth challenger.

The familywise reality check at 25bp found that even the best challenger had
an observed annualized log-growth difference of -1.21 points. The adjusted
p-value was 0.8754 across the honest eight-path trial count. Every challenger
failed the historical growth gate and every path retains `live_authority=false`.

The SSO variants confirm that lower daily leverage is not automatically a
better long-run trade. The equal QLD/SSO base improved drawdown by 8.76 points
but lost 3.84 points of CAGR and did not improve Sharpe. Pure SSO lost 9.64
points of CAGR. The residual router reduced drawdown but lost 4.58 points of
CAGR and raised turnover to 7.92x. USD was the closest growth substitute for
SOXL, but its drawdown improvement was negligible and its edge was negative.

### Allocation implication

At the completed 2026-08-07 close, the selected live strategy remained
**100% QLD / 0% SOXL** because the semiconductor residual signal was negative.
The QLD/GLD alternative's same-date state was **76% QLD / 19% GLD / 5% SOXL**.
Moving to the latter is not an automatic model upgrade; it is a different
investor-objective choice between maximum historical compounding and better
historical risk efficiency.

No production allocation changes follow from this post-selection study. A new
candidate still needs 252 prospectively recorded sessions and three future
structural satellite decisions before it can be considered for live authority.

### Research rationale

- Residual momentum evidence motivates retaining the transparent existing
  satellite signal rather than adding another high-capacity predictor:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2319861
- Volatility-managed portfolio evidence motivates bounded inverse-risk sizing,
  while the live design conservatively combines the forecast with trailing
  risk rather than treating forecast variance as known:
  https://www.nber.org/papers/w22208
- Broad return-prediction evidence warns that apparently successful predictive
  variables frequently fail out of sample, supporting the frozen small family
  and prospective promotion requirement:
  https://www.nber.org/papers/w10483
- The low-beta literature motivates testing lower-leverage broad-market cores,
  but does not overcome their observed compounding shortfall in this sample:
  https://www.nber.org/papers/w16601
- ProShares states that SSO and UPRO seek 2x and 3x daily S&P 500 returns,
  respectively, so multi-session results remain path dependent:
  https://www.proshares.com/our-etfs/leveraged-and-inverse/sso
  and https://www.proshares.com/our-etfs/leveraged-and-inverse/upro
- ProShares states that USD seeks 2x the daily semiconductor index return, and
  Direxion states that TECL seeks 3x the daily technology index return:
  https://www.proshares.com/our-etfs/leveraged-and-inverse/usd
  and https://www.direxion.com/product/daily-technology-bull-bear-3x-etfs
