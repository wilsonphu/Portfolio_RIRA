# Quantitative research workflow

`alpha_paired_research.py` is the primary research entry point. It supplies
both the redesigned seven-path family and the frozen Original strategy with
views of one exact eight-ticker Open/Close/Volume snapshot. It cannot access
production state, email, or broker holdings.

The decision rules, candidate registry, costs, and promotion gates are frozen
in `ALPHA_RESEARCH_PROTOCOL.md`. Historical results remain exploratory because
parts of the sample and the residual-momentum idea were viewed during design.

## Reproduce the 2026-08-06 paired study

Download the union once and atomically preserve every IEEE-754 value:

```powershell
python alpha_paired_research.py `
  --start 2010-01-01 `
  --end 2026-08-07 `
  --save-snapshot research_outputs/alpha_paired_union_snapshot.csv `
  --bootstrap-samples 100 `
  --output research_outputs/alpha_paired_smoke.json
```

Replay that immutable snapshot with the full inference budget:

```powershell
python alpha_paired_research.py `
  --start 2010-01-01 `
  --end 2026-08-07 `
  --snapshot research_outputs/alpha_paired_union_snapshot.csv `
  --bootstrap-samples 10000 `
  --output research_outputs/alpha_paired_full_results.json `
  --ledger-dir research_outputs/alpha_paired_ledgers
```

Research outputs are deliberately ignored by Git. The committed findings
record the exact snapshot, view, family, software, strategy, and schedule
fingerprints needed to identify the run.

`alpha_research.py` remains a standalone five-ticker runner for the redesigned
family. `legacy_original_research.py` is the isolated, provenance-locked
Original comparator. Neither is imported by the production engine.

## Integrity rules

- One union of QQQ, SMH, QLD, SOXL, SPY, TECL, SPMO, and GLD.
- Actual adjusted fund history only; no synthetic pre-inception series.
- No fill, interpolation, substitution, or silent row deletion.
- The redesigned model history must begin at SOXL's actual 2010-03-11
  inception; a shifted start is rejected without changing family identity.
- A current-day bar is not final until 15 minutes after the XNYS close.
- Signal at completed close `t`; fractional execution at adjusted open `t+1`.
- Actual shares and cash drive every later drift decision.
- Costs are charged on gross security buys plus sells at 0, 10, 25, and 50bp.
- Initial deployment affects wealth but is excluded from ongoing turnover.
- HAR and ridge scalers/fits use training observations only.
- A 21-session label is unavailable until every future return is realized.
- Snapshot output uses `%.17g`; input uses round-trip float parsing.
- Current and Original views are checked for bit-identical overlapping values.
- Every frozen current candidate stays in the seven-trial multiplicity count.
- Original is an external benchmark, not an eighth selected trial.

## What is and is not machine learning

The production risk forecast is a small, fixed ridge log-variance model using
5-, 21-, and 63-session daily squared-return variance plus 21-session downside
variance. Its forecast is conservatively combined with trailing risk and only
limits the incremental SOXL sleeve.

The directional ridge model is a shadow challenger. It predicts 21-session
relative return using five frozen features and may only veto the transparent
overlay in research. It has no live authority. Deep networks, boosted trees,
HMMs, and reinforcement learning were excluded because a single short ETF
history cannot support their capacity without severe selection risk.

## Statistical outputs

The paired JSON report includes:

- net CAGR, annualized log growth, drawdown, volatility, tail loss, recovery,
  underwater time, turnover, costs, and liquidity;
- Original results over its own executable history and the exact shared
  current-family interval;
- rolling three- and five-year comparisons and four chronological slices;
- paired moving-block intervals at 10bp and 25bp;
- White Reality Checks versus QLD and static 65/35;
- deflated-Sharpe and CSCV/PBO diagnostics using the honest seven-trial count;
- rolling constrained return-based style attribution;
- return contribution, complete ridge fit history, and all provenance hashes.

These diagnostics measure uncertainty. They do not transform a contaminated
historical backtest into proof of future alpha.

## Preregistered TCN and graph shadow study

`tcn_shadow_research.py` implements the separately preregistered experiment in
`TCN_SHADOW_PROTOCOL.md`. It preserves the frozen OLS residual signal and
55%-volatility-budget control, tests a 1,937-parameter causal TCN only as a
one-sided SOXL confidence modifier, and uses graph diffusion only as a shock
haircut. It has no production-state, notification, or allocation authority.

Install the optional, version-pinned research dependency separately:

```powershell
python -m pip install -r requirements-research-ml.txt
```

Create the immutable extended-universe snapshot once:

```powershell
python tcn_shadow_research.py `
  --start 2010-01-01 `
  --end 2026-08-08 `
  --save-snapshot research_outputs/tcn_shadow_union_snapshot.csv `
  --output research_outputs/tcn_shadow_smoke.json `
  --bootstrap-samples 100
```

Replay the exact snapshot with the frozen inference budget:

```powershell
python tcn_shadow_research.py `
  --start 2010-01-01 `
  --end 2026-08-08 `
  --snapshot research_outputs/tcn_shadow_union_snapshot.csv `
  --output research_outputs/tcn_shadow_results.json `
  --bootstrap-samples 10000
```

The observed 2026-08-09 result rejected the TCN, graph, hybrid, and cash-gated
challengers. It is recorded in `RESEARCH_FINDINGS.md`; no challenger advances
to the prospective shadow ledger, and production remains unchanged.

## Leveraged core and satellite universe study

`core_universe_research.py` implements the separately frozen experiment in
`CORE_UNIVERSE_PROTOCOL.md`. It tests QLD, SSO, UPRO, USD, and TECL alongside
GLD and IEF diversifiers while preserving the live residual signal, 55% risk
budget, state transitions, drift controls, next-open fills, and actual-share
accounting. It is research-only and cannot change a live position.

Replay the immutable common-universe snapshot with the full inference budget:

```powershell
python core_universe_research.py `
  --start 2010-01-01 `
  --end 2026-08-08 `
  --snapshot research_outputs/core_universe_snapshot.csv `
  --bootstrap-samples 10000 `
  --output research_outputs/core_universe_results.json
```

The observed 2026-08-09 result retained `live_qld_soxl` as the maximum-growth
choice. The 80% QLD / 20% GLD base qualified as a lower-growth risk-efficiency
alternative, not a replacement. Every SSO, UPRO, USD, TECL, equal-weight, and
router challenger failed the frozen growth gate. Exact results and provenance
are recorded in `RESEARCH_FINDINGS.md`.
