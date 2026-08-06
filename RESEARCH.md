# Portfolio research protocol

`research_backtest.py` is an isolated research tool. It imports the reviewed
production decision functions, but it never reads or writes
`roth_ira_state.json`, never writes a decision audit, and never sends email.
Nothing in this harness changes the live allocation.

## Locked execution model

- Signals use adjusted data through completed XNYS close `t`.
- A queued action fills at the adjusted open of the next validated XNYS
  session.
- Fractional shares and cash—not a prior target—drive every later weight.
- Trading costs are deducted from cash on gross security notional. The solver
  targets exact post-cost weights.
- Initial deployment costs affect wealth, but initial deployment is excluded
  from ongoing turnover.
- Missing sessions, required prices, QQQ signal volume, or execution opens fail
  closed. Data is never filled, dropped, or replaced with another ticker.
- Primary results use the first actual common inception of the full ETF
  universe. A later `--score-start` is a labeled sensitivity run.

## Prespecified first-round family

The locked Original is the baseline. Each candidate changes one mechanism:

1. `c1_wide_buffer`: 7.5pp drift trigger / 3.75pp destination.
2. `c2_bull_reentry_2`: immediate bearish exit / two bullish closes to re-enter.
3. `c3_vol_hysteresis`: immediate de-risking / buffered two-close re-risking.
4. `c4_momentum_63`: 63-session SOXL/TECL leader momentum.
5. `c5_equal_leaders`: equal SOXL/TECL weights in low/moderate tiers.

The descriptive benchmarks are buffered 50/50 SPMO/SMH, QLD buy-and-hold, and
SPY buy-and-hold. Candidates are not combined after seeing first-round results.

## Reproducible use

Download once and freeze the exact adjusted inputs:

```powershell
python research_backtest.py `
  --start 2014-01-01 `
  --end 2026-08-06 `
  --snapshot research_outputs/market_snapshot.csv `
  --output research_outputs/download_results.json
```

Replay the immutable local snapshot for final analysis:

```powershell
python research_backtest.py `
  --start 2014-01-01 `
  --end 2026-08-06 `
  --input-snapshot research_outputs/market_snapshot.csv `
  --cost-bps 0 5 10 25 `
  --bootstrap-samples 20000 `
  --output research_outputs/verified_results.json `
  --ledger-dir research_outputs/verified_ledgers
```

Research outputs and market snapshots are intentionally ignored by Git. The
JSON report records the full-data SHA-256, production fingerprints, locked
variant definitions, dependency versions, return checksums, and deterministic
random seed.

## Interpretation

The primary comparison is paired annualized log-growth difference at 10bp.
The report also provides 0/5/25bp sensitivity, four chronological robustness
slices, a centered paired moving-block bootstrap, a prespecified
Benjamini–Hochberg adjustment, complete-calendar-year diagnostics, and a
CSCV-style candidate-matrix selection-instability diagnostic.

These are exploratory results because the historical span was already viewed.
The unavailable daily paths for the earlier Strategies A and B cannot be
reconstructed from summary metrics and therefore cannot honestly enter paired
inference, DSR, BH, or historical-selection PBO. No candidate is eligible for
live promotion without an untouched/future shadow period, complete trial
lineage, acceptable liquidity, and explicit approval.
