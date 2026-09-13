# Roth IRA production allocator

This repository contains one stateful, notification-only Roth IRA allocator.
The current production revision is `tqqq40-dbmf20-zroz20-gld20-soxl15-v1`.
It does not connect to a broker or place trades automatically: it calculates a
next-session plan, emails only when a decision requires action, and treats
confirmed broker shares and cash as the source of truth.

## Current strategy

At the current SOXL tier `s`, the sprint target is:

```text
TQQQ = 40% * (1 - s)
DBMF = 20% * (1 - s)
ZROZ = 20% * (1 - s)
GLD  = 20% * (1 - s)
SOXL = s
```

`s` has only two production values: `0%` and `15%`. The 15% SOXL tier is
admitted only when both of these completed-close conditions are satisfied:

- QQQ is strictly above its 200-session SMA; and
- 63-session residual momentum for the separated-window SMH-on-QQQ OLS model
  is positive.

The causal HAR-style volatility forecast and its 55% overlay budget decide
whether the 15% tier fits. This is a risk-sizing gate, not a promise that
realized volatility will remain below 55%. Trend failure exits SOXL
immediately; re-entry requires two distinct eligible closes. The current
revision deliberately has no SMA-based TQQQ-to-QLD/QQQ de-leveraging rule yet;
that is a separate risk-management design discussion.

The maximum advertised daily exposure is 1.80x with no SOXL and 1.98x when the
15% tier is active. The core sleeves are intentionally simple and fixed; the
only tactical sleeve in this first revision is the single SOXL tier.

## Lifecycle reserve policy

The existing one-way lifecycle ratchet remains enabled. Account value and age
independently impose an exposure ceiling, and the safer ceiling wins. Until a
future revision specifies product-level SMA deleveraging, a lifecycle ceiling
is implemented by scaling the current TQQQ/DBMF/ZROZ/GLD/SOXL target
proportionally into `SGOV`. A stage never re-levers after it advances.

| Stage | 2026-dollar value gate | Age gate | Ceiling |
|---|---:|---:|---:|
| `SPRINT` | below $250,000 | below 45 | current target (1.80x–1.98x) |
| `GLIDE_225` | $250,000 | 45 | 2.25x maximum |
| `TWO_X` | $500,000 | 50 | 2.00x maximum |
| `PHI` | $1,000,000 | 55 | 1.618x |
| `ONE_THREE` | $2,000,000 | 59.5 | 1.30x |
| `ONE_X` | $5,000,000 | 65 | 1.00x |
| `RETIREMENT` | age only | 70 | 0.75x plus 25% SGOV |

The value gates are indexed at 2.5% annually from August 14, 2026. Set the
optional `INVESTOR_BIRTH_DATE` repository secret (`YYYY-MM-DD`) for exact age
boundaries; otherwise the dated age-23 anchor is used.

## Rebalancing and notifications

The engine rebalances only for a structural change, a five-percentage-point
individual drift, or a five-point aggregate equity drift. Risk-off SOXL exits
and lifecycle transitions bypass the drift band. Ordinary HOLD runs are
silent. Email is sent only for a new action, a material update, a one-time
cancellation, or a delivery retry. Duplicate pending recommendations are
suppressed.

Legacy QLD/UGL/TECL/other positions remain priceable during migration so that
the first run can sell them explicitly; they are not strategic targets in this
revision. No state, balances, addresses, or credentials belong in Git.

## Production workflow

The GitHub Actions workflow runs after completed NYSE closes. When an action
email arrives:

1. Recalculate quantities using executable next-session prices.
2. Execute the trades manually at the broker.
3. Run the workflow with `run_mode: confirm-execution`.
4. Supply the email's signal date and every final holding as `TICKER=SHARES`,
   including `CASH=...`.

Use `run_mode: sync-holdings` for a contribution, withdrawal, dividend, or
broker correction when no recommendation is pending. For a genuinely new
all-cash account, use `run_mode: signal`, `initialize_portfolio: true`, and
set the `ROTH_IRA_AMOUNT` secret. Do not use that secret to replace existing
broker holdings.

## Research governance

The runner records non-trading comparison targets in a hash-chained shadow
ledger. Historical QLD/UGL and other variants remain research or migration
artifacts; they cannot alter production holdings, orders, notifications, or
the live strategy fingerprint. A future TQQQ deleveraging rule must be tested
as a new frozen revision before promotion.

## Local validation

```powershell
python -m unittest discover -s tests -v
python -m py_compile port12_cloud.py alpha_core.py
python port12_cloud.py --test --roth-amount 10000
git diff --check
git status --short
```

Test mode may download live market data, but it never saves production state,
writes logs or audits, or sends email.
