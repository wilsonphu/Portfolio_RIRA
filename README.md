# Roth IRA production allocator

This repository contains one stateful, notification-only Roth IRA allocator.
The production revision is `tqqq-upro40-dbmf20-zroz20-ugl20-sma200-v1`.
It does not place broker trades. It calculates next-session instructions,
emails only when action is required, and values the account from confirmed
shares and cash rather than an old target allocation.

## Production allocation

The base target always contains:

| Sleeve | Weight |
|---|---:|
| Routed equity ETF | 40% |
| DBMF | 20% |
| ZROZ | 20% |
| UGL | 20% |

The 40% equity sleeve uses one completed-close QQQ trend rule:

- Hold `TQQQ` after two distinct QQQ closes strictly above its 200-session SMA.
- Switch immediately to `UPRO` after one QQQ close at or below its 200-session SMA.
- A duplicate run for the same close cannot advance the bullish count.
- Missed completed sessions are replayed in order.

TQQQ and UPRO both target three times their index's **daily** return. The switch
changes the equity engine from Nasdaq-100 to S&P 500 exposure; it does not
reduce the leverage multiplier. The base portfolio's advertised daily exposure
is 2.00x: 1.20x equity, 0.20x managed futures, 0.20x long Treasuries, and 0.40x
gold. Actual returns and risk are path-dependent.

SOXL is no longer a strategic sleeve. The engine retains migration-only pricing
for SOXL and other prior holdings so an existing position appears as an explicit
SELL instead of disappearing from account state.

## Lifecycle reserve

The one-way age/value ratchet remains enabled. When its exposure ceiling falls
below the base target, all risky sleeves are scaled proportionally and the
remainder moves to `SGOV`. A stage never moves backward.

| Stage | 2026-dollar value gate | Age gate | Exposure ceiling |
|---|---:|---:|---:|
| `SPRINT` | below $250,000 | below 45 | base target |
| `GLIDE_225` | $250,000 | 45 | 2.25x |
| `TWO_X` | $500,000 | 50 | 2.00x |
| `PHI` | $1,000,000 | 55 | 1.618x |
| `ONE_THREE` | $2,000,000 | 59.5 | 1.30x |
| `ONE_X` | $5,000,000 | 65 | 1.00x |
| `RETIREMENT` | age only | 70 | 0.75x |

Value gates rise 2.5% annually from August 14, 2026. Set the optional
`INVESTOR_BIRTH_DATE` secret in `YYYY-MM-DD` form for exact age boundaries.

## Rebalancing and email

The engine issues an action for a TQQQ/UPRO switch, a lifecycle transition, an
individual position drift of at least five percentage points, aggregate equity
drift of at least five points, or an obsolete holding that must be sold. An
ordinary drift rebalance trades only far enough to return inside a 2.5-point
band. Structural transitions use the exact target.

HOLD runs do not email. Identical pending instructions are suppressed. A
material change replaces the pending action, a no-longer-needed action gets one
cancellation, and failed SMTP delivery stays pending for retry.

## Operating cycle

The scheduled GitHub workflow runs after completed NYSE closes. An emailed
quantity is a signal-close estimate, not a guaranteed fill.

1. Recalculate quantities from executable prices during the next session.
2. Execute the trades manually at the broker.
3. Run the workflow with `run_mode: confirm-execution`.
4. Enter the email's signal date and every final holding as `TICKER=SHARES`,
   including `CASH=...`.

Use `run_mode: sync-holdings` after a contribution, withdrawal, dividend, or
broker correction when no action is pending. A truly new all-cash account must
be explicitly initialized with `run_mode: signal`, `initialize_portfolio: true`,
and the `ROTH_IRA_AMOUNT` secret. Never use that secret to overwrite an existing
account.

## Safety and validation

- Latest prices must cover every held and strategic ticker.
- QQQ signal history must contain the latest completed NYSE sessions with no
  synthetic fill or stale daily bar.
- State is saved atomically and stored as a private workflow artifact, never in
  Git.
- Test mode never saves state, writes an audit/log, or sends email.

```powershell
python -m unittest discover -s tests -v
python -m py_compile port12_cloud.py alpha_core.py
python port12_cloud.py --test --roth-amount 10000
git diff --check
git status --short
```
