# Roth IRA production allocator

This repository contains one stateful, notification-only Roth IRA allocator.
The production revision is
`tqqq40-btal10-extreme-bear-8-2-annual-v4`.
It does not place broker trades. It calculates next-session instructions,
emails only when action is required, and values the account from confirmed
shares and cash rather than an old target allocation.

The dashboard leads with the required action, then shows the target, signal,
contribution status, and only the orders that need to be placed. Detailed risk
and turnover diagnostics remain in the structured audit artifact rather than
cluttering the email. Each audit also records measurement-only paper-track
comparisons, realized volatility, sleeve risk contribution, and estimated
leveraged-fund financing drag. These diagnostics cannot change or block a
portfolio decision and are explicitly not the account's dollar-weighted return.

## Production allocation

The base target always contains:

| Sleeve | Weight |
|---|---:|
| TQQQ | 40% normally; 30% with the crisis hedge active |
| BTAL | 0% normally; 10% with the crisis hedge active |
| DBMF | 20% |
| ZROZ | 20% |
| UGL | 20% |

The tactical sleeve uses one stateful extreme-bear rule:

- Normally hold 40% `TQQQ` and no `BTAL`.
- Enter 30% TQQQ / 10% BTAL after two distinct completed sessions where QQQ
  closes at or below 92% of its SMA200 and SPY closes below its SMA200.
- Exit BTAL and restore 40% TQQQ after two distinct completed sessions where
  QQQ closes at or above 98% of its SMA200 and SPY closes above its SMA50.
- The asymmetric 8% entry and 2% recovery boundaries form a deadband.
- A duplicate run for the same close cannot advance either count.
- Missed completed sessions are replayed in order.

The compact dashboard reports the QQQ/SMA200 distance, SPY confirmation
distances, and the two stateful confirmation counters. QQQ's SMA50 and
252-session momentum remain available to contribution and diagnostic code but
cannot activate the crisis hedge.

## Contribution deployment

An optional stateful contribution plan emails only when new Roth funding is
due. The configured budget means the amount still available to contribute for
that tax year; it is not assumed to equal the statutory maximum or the account
balance.

- 50% is released immediately.
- Five additional 10% calendar tranches become due in March, May, July,
  September, and November.
- A QQQ close below its 50-session SMA but still above its 200-session SMA
  advances one future tranche once.
- A 10% QQQ drawdown from its trailing 63-session high advances one future
  tranche once.
- A 20% drawdown releases every remaining tranche.
- Calendar dates are floors: market strength can never postpone a tranche, and
  the budget is fully released by November.

Each email allocates the new dollars across the current target's underweight
holdings and includes estimated units. Volume is not a gate. Notices have
retry-safe state, but emailed amounts are recorded as notices—not fabricated
broker deposits or fills. Confirmed shares and cash remain the accounting source
of truth.

After holdings have been initialized, configure the remaining annual budget:

```powershell
python port12_cloud.py --configure-contributions --contribution-budget 7500 --contribution-year 2026
```

The GitHub workflow exposes the same `configure-contributions` operation. Use
`disable-contributions` to turn the scheduler off. Configure a new remaining
budget each tax year; the engine never assumes Roth eligibility or contribution
room. A plan must be configured for the current New York calendar year because
the scheduler cannot safely act on prior- or future-year contribution room.

TQQQ targets three times the Nasdaq-100's **daily** return. BTAL is a
dollar-neutral long-low-beta/short-high-beta strategy, not a guaranteed inverse
fund. The normal portfolio's advertised gross daily exposure is 2.00x; the
hedged target is 1.80x. Actual beta, returns, and risk remain path-dependent.

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

On the first completed NYSE signal of each calendar year, the engine performs
an exact annual rebalance. If the portfolio is already exact, it sends one
annual-review email and records the completed year without requesting a trade.

The engine also issues an action for a BTAL crisis-hedge transition or a
lifecycle transition. A midyear hedge transition moves one quarter of the
current TQQQ/BTAL tactical sleeve between TQQQ and BTAL without rebalancing
DBMF, ZROZ, UGL, or cash. Ordinary allocation drift waits for the annual
rebalance.

Other HOLD runs do not email. Identical pending instructions are suppressed. A
material change replaces the pending action, a no-longer-needed action gets one
cancellation, and failed SMTP delivery stays pending for retry.
Use the explicit `resend-notification` workflow operation when another copy of
the current pending portfolio action is needed; ordinary scheduled runs remain
quiet.

Confirmed holdings also determine whether the crisis hedge is actually
aligned. Missing TQQQ exposure, an unexpected BTAL position, or a legacy UPRO
position cannot be hidden by stale state metadata. Confirming a late fill
from an older signal updates the holdings without erasing a newer pending action.

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
- QQQ and SPY signal history must contain the latest completed NYSE sessions with no
  synthetic fill or stale daily bar.
- State is saved atomically and stored as a private workflow artifact, never in
  Git.
- Test mode never saves state, writes an audit/log, or sends email.

```powershell
python -m unittest discover -s tests -v
python -m py_compile port12_cloud.py alpha_core.py contribution_core.py performance_core.py
python port12_cloud.py --test --roth-amount 10000
git diff --check
git status --short
```
