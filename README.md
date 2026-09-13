# Roth IRA production allocator

This repository contains one stateful, notification-only Roth IRA allocator.
The production revision is
`tqqq-upro40-dbmf20-zroz20-ugl20-sma200-annual-v3`.
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

The compact dashboard also reports QQQ versus its 50-session average and its
252-session momentum. These are supporting health checks, not extra trade
triggers. Tests found that attaching short-horizon filters to this router
sharply increased whipsaw and reduced net growth.

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
room.

TQQQ and UPRO both target three times their index's **daily** return. The switch
changes the equity engine from Nasdaq-100 to S&P 500 exposure; it does not
reduce the leverage multiplier. The base portfolio's advertised daily exposure
is 2.00x: 1.20x equity, 0.20x managed futures, 0.20x long Treasuries, and 0.40x
gold. Actual returns and risk are path-dependent.

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

The engine also issues an action for a TQQQ/UPRO switch or a lifecycle transition.
A midyear equity switch replaces the
current equity fund without rebalancing DBMF, ZROZ, UGL, or cash. Ordinary
allocation drift waits for the annual rebalance.

Other HOLD runs do not email. Identical pending instructions are suppressed. A
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
python -m py_compile port12_cloud.py alpha_core.py contribution_core.py
python port12_cloud.py --test --roth-amount 10000
git diff --check
git status --short
```
