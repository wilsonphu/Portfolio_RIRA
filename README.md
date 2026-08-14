# Roth IRA production allocator

This repository contains one production system: a stateful, notification-only
TQQQ/UGL core with a bounded SOXL overlay. It does not connect to a broker or
place trades automatically.

## Strategy

At strategic SOXL weight `s`, the portfolio target is:

```text
TQQQ = 65% * (1 - s)
UGL  = 35% * (1 - s)
SOXL = s
```

`s` is restricted to four deliberate tiers: `0%`, `15%`, `25%`, or `35%`.
The SOXL sleeve requires both:

- QQQ strictly above its completed-close 200-session SMA; and
- positive 21-session residual momentum from a separated-window SMH-on-QQQ
  OLS model.

The sleeve is sized with the existing causal QLD/SOXL HAR-style volatility
forecast and a 55% overlay budget. Trend failure exits SOXL immediately.
Residual exits occur on the fixed 21-session review clock. Re-entry requires
two distinct eligible closes; volatility reductions are immediate and
increases require five completed sessions.

When both alpha gates are open, the volatility model chooses the highest tier
that fits the risk budget. If 15% does not fit, SOXL remains at 0%. Any
triggered rebalance returns SOXL to its exact strategic tier; other positions
normally stop at the inner drift band.

The strategy is deployed to production by investor authorization. Historical
backtests are not evidence that its return advantage will persist. A roughly
two-thirds portfolio drawdown remains plausible.

## Portfolio state and notifications

Confirmed broker shares and cash—not calculated target weights—are the sole
source of truth. The engine rebalances only for a structural change, a
five-percentage-point individual drift, or a five-point aggregate equity
drift. SOXL returns to its exact strategic tier; other positions normally
trade back to the inner 2.5-point band.

Ordinary HOLD runs are silent. Email is sent only for a new action, a material
update, a one-time cancellation, or a delivery retry. Every unseen completed
NYSE session is replayed after an outage, and the immutable signal ledger is
hash-chained and anchored in state.

## Production workflow

The GitHub Actions workflow runs after completed NYSE closes. When an action
email arrives:

1. Recalculate the quantities using executable next-session prices.
2. Execute the trades manually at the broker.
3. Run the workflow with `run_mode: confirm-execution`.
4. Supply the email's signal date and every final holding as
   `TICKER=SHARES`, including `CASH=...`.

Use `run_mode: sync-holdings` only for a contribution, withdrawal, dividend,
or broker correction when no recommendation is pending. Always supply the
complete account.

### First initialization

For a genuinely new all-cash account with no saved state:

- dispatch `run_mode: signal`;
- set `initialize_portfolio: true`; and
- create the `ROTH_IRA_AMOUNT` repository secret.

For an already-invested account, initialize with `sync-holdings` and complete
broker holdings. Never use `ROTH_IRA_AMOUNT` to replace existing state.

### Repository secrets

- `ROTH_IRA_AMOUNT`: first-run cash only.
- `GMAIL_ADDRESS`: notification sender.
- `GMAIL_APP_PASSWORD`: Gmail app password.
- `RECEIVER_EMAIL`: notification recipient.

State, holdings, balances, addresses, and credentials must never be committed.
Production state is restored from and saved to the `roth-ira-state` workflow
artifact.

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
