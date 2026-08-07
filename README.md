# Roth IRA QLD/SOXL allocation engine

This repository runs a stateful, notification-only Roth IRA allocation engine.
QLD is the permanent core. A maximum 35% SOXL satellite is admitted by QQQ
trend, SMH residual strength, and a bounded volatility-risk forecast.
This allocator is explicitly experimental and has not completed the frozen
research-promotion requirements.

The engine does not connect to a broker and does not place trades. It emails
only when an action is new, materially changed, cancelled, or needs delivery
retry. Ordinary HOLD runs are saved silently.
Every unseen completed session is replayed after an outage. A hash-chained,
signal-only shadow ledger is persisted with state so future evidence cannot be
silently rewritten when the workflow misses a day. Its count, final date, and
chain hash are anchored in portfolio state; a missing or truncated initialized
ledger fails closed.

## Production workflow

The scheduled GitHub Actions job runs after each completed XNYS close.

When an email arrives:

1. Treat its quantities as signal-close estimates.
2. Recalculate orders from executable next-session prices.
3. Trade manually at the broker.
4. Run the workflow with `run_mode: confirm-execution`.
5. Supply the email's signal date and every final broker holding as
   `TICKER=SHARES`, including `CASH=...`.

Confirmed shares and cash—not a prior target—drive every later valuation and
decision. The engine cannot safely infer fills, partial fills, price
improvement, dividends, or broker cash.

This confirmation is needed only after an action email, not after silent HOLD
runs. The workflow remembers the last confirmed holdings and suppresses an
identical pending recommendation.

Use `run_mode: sync-holdings` only for a contribution, withdrawal, dividend, or
broker correction when no recommendation is pending. Supply the complete
account, including cash.

## First initialization

For a genuinely new all-cash account with no saved state:

- dispatch `run_mode: signal`;
- set `initialize_portfolio: true`;
- create the `ROTH_IRA_AMOUNT` repository secret.

For an existing invested account, initialize using `sync-holdings` and the
complete broker holdings. Do not use `ROTH_IRA_AMOUNT` to replace existing
state.

## Repository secrets

- `ROTH_IRA_AMOUNT`: first-run cash only.
- `GMAIL_ADDRESS`: sender address, read only when a notification is due.
- `GMAIL_APP_PASSWORD`: Gmail app password, read only when a notification is
  due.
- `RECEIVER_EMAIL`: notification recipient, read only when a notification is
  due.

No state, holdings, balances, addresses, or credentials belong in Git.
Production state and the non-sensitive shadow chain are restored and saved as
the `roth-ira-state` workflow artifact.

SMTP and GitHub artifacts cannot form one atomic transaction. If the runner
disappears after Gmail accepts a message but before delivery proof is uploaded,
a rare duplicate retry is possible. Every message includes its signal date and
exact destination; never execute the same recommendation twice.

## Local validation

```powershell
python -m unittest discover -s tests -v
python -m py_compile port12_cloud.py alpha_core.py alpha_research.py `
  alpha_paired_research.py legacy_original_research.py
python port12_cloud.py --test --roth-amount 10000
git diff --check
git status --short
```

Test mode may download live yfinance data, but it never saves production state,
writes an audit/log, or sends email.

## Research

Read `ALPHA_RESEARCH_PROTOCOL.md`, `RESEARCH.md`, and
`RESEARCH_FINDINGS.md`. The current strategy remains experimental: historical
results do not prove future alpha, and the committed findings explicitly
preserve failed promotion flags and model-selection risk.
