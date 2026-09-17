# Roth IRA Static Annual Allocation

This repository contains a notification-only Roth IRA allocator. It does not
place broker trades. The engine values confirmed shares and cash, calculates
the exact target, and emails only when a new annual allocation or configured
annual contribution requires attention.

## Production target

| Holding | Target |
|---|---:|
| TQQQ | 40% |
| DBMF | 32% |
| UGL | 15% |
| ZROZ | 5% |
| BTAL | 8% |

The target is static for the full calendar year. There are no SMA, volatility,
regime, deadband, lifecycle, or tactical signals. The first completed NYSE
session of each new calendar year requests an exact rebalance. Midyear price
drift is intentionally ignored.

The advertised gross daily exposure is approximately 1.90x (TQQQ 3x and UGL
2x are daily leveraged products). Actual returns, correlation, decay, and
drawdown are path-dependent and are not guaranteed by the advertised figure.

## Operating cycle

1. On the first run, initialize an all-cash account with `ROTH_IRA_AMOUNT`, or
   synchronize complete broker holdings with `TICKER=SHARES` and `CASH=...`.
2. When an annual email arrives, place the listed orders manually using
   executable prices during the next session.
3. Confirm the complete post-trade shares and remaining cash with
   `run_mode: confirm-execution`. The engine stores those confirmed holdings as
   the source of truth.
4. Configure an optional annual contribution budget. It is released once at
   the annual allocation review and allocated to the static target; no market
   condition can accelerate or delay it.

HOLD runs are silent. Identical pending annual recommendations are suppressed.

## CLI examples

```powershell
python port12_cloud.py --test --roth-amount 10000
python port12_cloud.py --sync-holdings --executed-shares TQQQ=10 DBMF=20 UGL=5 ZROZ=4 BTAL=2 CASH=123.45
python port12_cloud.py --configure-contributions --contribution-budget 7500 --contribution-year 2026
```

Use the GitHub Actions workflow for production state and email delivery. The
`send-test-email` operation sends a connectivity test without reading or
changing portfolio state.

## Validation

```powershell
python -m unittest discover -s tests -v
python -m py_compile port12_cloud.py alpha_core.py contribution_core.py
python port12_cloud.py --test --roth-amount 10000
git diff --check
```

Credentials, balances, shares, and state files belong in GitHub secrets or
private workflow artifacts and must never be committed.
