from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

import alpha_research as research
import tcn_shadow as shadow
import tcn_shadow_research as runner


def alpha_data(
    *,
    periods: int = 8,
    start: str = "2024-01-02",
) -> research.AlphaMarketData:
    sessions = pd.bdate_range(start, periods=periods)
    positions = np.arange(periods, dtype=float)
    opens = pd.DataFrame(index=sessions)
    closes = pd.DataFrame(index=sessions)
    volumes = pd.DataFrame(index=sessions)
    for number, ticker in enumerate(research.TICKERS):
        base = 50.0 + 20.0 * number
        opens[ticker] = base * np.exp(0.002 * positions)
        closes[ticker] = opens[ticker] * 1.001
        volumes[ticker] = 1_000_000.0
    return research.AlphaMarketData(
        opens=opens,
        closes=closes,
        volumes=volumes,
        sessions=sessions,
        common_start=sessions[0],
        final_session=sessions[-1],
        fingerprint=research._full_data_fingerprint(opens, closes, volumes),
        requested_start=sessions[0].date().isoformat(),
        requested_end_exclusive=(
            sessions[-1] + pd.Timedelta(days=1)
        ).date().isoformat(),
        source="synthetic shadow test data",
    )


def extended_raw(
    *,
    final_session: str = "2010-06-01",
) -> tuple[pd.DataFrame, str, str]:
    start = research.MODEL_HISTORY_START
    sessions = research._calendar_sessions(
        pd.Timestamp(start),
        pd.Timestamp(final_session),
    )
    columns = pd.MultiIndex.from_product(
        [["Open", "Close", "Volume"], shadow.REQUIRED_TICKERS]
    )
    raw = pd.DataFrame(index=sessions, columns=columns, dtype=float)
    positions = np.arange(len(sessions), dtype=float)
    for number, ticker in enumerate(shadow.REQUIRED_TICKERS):
        price = (40.0 + 5.0 * number) * np.exp(0.0005 * positions)
        raw[("Open", ticker)] = price * 0.999
        raw[("Close", ticker)] = price
        raw[("Volume", ticker)] = 1_000_000.0
    end = (sessions[-1] + pd.Timedelta(days=1)).date().isoformat()
    return raw, start, end


def constant_target(
    sessions: pd.DatetimeIndex,
    *,
    soxl_weight: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "QLD": 1.0 - soxl_weight,
            "SOXL": soxl_weight,
            "CASH": 0.0,
        },
        index=sessions,
    )


def simple_ledger(index: pd.DatetimeIndex) -> pd.DataFrame:
    nav = runner.STARTING_CASH * 1.01 ** np.arange(1, len(index) + 1)
    return pd.DataFrame(
        {
            "nav": nav,
            "gross_trade_fraction": 0.0,
            "one_way_turnover": 0.0,
            "transaction_cost": 0.0,
            "security_orders": 0,
            "actual_soxl_weight": 0.0,
            "actual_cash_weight": 0.0,
        },
        index=index,
    )


class ExtendedMarketDataTests(unittest.TestCase):
    def test_context_only_non_xnys_row_is_excluded(self):
        raw, start, end = extended_raw()
        holiday = pd.Timestamp("2010-05-31")
        raw.loc[holiday] = np.nan
        for field, value in (("Open", 20.0), ("Close", 21.0), ("Volume", 1.0)):
            raw.loc[holiday, (field, "^VIX")] = value
        raw = raw.sort_index()

        data = runner.prepare_extended_market_data(
            raw,
            requested_start=start,
            requested_end_exclusive=end,
            source="calendar test",
        )

        self.assertNotIn(holiday, data.closes.index)
        self.assertTrue(data.closes.loc[data.base.common_start :].notna().all().all())

    def test_missing_context_value_on_xnys_session_fails_closed(self):
        raw, start, end = extended_raw()
        session = raw.index[40]
        raw.loc[session, ("Close", "^VIX")] = np.nan

        with self.assertRaisesRegex(RuntimeError, "invalid.*\\^VIX"):
            runner.prepare_extended_market_data(
                raw,
                requested_start=start,
                requested_end_exclusive=end,
                source="missing context test",
            )


class ShadowExecutionTests(unittest.TestCase):
    def test_initial_signal_and_later_change_fill_at_next_open(self):
        data = alpha_data()
        targets = constant_target(data.sessions, soxl_weight=0.0)
        signal = data.sessions[0]
        change_signal = data.sessions[1]
        targets.loc[change_signal:, ["QLD", "SOXL"]] = [0.8, 0.2]

        result = runner.simulate_target_schedule(
            data,
            targets,
            name="live_residual_vol55",
            cost_bps=0.0,
            first_signal_date=signal,
            mode="exact",
        )

        first_fill = result.ledger.index[0]
        second_fill = result.ledger.index[1]
        self.assertEqual(first_fill, data.sessions[1])
        self.assertEqual(result.ledger.loc[first_fill, "fill_signal_date"], signal)
        self.assertEqual(second_fill, data.sessions[2])
        self.assertEqual(
            result.ledger.loc[second_fill, "fill_signal_date"],
            change_signal,
        )
        self.assertAlmostEqual(
            result.ledger.loc[second_fill, "executed_soxl_weight"],
            0.2,
        )

    def test_modifier_reductions_are_exact_and_drift_triggers_at_five_points(self):
        desired = runner._target(0.20)
        reduction, reason = runner._queue_target(
            mode="modifier",
            desired=runner._target(0.15),
            executed=desired,
            actual=desired,
        )
        self.assertEqual(reason, "EXACT_RISK_REDUCTION")
        self.assertEqual(reduction, runner._target(0.15))

        no_trade, reason = runner._queue_target(
            mode="modifier",
            desired=desired,
            executed=desired,
            actual=runner._target(0.249999),
        )
        self.assertIsNone(no_trade)
        self.assertEqual(reason, "")

        rebalance, reason = runner._queue_target(
            mode="modifier",
            desired=desired,
            executed=desired,
            actual=runner._target(0.25),
        )
        self.assertEqual(reason, "DRIFT_REBALANCE")
        assert rebalance is not None
        self.assertAlmostEqual(rebalance["SOXL"], 0.225)

        no_cleanup, reason = runner._queue_target(
            mode="modifier",
            desired=desired,
            executed=desired,
            actual=runner._target(0.225),
        )
        self.assertIsNone(no_cleanup)
        self.assertEqual(reason, "")

    def test_exact_path_applies_ordinary_drift_after_structural_target(self):
        desired = runner._target(0.20)
        rebalance, reason = runner._queue_target(
            mode="exact",
            desired=desired,
            executed=desired,
            actual=runner._target(0.25),
        )
        self.assertEqual(reason, "DRIFT_REBALANCE")
        assert rebalance is not None
        self.assertAlmostEqual(rebalance["SOXL"], 0.225)


class MetricAndComparisonTests(unittest.TestCase):
    def test_metrics_include_the_initial_deployment_return(self):
        index = pd.bdate_range("2024-01-02", periods=5)
        metrics = runner.calculate_path_metrics(simple_ledger(index))
        self.assertAlmostEqual(
            metrics["annualized_log_growth"],
            252.0 * math.log(1.01),
        )
        self.assertAlmostEqual(metrics["cagr"], 1.01**252 - 1.0)

    def test_comparisons_reject_misaligned_dates(self):
        first_index = pd.bdate_range("2024-01-02", periods=30)
        second_index = pd.bdate_range("2024-01-03", periods=30)
        first = runner.PathResult(
            "tcn_modifier",
            10.0,
            simple_ledger(first_index),
            {},
            False,
        )
        second = runner.PathResult(
            "live_residual_vol55",
            10.0,
            simple_ledger(second_index),
            {},
            False,
        )
        with self.assertRaisesRegex(RuntimeError, "dates differ"):
            runner.chronological_differences(first, second)
        with self.assertRaisesRegex(RuntimeError, "dates differ"):
            runner.moving_block_difference(first, second, samples=100)


if __name__ == "__main__":
    unittest.main()
