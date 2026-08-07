from __future__ import annotations

import inspect
import json
import math
import unittest

import exchange_calendars as xcals
import numpy as np
import pandas as pd

import legacy_original_research as legacy


def _sessions(count: int = 320) -> pd.DatetimeIndex:
    calendar = xcals.get_calendar("XNYS")
    values = pd.DatetimeIndex(
        calendar.sessions_in_range("2022-01-03", "2024-12-31")
    )[:count]
    if values.tz is not None:
        values = values.tz_convert(None)
    return values.normalize()


def _union_frames(
    *,
    count: int = 320,
    common_position: int = 205,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    sessions = _sessions(count)
    t = np.arange(len(sessions), dtype=float)
    closes = pd.DataFrame(index=sessions, columns=legacy.PAIRED_TICKERS, dtype=float)
    parameters = {
        "QQQ": (100.0, 0.0010, 0.0060, 0.15),
        "SMH": (80.0, 0.0012, 0.0080, 0.41),
        "QLD": (40.0, 0.0018, 0.0120, 0.73),
        "SOXL": (25.0, 0.0020, 0.0180, 1.07),
        "SPY": (110.0, 0.0007, 0.0045, 1.31),
        "TECL": (35.0, 0.0016, 0.0130, 1.61),
        "SPMO": (45.0, 0.0009, 0.0070, 1.91),
        "GLD": (70.0, 0.0002, 0.0035, 2.17),
    }
    for ticker, (initial, drift, amplitude, phase) in parameters.items():
        log_returns = (
            drift
            + amplitude * np.sin(0.29 * t + phase)
            + amplitude * 0.35 * np.cos(0.071 * t + phase)
        )
        closes[ticker] = initial * np.exp(np.cumsum(log_returns))

    opens = closes.copy()
    for index, ticker in enumerate(legacy.PAIRED_TICKERS):
        opens[ticker] = closes[ticker] * (
            1.0 + 0.002 * np.cos(0.19 * t + index * 0.23)
        )
    volumes = pd.DataFrame(
        {
            ticker: (
                1_000_000.0
                + 100_000.0 * np.sin(0.11 * t + index * 0.31)
            )
            for index, ticker in enumerate(legacy.PAIRED_TICKERS)
        },
        index=sessions,
    )

    # SPMO is the latest-inception instrument.  Missing pre-inception values
    # remain visible so the validator can choose a real common start.
    opens.loc[sessions[:common_position], "SPMO"] = np.nan
    closes.loc[sessions[:common_position], "SPMO"] = np.nan
    return opens, closes, volumes, sessions


class ProvenanceTests(unittest.TestCase):
    def test_frozen_manifest_matches_historical_fingerprint(self) -> None:
        self.assertEqual(
            legacy.CALCULATED_LEGACY_STRATEGY_FINGERPRINT,
            legacy.LEGACY_STRATEGY_FINGERPRINT,
        )
        self.assertEqual(len(legacy.LEGACY_SOURCE_COMMIT), 40)
        self.assertEqual(len(legacy.LEGACY_PORT12_BLOB_SHA), 40)
        self.assertEqual(len(legacy.LEGACY_RESEARCH_BLOB_SHA), 40)
        self.assertEqual(len(legacy.LEGACY_TESTS_BLOB_SHA), 40)
        source = inspect.getsource(legacy)
        self.assertNotIn("import port12_cloud", source)
        self.assertNotIn("import alpha_research", source)
        self.assertNotIn("smtplib", source)


class UnionValidationTests(unittest.TestCase):
    def test_common_start_uses_real_union_inception_without_filling(self) -> None:
        opens, closes, volumes, sessions = _union_frames()
        data = legacy.validate_union_market_data(opens, closes, volumes)
        self.assertEqual(data.common_start, sessions[205])
        self.assertTrue(math.isnan(float(data.closes.loc[sessions[204], "SPMO"])))
        self.assertEqual(data.scoring_dates[0], sessions[205])
        self.assertEqual(len(data.fingerprint), 64)

    def test_missing_session_ticker_and_scored_nan_fail_closed(self) -> None:
        opens, closes, volumes, sessions = _union_frames()
        with self.assertRaisesRegex(RuntimeError, "missing union tickers"):
            legacy.validate_union_market_data(
                opens.drop(columns=["SPY"]),
                closes,
                volumes,
            )

        missing_session = sessions[100]
        with self.assertRaisesRegex(RuntimeError, "continuity"):
            legacy.validate_union_market_data(
                opens.drop(index=missing_session),
                closes.drop(index=missing_session),
                volumes.drop(index=missing_session),
            )

        invalid = closes.copy()
        invalid.loc[sessions[240], "GLD"] = np.nan
        with self.assertRaisesRegex(RuntimeError, "Scored adjusted Closes"):
            legacy.validate_union_market_data(opens, invalid, volumes)

    def test_zero_volatility_fails_when_schedule_is_scored(self) -> None:
        opens, closes, volumes, _ = _union_frames()
        closes["QQQ"] = 100.0
        opens["QQQ"] = 100.0
        data = legacy.validate_union_market_data(opens, closes, volumes)
        with self.assertRaisesRegex(RuntimeError, "volatility"):
            legacy.build_original_target_schedule(data)


class StrategyRuleTests(unittest.TestCase):
    def test_exact_tier_boundaries_and_target_templates(self) -> None:
        self.assertEqual(legacy.classify_volatility(0.149999), "LOW")
        self.assertEqual(legacy.classify_volatility(0.15), "MODERATE")
        self.assertEqual(legacy.classify_volatility(0.22), "MODERATE")
        self.assertEqual(legacy.classify_volatility(0.220001), "HIGH")
        for invalid in (0.0, -0.1, float("nan")):
            with self.assertRaises(ValueError):
                legacy.classify_volatility(invalid)

        latest = pd.Series(
            {
                "annualized_volatility": 0.10,
                "bullish_consensus": 1,
                "soxl_momentum": 0.20,
                "tecl_momentum": 0.10,
            }
        )
        low = legacy.determine_target_allocation(
            latest,
            existing_leader=None,
            allow_leader_review=True,
            tier_decision=legacy.VolatilityTierDecision("LOW", "LOW"),
        )
        self.assertEqual(
            low.target_weights,
            {"SOXL": 0.45, "TECL": 0.15, "SMH": 0.25, "QLD": 0.15},
        )

        moderate = legacy.determine_target_allocation(
            latest,
            existing_leader="SOXL",
            allow_leader_review=False,
            tier_decision=legacy.VolatilityTierDecision(
                "MODERATE", "MODERATE"
            ),
        )
        self.assertEqual(
            moderate.target_weights,
            {"SOXL": 0.25, "TECL": 0.10, "SMH": 0.45, "QLD": 0.20},
        )
        high = legacy.determine_target_allocation(
            latest,
            existing_leader="SOXL",
            allow_leader_review=False,
            tier_decision=legacy.VolatilityTierDecision("HIGH", "HIGH"),
        )
        self.assertEqual(high.target_weights, {"SMH": 0.85, "GLD": 0.15})

        bearish = latest.copy()
        bearish["bullish_consensus"] = 0
        bear = legacy.determine_target_allocation(
            bearish,
            existing_leader="SOXL",
            allow_leader_review=False,
            tier_decision=legacy.VolatilityTierDecision("N/A", "N/A"),
        )
        self.assertEqual(bear.target_weights, {"SPMO": 0.80, "GLD": 0.20})

    def test_volatility_persistence_and_strict_leader_switch(self) -> None:
        de_risk = legacy.classify_volatility_with_persistence(
            0.23, "LOW", "", 0
        )
        self.assertEqual((de_risk.tier, de_risk.transition), ("HIGH", "DE_RISK"))

        first = legacy.classify_volatility_with_persistence(
            0.20, "HIGH", "", 0
        )
        self.assertEqual(first.tier, "HIGH")
        self.assertEqual(first.pending_tier, "MODERATE")
        self.assertEqual(first.pending_days, 1)
        second = legacy.classify_volatility_with_persistence(
            0.20,
            first.tier,
            first.pending_tier,
            first.pending_days,
        )
        self.assertEqual((second.tier, second.transition), ("MODERATE", "RE_RISK"))

        self.assertEqual(legacy.select_leader(0.0, 0.05, "SOXL"), "SOXL")
        self.assertEqual(
            legacy.select_leader(0.0, 0.050001, "SOXL"),
            "TECL",
        )
        self.assertEqual(legacy.select_leader(0.10, 0.10, None), "SOXL")

    def test_exact_five_point_drift_projects_to_inner_band(self) -> None:
        target = {"SMH": 0.50, "QLD": 0.50}
        exact = {"SMH": 0.55, "QLD": 0.45}
        below = {"SMH": 0.54999, "QLD": 0.45001}
        self.assertTrue(legacy.should_rebalance(exact, target))
        self.assertFalse(legacy.should_rebalance(below, target))
        projected = legacy.inner_band_rebalance_weights(exact, target)
        self.assertAlmostEqual(projected["SMH"], 0.525)
        self.assertAlmostEqual(projected["QLD"], 0.475)


class ScheduleTests(unittest.TestCase):
    def test_schedule_review_cadence_and_future_mutation_causality(self) -> None:
        opens, closes, volumes, _ = _union_frames()
        data = legacy.validate_union_market_data(opens, closes, volumes)
        schedule = legacy.build_original_target_schedule(data)
        due_positions = np.flatnonzero(
            schedule.frame["sector_review_due"].to_numpy(dtype=bool)
        ).tolist()
        self.assertEqual(due_positions[:5], [0, 21, 42, 63, 84])

        cutoff = data.scoring_dates[45]
        changed_closes = closes.copy()
        future = changed_closes.index > cutoff
        changed_closes.loc[future, "QQQ"] *= 1.35
        changed_closes.loc[future, "SOXL"] *= 0.70
        changed_data = legacy.validate_union_market_data(
            opens,
            changed_closes,
            volumes,
        )
        changed_schedule = legacy.build_original_target_schedule(changed_data)
        pd.testing.assert_frame_equal(
            schedule.frame.loc[:cutoff],
            changed_schedule.frame.loc[:cutoff],
        )
        self.assertNotEqual(data.fingerprint, changed_data.fingerprint)


class ExecutionTests(unittest.TestCase):
    def test_initial_signal_fills_at_next_open_from_actual_shares(self) -> None:
        opens, closes, volumes, _ = _union_frames()
        data = legacy.validate_union_market_data(opens, closes, volumes)
        schedule = legacy.build_original_target_schedule(data)
        start = data.scoring_dates[1]
        prior = data.scoring_dates[0]
        result = legacy.simulate_original(
            data,
            schedule,
            cost_bps=0.0,
            execution_start=start,
        )
        first = result.ledger.iloc[0]
        self.assertTrue(bool(first["initial_deployment"]))
        self.assertEqual(pd.Timestamp(first["fill_signal_date"]), prior)
        self.assertEqual(first["filled_reason"], "INITIAL_DEPLOYMENT")
        self.assertEqual(float(first["ongoing_gross_trade_fraction"]), 0.0)

        target = schedule.frame.loc[prior, "target_weights"]
        shares = json.loads(first["shares"])
        for ticker, weight in target.items():
            expected = weight * legacy.STARTING_CASH / data.opens.loc[start, ticker]
            self.assertAlmostEqual(shares[ticker], expected, places=10)

        log_returns = legacy.result_log_returns(result)
        self.assertEqual(log_returns.index.tolist(), result.ledger.index.tolist())
        self.assertAlmostEqual(
            float(log_returns.iloc[0]),
            math.log(float(first["nav"]) / legacy.STARTING_CASH),
        )
        self.assertEqual(
            result.metrics["starting_nav"],
            legacy.STARTING_CASH,
        )
        self.assertEqual(
            legacy.adapter_record(result)["name"],
            "legacy_original",
        )

    def test_missing_traded_volume_marks_liquidity_unmeasurable(self) -> None:
        opens, closes, volumes, _ = _union_frames()
        for ticker in legacy.TRADED_TICKERS:
            volumes[ticker] = 0.0
        data = legacy.validate_union_market_data(opens, closes, volumes)
        schedule = legacy.build_original_target_schedule(data)
        result = legacy.simulate_original(data, schedule, cost_bps=10.0)
        self.assertTrue(
            math.isinf(
                float(
                    result.ledger.iloc[0][
                        "max_trade_to_median_dollar_volume20"
                    ]
                )
            )
        )
        self.assertGreater(result.metrics["total_modeled_cost"], 0.0)
        self.assertGreaterEqual(
            result.metrics["unmeasurable_liquidity_trade_sessions"],
            1,
        )

    def test_repeated_runs_are_stateless_and_deterministic(self) -> None:
        opens, closes, volumes, _ = _union_frames()
        data = legacy.validate_union_market_data(opens, closes, volumes)
        schedule = legacy.build_original_target_schedule(data)
        first = legacy.simulate_original(data, schedule, cost_bps=10.0)
        second = legacy.simulate_original(data, schedule, cost_bps=10.0)
        pd.testing.assert_frame_equal(first.ledger, second.ledger)
        self.assertEqual(first.metrics, second.metrics)


if __name__ == "__main__":
    unittest.main()
