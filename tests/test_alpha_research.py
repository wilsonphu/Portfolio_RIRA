from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import alpha_research as research


def alpha_data(
    *,
    periods: int = 80,
    start: str = "2024-01-02",
) -> research.AlphaMarketData:
    index = pd.bdate_range(start, periods=periods)
    positions = np.arange(periods, dtype=float)
    closes = pd.DataFrame(index=index)
    opens = pd.DataFrame(index=index)
    volumes = pd.DataFrame(index=index)
    for number, ticker in enumerate(research.TICKERS):
        returns = (
            0.0005
            + 0.003 * np.sin(positions / (5.0 + number))
            + 0.0004 * np.cos(positions / (2.0 + number))
        )
        close = (50.0 + 10.0 * number) * np.exp(np.cumsum(returns))
        closes[ticker] = close
        opens[ticker] = close * (1.0 + 0.001 * np.cos(positions / 3.0))
        volumes[ticker] = 2_000_000.0 + 10_000.0 * number
    return research.AlphaMarketData(
        opens=opens,
        closes=closes,
        volumes=volumes,
        sessions=index,
        common_start=index[0],
        final_session=index[-1],
        fingerprint=research._full_data_fingerprint(opens, closes, volumes),
        requested_start=index[0].date().isoformat(),
        requested_end_exclusive=(index[-1] + pd.Timedelta(days=1)).date().isoformat(),
        source="synthetic test data",
    )


def signal_panel(
    data: research.AlphaMarketData,
    *,
    raw_weight: float = 0.20,
) -> research.SignalPanel:
    frame = pd.DataFrame(index=data.sessions)
    frame["trend_positive"] = True
    frame["residual_available"] = True
    frame["residual_positive"] = True
    frame["residual_momentum"] = 0.01
    frame["residual_z"] = 1.0
    frame["volatility_available"] = True
    frame["raw_soxl_weight"] = raw_weight
    frame["ridge_available"] = True
    frame["ridge_prediction"] = 0.01
    return research.SignalPanel(
        frame=frame,
        ridge_details={},
        data_fingerprint=data.fingerprint,
    )


class FrozenRegistryTests(unittest.TestCase):
    def test_software_fingerprint_includes_shared_alpha_core(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_core = Path(directory) / "alpha_core.py"
            fake_core.write_text("version = 1\n", encoding="utf-8")
            with mock.patch.object(
                research.core,
                "__file__",
                str(fake_core),
            ):
                first = research.software_fingerprint()
                fake_core.write_text("version = 2\n", encoding="utf-8")
                second = research.software_fingerprint()
        self.assertNotEqual(first, second)

    def test_registry_and_fingerprint_are_complete_and_deterministic(self):
        self.assertEqual(
            tuple(research.CANDIDATES),
            research.FROZEN_CANDIDATE_NAMES,
        )
        self.assertEqual(len(research.FROZEN_CANDIDATE_NAMES), 7)
        self.assertEqual(len(research.FROZEN_FAMILY_FINGERPRINT), 64)
        self.assertEqual(
            research.FROZEN_FAMILY_FINGERPRINT,
            research._canonical_sha256(research.FROZEN_CONFIGURATION),
        )
        self.assertFalse(research.CANDIDATES["ridge_shadow"].production_eligible)


class MarketDataTests(unittest.TestCase):
    @staticmethod
    def raw_market() -> tuple[pd.DataFrame, str, str]:
        sessions = research._calendar_sessions(
            pd.Timestamp(research.MODEL_HISTORY_START),
            pd.Timestamp("2010-06-11"),
        )
        columns = pd.MultiIndex.from_product(
            [["Open", "Close", "Volume"], research.TICKERS]
        )
        raw = pd.DataFrame(index=sessions, columns=columns, dtype=float)
        positions = np.arange(len(sessions), dtype=float)
        for number, ticker in enumerate(research.TICKERS):
            close = (50.0 + number) * np.exp(0.001 * positions)
            raw[("Close", ticker)] = close
            raw[("Open", ticker)] = close * 0.999
            raw[("Volume", ticker)] = 1_000_000.0
        return (
            raw,
            sessions[0].date().isoformat(),
            (sessions[-1] + pd.Timedelta(days=1)).date().isoformat(),
        )

    def test_late_model_start_is_rejected_without_changing_family_identity(self):
        raw, _, _ = self.raw_market()
        late = raw.iloc[5:].copy()
        start = late.index[0].date().isoformat()
        end = (late.index[-1] + pd.Timedelta(days=1)).date().isoformat()
        fingerprint = research.FROZEN_FAMILY_FINGERPRINT
        with self.assertRaisesRegex(RuntimeError, "Frozen model history"):
            research.prepare_market_data(
                late,
                requested_start=start,
                requested_end_exclusive=end,
            )
        self.assertEqual(research.FROZEN_FAMILY_FINGERPRINT, fingerprint)

    def test_latest_session_waits_for_the_close_finalization_buffer(self):
        calendar = research.xcals.get_calendar("XNYS")
        session = pd.Timestamp("2024-03-28")
        close = calendar.session_close(session).tz_convert(research.NEW_YORK)
        self.assertEqual(
            research._expected_completed_session(
                close + pd.Timedelta(minutes=14)
            ),
            pd.Timestamp("2024-03-27"),
        )
        self.assertEqual(
            research._expected_completed_session(
                close
                + pd.Timedelta(
                    minutes=research.MARKET_CLOSE_BUFFER_MINUTES
                )
            ),
            session,
        )

    def test_missing_session_and_scored_nan_fail_closed(self):
        raw, start, end = self.raw_market()
        with self.assertRaisesRegex(RuntimeError, "session continuity"):
            research.prepare_market_data(
                raw.drop(index=raw.index[10]),
                requested_start=start,
                requested_end_exclusive=end,
            )
        invalid = raw.copy()
        invalid.loc[invalid.index[20], ("Close", "SOXL")] = np.nan
        with self.assertRaisesRegex(RuntimeError, "missing or invalid"):
            research.prepare_market_data(
                invalid,
                requested_start=start,
                requested_end_exclusive=end,
            )

    def test_atomic_snapshot_round_trip_preserves_fingerprint(self):
        raw, start, end = self.raw_market()
        data = research.prepare_market_data(
            raw,
            requested_start=start,
            requested_end_exclusive=end,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alpha.csv"
            research.save_market_snapshot(data, path)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            loaded = research.load_market_snapshot(
                path,
                requested_start=start,
                requested_end_exclusive=end,
            )
        self.assertEqual(loaded.fingerprint, data.fingerprint)
        pd.testing.assert_frame_equal(
            loaded.opens,
            data.opens,
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            loaded.closes,
            data.closes,
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            loaded.volumes,
            data.volumes,
            check_exact=True,
        )


class ResidualAndRidgeTests(unittest.TestCase):
    def test_residual_uses_strictly_separated_windows(self):
        count = research.ESTIMATION_RETURNS + research.SCORING_RETURNS
        positions = np.arange(count, dtype=float)
        qqq_returns = 0.0003 + 0.004 * np.sin(positions / 6.0)
        smh_returns = 0.0002 + 1.2 * qqq_returns
        smh_returns[: research.ESTIMATION_RETURNS] += (
            0.0005
            * np.cos(np.arange(research.ESTIMATION_RETURNS, dtype=float) / 4.0)
        )
        smh_returns[research.ESTIMATION_RETURNS :] += 0.001
        index = pd.bdate_range("2020-01-02", periods=count + 1)
        closes = pd.DataFrame(
            {
                "QQQ": 100.0 * np.exp(np.r_[0.0, np.cumsum(qqq_returns)]),
                "SMH": 80.0 * np.exp(np.r_[0.0, np.cumsum(smh_returns)]),
            },
            index=index,
        )
        estimate = research.residual_estimate_at(closes, index[-1])
        self.assertTrue(estimate.available)
        self.assertLess(estimate.estimation_end, estimate.scoring_start)
        self.assertEqual(estimate.scoring_end, index[-1])
        self.assertGreater(estimate.residual_momentum, 0)
        self.assertGreater(estimate.z_score, 0)

    def test_har_ridge_does_not_use_unrealized_labels(self):
        rng = np.random.default_rng(11)
        index = pd.bdate_range("2018-01-02", periods=940)
        returns = rng.normal(0.0004, 0.012, len(index) - 1)
        close = pd.Series(
            100.0 * np.exp(np.r_[0.0, np.cumsum(returns)]),
            index=index,
        )
        signal_date = index[900]
        baseline = research.har_forecast_at(
            research.har_design(close),
            signal_date,
            minimum_observations=700,
        )
        changed = close.copy()
        changed.loc[changed.index > signal_date] *= np.linspace(
            1.0,
            4.0,
            int((changed.index > signal_date).sum()),
        )
        repeated = research.har_forecast_at(
            research.har_design(changed),
            signal_date,
            minimum_observations=700,
        )
        self.assertTrue(baseline.available)
        self.assertEqual(baseline.training_cutoff, signal_date)
        self.assertEqual(baseline.last_label_origin, index[879])
        self.assertGreater(baseline.smearing_factor, 0)
        self.assertAlmostEqual(baseline.prediction, repeated.prediction, places=14)
        self.assertEqual(baseline.sample_count, repeated.sample_count)

    def test_ridge_shadow_cutoff_and_metadata_are_causal(self):
        index = pd.bdate_range("2019-01-02", periods=850)
        x = np.arange(len(index), dtype=float)
        design = pd.DataFrame(index=index)
        for number, name in enumerate(research.RIDGE_FEATURES, start=1):
            design[name] = (
                np.sin(x / (7.0 + number))
                + number * np.cos(x / (11.0 + number))
            )
        design["ridge_relative_log_return_21"] = (
            0.001 * design[research.RIDGE_FEATURES[0]]
            - 0.0005 * design[research.RIDGE_FEATURES[1]]
        )
        estimate = research.ridge_shadow_estimate_at(
            design,
            index[-1],
            minimum_observations=700,
        )
        self.assertTrue(estimate.available)
        self.assertEqual(estimate.training_cutoff, index[-1])
        self.assertEqual(estimate.last_label_origin, index[-23])
        self.assertEqual(estimate.sample_count, len(index) - 22)
        self.assertEqual(estimate.feature_names, research.RIDGE_FEATURES)
        self.assertEqual(len(estimate.training_fingerprint), 64)
        self.assertEqual(
            estimate.challenger_fingerprint,
            research.RIDGE_CHALLENGER_FINGERPRINT,
        )

    def test_open_to_open_label_has_exact_21_holding_intervals(self):
        index = pd.bdate_range("2025-01-02", periods=40)
        qld = 100.0 * np.exp(0.01 * np.arange(40))
        soxl = 50.0 * np.exp(0.02 * np.arange(40))
        frame = research.ridge_relative_label_frame(
            pd.DataFrame({"QLD": qld, "SOXL": soxl}, index=index)
        )
        qld_growth = math.exp(0.01 * 21)
        soxl_growth = math.exp(0.02 * 21)
        expected = math.log(0.65 * qld_growth + 0.35 * soxl_growth) - math.log(
            qld_growth
        )
        self.assertAlmostEqual(
            frame.iloc[0]["ridge_relative_log_return_21"],
            expected,
            places=14,
        )
        self.assertEqual(frame.iloc[0]["ridge_label_entry_date"], index[1])
        self.assertEqual(frame.iloc[0]["ridge_label_exit_date"], index[22])


class VolatilityAndStateTests(unittest.TestCase):
    def test_weight_grid_uses_shared_audited_equation_and_fails_closed(self):
        weight = research.choose_volatility_weight(0.40, 1.20, 0.80)
        audited, _ = research.core.choose_soxl_weight(0.40, 1.20, 0.80)
        self.assertEqual(weight, audited)
        self.assertEqual(
            research.choose_volatility_weight(np.nan, 1.0, 0.8),
            0.0,
        )
        self.assertEqual(
            research.choose_volatility_weight(0.60, 0.90, 0.8),
            0.0,
        )

    def test_schedule_reentry_upshift_and_trend_exit_are_stateful(self):
        data = alpha_data(periods=35)
        panel = signal_panel(data, raw_weight=0.10)
        schedule = research.build_target_schedule(panel, "residual_vol55")
        # The first alpha review has only one distinct eligible close.  The
        # next 21-session review enters at the fresh 10% scale.
        self.assertEqual(schedule.iloc[0]["strategic_soxl_weight"], 0.0)
        self.assertEqual(schedule.iloc[21]["strategic_soxl_weight"], 0.10)
        panel.frame.loc[panel.frame.index[22:27], "raw_soxl_weight"] = 0.30
        schedule = research.build_target_schedule(panel, "residual_vol55")
        self.assertEqual(schedule.iloc[25]["strategic_soxl_weight"], 0.10)
        self.assertEqual(schedule.iloc[26]["strategic_soxl_weight"], 0.30)
        panel.frame.loc[panel.frame.index[27], "trend_positive"] = False
        exited = research.build_target_schedule(panel, "residual_vol55")
        self.assertEqual(exited.iloc[27]["strategic_soxl_weight"], 0.0)
        self.assertIn("TREND_EXIT", exited.iloc[27]["decision_reason"])

    def test_invalid_residual_exits_active_overlay_immediately(self):
        data = alpha_data(periods=35)
        panel = signal_panel(data, raw_weight=0.10)
        panel.frame.loc[
            panel.frame.index[22],
            "residual_available",
        ] = False
        panel.frame.loc[
            panel.frame.index[22],
            "residual_positive",
        ] = False
        schedule = research.build_target_schedule(
            panel,
            "residual_vol55",
        )
        self.assertEqual(schedule.iloc[21]["strategic_soxl_weight"], 0.10)
        self.assertEqual(schedule.iloc[22]["strategic_soxl_weight"], 0.0)
        self.assertIn("TREND_EXIT", schedule.iloc[22]["decision_reason"])


class ExecutionAndDiagnosticsTests(unittest.TestCase):
    def test_drawdown_includes_initial_deployment_loss(self):
        index = pd.bdate_range("2026-08-04", periods=2)
        diagnostics = research._drawdown_diagnostics(
            pd.Series([50.0, 60.0], index=index),
            starting_capital=100.0,
            starting_date=pd.Timestamp("2026-08-03"),
        )
        self.assertEqual(diagnostics["maximum_drawdown"], -0.5)
        self.assertEqual(
            diagnostics["maximum_drawdown_peak"],
            "2026-08-03",
        )

    def test_next_open_fractional_fill_and_initial_turnover_exclusion(self):
        data = alpha_data(periods=15)
        panel = signal_panel(data)
        start = data.sessions[2]
        result = research.simulate_candidate(
            data,
            panel,
            "qld_buy_hold",
            cost_bps=10.0,
            scoring_start=start,
        )
        first = result.ledger.iloc[0]
        expected_post_cost_nav = research.STARTING_CASH / 1.001
        expected_shares = expected_post_cost_nav / data.opens.loc[start, "QLD"]
        self.assertAlmostEqual(first["qld_shares"], expected_shares)
        self.assertTrue(bool(first["initial_deployment"]))
        self.assertEqual(first["ongoing_gross_trade_fraction"], 0.0)
        self.assertEqual(result.metrics["annual_gross_turnover"], 0.0)
        self.assertGreater(result.metrics["total_modeled_cost"], 0.0)

    def test_exact_drift_threshold_and_destination(self):
        self.assertAlmostEqual(
            research._buffered_drift_target(0.25, 0.20),
            0.225,
        )
        self.assertIsNone(research._buffered_drift_target(0.249, 0.20))
        self.assertAlmostEqual(
            research._buffered_drift_target(0.15, 0.20),
            0.175,
        )
        self.assertAlmostEqual(
            research._buffered_drift_target(0.40, 0.35),
            0.35,
        )

    def test_bootstrap_and_style_attribution_are_deterministic(self):
        data = alpha_data(periods=90)
        panel = signal_panel(data)
        qld = research.simulate_candidate(
            data,
            panel,
            "qld_buy_hold",
            cost_bps=0.0,
            scoring_start=data.sessions[2],
        )
        static = research.simulate_candidate(
            data,
            panel,
            "static_65_35",
            cost_bps=0.0,
            scoring_start=data.sessions[2],
        )
        first = research.moving_block_bootstrap(
            static,
            qld,
            samples=200,
            block_length=5,
            seed=17,
        )
        second = research.moving_block_bootstrap(
            static,
            qld,
            samples=200,
            block_length=5,
            seed=17,
        )
        self.assertEqual(first, second)
        style = research.rolling_style_attribution(
            static,
            data,
            window=40,
            step=10,
        )
        self.assertGreater(len(style), 0)
        weights = style.iloc[-1][
            ["qld_exposure", "soxl_exposure", "spy_exposure"]
        ].sum()
        self.assertAlmostEqual(weights, 1.0, places=10)
        self.assertTrue((style["r_squared"] <= 1.0 + 1e-10).all())


if __name__ == "__main__":
    unittest.main()
