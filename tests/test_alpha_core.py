from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

import alpha_core as alpha


def closes_from_returns(
    qqq_returns: np.ndarray,
    smh_returns: np.ndarray,
) -> pd.DataFrame:
    if len(qqq_returns) != len(smh_returns):
        raise ValueError("Return arrays must have equal length")
    index = pd.bdate_range("2020-01-02", periods=len(qqq_returns) + 1)
    qqq = 100.0 * np.exp(np.r_[0.0, np.cumsum(qqq_returns)])
    smh = 80.0 * np.exp(np.r_[0.0, np.cumsum(smh_returns)])
    return pd.DataFrame({alpha.QQQ: qqq, alpha.SMH: smh}, index=index)


class ResidualSignalTests(unittest.TestCase):
    def test_separated_ols_and_residual_score_match_hand_calculation(self):
        count = alpha.BETA_WINDOW + alpha.RESIDUAL_SCORE_WINDOW
        positions = np.arange(count, dtype=float)
        qqq_returns = 0.0004 + 0.004 * np.sin(positions / 7.0)
        estimation_noise = 0.0007 * np.cos(
            np.arange(alpha.BETA_WINDOW, dtype=float) / 3.0
        )
        smh_returns = np.empty(count, dtype=float)
        smh_returns[: alpha.BETA_WINDOW] = (
            0.0002
            + 1.25 * qqq_returns[: alpha.BETA_WINDOW]
            + estimation_noise
        )

        x = qqq_returns[: alpha.BETA_WINDOW]
        y = smh_returns[: alpha.BETA_WINDOW]
        beta = float(((x - x.mean()) @ (y - y.mean())) / ((x - x.mean()) @ (x - x.mean())))
        intercept = float(y.mean() - beta * x.mean())
        fit_residuals = y - (intercept + beta * x)
        sigma = float(
            np.sqrt(
                (fit_residuals @ fit_residuals)
                / (alpha.BETA_WINDOW - 2)
            )
        )
        smh_returns[alpha.BETA_WINDOW :] = (
            intercept
            + beta * qqq_returns[alpha.BETA_WINDOW :]
            + 0.001
        )
        prices = closes_from_returns(qqq_returns, smh_returns)

        signal = alpha.calculate_residual_signal(prices)

        expected_momentum = 0.001 * alpha.RESIDUAL_SCORE_WINDOW
        expected_z = expected_momentum / (
            sigma * np.sqrt(alpha.RESIDUAL_SCORE_WINDOW)
        )
        self.assertAlmostEqual(signal.beta, beta, places=12)
        self.assertAlmostEqual(signal.intercept, intercept, places=12)
        self.assertAlmostEqual(signal.residual_sigma, sigma, places=12)
        self.assertAlmostEqual(
            signal.residual_momentum,
            expected_momentum,
            places=12,
        )
        self.assertAlmostEqual(signal.residual_z, expected_z, places=12)
        self.assertLess(signal.estimation_end, signal.scoring_start)
        self.assertEqual(
            len(prices.loc[signal.scoring_start : signal.scoring_end]) - 1,
            alpha.RESIDUAL_SCORE_WINDOW - 1,
        )

    def test_singular_qqq_window_fails_closed(self):
        count = alpha.BETA_WINDOW + alpha.RESIDUAL_SCORE_WINDOW
        qqq_returns = np.full(count, 0.001)
        smh_returns = np.linspace(-0.002, 0.003, count)
        with self.assertRaisesRegex(ValueError, "QQQ estimation variance"):
            alpha.calculate_residual_signal(
                closes_from_returns(qqq_returns, smh_returns)
            )

    def test_future_prices_do_not_change_prior_signal(self):
        rng = np.random.default_rng(20260806)
        count = alpha.BETA_WINDOW + alpha.RESIDUAL_SCORE_WINDOW + 20
        qqq_returns = rng.normal(0.0004, 0.01, count)
        smh_returns = 0.0001 + 1.1 * qqq_returns + rng.normal(0, 0.004, count)
        full = closes_from_returns(qqq_returns, smh_returns)
        cutoff = full.index[-11]
        baseline = alpha.calculate_residual_signal_frame(full).loc[cutoff]
        changed = full.copy()
        changed.loc[changed.index > cutoff, alpha.QQQ] *= 20.0
        changed.loc[changed.index > cutoff, alpha.SMH] *= 0.05
        repeated = alpha.calculate_residual_signal_frame(changed).loc[cutoff]
        pd.testing.assert_series_equal(baseline, repeated)


class VarianceModelTests(unittest.TestCase):
    def test_future_variance_label_uses_exact_next_21_returns(self):
        index = pd.bdate_range("2022-01-03", periods=120)
        returns = np.linspace(-0.015, 0.02, len(index) - 1)
        close = pd.Series(
            100.0 * np.exp(np.r_[0.0, np.cumsum(returns)]),
            index=index,
        )
        frame = alpha.build_variance_learning_frame(close)
        position = 70
        expected = np.log(
            np.mean(np.square(returns[position : position + 21])) * 252.0
        )
        self.assertAlmostEqual(
            float(frame.iloc[position]["future_log_var_21"]),
            expected,
            places=12,
        )
        self.assertTrue(
            frame["future_log_var_21"].iloc[-alpha.VARIANCE_HORIZON :].isna().all()
        )

    def test_ridge_standardizes_training_only_and_is_deterministic(self):
        index = pd.bdate_range("2024-01-02", periods=20)
        x = np.arange(20, dtype=float)
        features = pd.DataFrame({"x": x}, index=index)
        target = pd.Series(1.0 + 2.0 * x, index=index)
        current = pd.Series(
            {"x": 1_000_000.0},
            name=pd.Timestamp("2024-02-01"),
        )
        first = alpha.standardized_ridge_fit_predict(
            features,
            target,
            current,
            alpha=1e-9,
            min_samples=10,
        )
        second = alpha.standardized_ridge_fit_predict(
            features,
            target,
            current,
            alpha=1e-9,
            min_samples=10,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.feature_means[0], x.mean(), places=12)
        self.assertAlmostEqual(
            first.feature_scales[0],
            x.std(ddof=1),
            places=12,
        )
        self.assertAlmostEqual(first.prediction, 2_000_001.0, places=3)
        self.assertAlmostEqual(first.smearing_factor, 1.0, places=12)

    def test_reserved_target_feature_is_rejected(self):
        index = pd.bdate_range("2024-01-02", periods=20)
        features = pd.DataFrame({"_target": np.arange(20.0)}, index=index)
        target = pd.Series(np.arange(20.0), index=index)
        with self.assertRaisesRegex(ValueError, "reserved"):
            alpha.standardized_ridge_fit_predict(
                features,
                target,
                pd.Series(
                    {"_target": 1.0},
                    name=pd.Timestamp("2024-02-01"),
                ),
                min_samples=10,
            )

    def test_ridge_requires_explicit_information_cutoff(self):
        index = pd.bdate_range("2024-01-02", periods=20)
        features = pd.DataFrame({"x": np.arange(20.0)}, index=index)
        target = pd.Series(np.arange(20.0), index=index)
        with self.assertRaisesRegex(ValueError, "information cutoff"):
            alpha.standardized_ridge_fit_predict(
                features,
                target,
                pd.Series({"x": 1.0}),
                min_samples=10,
            )

    def test_forecast_uses_only_fully_realized_labels(self):
        rng = np.random.default_rng(7)
        index = pd.bdate_range("2019-01-02", periods=950)
        returns = rng.normal(0.0004, 0.012, len(index) - 1)
        close = pd.Series(
            100.0 * np.exp(np.r_[0.0, np.cumsum(returns)]),
            index=index,
        )
        forecast = alpha.forecast_daily_bar_volatility(
            close,
            min_samples=700,
        )
        self.assertEqual(
            forecast.ridge_fit.training_end,
            index[-1],
        )
        self.assertEqual(
            forecast.ridge_fit.last_label_origin,
            index[-alpha.VARIANCE_HORIZON - 1],
        )
        self.assertGreater(forecast.model_volatility, 0)
        self.assertGreaterEqual(
            forecast.sizing_volatility,
            forecast.model_volatility,
        )
        self.assertGreaterEqual(
            forecast.sizing_volatility,
            forecast.trailing_volatility_21,
        )
        self.assertGreaterEqual(
            forecast.sizing_volatility,
            forecast.trailing_volatility_63,
        )


class VolatilitySizingTests(unittest.TestCase):
    def test_soxl_allocations_are_only_the_four_frozen_tiers(self):
        self.assertEqual(alpha.SOXL_WEIGHT_GRID, (0.0, 0.15))
        self.assertEqual(alpha.floor_soxl_tier(0.20), 0.15)
        self.assertEqual(alpha.floor_soxl_tier(0.30), 0.15)
        with self.assertRaisesRegex(ValueError, "strategic tier"):
            alpha.target_weights(0.20)

    def test_each_soxl_tier_is_reachable_from_the_volatility_budget(self):
        cases = ((0.50, 1.00, 0.0), (0.30, 1.50, 0.15))
        for qld_volatility, soxl_volatility, expected in cases:
            with self.subTest(expected=expected):
                selected, _ = alpha.choose_soxl_weight(
                    qld_volatility,
                    soxl_volatility,
                    0.80,
                )
                self.assertEqual(selected, expected)

    def test_correlation_uses_the_maximum_available_finite_window(self):
        self.assertEqual(
            alpha.conservative_finite_correlation(np.nan, 0.82),
            0.82,
        )
        self.assertEqual(
            alpha.conservative_finite_correlation(0.91, np.nan),
            0.91,
        )
        with self.assertRaisesRegex(ValueError, "No finite"):
            alpha.conservative_finite_correlation(np.nan, np.nan)

    def test_exact_budget_boundary_selects_maximum_weight(self):
        weight, volatility = alpha.choose_soxl_weight(
            0.50,
            0.50,
            1.0,
            budget=0.50,
        )
        self.assertEqual(weight, 0.15)
        self.assertAlmostEqual(volatility, 0.50, places=12)
        self.assertAlmostEqual(
            alpha.advertised_daily_exposure(weight),
            alpha.MAX_ADVERTISED_DAILY_EXPOSURE,
            places=12,
        )

    def test_qld_above_budget_keeps_permanent_core(self):
        weight, volatility = alpha.choose_soxl_weight(
            0.60,
            0.90,
            0.8,
            budget=0.55,
        )
        self.assertEqual(weight, 0.0)
        self.assertAlmostEqual(volatility, 0.60, places=12)
        self.assertEqual(
            alpha.target_weights(weight),
            {"TQQQ": 0.40, "DBMF": 0.20, "ZROZ": 0.20, "GLD": 0.20, "SOXL": 0.0},
        )

    def test_intermediate_grid_weight_is_largest_feasible(self):
        weight, volatility = alpha.choose_soxl_weight(
            0.40,
            1.20,
            0.80,
            budget=0.55,
        )
        all_vols = {
            candidate: alpha.forecast_portfolio_volatility(
                0.40,
                1.20,
                0.80,
                candidate,
            )
            for candidate in alpha.SOXL_WEIGHT_GRID
        }
        expected = max(
            candidate
            for candidate, candidate_vol in all_vols.items()
            if candidate_vol <= 0.55 + 1e-12
        )
        self.assertEqual(weight, expected)
        self.assertAlmostEqual(volatility, all_vols[expected], places=12)
        higher = [
            candidate_vol
            for candidate, candidate_vol in all_vols.items()
            if candidate > weight
        ]
        self.assertTrue(all(item > 0.55 for item in higher))

    def test_invalid_or_zero_volatility_fails_closed(self):
        for invalid in (0.0, np.nan, np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    alpha.choose_soxl_weight(invalid, 0.8, 0.8)


class OverlayTransitionTests(unittest.TestCase):
    def test_bearish_review_advances_the_completed_session_clock(self):
        transition = alpha.advance_overlay_state(
            alpha.OverlayState(last_alpha_review_date="2026-07-01"),
            signal_date=pd.Timestamp("2026-08-04"),
            trend_positive=False,
            residual_positive=True,
            raw_soxl_weight=0.0,
            alpha_review_due=True,
        )
        self.assertTrue(transition.alpha_reviewed)
        self.assertEqual(
            transition.state.last_alpha_review_date,
            "2026-08-04",
        )

    def test_two_distinct_eligible_closes_reenter_and_duplicate_is_idempotent(self):
        state = alpha.OverlayState()
        first = alpha.advance_overlay_state(
            state,
            signal_date=pd.Timestamp("2026-08-03"),
            trend_positive=True,
            residual_positive=True,
            raw_soxl_weight=0.15,
            alpha_review_due=False,
        )
        self.assertFalse(first.state.overlay_active)
        self.assertEqual(first.state.eligible_streak, 1)

        second = alpha.advance_overlay_state(
            first.state,
            signal_date=pd.Timestamp("2026-08-04"),
            trend_positive=True,
            residual_positive=True,
            raw_soxl_weight=0.15,
            alpha_review_due=True,
        )
        self.assertTrue(second.state.overlay_active)
        self.assertEqual(second.state.soxl_weight, 0.15)
        self.assertEqual(second.reason, "OVERLAY_REENTRY")

        duplicate = alpha.advance_overlay_state(
            second.state,
            signal_date=pd.Timestamp("2026-08-04"),
            trend_positive=True,
            residual_positive=True,
            raw_soxl_weight=0.15,
            alpha_review_due=True,
        )
        self.assertEqual(duplicate.state, second.state)
        self.assertEqual(duplicate.reason, "SAME_DATE")

    def test_trend_exit_and_volatility_downshift_are_immediate(self):
        active = alpha.OverlayState(
            overlay_active=True,
            eligible_streak=10,
            soxl_weight=0.15,
            soxl_weight_date="2026-08-01",
            last_processed_signal_date="2026-08-03",
        )
        downshift = alpha.advance_overlay_state(
            active,
            signal_date=pd.Timestamp("2026-08-04"),
            trend_positive=True,
            residual_positive=True,
            raw_soxl_weight=0.15,
            alpha_review_due=False,
        )
        self.assertEqual(downshift.state.soxl_weight, 0.15)
        self.assertEqual(downshift.reason, "HOLD")

        exited = alpha.advance_overlay_state(
            downshift.state,
            signal_date=pd.Timestamp("2026-08-05"),
            trend_positive=False,
            residual_positive=True,
            raw_soxl_weight=0.15,
            alpha_review_due=False,
        )
        self.assertFalse(exited.state.overlay_active)
        self.assertEqual(exited.state.soxl_weight, 0.0)
        self.assertEqual(exited.reason, "TREND_EXIT")

    def test_volatility_upshift_requires_five_distinct_closes(self):
        state = alpha.OverlayState(
            overlay_active=True,
            eligible_streak=5,
            soxl_weight=0.15,
            last_processed_signal_date="2026-08-02",
        )
        transition = alpha.advance_overlay_state(
            state,
            signal_date=pd.Timestamp("2026-08-03"),
            trend_positive=True,
            residual_positive=True,
            raw_soxl_weight=0.15,
            alpha_review_due=False,
        )
        self.assertEqual(transition.state.soxl_weight, 0.15)
        self.assertEqual(transition.state.pending_scale_days, 0)
        self.assertEqual(transition.reason, "HOLD")

    def test_residual_exit_waits_for_alpha_review(self):
        active = alpha.OverlayState(
            overlay_active=True,
            eligible_streak=4,
            soxl_weight=0.15,
            last_processed_signal_date="2026-08-03",
        )
        held = alpha.advance_overlay_state(
            active,
            signal_date=pd.Timestamp("2026-08-04"),
            trend_positive=True,
            residual_positive=False,
            raw_soxl_weight=0.15,
            alpha_review_due=False,
        )
        self.assertTrue(held.state.overlay_active)
        reviewed = alpha.advance_overlay_state(
            held.state,
            signal_date=pd.Timestamp("2026-08-05"),
            trend_positive=True,
            residual_positive=False,
            raw_soxl_weight=0.15,
            alpha_review_due=True,
        )
        self.assertFalse(reviewed.state.overlay_active)
        self.assertEqual(reviewed.state.soxl_weight, 0.0)
        self.assertEqual(reviewed.reason, "RESIDUAL_EXIT")

    def test_non_boolean_signal_inputs_fail_closed(self):
        state = alpha.OverlayState()
        for field_name, trend, residual in (
            ("trend_positive", np.nan, True),
            ("residual_positive", True, np.nan),
            ("trend_positive", "yes", True),
            ("residual_positive", True, 1),
        ):
            with self.subTest(field_name=field_name, value=(trend, residual)):
                with self.assertRaisesRegex(ValueError, field_name):
                    alpha.advance_overlay_state(
                        state,
                        signal_date=pd.Timestamp("2026-08-03"),
                        trend_positive=trend,
                        residual_positive=residual,
                        raw_soxl_weight=0.15,
                        alpha_review_due=True,
                    )
        with self.assertRaisesRegex(ValueError, "alpha_review_due"):
            alpha.advance_overlay_state(
                state,
                signal_date=pd.Timestamp("2026-08-03"),
                trend_positive=True,
                residual_positive=True,
                raw_soxl_weight=0.15,
                alpha_review_due=np.nan,
            )


if __name__ == "__main__":
    unittest.main()
