from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import port12_cloud as engine
import research_backtest as research


def synthetic_raw_market_data(
    *,
    sessions_count: int = 285,
    common_position: int = 220,
) -> tuple[pd.DataFrame, str, str]:
    calendar = engine._nyse_calendar()
    available = pd.DatetimeIndex(
        calendar.sessions_in_range("2023-01-03", "2025-12-31")
    )
    if available.tz is not None:
        available = available.tz_convert(None)
    sessions = available.normalize()[:sessions_count]
    tickers = list(research.RESEARCH_TICKERS)
    columns = pd.MultiIndex.from_product(
        [["Open", "Close", "Volume"], tickers]
    )
    raw = pd.DataFrame(index=sessions, columns=columns, dtype=float)

    positions = np.arange(len(sessions), dtype=float)
    qqq_returns = 0.0007 + 0.0025 * np.sin(positions / 4.0)
    base_qqq = 100.0 * np.cumprod(1.0 + qqq_returns)
    for number, ticker in enumerate(tickers):
        ticker_returns = (
            0.0004
            + (0.0015 + number * 0.00003) * np.sin(positions / (5.0 + number))
        )
        close = (
            base_qqq
            if ticker == engine.MARKET_INDEX
            else (50.0 + number * 3.0) * np.cumprod(1.0 + ticker_returns)
        )
        raw[("Close", ticker)] = close
        raw[("Open", ticker)] = close * (
            1.0 + 0.0008 * np.cos(positions / (3.0 + number))
        )
        raw[("Volume", ticker)] = 1_000_000.0 + number * 10_000.0

    # SPMO is the binding actual inception, leaving genuine QQQ/leader warm-up.
    for field_name in ("Open", "Close", "Volume"):
        raw.loc[sessions[:common_position], (field_name, engine.DEFENSIVE_EQUITY)] = np.nan

    requested_start = sessions[0].date().isoformat()
    requested_end = (sessions[-1] + pd.Timedelta(days=1)).date().isoformat()
    return raw, requested_start, requested_end


def prepared_synthetic_data() -> research.MarketData:
    raw, start, end = synthetic_raw_market_data()
    return research.prepare_market_data(
        raw,
        requested_start=start,
        requested_end_exclusive=end,
    )


class ResearchDataTests(unittest.TestCase):
    def test_common_actual_inception_preserves_real_warmup_without_filling(self):
        raw, start, end = synthetic_raw_market_data()
        data = research.prepare_market_data(
            raw,
            requested_start=start,
            requested_end_exclusive=end,
        )
        self.assertEqual(data.common_start, raw.index[220])
        self.assertEqual(data.scoring_dates[0], raw.index[220])
        self.assertTrue(
            data.closes[engine.DEFENSIVE_EQUITY].iloc[:220].isna().all()
        )
        self.assertEqual(len(data.fingerprint), 64)

    def test_missing_session_and_scored_nan_fail_closed(self):
        raw, start, end = synthetic_raw_market_data()
        missing = raw.drop(index=raw.index[100])
        with self.assertRaisesRegex(RuntimeError, "session continuity"):
            research.prepare_market_data(
                missing,
                requested_start=start,
                requested_end_exclusive=end,
            )

        invalid = raw.copy()
        invalid.loc[
            invalid.index[240],
            ("Open", engine.SEMICONDUCTOR_ETF),
        ] = np.nan
        with self.assertRaisesRegex(RuntimeError, "Scored adjusted opens"):
            research.prepare_market_data(
                invalid,
                requested_start=start,
                requested_end_exclusive=end,
            )

    def test_fingerprint_changes_when_historical_input_changes(self):
        raw, start, end = synthetic_raw_market_data()
        original = research.prepare_market_data(
            raw,
            requested_start=start,
            requested_end_exclusive=end,
        )
        changed = raw.copy()
        changed.loc[
            changed.index[50],
            ("Close", engine.MARKET_INDEX),
        ] += 0.01
        revised = research.prepare_market_data(
            changed,
            requested_start=start,
            requested_end_exclusive=end,
        )
        self.assertNotEqual(original.fingerprint, revised.fingerprint)

    def test_snapshot_round_trip_preserves_fingerprint_and_values(self):
        data = prepared_synthetic_data()
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot.csv"
            research.save_market_snapshot(data, snapshot)
            loaded = research.load_market_snapshot(
                snapshot,
                requested_start=data.requested_start,
                requested_end_exclusive=data.requested_end_exclusive,
            )
        self.assertEqual(data.fingerprint, loaded.fingerprint)
        pd.testing.assert_frame_equal(data.opens, loaded.opens)
        pd.testing.assert_frame_equal(data.closes, loaded.closes)
        pd.testing.assert_frame_equal(data.volumes, loaded.volumes)
        self.assertIn("frozen local snapshot", loaded.source)

    def test_current_partial_session_is_rejected_before_data_use(self):
        raw, start, end = synthetic_raw_market_data()
        today = pd.Timestamp.now(tz=engine.NEW_YORK).tz_localize(None).normalize()
        with (
            mock.patch.object(research, "_last_session_before", return_value=today),
            mock.patch.object(
                engine,
                "expected_completed_session",
                side_effect=RuntimeError("daily bar is not final"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "daily bar is not final"):
                research.prepare_market_data(
                    raw,
                    requested_start=start,
                    requested_end_exclusive=end,
                )


class CostAccountingTests(unittest.TestCase):
    def test_cash_to_stock_and_full_rotation_solve_post_cost_nav(self):
        cash_fill = research.execute_target_at_open(
            shares={},
            cash=1_000.0,
            open_prices=pd.Series({"A": 100.0}),
            target_weights={"A": 1.0, engine.CASH_ASSET: 0.0},
            cost_bps=10.0,
        )
        self.assertAlmostEqual(cash_fill.posttrade_nav, 1_000.0 / 1.001)
        self.assertAlmostEqual(cash_fill.shares["A"], cash_fill.posttrade_nav / 100.0)
        self.assertAlmostEqual(cash_fill.gross_fraction, cash_fill.posttrade_nav / 1_000.0)
        self.assertAlmostEqual(cash_fill.one_way_fraction, 1.0)
        self.assertAlmostEqual(
            cash_fill.posttrade_nav + cash_fill.cost,
            cash_fill.pretrade_nav,
        )

        rotation = research.execute_target_at_open(
            shares={"A": 10.0},
            cash=0.0,
            open_prices=pd.Series({"A": 100.0, "B": 100.0}),
            target_weights={"B": 1.0, engine.CASH_ASSET: 0.0},
            cost_bps=10.0,
        )
        self.assertAlmostEqual(rotation.posttrade_nav, 999.0 / 1.001)
        self.assertAlmostEqual(rotation.gross_notional, 1_000.0 + rotation.posttrade_nav)
        self.assertAlmostEqual(rotation.gross_fraction, rotation.gross_notional / 1_000.0)
        self.assertAlmostEqual(rotation.one_way_fraction, 1.0)
        self.assertNotIn("A", rotation.shares)
        self.assertAlmostEqual(rotation.shares["B"], rotation.posttrade_nav / 100.0)

    def test_invalid_execution_open_fails(self):
        with self.assertRaisesRegex(RuntimeError, "Invalid execution Open"):
            research.execute_target_at_open(
                shares={},
                cash=1_000.0,
                open_prices=pd.Series({"A": np.nan}),
                target_weights={"A": 1.0, engine.CASH_ASSET: 0.0},
                cost_bps=10.0,
            )


class ResearchSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = prepared_synthetic_data()

    def test_buy_and_hold_excludes_initial_deployment_from_turnover(self):
        result = research.simulate_variant(
            self.data,
            research.VARIANTS["benchmark_qld"],
            cost_bps=10.0,
        )
        fills = result.ledger[result.ledger["security_orders"] > 0]
        self.assertEqual(len(fills), 1)
        self.assertTrue(bool(fills.iloc[0]["initial_deployment"]))
        self.assertGreater(result.metrics["total_modeled_cost"], 0.0)
        self.assertAlmostEqual(result.metrics["annual_gross_turnover"], 0.0)
        self.assertAlmostEqual(result.metrics["annual_one_way_turnover"], 0.0)

    def test_signal_is_causal_and_fill_uses_only_next_open(self):
        baseline = research.simulate_variant(
            self.data,
            research.VARIANTS["original"],
            cost_bps=0.0,
        )
        first_signal = self.data.scoring_dates[0]
        next_session = self.data.scoring_dates[1]

        open_mutated = copy.deepcopy(self.data)
        open_mutated.opens.loc[next_session, :] *= 1.17
        changed_open = research.simulate_variant(
            open_mutated,
            research.VARIANTS["original"],
            cost_bps=0.0,
        )

        for column in (
            "regime",
            "volatility_tier",
            "leader",
            "signal_reason",
            "queued_target",
        ):
            self.assertEqual(
                baseline.ledger.loc[first_signal, column],
                changed_open.ledger.loc[first_signal, column],
            )
        self.assertNotEqual(
            baseline.ledger.loc[next_session, "shares"],
            changed_open.ledger.loc[next_session, "shares"],
        )

        close_mutated = copy.deepcopy(self.data)
        close_mutated.closes.loc[next_session, :] *= 0.83
        changed_close = research.simulate_variant(
            close_mutated,
            research.VARIANTS["original"],
            cost_bps=0.0,
        )
        self.assertEqual(
            baseline.ledger.loc[next_session, "shares"],
            changed_close.ledger.loc[next_session, "shares"],
        )
        self.assertNotEqual(
            baseline.ledger.loc[next_session, "nav"],
            changed_close.ledger.loc[next_session, "nav"],
        )

        queued = json.loads(baseline.ledger.loc[first_signal, "queued_target"])
        filled_shares = json.loads(baseline.ledger.loc[next_session, "shares"])
        ticker = next(item for item in queued if item != engine.CASH_ASSET)
        self.assertAlmostEqual(
            filled_shares[ticker],
            queued[ticker] * research.STARTING_CASH
            / self.data.opens.loc[next_session, ticker],
        )

    def test_research_path_never_calls_production_io_or_notifications(self):
        with (
            mock.patch.object(
                engine,
                "load_state",
                side_effect=AssertionError("production load"),
            ),
            mock.patch.object(
                engine,
                "save_state",
                side_effect=AssertionError("production save"),
            ),
            mock.patch.object(
                engine,
                "send_email",
                side_effect=AssertionError("email"),
            ),
        ):
            result = research.simulate_variant(
                self.data,
                research.VARIANTS["original"],
                cost_bps=10.0,
            )
        self.assertGreater(len(result.ledger), 1)

    def test_variant_order_has_no_mutable_state_leakage(self):
        original_first = research.simulate_variant(
            self.data,
            research.VARIANTS["original"],
            cost_bps=10.0,
        )
        research.simulate_variant(
            self.data,
            research.VARIANTS["c1_wide_buffer"],
            cost_bps=10.0,
        )
        original_second = research.simulate_variant(
            self.data,
            research.VARIANTS["original"],
            cost_bps=10.0,
        )
        pd.testing.assert_series_equal(
            original_first.ledger["nav"],
            original_second.ledger["nav"],
        )
        self.assertEqual(original_first.metrics, original_second.metrics)

    def test_registry_name_cannot_mask_modified_parameters(self):
        altered = research.replace(
            research.VARIANTS["original"],
            rebalance_band=0.20,
        )
        with self.assertRaisesRegex(ValueError, "locked registry definition"):
            research.simulate_variant(
                self.data,
                altered,
                cost_bps=10.0,
            )

    def test_first_signal_plan_matches_direct_production_oracle(self):
        signal_date = self.data.scoring_dates[0]
        position = self.data.closes.index.get_loc(signal_date)
        indicators = engine.calculate_indicators(
            self.data.closes,
            self.data.volumes,
        )
        latest = indicators.loc[signal_date]
        state = engine.PortfolioState(cash_balance=research.STARTING_CASH)
        tier = engine.replay_volatility_state(
            indicators.iloc[: position + 1],
            state,
        )
        sector_due = engine.sector_review_due(
            state.last_sector_rebalance,
            signal_date,
            self.data.closes.index[: position + 1],
        )
        expected_result = engine.determine_target_allocation(
            latest,
            state.leader,
            sector_due,
            tier,
        )
        expected_plan = engine.build_rebalance_plan(
            {engine.CASH_ASSET: 1.0},
            expected_result,
            state,
        )

        simulated = research.simulate_variant(
            self.data,
            research.VARIANTS["original"],
            cost_bps=0.0,
        )
        row = simulated.ledger.loc[signal_date]
        self.assertEqual(row["regime"], expected_result.regime)
        self.assertEqual(row["volatility_tier"], expected_result.volatility_tier)
        self.assertEqual(row["leader"], expected_result.leader)
        self.assertEqual(row["signal_reason"], expected_plan.reason)
        self.assertEqual(
            json.loads(row["queued_target"]),
            expected_plan.execution_weights,
        )

    def test_missing_trade_liquidity_is_explicitly_fail_closed(self):
        raw, start, end = synthetic_raw_market_data()
        raw.loc[
            raw.index[200:221],
            ("Volume", engine.LEVERAGED_INDEX),
        ] = 0.0
        data = research.prepare_market_data(
            raw,
            requested_start=start,
            requested_end_exclusive=end,
        )
        result = research.simulate_variant(
            data,
            research.VARIANTS["original"],
            cost_bps=10.0,
        )
        self.assertGreaterEqual(
            result.metrics["unmeasurable_liquidity_trade_sessions"],
            1,
        )
        self.assertTrue(
            np.isinf(
                result.metrics[
                    "maximum_trade_to_median_dollar_volume20"
                ]
            )
        )

    def test_custom_wide_band_holds_six_points_and_targets_inner_band(self):
        result = engine.StrategyResult(
            target_weights={"A": 0.50, "B": 0.50},
            regime="STATIC",
            leader=engine.LEVERAGED_SEMICONDUCTOR,
            volatility_tier="N/A",
            annualized_volatility=0.20,
            raw_volatility_tier="MODERATE",
        )
        state = engine.PortfolioState(
            executed_regime="STATIC",
            executed_strategy_fingerprint=engine.STRATEGY_FINGERPRINT,
        )
        hold = engine.build_rebalance_plan(
            {"A": 0.56, "B": 0.44},
            result,
            state,
            rebalance_band=0.075,
            rebalance_destination=0.0375,
        )
        self.assertFalse(hold.rebalance_due)

        rebalance = engine.build_rebalance_plan(
            {"A": 0.575, "B": 0.425},
            result,
            state,
            rebalance_band=0.075,
            rebalance_destination=0.0375,
        )
        self.assertTrue(rebalance.rebalance_due)
        self.assertAlmostEqual(rebalance.execution_weights["A"], 0.5375)
        self.assertAlmostEqual(rebalance.execution_weights["B"], 0.4625)


class StatisticalControlTests(unittest.TestCase):
    def test_benjamini_hochberg_is_monotone_and_exact(self):
        adjusted = research.benjamini_hochberg(
            {"a": 0.01, "b": 0.04, "c": 0.03}
        )
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.04)
        self.assertAlmostEqual(adjusted["c"], 0.04)

    def test_bootstrap_block_draws_are_deterministic(self):
        first = research.bootstrap_block_starts(
            250,
            samples=20,
            block_length=10,
        )
        second = research.bootstrap_block_starts(
            250,
            samples=20,
            block_length=10,
        )
        np.testing.assert_array_equal(first, second)

    def test_hand_calculated_metrics_and_turnover(self):
        index = pd.date_range("2025-01-02", periods=3, freq="B")
        ledger = pd.DataFrame(
            {
                "nav": [100.0, 110.0, 99.0],
                "ongoing_gross_trade_fraction": [0.0, 0.20, 0.30],
                "one_way_turnover": [0.0, 0.10, 0.15],
                "initial_deployment": [False, False, False],
                "security_orders": [0, 1, 1],
                "transaction_cost": [0.0, 1.0, 2.0],
                "transaction_cost_fraction": [0.0, 0.01, 0.02],
                "max_trade_to_median_dollar_volume20": [0.0, 0.001, 0.002],
                "advertised_daily_exposure": [1.0, 1.0, 1.0],
            },
            index=index,
        )
        metrics = research.calculate_metrics(ledger)
        self.assertAlmostEqual(metrics["terminal_wealth_multiple"], 0.99)
        self.assertAlmostEqual(metrics["maximum_drawdown"], -0.10)
        self.assertAlmostEqual(metrics["annual_gross_turnover"], 63.0)
        self.assertAlmostEqual(metrics["annual_one_way_turnover"], 31.5)
        self.assertEqual(metrics["total_modeled_cost"], 3.0)

    def test_identical_paths_have_degenerate_bootstrap(self):
        index = pd.date_range("2024-01-02", periods=260, freq="B")
        nav = pd.Series(np.cumprod(np.full(len(index), 1.0002)), index=index)
        ledger = pd.DataFrame({"nav": nav})
        result = research.BacktestResult(
            variant=research.VARIANTS["original"],
            cost_bps=10.0,
            ledger=ledger,
            metrics={},
            unfilled_final_order=False,
        )
        output = research.paired_moving_block_bootstrap(
            result,
            result,
            samples=100,
            block_length=5,
        )
        self.assertEqual(output["annualized_excess_geometric_growth"], 0.0)
        self.assertEqual(output["ci95_excess_geometric_growth"], [0.0, 0.0])
        self.assertEqual(output["one_sided_p_value"], 1.0)

    def test_pbo_uses_every_observation_and_validates_blocks(self):
        index = pd.date_range("2022-01-03", periods=506, freq="B")
        results = {}
        for offset, name in enumerate(
            ("original", "c1_wide_buffer", "c2_bull_reentry_2")
        ):
            returns = 0.0002 + offset * 0.00001 + 0.001 * np.sin(
                np.arange(len(index)) / (7.0 + offset)
            )
            ledger = pd.DataFrame(
                {"nav": 100.0 * np.cumprod(1.0 + returns)},
                index=index,
            )
            results[name] = research.BacktestResult(
                variant=research.VARIANTS[name],
                cost_bps=10.0,
                ledger=ledger,
                metrics={},
                unfilled_final_order=False,
            )
        output = research.pbo_diagnostic(results, blocks=8)
        self.assertEqual(output["usable_observations"], 505)
        self.assertEqual(output["splits"], 70)
        self.assertGreaterEqual(output["pbo"], 0.0)
        self.assertLessEqual(output["pbo"], 1.0)
        with self.assertRaises(ValueError):
            research.pbo_diagnostic(results, blocks=7)


if __name__ == "__main__":
    unittest.main()
