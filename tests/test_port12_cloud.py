import copy
import json
import os
import sys
import tempfile
import types
import unittest
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import alpha_core as core  # noqa: E402
import port12_cloud as engine  # noqa: E402


SIGNAL_DATE = pd.Timestamp("2026-08-04")


def market_prices(rows=900, *, end=SIGNAL_DATE):
    index = engine.required_nyse_sessions(pd.Timestamp(end), rows)
    position = np.arange(rows, dtype=float)
    returns = {
        engine.MARKET_INDEX: (
            0.00045 + 0.006 * np.sin(position / 2.1)
        ),
        engine.SEMICONDUCTOR_SIGNAL: (
            0.00060
            + 0.009 * np.sin(position / 2.3)
            + 0.002 * np.cos(position / 7)
        ),
        engine.VOLATILITY_INDEX: (
            0.00065 + 0.012 * np.sin(position / 2.05)
        ),
        engine.LEVERAGED_INDEX: (
            0.00090 + 0.019 * np.sin(position / 2.0)
        ),
        engine.LEVERAGED_GOLD: (
            0.00020 + 0.010 * np.sin(position / 2.6)
        ),
        engine.LEVERAGED_SEMICONDUCTOR: (
            0.0010
            + 0.025 * np.sin(position / 1.9)
            + 0.004 * np.cos(position / 5)
        ),
        engine.LEGACY_TECH: (
            0.0008 + 0.018 * np.sin(position / 2.2)
        ),
        engine.LEGACY_DEFENSIVE_EQUITY: (
            0.0003 + 0.007 * np.sin(position / 2.5)
        ),
        engine.LEGACY_HEDGE: (
            0.0001 + 0.004 * np.sin(position / 2.7)
        ),
    }
    return pd.DataFrame(
        {
            ticker: (70.0 + index_number * 5.0)
            * np.exp(np.cumsum(values))
            for index_number, (ticker, values) in enumerate(returns.items())
        },
        index=index,
    )


def yfinance_payload(prices):
    return pd.concat({"Close": prices}, axis=1)


def one_row_prices(price=100.0):
    return pd.DataFrame(
        {ticker: [float(price)] for ticker in engine.ALL_TICKERS},
        index=pd.DatetimeIndex([SIGNAL_DATE]),
    )


def make_decision(
    weight=0.0,
    *,
    active=None,
    reason="HOLD",
    structural=False,
    last_processed="",
):
    if active is None:
        active = weight > 0
    overlay = core.OverlayState(
        overlay_active=active,
        eligible_streak=2 if active else 0,
        soxl_weight=weight,
        soxl_weight_date=(
            SIGNAL_DATE.date().isoformat() if weight else ""
        ),
        last_processed_signal_date=last_processed,
    )
    return engine.StrategyDecision(
        target_weights=engine.target_weights(weight),
        overlay_state=overlay,
        transition_reason=reason,
        alpha_reviewed=False,
        structural_change=structural,
        trend_positive=True,
        residual_positive=True,
        raw_soxl_weight=weight,
        qqq_close=120.0,
        qqq_sma_200=100.0,
        residual_signal=None,
        portfolio_volatility=None,
    )


def make_run(
    *,
    state=None,
    decision=None,
    plan=None,
    current_weights=None,
    prices=None,
):
    state = state or engine.PortfolioState()
    decision = decision or make_decision()
    prices = prices if prices is not None else one_row_prices()
    current_weights = dict(current_weights or {})
    if plan is None:
        plan = engine.RebalancePlan(
            execution_weights=engine._with_cash_target(
                decision.target_weights
            ),
            rebalance_due=False,
            full_transition=False,
            reason="HOLD",
            one_way_turnover=0.0,
            individual_orders=0,
        )
    planning = copy.deepcopy(state)
    engine._apply_overlay_state(planning, decision.overlay_state)
    table = engine.calculate_execution_table(
        prices,
        plan.execution_weights,
        1_000.0,
        planning,
        actionable=plan.rebalance_due,
    )
    diagnostics = engine.calculate_execution_diagnostics(
        table,
        1_000.0,
        current_weights,
        decision.target_weights,
        plan.execution_weights,
    )
    return engine.StrategyRun(
        price_data=prices,
        decision=decision,
        state=state,
        planning_state=planning,
        portfolio_value=1_000.0,
        current_weights=current_weights,
        execution_table=table,
        signal_date=SIGNAL_DATE,
        market_data_fingerprint="a" * 64,
        rebalance_plan=plan,
        execution_diagnostics=diagnostics,
    )


def actionable_plan(weights, *, reason="INDIVIDUAL_DRIFT_BAND"):
    return engine.RebalancePlan(
        execution_weights=engine._with_cash_target(weights),
        rebalance_due=True,
        full_transition=False,
        reason=reason,
        one_way_turnover=0.10,
        individual_orders=2,
        individual_drift_triggered=True,
    )


class EngineTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temporary_directory.name)
        self.state_file = self.temp_path / "state.json"
        self.log_file = self.temp_path / "engine.log"
        self.audit_file = self.temp_path / "decision.json"
        self.shadow_file = self.temp_path / "shadow.jsonl"
        self.patches = [
            mock.patch.object(engine, "STATE_FILE", self.state_file),
            mock.patch.object(engine, "LOG_FILE", self.log_file),
            mock.patch.object(
                engine,
                "DECISION_AUDIT_FILE",
                self.audit_file,
            ),
            mock.patch.object(
                engine,
                "SHADOW_LEDGER_FILE",
                self.shadow_file,
            ),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for handler in list(engine.logger.handlers):
            handler.close()
        engine.logger.handlers.clear()
        for patch in reversed(self.patches):
            patch.stop()
        self.temporary_directory.cleanup()

    def write_state(self, value):
        payload = asdict(value) if isinstance(
            value,
            engine.PortfolioState,
        ) else value
        self.state_file.write_text(
            json.dumps(payload, allow_nan=True),
            encoding="utf-8",
        )

    def download(self, prices, *, expected=SIGNAL_DATE):
        fake_yfinance = types.ModuleType("yfinance")
        fake_yfinance.download = mock.Mock(
            return_value=yfinance_payload(prices)
        )
        with mock.patch.dict(
            sys.modules,
            {"yfinance": fake_yfinance},
        ), mock.patch.object(
            engine,
            "expected_completed_session",
            return_value=pd.Timestamp(expected),
        ), mock.patch.object(
            engine,
            "MODEL_START_DATE",
            prices.index[0].date().isoformat(),
        ):
            result = engine.download_market_data(engine.ALL_TICKERS)
        return result, fake_yfinance.download


class StrategyTransitionTests(unittest.TestCase):
    def test_missed_sessions_are_replayed_before_latest_decision(self):
        prices = market_prices(900)
        sessions = prices.index[-3:]
        bearish_session = pd.Timestamp(sessions[-2])
        state = engine.PortfolioState(
            overlay_active=True,
            eligible_streak=5,
            soxl_weight=0.25,
            soxl_weight_date=sessions[-3].date().isoformat(),
            last_alpha_review_date=sessions[-3].date().isoformat(),
            last_processed_signal_date=sessions[-3].date().isoformat(),
        )

        def calculate_one(history, working):
            session = pd.Timestamp(history.index[-1])
            transition = core.advance_overlay_state(
                engine._overlay_state_from_portfolio(working),
                signal_date=session,
                trend_positive=session != bearish_session,
                residual_positive=True,
                raw_soxl_weight=0.25,
                alpha_review_due=False,
            )
            base = make_decision()
            return replace(
                base,
                target_weights=engine.target_weights(
                    transition.state.soxl_weight
                ),
                overlay_state=transition.state,
                transition_reason=transition.reason,
                structural_change=transition.structural_change,
            )

        with mock.patch.object(
            engine,
            "_calculate_latest_strategy_decision",
            side_effect=calculate_one,
        ):
            decision = engine.calculate_strategy_decision(prices, state)

        self.assertFalse(decision.overlay_state.overlay_active)
        self.assertEqual(decision.overlay_state.soxl_weight, 0.0)
        self.assertEqual(
            decision.processed_signal_dates,
            tuple(item.date().isoformat() for item in sessions[-2:]),
        )
        self.assertEqual(decision.transition_path[0], "TREND_EXIT")

    def test_bearish_trend_exits_overlay_immediately(self):
        state = core.OverlayState(
            overlay_active=True,
            eligible_streak=5,
            soxl_weight=0.35,
        )
        transition = core.advance_overlay_state(
            state,
            signal_date=SIGNAL_DATE,
            trend_positive=False,
            residual_positive=True,
            raw_soxl_weight=0.35,
            alpha_review_due=False,
        )
        self.assertFalse(transition.state.overlay_active)
        self.assertEqual(transition.state.soxl_weight, 0.0)
        self.assertEqual(transition.reason, "TREND_EXIT")

    def test_two_distinct_eligible_closes_reenter_and_duplicate_does_not_count(self):
        first = core.advance_overlay_state(
            core.OverlayState(),
            signal_date=SIGNAL_DATE,
            trend_positive=True,
            residual_positive=True,
            raw_soxl_weight=0.25,
            alpha_review_due=True,
        )
        duplicate = core.advance_overlay_state(
            first.state,
            signal_date=SIGNAL_DATE,
            trend_positive=True,
            residual_positive=True,
            raw_soxl_weight=0.25,
            alpha_review_due=True,
        )
        self.assertEqual(duplicate.reason, "SAME_DATE")
        self.assertEqual(duplicate.state.eligible_streak, 1)
        second = core.advance_overlay_state(
            duplicate.state,
            signal_date=SIGNAL_DATE + pd.Timedelta(days=1),
            trend_positive=True,
            residual_positive=True,
            raw_soxl_weight=0.25,
            alpha_review_due=True,
        )
        self.assertTrue(second.state.overlay_active)
        self.assertEqual(second.state.soxl_weight, 0.25)

    def test_volatility_downshift_is_immediate_and_upshift_needs_five_closes(self):
        state = core.OverlayState(
            overlay_active=True,
            eligible_streak=2,
            soxl_weight=0.35,
        )
        down = core.advance_overlay_state(
            state,
            signal_date=SIGNAL_DATE,
            trend_positive=True,
            residual_positive=True,
            raw_soxl_weight=0.15,
            alpha_review_due=False,
        )
        self.assertEqual(down.state.soxl_weight, 0.15)
        current = down.state
        for day in range(1, 5):
            current = core.advance_overlay_state(
                current,
                signal_date=SIGNAL_DATE + pd.Timedelta(days=day),
                trend_positive=True,
                residual_positive=True,
                raw_soxl_weight=0.25,
                alpha_review_due=False,
            ).state
            self.assertEqual(current.soxl_weight, 0.15)
        fifth = core.advance_overlay_state(
            current,
            signal_date=SIGNAL_DATE + pd.Timedelta(days=5),
            trend_positive=True,
            residual_positive=True,
            raw_soxl_weight=0.25,
            alpha_review_due=False,
        )
        self.assertEqual(fifth.state.soxl_weight, 0.25)

    def test_fresh_reentry_uses_current_scale_immediately(self):
        first = core.advance_overlay_state(
            core.OverlayState(),
            signal_date=SIGNAL_DATE,
            trend_positive=True,
            residual_positive=True,
            raw_soxl_weight=0.35,
            alpha_review_due=True,
        )
        second = core.advance_overlay_state(
            first.state,
            signal_date=SIGNAL_DATE + pd.Timedelta(days=1),
            trend_positive=True,
            residual_positive=True,
            raw_soxl_weight=0.35,
            alpha_review_due=True,
        )
        self.assertEqual(second.reason, "OVERLAY_REENTRY")
        self.assertEqual(second.state.soxl_weight, 0.35)
        self.assertEqual(second.state.pending_scale_days, 0)

    def test_alpha_review_uses_fixed_model_history_phase(self):
        dates = engine.required_nyse_sessions(SIGNAL_DATE, 30)
        for position, session in enumerate(dates):
            self.assertEqual(
                engine.alpha_review_due(
                    "",
                    session,
                    dates,
                ),
                position % engine.ALPHA_REVIEW_SESSIONS == 0,
            )

class SignalIntegrationTests(EngineTestCase):
    def test_exact_no_crypto_core_target_math(self):
        self.assertEqual(
            engine.target_weights(0.0),
            {
                engine.LEVERAGED_INDEX: 0.65,
                engine.LEVERAGED_GOLD: 0.35,
                engine.LEVERAGED_SEMICONDUCTOR: 0.0,
            },
        )
        middle = engine.target_weights(0.25)
        self.assertAlmostEqual(middle[engine.LEVERAGED_INDEX], 0.4875)
        self.assertAlmostEqual(middle[engine.LEVERAGED_GOLD], 0.2625)
        self.assertAlmostEqual(middle[engine.LEVERAGED_SEMICONDUCTOR], 0.25)
        maximum = engine.target_weights(core.MAX_SOXL_WEIGHT)
        self.assertAlmostEqual(maximum[engine.LEVERAGED_INDEX], 0.4225)
        self.assertAlmostEqual(maximum[engine.LEVERAGED_GOLD], 0.2275)
        self.assertAlmostEqual(maximum[engine.LEVERAGED_SEMICONDUCTOR], 0.35)
        self.assertTrue(
            {"GBTC", "IBIT", "BTC-USD"}.isdisjoint(engine.ALL_TICKERS)
        )

    def test_strategy_uses_shared_residual_and_volatility_primitives(self):
        prices = market_prices(900)
        state = engine.PortfolioState(cash_balance=10_000.0)
        with mock.patch.object(
            core,
            "calculate_residual_signal",
            wraps=core.calculate_residual_signal,
        ) as residual, mock.patch.object(
            core,
            "calculate_portfolio_volatility",
            wraps=core.calculate_portfolio_volatility,
        ) as volatility:
            result = engine.calculate_strategy_decision(prices, state)
        residual.assert_called_once()
        volatility.assert_called_once()
        self.assertEqual(
            result.target_weights,
            engine.target_weights(result.overlay_state.soxl_weight),
        )
        volatility_prices = volatility.call_args.args[0]
        self.assertEqual(
            list(volatility_prices.columns),
            [engine.VOLATILITY_INDEX, engine.LEVERAGED_SEMICONDUCTOR],
        )

    def test_flat_volatility_fails_closed_and_immediately_removes_soxl(self):
        prices = market_prices(900)
        prices[engine.VOLATILITY_INDEX] = 100.0
        prices[engine.LEVERAGED_SEMICONDUCTOR] = 100.0
        state = engine.PortfolioState(
            cash_balance=1_000.0,
            overlay_active=True,
            eligible_streak=3,
            soxl_weight=0.35,
            last_processed_signal_date="2026-08-03",
        )
        result = engine.calculate_strategy_decision(prices, state)
        self.assertIsNone(result.portfolio_volatility)
        self.assertEqual(result.raw_soxl_weight, 0.0)
        self.assertEqual(result.overlay_state.soxl_weight, 0.0)
        self.assertIn("volatility=", result.failure_reason)

    def test_nan_signal_price_is_rejected(self):
        prices = market_prices(900)
        prices.iloc[-1, prices.columns.get_loc(engine.MARKET_INDEX)] = np.nan
        with self.assertRaisesRegex(RuntimeError, "trend inputs"):
            engine.calculate_strategy_decision(
                prices,
                engine.PortfolioState(cash_balance=1_000.0),
            )

    def test_strategy_manifest_has_exact_maximum_exposure(self):
        target = engine.target_weights(core.MAX_SOXL_WEIGHT)
        self.assertAlmostEqual(
            engine.advertised_daily_exposure(target),
            engine.MAX_ADVERTISED_DAILY_EXPOSURE,
        )
        self.assertEqual(
            engine.STRATEGY_FINGERPRINT,
            engine.calculate_strategy_fingerprint(),
        )
        self.assertEqual(
            engine.STRATEGY_FINGERPRINT,
            engine.EXPECTED_STRATEGY_FINGERPRINT,
        )


class RebalanceTests(EngineTestCase):
    def aligned_state(self, weight):
        return engine.PortfolioState(
            cash_balance=1.0,
            executed_overlay_active=weight > 0,
            executed_soxl_weight=weight,
            executed_strategy_fingerprint=engine.STRATEGY_FINGERPRINT,
        )

    def test_exact_five_point_individual_drift_triggers(self):
        decision = make_decision(0.35)
        state = self.aligned_state(0.35)
        plan = engine.build_rebalance_plan(
            {
                engine.LEVERAGED_INDEX: 0.4725,
                engine.LEVERAGED_GOLD: 0.1775,
                engine.LEVERAGED_SEMICONDUCTOR: 0.35,
            },
            decision,
            state,
        )
        self.assertTrue(plan.rebalance_due)
        self.assertTrue(plan.individual_drift_triggered)
        below = engine.build_rebalance_plan(
            {
                engine.LEVERAGED_INDEX: 0.4724,
                engine.LEVERAGED_GOLD: 0.1776,
                engine.LEVERAGED_SEMICONDUCTOR: 0.35,
            },
            decision,
            state,
        )
        self.assertFalse(below.rebalance_due)

    def test_exact_five_point_aggregate_equity_drift_triggers(self):
        individual, aggregate = engine.drift_triggers(
            {
                engine.LEVERAGED_INDEX: 0.3975,
                engine.LEVERAGED_SEMICONDUCTOR: 0.325,
                engine.LEVERAGED_GOLD: 0.2525,
                engine.CASH_ASSET: 0.025,
            },
            engine.target_weights(0.35),
        )
        self.assertFalse(individual)
        self.assertTrue(aggregate)

    def test_drift_trade_stops_at_inner_destination(self):
        result = engine.inner_band_rebalance_weights(
            {
                engine.LEVERAGED_INDEX: 0.4725,
                engine.LEVERAGED_GOLD: 0.1775,
                engine.LEVERAGED_SEMICONDUCTOR: 0.35,
            },
            engine.target_weights(0.35),
        )
        self.assertAlmostEqual(result[engine.LEVERAGED_INDEX], 0.4475)
        self.assertAlmostEqual(result[engine.LEVERAGED_GOLD], 0.2025)
        self.assertAlmostEqual(
            result[engine.LEVERAGED_SEMICONDUCTOR],
            0.35,
        )

    def test_inner_destination_never_exceeds_the_soxl_cap(self):
        plan = engine.build_rebalance_plan(
            {
                engine.LEVERAGED_INDEX: 0.35,
                engine.LEVERAGED_GOLD: 0.25,
                engine.LEVERAGED_SEMICONDUCTOR: 0.40,
            },
            make_decision(core.MAX_SOXL_WEIGHT),
            self.aligned_state(core.MAX_SOXL_WEIGHT),
        )
        self.assertAlmostEqual(
            plan.execution_weights[engine.LEVERAGED_SEMICONDUCTOR],
            core.MAX_SOXL_WEIGHT,
        )
        self.assertAlmostEqual(
            plan.execution_weights[engine.LEVERAGED_INDEX],
            0.3975,
        )
        self.assertAlmostEqual(
            plan.execution_weights[engine.LEVERAGED_GOLD],
            0.25125,
        )
        self.assertAlmostEqual(
            plan.execution_weights[engine.CASH_ASSET],
            0.00125,
        )

    def test_ordinary_drift_rebalance_returns_soxl_to_exact_tier(self):
        plan = engine.build_rebalance_plan(
            {
                engine.LEVERAGED_INDEX: 0.50,
                engine.LEVERAGED_GOLD: 0.30,
                engine.LEVERAGED_SEMICONDUCTOR: 0.20,
            },
            make_decision(0.15),
            self.aligned_state(0.15),
        )
        self.assertTrue(plan.rebalance_due)
        self.assertEqual(plan.reason, "INDIVIDUAL_DRIFT_BAND")
        self.assertAlmostEqual(
            plan.execution_weights[engine.LEVERAGED_SEMICONDUCTOR],
            0.15,
        )

    def test_risk_off_soxl_is_exactly_sold_below_the_drift_band(self):
        state = self.aligned_state(0.0)
        decision = make_decision(0.0)
        plan = engine.build_rebalance_plan(
            {
                engine.LEVERAGED_INDEX: 0.99,
                engine.LEVERAGED_SEMICONDUCTOR: 0.01,
            },
            decision,
            state,
        )
        self.assertTrue(plan.full_transition)
        self.assertEqual(plan.reason, "RISK_OFF_RESIDUAL_EXIT")
        self.assertEqual(
            plan.execution_weights[engine.LEVERAGED_SEMICONDUCTOR],
            0.0,
        )

    def test_legacy_position_is_exactly_sold_below_drift_band(self):
        state = self.aligned_state(0.0)
        state.shares = {engine.LEGACY_HEDGE: 0.10, engine.LEVERAGED_INDEX: 9.9}
        current = {
            engine.LEGACY_HEDGE: 0.01,
            engine.LEVERAGED_INDEX: 0.99,
        }
        decision = make_decision(0.0)
        plan = engine.build_rebalance_plan(current, decision, state)
        self.assertTrue(plan.full_transition)
        self.assertEqual(plan.reason, "LEGACY_POSITION_EXIT")
        table = engine.calculate_execution_table(
            one_row_prices(),
            plan.execution_weights,
            1_000.0,
            state,
            actionable=True,
        )
        row = table.loc[table["Ticker"] == engine.LEGACY_HEDGE].iloc[0]
        self.assertEqual(row["TargetPct"], 0.0)
        self.assertEqual(row["Action"], "SELL")
        self.assertAlmostEqual(row["EstimatedUnits"], 0.0)

        tiny_state = self.aligned_state(0.0)
        tiny_state.shares = {
            engine.LEGACY_HEDGE: 0.00000001,
            engine.LEVERAGED_INDEX: 10.0,
        }
        tiny_table = engine.calculate_execution_table(
            one_row_prices(),
            engine._with_cash_target(engine.target_weights(0.0)),
            1_000.0,
            tiny_state,
            actionable=True,
        )
        tiny_row = tiny_table.loc[
            tiny_table["Ticker"] == engine.LEGACY_HEDGE
        ].iloc[0]
        self.assertEqual(tiny_row["Action"], "SELL")

    def test_legacy_qld_is_explicitly_sold_for_new_core(self):
        state = self.aligned_state(0.0)
        state.shares = {engine.VOLATILITY_INDEX: 10.0}
        plan = engine.build_rebalance_plan(
            {engine.VOLATILITY_INDEX: 1.0},
            make_decision(0.0),
            state,
        )
        self.assertTrue(plan.full_transition)
        self.assertEqual(plan.reason, "LEGACY_POSITION_EXIT")
        table = engine.calculate_execution_table(
            one_row_prices(),
            plan.execution_weights,
            1_000.0,
            state,
            actionable=True,
        )
        qld = table.loc[
            table["Ticker"] == engine.VOLATILITY_INDEX
        ].iloc[0]
        self.assertEqual(qld["TargetPct"], 0.0)
        self.assertEqual(qld["Action"], "SELL")

    def test_strategy_revision_forces_exact_transition(self):
        state = engine.PortfolioState(
            cash_balance=1.0,
            executed_strategy_fingerprint="b" * 64,
        )
        decision = make_decision()
        plan = engine.build_rebalance_plan(
            {engine.LEVERAGED_INDEX: 0.97, engine.CASH_ASSET: 0.03},
            decision,
            state,
        )
        self.assertTrue(plan.full_transition)
        self.assertEqual(
            plan.execution_weights,
            {
                engine.LEVERAGED_INDEX: 0.65,
                engine.LEVERAGED_GOLD: 0.35,
                engine.LEVERAGED_SEMICONDUCTOR: 0.0,
                engine.CASH_ASSET: 0.0,
            },
        )


class MarketDataTests(EngineTestCase):
    def test_partial_latest_data_is_rejected(self):
        prices = market_prices(900)
        prices.iloc[-1, prices.columns.get_loc(engine.LEGACY_HEDGE)] = np.nan
        with self.assertRaisesRegex(RuntimeError, "Latest prices"):
            self.download(prices)

    def test_missing_ticker_is_rejected(self):
        prices = market_prices(900).drop(columns=[engine.LEGACY_TECH])
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            self.download(prices)

    def test_signal_window_gap_is_rejected_without_substitution(self):
        prices = market_prices(900)
        original_index = prices.index.copy()
        prices.iloc[-50, prices.columns.get_loc(engine.LEVERAGED_INDEX)] = np.nan
        with self.assertRaisesRegex(
            RuntimeError,
            "complete quantitative model history",
        ):
            self.download(prices)
        self.assertTrue(prices.index.equals(original_index))
        self.assertTrue(prices.iloc[-50].isna().any())

    def test_unrelated_legacy_history_gap_is_not_filled_or_dropped(self):
        prices = market_prices(900)
        prices.iloc[-50, prices.columns.get_loc(engine.LEGACY_HEDGE)] = np.nan
        result, _ = self.download(prices)
        self.assertEqual(len(result), len(prices))
        self.assertTrue(result[engine.LEGACY_HEDGE].iloc[-50] != result[engine.LEGACY_HEDGE].iloc[-50])

    def test_missing_required_session_is_rejected(self):
        prices = market_prices(engine.REQUIRED_SIGNAL_ROWS + 1)
        prices = prices.drop(prices.index[-100])
        with self.assertRaisesRegex(RuntimeError, "continuity"):
            self.download(prices)

    def test_stale_latest_session_is_rejected(self):
        prices = market_prices(900, end=SIGNAL_DATE - pd.Timedelta(days=1))
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.download(prices)

    def test_fingerprint_is_canonical_and_input_sensitive(self):
        prices = market_prices(900)
        first = engine.market_data_fingerprint(prices)
        second = prices.copy()
        second.iloc[-1, second.columns.get_loc(engine.LEVERAGED_INDEX)] *= 1.001
        self.assertTrue(engine._is_sha256(first))
        self.assertNotEqual(first, engine.market_data_fingerprint(second))
        older = prices.copy()
        older.iloc[-850, older.columns.get_loc(engine.LEVERAGED_INDEX)] *= 1.001
        self.assertNotEqual(first, engine.market_data_fingerprint(older))

    def test_same_date_revision_is_rejected(self):
        state = engine.PortfolioState(
            last_processed_signal_date=SIGNAL_DATE.date().isoformat(),
            last_processed_data_fingerprint="a" * 64,
        )
        with self.assertRaisesRegex(RuntimeError, "changed"):
            engine.validate_same_date_data_fingerprint(
                state,
                SIGNAL_DATE,
                "b" * 64,
            )

    def test_latest_completed_xnys_session_freshness(self):
        before_close = datetime(2026, 8, 3, 15, 0, tzinfo=engine.NEW_YORK)
        after_buffer = datetime(2026, 8, 3, 16, 20, tzinfo=engine.NEW_YORK)
        with self.assertRaisesRegex(RuntimeError, "not final"):
            engine.expected_completed_session(before_close)
        self.assertEqual(
            engine.expected_completed_session(after_buffer),
            pd.Timestamp("2026-08-03"),
        )


class StateAndHoldingsTests(EngineTestCase):
    def test_state_save_is_atomic_when_replace_fails(self):
        self.state_file.write_text("original", encoding="utf-8")
        with mock.patch.object(
            engine.os,
            "replace",
            side_effect=OSError("no"),
        ):
            with self.assertRaises(OSError):
                engine.save_state(engine.PortfolioState(cash_balance=10.0))
        self.assertEqual(
            self.state_file.read_text(encoding="utf-8"),
            "original",
        )

    def test_v7_migration_preserves_holdings_cash_and_delivery_evidence(self):
        payload = {
            "state_version": 7,
            "shares": {engine.SEMICONDUCTOR_SIGNAL: 2.5, engine.LEGACY_HEDGE: 1.0},
            "cash_balance": 19.87,
            "target_weights": {engine.SEMICONDUCTOR_SIGNAL: 0.8, engine.LEGACY_HEDGE: 0.2},
            "portfolio_value": 1_000.0,
            "pending_recommendation_date": "2026-08-04",
            "pending_recommendation_weights": {engine.SEMICONDUCTOR_SIGNAL: 0.8, engine.LEGACY_HEDGE: 0.2},
            "pending_recommendation_notified": True,
            "pending_recommendation_supersedes_date": "",
            "pending_recommendation_fingerprint": "b" * 64,
            "executed_strategy_fingerprint": "c" * 64,
            "last_processed_signal_date": "2026-08-04",
            "last_processed_data_fingerprint": "d" * 64,
            "last_delivered_decision_hash": "e" * 64,
            "last_delivered_signal_date": "2026-08-04",
            "last_delivered_notification_kind": "ACTION",
        }
        self.write_state(payload)
        state = engine.load_state(backup_legacy=False)
        self.assertEqual(state.state_version, engine.STATE_VERSION)
        self.assertEqual(state.shares, payload["shares"])
        self.assertEqual(state.cash_balance, 19.87)
        self.assertEqual(state.pending_recommendation_date, "2026-08-04")
        self.assertTrue(state.pending_recommendation_notified)
        self.assertEqual(state.last_delivered_decision_hash, "e" * 64)
        self.assertEqual(state.last_processed_data_fingerprint, "")

    def test_v2_migration_preserves_and_prices_legacy_shares(self):
        self.write_state(
            {
                "state_version": 2,
                "shares": {
                    engine.SEMICONDUCTOR_SIGNAL: 2.0,
                    engine.LEGACY_HEDGE: 3.0,
                },
                "cash_balance": 5.0,
                "target_weights": {},
                "portfolio_value": 0.0,
            }
        )
        state = engine.load_state(backup_legacy=False)
        self.assertAlmostEqual(
            engine.existing_portfolio_value(state, one_row_prices()),
            505.0,
        )

    def test_v8_migration_preserves_state_and_resets_old_data_hash(self):
        original = engine.PortfolioState(
            shares={engine.LEVERAGED_INDEX: 4.5},
            cash_balance=12.25,
            target_weights={engine.LEVERAGED_INDEX: 1.0},
            portfolio_value=1_234.0,
            overlay_active=True,
            eligible_streak=4,
            soxl_weight=0.20,
            soxl_weight_date="2026-08-04",
            pending_soxl_weight=0.25,
            pending_scale_days=2,
            last_alpha_review_date="2026-08-04",
            last_processed_signal_date="2026-08-04",
            shadow_ledger_sessions=1,
            shadow_ledger_last_signal_date="2026-08-04",
            shadow_ledger_chain_hash="a" * 64,
            last_processed_data_fingerprint="b" * 64,
        )
        payload = asdict(original)
        payload["state_version"] = 8
        self.write_state(payload)
        migrated = engine.load_state(backup_legacy=False)
        self.assertEqual(migrated.state_version, engine.STATE_VERSION)
        for name, value in payload.items():
            if name not in {
                "state_version",
                "last_processed_data_fingerprint",
                "soxl_weight",
            }:
                self.assertEqual(getattr(migrated, name), value)
        self.assertEqual(migrated.soxl_weight, 0.15)
        self.assertEqual(migrated.last_processed_data_fingerprint, "")
        engine.validate_same_date_data_fingerprint(
            migrated,
            pd.Timestamp("2026-08-04"),
            "c" * 64,
        )

    def test_v10_migration_drops_retired_research_only_anchors(self):
        original = engine.PortfolioState(
            shares={engine.LEVERAGED_INDEX: 4.5},
            cash_balance=12.25,
            pending_recommendation_date="2026-08-04",
            pending_recommendation_weights={engine.LEVERAGED_INDEX: 1.0},
            pending_recommendation_fingerprint="a" * 64,
            last_processed_signal_date="2026-08-04",
            last_processed_data_fingerprint="c" * 64,
        )
        payload = asdict(original)
        payload["state_version"] = 10
        payload.update(
            {
                "downside_shadow_ledger_sessions": 1,
                "downside_shadow_ledger_last_signal_date": "2026-08-04",
                "downside_shadow_ledger_chain_hash": "b" * 64,
            }
        )
        self.write_state(payload)
        migrated = engine.load_state(backup_legacy=False)
        self.assertEqual(migrated.state_version, engine.STATE_VERSION)
        self.assertEqual(migrated.shares, original.shares)
        self.assertEqual(migrated.cash_balance, original.cash_balance)
        self.assertEqual(
            migrated.pending_recommendation_weights,
            original.pending_recommendation_weights,
        )
        self.assertEqual(
            migrated.last_processed_data_fingerprint,
            original.last_processed_data_fingerprint,
        )
        self.assertFalse(
            hasattr(migrated, "downside_shadow_ledger_sessions")
        )

    def test_v11_migration_tiers_model_state_without_losing_broker_or_outbox(self):
        original = engine.PortfolioState(
            shares={engine.LEVERAGED_SEMICONDUCTOR: 2.5},
            cash_balance=12.25,
            overlay_active=True,
            eligible_streak=7,
            soxl_weight=0.30,
            soxl_weight_date="2026-08-04",
            pending_soxl_weight=0.35,
            pending_scale_days=3,
            last_processed_signal_date="2026-08-04",
            executed_overlay_active=True,
            executed_soxl_weight=0.30,
            executed_strategy_fingerprint="a" * 64,
            pending_recommendation_date="2026-08-04",
            pending_recommendation_weights={
                engine.LEVERAGED_INDEX: 0.52,
                engine.LEVERAGED_GOLD: 0.28,
                engine.LEVERAGED_SEMICONDUCTOR: 0.20,
                engine.CASH_ASSET: 0.0,
            },
            pending_recommendation_overlay_active=True,
            pending_recommendation_soxl_weight=0.20,
            pending_recommendation_notified=True,
            pending_recommendation_fingerprint="b" * 64,
            last_processed_data_fingerprint="c" * 64,
        )
        payload = asdict(original)
        payload["state_version"] = 11
        self.write_state(payload)
        migrated = engine.load_state(backup_legacy=False)
        self.assertEqual(
            migrated.shares,
            {engine.LEVERAGED_SEMICONDUCTOR: 2.5},
        )
        self.assertEqual(migrated.cash_balance, 12.25)
        self.assertEqual(migrated.soxl_weight, 0.25)
        self.assertEqual(migrated.pending_soxl_weight, 0.35)
        self.assertEqual(migrated.pending_scale_days, 3)
        self.assertEqual(migrated.executed_soxl_weight, 0.30)
        self.assertEqual(migrated.pending_recommendation_soxl_weight, 0.20)
        self.assertTrue(migrated.pending_recommendation_notified)
        self.assertEqual(migrated.last_processed_data_fingerprint, "c" * 64)

    def test_v9_migration_preserves_qld_and_forces_new_core_transition(self):
        original = engine.PortfolioState(
            shares={engine.VOLATILITY_INDEX: 10.0},
            target_weights={engine.VOLATILITY_INDEX: 1.0},
            portfolio_value=1_000.0,
            last_processed_signal_date="2026-08-04",
            last_processed_data_fingerprint="b" * 64,
            executed_strategy_fingerprint=(
                "d9ce9aaf3fbc39962598fc09986f1b37"
                "b87d4e62559980d81820326534d7e837"
            ),
        )
        payload = asdict(original)
        payload["state_version"] = 9
        self.write_state(payload)
        migrated = engine.load_state(backup_legacy=False)
        self.assertEqual(
            migrated.shares,
            {engine.VOLATILITY_INDEX: 10.0},
        )
        self.assertEqual(
            migrated.target_weights,
            {engine.VOLATILITY_INDEX: 1.0},
        )
        self.assertEqual(migrated.last_processed_data_fingerprint, "")
        plan = engine.build_rebalance_plan(
            {engine.VOLATILITY_INDEX: 1.0},
            make_decision(0.0),
            migrated,
        )
        self.assertTrue(plan.full_transition)
        self.assertEqual(plan.reason, "STRATEGY_REVISION_TRANSITION")
        self.assertEqual(
            plan.execution_weights,
            engine._with_cash_target(engine.target_weights(0.0)),
        )

    def test_migration_creates_recoverable_backup_when_enabled(self):
        payload = {
            "state_version": 2,
            "shares": {engine.LEGACY_HEDGE: 1.0},
            "cash_balance": 5.0,
            "target_weights": {},
            "portfolio_value": 105.0,
        }
        self.write_state(payload)
        original = self.state_file.read_bytes()
        engine.load_state(backup_legacy=True)
        backups = list(self.temp_path.glob("state.v2.*.backup.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)

    def test_old_unproven_outbox_migrates_to_retryable_legacy_identity(self):
        self.write_state(
            {
                "state_version": 4,
                "shares": {},
                "cash_balance": 1_000.0,
                "target_weights": {},
                "portfolio_value": 1_000.0,
                "pending_recommendation_date": "2026-08-04",
                "pending_recommendation_weights": {engine.LEGACY_HEDGE: 1.0},
                "pending_recommendation_notified": True,
            }
        )
        state = engine.load_state(backup_legacy=False)
        self.assertFalse(state.pending_recommendation_notified)
        self.assertEqual(state.pending_recommendation_fingerprint, "0" * 64)

    def test_confirmed_shares_not_saved_target_drive_weights(self):
        state = engine.PortfolioState(
            shares={engine.LEVERAGED_INDEX: 5.0},
            cash_balance=500.0,
            target_weights={engine.LEVERAGED_SEMICONDUCTOR: 1.0},
        )
        weights = engine.existing_weights(state, one_row_prices())
        self.assertEqual(
            weights,
            {engine.LEVERAGED_INDEX: 0.5, engine.CASH_ASSET: 0.5},
        )

    def test_invalid_price_for_confirmed_holding_is_rejected(self):
        state = engine.PortfolioState(
            shares={engine.LEGACY_HEDGE: 1.0},
        )
        prices = one_row_prices()
        prices.loc[SIGNAL_DATE, engine.LEGACY_HEDGE] = np.nan
        with self.assertRaisesRegex(RuntimeError, "Invalid latest price"):
            engine.existing_portfolio_value(state, prices)

    def test_sync_and_confirmation_store_complete_actual_holdings(self):
        state = engine.PortfolioState(
            cash_balance=100.0,
            pending_recommendation_date="2026-08-04",
            pending_recommendation_weights=engine._with_cash_target(engine.target_weights(0.25)),
            pending_recommendation_overlay_active=True,
            pending_recommendation_soxl_weight=0.25,
            pending_recommendation_notified=True,
            pending_recommendation_fingerprint=engine.STRATEGY_FINGERPRINT,
        )
        self.write_state(state)
        confirmed = engine.confirm_execution(
            {engine.LEVERAGED_INDEX: 8.0, engine.LEVERAGED_SEMICONDUCTOR: 2.0},
            3.25,
            "2026-08-04",
        )
        self.assertEqual(confirmed.cash_balance, 3.25)
        self.assertEqual(confirmed.executed_soxl_weight, 0.25)
        self.assertFalse(confirmed.pending_recommendation_date)
        synced = engine.sync_holdings(
            {engine.LEVERAGED_INDEX: 8.1, engine.LEVERAGED_SEMICONDUCTOR: 2.0},
            4.50,
        )
        self.assertEqual(synced.cash_balance, 4.50)
        self.assertEqual(synced.shares[engine.LEVERAGED_INDEX], 8.1)


class NotificationTests(EngineTestCase):
    def pending_state(self, weights, *, notified=True, supersedes=""):
        return engine.PortfolioState(
            shares={engine.LEVERAGED_INDEX: 10.0},
            pending_recommendation_date="2026-08-04",
            pending_recommendation_weights=engine._with_cash_target(weights),
            pending_recommendation_overlay_active=weights.get(engine.LEVERAGED_SEMICONDUCTOR, 0.0) > 0,
            pending_recommendation_soxl_weight=weights.get(engine.LEVERAGED_SEMICONDUCTOR, 0.0),
            pending_recommendation_notified=notified,
            pending_recommendation_supersedes_date=supersedes,
            pending_recommendation_fingerprint=engine.STRATEGY_FINGERPRINT,
        )

    def test_hold_saves_state_without_email(self):
        state = engine.PortfolioState(
            shares={engine.LEVERAGED_INDEX: 10.0},
            executed_strategy_fingerprint=engine.STRATEGY_FINGERPRINT,
        )
        run = make_run(
            state=state,
            current_weights={engine.LEVERAGED_INDEX: 1.0},
        )
        notice = engine.decide_notification(run)
        self.assertEqual(notice.kind, "NONE")
        engine.persist_signal_run(run)
        self.assertTrue(self.state_file.exists())

    def test_identical_delivered_pending_action_is_suppressed(self):
        weights = engine.target_weights(0.25)
        state = self.pending_state(weights)
        run = make_run(
            state=state,
            decision=make_decision(0.25),
            plan=actionable_plan(weights),
        )
        self.assertEqual(engine.decide_notification(run).kind, "NONE")

    def test_undelivered_identical_action_retries_exact_destination(self):
        weights = engine.target_weights(0.25)
        state = self.pending_state(weights, notified=False)
        plan = actionable_plan(weights)
        preserved = engine.preserve_pending_delivery_plan(
            plan,
            state,
            {engine.LEVERAGED_INDEX: 1.0},
        )
        self.assertEqual(preserved.reason, "PENDING_DELIVERY_RETRY")
        run = make_run(
            state=state,
            decision=make_decision(0.25),
            plan=preserved,
        )
        self.assertEqual(engine.decide_notification(run).kind, "RETRY")

    def test_material_update_replaces_pending_action(self):
        state = self.pending_state(engine.target_weights(0.15))
        run = make_run(
            state=state,
            decision=make_decision(0.25),
            plan=actionable_plan(engine.target_weights(0.25)),
        )
        notice = engine.decide_notification(run)
        self.assertEqual(notice.kind, "UPDATE")
        engine.prepare_notification_delivery(run, notice)
        self.assertEqual(
            state.pending_recommendation_soxl_weight,
            0.25,
        )
        self.assertEqual(
            state.pending_recommendation_supersedes_date,
            "2026-08-04",
        )

    def test_cancellation_is_one_time_and_clears_pending_after_delivery(self):
        state = self.pending_state(engine.target_weights(0.25))
        run = make_run(state=state, decision=make_decision(0.25))
        notice = engine.decide_notification(run)
        self.assertEqual(notice.kind, "CANCELLATION")
        with mock.patch.object(engine, "save_state"):
            engine.persist_notification_delivery(
                run,
                notice,
                delivered_decision_hash="f" * 64,
            )
        self.assertFalse(state.pending_recommendation_date)
        self.assertEqual(engine.decide_notification(run).kind, "NONE")

    def test_smtp_failure_retains_retry_and_actual_holdings(self):
        state = engine.PortfolioState(
            shares={engine.LEVERAGED_INDEX: 10.0},
        )
        weights = engine.target_weights(0.25)
        run = make_run(
            state=state,
            decision=make_decision(0.25),
            plan=actionable_plan(weights),
        )
        notice = engine.decide_notification(run)
        engine.prepare_notification_delivery(run, notice)
        engine.persist_signal_run(run)
        with mock.patch.object(
            engine,
            "send_email",
            side_effect=RuntimeError("SMTP"),
        ):
            with self.assertRaises(RuntimeError):
                engine.send_email("x", "x", "x")
        persisted = engine.load_state()
        self.assertEqual(
            persisted.shares,
            {engine.LEVERAGED_INDEX: 10.0},
        )
        self.assertFalse(persisted.pending_recommendation_notified)
        self.assertEqual(
            persisted.pending_recommendation_soxl_weight,
            0.25,
        )

    def test_missing_credentials_fail_only_for_actual_notification(self):
        with mock.patch.dict(
            os.environ,
            {
                "GMAIL_ADDRESS": "",
                "GMAIL_APP_PASSWORD": "",
                "RECEIVER_EMAIL": "",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "required only"):
                engine.send_email("x", "x", "x")
        run = make_run()
        self.assertFalse(engine.decide_notification(run).should_send)

    def test_test_mode_never_persists_emails_audit_or_log(self):
        run = make_run(
            state=engine.PortfolioState(cash_balance=1_000.0),
            plan=actionable_plan(engine.target_weights(0.0)),
        )
        with mock.patch.object(sys, "argv", ["port12_cloud.py", "--test", "--roth-amount", "1000"]), mock.patch.object(
            engine,
            "run_strategy",
            return_value=run,
        ), mock.patch.object(engine, "save_state") as save, mock.patch.object(
            engine,
            "send_email",
        ) as send, mock.patch.object(
            engine,
            "write_decision_audit",
        ) as audit:
            engine.main()
        save.assert_not_called()
        send.assert_not_called()
        audit.assert_not_called()
        self.assertFalse(self.log_file.exists())
        self.assertFalse(self.audit_file.exists())
        self.assertFalse(self.state_file.exists())
        self.assertFalse(self.shadow_file.exists())

    def test_delivery_evidence_survives_later_audit_failure(self):
        state = engine.PortfolioState(shares={engine.LEVERAGED_INDEX: 10.0})
        weights = engine.target_weights(0.25)
        run = make_run(
            state=state,
            decision=make_decision(0.25),
            plan=actionable_plan(weights),
        )
        notice = engine.decide_notification(run)
        engine.prepare_notification_delivery(run, notice)
        engine.persist_signal_run(run)
        engine.persist_notification_delivery(
            run,
            notice,
            delivered_decision_hash="9" * 64,
        )
        with mock.patch.object(
            engine,
            "write_decision_audit",
            side_effect=OSError("disk"),
        ):
            with self.assertRaises(OSError):
                engine.write_decision_audit(run, notice, "DELIVERED")
        persisted = engine.load_state()
        self.assertTrue(persisted.pending_recommendation_notified)
        self.assertEqual(persisted.last_delivered_decision_hash, "9" * 64)


class AuditAndCliTests(EngineTestCase):
    @staticmethod
    def shadow_observation(
        signal_date="2026-08-04",
        *,
        structural=False,
    ):
        return engine.ShadowObservation(
            signal_date=signal_date,
            data_fingerprint="a" * 64,
            trend_positive=True,
            residual_positive=False,
            raw_soxl_weight=0.0,
            overlay_active=False,
            soxl_weight=0.0,
            transition_reason="ALPHA_BLOCK",
            structural_change=structural,
            failure_reason="",
        )

    def test_shadow_ledger_is_chained_idempotent_and_tamper_evident(self):
        shared_state = engine.PortfolioState()
        first_decision = replace(
            make_decision(),
            shadow_observations=(self.shadow_observation(),),
        )
        first_run = make_run(
            state=shared_state,
            decision=first_decision,
        )
        summary = engine.append_shadow_ledger(first_run)
        self.assertEqual(summary["sessions"], 1)
        self.assertEqual(summary["structural_decisions"], 0)
        self.assertEqual(
            engine.append_shadow_ledger(first_run)["sessions"],
            1,
        )

        second_decision = replace(
            make_decision(),
            shadow_observations=(
                self.shadow_observation(
                    "2026-08-05",
                    structural=True,
                ),
            ),
        )
        summary = engine.append_shadow_ledger(
            make_run(
                state=shared_state,
                decision=second_decision,
            )
        )
        self.assertEqual(summary["sessions"], 2)
        self.assertEqual(summary["structural_decisions"], 1)
        self.assertTrue(engine._is_sha256(summary["chain_hash"]))

        records = self.shadow_file.read_text(
            encoding="utf-8"
        ).splitlines()
        tampered = json.loads(records[0])
        tampered["observation"]["transition_reason"] = "TAMPERED"
        records[0] = json.dumps(tampered)
        self.shadow_file.write_text(
            "\n".join(records) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "record hash"):
            engine.shadow_ledger_summary()

    def test_shadow_anchor_rejects_missing_or_truncated_chain(self):
        state = engine.PortfolioState(
            last_processed_signal_date="2026-08-05",
        )
        decision = replace(
            make_decision(),
            shadow_observations=(
                self.shadow_observation("2026-08-04"),
                self.shadow_observation("2026-08-05"),
            ),
        )
        run = make_run(state=state, decision=decision)
        summary = engine.append_shadow_ledger(run)
        self.assertEqual(state.shadow_ledger_sessions, 2)
        self.assertEqual(
            state.shadow_ledger_chain_hash,
            summary["chain_hash"],
        )
        complete = self.shadow_file.read_text(encoding="utf-8")

        self.shadow_file.unlink()
        with self.assertRaisesRegex(RuntimeError, "missing or truncated"):
            engine.append_shadow_ledger(run)

        lines = complete.splitlines()
        self.shadow_file.write_text(
            lines[0] + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "missing or truncated"):
            engine.append_shadow_ledger(run)

    def test_shadow_chain_ahead_of_state_requires_exact_replay(self):
        state = engine.PortfolioState()
        first = replace(
            make_decision(),
            shadow_observations=(self.shadow_observation("2026-08-04"),),
        )
        engine.append_shadow_ledger(make_run(state=state, decision=first))

        # Model a crash after the atomic ledger write but before state save.
        stale = engine.PortfolioState()
        exact = engine.append_shadow_ledger(
            make_run(state=stale, decision=first)
        )
        self.assertEqual(exact["sessions"], 1)
        self.assertEqual(stale.shadow_ledger_sessions, 1)

        mismatched = replace(
            make_decision(),
            shadow_observations=(
                replace(
                    self.shadow_observation("2026-08-04"),
                    residual_positive=True,
                ),
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "deterministic state replay"):
            engine.append_shadow_ledger(
                make_run(
                    state=engine.PortfolioState(),
                    decision=mismatched,
                )
            )

    def test_shadow_append_failure_cannot_advance_persisted_signal_state(self):
        decision = replace(
            make_decision(),
            shadow_observations=(
                self.shadow_observation("2026-08-04"),
                self.shadow_observation(
                    "2026-08-05",
                    structural=True,
                ),
            ),
        )
        run = make_run(
            state=engine.PortfolioState(cash_balance=1_000.0),
            decision=decision,
        )
        with mock.patch.object(
            sys,
            "argv",
            ["port12_cloud.py"],
        ), mock.patch.object(
            engine,
            "run_strategy",
            return_value=run,
        ), mock.patch.object(
            engine,
            "append_shadow_ledger",
            side_effect=OSError("disk"),
        ), mock.patch.object(
            engine,
            "persist_signal_run",
        ) as persist:
            with self.assertRaises(OSError):
                engine.main()
        persist.assert_not_called()
        self.assertEqual(engine.append_shadow_ledger(run)["sessions"], 2)

    def test_implementation_fingerprint_includes_shared_alpha_core(self):
        fake_core = self.temp_path / "alpha_core.py"
        fake_core.write_text("version = 1\n", encoding="utf-8")
        with mock.patch.object(engine.core, "__file__", str(fake_core)):
            strategy_first = engine.calculate_strategy_fingerprint()
            first = engine.calculate_implementation_fingerprint()
            fake_core.write_text("version = 2\n", encoding="utf-8")
            strategy_second = engine.calculate_strategy_fingerprint()
            second = engine.calculate_implementation_fingerprint()
        self.assertNotEqual(first, second)
        self.assertEqual(strategy_first, strategy_second)

    def test_audit_hash_is_stable_across_delivery_and_has_model_lineage(self):
        run = make_run()
        notice = engine.NotificationDecision("NONE", "HOLD")
        first = engine.build_decision_audit(run, notice, "NOT_REQUIRED")
        second = engine.build_decision_audit(run, notice, "NOT_REQUIRED")
        self.assertEqual(first["decision_hash"], second["decision_hash"])
        serialized = json.dumps(first)
        self.assertNotIn("GMAIL_APP_PASSWORD", serialized)
        self.assertEqual(
            first["strategy"]["fingerprint"],
            engine.STRATEGY_FINGERPRINT,
        )

    def test_audit_write_is_atomic(self):
        run = make_run()
        notice = engine.NotificationDecision("NONE", "HOLD")
        self.audit_file.write_text("old", encoding="utf-8")
        with mock.patch.object(
            engine.os,
            "replace",
            side_effect=OSError("no"),
        ):
            with self.assertRaises(OSError):
                engine.write_decision_audit(run, notice, "NOT_REQUIRED")
        self.assertEqual(self.audit_file.read_text(encoding="utf-8"), "old")

    def test_execution_parser_requires_complete_cash_and_accepts_legacy(self):
        holdings, cash = engine.parse_executed_shares(
            [
                f"{engine.LEVERAGED_INDEX}=1.5",
                f"{engine.LEGACY_HEDGE}=2",
                "CASH=3.25",
            ]
        )
        self.assertEqual(cash, 3.25)
        self.assertEqual(holdings[engine.LEGACY_HEDGE], 2.0)
        with self.assertRaisesRegex(ValueError, "include CASH"):
            engine.parse_executed_shares(
                [f"{engine.LEVERAGED_INDEX}=1"]
            )
        with self.assertRaisesRegex(ValueError, "Invalid"):
            engine.parse_executed_shares(
                [f"{engine.MARKET_INDEX}=1", "CASH=0"]
            )

    def test_workflow_keeps_stateful_ref_guard_and_artifacts(self):
        workflow = (
            PROJECT_ROOT / ".github" / "workflows" / "run_portfolio.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("Guard stateful production ref", workflow)
        self.assertIn("roth-ira-state", workflow)
        self.assertIn("roth_ira_shadow_ledger.jsonl", workflow)
        self.assertNotIn("downside_shadow", workflow)
        self.assertNotIn("alpha_research.py", workflow)
        self.assertIn("roth-ira-decision", workflow)
        self.assertIn("--confirm-execution", workflow)
        self.assertIn("--sync-holdings", workflow)


if __name__ == "__main__":
    unittest.main()
