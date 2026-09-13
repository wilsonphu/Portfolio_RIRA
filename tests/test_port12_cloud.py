from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import port12_cloud as portfolio


def price_frame(*, bullish: bool = True, periods: int = 280) -> pd.DataFrame:
    index = pd.bdate_range("2025-11-03", periods=periods)
    frame = pd.DataFrame(100.0, index=index, columns=portfolio.ALL_TICKERS)
    if bullish:
        frame.loc[index[-2], portfolio.MARKET_INDEX] = 110.0
        frame.loc[index[-1], portfolio.MARKET_INDEX] = 111.0
    else:
        frame.loc[index[-1], portfolio.MARKET_INDEX] = 80.0
    return frame


def decision(active: bool) -> portfolio.StrategyDecision:
    router = portfolio.core.EquityRouterState(
        tqqq_active=active,
        bullish_streak=2 if active else 0,
        last_processed_signal_date="2026-09-11",
    )
    return portfolio.StrategyDecision(
        target_weights=portfolio.target_weights(active),
        router_state=router,
        transition_reason="TQQQ_HOLD" if active else "UPRO_HOLD",
        structural_change=False,
        trend_positive=active,
        qqq_close=110.0 if active else 90.0,
        qqq_sma_200=100.0,
        qqq_sma_50=102.0,
        qqq_momentum_252=0.10,
        lifecycle_stage=portfolio.LIFECYCLE_SPRINT,
    )


def run_fixture() -> portfolio.StrategyRun:
    prices = price_frame()
    state = portfolio.PortfolioState(
        shares={"TQQQ": 40.0, "DBMF": 20.0, "ZROZ": 20.0, "UGL": 20.0},
        target_weights=portfolio.target_weights(True),
        strategy_initialized=True,
        tqqq_active=True,
        tqqq_bullish_streak=2,
        last_processed_signal_date="2026-09-11",
        executed_tqqq_active=True,
        executed_strategy_fingerprint=portfolio.STRATEGY_FINGERPRINT,
        last_completed_annual_rebalance_year=2026,
    )
    current = portfolio.target_weights(True)
    plan = portfolio.RebalancePlan(current, False, False, "HOLD", 0.0, 0)
    table = portfolio.calculate_execution_table(prices, current, 10_000.0, state, actionable=False)
    diagnostics = portfolio.calculate_execution_diagnostics(table, 10_000.0, current, current, current)
    return portfolio.StrategyRun(
        prices, decision(True), state, deepcopy(state), 10_000.0, current,
        table, pd.Timestamp("2026-09-11"), "1" * 64, plan, diagnostics,
    )


def action_run_fixture() -> portfolio.StrategyRun:
    """A coherent TQQQ-to-UPRO action, including its executable trade rows."""
    run = run_fixture()
    routed = decision(False)
    plan = portfolio.build_rebalance_plan(run.current_weights, routed, run.state)
    table = portfolio.calculate_execution_table(
        run.price_data,
        plan.execution_weights,
        run.portfolio_value,
        run.state,
        actionable=True,
    )
    diagnostics = portfolio.calculate_execution_diagnostics(
        table,
        run.portfolio_value,
        run.current_weights,
        routed.target_weights,
        plan.execution_weights,
    )
    return replace(
        run,
        decision=routed,
        execution_table=table,
        rebalance_plan=plan,
        execution_diagnostics=diagnostics,
    )


class StrategyTests(unittest.TestCase):
    def test_configuration_and_fingerprint_are_stable(self):
        portfolio.validate_configuration()
        self.assertEqual(portfolio.STRATEGY_FINGERPRINT, portfolio.calculate_strategy_fingerprint())

    def test_initialization_replays_two_completed_bullish_closes(self):
        prices = price_frame()
        result = portfolio.calculate_strategy_decision(prices, portfolio.PortfolioState())
        self.assertTrue(result.router_state.tqqq_active)
        self.assertEqual(result.processed_signal_dates, tuple(item.date().isoformat() for item in prices.index[-2:]))

    def test_bearish_close_exits_tqqq_immediately(self):
        prices = price_frame(bullish=False)
        state = portfolio.PortfolioState(
            strategy_initialized=True,
            tqqq_active=True,
            tqqq_bullish_streak=9,
            last_processed_signal_date=prices.index[-2].date().isoformat(),
        )
        result = portfolio.calculate_strategy_decision(prices, state)
        self.assertFalse(result.router_state.tqqq_active)
        self.assertEqual(result.transition_reason, "TREND_SWITCH_TO_UPRO")

    def test_same_date_run_is_idempotent(self):
        prices = price_frame()
        state = portfolio.PortfolioState(
            strategy_initialized=True,
            tqqq_active=False,
            tqqq_bullish_streak=1,
            last_processed_signal_date=prices.index[-1].date().isoformat(),
        )
        result = portfolio.calculate_strategy_decision(prices, state)
        self.assertFalse(result.router_state.tqqq_active)
        self.assertEqual(result.router_state.bullish_streak, 1)

    def test_missed_sessions_are_replayed(self):
        prices = price_frame()
        state = portfolio.PortfolioState(
            strategy_initialized=True,
            last_processed_signal_date=prices.index[-3].date().isoformat(),
        )
        result = portfolio.calculate_strategy_decision(prices, state)
        self.assertEqual(len(result.processed_signal_dates), 2)
        self.assertTrue(result.router_state.tqqq_active)

    def test_lifecycle_ceiling_scales_risk_into_sgov(self):
        weights = portfolio.target_weights(True, portfolio.LIFECYCLE_PHI)
        self.assertAlmostEqual(portfolio.advertised_daily_exposure(weights), (1 + 5**0.5) / 2)
        self.assertGreater(weights[portfolio.TREASURY_RESERVE], 0)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_lifecycle_ratchet_never_moves_backward(self):
        chosen = portfolio.select_lifecycle_stage(
            portfolio.LIFECYCLE_ONE_THREE, 10_000.0, pd.Timestamp("2026-09-11").date()
        )
        self.assertEqual(chosen.stage, portfolio.LIFECYCLE_ONE_THREE)


class HoldingsAndRebalanceTests(unittest.TestCase):
    def test_confirmed_shares_drive_value_and_weights(self):
        prices = price_frame()
        state = portfolio.PortfolioState(shares={"TQQQ": 3.0}, cash_balance=200.0)
        self.assertEqual(portfolio.existing_portfolio_value(state, prices), 500.0)
        self.assertEqual(portfolio.existing_weights(state, prices), {"TQQQ": 0.6, "CASH": 0.4})

    def test_nan_holding_price_fails_closed(self):
        prices = price_frame()
        prices.loc[prices.index[-1], "TQQQ"] = np.nan
        with self.assertRaises(RuntimeError):
            portfolio.existing_portfolio_value(portfolio.PortfolioState(shares={"TQQQ": 1.0}), prices)

    def test_legacy_soxl_is_explicitly_sold(self):
        state = portfolio.PortfolioState(
            executed_tqqq_active=True,
            executed_strategy_fingerprint=portfolio.STRATEGY_FINGERPRINT,
            last_completed_annual_rebalance_year=2026,
            shares={"TQQQ": 39.0, "SOXL": 1.0, "DBMF": 20.0, "ZROZ": 20.0, "UGL": 20.0},
        )
        existing = {"TQQQ": 0.39, "SOXL": 0.01, "DBMF": 0.2, "ZROZ": 0.2, "UGL": 0.2}
        plan = portfolio.build_rebalance_plan(existing, decision(True), state)
        self.assertTrue(plan.full_transition)
        self.assertEqual(plan.execution_weights.get("SOXL", 0), 0)
        table = portfolio.calculate_execution_table(price_frame(), plan.execution_weights, 10_000.0, state, actionable=True)
        self.assertEqual(table.set_index("Ticker").loc["SOXL", "Action"], "SELL")

    def test_exact_holdings_confirm_new_strategy_without_trade(self):
        state = portfolio.PortfolioState(
            executed_tqqq_active=True,
            last_completed_annual_rebalance_year=2026,
        )
        current = portfolio.target_weights(True)
        plan = portfolio.build_rebalance_plan(current, decision(True), state)
        self.assertFalse(plan.rebalance_due)
        self.assertEqual(plan.reason, "CONFIRMED_TARGET_STATE")

    def test_router_switch_preserves_non_equity_weights(self):
        state = portfolio.PortfolioState(
            executed_tqqq_active=True,
            executed_strategy_fingerprint=portfolio.STRATEGY_FINGERPRINT,
            last_completed_annual_rebalance_year=2026,
        )
        current = {"TQQQ": 0.44, "DBMF": 0.18, "ZROZ": 0.19, "UGL": 0.17, "CASH": 0.02}
        plan = portfolio.build_rebalance_plan(current, decision(False), state)
        self.assertTrue(plan.rebalance_due)
        self.assertFalse(plan.full_transition)
        self.assertEqual(plan.execution_weights, {"DBMF": 0.18, "ZROZ": 0.19, "UGL": 0.17, "CASH": 0.02, "UPRO": 0.44})

    def test_midyear_drift_waits_for_annual_rebalance(self):
        state = portfolio.PortfolioState(
            executed_tqqq_active=True,
            executed_strategy_fingerprint=portfolio.STRATEGY_FINGERPRINT,
            last_completed_annual_rebalance_year=2026,
        )
        current = {"TQQQ": 0.50, "DBMF": 0.15, "ZROZ": 0.15, "UGL": 0.20}
        plan = portfolio.build_rebalance_plan(current, decision(True), state)
        self.assertFalse(plan.rebalance_due)
        self.assertEqual(plan.reason, "HOLD")

    def test_new_calendar_year_forces_exact_rebalance(self):
        state = portfolio.PortfolioState(
            executed_tqqq_active=True,
            executed_strategy_fingerprint=portfolio.STRATEGY_FINGERPRINT,
            last_completed_annual_rebalance_year=2025,
        )
        current = {"TQQQ": 0.39, "DBMF": 0.21, "ZROZ": 0.20, "UGL": 0.20}
        plan = portfolio.build_rebalance_plan(current, decision(True), state)
        self.assertTrue(plan.rebalance_due)
        self.assertTrue(plan.annual_rebalance_due)
        self.assertEqual(plan.annual_rebalance_year, 2026)
        self.assertEqual(plan.execution_weights, {**portfolio.target_weights(True), "CASH": 0.0})

    def test_exact_new_year_allocation_sends_one_review(self):
        run = run_fixture()
        run.state.last_completed_annual_rebalance_year = 2025
        plan = portfolio.build_rebalance_plan(run.current_weights, run.decision, run.state)
        run = replace(run, rebalance_plan=plan)
        self.assertFalse(plan.rebalance_due)
        self.assertEqual(plan.reason, "ANNUAL_REVIEW")
        notice = portfolio.decide_notification(run)
        self.assertEqual(notice.kind, "ANNUAL_REVIEW")
        with patch.object(portfolio, "save_state"):
            portfolio.persist_notification_delivery(run, notice, delivered_decision_hash="3" * 64)
        self.assertEqual(run.state.last_completed_annual_rebalance_year, 2026)


class StateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state_path = Path(self.temp.name) / "state.json"
        self.patcher = patch.object(portfolio, "STATE_FILE", self.state_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp.cleanup()

    def test_version_14_migration_preserves_legacy_shares_cash_and_pending(self):
        payload = {
            "state_version": 14,
            "shares": {"SOXL": 1.5, "GLD": 2.0},
            "cash_balance": 123.45,
            "portfolio_value": 999.0,
            "target_weights": {"SOXL": 0.15, "GLD": 0.85},
            "pending_recommendation_date": "2026-09-10",
            "pending_recommendation_weights": {"SOXL": 0.15, "GLD": 0.85},
            "pending_recommendation_notified": True,
        }
        self.state_path.write_text(json.dumps(payload), encoding="utf-8")
        state = portfolio.load_state(backup_legacy=False)
        self.assertEqual(state.shares, {"SOXL": 1.5, "GLD": 2.0})
        self.assertEqual(state.cash_balance, 123.45)
        self.assertEqual(state.pending_recommendation_date, "2026-09-10")
        self.assertFalse(state.strategy_initialized)

    def test_version_15_migration_defers_annual_rebalance_until_next_year(self):
        payload = {
            "state_version": 15,
            "shares": {"TQQQ": 1.0},
            "cash_balance": 0.0,
            "portfolio_value": 100.0,
            "last_processed_signal_date": "2026-09-11",
        }
        self.state_path.write_text(json.dumps(payload), encoding="utf-8")
        state = portfolio.load_state(backup_legacy=False)
        self.assertEqual(state.last_completed_annual_rebalance_year, 2026)

    def test_atomic_save_failure_preserves_original(self):
        self.state_path.write_text("original", encoding="utf-8")
        with patch.object(portfolio.os, "replace", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                portfolio.save_state(portfolio.PortfolioState())
        self.assertEqual(self.state_path.read_text(encoding="utf-8"), "original")

    def test_confirm_execution_stores_actual_shares(self):
        state = portfolio.PortfolioState(
            pending_recommendation_date="2026-09-11",
            pending_recommendation_weights=portfolio.target_weights(True),
            pending_recommendation_tqqq_active=True,
            pending_recommendation_lifecycle_stage=portfolio.LIFECYCLE_SPRINT,
            pending_recommendation_notified=True,
            pending_recommendation_fingerprint=portfolio.STRATEGY_FINGERPRINT,
            pending_recommendation_annual_year=2026,
        )
        portfolio.save_state(state)
        confirmed = portfolio.confirm_execution({"TQQQ": 12.25}, 3.21, "2026-09-11")
        self.assertEqual(confirmed.shares, {"TQQQ": 12.25})
        self.assertEqual(confirmed.cash_balance, 3.21)
        self.assertFalse(confirmed.pending_recommendation_date)
        self.assertEqual(confirmed.last_completed_annual_rebalance_year, 2026)


class NotificationTests(unittest.TestCase):
    def test_hold_is_silent(self):
        self.assertEqual(portfolio.decide_notification(run_fixture()).kind, "NONE")

    def test_identical_delivered_pending_is_suppressed(self):
        run = run_fixture()
        run.state.pending_recommendation_date = "2026-09-10"
        run.state.pending_recommendation_weights = portfolio.target_weights(True)
        run.state.pending_recommendation_tqqq_active = True
        run.state.pending_recommendation_lifecycle_stage = portfolio.LIFECYCLE_SPRINT
        run.state.pending_recommendation_notified = True
        run.state.pending_recommendation_fingerprint = portfolio.STRATEGY_FINGERPRINT
        run = replace(run, rebalance_plan=portfolio.RebalancePlan(portfolio.target_weights(True), True, True, "TEST", 0.1, 1))
        self.assertEqual(portfolio.decide_notification(run).kind, "NONE")

    def test_cancellation_is_sent_once_then_cleared(self):
        run = run_fixture()
        run.state.pending_recommendation_date = "2026-09-10"
        run.state.pending_recommendation_weights = portfolio.target_weights(True)
        run.state.pending_recommendation_tqqq_active = True
        run.state.pending_recommendation_lifecycle_stage = portfolio.LIFECYCLE_SPRINT
        run.state.pending_recommendation_notified = True
        run.state.pending_recommendation_fingerprint = portfolio.STRATEGY_FINGERPRINT
        notice = portfolio.decide_notification(run)
        self.assertEqual(notice.kind, "CANCELLATION")
        with patch.object(portfolio, "save_state"):
            portfolio.persist_notification_delivery(run, notice, delivered_decision_hash="2" * 64)
        self.assertFalse(run.state.pending_recommendation_date)

    def test_missing_email_credentials_fail_only_when_send_is_called(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(portfolio.decide_notification(run_fixture()).kind, "NONE")
            with self.assertRaises(RuntimeError):
                portfolio.send_email("subject", "text", "<p>html</p>")

    def test_material_update_replaces_pending_action(self):
        run = run_fixture()
        run.state.pending_recommendation_date = "2026-09-10"
        run.state.pending_recommendation_weights = portfolio.target_weights(False)
        run.state.pending_recommendation_tqqq_active = False
        run.state.pending_recommendation_lifecycle_stage = portfolio.LIFECYCLE_SPRINT
        run.state.pending_recommendation_notified = True
        run.state.pending_recommendation_fingerprint = portfolio.STRATEGY_FINGERPRINT
        run = replace(run, rebalance_plan=portfolio.RebalancePlan(portfolio.target_weights(True), True, True, "TREND_SWITCH_TO_TQQQ", 0.4, 2))
        notice = portfolio.decide_notification(run)
        self.assertEqual(notice.kind, "UPDATE")
        portfolio.prepare_notification_delivery(run, notice)
        self.assertEqual(run.state.pending_recommendation_weights, portfolio.target_weights(True))
        self.assertEqual(run.state.pending_recommendation_supersedes_date, "2026-09-10")

    def test_smtp_failure_retains_pending_action_and_confirmed_shares(self):
        run = action_run_fixture()
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            audit_path = Path(directory) / "audit.json"
            with (
                patch.object(sys, "argv", ["port12_cloud.py"]),
                patch.object(portfolio, "STATE_FILE", state_path),
                patch.object(portfolio, "DECISION_AUDIT_FILE", audit_path),
                patch.object(portfolio, "configure_logging"),
                patch.object(portfolio, "run_strategy", return_value=run),
                patch.object(portfolio, "send_email", side_effect=RuntimeError("smtp failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "smtp failed"):
                    portfolio.main()
            restored = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(restored["shares"], run.state.shares)
            self.assertEqual(restored["pending_recommendation_date"], "2026-09-11")
            self.assertFalse(restored["pending_recommendation_notified"])


class DecisionAuditTests(unittest.TestCase):
    def test_audit_records_lifecycle_rationale(self):
        run = run_fixture()
        audit = portfolio.build_decision_audit(
            run, portfolio.NotificationDecision("ACTION", "NEW_RECOMMENDATION"), "STAGED"
        )
        self.assertEqual(audit["schema_version"], 7)
        lifecycle = audit["lifecycle"]
        self.assertEqual(lifecycle["stage"], run.decision.lifecycle_stage)
        self.assertEqual(lifecycle["value_stage"], run.decision.lifecycle_value_stage)
        self.assertEqual(lifecycle["age_stage"], run.decision.lifecycle_age_stage)
        self.assertEqual(lifecycle["advanced"], run.decision.lifecycle_stage_advanced)

    def test_audit_records_exposure_diagnostics(self):
        run = run_fixture()
        audit = portfolio.build_decision_audit(
            run, portfolio.NotificationDecision("ACTION", "NEW_RECOMMENDATION"), "STAGED"
        )
        exposure = audit["exposure"]
        self.assertEqual(
            exposure["strategic"], run.execution_diagnostics.strategic_daily_exposure
        )
        self.assertEqual(
            exposure["destination"], run.execution_diagnostics.destination_daily_exposure
        )
        self.assertEqual(
            sorted(exposure["estimated_costs_by_bps"]),
            sorted(str(bps) for bps in portfolio.TRANSACTION_COST_SCENARIOS_BPS),
        )

    def test_audit_payload_is_canonically_hashable(self):
        run = run_fixture()
        notice = portfolio.NotificationDecision("ACTION", "NEW_RECOMMENDATION")
        first = portfolio.build_decision_audit(run, notice, "STAGED")
        second = portfolio.build_decision_audit(run, notice, "STAGED")
        self.assertEqual(first["decision_hash"], second["decision_hash"])
        self.assertEqual(len(first["decision_hash"]), 64)


class RenderingTests(unittest.TestCase):
    def test_action_dashboard_lists_both_sides_of_router_switch(self):
        dashboard = portfolio.build_dashboard(action_run_fixture())
        self.assertIn("TQQQ", dashboard)
        self.assertIn("SELL", dashboard)
        self.assertIn("UPRO", dashboard)
        self.assertIn("BUY", dashboard)

    def test_hold_dashboard_remains_compact_and_omits_trade_table(self):
        dashboard = portfolio.build_dashboard(run_fixture())
        self.assertIn("NO TRADES", dashboard)
        self.assertNotIn("TRADES  (estimated", dashboard)

    def test_action_email_contains_trade_and_diagnostic_rows(self):
        run = action_run_fixture()
        html = portfolio.build_email_html(
            run, portfolio.NotificationDecision("ACTION", "NEW_RECOMMENDATION")
        )
        self.assertIn("ACTION REQUIRED", html)
        self.assertIn("TQQQ", html)
        self.assertIn("UPRO", html)
        self.assertIn("Turnover 40.0% one-way", html)


class DataAndCliTests(unittest.TestCase):
    def test_missing_yfinance_ticker_is_rejected(self):
        index = pd.DatetimeIndex(["2026-09-11"])
        columns = pd.MultiIndex.from_product([["Close"], ["QQQ"]])
        raw = pd.DataFrame([[100.0]], index=index, columns=columns)
        with self.assertRaises(RuntimeError):
            portfolio._extract_yfinance_prices(raw, ["QQQ", "TQQQ"])

    def test_partial_latest_prices_are_rejected(self):
        frame = price_frame()
        frame.loc[frame.index[-1], "UGL"] = np.nan
        with self.assertRaises(RuntimeError):
            portfolio._require_positive(frame.loc[frame.index[-1], list(portfolio.ALL_TICKERS)], "latest")

    def test_latest_completed_nyse_session(self):
        after_close = datetime(2026, 9, 11, 17, 0, tzinfo=ZoneInfo("America/New_York"))
        self.assertEqual(portfolio.expected_completed_session(after_close), pd.Timestamp("2026-09-11"))
        during_market = datetime(2026, 9, 11, 15, 0, tzinfo=ZoneInfo("America/New_York"))
        with self.assertRaises(RuntimeError):
            portfolio.expected_completed_session(during_market)

    def test_complete_holdings_parser_requires_cash(self):
        with self.assertRaises(ValueError):
            portfolio.parse_executed_shares(["TQQQ=1"])
        shares, cash = portfolio.parse_executed_shares(["TQQQ=1", "SOXL=2", "CASH=3"])
        self.assertEqual(shares, {"TQQQ": 1.0, "SOXL": 2.0})
        self.assertEqual(cash, 3.0)

    def test_test_mode_never_persists_or_sends(self):
        run = run_fixture()
        with (
            patch.object(sys, "argv", ["port12_cloud.py", "--test", "--roth-amount", "10000"]),
            patch.object(portfolio, "run_strategy", return_value=run),
            patch.object(portfolio, "save_state") as save,
            patch.object(portfolio, "send_email") as send,
            patch.object(portfolio, "write_decision_audit") as audit,
        ):
            portfolio.main()
        save.assert_not_called()
        send.assert_not_called()
        audit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
