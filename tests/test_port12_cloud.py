import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import port12_cloud as portfolio


def prices():
    index = pd.date_range("2026-01-02", periods=3, freq="B")
    return pd.DataFrame(
        {
            "TQQQ": [100.0, 101.0, 102.0],
            "DBMF": [100.0, 100.5, 101.0],
            "UGL": [100.0, 99.0, 101.0],
            "ZROZ": [100.0, 101.0, 100.0],
            "BTAL": [100.0, 100.2, 100.1],
        },
        index=index,
    )


class StaticEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state_path = Path(self.temp.name) / "roth_ira_state.json"
        self.audit_path = Path(self.temp.name) / "audit.json"
        self.state_patch = patch.object(portfolio, "STATE_FILE", self.state_path)
        self.state_patch.start()

    def tearDown(self):
        self.state_patch.stop()
        self.temp.cleanup()

    def test_decision_never_reads_signals(self):
        state = portfolio.PortfolioState()
        decision = portfolio.calculate_strategy_decision(prices(), state)
        self.assertEqual(decision.target_weights, {**portfolio.core.target_weights(), "CASH": 0.0})
        self.assertEqual(decision.transition_reason, "STATIC_ANNUAL_HOLD")

    def test_initial_all_cash_creates_full_annual_target(self):
        state = portfolio.PortfolioState(cash_balance=10000.0)
        current = portfolio.existing_weights(state, prices())
        decision = portfolio.calculate_strategy_decision(prices(), state)
        plan = portfolio.build_rebalance_plan(current, decision, state)
        self.assertTrue(plan.rebalance_due)
        self.assertEqual(plan.execution_weights["TQQQ"], 0.35)
        self.assertEqual(plan.execution_weights["BTAL"], 0.05)

    def test_midyear_static_hold_is_silent(self):
        state = portfolio.PortfolioState(
            shares={"TQQQ": 35, "DBMF": 25, "UGL": 20, "ZROZ": 15, "BTAL": 5},
            target_weights=portfolio.target_with_cash(),
            last_completed_annual_rebalance_year=2026,
            executed_strategy_fingerprint=portfolio.STRATEGY_FINGERPRINT,
        )
        state.cash_balance = 0.0
        current = portfolio.existing_weights(state, prices())
        decision = portfolio.calculate_strategy_decision(prices(), state)
        plan = portfolio.build_rebalance_plan(current, decision, state)
        self.assertFalse(plan.rebalance_due)
        self.assertEqual(plan.reason, "HOLD")

    def test_unsupported_holdings_are_rejected(self):
        state = portfolio.PortfolioState(
            shares={"TQQQ": 10.0}, cash_balance=0.0,
            last_completed_annual_rebalance_year=2026,
            executed_strategy_fingerprint=portfolio.STRATEGY_FINGERPRINT,
        )
        current = portfolio.existing_weights(state, prices())
        decision = portfolio.calculate_strategy_decision(prices(), state)
        plan = portfolio.build_rebalance_plan(current, decision, state)
        self.assertFalse(plan.rebalance_due)

        with self.assertRaises(ValueError):
            portfolio.parse_executed_shares(["SOXL=10", "CASH=0"])

    def test_parse_complete_holdings(self):
        shares, cash = portfolio.parse_executed_shares(["TQQQ=1.5", "CASH=20"])
        self.assertEqual(shares, {"TQQQ": 1.5})
        self.assertEqual(cash, 20.0)
        with self.assertRaises(ValueError):
            portfolio.parse_executed_shares(["TQQQ=1"])

    def test_annual_contribution_has_no_price_signal(self):
        state = portfolio.PortfolioState(
            contribution_plan_year=2026,
            contribution_policy_revision=portfolio.contribution.POLICY_REVISION,
            contribution_budget=7500.0,
        )
        decision = portfolio.calculate_strategy_decision(prices(), state)
        plan = portfolio.RebalancePlan(portfolio.target_with_cash(), False, False, "HOLD", 0.0, 0, True, 2026)
        contribution_plan = portfolio.build_contribution_plan(state, prices(), 10000.0, decision, plan)
        self.assertEqual(contribution_plan.due_amount, 7500.0)
        self.assertEqual(contribution_plan.reasons, ("ANNUAL_CONTRIBUTION",))

    def test_test_mode_never_persists(self):
        with patch.object(portfolio, "download_market_data", return_value=prices()), patch.object(
            portfolio, "ROTH_IRA_AMOUNT", 10000.0
        ), patch("sys.argv", ["port12_cloud.py", "--test"]):
            portfolio.main()
        self.assertFalse(self.state_path.exists())

    def test_dashboard_snapshot_sends_without_persisting(self):
        with patch.object(portfolio, "download_market_data", return_value=prices()), patch.object(
            portfolio, "ROTH_IRA_AMOUNT", 10000.0
        ), patch.object(portfolio, "send_email") as send_email, patch(
            "sys.argv", ["port12_cloud.py", "--send-dashboard-email"]
        ):
            portfolio.main()
        send_email.assert_called_once()
        self.assertIn("CURRENT HOLDINGS", send_email.call_args.args[1])
        self.assertIn("Current holdings", send_email.call_args.args[2])
        self.assertIn("Target allocation", send_email.call_args.args[2])
        self.assertIn("<table", send_email.call_args.args[2])
        self.assertFalse(self.state_path.exists())

    def test_dashboard_shows_current_holdings_and_orders(self):
        with patch.object(portfolio, "download_market_data", return_value=prices()), patch.object(
            portfolio, "ROTH_IRA_AMOUNT", 10000.0
        ):
            run = portfolio.run_strategy(10000.0)
        dashboard = portfolio.build_dashboard(run)
        self.assertIn("CURRENT HOLDINGS", dashboard)
        self.assertIn("CASH", dashboard)
        self.assertIn("ORDERS", dashboard)
        self.assertIn("TQQQ", dashboard)

    def test_migration_preserves_supported_holdings_and_forces_revision(self):
        self.state_path.write_text(json.dumps({
            "state_version": 19,
            "shares": {"TQQQ": 2.0},
            "cash_balance": 4.0,
            "portfolio_value": 204.0,
            "contribution_plan_year": 0,
        }), encoding="utf-8")
        state = portfolio.load_state()
        self.assertEqual(state.state_version, portfolio.STATE_VERSION)
        self.assertEqual(state.shares, {"TQQQ": 2.0})
        self.assertEqual(state.executed_strategy_fingerprint, "")


if __name__ == "__main__":
    unittest.main()
