import copy
import json
import sys
import tempfile
import types
import unittest
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import port12_cloud as engine  # noqa: E402


SIGNAL_DATE = pd.Timestamp("2026-08-04")


def market_frames(rows=230, *, end=SIGNAL_DATE):
    index = pd.bdate_range(end=end, periods=rows)
    step = np.arange(rows, dtype=float)
    prices = pd.DataFrame(
        {
            ticker: 100.0
            + position
            + 0.12 * step
            + 0.35 * np.sin(step / (3.0 + position))
            for position, ticker in enumerate(engine.ALL_TICKERS)
        },
        index=index,
    )
    volumes = pd.DataFrame(
        {
            ticker: 1_000_000.0 + 1_000.0 * np.cos(step / (5.0 + position))
            for position, ticker in enumerate(engine.ALL_TICKERS)
        },
        index=index,
    )
    payload = pd.concat({"Close": prices, "Volume": volumes}, axis=1)
    return prices, volumes, payload


def one_row_prices(price=100.0):
    return pd.DataFrame(
        {ticker: [float(price)] for ticker in engine.ALL_TICKERS},
        index=pd.DatetimeIndex([SIGNAL_DATE]),
    )


def complete_latest(
    *,
    bullish=1,
    volatility=0.10,
    soxl_momentum=0.08,
    tecl_momentum=0.02,
):
    return pd.Series(
        {
            "sma_200": 100.0,
            "donchian_mid": 100.0,
            "vwma_50": 100.0,
            "sma_signal": int(bool(bullish)),
            "donchian_signal": int(bool(bullish)),
            "vwma_signal": int(bool(bullish)),
            "bullish_consensus": int(bool(bullish)),
            "volatility_10": volatility,
            "volatility_30": volatility * 0.9,
            "annualized_volatility": volatility,
            "soxl_momentum": soxl_momentum,
            "tecl_momentum": tecl_momentum,
        }
    )


def weights_with_cash(weights):
    result = dict(weights)
    result.setdefault(engine.CASH_ASSET, 0.0)
    return result


def make_result(*, tier="LOW", leader=None, regime="BULL", weights=None):
    leader = leader or engine.LEVERAGED_SEMICONDUCTOR
    if weights is None:
        if regime == "BEAR":
            weights = dict(engine.BEAR_ALLOCATION)
            tier = "N/A"
        else:
            template = {
                "LOW": engine.LOW_VOL_ALLOCATION,
                "MODERATE": engine.MODERATE_VOL_ALLOCATION,
                "HIGH": engine.HIGH_VOL_ALLOCATION,
            }[tier]
            weights = engine.apply_allocation_template(template, leader)
    return engine.StrategyResult(
        target_weights=dict(weights),
        regime=regime,
        leader=leader,
        volatility_tier=tier,
        annualized_volatility=0.10 if tier != "HIGH" else 0.25,
        raw_volatility_tier=tier,
    )


def pending_state(
    weights,
    *,
    tier="LOW",
    leader=None,
    regime="BULL",
    shares=None,
    notified=True,
    supersedes_date="",
):
    leader = leader or engine.LEVERAGED_SEMICONDUCTOR
    return engine.PortfolioState(
        shares=dict(shares or {}),
        pending_recommendation_date=SIGNAL_DATE.date().isoformat(),
        pending_recommendation_leader=leader,
        pending_recommendation_regime=regime,
        pending_recommendation_tier=tier,
        pending_recommendation_weights=dict(weights),
        pending_recommendation_notified=notified,
        pending_recommendation_supersedes_date=supersedes_date,
    )


def make_strategy_run(
    *,
    state=None,
    result=None,
    execution_weights=None,
    rebalance_due=False,
    individual_orders=None,
    reason=None,
    signal_date=SIGNAL_DATE,
):
    state = state or engine.PortfolioState()
    result = result or make_result()
    execution_weights = (
        dict(execution_weights)
        if execution_weights is not None
        else weights_with_cash(result.target_weights)
    )
    if individual_orders is None:
        individual_orders = 2 if rebalance_due else 0
    if reason is None:
        reason = "DRIFT_BAND" if rebalance_due else "HOLD"
    plan = engine.RebalancePlan(
        execution_weights=execution_weights,
        rebalance_due=rebalance_due,
        full_transition=reason != "DRIFT_BAND" and rebalance_due,
        reason=reason,
        one_way_turnover=0.10 if rebalance_due else 0.0,
        individual_orders=individual_orders,
    )
    tier = result.volatility_tier
    tier_decision = engine.VolatilityTierDecision(
        tier=tier,
        raw_tier=tier,
        transition="UNCHANGED",
    )
    return engine.StrategyRun(
        price_data=one_row_prices(),
        latest_indicators=complete_latest(
            bullish=result.regime == "BULL",
            volatility=result.annualized_volatility,
        ),
        result=result,
        state=state,
        planning_state=copy.deepcopy(state),
        portfolio_value=1_000.0,
        current_weights={},
        execution_table=pd.DataFrame(),
        signal_date=pd.Timestamp(signal_date),
        sector_review_due=False,
        rebalance_plan=plan,
        tier_decision=tier_decision,
    )


class EngineTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temporary_directory.name)
        self.state_file = self.temp_path / "state.json"
        self.log_file = self.temp_path / "engine.log"
        self.state_patch = mock.patch.object(engine, "STATE_FILE", self.state_file)
        self.log_patch = mock.patch.object(engine, "LOG_FILE", self.log_file)
        self.state_patch.start()
        self.log_patch.start()

    def tearDown(self):
        for handler in list(engine.logger.handlers):
            handler.close()
        engine.logger.handlers.clear()
        self.log_patch.stop()
        self.state_patch.stop()
        self.temporary_directory.cleanup()

    def write_state(self, state_or_payload):
        payload = (
            asdict(state_or_payload)
            if isinstance(state_or_payload, engine.PortfolioState)
            else state_or_payload
        )
        self.state_file.write_text(
            json.dumps(payload, allow_nan=True),
            encoding="utf-8",
        )

    def download_from_payload(self, payload, expected_session):
        fake_yfinance = types.ModuleType("yfinance")
        fake_yfinance.download = mock.Mock(return_value=payload)
        with mock.patch.dict(sys.modules, {"yfinance": fake_yfinance}), mock.patch.object(
            engine,
            "expected_completed_session",
            return_value=pd.Timestamp(expected_session).normalize(),
        ):
            result = engine.download_market_data(engine.ALL_TICKERS)
        return result, fake_yfinance.download


class OriginalStrategyTests(EngineTestCase):
    def test_all_locked_allocations_and_leader_follower_resolution(self):
        cases = [
            (
                "LOW",
                engine.LEVERAGED_SEMICONDUCTOR,
                {
                    engine.LEVERAGED_SEMICONDUCTOR: 0.45,
                    engine.LEVERAGED_TECH: 0.15,
                    engine.SEMICONDUCTOR_ETF: 0.25,
                    engine.LEVERAGED_INDEX: 0.15,
                },
            ),
            (
                "LOW",
                engine.LEVERAGED_TECH,
                {
                    engine.LEVERAGED_TECH: 0.45,
                    engine.LEVERAGED_SEMICONDUCTOR: 0.15,
                    engine.SEMICONDUCTOR_ETF: 0.25,
                    engine.LEVERAGED_INDEX: 0.15,
                },
            ),
            (
                "MODERATE",
                engine.LEVERAGED_SEMICONDUCTOR,
                {
                    engine.LEVERAGED_SEMICONDUCTOR: 0.25,
                    engine.LEVERAGED_TECH: 0.10,
                    engine.SEMICONDUCTOR_ETF: 0.45,
                    engine.LEVERAGED_INDEX: 0.20,
                },
            ),
            (
                "HIGH",
                engine.LEVERAGED_TECH,
                {
                    engine.SEMICONDUCTOR_ETF: 0.85,
                    engine.HEDGE_ASSET: 0.15,
                },
            ),
        ]
        for tier, leader, expected in cases:
            with self.subTest(tier=tier, leader=leader):
                latest = complete_latest(volatility=0.10)
                decision = engine.VolatilityTierDecision(tier, tier)
                result = engine.determine_target_allocation(
                    latest,
                    leader,
                    allow_leader_review=False,
                    tier_decision=decision,
                )
                self.assertEqual(result.regime, "BULL")
                self.assertEqual(result.leader, leader)
                self.assertEqual(result.volatility_tier, tier)
                self.assertEqual(result.target_weights, expected)

        bear = engine.determine_target_allocation(
            complete_latest(bullish=0, volatility=0.25),
            engine.LEVERAGED_TECH,
            allow_leader_review=False,
            tier_decision=engine.VolatilityTierDecision("N/A", "N/A"),
        )
        self.assertEqual(bear.regime, "BEAR")
        self.assertEqual(
            bear.target_weights,
            {engine.DEFENSIVE_EQUITY: 0.80, engine.HEDGE_ASSET: 0.20},
        )

    def test_indicator_consensus_is_two_of_three_and_volatility_uses_maximum(self):
        prices, volumes, _ = market_frames()
        rows = len(prices)
        prices[engine.MARKET_INDEX] = np.concatenate(
            (
                np.full(rows - 50, 200.0),
                np.full(49, 100.0),
                np.array([150.0]),
            )
        )
        latest = engine.calculate_indicators(prices, volumes).iloc[-1]
        self.assertEqual(
            (
                int(latest["sma_signal"]),
                int(latest["donchian_signal"]),
                int(latest["vwma_signal"]),
            ),
            (0, 1, 1),
        )
        self.assertEqual(int(latest["bullish_consensus"]), 1)
        self.assertAlmostEqual(
            latest["annualized_volatility"],
            max(latest["volatility_10"], latest["volatility_30"]),
        )

    def test_volatility_threshold_boundaries(self):
        self.assertEqual(engine.classify_volatility(0.149999), "LOW")
        self.assertEqual(engine.classify_volatility(0.15), "MODERATE")
        self.assertEqual(engine.classify_volatility(0.22), "MODERATE")
        self.assertEqual(engine.classify_volatility(0.220001), "HIGH")

    def test_defensive_tier_transition_is_immediate(self):
        decision = engine.classify_volatility_with_persistence(
            0.23,
            "LOW",
            "",
            0,
        )
        self.assertEqual(decision.tier, "HIGH")
        self.assertEqual(decision.transition, "DE_RISK")
        self.assertEqual(decision.pending_days, 0)

    def test_re_risk_requires_two_distinct_completed_closes_and_same_date_is_idempotent(self):
        dates = pd.DatetimeIndex(
            [pd.Timestamp("2026-08-03"), pd.Timestamp("2026-08-04")]
        )
        indicators = pd.DataFrame(
            {
                "bullish_consensus": [1, 1],
                "annualized_volatility": [0.25, 0.14],
            },
            index=dates,
        )
        state = engine.PortfolioState(
            leader=engine.LEVERAGED_SEMICONDUCTOR,
            regime="BULL",
            volatility_tier="HIGH",
            last_processed_signal_date="2026-08-03",
        )
        first = engine.replay_volatility_state(indicators, state)
        self.assertEqual(first.tier, "HIGH")
        self.assertEqual(first.pending_tier, "LOW")
        self.assertEqual(first.pending_days, 1)
        self.assertEqual(first.processed_sessions, 1)

        state.pending_volatility_tier = first.pending_tier
        state.pending_volatility_days = first.pending_days
        state.last_processed_signal_date = "2026-08-04"
        duplicate = engine.replay_volatility_state(indicators, state)
        self.assertEqual(duplicate.transition, "SAME_DATE")
        self.assertEqual(duplicate.pending_days, 1)
        self.assertEqual(duplicate.processed_sessions, 0)

        indicators.loc[pd.Timestamp("2026-08-05")] = [1, 0.14]
        second = engine.replay_volatility_state(indicators, state)
        self.assertEqual(second.tier, "LOW")
        self.assertEqual(second.transition, "RE_RISK")
        self.assertEqual(second.pending_days, 0)
        self.assertEqual(second.processed_sessions, 1)

    def test_bear_close_resets_pending_volatility_state(self):
        dates = pd.DatetimeIndex(
            [pd.Timestamp("2026-08-03"), pd.Timestamp("2026-08-04")]
        )
        indicators = pd.DataFrame(
            {
                "bullish_consensus": [1, 0],
                "annualized_volatility": [0.14, np.nan],
            },
            index=dates,
        )
        state = engine.PortfolioState(
            leader=engine.LEVERAGED_SEMICONDUCTOR,
            regime="BULL",
            volatility_tier="HIGH",
            pending_volatility_tier="LOW",
            pending_volatility_days=1,
            last_processed_signal_date="2026-08-03",
        )
        decision = engine.replay_volatility_state(indicators, state)
        self.assertEqual((decision.tier, decision.transition), ("N/A", "BEAR"))
        self.assertEqual(decision.pending_tier, "")
        self.assertEqual(decision.pending_days, 0)

    def test_leader_review_cadence_and_momentum_hysteresis(self):
        dates = pd.bdate_range("2026-07-01", periods=22)
        first_date = dates[0].date().isoformat()
        self.assertFalse(engine.sector_review_due(first_date, dates[20], dates))
        self.assertTrue(engine.sector_review_due(first_date, dates[21], dates))

        self.assertEqual(
            engine.select_leader(0.00, 0.05, engine.LEVERAGED_SEMICONDUCTOR),
            engine.LEVERAGED_SEMICONDUCTOR,
        )
        self.assertEqual(
            engine.select_leader(0.00, 0.050001, engine.LEVERAGED_SEMICONDUCTOR),
            engine.LEVERAGED_TECH,
        )
        self.assertEqual(
            engine.select_leader(0.050001, 0.00, engine.LEVERAGED_TECH),
            engine.LEVERAGED_SEMICONDUCTOR,
        )

    def test_leader_changes_only_when_review_is_allowed(self):
        latest = complete_latest(soxl_momentum=0.00, tecl_momentum=0.06)
        tier = engine.VolatilityTierDecision("LOW", "LOW")
        held = engine.determine_target_allocation(
            latest,
            engine.LEVERAGED_SEMICONDUCTOR,
            allow_leader_review=False,
            tier_decision=tier,
        )
        reviewed = engine.determine_target_allocation(
            latest,
            engine.LEVERAGED_SEMICONDUCTOR,
            allow_leader_review=True,
            tier_decision=tier,
        )
        self.assertEqual(held.leader, engine.LEVERAGED_SEMICONDUCTOR)
        self.assertEqual(reviewed.leader, engine.LEVERAGED_TECH)


class RebalanceTests(EngineTestCase):
    def test_exact_five_point_drift_triggers_and_below_band_holds(self):
        target = {"A": 0.50, "B": 0.50}
        self.assertFalse(
            engine.should_rebalance({"A": 0.549999, "B": 0.450001}, target)
        )
        self.assertTrue(engine.should_rebalance({"A": 0.55, "B": 0.45}, target))

    def test_drift_projection_stops_at_inner_destination_and_preserves_sum(self):
        projected = engine.inner_band_rebalance_weights(
            {"A": 0.56, "B": 0.44},
            {"A": 0.50, "B": 0.50},
        )
        self.assertAlmostEqual(projected["A"], 0.525)
        self.assertAlmostEqual(projected["B"], 0.475)
        self.assertAlmostEqual(sum(projected.values()), 1.0)

        capped = engine.inner_band_rebalance_weights(
            {
                engine.LEVERAGED_SEMICONDUCTOR: 0.70,
                engine.SEMICONDUCTOR_ETF: 0.30,
            },
            {
                engine.LEVERAGED_SEMICONDUCTOR: 0.45,
                engine.SEMICONDUCTOR_ETF: 0.55,
            },
        )
        self.assertLessEqual(
            capped[engine.LEVERAGED_SEMICONDUCTOR],
            engine.MAX_LEVERAGED_POSITION,
        )
        self.assertAlmostEqual(sum(capped.values()), 1.0)

    def test_regime_tier_and_leader_changes_are_full_exact_transitions(self):
        low_soxl = make_result(tier="LOW", leader=engine.LEVERAGED_SEMICONDUCTOR)
        existing = dict(low_soxl.target_weights)
        state = engine.PortfolioState(
            executed_leader=engine.LEVERAGED_SEMICONDUCTOR,
            executed_regime="BULL",
            executed_volatility_tier="LOW",
        )
        hold = engine.build_rebalance_plan(existing, low_soxl, state)
        self.assertFalse(hold.rebalance_due)
        self.assertFalse(hold.full_transition)
        self.assertEqual(hold.reason, "HOLD")

        transitions = [
            (
                make_result(regime="BEAR"),
                "REGIME_TRANSITION",
            ),
            (
                make_result(tier="MODERATE"),
                "VOLATILITY_TIER_TRANSITION",
            ),
            (
                make_result(tier="LOW", leader=engine.LEVERAGED_TECH),
                "LEADER_TRANSITION",
            ),
        ]
        for result, reason in transitions:
            with self.subTest(reason=reason):
                plan = engine.build_rebalance_plan(existing, result, state)
                self.assertTrue(plan.rebalance_due)
                self.assertTrue(plan.full_transition)
                self.assertEqual(plan.reason, reason)
                self.assertEqual(
                    plan.execution_weights,
                    weights_with_cash(result.target_weights),
                )

    def test_drift_only_plan_uses_inner_destination(self):
        result = make_result(
            weights={
                engine.SEMICONDUCTOR_ETF: 0.50,
                engine.HEDGE_ASSET: 0.50,
            }
        )
        state = engine.PortfolioState(
            executed_leader=engine.LEVERAGED_SEMICONDUCTOR,
            executed_regime="BULL",
            executed_volatility_tier="LOW",
        )
        plan = engine.build_rebalance_plan(
            {
                engine.SEMICONDUCTOR_ETF: 0.56,
                engine.HEDGE_ASSET: 0.44,
            },
            result,
            state,
        )
        self.assertTrue(plan.rebalance_due)
        self.assertFalse(plan.full_transition)
        self.assertEqual(plan.reason, "DRIFT_BAND")
        self.assertAlmostEqual(plan.execution_weights[engine.SEMICONDUCTOR_ETF], 0.525)
        self.assertAlmostEqual(plan.execution_weights[engine.HEDGE_ASSET], 0.475)

    def test_execution_table_explicitly_sells_zero_target_holdings(self):
        prices = one_row_prices(100.0)
        state = engine.PortfolioState(
            shares={
                engine.LEVERAGED_SEMICONDUCTOR: 1.0,
                engine.LEVERAGED_TECH: 1.0,
            }
        )
        table = engine.calculate_execution_table(
            prices,
            weights_with_cash(engine.BEAR_ALLOCATION),
            200.0,
            state,
            actionable=True,
        ).set_index("Ticker")
        self.assertEqual(table.loc[engine.LEVERAGED_SEMICONDUCTOR, "Action"], "SELL")
        self.assertEqual(table.loc[engine.LEVERAGED_TECH, "Action"], "SELL")
        self.assertEqual(table.loc[engine.LEVERAGED_SEMICONDUCTOR, "TargetPct"], 0.0)
        self.assertEqual(table.loc[engine.LEVERAGED_TECH, "TargetPct"], 0.0)

    def test_zero_order_structural_transition_reconciles_execution_metadata(self):
        result = make_result(tier="LOW", leader=engine.LEVERAGED_SEMICONDUCTOR)
        existing = dict(result.target_weights)
        state = engine.PortfolioState(
            executed_leader=engine.LEVERAGED_SEMICONDUCTOR,
            executed_regime="BULL",
            executed_volatility_tier="MODERATE",
        )
        plan = engine.build_rebalance_plan(existing, result, state)
        self.assertFalse(plan.rebalance_due)
        self.assertEqual(plan.individual_orders, 0)
        self.assertEqual(plan.reason, "CONFIRMED_TARGET_STATE")

        run = make_strategy_run(
            state=state,
            result=result,
            execution_weights=plan.execution_weights,
            rebalance_due=plan.rebalance_due,
            individual_orders=plan.individual_orders,
            reason=plan.reason,
        )
        with mock.patch.object(engine, "save_state") as save:
            engine.persist_signal_run(run)
        self.assertEqual(state.executed_regime, "BULL")
        self.assertEqual(state.executed_volatility_tier, "LOW")
        self.assertEqual(state.executed_leader, engine.LEVERAGED_SEMICONDUCTOR)
        self.assertEqual(state.target_weights, plan.execution_weights)
        save.assert_called_once_with(state)


class MarketDataTests(EngineTestCase):
    def test_partial_latest_prices_and_required_volume_are_rejected(self):
        _, _, payload = market_frames()
        expected = payload.index[-1]

        missing_price = payload.copy()
        missing_price.loc[expected, ("Close", engine.HEDGE_ASSET)] = np.nan
        with self.assertRaisesRegex(RuntimeError, "latest prices.*missing or invalid"):
            self.download_from_payload(missing_price, expected)

        missing_volume = payload.copy()
        missing_volume.loc[expected, ("Volume", engine.MARKET_INDEX)] = np.nan
        with self.assertRaisesRegex(RuntimeError, "QQQ volumes.*missing or invalid"):
            self.download_from_payload(missing_volume, expected)

    def test_missing_ticker_column_is_rejected(self):
        _, _, payload = market_frames()
        payload = payload.drop(columns=[("Close", engine.HEDGE_ASSET)])
        with self.assertRaisesRegex(RuntimeError, "Incomplete market-data universe.*GLD"):
            self.download_from_payload(payload, payload.index[-1])

    def test_unrelated_historical_gap_is_neither_filled_nor_dropped(self):
        _, _, payload = market_frames()
        gap_date = payload.index[-20]
        payload.loc[gap_date, ("Close", engine.SEMICONDUCTOR_ETF)] = np.nan
        (prices, volumes), downloader = self.download_from_payload(
            payload,
            payload.index[-1],
        )
        self.assertEqual(len(prices), len(payload))
        self.assertTrue(np.isnan(prices.loc[gap_date, engine.SEMICONDUCTOR_ETF]))
        self.assertTrue(prices.index.equals(volumes.index))
        downloader.assert_called_once()

    def test_gap_inside_required_signal_window_fails_closed(self):
        _, _, payload = market_frames()
        payload.loc[payload.index[-10], ("Close", engine.MARKET_INDEX)] = np.nan
        with self.assertRaisesRegex(RuntimeError, "QQQ prices.*missing or invalid"):
            self.download_from_payload(payload, payload.index[-1])

    def test_stale_latest_session_is_rejected(self):
        _, _, payload = market_frames()
        expected = payload.index[-1] + pd.offsets.BDay(1)
        with self.assertRaisesRegex(RuntimeError, "stale or misdated"):
            self.download_from_payload(payload, expected)

    def test_latest_completed_nyse_session_freshness(self):
        before_open = datetime(2026, 8, 3, 8, 0, tzinfo=engine.NEW_YORK)
        during_session = datetime(2026, 8, 3, 10, 0, tzinfo=engine.NEW_YORK)
        after_buffer = datetime(2026, 8, 3, 16, 16, tzinfo=engine.NEW_YORK)
        weekend = datetime(2026, 8, 2, 12, 0, tzinfo=engine.NEW_YORK)

        self.assertEqual(
            engine.expected_completed_session(before_open),
            pd.Timestamp("2026-07-31"),
        )
        with self.assertRaisesRegex(RuntimeError, "daily bar is not final"):
            engine.expected_completed_session(during_session)
        self.assertEqual(
            engine.expected_completed_session(after_buffer),
            pd.Timestamp("2026-08-03"),
        )
        self.assertEqual(
            engine.expected_completed_session(weekend),
            pd.Timestamp("2026-07-31"),
        )

    def test_zero_nan_volatility_and_invalid_prices_fail_closed(self):
        for value in (0.0, -0.01, np.nan):
            with self.subTest(volatility=value):
                with self.assertRaises(ValueError):
                    engine.classify_volatility(value)

        invalid_latest = complete_latest()
        invalid_latest["volatility_10"] = 0.0
        with self.assertRaisesRegex(RuntimeError, "volatility_10"):
            engine.validate_latest_indicators(invalid_latest)

        prices = one_row_prices()
        state = engine.PortfolioState(
            shares={engine.LEVERAGED_SEMICONDUCTOR: 1.0}
        )
        for price in (0.0, np.nan):
            with self.subTest(price=price):
                prices.loc[SIGNAL_DATE, engine.LEVERAGED_SEMICONDUCTOR] = price
                with self.assertRaisesRegex(RuntimeError, "Invalid latest price"):
                    engine.existing_portfolio_value(state, prices)


class StateAndHoldingsTests(EngineTestCase):
    def test_invalid_state_mapping_fails_with_runtime_error(self):
        payload = asdict(engine.PortfolioState())
        payload["shares"] = []
        self.write_state(payload)
        with self.assertRaisesRegex(RuntimeError, "shares must be a mapping"):
            engine.load_state()

    def test_state_save_is_atomic_when_replace_fails(self):
        original = '{"preserve": true}'
        self.state_file.write_text(original, encoding="utf-8")
        with mock.patch.object(engine.os, "replace", side_effect=OSError("disk error")):
            with self.assertRaisesRegex(OSError, "disk error"):
                engine.save_state(engine.PortfolioState())
        self.assertEqual(self.state_file.read_text(encoding="utf-8"), original)

    def test_version_two_migration_preserves_and_prices_confirmed_shares(self):
        legacy = {
            "state_version": 2,
            "shares": {engine.LEVERAGED_SEMICONDUCTOR: 2.0},
            "cash_balance": 25.0,
            "target_weights": {engine.LEVERAGED_SEMICONDUCTOR: 1.0},
            "portfolio_value": 9_999.0,
            "leader": engine.LEVERAGED_SEMICONDUCTOR,
            "volatility_tier": "LOW",
            "regime": "BULL",
            "last_sector_rebalance": "",
            "last_updated": "",
        }
        self.write_state(legacy)
        migrated = engine.load_state()
        self.assertEqual(migrated.state_version, engine.STATE_VERSION)
        self.assertEqual(migrated.shares, legacy["shares"])
        self.assertEqual(migrated.cash_balance, 25.0)
        self.assertEqual(migrated.executed_regime, "UNKNOWN")
        backups = list(self.temp_path.glob("state.v2.*.backup.json"))
        self.assertEqual(len(backups), 1)

        prices = one_row_prices()
        prices.loc[SIGNAL_DATE, engine.LEVERAGED_SEMICONDUCTOR] = 125.0
        self.assertEqual(engine.existing_portfolio_value(migrated, prices), 275.0)

    def test_migration_backup_can_be_disabled_for_test_mode(self):
        legacy = {
            "state_version": 2,
            "shares": {},
            "cash_balance": 100.0,
            "target_weights": {},
            "portfolio_value": 100.0,
            "leader": None,
            "volatility_tier": "N/A",
            "regime": "UNKNOWN",
            "last_sector_rebalance": "",
            "last_updated": "",
        }
        self.write_state(legacy)
        original = self.state_file.read_bytes()
        migrated = engine.load_state(backup_legacy=False)
        self.assertEqual(migrated.state_version, engine.STATE_VERSION)
        self.assertEqual(self.state_file.read_bytes(), original)
        self.assertEqual(list(self.temp_path.glob("*.backup.json")), [])

    def test_version_four_pending_action_migrates_to_one_time_retry(self):
        weights = weights_with_cash(make_result().target_weights)
        legacy = asdict(
            pending_state(
                weights,
                notified=True,
            )
        )
        legacy["state_version"] = 4
        legacy.pop("pending_recommendation_notified")
        legacy.pop("pending_recommendation_supersedes_date")
        self.write_state(legacy)

        migrated = engine.load_state(backup_legacy=False)
        self.assertFalse(migrated.pending_recommendation_notified)
        retry = make_strategy_run(
            state=migrated,
            execution_weights=weights,
            rebalance_due=True,
        )
        decision = engine.decide_notification(retry)
        self.assertEqual(decision.kind, "RETRY")
        self.assertEqual(
            decision.previous_recommendation_date,
            SIGNAL_DATE.date().isoformat(),
        )

    def test_confirmed_shares_and_cash_not_model_target_drive_next_decision(self):
        model_target = weights_with_cash(
            engine.apply_allocation_template(
                engine.LOW_VOL_ALLOCATION,
                engine.LEVERAGED_SEMICONDUCTOR,
            )
        )
        state = pending_state(
            model_target,
            shares={engine.HEDGE_ASSET: 10.0},
        )
        state.cash_balance = 5.0
        state.portfolio_value = 99_999.0
        self.write_state(state)

        confirmed = engine.confirm_execution(
            {engine.LEVERAGED_TECH: 2.0},
            50.0,
            SIGNAL_DATE.date().isoformat(),
        )
        self.assertEqual(
            confirmed.shares,
            {engine.LEVERAGED_TECH: 2.0},
        )
        self.assertEqual(confirmed.cash_balance, 50.0)
        self.assertEqual(confirmed.target_weights, model_target)
        self.assertEqual(confirmed.pending_recommendation_date, "")
        self.assertFalse(confirmed.pending_recommendation_notified)

        prices = one_row_prices()
        prices.loc[SIGNAL_DATE, engine.LEVERAGED_TECH] = 125.0
        self.assertEqual(engine.existing_portfolio_value(confirmed, prices), 300.0)
        actual = engine.existing_weights(confirmed, prices)
        self.assertAlmostEqual(actual[engine.LEVERAGED_TECH], 250.0 / 300.0)
        self.assertAlmostEqual(actual[engine.CASH_ASSET], 50.0 / 300.0)
        self.assertNotEqual(actual, confirmed.target_weights)

        plan = engine.build_rebalance_plan(
            actual,
            make_result(tier="LOW", leader=engine.LEVERAGED_SEMICONDUCTOR),
            confirmed,
        )
        self.assertTrue(plan.rebalance_due)

    def test_missing_or_invalid_price_for_confirmed_holding_is_rejected(self):
        prices = one_row_prices()
        state = engine.PortfolioState(shares={"UNKNOWN": 1.0})
        with self.assertRaisesRegex(RuntimeError, "no current price data"):
            engine.existing_portfolio_value(state, prices)

        state = engine.PortfolioState(
            shares={engine.LEVERAGED_SEMICONDUCTOR: 1.0}
        )
        prices.loc[SIGNAL_DATE, engine.LEVERAGED_SEMICONDUCTOR] = np.nan
        with self.assertRaisesRegex(RuntimeError, "Invalid latest price"):
            engine.existing_portfolio_value(state, prices)

    def test_fills_from_superseded_signal_are_reconciled_conservatively(self):
        current_weights = weights_with_cash(
            make_result(tier="MODERATE").target_weights
        )
        state = pending_state(
            current_weights,
            tier="MODERATE",
            notified=True,
        )
        self.write_state(state)

        confirmed = engine.confirm_execution(
            {engine.LEVERAGED_TECH: 3.0},
            25.0,
            "2026-08-03",
        )
        self.assertEqual(
            confirmed.shares,
            {engine.LEVERAGED_TECH: 3.0},
        )
        self.assertEqual(confirmed.cash_balance, 25.0)
        self.assertEqual(confirmed.pending_recommendation_date, "")
        self.assertFalse(confirmed.pending_recommendation_notified)
        self.assertEqual(confirmed.target_weights, {})
        self.assertEqual(confirmed.executed_regime, "UNKNOWN")
        self.assertEqual(confirmed.executed_volatility_tier, "N/A")


class NotificationTests(EngineTestCase):
    def test_hold_persists_signal_state_but_sends_no_email(self):
        run = make_strategy_run(rebalance_due=False)
        self.assertEqual(engine.decide_notification(run).kind, "NONE")
        with mock.patch.dict(engine.os.environ, {}, clear=True), mock.patch.object(
            sys,
            "argv",
            ["port12_cloud.py"],
        ), mock.patch.object(engine, "configure_logging"), mock.patch.object(
            engine,
            "run_strategy",
            return_value=run,
        ), mock.patch.object(engine, "log_decision"), mock.patch.object(
            engine,
            "build_dashboard",
            return_value="dashboard",
        ), mock.patch.object(engine, "persist_signal_run") as persist, mock.patch.object(
            engine,
            "send_email",
        ) as send, mock.patch.object(
            engine,
            "persist_notification_delivery",
        ) as delivery, mock.patch("builtins.print"):
            engine.main()
        persist.assert_called_once_with(run)
        send.assert_not_called()
        delivery.assert_not_called()

    def test_new_action_sends_once_and_identical_pending_action_is_suppressed(self):
        weights = weights_with_cash(make_result().target_weights)
        new_run = make_strategy_run(
            execution_weights=weights,
            rebalance_due=True,
        )
        self.assertEqual(engine.decide_notification(new_run).kind, "ACTION")

        state = pending_state(weights)
        duplicate = make_strategy_run(
            state=state,
            execution_weights=weights,
            rebalance_due=True,
        )
        decision = engine.decide_notification(duplicate)
        self.assertEqual(decision.kind, "NONE")
        self.assertEqual(decision.reason, "IDENTICAL_PENDING_RECOMMENDATION")

    def test_sub_half_point_destination_change_is_suppressed(self):
        saved = weights_with_cash(make_result().target_weights)
        state = pending_state(saved)
        tiny_change = dict(saved)
        tiny_change[engine.LEVERAGED_SEMICONDUCTOR] += 0.004
        tiny_change[engine.LEVERAGED_TECH] -= 0.004
        run = make_strategy_run(
            state=state,
            execution_weights=tiny_change,
            rebalance_due=True,
        )
        self.assertEqual(engine.decide_notification(run).kind, "NONE")

    def test_material_pending_action_update_replaces_prior_instruction(self):
        low = weights_with_cash(make_result(tier="LOW").target_weights)
        state = pending_state(low)
        moderate_result = make_result(tier="MODERATE")
        moderate = weights_with_cash(moderate_result.target_weights)
        run = make_strategy_run(
            state=state,
            result=moderate_result,
            execution_weights=moderate,
            rebalance_due=True,
            reason="VOLATILITY_TIER_TRANSITION",
        )
        decision = engine.decide_notification(run)
        self.assertEqual(decision.kind, "UPDATE")
        engine.prepare_notification_delivery(run, decision)
        with mock.patch.object(engine, "save_state") as save:
            engine.persist_notification_delivery(run, decision)
        self.assertEqual(state.pending_recommendation_weights, moderate)
        self.assertEqual(state.pending_recommendation_tier, "MODERATE")
        self.assertTrue(state.pending_recommendation_notified)
        save.assert_called_once_with(state)

    def test_failed_update_retry_retains_the_last_delivered_action_identity(self):
        low = weights_with_cash(make_result(tier="LOW").target_weights)
        state = pending_state(low, notified=True)
        state.pending_recommendation_date = "2026-08-03"

        moderate_result = make_result(tier="MODERATE")
        moderate = weights_with_cash(moderate_result.target_weights)
        update_run = make_strategy_run(
            state=state,
            result=moderate_result,
            execution_weights=moderate,
            rebalance_due=True,
            reason="VOLATILITY_TIER_TRANSITION",
        )
        update = engine.decide_notification(update_run)
        self.assertEqual(update.kind, "UPDATE")
        self.assertEqual(update.previous_recommendation_date, "2026-08-03")

        engine.prepare_notification_delivery(update_run, update)
        self.assertEqual(
            state.pending_recommendation_date,
            SIGNAL_DATE.date().isoformat(),
        )
        self.assertEqual(
            state.pending_recommendation_supersedes_date,
            "2026-08-03",
        )
        self.assertFalse(state.pending_recommendation_notified)

        retry_run = make_strategy_run(
            state=state,
            result=moderate_result,
            execution_weights=moderate,
            rebalance_due=True,
            reason="VOLATILITY_TIER_TRANSITION",
            signal_date="2026-08-05",
        )
        retry = engine.decide_notification(retry_run)
        self.assertEqual(retry.kind, "UPDATE_RETRY")
        self.assertEqual(
            retry.previous_recommendation_date,
            SIGNAL_DATE.date().isoformat(),
        )
        self.assertEqual(retry.supersedes_recommendation_date, "2026-08-03")
        self.assertIn("2026-08-04", engine.notification_subject(retry_run, retry))
        retry_text = engine.notification_text_body(
            retry_run,
            retry,
            "dashboard",
        )
        self.assertIn("2026-08-04", retry_text)
        self.assertIn("2026-08-03", retry_text)
        self.assertIn(
            "UPDATED ACTION REQUIRED - DELIVERY RETRY",
            engine.build_email_html(retry_run, retry),
        )

        cancellation_run = make_strategy_run(
            state=copy.deepcopy(state),
            result=moderate_result,
            execution_weights=moderate,
            rebalance_due=False,
            signal_date="2026-08-05",
        )
        cancellation = engine.decide_notification(cancellation_run)
        self.assertEqual(cancellation.kind, "CANCELLATION")
        self.assertEqual(cancellation.previous_recommendation_date, "2026-08-03")

        with mock.patch.object(engine, "save_state") as save:
            engine.persist_notification_delivery(retry_run, retry)
        self.assertTrue(state.pending_recommendation_notified)
        self.assertEqual(state.pending_recommendation_supersedes_date, "")
        save.assert_called_once_with(state)

    def test_rebalance_flag_without_orders_is_not_an_action(self):
        no_orders = make_strategy_run(
            rebalance_due=True,
            individual_orders=0,
        )
        self.assertEqual(engine.decide_notification(no_orders).kind, "NONE")

        weights = weights_with_cash(make_result().target_weights)
        pending = make_strategy_run(
            state=pending_state(weights),
            execution_weights=weights,
            rebalance_due=True,
            individual_orders=0,
        )
        self.assertEqual(engine.decide_notification(pending).kind, "CANCELLATION")

    def test_cancellation_is_sent_once_then_pending_action_is_cleared(self):
        weights = weights_with_cash(make_result().target_weights)
        state = pending_state(weights)
        run = make_strategy_run(
            state=state,
            execution_weights=weights,
            rebalance_due=False,
        )
        cancellation = engine.decide_notification(run)
        self.assertEqual(cancellation.kind, "CANCELLATION")
        with mock.patch.object(engine, "save_state") as save:
            engine.persist_notification_delivery(run, cancellation)
        self.assertEqual(state.pending_recommendation_date, "")
        self.assertEqual(state.pending_recommendation_weights, {})
        self.assertEqual(engine.decide_notification(run).kind, "NONE")
        save.assert_called_once_with(state)

    def test_smtp_failure_stages_confirmable_action_and_preserves_fills_for_retry(self):
        state = engine.PortfolioState(
            shares={engine.LEVERAGED_TECH: 2.0},
        )
        state.cash_balance = 50.0
        state.executed_leader = engine.LEVERAGED_SEMICONDUCTOR
        state.executed_regime = "BULL"
        state.executed_volatility_tier = "LOW"
        new_result = make_result(tier="MODERATE")
        new_weights = weights_with_cash(new_result.target_weights)
        run = make_strategy_run(
            state=state,
            result=new_result,
            execution_weights=new_weights,
            rebalance_due=True,
            reason="VOLATILITY_TIER_TRANSITION",
        )

        with mock.patch.object(
            sys,
            "argv",
            ["port12_cloud.py"],
        ), mock.patch.object(engine, "configure_logging"), mock.patch.object(
            engine,
            "run_strategy",
            return_value=run,
        ), mock.patch.object(engine, "log_decision"), mock.patch.object(
            engine,
            "build_dashboard",
            return_value="dashboard",
        ), mock.patch.object(
            engine,
            "build_email_html",
            return_value="<html></html>",
        ), mock.patch.object(
            engine,
            "send_email",
            side_effect=RuntimeError("SMTP down"),
        ), mock.patch.object(
            engine,
            "persist_notification_delivery",
        ) as delivered, mock.patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "SMTP down"):
                engine.main()
        delivered.assert_not_called()

        persisted = engine.load_state()
        self.assertEqual(
            persisted.shares,
            {engine.LEVERAGED_TECH: 2.0},
        )
        self.assertEqual(persisted.cash_balance, 50.0)
        self.assertEqual(persisted.pending_recommendation_weights, new_weights)
        self.assertEqual(
            persisted.pending_recommendation_date,
            SIGNAL_DATE.date().isoformat(),
        )
        self.assertFalse(persisted.pending_recommendation_notified)
        engine.validate_execution_confirmation(
            persisted,
            SIGNAL_DATE.date().isoformat(),
        )
        retry = make_strategy_run(
            state=persisted,
            result=new_result,
            execution_weights=new_weights,
            rebalance_due=True,
            reason="VOLATILITY_TIER_TRANSITION",
        )
        self.assertEqual(engine.decide_notification(retry).kind, "RETRY")

    def test_update_and_cancellation_bodies_name_superseded_signal(self):
        prior_date = "2026-08-01"
        update = engine.NotificationDecision(
            "UPDATE",
            "MATERIAL_RECOMMENDATION_UPDATE",
            prior_date,
        )
        cancellation = engine.NotificationDecision(
            "CANCELLATION",
            "PENDING_ACTION_NO_LONGER_REQUIRED",
            prior_date,
        )
        run = make_strategy_run(rebalance_due=True)
        self.assertIn(
            prior_date,
            engine.notification_text_body(run, update, "dashboard"),
        )
        self.assertIn(
            prior_date,
            engine.notification_text_body(run, cancellation, "dashboard"),
        )

    def test_missing_credentials_fail_only_when_notification_is_due(self):
        with mock.patch.dict(engine.os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "required.*notification"):
                engine.send_email("subject", "text", "<html></html>")

        hold = make_strategy_run(rebalance_due=False)
        self.assertFalse(engine.decide_notification(hold).should_send)

    def test_test_mode_never_saves_sends_logs_or_backs_up_state(self):
        run = make_strategy_run(rebalance_due=True)
        with mock.patch.object(
            sys,
            "argv",
            ["port12_cloud.py", "--test", "--roth-amount", "100"],
        ), mock.patch.object(engine, "configure_logging") as configure, mock.patch.object(
            engine,
            "run_strategy",
            return_value=run,
        ) as run_strategy, mock.patch.object(engine, "log_decision"), mock.patch.object(
            engine,
            "build_dashboard",
            return_value="dashboard",
        ), mock.patch.object(
            engine,
            "persist_signal_run",
        ) as persist, mock.patch.object(
            engine,
            "prepare_notification_delivery",
        ) as prepare, mock.patch.object(
            engine,
            "persist_notification_delivery",
        ) as delivery, mock.patch.object(
            engine,
            "send_email",
        ) as send, mock.patch.object(
            engine,
            "_backup_legacy_state",
        ) as backup, mock.patch("builtins.print"):
            engine.main()
        configure.assert_called_once_with(persist_log=False)
        run_strategy.assert_called_once_with(
            100.0,
            backup_legacy_state=False,
        )
        persist.assert_not_called()
        prepare.assert_not_called()
        delivery.assert_not_called()
        send.assert_not_called()
        backup.assert_not_called()
        self.assertFalse(self.log_file.exists())


if __name__ == "__main__":
    unittest.main()
