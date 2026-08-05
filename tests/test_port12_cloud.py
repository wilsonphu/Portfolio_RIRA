import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import port12_cloud as engine  # noqa: E402


class FixedAfterCloseDateTime(datetime):
    @classmethod
    def now(cls, timezone=None):
        return cls(2026, 8, 3, 17, 0, tzinfo=timezone)


def market_data(rows=230):
    index = pd.bdate_range("2025-01-02", periods=rows)
    values = {
        ticker: np.arange(rows, dtype=float) + 100.0 + position
        for position, ticker in enumerate(engine.ALL_TICKERS)
    }
    prices = pd.DataFrame(values, index=index)
    volumes = pd.DataFrame(
        {ticker: np.full(rows, 1_000_000.0) for ticker in engine.ALL_TICKERS},
        index=index,
    )
    return prices, volumes, pd.concat({"Close": prices, "Volume": volumes}, axis=1)


class PortfolioEngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.state_file = Path(self.temporary_directory.name) / "state.json"
        self.state_file_patch = mock.patch.object(engine, "STATE_FILE", self.state_file)
        self.state_file_patch.start()

    def tearDown(self):
        self.state_file_patch.stop()
        self.temporary_directory.cleanup()

    def write_state(self, payload):
        self.state_file.write_text(json.dumps(payload), encoding="utf-8")

    def make_strategy_run(self, prices, state, result, *, rebalance_due=True,
                          sector_due=False, execution_weights=None, reason="DRIFT_BAND"):
        latest = pd.Series({
            "sma_signal": 1,
            "donchian_signal": 1,
            "vwma_signal": 1,
            "volatility_10": 0.10,
            "volatility_30": 0.09,
            "soxl_momentum": 0.01,
            "tecl_momentum": 0.02,
        })
        execution_weights = execution_weights or result.target_weights
        return engine.StrategyRun(
            price_data=prices,
            latest_indicators=latest,
            result=result,
            state=state,
            portfolio_value=100.0,
            target_portfolio=engine.calculate_target_portfolio(prices, execution_weights, 100.0),
            signal_date=prices.index[-1],
            rebalance_due=rebalance_due,
            sector_review_due=sector_due,
            execution_weights=execution_weights,
            full_transition=False,
            rebalance_reason=reason,
            one_way_turnover=0.05 if rebalance_due else 0.0,
            individual_orders=2 if rebalance_due else 0,
            tier_decision=engine.VolatilityTierDecision(
                result.volatility_tier,
                result.raw_volatility_tier,
            ),
        )

    def test_download_rejects_partial_latest_row_without_substitution(self):
        _, _, data = market_data()
        data.loc[data.index[-1], ("Close", engine.HEDGE_ASSET)] = np.nan
        with mock.patch.object(engine.yf, "download", return_value=data), \
             mock.patch.object(engine, "datetime", FixedAfterCloseDateTime):
            with self.assertRaisesRegex(RuntimeError, "Latest market-data row is partial.*GLD"):
                engine.download_market_data(engine.ALL_TICKERS)

    def test_download_drops_incomplete_historical_row_and_preserves_alignment(self):
        prices, _, data = market_data()
        partial_date = data.index[20]
        data.loc[partial_date, ("Volume", engine.LEVERAGED_TECH)] = np.nan
        with mock.patch.object(engine.yf, "download", return_value=data), \
             mock.patch.object(engine, "datetime", FixedAfterCloseDateTime):
            downloaded_prices, downloaded_volumes = engine.download_market_data(engine.ALL_TICKERS)
        self.assertNotIn(partial_date, downloaded_prices.index)
        self.assertTrue(downloaded_prices.index.equals(downloaded_volumes.index))
        self.assertEqual(list(downloaded_prices.columns), engine.ALL_TICKERS)
        self.assertEqual(len(downloaded_prices), len(prices) - 1)

    def test_download_rejects_missing_ticker_column(self):
        _, _, data = market_data()
        data = data.drop(columns=("Close", engine.HEDGE_ASSET))
        with mock.patch.object(engine.yf, "download", return_value=data), \
             mock.patch.object(engine, "datetime", FixedAfterCloseDateTime):
            with self.assertRaisesRegex(RuntimeError, "Incomplete market-data universe.*GLD"):
                engine.download_market_data(engine.ALL_TICKERS)

    def test_fast_slow_volatility_uses_the_conservative_maximum(self):
        prices, volumes, _ = market_data(rows=80)
        returns = 0.01 * np.sin(np.arange(len(prices)) / 3.0)
        prices[engine.MARKET_INDEX] = 100.0 * np.exp(np.cumsum(returns))
        indicators = engine.calculate_indicators(prices, volumes)
        latest = indicators.iloc[-1]
        self.assertAlmostEqual(
            latest["annualized_volatility"],
            max(latest["volatility_10"], latest["volatility_30"]),
        )

    def test_load_state_fails_closed_for_invalid_or_incompatible_data(self):
        invalid_payloads = [
            {"state_version": engine.STATE_VERSION},
            {"state_version": engine.STATE_VERSION, "shares": [], "target_weights": {}},
            {"state_version": engine.STATE_VERSION, "shares": {"QQQ": 1}, "target_weights": {}},
            {"state_version": engine.STATE_VERSION, "shares": {"SOXL": float("nan")}, "target_weights": {}},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                self.write_state(payload)
                with self.assertRaises(RuntimeError):
                    engine.load_state()

    def test_version_one_state_migrates_without_claiming_execution_history(self):
        version_one_state = {
            "state_version": 1,
            "shares": {engine.LEVERAGED_SEMICONDUCTOR: 1.0},
            "target_weights": {engine.LEVERAGED_SEMICONDUCTOR: 1.0},
            "portfolio_value": 100.0,
            "leader": engine.LEVERAGED_SEMICONDUCTOR,
            "volatility_tier": "LOW",
            "regime": "BULL",
            "last_sector_rebalance": "",
            "last_updated": "",
        }
        self.write_state(version_one_state)
        migrated = engine.load_state()
        self.assertEqual(migrated.state_version, engine.STATE_VERSION)
        self.assertEqual(migrated.executed_regime, "UNKNOWN")
        self.assertEqual(migrated.shares, version_one_state["shares"])

    def test_save_state_is_atomic_and_does_not_suppress_failure(self):
        original_contents = "{\"preserve\": true}"
        self.state_file.write_text(original_contents, encoding="utf-8")
        with mock.patch.object(engine.os, "replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                engine.save_state(engine.PortfolioState())
        self.assertEqual(self.state_file.read_text(encoding="utf-8"), original_contents)

    def test_leader_changes_only_on_sector_review(self):
        latest = pd.Series({
            "bullish_consensus": 1,
            "annualized_volatility": 0.10,
            "soxl_momentum": 0.00,
            "tecl_momentum": 0.06,
        })
        held_leader = engine.determine_target_allocation(
            latest, engine.LEVERAGED_SEMICONDUCTOR, allow_leader_review=False
        )
        reviewed_leader = engine.determine_target_allocation(
            latest, engine.LEVERAGED_SEMICONDUCTOR, allow_leader_review=True
        )
        self.assertEqual(held_leader.leader, engine.LEVERAGED_SEMICONDUCTOR)
        self.assertEqual(reviewed_leader.leader, engine.LEVERAGED_TECH)

    def test_volatility_tier_persistence_de_risks_immediately_and_delays_re_risking(self):
        de_risk = engine.classify_volatility_with_persistence(0.23, "LOW", "", 0)
        self.assertEqual((de_risk.tier, de_risk.transition), ("HIGH", "DE_RISK"))

        first_re_risk_close = engine.classify_volatility_with_persistence(0.14, "HIGH", "", 0)
        self.assertEqual(first_re_risk_close.tier, "HIGH")
        self.assertEqual(first_re_risk_close.pending_tier, "LOW")
        self.assertEqual(first_re_risk_close.pending_days, 1)

        second_re_risk_close = engine.classify_volatility_with_persistence(
            0.14,
            "HIGH",
            first_re_risk_close.pending_tier,
            first_re_risk_close.pending_days,
        )
        self.assertEqual((second_re_risk_close.tier, second_re_risk_close.transition), ("LOW", "RE_RISK"))

    def test_sector_review_uses_completed_trading_sessions(self):
        dates = pd.bdate_range("2026-01-02", periods=22)
        self.assertFalse(engine.sector_review_due(dates[0].isoformat(), dates[20], dates))
        self.assertTrue(engine.sector_review_due(dates[0].isoformat(), dates[21], dates))

    def test_review_without_execution_preserves_holdings_and_target_weights(self):
        prices, _, _ = market_data(rows=1)
        state = engine.PortfolioState(
            shares={engine.LEVERAGED_SEMICONDUCTOR: 2.0},
            target_weights={engine.LEVERAGED_SEMICONDUCTOR: 1.0},
        )
        result = engine.StrategyResult(
            target_weights={engine.LEVERAGED_TECH: 1.0},
            regime="BULL",
            leader=engine.LEVERAGED_TECH,
            volatility_tier="LOW",
            annualized_volatility=0.10,
        )
        strategy_run = self.make_strategy_run(
            prices, state, result, rebalance_due=False, sector_due=True, reason="HOLD"
        )
        engine.persist_state(strategy_run, None, None)
        self.assertEqual(state.shares, {engine.LEVERAGED_SEMICONDUCTOR: 2.0})
        self.assertEqual(state.target_weights, {engine.LEVERAGED_SEMICONDUCTOR: 1.0})
        self.assertEqual(state.leader, engine.LEVERAGED_TECH)

    def test_mark_to_market_uses_confirmed_holdings_not_stale_state_value(self):
        prices, _, _ = market_data(rows=1)
        prices.loc[prices.index[-1], engine.LEVERAGED_SEMICONDUCTOR] = 125.0
        state = engine.PortfolioState(
            shares={engine.LEVERAGED_SEMICONDUCTOR: 2.0},
            portfolio_value=1_000.0,
        )
        with mock.patch.object(engine, "ROTH_IRA_AMOUNT", None):
            value = engine.resolve_portfolio_value(None, state, prices)
        self.assertEqual(value, 250.0)

    def test_holdings_without_current_prices_are_rejected(self):
        prices, _, _ = market_data(rows=1)
        state = engine.PortfolioState(shares={"UNKNOWN": 1.0})
        with self.assertRaisesRegex(RuntimeError, "holdings with no current price data"):
            engine.existing_portfolio_value(state, prices)

    def test_confirmed_fills_store_actual_weights_not_model_targets(self):
        prices, _, _ = market_data(rows=1)
        state = engine.PortfolioState()
        result = engine.StrategyResult(
            target_weights={engine.LEVERAGED_SEMICONDUCTOR: 1.0},
            regime="BULL", leader=engine.LEVERAGED_SEMICONDUCTOR,
            volatility_tier="LOW", annualized_volatility=0.10,
        )
        executed_shares = {engine.LEVERAGED_TECH: 2.0}
        pending_date = prices.index[-1].date().isoformat()
        state.pending_recommendation_date = pending_date
        state.pending_recommendation_regime = "BULL"
        state.pending_recommendation_tier = "LOW"
        strategy_run = self.make_strategy_run(prices, state, result, rebalance_due=False)
        engine.persist_state(strategy_run, executed_shares, pending_date)
        self.assertEqual(state.shares, executed_shares)
        self.assertEqual(state.target_weights, {engine.LEVERAGED_TECH: 1.0})

    def test_confirmed_fills_use_the_pending_signal_regime_not_todays_signal(self):
        prices, _, _ = market_data(rows=1)
        pending_date = prices.index[-1].date().isoformat()
        state = engine.PortfolioState(
            pending_recommendation_date=pending_date,
            pending_recommendation_regime="BEAR",
            pending_recommendation_tier="N/A",
        )
        current_result = engine.StrategyResult(
            target_weights={engine.LEVERAGED_SEMICONDUCTOR: 1.0},
            regime="BULL", leader=engine.LEVERAGED_SEMICONDUCTOR,
            volatility_tier="LOW", annualized_volatility=0.10, raw_volatility_tier="LOW",
        )
        strategy_run = self.make_strategy_run(prices, state, current_result, rebalance_due=False)
        engine.persist_state(
            strategy_run,
            {engine.DEFENSIVE_EQUITY: 2.0},
            pending_date,
        )
        self.assertEqual(state.executed_regime, "BEAR")
        self.assertEqual(state.executed_volatility_tier, "N/A")
        self.assertEqual(state.pending_recommendation_date, "")

    def test_confirmed_fill_must_match_a_pending_recommendation(self):
        prices, _, _ = market_data(rows=1)
        state = engine.PortfolioState()
        result = engine.StrategyResult(
            target_weights={engine.LEVERAGED_SEMICONDUCTOR: 1.0},
            regime="BULL", leader=engine.LEVERAGED_SEMICONDUCTOR,
            volatility_tier="LOW", annualized_volatility=0.10, raw_volatility_tier="LOW",
        )
        strategy_run = self.make_strategy_run(prices, state, result, rebalance_due=False)
        with self.assertRaisesRegex(RuntimeError, "do not match the pending recommendation"):
            engine.persist_state(
                strategy_run,
                {engine.LEVERAGED_SEMICONDUCTOR: 1.0},
                prices.index[-1].date().isoformat(),
            )

    def test_inner_band_destination_is_sum_preserving_and_respects_cap(self):
        two_asset = engine.inner_band_rebalance_weights(
            {"A": 0.52, "B": 0.48}, {"A": 0.45, "B": 0.55}
        )
        self.assertAlmostEqual(two_asset["A"], 0.475)
        self.assertAlmostEqual(two_asset["B"], 0.525)
        self.assertAlmostEqual(sum(two_asset.values()), 1.0)

        multi_asset = engine.inner_band_rebalance_weights(
            {"A": 0.52, "B": 0.41, "C": 0.07},
            {"A": 0.45, "B": 0.45, "C": 0.10},
        )
        self.assertAlmostEqual(sum(multi_asset.values()), 1.0)
        self.assertTrue(all(
            abs(multi_asset[ticker] - target) <= engine.REBALANCE_DESTINATION + 1e-9
            for ticker, target in {"A": 0.45, "B": 0.45, "C": 0.10}.items()
        ))

        capped = engine.inner_band_rebalance_weights(
            {engine.LEVERAGED_SEMICONDUCTOR: 0.70, engine.SEMICONDUCTOR_ETF: 0.30},
            {engine.LEVERAGED_SEMICONDUCTOR: 0.45, engine.SEMICONDUCTOR_ETF: 0.55},
        )
        self.assertLessEqual(capped[engine.LEVERAGED_SEMICONDUCTOR], engine.MAX_LEVERAGED_POSITION)
        self.assertAlmostEqual(sum(capped.values()), 1.0)

    def test_rebalance_plan_uses_full_targets_for_regime_tier_changes_only(self):
        result = engine.StrategyResult(
            target_weights={"A": 0.45, "B": 0.55},
            regime="BULL", leader=engine.LEVERAGED_SEMICONDUCTOR,
            volatility_tier="LOW", annualized_volatility=0.10, raw_volatility_tier="LOW",
        )
        existing = {"A": 0.52, "B": 0.48}
        state = engine.PortfolioState(
            executed_regime="BULL",
            executed_volatility_tier="LOW",
        )
        drift_plan = engine.build_rebalance_plan(existing, result, state)
        self.assertFalse(drift_plan.full_transition)
        self.assertAlmostEqual(drift_plan.execution_weights["A"], 0.475)

        state.executed_volatility_tier = "MODERATE"
        tier_plan = engine.build_rebalance_plan(existing, result, state)
        self.assertTrue(tier_plan.full_transition)
        self.assertEqual(tier_plan.execution_weights, result.target_weights)

    def test_invalid_prices_and_volatility_fail_closed(self):
        prices, _, _ = market_data(rows=1)
        prices.loc[prices.index[-1], engine.LEVERAGED_SEMICONDUCTOR] = 0.0
        with self.assertRaisesRegex(RuntimeError, "Invalid latest price"):
            engine.calculate_target_portfolio(prices, {engine.LEVERAGED_SEMICONDUCTOR: 1.0}, 100.0)
        self.assertEqual(engine.classify_volatility(0.22), "MODERATE")
        with self.assertRaises(ValueError):
            engine.classify_volatility(float("nan"))

    def test_test_mode_never_persists_or_sends(self):
        prices, _, _ = market_data(rows=1)
        result = engine.StrategyResult(
            target_weights={engine.LEVERAGED_SEMICONDUCTOR: 1.0},
            regime="BULL", leader=engine.LEVERAGED_SEMICONDUCTOR,
            volatility_tier="LOW", annualized_volatility=0.10,
        )
        strategy_run = self.make_strategy_run(prices, engine.PortfolioState(), result, sector_due=True)
        with mock.patch.object(engine, "run_strategy", return_value=strategy_run), \
             mock.patch.object(engine, "persist_state") as persist, \
             mock.patch.object(engine, "send_email") as send, \
             mock.patch.object(sys, "argv", ["port12_cloud.py", "--test", "--roth-amount", "100"]):
            engine.main()
        persist.assert_not_called()
        send.assert_not_called()

    def test_email_failure_prevents_state_commit(self):
        prices, _, _ = market_data(rows=1)
        result = engine.StrategyResult(
            target_weights={engine.LEVERAGED_SEMICONDUCTOR: 1.0},
            regime="BULL", leader=engine.LEVERAGED_SEMICONDUCTOR,
            volatility_tier="LOW", annualized_volatility=0.10,
        )
        strategy_run = self.make_strategy_run(prices, engine.PortfolioState(), result, sector_due=True)
        with mock.patch.object(engine, "run_strategy", return_value=strategy_run), \
             mock.patch.object(engine, "persist_state") as persist, \
             mock.patch.object(engine, "send_email", side_effect=RuntimeError("SMTP down")), \
             mock.patch.object(sys, "argv", ["port12_cloud.py", "--roth-amount", "100"]):
            with self.assertRaisesRegex(RuntimeError, "SMTP down"):
                engine.main()
        persist.assert_not_called()


if __name__ == "__main__":
    unittest.main()
