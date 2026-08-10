from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

import alpha_core as core
import alpha_research as research
import core_universe_research as universe


def price_series(rows: int = 1000, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-03-11", periods=rows)
    market = rng.normal(0.0005, 0.01, rows)
    return pd.DataFrame(
        {
            "QQQ": 100.0
            * np.exp(np.cumsum(1.1 * market + rng.normal(0.0, 0.003, rows))),
            "SMH": 80.0
            * np.exp(np.cumsum(1.3 * market + rng.normal(0.0, 0.006, rows))),
        },
        index=dates,
    )


def market_data(periods: int = 10) -> universe.UniverseData:
    sessions = pd.bdate_range("2024-01-02", periods=periods)
    positions = np.arange(periods, dtype=float)
    opens = pd.DataFrame(index=sessions)
    closes = pd.DataFrame(index=sessions)
    volumes = pd.DataFrame(index=sessions)
    for number, ticker in enumerate(universe.REQUIRED_TICKERS):
        base = 30.0 + 10.0 * number
        opens[ticker] = base * np.exp((0.001 + number * 0.00001) * positions)
        closes[ticker] = opens[ticker] * 1.001
        volumes[ticker] = 1_000_000.0
    return universe.UniverseData(
        opens=opens,
        closes=closes,
        volumes=volumes,
        sessions=sessions,
        common_start=sessions[0],
        fingerprint=universe._fingerprint_frames(opens, closes, volumes),
        source="synthetic universe test data",
    )


def constant_schedule(
    sessions: pd.DatetimeIndex,
    weights: dict[str, float],
) -> pd.DataFrame:
    completed = universe._complete_weights(weights)
    return pd.DataFrame([completed] * len(sessions), index=sessions)


class FrozenRegistryTests(unittest.TestCase):
    def test_registry_is_exact_and_unique(self):
        self.assertEqual(len(universe.CANDIDATES), 8)
        self.assertEqual(len(set(universe.CANDIDATE_NAMES)), 8)
        self.assertEqual(universe.BENCHMARK, "live_qld_soxl")
        self.assertEqual(
            set(universe.TRADED_TICKERS),
            set(universe.LEVERAGE),
        )


class UniverseDataTests(unittest.TestCase):
    @staticmethod
    def raw_universe() -> tuple[pd.DataFrame, str, str]:
        sessions = research._calendar_sessions(
            pd.Timestamp("2010-01-04"),
            pd.Timestamp("2010-06-01"),
        )
        columns = pd.MultiIndex.from_product(
            [["Open", "Close", "Volume"], universe.REQUIRED_TICKERS]
        )
        raw = pd.DataFrame(index=sessions, columns=columns, dtype=float)
        positions = np.arange(len(sessions), dtype=float)
        for number, ticker in enumerate(universe.REQUIRED_TICKERS):
            price = (40.0 + number) * np.exp(0.0005 * positions)
            raw[("Open", ticker)] = price * 0.999
            raw[("Close", ticker)] = price
            raw[("Volume", ticker)] = 1_000_000.0
        return raw, "2010-01-04", "2010-06-02"

    def test_zero_reported_volume_is_preserved_but_missing_volume_fails(self):
        raw, start, end = self.raw_universe()
        session = raw.index[20]
        raw.loc[session, ("Volume", "USD")] = 0.0
        data = universe.prepare_universe_data(
            raw,
            requested_start=start,
            requested_end_exclusive=end,
            source="zero-volume test",
        )
        self.assertEqual(data.volumes.loc[session, "USD"], 0.0)

        missing = raw.copy()
        missing.loc[session, ("Volume", "USD")] = np.nan
        with self.assertRaisesRegex(RuntimeError, "Volume is invalid"):
            universe.prepare_universe_data(
                missing,
                requested_start=start,
                requested_end_exclusive=end,
                source="missing-volume test",
            )


class ResidualAndVolatilityTests(unittest.TestCase):
    def test_generalized_residual_matches_frozen_semiconductor_equation(self):
        prices = price_series(420)
        generalized = universe.generalized_residual_frame(
            prices["SMH"],
            prices["QQQ"],
        )
        frozen = core.calculate_residual_signal_frame(prices)
        available = generalized["available"]
        np.testing.assert_allclose(
            generalized.loc[available, "momentum"],
            frozen.loc[available, "residual_momentum"],
            rtol=0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            generalized.loc[available, "z"],
            frozen.loc[available, "residual_z"],
            rtol=0,
            atol=1e-14,
        )

    def test_fast_har_matches_audited_expanding_fit(self):
        close = price_series(1050)["QQQ"]
        minimum = 200
        fast = universe.expanding_har_volatility(
            close,
            minimum_observations=minimum,
        )
        design = research.har_design(close)
        date = close.index[-1]
        audited = research.har_forecast_at(
            design,
            date,
            minimum_observations=minimum,
        )
        assert audited.prediction is not None
        assert audited.smearing_factor is not None
        expected = math.sqrt(
            math.exp(audited.prediction) * audited.smearing_factor
        )
        self.assertEqual(
            fast.loc[date, "sample_count"],
            audited.sample_count,
        )
        self.assertAlmostEqual(fast.loc[date, "model"], expected, places=12)

    def test_two_asset_forecast_matches_frozen_portfolio_equation(self):
        weights = universe.target_with_satellite({"QLD": 1.0}, "SOXL", 0.25)
        volatility = pd.Series({"QLD": 0.40, "SOXL": 0.80})
        first = pd.DataFrame(
            [[1.0, 0.7], [0.7, 1.0]],
            index=["QLD", "SOXL"],
            columns=["QLD", "SOXL"],
        )
        second = pd.DataFrame(
            [[1.0, 0.8], [0.8, 1.0]],
            index=["QLD", "SOXL"],
            columns=["QLD", "SOXL"],
        )
        actual = universe.forecast_weight_volatility(
            weights,
            volatility,
            (first, second),
        )
        expected = core.forecast_portfolio_volatility(0.40, 0.80, 0.8, 0.25)
        self.assertAlmostEqual(actual, expected)

    def test_satellite_target_replaces_base_pro_rata(self):
        target = universe.target_with_satellite(
            {"QLD": 0.8, "GLD": 0.2},
            "SOXL",
            0.35,
        )
        self.assertAlmostEqual(target["QLD"], 0.52)
        self.assertAlmostEqual(target["GLD"], 0.13)
        self.assertAlmostEqual(target["SOXL"], 0.35)
        self.assertAlmostEqual(sum(target.values()), 1.0)


class DriftAndExecutionTests(unittest.TestCase):
    def test_individual_and_aggregate_drift_trigger_at_exactly_five_points(self):
        strategic = universe._complete_weights({"QLD": 0.8, "GLD": 0.2})
        below = dict(strategic)
        below["QLD"] = 0.750001
        below["GLD"] = 0.249999
        exact = dict(strategic)
        exact["QLD"] = 0.75
        exact["GLD"] = 0.25
        self.assertFalse(universe.drift_triggered(below, strategic))
        self.assertTrue(universe.drift_triggered(exact, strategic))

        equity_strategic = universe._complete_weights(
            {"QLD": 0.35, "SOXL": 0.35, "GLD": 0.30}
        )
        equity_actual = dict(equity_strategic)
        equity_actual["QLD"] = 0.325
        equity_actual["SOXL"] = 0.325
        equity_actual["GLD"] = 0.35
        self.assertTrue(
            all(
                abs(equity_actual[ticker] - equity_strategic[ticker])
                < universe.DRIFT_TRIGGER
                for ticker in universe.TRADED_TICKERS
            )
        )
        self.assertTrue(
            universe.drift_triggered(equity_actual, equity_strategic)
        )

    def test_inner_projection_finishes_inside_the_2_5_point_band(self):
        strategic = universe._complete_weights(
            {"UPRO": 0.6, "GLD": 0.2, "IEF": 0.2}
        )
        actual = dict(strategic)
        actual["UPRO"] = 0.72
        actual["GLD"] = 0.13
        actual["IEF"] = 0.15
        target = universe.inner_band_target(actual, strategic)
        self.assertAlmostEqual(sum(target.values()), 1.0)
        for ticker in universe.TRADED_TICKERS:
            self.assertLessEqual(
                abs(target[ticker] - strategic[ticker]),
                universe.DRIFT_DESTINATION + 1e-10,
            )
            if strategic[ticker] == 0.0:
                self.assertEqual(target[ticker], 0.0)

    def test_signal_executes_at_the_next_open_from_actual_shares(self):
        data = market_data()
        sessions = data.sessions
        schedule = constant_schedule(sessions, {"QLD": 1.0})
        replacement = pd.Series(
            universe._complete_weights({"SSO": 1.0})
        ).reindex(schedule.columns)
        schedule.loc[sessions[1]:, :] = replacement.to_numpy()
        result = universe.simulate_schedule(
            data,
            schedule,
            candidate="live_qld_soxl",
            cost_bps=0.0,
            first_signal_date=sessions[0],
        )
        self.assertEqual(result.ledger.index[0], sessions[1])
        self.assertEqual(
            result.ledger.iloc[0]["fill_signal_date"],
            sessions[0],
        )
        self.assertEqual(
            result.ledger.iloc[1]["fill_signal_date"],
            sessions[1],
        )
        self.assertGreater(result.ledger.iloc[0]["qld_shares"], 0.0)
        self.assertEqual(result.ledger.iloc[1]["qld_shares"], 0.0)
        self.assertGreater(result.ledger.iloc[1]["sso_shares"], 0.0)


class ComparisonTests(unittest.TestCase):
    def test_drawdown_recovery_is_measured_from_starting_capital(self):
        index = pd.bdate_range("2024-01-02", periods=4)
        diagnostics = universe.drawdown_diagnostics(
            pd.Series([9_000.0, 8_000.0, 9_500.0, 10_100.0], index=index)
        )
        self.assertAlmostEqual(diagnostics["maximum_drawdown"], -0.20)
        self.assertEqual(diagnostics["drawdown_trough_date"], "2024-01-03")
        self.assertEqual(diagnostics["drawdown_recovery_date"], "2024-01-05")
        self.assertEqual(diagnostics["peak_to_trough_sessions"], 2)
        self.assertEqual(diagnostics["trough_to_recovery_sessions"], 2)

    def test_pareto_frontier_removes_fully_dominated_path(self):
        index = pd.bdate_range("2024-01-02", periods=3)
        ledger = pd.DataFrame({"nav": [10_000.0, 10_100.0, 10_200.0]}, index=index)
        better = universe.PathResult(
            "live_qld_soxl",
            25.0,
            ledger,
            {
                "annualized_log_growth": 0.30,
                "maximum_drawdown": -0.40,
                "sharpe_zero_cash_rate": 1.0,
                "calmar": 0.8,
            },
            False,
        )
        worse = universe.PathResult(
            "sso_soxl",
            25.0,
            ledger,
            {
                "annualized_log_growth": 0.20,
                "maximum_drawdown": -0.50,
                "sharpe_zero_cash_rate": 0.8,
                "calmar": 0.4,
            },
            False,
        )
        self.assertEqual(
            universe.pareto_frontier(
                {"live_qld_soxl": better, "sso_soxl": worse}
            ),
            ["live_qld_soxl"],
        )


if __name__ == "__main__":
    unittest.main()
