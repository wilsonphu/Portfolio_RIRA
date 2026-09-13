import unittest

import numpy as np
import pandas as pd

import performance_core as perf


def synthetic_prices(sessions: int = 900, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-01-04", periods=sessions)
    spec = {
        "QQQ": (0.0004, 0.013),
        "TQQQ": (0.0009, 0.039),
        "UPRO": (0.0008, 0.030),
        "DBMF": (0.0002, 0.006),
        "ZROZ": (0.0001, 0.012),
        "UGL": (0.0002, 0.019),
        "SGOV": (0.00017, 0.00002),
    }
    data = {}
    for ticker, (mu, sigma) in spec.items():
        steps = rng.normal(mu, sigma, sessions)
        data[ticker] = 100.0 * np.cumprod(1.0 + steps)
    return pd.DataFrame(data, index=index)


class CurveTests(unittest.TestCase):
    def test_annual_rebalance_matches_single_asset_growth(self):
        prices = synthetic_prices()
        returns = perf.daily_returns(prices, ["QQQ"])
        curve = perf.annual_rebalanced_curve(returns, {"QQQ": 1.0})
        expected = float(prices["QQQ"].iloc[-1] / prices["QQQ"].iloc[0])
        self.assertAlmostEqual(float(curve.iloc[-1]), expected, places=6)

    def test_weights_are_normalised(self):
        prices = synthetic_prices()
        returns = perf.daily_returns(prices, ["QQQ", "ZROZ"])
        a = perf.annual_rebalanced_curve(returns, {"QQQ": 0.6, "ZROZ": 0.4})
        b = perf.annual_rebalanced_curve(returns, {"QQQ": 60.0, "ZROZ": 40.0})
        self.assertAlmostEqual(float(a.iloc[-1]), float(b.iloc[-1]), places=9)

    def test_empty_book_is_safe(self):
        prices = synthetic_prices()
        returns = perf.daily_returns(prices, ["QQQ"])
        self.assertTrue(perf.annual_rebalanced_curve(returns, {}).empty)
        self.assertTrue(perf.annual_rebalanced_curve(returns, {"QQQ": 0.0}).empty)

    def test_full_return_and_drawdown_include_the_first_observation(self):
        index = pd.bdate_range("2026-01-02", periods=31)
        curve = pd.Series([0.5] * len(index), index=index)
        metrics = perf.trailing_returns(curve)
        self.assertAlmostEqual(metrics["full"], 0.5 ** (252 / 31) - 1.0)
        self.assertEqual(metrics["max_drawdown"], -0.5)

    def test_exact_one_year_sample_reports_one_year_return(self):
        index = pd.bdate_range("2025-01-02", periods=252)
        curve = pd.Series(np.linspace(1.0, 1.25, len(index)), index=index)
        self.assertAlmostEqual(perf.trailing_returns(curve)["1y"], 0.25)


class RiskTests(unittest.TestCase):
    def test_risk_contributions_sum_to_one(self):
        prices = synthetic_prices()
        returns = perf.daily_returns(prices, ["TQQQ", "DBMF", "ZROZ", "UGL"])
        weights = {"TQQQ": 0.4, "DBMF": 0.2, "ZROZ": 0.2, "UGL": 0.2}
        contributions = perf.sleeve_risk_contribution(returns, weights)
        self.assertAlmostEqual(sum(contributions.values()), 1.0, places=6)

    def test_risk_contribution_differs_from_advertised_multiplier(self):
        """The whole point: ZROZ carries a 1.0x multiplier and real volatility."""
        prices = synthetic_prices()
        returns = perf.daily_returns(prices, ["TQQQ", "DBMF", "ZROZ", "UGL"])
        weights = {"TQQQ": 0.4, "DBMF": 0.2, "ZROZ": 0.2, "UGL": 0.2}
        contributions = perf.sleeve_risk_contribution(returns, weights)
        self.assertGreater(contributions["ZROZ"], contributions["DBMF"])

    def test_volatility_windows_are_annualised(self):
        prices = synthetic_prices()
        returns = perf.daily_returns(prices, ["QQQ"])["QQQ"]
        vols = perf.realized_volatility(returns)
        self.assertIn("252d", vols)
        self.assertGreater(vols["252d"], 0.0)

    def test_short_history_returns_empty_rather_than_raising(self):
        prices = synthetic_prices(sessions=10)
        returns = perf.daily_returns(prices, ["QQQ", "ZROZ"])
        self.assertEqual(perf.sleeve_risk_contribution(returns, {"QQQ": 1.0}), {})


class FinancingTests(unittest.TestCase):
    def test_short_rate_recovered_from_cash_proxy(self):
        prices = synthetic_prices()
        rate = perf.estimate_short_rate(prices, "SGOV")
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 0.25)

    def test_missing_cash_proxy_is_zero(self):
        prices = synthetic_prices().drop(columns=["SGOV"])
        self.assertEqual(perf.estimate_short_rate(prices, "SGOV"), 0.0)

    def test_financing_drag_counts_only_borrowed_notional(self):
        weights = {"TQQQ": 0.40, "UGL": 0.20, "DBMF": 0.20, "ZROZ": 0.20}
        multipliers = {"TQQQ": 3.0, "UGL": 2.0, "DBMF": 1.0, "ZROZ": 1.0}
        drag = perf.financing_drag(weights, multipliers, 0.04)
        # 0.40 * 2 borrowed + 0.20 * 1 borrowed = 1.0x notional at 4%
        self.assertAlmostEqual(drag, 0.04, places=9)

    def test_unlevered_book_has_no_financing_drag(self):
        drag = perf.financing_drag({"QQQ": 1.0}, {"QQQ": 1.0}, 0.05)
        self.assertEqual(drag, 0.0)


class ReportTests(unittest.TestCase):
    def report(self, **overrides):
        kwargs = dict(
            reference_books={
                "unlevered_qqq": {"QQQ": 1.0},
                "permanent_growth": {"TQQQ": .4, "DBMF": .2, "ZROZ": .2, "UGL": .2},
            },
            live_book={"TQQQ": .4, "DBMF": .2, "ZROZ": .2, "UGL": .2},
            index_ticker="QQQ", growth="TQQQ", defensive="UPRO",
            cash_proxy="SGOV", sma_window=200,
            multipliers={"TQQQ": 3.0, "UPRO": 3.0, "UGL": 2.0, "DBMF": 1.0, "ZROZ": 1.0},
            current_weights={"TQQQ": .4, "DBMF": .2, "ZROZ": .2, "UGL": .2},
        )
        kwargs.update(overrides)
        return perf.build_performance_report(synthetic_prices(), **kwargs)

    def test_report_includes_live_rule_and_references(self):
        report = self.report()
        self.assertIn("live_rule", report.paper_track)
        self.assertIn("unlevered_qqq", report.paper_track)
        self.assertIn("permanent_growth", report.paper_track)

    def test_all_values_are_json_safe(self):
        report = self.report()
        for block in (report.paper_track.values()):
            for value in block.values():
                self.assertTrue(np.isfinite(value))
        for value in report.realized_volatility.values():
            self.assertTrue(np.isfinite(value))
        self.assertTrue(np.isfinite(report.financing_drag))

    def test_drawdown_is_non_positive(self):
        report = self.report()
        for block in report.paper_track.values():
            self.assertLessEqual(block["max_drawdown"], 0.0)

    def test_insufficient_history_degrades_gracefully(self):
        tiny = synthetic_prices(sessions=5)
        report = perf.build_performance_report(
            tiny,
            reference_books={"unlevered_qqq": {"QQQ": 1.0}},
            live_book={"TQQQ": 1.0},
            index_ticker="QQQ", growth="TQQQ", defensive="UPRO",
            cash_proxy="SGOV", sma_window=200,
            multipliers={"TQQQ": 3.0}, current_weights={"TQQQ": 1.0},
        )
        self.assertEqual(report.sample_sessions, 0)
        self.assertEqual(report.paper_track, {})

    def test_cash_proxy_does_not_truncate_the_common_sample(self):
        prices = synthetic_prices().copy()
        prices.loc[prices.index[:500], "SGOV"] = np.nan
        baseline = self.report()
        delayed_proxy = perf.build_performance_report(
            prices,
            reference_books={
                "unlevered_qqq": {"QQQ": 1.0},
                "permanent_growth": {
                    "TQQQ": .4, "DBMF": .2, "ZROZ": .2, "UGL": .2
                },
            },
            live_book={"TQQQ": .4, "DBMF": .2, "ZROZ": .2, "UGL": .2},
            index_ticker="QQQ", growth="TQQQ", defensive="UPRO",
            cash_proxy="SGOV", sma_window=200,
            multipliers={"TQQQ": 3.0, "UPRO": 3.0, "UGL": 2.0},
            current_weights={"TQQQ": .4, "DBMF": .2, "ZROZ": .2, "UGL": .2},
        )
        self.assertEqual(delayed_proxy.sample_sessions, baseline.sample_sessions)

    def test_all_cash_has_no_reported_risk_or_financing_drag(self):
        report = self.report(current_weights={"CASH": 1.0})
        self.assertEqual(report.sleeve_risk_contribution, {})
        self.assertEqual(report.financing_drag, 0.0)

    def test_missing_required_asset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing required performance data"):
            perf.build_performance_report(
                synthetic_prices().drop(columns=["DBMF"]),
                reference_books={"live": {"TQQQ": .5, "DBMF": .5}},
                live_book={"TQQQ": .5, "DBMF": .5},
                index_ticker="QQQ", growth="TQQQ", defensive="UPRO",
                cash_proxy="SGOV", sma_window=200,
                multipliers={"TQQQ": 3.0}, current_weights={},
            )


if __name__ == "__main__":
    unittest.main()
