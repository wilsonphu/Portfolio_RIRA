from __future__ import annotations

import unittest
from datetime import date

import contribution_core as contribution


class CalendarTests(unittest.TestCase):
    def test_calendar_releases_half_immediately_and_finishes_in_november(self):
        expected = {1: 0.5, 3: 0.6, 5: 0.7, 7: 0.8, 9: 0.9, 11: 1.0}
        for month, fraction in expected.items():
            with self.subTest(month=month):
                self.assertAlmostEqual(
                    contribution.calendar_fraction(date(2026, month, 15)), fraction
                )


class ReleaseTests(unittest.TestCase):
    def evaluate(self, **overrides):
        values = {
            "as_of": date(2026, 1, 15),
            "qqq_close": 110.0,
            "qqq_sma_50": 105.0,
            "qqq_sma_200": 100.0,
            "qqq_high_63": 110.0,
            "pullback_used": False,
            "drawdown_10_used": False,
            "drawdown_20_used": False,
        }
        values.update(overrides)
        return contribution.evaluate_release(**values)

    def test_bull_market_does_not_delay_calendar_deployment(self):
        result = self.evaluate()
        self.assertEqual(result.target_fraction, 0.5)
        self.assertFalse(result.use_pullback)

    def test_pullback_above_sma200_advances_one_tranche_once(self):
        result = self.evaluate(qqq_close=102.0)
        self.assertEqual(result.target_fraction, 0.6)
        self.assertTrue(result.use_pullback)
        repeated = self.evaluate(qqq_close=102.0, pullback_used=True)
        self.assertEqual(repeated.target_fraction, 0.6)
        self.assertFalse(repeated.use_pullback)

    def test_ten_percent_drawdown_advances_one_tranche(self):
        result = self.evaluate(qqq_close=90.0, qqq_high_63=100.0)
        self.assertEqual(result.target_fraction, 0.6)
        self.assertTrue(result.use_drawdown_10)

    def test_twenty_percent_drawdown_releases_everything(self):
        result = self.evaluate(qqq_close=80.0, qqq_high_63=100.0)
        self.assertEqual(result.target_fraction, 1.0)
        self.assertTrue(result.use_drawdown_20)

    def test_invalid_market_input_fails_closed(self):
        with self.assertRaises(ValueError):
            self.evaluate(qqq_high_63=0.0)


if __name__ == "__main__":
    unittest.main()
