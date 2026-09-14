import unittest
from datetime import date

import contribution_core as contribution


class ContributionTests(unittest.TestCase):
    def test_annual_release_has_no_market_inputs(self):
        result = contribution.evaluate_release(
            as_of=date(2026, 1, 2), already_released=0.0, budget=7500.0
        )
        self.assertEqual(result.reasons, ("ANNUAL_CONTRIBUTION",))
        self.assertEqual(result.target_fraction, 0.0)

    def test_invalid_values_fail_closed(self):
        with self.assertRaises(ValueError):
            contribution.evaluate_release(
                as_of=date(2026, 1, 2), already_released=-1.0, budget=7500.0
            )


if __name__ == "__main__":
    unittest.main()
