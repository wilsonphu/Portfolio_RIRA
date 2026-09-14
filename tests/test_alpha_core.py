import unittest

import alpha_core as alpha


class StaticAllocationTests(unittest.TestCase):
    def test_target_is_exact(self):
        self.assertEqual(alpha.target_weights(), {
            "TQQQ": 0.35,
            "DBMF": 0.25,
            "UGL": 0.20,
            "ZROZ": 0.15,
            "BTAL": 0.05,
        })

    def test_target_is_a_fresh_copy(self):
        target = alpha.target_weights()
        target["TQQQ"] = 0.0
        self.assertEqual(alpha.target_weights()["TQQQ"], 0.35)

    def test_advertised_exposure(self):
        self.assertAlmostEqual(alpha.advertised_daily_exposure(), 1.90)

    def test_validation(self):
        alpha.validate_target()


if __name__ == "__main__":
    unittest.main()
