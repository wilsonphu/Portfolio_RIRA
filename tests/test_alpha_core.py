from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import alpha_core as alpha


class AllocationTests(unittest.TestCase):
    def test_normal_target_is_exact(self):
        self.assertEqual(
            alpha.target_weights(False),
            {"TQQQ": 0.4, "DBMF": 0.2, "ZROZ": 0.2, "UGL": 0.2},
        )

    def test_extreme_bear_target_is_exact_and_btal_is_capped(self):
        self.assertEqual(
            alpha.target_weights(True),
            {"TQQQ": 0.3, "DBMF": 0.2, "ZROZ": 0.2, "UGL": 0.2, "BTAL": 0.1},
        )

    def test_hedge_reduces_advertised_exposure(self):
        self.assertAlmostEqual(alpha.advertised_daily_exposure(False), 2.0)
        self.assertAlmostEqual(alpha.advertised_daily_exposure(True), 1.8)

    def test_boolean_inputs_are_strict(self):
        with self.assertRaises(ValueError):
            alpha.target_weights(1)  # type: ignore[arg-type]


class CrisisHedgeTests(unittest.TestCase):
    def test_exact_eight_percent_entry_requires_broad_confirmation(self):
        extreme, recovery = alpha.crisis_conditions(
            qqq_close=92.0,
            qqq_sma_200=100.0,
            spy_close=99.0,
            spy_sma_200=100.0,
            spy_sma_50=101.0,
        )
        self.assertTrue(extreme)
        self.assertFalse(recovery)
        extreme_without_spy, _ = alpha.crisis_conditions(
            qqq_close=92.0,
            qqq_sma_200=100.0,
            spy_close=100.0,
            spy_sma_200=100.0,
            spy_sma_50=101.0,
        )
        self.assertFalse(extreme_without_spy)

    def test_exact_two_percent_recovery_requires_spy_above_sma50(self):
        extreme, recovery = alpha.crisis_conditions(
            qqq_close=98.0,
            qqq_sma_200=100.0,
            spy_close=102.0,
            spy_sma_200=105.0,
            spy_sma_50=101.0,
        )
        self.assertFalse(extreme)
        self.assertTrue(recovery)

    def test_two_distinct_extreme_closes_activate_btal(self):
        first = alpha.advance_crisis_hedge(
            alpha.CrisisHedgeState(),
            signal_date=pd.Timestamp("2026-09-09"),
            extreme_bearish=True,
            recovery_confirmed=False,
        )
        self.assertFalse(first.state.btal_active)
        self.assertEqual(first.state.entry_streak, 1)
        second = alpha.advance_crisis_hedge(
            first.state,
            signal_date=pd.Timestamp("2026-09-10"),
            extreme_bearish=np.bool_(True),
            recovery_confirmed=False,
        )
        self.assertTrue(second.state.btal_active)
        self.assertTrue(second.structural_change)
        self.assertEqual(second.reason, "EXTREME_HEDGE_ENTRY")

    def test_non_extreme_close_resets_entry_streak(self):
        state = alpha.CrisisHedgeState(entry_streak=1)
        result = alpha.advance_crisis_hedge(
            state,
            signal_date=pd.Timestamp("2026-09-10"),
            extreme_bearish=False,
            recovery_confirmed=False,
        )
        self.assertEqual(result.state.entry_streak, 0)
        self.assertFalse(result.state.btal_active)

    def test_two_distinct_recovery_closes_exit_btal(self):
        state = alpha.CrisisHedgeState(btal_active=True)
        first = alpha.advance_crisis_hedge(
            state,
            signal_date=pd.Timestamp("2026-09-09"),
            extreme_bearish=False,
            recovery_confirmed=True,
        )
        self.assertTrue(first.state.btal_active)
        self.assertEqual(first.state.exit_streak, 1)
        second = alpha.advance_crisis_hedge(
            first.state,
            signal_date=pd.Timestamp("2026-09-10"),
            extreme_bearish=False,
            recovery_confirmed=True,
        )
        self.assertFalse(second.state.btal_active)
        self.assertTrue(second.structural_change)
        self.assertEqual(second.reason, "EXTREME_HEDGE_EXIT")

    def test_duplicate_date_does_not_advance_streak(self):
        state = alpha.CrisisHedgeState(
            entry_streak=1, last_processed_signal_date="2026-09-10"
        )
        result = alpha.advance_crisis_hedge(
            state,
            signal_date=pd.Timestamp("2026-09-10"),
            extreme_bearish=True,
            recovery_confirmed=False,
        )
        self.assertEqual(result.state, state)
        self.assertEqual(result.reason, "SAME_DATE")

    def test_future_state_is_rejected(self):
        state = alpha.CrisisHedgeState(last_processed_signal_date="2026-09-11")
        with self.assertRaises(ValueError):
            alpha.advance_crisis_hedge(
                state,
                signal_date=pd.Timestamp("2026-09-10"),
                extreme_bearish=True,
                recovery_confirmed=False,
            )


if __name__ == "__main__":
    unittest.main()
