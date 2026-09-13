from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import alpha_core as alpha


class AllocationTests(unittest.TestCase):
    def test_tqqq_target_is_exact(self):
        self.assertEqual(alpha.target_weights(True), {"TQQQ": 0.4, "DBMF": 0.2, "ZROZ": 0.2, "UGL": 0.2})

    def test_upro_target_is_exact(self):
        self.assertEqual(alpha.target_weights(False), {"UPRO": 0.4, "DBMF": 0.2, "ZROZ": 0.2, "UGL": 0.2})

    def test_both_states_have_same_advertised_exposure(self):
        self.assertAlmostEqual(alpha.advertised_daily_exposure(True), 2.0)
        self.assertAlmostEqual(alpha.advertised_daily_exposure(False), 2.0)

    def test_boolean_inputs_are_strict(self):
        with self.assertRaises(ValueError):
            alpha.target_weights(1)  # type: ignore[arg-type]


class RouterTests(unittest.TestCase):
    def test_bearish_close_switches_immediately_to_upro(self):
        state = alpha.EquityRouterState(tqqq_active=True, bullish_streak=12)
        result = alpha.advance_equity_router(state, signal_date=pd.Timestamp("2026-09-10"), trend_positive=False)
        self.assertFalse(result.state.tqqq_active)
        self.assertEqual(result.state.bullish_streak, 0)
        self.assertTrue(result.structural_change)
        self.assertEqual(result.reason, "TREND_SWITCH_TO_UPRO")

    def test_two_distinct_bullish_closes_are_required(self):
        first = alpha.advance_equity_router(alpha.EquityRouterState(), signal_date=pd.Timestamp("2026-09-09"), trend_positive=True)
        self.assertFalse(first.state.tqqq_active)
        self.assertEqual(first.state.bullish_streak, 1)
        second = alpha.advance_equity_router(first.state, signal_date=pd.Timestamp("2026-09-10"), trend_positive=np.bool_(True))
        self.assertTrue(second.state.tqqq_active)
        self.assertTrue(second.structural_change)

    def test_duplicate_date_does_not_advance_streak(self):
        state = alpha.EquityRouterState(bullish_streak=1, last_processed_signal_date="2026-09-10")
        result = alpha.advance_equity_router(state, signal_date=pd.Timestamp("2026-09-10"), trend_positive=True)
        self.assertEqual(result.state, state)
        self.assertEqual(result.reason, "SAME_DATE")

    def test_future_state_is_rejected(self):
        state = alpha.EquityRouterState(last_processed_signal_date="2026-09-11")
        with self.assertRaises(ValueError):
            alpha.advance_equity_router(state, signal_date=pd.Timestamp("2026-09-10"), trend_positive=True)

    def test_active_tqqq_remains_active_on_bullish_close(self):
        state = alpha.EquityRouterState(tqqq_active=True, bullish_streak=2)
        result = alpha.advance_equity_router(state, signal_date=pd.Timestamp("2026-09-10"), trend_positive=True)
        self.assertTrue(result.state.tqqq_active)
        self.assertFalse(result.structural_change)
        self.assertEqual(result.reason, "TQQQ_HOLD")


if __name__ == "__main__":
    unittest.main()
