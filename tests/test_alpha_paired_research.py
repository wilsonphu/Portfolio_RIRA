from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import exchange_calendars as xcals
import numpy as np
import pandas as pd

import alpha_paired_research as paired
import alpha_research as alpha
import legacy_original_research as legacy


def _sessions(count: int = 320) -> pd.DatetimeIndex:
    calendar = xcals.get_calendar("XNYS")
    values = pd.DatetimeIndex(
        calendar.sessions_in_range("2010-01-01", "2024-12-31")
    )[:count]
    if values.tz is not None:
        values = values.tz_convert(None)
    return values.normalize()


def _union_frames(
    *,
    count: int = 320,
    common_position: int = 260,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    sessions = _sessions(count)
    t = np.arange(len(sessions), dtype=float)
    closes = pd.DataFrame(
        index=sessions,
        columns=paired.UNION_TICKERS,
        dtype=float,
    )
    parameters = {
        "QQQ": (100.0, 0.0010, 0.0060, 0.15),
        "SMH": (80.0, 0.0012, 0.0080, 0.41),
        "QLD": (40.0, 0.0018, 0.0120, 0.73),
        "SOXL": (25.0, 0.0020, 0.0180, 1.07),
        "SPY": (110.0, 0.0007, 0.0045, 1.31),
        "TECL": (35.0, 0.0016, 0.0130, 1.61),
        "SPMO": (45.0, 0.0009, 0.0070, 1.91),
        "GLD": (70.0, 0.0002, 0.0035, 2.17),
    }
    for ticker, (initial, drift, amplitude, phase) in parameters.items():
        log_returns = (
            drift
            + amplitude * np.sin(0.29 * t + phase)
            + amplitude * 0.35 * np.cos(0.071 * t + phase)
        )
        closes[ticker] = initial * np.exp(np.cumsum(log_returns))

    opens = closes.copy()
    for index, ticker in enumerate(paired.UNION_TICKERS):
        opens[ticker] = closes[ticker] * (
            1.0 + 0.002 * np.cos(0.19 * t + index * 0.23)
        )
    volumes = pd.DataFrame(
        {
            ticker: (
                1_000_000.0
                + 100_000.0 * np.sin(0.11 * t + index * 0.31)
            )
            for index, ticker in enumerate(paired.UNION_TICKERS)
        },
        index=sessions,
    )
    soxl_preinception = sessions < pd.Timestamp(
        alpha.MODEL_HISTORY_START
    )
    for frame in (opens, closes, volumes):
        frame.loc[soxl_preinception, "SOXL"] = np.nan
    opens.loc[sessions[:common_position], "SPMO"] = np.nan
    closes.loc[sessions[:common_position], "SPMO"] = np.nan
    return opens, closes, volumes, sessions


def _validated_union() -> paired.UnionMarketData:
    opens, closes, volumes, sessions = _union_frames()
    return paired.validate_union_frames(
        opens,
        closes,
        volumes,
        requested_start=sessions[0].date().isoformat(),
        requested_end_exclusive=(
            sessions[-1] + pd.Timedelta(days=1)
        ).date().isoformat(),
        source="deterministic paired unit fixture",
    )


class SnapshotAndViewTests(unittest.TestCase):
    def test_union_snapshot_round_trip_is_exact_and_views_are_identical(
        self,
    ) -> None:
        data = _validated_union()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "paired_union.csv"
            paired.save_union_snapshot(data, path)
            loaded = paired.load_union_snapshot(
                path,
                requested_start=data.requested_start,
                requested_end_exclusive=data.requested_end_exclusive,
            )

        self.assertEqual(loaded.fingerprint, data.fingerprint)
        for expected, actual in (
            (data.opens, loaded.opens),
            (data.closes, loaded.closes),
            (data.volumes, loaded.volumes),
        ):
            left = expected.to_numpy(dtype=np.float64)
            right = actual.to_numpy(dtype=np.float64)
            self.assertTrue(np.array_equal(np.isnan(left), np.isnan(right)))
            finite = ~np.isnan(left)
            self.assertTrue(
                np.array_equal(
                    left[finite].view(np.uint64),
                    right[finite].view(np.uint64),
                )
            )

        views = paired.create_engine_views(loaded)
        self.assertEqual(len(views.verification_fingerprint), 64)
        self.assertEqual(
            views.original.common_start,
            loaded.sessions[260],
        )
        for ticker in alpha.TICKERS:
            self.assertTrue(
                views.current.closes[ticker].equals(
                    views.original.closes[ticker]
                )
            )
        self.assertTrue(
            math.isnan(
                float(views.original.closes.loc[loaded.sessions[204], "SPMO"])
            )
        )


class AlignmentTests(unittest.TestCase):
    def test_shared_date_alignment_covers_every_candidate_and_cost(self) -> None:
        dates = _sessions(12)

        def result(index: pd.DatetimeIndex) -> SimpleNamespace:
            return SimpleNamespace(
                ledger=pd.DataFrame({"nav": np.arange(len(index)) + 1.0}, index=index)
            )

        current = {
            (name, cost): result(dates)
            for name in alpha.FROZEN_CANDIDATE_NAMES
            for cost in paired.CONFIRMATORY_COSTS_BPS
        }
        original = {
            cost: result(dates)
            for cost in paired.CONFIRMATORY_COSTS_BPS
        }
        audit = paired.verify_shared_date_alignment(current, original)
        self.assertTrue(audit["aligned"])
        self.assertEqual(audit["observations"], len(dates))
        self.assertEqual(
            audit["current_candidates"],
            list(alpha.FROZEN_CANDIDATE_NAMES),
        )

        broken = dict(current)
        broken[(alpha.FROZEN_CANDIDATE_NAMES[-1], 25.0)] = result(dates[1:])
        with self.assertRaisesRegex(RuntimeError, "alignment failed"):
            paired.verify_shared_date_alignment(broken, original)


class MultiplicityTests(unittest.TestCase):
    def _family_report(self) -> dict[str, object]:
        trial_count = len(alpha.FROZEN_CANDIDATE_NAMES)
        challengers = [
            name
            for name in alpha.FROZEN_CANDIDATE_NAMES
            if name != "qld_buy_hold"
        ]
        return {
            "white_reality_checks": {
                "vs_qld_buy_hold": {
                    "honest_trial_count": trial_count,
                    "challengers": challengers,
                },
                "vs_static_65_35": {
                    "honest_trial_count": trial_count,
                    "challengers": [
                        name
                        for name in alpha.FROZEN_CANDIDATE_NAMES
                        if name != "static_65_35"
                    ],
                },
            },
            "cscv_pbo": {
                "available": True,
                "honest_trial_count": trial_count,
            },
            "deflated_sharpe": {
                "honest_trial_count": trial_count,
            },
        }

    def test_original_is_external_and_excluded_from_trial_count(self) -> None:
        report = self._family_report()
        audit = paired.external_baseline_trial_audit(report)
        self.assertEqual(
            audit["frozen_current_family_trial_count"],
            len(alpha.FROZEN_CANDIDATE_NAMES),
        )
        self.assertFalse(audit["external_baseline_in_trial_count"])
        self.assertFalse(audit["external_baseline_in_reality_check"])
        self.assertEqual(
            audit["external_baselines"],
            [legacy.LEGACY_CANDIDATE.name],
        )

        report["white_reality_checks"]["vs_qld_buy_hold"][
            "challengers"
        ].append(legacy.LEGACY_CANDIDATE.name)
        with self.assertRaisesRegex(RuntimeError, "included"):
            paired.external_baseline_trial_audit(report)

    def test_trial_count_mismatch_is_rejected(self) -> None:
        report = self._family_report()
        report["white_reality_checks"]["vs_qld_buy_hold"][
            "honest_trial_count"
        ] += 1
        with self.assertRaisesRegex(RuntimeError, "trial count"):
            paired.external_baseline_trial_audit(report)


if __name__ == "__main__":
    unittest.main()
