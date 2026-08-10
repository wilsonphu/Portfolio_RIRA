import math
import unittest

import numpy as np
import pandas as pd

import tcn_shadow as shadow


def synthetic_prices(rows: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-03-11", periods=rows)
    common = rng.normal(0.0004, 0.009, rows)
    data = {}
    for index, ticker in enumerate(shadow.REQUIRED_TICKERS):
        scale = 1.0 + 0.07 * index
        independent = rng.normal(0.0, 0.004 + 0.0003 * index, rows)
        returns = scale * common + independent
        if ticker == "^VIX":
            returns = -1.5 * common + rng.normal(0.0, 0.025, rows)
        data[ticker] = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(data, index=dates)


class FeatureAndLabelTests(unittest.TestCase):
    def test_feature_frame_has_exact_channels_and_is_causal(self):
        prices = synthetic_prices(340)
        first = shadow.build_feature_frame(prices)
        changed = prices.copy()
        changed.iloc[-1, changed.columns.get_loc("QQQ")] *= 1.5
        second = shadow.build_feature_frame(changed)
        self.assertEqual(tuple(first.columns), shadow.FEATURE_NAMES)
        pd.testing.assert_frame_equal(first.iloc[:-1], second.iloc[:-1])
        self.assertFalse(first.iloc[-1].equals(second.iloc[-1]))

    def test_relative_label_uses_next_open_and_t_plus_22_open(self):
        dates = pd.bdate_range("2020-01-01", periods=30)
        opens = pd.DataFrame(
            {
                "QLD": np.exp(np.arange(30) * 0.01),
                "SOXL": np.exp(np.arange(30) * 0.02),
            },
            index=dates,
        )
        labels = shadow.build_relative_label_frame(opens)
        expected = 21 * 0.01 - shadow.LABEL_HURDLE
        self.assertAlmostEqual(labels.iloc[0]["relative_log_growth"], expected)
        self.assertEqual(labels.iloc[0]["label"], 1.0)
        self.assertEqual(labels.iloc[0]["label_end_date"], dates[22])
        self.assertTrue(labels.iloc[-22:]["label"].isna().all())

    def test_sequences_never_use_rows_after_origin(self):
        prices = synthetic_prices(340)
        features = shadow.build_feature_frame(prices)
        dates = shadow.available_sequence_dates(features)
        origin = dates[-2]
        training = features.loc[:origin].dropna()
        mean = training.to_numpy().mean(axis=0)
        std = training.to_numpy().std(axis=0, ddof=1)
        before = shadow.make_sequence_array(features, [origin], mean, std)
        changed = features.copy()
        changed.loc[changed.index > origin] = 999.0
        after = shadow.make_sequence_array(changed, [origin], mean, std)
        np.testing.assert_array_equal(before, after)


class GraphTests(unittest.TestCase):
    def test_graph_score_is_finite_and_nonnegative(self):
        prices = synthetic_prices(100)
        returns = np.log(prices.loc[:, list(shadow.GRAPH_NODES)]).diff().dropna()
        score = shadow.graph_shock_score(returns.iloc[-shadow.GRAPH_WINDOW :])
        self.assertTrue(math.isfinite(score))
        self.assertGreaterEqual(score, 0.0)

    def test_graph_haircut_boundaries(self):
        self.assertEqual(shadow.graph_haircut_from_percentile(0.90), 1.0)
        self.assertAlmostEqual(shadow.graph_haircut_from_percentile(0.95), 0.5)
        self.assertEqual(shadow.graph_haircut_from_percentile(1.0), 0.0)
        with self.assertRaises(ValueError):
            shadow.graph_haircut_from_percentile(1.01)

    def test_graph_percentile_excludes_current_score(self):
        prices = synthetic_prices(900)
        frame = shadow.build_graph_haircut_frame(prices)
        first = frame.index[frame["graph_available"]][0]
        position = frame.index.get_loc(first)
        prior = frame["graph_score"].iloc[
            position - shadow.GRAPH_HISTORY : position
        ].dropna()
        expected = float(np.mean(prior.to_numpy() <= frame.loc[first, "graph_score"]))
        self.assertEqual(len(prior), shadow.GRAPH_HISTORY)
        self.assertAlmostEqual(frame.loc[first, "graph_percentile"], expected)


class CalibrationTests(unittest.TestCase):
    def test_deadband_confidence(self):
        self.assertEqual(shadow.deadband_confidence(0.55), 0.0)
        self.assertAlmostEqual(shadow.deadband_confidence(0.60), 0.5)
        self.assertEqual(shadow.deadband_confidence(0.65), 1.0)
        self.assertEqual(shadow.deadband_confidence(0.90, qualified=False), 0.0)

    def test_platt_map_is_finite_monotone_and_calibrates(self):
        logits = np.linspace(-3.0, 3.0, 200)
        probabilities = 1.0 / (1.0 + np.exp(-(0.6 * logits - 0.3)))
        labels = (np.arange(200) / 200.0 < probabilities).astype(float)
        slope, intercept = shadow.fit_platt_map(logits, labels)
        calibrated = shadow.apply_platt_map(logits, slope, intercept)
        self.assertGreaterEqual(slope, 0.0)
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertTrue((np.diff(calibrated) >= 0.0).all())

    def test_ece_perfect_predictions(self):
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertEqual(
            shadow.expected_calibration_error(labels, labels, bins=4),
            0.0,
        )


class PartitionAndModelTests(unittest.TestCase):
    def test_partition_sizes_and_purges(self):
        dates = pd.bdate_range("2010-01-01", periods=1400)
        split = shadow.split_walk_forward_partitions(dates)
        self.assertIsNotNone(split)
        assert split is not None
        self.assertGreaterEqual(len(split.train), shadow.TRAIN_MINIMUM)
        self.assertEqual(len(split.validation), shadow.VALIDATION_SIZE)
        self.assertEqual(len(split.calibration), shadow.CALIBRATION_SIZE)
        train_end = dates.get_loc(split.train[-1])
        validation_start = dates.get_loc(split.validation[0])
        validation_end = dates.get_loc(split.validation[-1])
        calibration_start = dates.get_loc(split.calibration[0])
        self.assertEqual(validation_start - train_end - 1, shadow.PARTITION_PURGE)
        self.assertEqual(calibration_start - validation_end - 1, shadow.PARTITION_PURGE)

    def test_small_tcn_parameter_count_and_output_shape(self):
        try:
            torch, _, _ = shadow._load_torch()
        except RuntimeError:
            self.skipTest("optional PyTorch research dependency is absent")
        model = shadow.build_tcn_model()
        model.eval()
        values = torch.zeros((3, shadow.LOOKBACK, len(shadow.FEATURE_NAMES)))
        with torch.no_grad():
            output = model(values)
        self.assertEqual(tuple(output.shape), (3,))
        self.assertLess(shadow.tcn_parameter_count(), 10_000)


if __name__ == "__main__":
    unittest.main()
