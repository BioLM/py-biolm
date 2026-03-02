"""
Unit tests for filters.
"""

import unittest

import numpy as np
import pandas as pd

from biolmai.pipeline.filters import (
    ConservedResidueFilter,
    CustomFilter,
    DiversitySamplingFilter,
    HammingDistanceFilter,
    SequenceLengthFilter,
    ThresholdFilter,
    combine_filters,
)


class TestFilters(unittest.TestCase):
    """Test filter implementations."""

    def setUp(self):
        """Create test DataFrame."""
        self.df = pd.DataFrame(
            {
                "sequence": [
                    "MKTAYIAKQRQ",
                    "MKLAVIDSAQ",
                    "MKTAYIDSAQ",
                    "MKTAY",  # Short
                    "MKTAYIAKQRQGHQAMAEIKQGHQAMAEIKQ",  # Long
                ],
                "tm": [65.0, 55.0, 70.0, 45.0, 60.0],
                "plddt": [85.0, 75.0, 90.0, 70.0, 80.0],
                "temperature": [0.5, 1.0, 1.0, 1.5, 0.5],
            }
        )

    def test_threshold_filter_min(self):
        """Test ThresholdFilter with minimum."""
        filter_obj = ThresholdFilter("tm", min_value=60)

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 3)
        self.assertTrue((df_filtered["tm"] >= 60).all())

    def test_threshold_filter_max(self):
        """Test ThresholdFilter with maximum."""
        filter_obj = ThresholdFilter("tm", max_value=65)

        df_filtered = filter_obj(self.df)

        # tm values: [65.0, 55.0, 70.0, 45.0, 60.0]
        # Values <= 65 (inclusive): 65.0, 55.0, 45.0, 60.0 = 4 values
        self.assertEqual(len(df_filtered), 4)
        self.assertTrue((df_filtered["tm"] <= 65).all())

    def test_threshold_filter_range(self):
        """Test ThresholdFilter with range."""
        filter_obj = ThresholdFilter("tm", min_value=55, max_value=65)

        df_filtered = filter_obj(self.df)

        # tm values: [65.0, 55.0, 70.0, 45.0, 60.0]
        # Values in [55, 65] (inclusive): 65.0, 55.0, 60.0 = 3 values
        self.assertEqual(len(df_filtered), 3)
        self.assertTrue((df_filtered["tm"] >= 55).all())
        self.assertTrue((df_filtered["tm"] <= 65).all())

    def test_threshold_filter_with_nan(self):
        """Test ThresholdFilter with NaN values."""
        df = self.df.copy()
        df.loc[0, "tm"] = np.nan

        # Default: exclude NaN
        filter_obj = ThresholdFilter("tm", min_value=60, keep_na=False)
        df_filtered = filter_obj(df)
        self.assertFalse(df_filtered["tm"].isna().any())

        # Keep NaN
        filter_obj = ThresholdFilter("tm", min_value=60, keep_na=True)
        df_filtered = filter_obj(df)
        self.assertTrue(df_filtered["tm"].isna().any())

    def test_sequence_length_filter_min(self):
        """Test SequenceLengthFilter with minimum."""
        filter_obj = SequenceLengthFilter(min_length=10)

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 4)  # Excludes short one
        self.assertTrue((df_filtered["sequence"].str.len() >= 10).all())

    def test_sequence_length_filter_max(self):
        """Test SequenceLengthFilter with maximum."""
        filter_obj = SequenceLengthFilter(max_length=15)

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 4)  # Excludes long one
        self.assertTrue((df_filtered["sequence"].str.len() <= 15).all())

    def test_sequence_length_filter_range(self):
        """Test SequenceLengthFilter with range."""
        filter_obj = SequenceLengthFilter(min_length=10, max_length=15)

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 3)

    def test_hamming_distance_filter(self):
        """Test HammingDistanceFilter."""
        reference = "MKTAYIAKQRQ"

        filter_obj = HammingDistanceFilter(reference, max_distance=5)

        df_filtered = filter_obj(self.df)

        # Should have hamming_distance column
        self.assertIn("hamming_distance", df_filtered.columns)
        self.assertTrue((df_filtered["hamming_distance"] <= 5).all())

    def test_hamming_distance_filter_normalized(self):
        """Test HammingDistanceFilter with normalization."""
        reference = "MKTAYIAKQRQ"

        filter_obj = HammingDistanceFilter(reference, max_distance=0.5, normalize=True)

        df_filtered = filter_obj(self.df)

        self.assertIn("hamming_distance_normalized", df_filtered.columns)
        self.assertTrue((df_filtered["hamming_distance_normalized"] <= 0.5).all())

    def test_conserved_residue_filter(self):
        """Test ConservedResidueFilter."""
        # All sequences start with 'MK'
        filter_obj = ConservedResidueFilter({0: ["M"], 1: ["K"]})

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 5)  # All pass

        # Stricter filter
        filter_obj = ConservedResidueFilter(
            {
                2: ["T"],  # Position 2 must be T
            }
        )

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 4)  # One has L at position 2

    def test_diversity_sampling_filter_random(self):
        """Test DiversitySamplingFilter with random method."""
        filter_obj = DiversitySamplingFilter(
            n_samples=3, method="random", random_seed=42
        )

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 3)

    def test_diversity_sampling_filter_top(self):
        """Test DiversitySamplingFilter with top method."""
        filter_obj = DiversitySamplingFilter(
            n_samples=3, method="top", score_column="tm"
        )

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 3)
        # Should be top 3 by tm
        expected_tms = sorted(self.df["tm"], reverse=True)[:3]
        actual_tms = sorted(df_filtered["tm"], reverse=True)
        self.assertEqual(actual_tms, expected_tms)

    def test_diversity_sampling_filter_spread(self):
        """Test DiversitySamplingFilter with spread method."""
        filter_obj = DiversitySamplingFilter(
            n_samples=3, method="spread", score_column="tm"
        )

        df_filtered = filter_obj(self.df)

        self.assertEqual(len(df_filtered), 3)

    def test_custom_filter(self):
        """Test CustomFilter."""

        def my_filter(df):
            return df[df["tm"] > 60]

        filter_obj = CustomFilter(my_filter, name="tm_above_60")

        df_filtered = filter_obj(self.df)

        self.assertTrue((df_filtered["tm"] > 60).all())
        self.assertIn("tm_above_60", str(filter_obj))

    def test_combine_filters(self):
        """Test combining filters."""
        filter1 = ThresholdFilter("tm", min_value=55)
        filter2 = SequenceLengthFilter(min_length=10)

        combined = combine_filters(filter1, filter2)

        df_filtered = combined(self.df)

        # Should satisfy both filters
        self.assertTrue((df_filtered["tm"] >= 55).all())
        self.assertTrue((df_filtered["sequence"].str.len() >= 10).all())

    def test_filter_repr(self):
        """Test filter __repr__ methods."""
        f1 = ThresholdFilter("tm", min_value=60)
        self.assertIn("tm", str(f1))
        self.assertIn("60", str(f1))

        f2 = SequenceLengthFilter(min_length=10)
        self.assertIn("10", str(f2))

        f3 = HammingDistanceFilter("MKTAY", max_distance=5)
        self.assertIn("5", str(f3))


class TestFilterEdgeCases(unittest.TestCase):
    """Test filter edge cases."""

    def test_empty_dataframe(self):
        """Test filters on empty DataFrame."""
        df = pd.DataFrame({"sequence": [], "tm": []})

        filter_obj = ThresholdFilter("tm", min_value=60)
        df_filtered = filter_obj(df)

        self.assertEqual(len(df_filtered), 0)

    def test_all_filtered_out(self):
        """Test when all rows are filtered out."""
        df = pd.DataFrame({"sequence": ["MKTAY", "MKLAY"], "tm": [30.0, 35.0]})

        filter_obj = ThresholdFilter("tm", min_value=60)
        df_filtered = filter_obj(df)

        self.assertEqual(len(df_filtered), 0)

    def test_missing_column(self):
        """Test filter on missing column."""
        df = pd.DataFrame({"sequence": ["MKTAY"]})

        filter_obj = ThresholdFilter("tm", min_value=60)

        with self.assertRaises(ValueError):
            filter_obj(df)

    def test_sampling_more_than_available(self):
        """Test sampling more items than available."""
        df = pd.DataFrame({"sequence": ["MKTAY", "MKLAY"], "tm": [60.0, 65.0]})

        filter_obj = DiversitySamplingFilter(n_samples=10, method="random")
        df_filtered = filter_obj(df)

        # Should return all available
        self.assertEqual(len(df_filtered), 2)


if __name__ == "__main__":
    unittest.main()
