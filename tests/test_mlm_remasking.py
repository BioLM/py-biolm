"""
Unit tests for MLM remasking functionality.
"""

import asyncio
import unittest

import numpy as np

from biolmai.pipeline.mlm_remasking import (
    AGGRESSIVE_CONFIG,
    CONSERVATIVE_CONFIG,
    MODERATE_CONFIG,
    MLMRemasker,
    RemaskingConfig,
    create_remasker_from_dict,
)


class TestRemaskingConfig(unittest.TestCase):
    """Test RemaskingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = RemaskingConfig()

        self.assertEqual(config.mask_fraction, 0.15)
        self.assertEqual(config.num_iterations, 1)
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.mask_token, "<mask>")

    def test_custom_config(self):
        """Test custom configuration."""
        config = RemaskingConfig(
            mask_fraction=0.2,
            num_iterations=5,
            temperature=1.5,
            conserved_positions=[0, 10, 20],
        )

        self.assertEqual(config.mask_fraction, 0.2)
        self.assertEqual(config.num_iterations, 5)
        self.assertEqual(config.conserved_positions, [0, 10, 20])


class TestMLMRemasker(unittest.TestCase):
    """Test MLMRemasker functionality."""

    def setUp(self):
        """Create test remasker."""
        self.config = RemaskingConfig(mask_fraction=0.15, num_iterations=1)
        self.remasker = MLMRemasker(self.config)
        self.test_sequence = "MKTAYIAKQRQGHQAMAEIKQ"

    def test_remasker_creation(self):
        """Test creating remasker."""
        self.assertIsNotNone(self.remasker)
        self.assertEqual(self.remasker.config.mask_fraction, 0.15)

    def test_select_mask_positions_random(self):
        """Test random mask position selection."""
        positions = self.remasker.select_mask_positions(self.test_sequence)

        # Should return positions
        self.assertIsInstance(positions, list)
        self.assertGreater(len(positions), 0)

        # All positions should be valid
        self.assertTrue(all(0 <= p < len(self.test_sequence) for p in positions))

        # Should respect mask_fraction
        expected_count = int(len(self.test_sequence) * self.config.mask_fraction)
        self.assertEqual(len(positions), max(1, expected_count))

    def test_select_mask_positions_with_conserved(self):
        """Test mask position selection with conserved positions."""
        config = RemaskingConfig(mask_fraction=0.5, conserved_positions=[0, 1, 2])
        remasker = MLMRemasker(config)

        positions = remasker.select_mask_positions(self.test_sequence)

        # Should not include conserved positions
        self.assertNotIn(0, positions)
        self.assertNotIn(1, positions)
        self.assertNotIn(2, positions)

    def test_select_mask_positions_explicit(self):
        """Test explicit mask positions."""
        config = RemaskingConfig(mask_positions=[5, 10, 15])
        remasker = MLMRemasker(config)

        positions = remasker.select_mask_positions(self.test_sequence)

        self.assertEqual(set(positions), {5, 10, 15})

    def test_select_mask_positions_blocks(self):
        """Test block masking strategy."""
        config = RemaskingConfig(
            mask_fraction=0.3, mask_strategy="blocks", block_size=3
        )
        remasker = MLMRemasker(config)

        positions = remasker.select_mask_positions(self.test_sequence)

        self.assertGreater(len(positions), 0)

    def test_create_masked_sequence(self):
        """Test creating masked sequence."""
        positions = [5, 10, 15]

        masked = self.remasker.create_masked_sequence(self.test_sequence, positions)

        # Should contain mask tokens
        self.assertIn("<mask>", masked)

        # Should have 3 mask tokens
        self.assertEqual(masked.count("<mask>"), 3)

        # First 5 characters should be unchanged
        self.assertEqual(masked[:5], self.test_sequence[:5])

    def test_predict_masked_positions_mock(self):
        """Test mock prediction (no API)."""
        masked_seq = "MKTAY<mask>AKQRQ"
        positions = [5]

        predicted, confidences = asyncio.run(
            self.remasker.predict_masked_positions(masked_seq, positions)
        )

        # Should return a sequence
        self.assertIsInstance(predicted, str)

        # Should not contain mask token
        self.assertNotIn("<mask>", predicted)

        # Should have confidences
        self.assertIsInstance(confidences, dict)
        self.assertEqual(len(confidences), 1)

    def test_generate_variant(self):
        """Test generating single variant."""
        variant, metadata = asyncio.run(
            self.remasker.generate_variant(self.test_sequence)
        )

        # Should return a sequence
        self.assertIsInstance(variant, str)

        # Variant should be similar length to original (may differ slightly due to masking)
        # but should not contain mask tokens
        self.assertNotIn("<mask>", variant)

        # Length should match original after prediction
        self.assertEqual(len(variant), len(self.test_sequence))

        # Should have metadata
        self.assertIn("parent_sequence", metadata)
        self.assertIn("num_mutations", metadata)
        self.assertIn("mutation_rate", metadata)
        self.assertEqual(metadata["parent_sequence"], self.test_sequence)

    def test_generate_variants_multiple(self):
        """Test generating multiple variants."""
        variants = asyncio.run(
            self.remasker.generate_variants(self.test_sequence, num_variants=10)
        )

        self.assertEqual(len(variants), 10)

        # All should be tuples of (sequence, metadata)
        for variant, metadata in variants:
            self.assertIsInstance(variant, str)
            self.assertIsInstance(metadata, dict)
            # Variant should not contain mask tokens
            self.assertNotIn("<mask>", variant)
            # Length should match original
            self.assertEqual(len(variant), len(self.test_sequence))

    def test_generate_variants_deduplication(self):
        """Test variant deduplication."""
        variants = asyncio.run(
            self.remasker.generate_variants(
                self.test_sequence, num_variants=50, deduplicate=True
            )
        )

        # Should have unique sequences
        sequences = [v[0] for v in variants]
        self.assertEqual(len(sequences), len(set(sequences)))

    def test_generate_variants_no_deduplication(self):
        """Test without deduplication."""
        variants = asyncio.run(
            self.remasker.generate_variants(
                self.test_sequence, num_variants=10, deduplicate=False
            )
        )

        # May have duplicates (with random generation, unlikely but possible)
        self.assertEqual(len(variants), 10)

    def test_iterative_refinement(self):
        """Test iterative refinement with fitness function."""

        # Simple fitness function (favor sequences with more A's)
        def fitness_func(seq):
            return seq.count("A") / len(seq)

        config = RemaskingConfig(mask_fraction=0.1, num_iterations=2)
        remasker = MLMRemasker(config)

        final_population = asyncio.run(
            remasker.iterative_refinement(
                self.test_sequence,
                fitness_func,
                num_iterations=3,
                population_size=10,
                keep_top_k=3,
            )
        )

        # Should return population
        self.assertGreater(len(final_population), 0)
        self.assertLessEqual(len(final_population), 3)

        # Each should be (sequence, fitness, metadata)
        for seq, fitness, metadata in final_population:
            self.assertIsInstance(seq, str)
            self.assertIsInstance(fitness, float)
            self.assertIsInstance(metadata, dict)


class TestRemaskingHelpers(unittest.TestCase):
    """Test helper functions."""

    def test_create_from_dict(self):
        """Test creating remasker from dict."""
        config_dict = {"mask_fraction": 0.2, "num_iterations": 5, "temperature": 1.5}

        remasker = create_remasker_from_dict(config_dict, model_name="esm2")

        self.assertEqual(remasker.config.mask_fraction, 0.2)
        self.assertEqual(remasker.config.num_iterations, 5)
        self.assertEqual(remasker.model_name, "esm2")

    def test_predefined_configs(self):
        """Test predefined configurations."""
        # Conservative
        self.assertEqual(CONSERVATIVE_CONFIG.mask_fraction, 0.10)
        self.assertEqual(CONSERVATIVE_CONFIG.temperature, 0.5)

        # Moderate
        self.assertEqual(MODERATE_CONFIG.mask_fraction, 0.15)
        self.assertEqual(MODERATE_CONFIG.temperature, 1.0)

        # Aggressive
        self.assertEqual(AGGRESSIVE_CONFIG.mask_fraction, 0.25)
        self.assertEqual(AGGRESSIVE_CONFIG.temperature, 1.5)


class TestRemaskingStrategies(unittest.TestCase):
    """Test different masking strategies."""

    def setUp(self):
        """Setup test sequence."""
        self.sequence = "MKTAYIAKQRQGHQAMAEIKQ"

    def test_random_strategy(self):
        """Test random masking strategy."""
        config = RemaskingConfig(mask_strategy="random", mask_fraction=0.2)
        remasker = MLMRemasker(config)

        positions = remasker.select_mask_positions(self.sequence)

        self.assertGreater(len(positions), 0)

    def test_block_strategy(self):
        """Test block masking strategy."""
        config = RemaskingConfig(
            mask_strategy="blocks", mask_fraction=0.3, block_size=3
        )
        remasker = MLMRemasker(config)

        positions = remasker.select_mask_positions(self.sequence)

        self.assertGreater(len(positions), 0)

    def test_low_confidence_strategy_fallback(self):
        """Test low confidence strategy without confidences (fallback)."""
        config = RemaskingConfig(mask_strategy="low_confidence", mask_fraction=0.2)
        remasker = MLMRemasker(config)

        # Without confidences, should fallback to random
        positions = remasker.select_mask_positions(self.sequence)

        self.assertGreater(len(positions), 0)

    def test_low_confidence_strategy_with_confidences(self):
        """Test low confidence strategy with confidence scores."""
        config = RemaskingConfig(mask_strategy="low_confidence", mask_fraction=0.2)
        remasker = MLMRemasker(config)

        # Mock confidences (low at certain positions)
        confidences = np.ones(len(self.sequence))
        confidences[[5, 10, 15]] = 0.3  # Low confidence at these positions

        positions = remasker.select_mask_positions(self.sequence, confidences)

        # Should prefer low confidence positions
        self.assertGreater(len(positions), 0)
        # At least some of the selected positions should be low-confidence ones
        low_conf_selected = sum(1 for p in positions if p in [5, 10, 15])
        self.assertGreater(low_conf_selected, 0)


if __name__ == "__main__":
    unittest.main()
