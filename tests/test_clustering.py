"""
Tests for sequence clustering and diversity analysis.
"""
import pytest
pytest.importorskip("duckdb")

import unittest

import numpy as np

from biolmai.pipeline.clustering import (
    ClusteringResult,
    DiversityAnalyzer,
    SequenceClusterer,
    analyze_diversity,
    cluster_sequences,
)


class TestSequenceClusterer(unittest.TestCase):
    """Test sequence clustering functionality."""

    def setUp(self):
        """Set up test sequences and mock embeddings."""
        self.sequences = [
            "MKTAYIAKQRQ",
            "MKTAYIAKQRR",  # 1 mutation from first
            "MKTAYIAKQRK",  # 1 mutation from first
            "MKLAVIDSAQR",  # More different
            "MKLAVIDSAQK",  # Similar to previous
            "GGGGGGGGGG",  # Very different
            "GGGGGGGGGA",  # Similar to previous
        ]
        self.embeddings = np.random.randn(len(self.sequences), 64)

    def test_kmeans_clustering(self):
        """Test K-means clustering with embeddings."""
        clusterer = SequenceClusterer(method="kmeans", n_clusters=3, similarity_metric="embedding")
        result = clusterer.cluster(self.sequences, embeddings=self.embeddings)

        self.assertIsInstance(result, ClusteringResult)
        self.assertEqual(len(result.cluster_ids), len(self.sequences))
        self.assertEqual(result.n_clusters, 3)
        self.assertEqual(len(result.centroids), 3)

    def test_dbscan_clustering(self):
        """Test DBSCAN clustering with embeddings."""
        clusterer = SequenceClusterer(method="dbscan", eps=5.0, min_samples=2, similarity_metric="embedding")
        result = clusterer.cluster(self.sequences, embeddings=self.embeddings)

        self.assertIsInstance(result, ClusteringResult)
        self.assertEqual(len(result.cluster_ids), len(self.sequences))
        self.assertTrue(result.n_clusters >= 1)

    def test_hierarchical_clustering(self):
        """Test hierarchical clustering with embeddings."""
        clusterer = SequenceClusterer(method="hierarchical", n_clusters=2, similarity_metric="embedding")
        result = clusterer.cluster(self.sequences, embeddings=self.embeddings)

        self.assertIsInstance(result, ClusteringResult)
        self.assertEqual(len(result.cluster_ids), len(self.sequences))
        self.assertEqual(result.n_clusters, 2)

    def test_centroids(self):
        """Test that centroids are actual sequences."""
        embeddings = np.random.randn(len(self.sequences), 64)
        result = cluster_sequences(self.sequences, method="kmeans", n_clusters=3, embeddings=embeddings)

        for centroid in result.centroids:
            self.assertIn(centroid, self.sequences)

    def test_cluster_sizes(self):
        """Test cluster size tracking."""
        embeddings = np.random.randn(len(self.sequences), 64)
        result = cluster_sequences(self.sequences, method="kmeans", n_clusters=2, embeddings=embeddings)

        self.assertIsNotNone(result.cluster_sizes)
        self.assertEqual(sum(result.cluster_sizes.values()), len(self.sequences))

    def test_embedding_clustering(self):
        """Test clustering with embeddings."""
        # Create mock embeddings
        embeddings = np.random.randn(len(self.sequences), 128)

        clusterer = SequenceClusterer(
            method="kmeans", n_clusters=2, similarity_metric="embedding"
        )
        result = clusterer.cluster(self.sequences, embeddings=embeddings)

        self.assertEqual(len(result.cluster_ids), len(self.sequences))


class TestDiversityAnalyzer(unittest.TestCase):
    """Test diversity analysis functionality."""

    def setUp(self):
        """Set up test sequences."""
        self.sequences = [
            "MKTAYIAKQRQ",
            "MKTAYIAKQRR",
            "MKLAVIDSAQR",
        ]

    def test_shannon_entropy(self):
        """Test Shannon entropy calculation."""
        entropy = DiversityAnalyzer.shannon_entropy(self.sequences)

        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, 1.0)  # Normalized

    def test_entropy_identical_sequences(self):
        """Test entropy for identical sequences."""
        identical = ["MKTAY", "MKTAY", "MKTAY"]
        entropy = DiversityAnalyzer.shannon_entropy(identical)

        # Identical sequences should have zero entropy
        self.assertAlmostEqual(entropy, 0.0)

    def test_pairwise_distances(self):
        """Placeholder: pairwise_distance_stats() removed (seq-to-seq analysis).
        TODO: implement using embedding distances and re-enable this test.
        """
        self.assertFalse(hasattr(DiversityAnalyzer, "pairwise_distance_stats"))

    def test_motif_diversity(self):
        """Test k-mer diversity analysis."""
        diversity = DiversityAnalyzer.motif_diversity(self.sequences, k=3)

        self.assertIn("unique_kmers", diversity)
        self.assertIn("total_kmers", diversity)
        self.assertIn("diversity_ratio", diversity)

        self.assertGreaterEqual(diversity["unique_kmers"], 0)
        self.assertGreaterEqual(diversity["total_kmers"], 0)
        self.assertGreaterEqual(diversity["diversity_ratio"], 0.0)
        self.assertLessEqual(diversity["diversity_ratio"], 1.0)

    def test_sequence_identity_matrix(self):
        """Placeholder: sequence_identity_matrix() removed (seq-to-seq analysis).
        TODO: implement using embedding cosine similarity and re-enable this test.
        """
        self.assertFalse(hasattr(DiversityAnalyzer, "sequence_identity_matrix"))

    def test_compute_all_metrics(self):
        """Test computing all metrics at once."""
        metrics = DiversityAnalyzer.compute_all_metrics(self.sequences)

        self.assertIn("n_sequences", metrics)
        self.assertIn("n_unique", metrics)
        self.assertIn("uniqueness_ratio", metrics)
        self.assertIn("shannon_entropy", metrics)
        self.assertNotIn("pairwise_distances", metrics)  # removed (seq-to-seq)
        self.assertIn("motif_diversity_3mer", metrics)
        self.assertIn("motif_diversity_5mer", metrics)

        self.assertEqual(metrics["n_sequences"], len(self.sequences))


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def setUp(self):
        """Set up test sequences."""
        self.sequences = [
            "MKTAYIAKQRQ",
            "MKTAYIAKQRR",
            "MKLAVIDSAQR",
            "MKLAVIDSAQK",
        ]

    def test_cluster_sequences(self):
        """Test cluster_sequences convenience function with embeddings."""
        embeddings = np.random.randn(len(self.sequences), 64)
        result = cluster_sequences(self.sequences, method="kmeans", n_clusters=2, embeddings=embeddings)

        self.assertIsInstance(result, ClusteringResult)
        self.assertEqual(len(result.cluster_ids), len(self.sequences))

    def test_analyze_diversity(self):
        """Test analyze_diversity convenience function."""
        metrics = analyze_diversity(self.sequences)

        self.assertIsInstance(metrics, dict)
        self.assertIn("shannon_entropy", metrics)
        self.assertNotIn("pairwise_distances", metrics)  # removed (seq-to-seq)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_sequences(self):
        """Test with empty sequence list."""
        metrics = analyze_diversity([])

        self.assertEqual(metrics["n_sequences"], 0)
        self.assertEqual(metrics["n_unique"], 0)

    def test_single_sequence(self):
        """Test with single sequence."""
        sequences = ["MKTAY"]
        metrics = analyze_diversity(sequences)

        self.assertEqual(metrics["n_sequences"], 1)
        self.assertEqual(metrics["n_unique"], 1)

    def test_different_length_sequences(self):
        """Test sequences of different lengths."""
        sequences = ["MKTAY", "MKLAVIDSAQR", "GG"]
        metrics = analyze_diversity(sequences)

        # Should handle different lengths by padding
        self.assertEqual(metrics["n_sequences"], 3)


if __name__ == "__main__":
    unittest.main()
