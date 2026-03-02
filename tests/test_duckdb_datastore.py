"""
Test DuckDB datastore performance and correctness.
"""

import time

import numpy as np
import pytest

from biolmai.pipeline.datastore_duckdb import DuckDBDataStore


@pytest.fixture
def duckdb_store(tmp_path):
    """Create temporary DuckDB datastore."""
    return DuckDBDataStore(db_path=tmp_path / "test.duckdb", data_dir=tmp_path / "data")


class TestDuckDBBasics:
    """Test basic DuckDB operations."""

    def test_add_single_sequence(self, duckdb_store):
        """Test adding single sequence."""
        seq_id = duckdb_store.add_sequence("MKLLIV")
        assert seq_id > 0
        print(f"  ✓ Added sequence with ID {seq_id}")

    def test_add_batch_sequences(self, duckdb_store):
        """Test batch sequence addition."""
        sequences = [f"SEQ{i}" * 10 for i in range(100)]

        start = time.time()
        seq_ids = duckdb_store.add_sequences_batch(sequences)
        elapsed = time.time() - start

        assert len(seq_ids) == 100
        assert len(set(seq_ids)) == 100  # All unique
        print(f"  ✓ Added 100 sequences in {elapsed*1000:.1f}ms")

    def test_deduplication_anti_join(self, duckdb_store):
        """Test anti-join deduplication."""
        # Add initial batch
        sequences = ["SEQ1", "SEQ2", "SEQ3"]
        ids1 = duckdb_store.add_sequences_batch(sequences)

        # Add overlapping batch (should dedupe)
        sequences2 = ["SEQ2", "SEQ3", "SEQ4", "SEQ5"]
        ids2 = duckdb_store.add_sequences_batch(sequences2)

        # Check total count (should be 5 unique)
        result = duckdb_store.query("SELECT COUNT(*) as cnt FROM sequences")
        assert result["cnt"].iloc[0] == 5

        # SEQ2 and SEQ3 should have same IDs
        assert ids1[1] == ids2[0]  # SEQ2
        assert ids1[2] == ids2[1]  # SEQ3

        print("  ✓ Anti-join deduplication works correctly")

    def test_batch_predictions(self, duckdb_store):
        """Test batch prediction insertion."""
        # Add sequences
        seq_ids = duckdb_store.add_sequences_batch(["AAA", "CCC", "GGG"])

        # Batch add predictions
        predictions = [
            {
                "sequence_id": seq_ids[0],
                "prediction_type": "tm",
                "model_name": "test",
                "value": 50.0,
            },
            {
                "sequence_id": seq_ids[1],
                "prediction_type": "tm",
                "model_name": "test",
                "value": 60.0,
            },
            {
                "sequence_id": seq_ids[2],
                "prediction_type": "tm",
                "model_name": "test",
                "value": 70.0,
            },
        ]

        start = time.time()
        duckdb_store.add_predictions_batch(predictions)
        elapsed = time.time() - start

        # Verify
        result = duckdb_store.query("SELECT COUNT(*) as cnt FROM predictions")
        assert result["cnt"].iloc[0] == 3

        print(f"  ✓ Added 3 predictions in {elapsed*1000:.1f}ms")

    def test_query_power(self, duckdb_store):
        """Test SQL query power (no memory explosion)."""
        # Add test data
        sequences = [f"SEQUENCE_{i}" * 5 for i in range(200)]
        seq_ids = duckdb_store.add_sequences_batch(sequences)

        # Add predictions
        predictions = [
            {
                "sequence_id": sid,
                "prediction_type": "score",
                "model_name": "test",
                "value": float(i),
            }
            for i, sid in enumerate(seq_ids)
        ]
        duckdb_store.add_predictions_batch(predictions)

        # Query with filter (columnar scan - super fast!)
        result = duckdb_store.query(
            """
            SELECT s.sequence, p.value
            FROM sequences s
            JOIN predictions p ON s.sequence_id = p.sequence_id
            WHERE s.length > 100
            AND p.value > 150
            ORDER BY p.value DESC
            LIMIT 10
        """
        )

        assert len(result) <= 10
        assert all(result["value"] > 150)

        print(f"  ✓ Complex query returned {len(result)} results efficiently")


class TestDuckDBPerformance:
    """Test DuckDB performance vs expectations."""

    def test_large_batch_performance(self, duckdb_store):
        """Test performance with large batch (1000 sequences)."""
        sequences = [f"PROT_{i:04d}" + "A" * 100 for i in range(1000)]

        start = time.time()
        seq_ids = duckdb_store.add_sequences_batch(sequences)
        elapsed = time.time() - start

        assert len(seq_ids) == 1000
        print(
            f"  ✓ Added 1000 sequences in {elapsed*1000:.0f}ms ({elapsed/1000*1000000:.0f}µs each)"
        )

        # Should be fast (< 1 second for 1000 sequences)
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s"

    def test_diff_mode_performance(self, duckdb_store):
        """Test diff mode anti-join performance."""
        # Initial batch: 500 sequences
        batch1 = [f"SEQ_{i:04d}" + "M" * 50 for i in range(500)]
        duckdb_store.add_sequences_batch(batch1)

        # Diff batch: 250 overlap + 250 new
        batch2 = [f"SEQ_{i:04d}" + "M" * 50 for i in range(250, 750)]

        start = time.time()
        duckdb_store.add_sequences_batch(batch2, deduplicate=True)
        elapsed = time.time() - start

        # Check only 250 new were added
        result = duckdb_store.query("SELECT COUNT(*) as cnt FROM sequences")
        assert result["cnt"].iloc[0] == 750

        print(f"  ✓ Diff mode: 250 overlapping + 250 new in {elapsed*1000:.0f}ms")
        print("    Anti-join deduplication is vectorized!")

    def test_embedding_storage(self, duckdb_store):
        """Test embedding storage in Parquet."""
        seq_id = duckdb_store.add_sequence("TESTSEQ")

        # Create fake embedding
        embedding = np.random.randn(1280).astype(np.float32)

        start = time.time()
        duckdb_store.add_embedding(seq_id, "esm2", embedding)
        elapsed = time.time() - start

        # Load it back
        embs = duckdb_store.get_embeddings_by_sequence("TESTSEQ", load_data=True)

        assert len(embs) == 1
        assert np.allclose(embs[0]["embedding"], embedding)

        print(f"  ✓ Stored 1280-dim embedding in Parquet ({elapsed*1000:.1f}ms)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
