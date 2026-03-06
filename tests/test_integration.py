"""
Integration tests for the BioLM Pipeline system.

These tests make real API calls and test end-to-end functionality.
Run with: make test-integration (requires BIOLMAI_TOKEN or BIOLM_API_KEY)
"""

import pytest

# Integration tests require the full pipeline extras (matplotlib, etc.) and a real
# API key.  Skip the entire module if either is missing.
pytest.importorskip(
    "matplotlib", reason="pip install biolmai[pipeline] for integration tests"
)

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HAS_PYTEST = True

from biolmai.pipeline import (  # noqa: E402
    DataPipeline,
    DataStore,
    Embed,
    Predict,
    SequenceLengthFilter,
    ThresholdFilter,
)
from biolmai.pipeline.data import EmbeddingSpec  # noqa: E402


def skip_if_no_api_key():
    """Skip test if API key not available."""
    if not (os.getenv("BIOLMAI_TOKEN") or os.getenv("BIOLM_API_KEY")):
        return unittest.skip("BIOLMAI_TOKEN or BIOLM_API_KEY not set")
    return lambda func: func


def integration_test(func):
    """Decorator for integration tests."""
    if HAS_PYTEST:
        func = pytest.mark.integration(func)
    return func


class TestDataPipelineIntegration(unittest.TestCase):
    """Integration tests for DataPipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.db_path = self.test_dir / "test_pipeline.db"
        self.test_sequences = [
            "MKTAYIAKQRQ",
            "MKLAVIDSAQ",
            "MKTAYIDSAQ",
        ]

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @integration_test
    @skip_if_no_api_key()
    def test_simple_prediction_pipeline(self):
        """Test basic prediction pipeline with API."""
        pipeline = DataPipeline(
            sequences=self.test_sequences, datastore=self.db_path, verbose=True
        )

        pipeline.add_prediction(
            "temberture-regression",
            extractions="prediction",
            columns="tm",
            stage_name="tm_pred",
        )

        pipeline.run()
        df = pipeline.get_final_data()

        self.assertEqual(len(df), len(self.test_sequences))
        self.assertIn("tm", df.columns)
        self.assertTrue(df["tm"].notna().all())
        df["tm"] = pd.to_numeric(df["tm"], errors="coerce")
        self.assertTrue(pd.api.types.is_numeric_dtype(df["tm"]))

        print(f"\n✅ Melting temperatures: {df['tm'].tolist()}")

    @integration_test
    @skip_if_no_api_key()
    def test_caching_behavior(self):
        """Test that predictions are cached and reused."""
        pipeline1 = DataPipeline(
            sequences=self.test_sequences, datastore=self.db_path, verbose=True
        )
        pipeline1.add_prediction(
            "temberture-regression",
            extractions="prediction",
            columns="tm",
            stage_name="tm_pred",
        )

        start1 = time.time()
        pipeline1.run()
        time1 = time.time() - start1

        pipeline2 = DataPipeline(
            sequences=self.test_sequences, datastore=self.db_path, verbose=True
        )
        pipeline2.add_prediction(
            "temberture-regression",
            extractions="prediction",
            columns="tm",
            stage_name="tm_pred",
        )

        start2 = time.time()
        pipeline2.run()
        time2 = time.time() - start2

        df1 = pipeline1.get_final_data()
        df2 = pipeline2.get_final_data()
        pd.testing.assert_frame_equal(
            df1[["sequence", "tm"]].sort_values("sequence").reset_index(drop=True),
            df2[["sequence", "tm"]].sort_values("sequence").reset_index(drop=True),
        )

        speedup = time1 / max(time2, 0.001)
        print(f"\n⏱️  First run: {time1:.2f}s")
        print(f"⏱️  Cached run: {time2:.2f}s")
        print(f"✅ Speedup: {speedup:.1f}x")

        self.assertGreater(speedup, 10.0, "Cache should provide significant speedup")

    @integration_test
    @skip_if_no_api_key()
    def test_multi_model_parallel_predictions(self):
        """Test running multiple models in parallel.

        Uses sequences ≥20 aa so soluprot validation passes.
        Verifies union merge: all sequences survive even though the two
        prediction stages are independent (not intersection).
        """
        # soluprot requires ≥20 aa — use long enough sequences
        long_sequences = [
            "MKTAYIAKQRQGHQAMAEIKQ",
            "ACDEFGHIKLMNPQRSTVWYAA",
        ]
        pipeline = DataPipeline(
            sequences=long_sequences,
            datastore=self.db_path,
            verbose=True,
        )

        pipeline.add_predictions([
            {"model_name": "temberture-regression", "extractions": "prediction", "columns": "tm"},
            {"model_name": "soluprot", "extractions": "soluble", "columns": "sol"},
        ])

        results = pipeline.run()
        df = pipeline.get_final_data()

        self.assertIn("predict_tm", results)
        self.assertIn("predict_sol", results)
        self.assertIn("tm", df.columns)
        self.assertIn("sol", df.columns)
        # union merge: all sequences survive (not just those with both predictions)
        self.assertEqual(len(df), len(long_sequences))

        print("\n✅ Both predictions completed in parallel")

    @integration_test
    @skip_if_no_api_key()
    def test_pipeline_with_filtering(self):
        """Test pipeline with filtering steps."""
        pipeline = DataPipeline(
            sequences=self.test_sequences, datastore=self.db_path, verbose=True
        )

        pipeline.add_prediction(
            "temberture-regression",
            extractions="prediction",
            columns="tm",
            stage_name="tm_pred",
        )
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=0),
            stage_name="filter_tm",
        )
        pipeline.add_prediction(
            "soluprot",
            extractions="soluble",
            columns="sol",
            stage_name="sol_pred",
        )

        results = pipeline.run()
        pipeline.get_final_data()

        self.assertIn("tm_pred", results)
        self.assertIn("filter_tm", results)
        self.assertIn("sol_pred", results)

        print("\n✅ Pipeline with filtering completed")

    @integration_test
    @skip_if_no_api_key()
    def test_rerun_with_different_filter(self):
        """Test re-running pipeline with different filter threshold."""
        pipeline1 = DataPipeline(
            sequences=self.test_sequences, datastore=self.db_path, verbose=True
        )
        pipeline1.add_prediction(
            "temberture-regression",
            extractions="prediction",
            columns="tm",
            stage_name="tm_pred",
        )
        pipeline1.add_filter(
            ThresholdFilter("tm", min_value=500),  # Impossible threshold
            stage_name="filter_1",
        )

        pipeline1.run()
        pipeline1.get_final_data()

        pipeline2 = DataPipeline(
            sequences=self.test_sequences,
            datastore=self.db_path,
            verbose=True,
            run_id="run2",
        )
        pipeline2.add_prediction(
            "temberture-regression",
            extractions="prediction",
            columns="tm",
            stage_name="tm_pred",
        )
        pipeline2.add_filter(
            ThresholdFilter("tm", min_value=0),  # Pass-all threshold
            stage_name="filter_1",
        )

        start = time.time()
        pipeline2.run()
        time2 = time.time() - start

        print(f"\n✅ Re-run completed in {time2:.2f}s (predictions cached)")
        self.assertLess(time2, 5.0, "Re-run with cached predictions should be fast")


class TestSingleStepPipelineIntegration(unittest.TestCase):
    """Integration tests for single-step shortcuts."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @integration_test
    @skip_if_no_api_key()
    def test_predict_shortcut(self):
        """Test Predict() shortcut function."""
        df = Predict(
            "temberture-regression",
            sequences=self.test_sequences,
            extractions="prediction",
            columns="tm",
            verbose=False,
        )

        self.assertEqual(len(df), len(self.test_sequences))
        self.assertIn("sequence", df.columns)
        self.assertIn("tm", df.columns)

        print("\n✅ Predict() shortcut works")

    @integration_test
    @skip_if_no_api_key()
    def test_embed_shortcut(self):
        """Test Embed() shortcut function."""
        df = Embed("esm2-650m", sequences=self.test_sequences[:1], verbose=False)

        self.assertEqual(len(df), 1)
        self.assertIn("sequence", df.columns)
        self.assertIn("embedding", df.columns)
        self.assertIsNotNone(df.iloc[0]["embedding"])

        print("\n✅ Embed() shortcut works")


class TestEmbeddingsAndPCAIntegration(unittest.TestCase):
    """Integration tests for embeddings and PCA."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.db_path = self.test_dir / "test_embeddings.db"
        self.test_sequences = [
            "MKTAYIAKQRQ",
            "MKLAVIDSAQ",
            "MKTAYIDSAQ",
            "MKLAVIYDSAQ",
            "MKTAY",
        ]

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @integration_test
    @skip_if_no_api_key()
    def test_embeddings_generation(self):
        """Test generating embeddings with ESM2-650M."""
        pipeline = DataPipeline(
            sequences=self.test_sequences, datastore=self.db_path, verbose=True
        )

        pipeline.add_prediction(
            "esm2-650m",
            action="encode",
            stage_name="embeddings",
            embedding_extractor=EmbeddingSpec(key="embeddings"),
        )

        results = pipeline.run()
        pipeline.get_final_data()

        embed_result = results["embeddings"]
        self.assertEqual(embed_result.output_count, len(self.test_sequences))

        with DataStore(self.db_path) as store:
            for seq in self.test_sequences:
                emb_list = store.get_embeddings_by_sequence(
                    seq, model_name="esm2-650m", load_data=True
                )
                self.assertGreater(len(emb_list), 0, f"No embeddings found for {seq}")

                embedding = emb_list[0].get("embedding")
                self.assertIsInstance(embedding, np.ndarray)
                self.assertGreater(len(embedding), 0)

                print(f"✅ Embedding for {seq[:10]}...: shape={embedding.shape}")

    @integration_test
    @skip_if_no_api_key()
    def test_pca_visualization(self):
        """Test PCA visualization of embeddings."""
        pipeline = DataPipeline(
            sequences=self.test_sequences, datastore=self.db_path, verbose=True
        )

        pipeline.add_prediction(
            "esm2-650m",
            action="encode",
            stage_name="embeddings",
            embedding_extractor=EmbeddingSpec(key="embeddings"),
        )
        pipeline.run()

        with DataStore(self.db_path) as store:
            df = store.export_to_dataframe()

            embeddings_list = []
            for seq in df["sequence"]:
                emb_list = store.get_embeddings_by_sequence(
                    seq, model_name="esm2-650m", load_data=True
                )
                if emb_list:
                    embeddings_list.append(emb_list[0].get("embedding"))
                else:
                    embeddings_list.append(None)

        self.assertEqual(len(embeddings_list), len(self.test_sequences))
        self.assertTrue(all(e is not None for e in embeddings_list))

        from biolmai.pipeline.visualization import plot_embedding_pca

        embeddings_array = np.stack(embeddings_list)
        fig = plot_embedding_pca(
            embeddings=embeddings_array, title="ESM2-650M Embeddings - PCA"
        )

        fig_path = self.test_dir / "pca_embeddings.png"
        fig.savefig(fig_path, dpi=100, bbox_inches="tight")

        self.assertTrue(fig_path.exists())
        file_size = fig_path.stat().st_size
        self.assertGreater(file_size, 1000, f"PCA plot too small: {file_size} bytes")

        print(f"\n✅ PCA plot created: {fig_path} ({file_size:,} bytes)")

        import matplotlib.pyplot as plt
        plt.close(fig)


class TestComplexMultiLevelPipeline(unittest.TestCase):
    """Integration test for complex multi-level pipeline with structure prediction."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.db_path = self.test_dir / "test_complex.db"
        self.test_sequences = [
            "MKTAY",
            "MKLAY",
            "MKSAY",
        ]

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @integration_test
    @skip_if_no_api_key()
    def test_complex_pipeline_with_structure(self):
        """
        Test complex multi-level pipeline:
        Predict (parallel) → Filter → Structure → Embed
        """
        pipeline = DataPipeline(
            sequences=self.test_sequences, datastore=self.db_path, verbose=True
        )

        print("\n" + "=" * 60)
        print("Complex Multi-Level Pipeline Test")
        print("=" * 60)

        # Level 1: Initial predictions (parallel)
        print("\n[Level 1] Adding initial predictions...")
        pipeline.add_prediction(
            "temberture-regression",
            extractions="prediction",
            columns="tm",
            stage_name="tm_pred",
        )
        pipeline.add_prediction(
            "soluprot",
            extractions="soluble",
            columns="sol",
            stage_name="sol_pred",
            depends_on=[],  # parallel with tm_pred
        )

        # Level 2: Filter based on Tm
        print("[Level 2] Adding filter...")
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=40),
            stage_name="tm_filter",
            depends_on=["tm_pred"],
        )

        # Level 3: Structure prediction
        print("[Level 3] Adding structure prediction...")
        pipeline.add_prediction(
            "esmfold",
            action="predict",
            extractions="mean_plddt",
            columns="plddt",
            stage_name="structure_pred",
            depends_on=["tm_filter"],
        )

        # Level 4: Embeddings on filtered+structured sequences
        print("[Level 4] Adding final embedding...")
        pipeline.add_prediction(
            "esm2-650m",
            action="encode",
            stage_name="final_embed",
            depends_on=["structure_pred"],
            embedding_extractor=EmbeddingSpec(key="embeddings"),
        )

        print("\nExecution Plan:")
        levels = pipeline._resolve_dependencies()
        for i, level in enumerate(levels):
            stage_names = [s.name for s in level]
            print(f"  Level {i+1}: {', '.join(stage_names)}")

        print("\nRunning pipeline...")
        results = pipeline.run()

        self.assertIn("tm_pred", results)
        self.assertIn("sol_pred", results)
        self.assertIn("tm_filter", results)
        self.assertIn("structure_pred", results)
        self.assertIn("final_embed", results)

        df = pipeline.get_final_data()

        print("\n✅ Complex pipeline completed!")
        print(f"   Input sequences: {len(self.test_sequences)}")
        print(f"   Final sequences: {len(df)}")
        print(f"   Total stages: {len(results)}")

        with DataStore(self.db_path) as store:
            for seq in df["sequence"]:
                structures = store.get_structures_by_sequence(seq, model_name="esmfold")
                if structures:
                    print(f"   ✅ Structure found for {seq}")


class TestEdgeCasesIntegration(unittest.TestCase):
    """Integration tests for edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.db_path = self.test_dir / "test_pipeline.db"

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @integration_test
    @skip_if_no_api_key()
    def test_empty_filter_result(self):
        """Test handling of filters that produce empty results."""
        pipeline = DataPipeline(
            sequences=["MKTAY", "MKLAVIY"], datastore=self.db_path, verbose=True
        )

        pipeline.add_prediction(
            "temberture-regression", extractions="prediction", columns="tm"
        )
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=500),  # Impossible threshold
            stage_name="filter_1",
        )

        pipeline.run()
        df = pipeline.get_final_data()

        print(f"\n✅ Filter produced {len(df)} sequences (threshold 500)")

    @integration_test
    @skip_if_no_api_key()
    def test_duplicate_sequences(self):
        """Test automatic deduplication."""
        sequences = ["MKTAY", "MKLAVIY", "MKTAY"]  # Duplicate

        pipeline = DataPipeline(
            sequences=sequences, datastore=self.db_path, verbose=True
        )

        pipeline.add_prediction(
            "temberture-regression", extractions="prediction", columns="tm"
        )
        pipeline.run()
        df = pipeline.get_final_data()

        unique_seqs = df["sequence"].nunique()
        self.assertEqual(unique_seqs, 2)

        print(
            f"\n✅ Deduplicated: {len(sequences)} input → {unique_seqs} unique sequences"
        )

    @integration_test
    @skip_if_no_api_key()
    def test_sequence_length_variation(self):
        """Test handling sequences of various lengths."""
        sequences = [
            "MKTAY",                            # 5 AA
            "MKTAYIAKQRQ",                      # 11 AA
            "MKTAYIAKQRQGHQAMAEIKQGHQAMAEIKQ",  # 32 AA
        ]

        pipeline = DataPipeline(
            sequences=sequences, datastore=self.db_path, verbose=True
        )

        pipeline.add_filter(SequenceLengthFilter(min_length=10, max_length=30))
        pipeline.add_prediction(
            "temberture-regression", extractions="prediction", columns="tm"
        )

        results = pipeline.run()
        pipeline.get_final_data()

        filter_result = results["filter_0"]
        self.assertEqual(
            filter_result.output_count, 1, "Only 1 sequence should pass filter"
        )

        print(
            f"\n✅ Length filter worked: {filter_result.output_count} sequence passed"
        )


class TestDataStoreIntegration(unittest.TestCase):
    """Integration tests for DataStore operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.db_path = self.test_dir / "test_pipeline.db"
        self.test_sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ", "MKTAYIDSAQ"]

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @integration_test
    @skip_if_no_api_key()
    def test_datastore_export_options(self):
        """Test various export options from DataStore."""
        pipeline = DataPipeline(
            sequences=self.test_sequences, datastore=self.db_path, verbose=True
        )

        pipeline.add_prediction(
            "temberture-regression", extractions="prediction", columns="tm"
        )
        pipeline.add_prediction(
            "soluprot", extractions="soluble", columns="sol"
        )

        pipeline.run()

        with DataStore(self.db_path) as store:
            df1 = store.export_to_dataframe()
            self.assertEqual(len(df1), 3)

            df2 = store.export_to_dataframe(include_predictions=True)
            self.assertIn("tm", df2.columns)
            self.assertIn("sol", df2.columns)

            print(
                f"\n✅ DataStore export works: {len(df2)} sequences, {len(df2.columns)} columns"
            )

    @integration_test
    @skip_if_no_api_key()
    def test_export_to_csv(self):
        """Test exporting pipeline results to CSV."""
        pipeline = DataPipeline(
            sequences=self.test_sequences, datastore=self.db_path, verbose=True
        )

        pipeline.add_predictions([
            {"model_name": "temberture-regression", "extractions": "prediction", "columns": "tm"},
            {"model_name": "soluprot", "extractions": "soluble", "columns": "sol"},
        ])
        pipeline.run()

        csv_path = self.test_dir / "results.csv"
        with DataStore(self.db_path) as store:
            store.export_to_csv(str(csv_path), include_predictions=True)

        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 3)
        self.assertIn("sequence", df.columns)

        print(f"\n✅ CSV export successful: {csv_path}")


class TestPredictionCorrelation(unittest.TestCase):
    """Verify that pipeline batch predictions are correctly aligned to their input sequences."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.db_path = self.test_dir / "test_correlator.db"
        self.test_sequences = [
            "MKTAYIAKQRQGHQAMAEIKQ",
            "ACDEFGHIKLMNPQRSTVWY",
            "MKLAVIDSAQGHILMNPQRSTVWY",
            "AAAAAAAAAAAAAAAAAAAAAA",
            "WRWWRWWRWWRWWRWWRWWRW",
        ]

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @integration_test
    @skip_if_no_api_key()
    def test_pipeline_predictions_match_individual_api_calls(self):
        """Each pipeline prediction value must match a direct one-sequence API call."""
        import asyncio

        from biolmai.client import BioLMApiClient

        pipeline = DataPipeline(
            sequences=self.test_sequences,
            datastore=self.db_path,
            verbose=False,
        )
        pipeline.add_prediction(
            "temberture-regression",
            extractions="prediction",
            columns="tm",
            stage_name="predict_tm",
        )
        pipeline.run()
        df_pipeline = pipeline.get_final_data()

        self.assertEqual(len(df_pipeline), len(self.test_sequences))
        self.assertIn("tm", df_pipeline.columns)
        self.assertIn("sequence", df_pipeline.columns)

        pipeline_preds: dict[str, float] = {}
        for _, row in df_pipeline.iterrows():
            seq = row["sequence"]
            val = row["tm"]
            self.assertIsNotNone(val, f"Pipeline prediction for '{seq}' is None")
            self.assertFalse(pd.isna(val), f"Pipeline prediction for '{seq}' is NaN")
            pipeline_preds[seq] = float(val)

        print(f"\nPipeline predictions: {pipeline_preds}")

        async def predict_one(seq: str) -> float:
            api = BioLMApiClient("temberture-regression")
            try:
                results = await api.predict(items=[{"sequence": seq}])
                result = results[0]
                if isinstance(result, dict):
                    return float(result["prediction"])
                return float(result)
            finally:
                await api.shutdown()

        individual_preds: dict[str, float] = {}
        for seq in self.test_sequences:
            val = asyncio.run(predict_one(seq))
            individual_preds[seq] = val
            print(f"  Individual API: {seq[:15]}... → {val:.4f}")

        for seq in self.test_sequences:
            pipeline_val = pipeline_preds[seq]
            individual_val = individual_preds[seq]
            self.assertAlmostEqual(
                pipeline_val,
                individual_val,
                places=4,
                msg=(
                    f"Prediction mismatch for sequence '{seq}': "
                    f"pipeline={pipeline_val}, individual={individual_val}."
                ),
            )
            print(f"  ✓ {seq[:15]}... pipeline={pipeline_val:.4f} == individual={individual_val:.4f}")

        print(f"\n✅ All {len(self.test_sequences)} predictions correctly aligned")

    @integration_test
    @skip_if_no_api_key()
    def test_cached_predictions_same_alignment(self):
        """Re-running the pipeline with cached predictions preserves alignment."""
        pipeline1 = DataPipeline(
            sequences=self.test_sequences,
            datastore=self.db_path,
            verbose=False,
        )
        pipeline1.add_prediction(
            "temberture-regression",
            extractions="prediction",
            columns="tm",
            stage_name="predict_tm",
        )
        pipeline1.run()
        df_first = pipeline1.get_final_data()
        first_preds = dict(zip(df_first["sequence"], df_first["tm"].astype(float)))

        pipeline2 = DataPipeline(
            sequences=self.test_sequences,
            datastore=self.db_path,
            verbose=False,
            run_id="run2",
        )
        pipeline2.add_prediction(
            "temberture-regression",
            extractions="prediction",
            columns="tm",
            stage_name="predict_tm",
        )
        result2 = pipeline2.run()
        df_second = pipeline2.get_final_data()
        second_preds = dict(zip(df_second["sequence"], df_second["tm"].astype(float)))

        stage_result = result2["predict_tm"]
        self.assertEqual(
            stage_result.cached_count,
            len(self.test_sequences),
            "Second run should be 100% cached",
        )

        for seq in self.test_sequences:
            self.assertAlmostEqual(
                first_preds[seq],
                second_preds[seq],
                places=4,
                msg=f"Cached prediction for '{seq}' differs from original",
            )
            print(f"  ✓ {seq[:15]}... first={first_preds[seq]:.4f} == cached={second_preds[seq]:.4f}")

        print(f"\n✅ Cached alignment verified for {len(self.test_sequences)} sequences")


if __name__ == "__main__":
    unittest.main(verbosity=2)
