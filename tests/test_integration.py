"""
Integration tests for the BioLM Pipeline system.

These tests make real API calls and test end-to-end functionality.
Run with: make test-integration (requires BIOLMAI_TOKEN or BIOLM_API_KEY)
"""

from biolmai.core.http import BioLMApiClient
import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import os
import time

# Mark for pytest
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from biolmai.pipeline import (
    DataPipeline,
    GenerativePipeline,
    GenerationConfig,
    SingleStepPipeline,
    Predict,
    Embed,
    ThresholdFilter,
    RankingFilter,
    SequenceLengthFilter,
    DataStore,
)
from biolmai.pipeline.visualization import PipelinePlotter


def skip_if_no_api_key():
    """Skip test if API key not available."""
    if not (os.getenv('BIOLMAI_TOKEN') or os.getenv('BIOLM_API_KEY')):
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
        self.db_path = self.test_dir / 'test_pipeline.db'
        self.test_sequences = [
            'MKTAYIAKQRQ',
            'MKLAVIDSAQ',
            'MKTAYIDSAQ',
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
            sequences=self.test_sequences,
            datastore=self.db_path,
            verbose=True
        )
        
        # Add esm2stabp prediction - returns melting_temperature
        pipeline.add_prediction('esm2stabp', stage_name='tm_pred')
        
        results = pipeline.run()
        df = pipeline.get_final_data()
        
        # Verify predictions were made
        self.assertEqual(len(df), len(self.test_sequences))
        # esm2stabp returns 'melting_temperature' field
        self.assertIn('melting_temperature', df.columns)
        
        # Check that values exist and are numeric
        self.assertTrue(df['melting_temperature'].notna().all())
        # Convert to numeric if needed
        df['melting_temperature'] = pd.to_numeric(df['melting_temperature'], errors='coerce')
        self.assertTrue(pd.api.types.is_numeric_dtype(df['melting_temperature']))
        
        print(f"\n✅ Melting temperatures: {df['melting_temperature'].tolist()}")
    
    @integration_test
    @skip_if_no_api_key()
    def test_caching_behavior(self):
        """Test that predictions are cached and reused."""
        # First run
        pipeline1 = DataPipeline(
            sequences=self.test_sequences,
            datastore=self.db_path,
            verbose=True
        )
        pipeline1.add_prediction('esm2stabp', stage_name='tm_pred')
        
        start1 = time.time()
        results1 = pipeline1.run()
        time1 = time.time() - start1
        
        # Second run with same sequences - should use cache
        pipeline2 = DataPipeline(
            sequences=self.test_sequences,
            datastore=self.db_path,
            verbose=True
        )
        pipeline2.add_prediction('esm2stabp', stage_name='tm_pred')
        
        start2 = time.time()
        results2 = pipeline2.run()
        time2 = time.time() - start2
        
        # Verify results are identical
        df1 = pipeline1.get_final_data()
        df2 = pipeline2.get_final_data()
        pd.testing.assert_frame_equal(
            df1[['sequence', 'melting_temperature']].sort_values('sequence').reset_index(drop=True),
            df2[['sequence', 'melting_temperature']].sort_values('sequence').reset_index(drop=True)
        )
        
        # Second run should be significantly faster
        speedup = time1 / max(time2, 0.001)  # Avoid division by zero
        print(f"\n⏱️  First run: {time1:.2f}s")
        print(f"⏱️  Cached run: {time2:.2f}s")
        print(f"✅ Speedup: {speedup:.1f}x")
        
        # Cache should provide at least 10x speedup
        self.assertGreater(speedup, 10.0, "Cache should provide significant speedup")
    
    @integration_test
    @skip_if_no_api_key()
    def test_multi_model_parallel_predictions(self):
        """Test running multiple models in parallel."""
        pipeline = DataPipeline(
            sequences=self.test_sequences[:2],  # Use 2 to minimize cost
            datastore=self.db_path,
            verbose=True
        )
        
        # Add multiple predictions that can run in parallel
        pipeline.add_predictions(['esm2stabp', 'biolmsol'])
        
        results = pipeline.run()
        df = pipeline.get_final_data()
        
        # Verify both models ran
        self.assertIn('esm2stabp_predict', results)
        self.assertIn('biolmsol_predict', results)
        
        # Verify both predictions in dataframe
        self.assertIn('melting_temperature', df.columns)
        self.assertIn('solubility_score', df.columns)
        
        print(f"\n✅ Both predictions completed in parallel")
    
    @integration_test
    @skip_if_no_api_key()
    def test_pipeline_with_filtering(self):
        """Test pipeline with filtering steps."""
        pipeline = DataPipeline(
            sequences=self.test_sequences,
            datastore=self.db_path,
            verbose=True
        )
        
        # Predict -> Filter -> Predict
        pipeline.add_prediction('esm2stabp', stage_name='tm_pred')
        pipeline.add_filter(
            ThresholdFilter('melting_temperature', min_value=0),  # Keep all that have values
            stage_name='filter_tm'
        )
        pipeline.add_prediction('biolmsol', stage_name='sol_pred', depends_on=['filter_tm'])
        
        results = pipeline.run()
        df = pipeline.get_final_data()
        
        # Verify all stages ran
        self.assertIn('tm_pred', results)
        self.assertIn('filter_tm', results)
        self.assertIn('sol_pred', results)
        
        print(f"\n✅ Pipeline with filtering completed")
    
    @integration_test
    @skip_if_no_api_key()
    def test_rerun_with_different_filter(self):
        """Test re-running pipeline with different filter threshold."""
        # First run with threshold
        pipeline1 = DataPipeline(
            sequences=self.test_sequences,
            datastore=self.db_path,
            verbose=True
        )
        pipeline1.add_prediction('esm2stabp', stage_name='tm_pred')
        pipeline1.add_filter(
            ThresholdFilter('melting_temperature', min_value=100),  # High threshold
            stage_name='filter_1'
        )
        
        results1 = pipeline1.run()
        df1 = pipeline1.get_final_data()
        
        # Second run with lower threshold - should use cached predictions
        pipeline2 = DataPipeline(
            sequences=self.test_sequences,
            datastore=self.db_path,
            verbose=True,
            run_id='run2'
        )
        pipeline2.add_prediction('esm2stabp', stage_name='tm_pred')
        pipeline2.add_filter(
            ThresholdFilter('melting_temperature', min_value=0),  # Lower threshold
            stage_name='filter_1'
        )
        
        start = time.time()
        results2 = pipeline2.run()
        time2 = time.time() - start
        
        # Second run should be fast (cached predictions)
        print(f"\n✅ Re-run completed in {time2:.2f}s (predictions cached)")
        self.assertLess(time2, 5.0, "Re-run with cached predictions should be fast")


class TestSingleStepPipelineIntegration(unittest.TestCase):
    """Integration tests for single-step shortcuts."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.test_sequences = ['MKTAYIAKQRQ', 'MKLAVIDSAQ']
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @integration_test
    @skip_if_no_api_key()
    def test_predict_shortcut(self):
        """Test Predict() shortcut function."""
        df = Predict('esm2stabp', sequences=self.test_sequences, verbose=False)
        
        # Verify DataFrame structure
        self.assertEqual(len(df), len(self.test_sequences))
        self.assertIn('sequence', df.columns)
        self.assertIn('melting_temperature', df.columns)
        
        print(f"\n✅ Predict() shortcut works")
    
    @integration_test
    @skip_if_no_api_key()
    def test_embed_shortcut(self):
        """Test Embed() shortcut function."""
        df = Embed('esm2-650m', sequences=self.test_sequences[:1], verbose=False)
        
        # Verify DataFrame structure
        self.assertEqual(len(df), 1)
        self.assertIn('sequence', df.columns)
        self.assertIn('embedding', df.columns)
        
        # Check embedding is valid
        self.assertIsNotNone(df.iloc[0]['embedding'])
        
        print(f"\n✅ Embed() shortcut works")


class TestEmbeddingsAndPCAIntegration(unittest.TestCase):
    """Integration tests for embeddings and PCA."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.db_path = self.test_dir / 'test_embeddings.db'
        self.test_sequences = [
            'MKTAYIAKQRQ',
            'MKLAVIDSAQ',
            'MKTAYIDSAQ',
            'MKLAVIYDSAQ',
            'MKTAY',
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
            sequences=self.test_sequences,
            datastore=self.db_path,
            verbose=True
        )
        
        # Generate embeddings
        pipeline.add_prediction('esm2-650m', action='encode', stage_name='embeddings')
        
        results = pipeline.run()
        df = pipeline.get_final_data()
        
        # Verify embeddings were created
        embed_result = results['embeddings']
        self.assertEqual(embed_result.output_count, len(self.test_sequences))
        
        # Check embeddings in datastore
        with DataStore(self.db_path) as store:
            for seq in self.test_sequences:
                emb_list = store.get_embeddings_by_sequence(seq, model_name='esm2-650m', load_data=True)
                self.assertGreater(len(emb_list), 0, f"No embeddings found for {seq}")
                
                _, embedding = emb_list[0]
                self.assertIsInstance(embedding, np.ndarray)
                self.assertGreater(len(embedding), 0)
                
                print(f"✅ Embedding for {seq[:10]}...: shape={embedding.shape}")
    
    @integration_test
    @skip_if_no_api_key()
    def test_pca_visualization(self):
        """Test PCA visualization of embeddings."""
        pipeline = DataPipeline(
            sequences=self.test_sequences,
            datastore=self.db_path,
            verbose=True
        )
        
        # Generate embeddings
        pipeline.add_prediction('esm2-650m', action='encode', stage_name='embeddings')
        results = pipeline.run()
        
        # Get data with embeddings
        with DataStore(self.db_path) as store:
            df = store.export_to_dataframe()
            
            # Manually load embeddings
            embeddings_list = []
            for seq in df['sequence']:
                emb_list = store.get_embeddings_by_sequence(seq, model_name='esm2-650m', load_data=True)
                if emb_list:
                    _, embedding = emb_list[0]
                    embeddings_list.append(embedding)
                else:
                    embeddings_list.append(None)
        
        # Verify all have embeddings
        self.assertEqual(len(embeddings_list), len(self.test_sequences))
        self.assertTrue(all(e is not None for e in embeddings_list))
        
        # Create PCA plot
        from biolmai.pipeline.visualization import plot_embedding_pca
        
        # Stack embeddings into array
        embeddings_array = np.stack(embeddings_list)
        
        fig = plot_embedding_pca(
            embeddings=embeddings_array,
            title='ESM2-650M Embeddings - PCA'
        )
        
        # Save plot
        fig_path = self.test_dir / 'pca_embeddings.png'
        fig.savefig(fig_path, dpi=100, bbox_inches='tight')
        
        # Verify
        self.assertTrue(fig_path.exists())
        file_size = fig_path.stat().st_size
        self.assertGreater(file_size, 1000, f"PCA plot too small: {file_size} bytes")
        
        print(f"\n✅ PCA plot created: {fig_path} ({file_size:,} bytes)")
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestComplexMultiLevelPipeline(unittest.TestCase):
    """Integration test for complex multi-level pipeline with structure prediction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.db_path = self.test_dir / 'test_complex.db'
        # Use very short sequences to minimize cost
        self.test_sequences = [
            'MKTAY',  # 5 AA
            'MKLAY',  # 5 AA
            'MKSAY',  # 5 AA
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
        Generate → Predict → Filter → Predict → Structure → Structure-based Prediction
        
        This test demonstrates the full power of the pipeline system.
        """
        pipeline = DataPipeline(
            sequences=self.test_sequences,
            datastore=self.db_path,
            verbose=True
        )
        
        print("\n" + "="*60)
        print("Complex Multi-Level Pipeline Test")
        print("="*60)
        
        # Level 1: Initial predictions (parallel)
        print("\n[Level 1] Adding initial predictions...")
        pipeline.add_prediction('esm2stabp', stage_name='tm_pred')
        pipeline.add_prediction('biolmsol', stage_name='sol_pred')
        
        # Level 2: Filter based on predictions
        print("[Level 2] Adding filter...")
        pipeline.add_filter(
            ThresholdFilter('melting_temperature', min_value=40),
            stage_name='tm_filter',
            depends_on=['tm_pred']
        )
        
        # Level 3: Structure prediction (expensive!)
        print("[Level 3] Adding structure prediction...")
        pipeline.add_prediction(
            'esmfold',
            action='predict',
            prediction_type='structure',
            stage_name='structure_pred',
            depends_on=['tm_filter']
        )
        
        # Level 4: Structure-based prediction (uses structure as input)
        # Note: Would need ProteinMPNN or similar that takes structure
        # For now, just do another prediction based on passing filter
        print("[Level 4] Adding final prediction...")
        pipeline.add_prediction(
            'esm2-650m',
            action='encode',
            stage_name='final_embed',
            depends_on=['structure_pred']
        )
        
        # Show execution plan
        print("\nExecution Plan:")
        levels = pipeline._resolve_dependencies()
        for i, level in enumerate(levels):
            stage_names = [s.name for s in level]
            print(f"  Level {i+1}: {', '.join(stage_names)}")
        
        # Run pipeline
        print("\nRunning pipeline...")
        results = pipeline.run()
        
        # Verify all stages completed
        self.assertIn('tm_pred', results)
        self.assertIn('sol_pred', results)
        self.assertIn('tm_filter', results)
        self.assertIn('structure_pred', results)
        self.assertIn('final_embed', results)
        
        # Get final results
        df = pipeline.get_final_data()
        
        print(f"\n✅ Complex pipeline completed!")
        print(f"   Input sequences: {len(self.test_sequences)}")
        print(f"   Final sequences: {len(df)}")
        print(f"   Total stages: {len(results)}")
        print(f"   Levels: {len(levels)}")
        
        # Verify structures were created
        with DataStore(self.db_path) as store:
            for seq in df['sequence']:
                structures = store.get_structures_by_sequence(seq, model_name='esmfold')
                if structures:
                    print(f"   ✅ Structure found for {seq}")



class TestEdgeCasesIntegration(unittest.TestCase):
    """Integration tests for edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.db_path = self.test_dir / 'test_pipeline.db'
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @integration_test
    @skip_if_no_api_key()
    def test_empty_filter_result(self):
        """Test handling of filters that produce empty results."""
        pipeline = DataPipeline(
            sequences=['MKTAY', 'MKLAVIY'],
            datastore=self.db_path,
            verbose=True
        )
        
        pipeline.add_prediction('esm2stabp')
        pipeline.add_filter(
            ThresholdFilter('melting_temperature', min_value=500),  # Impossible threshold
            stage_name='filter_1'
        )
        
        results = pipeline.run()
        df = pipeline.get_final_data()
        
        print(f"\n✅ Filter produced {len(df)} sequences (threshold 500)")
    
    @integration_test
    @skip_if_no_api_key()
    def test_duplicate_sequences(self):
        """Test automatic deduplication."""
        sequences = ['MKTAY', 'MKLAVIY', 'MKTAY']  # Duplicate
        
        pipeline = DataPipeline(
            sequences=sequences,
            datastore=self.db_path,
            verbose=True
        )
        
        pipeline.add_prediction('esm2stabp')
        results = pipeline.run()
        df = pipeline.get_final_data()
        
        # Should have deduplicated
        unique_seqs = df['sequence'].nunique()
        self.assertEqual(unique_seqs, 2)
        
        print(f"\n✅ Deduplicated: {len(sequences)} input → {unique_seqs} unique sequences")
    
    @integration_test
    @skip_if_no_api_key()
    def test_sequence_length_variation(self):
        """Test handling sequences of various lengths."""
        sequences = [
            'MKTAY',  # 5 AA
            'MKTAYIAKQRQ',  # 11 AA
            'MKTAYIAKQRQGHQAMAEIKQGHQAMAEIKQ',  # 32 AA
        ]
        
        pipeline = DataPipeline(
            sequences=sequences,
            datastore=self.db_path,
            verbose=True
        )
        
        # Filter to medium length first
        pipeline.add_filter(SequenceLengthFilter(min_length=10, max_length=30))
        pipeline.add_prediction('esm2stabp')
        
        results = pipeline.run()
        df = pipeline.get_final_data()
        
        # The filter runs in parallel with predictions, so all sequences get predicted
        # but only filtered ones should pass through the filter stage
        # Check the filter actually worked by looking at the result
        filter_result = results['filter_0']
        self.assertEqual(filter_result.output_count, 1, "Only 1 sequence should pass filter")
        
        print(f"\n✅ Length filter worked: {filter_result.output_count} sequence passed")


class TestDataStoreIntegration(unittest.TestCase):
    """Integration tests for DataStore operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.db_path = self.test_dir / 'test_pipeline.db'
        self.test_sequences = ['MKTAYIAKQRQ', 'MKLAVIDSAQ', 'MKTAYIDSAQ']
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @integration_test
    @skip_if_no_api_key()
    def test_datastore_export_options(self):
        """Test various export options from DataStore."""
        pipeline = DataPipeline(
            sequences=self.test_sequences,
            datastore=self.db_path,
            verbose=True
        )
        
        pipeline.add_prediction('esm2stabp')
        pipeline.add_prediction('biolmsol')
        
        results = pipeline.run()
        
        # Test different export options
        with DataStore(self.db_path) as store:
            # Basic export
            df1 = store.export_to_dataframe()
            self.assertEqual(len(df1), 3)
            
            # With predictions - note: datastore exports with column names like "prediction_type_model_name"
            df2 = store.export_to_dataframe(include_predictions=True)
            # The actual column names from datastore are like "esm2stabp_predict_esm2stabp"
            self.assertIn('esm2stabp_predict_esm2stabp', df2.columns)
            
            # Check data
            print(f"\n✅ DataStore export works: {len(df2)} sequences, {len(df2.columns)} columns")
    
    @integration_test
    @skip_if_no_api_key()
    def test_export_to_csv(self):
        """Test exporting pipeline results to CSV."""
        pipeline = DataPipeline(
            sequences=self.test_sequences,
            datastore=self.db_path,
            verbose=True
        )
        
        pipeline.add_predictions(['esm2stabp', 'biolmsol'])
        results = pipeline.run()
        
        # Export to CSV
        csv_path = self.test_dir / 'results.csv'
        with DataStore(self.db_path) as store:
            store.export_to_csv(str(csv_path), include_predictions=True)
        
        # Verify CSV
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 3)
        self.assertIn('sequence', df.columns)
        
        print(f"\n✅ CSV export successful: {csv_path}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
