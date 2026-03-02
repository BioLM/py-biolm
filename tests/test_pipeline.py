"""
Unit tests for pipeline functionality.
"""

import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from biolmai.pipeline import DataPipeline, SingleStepPipeline
from biolmai.pipeline.base import Stage, StageResult
from biolmai.pipeline.filters import SequenceLengthFilter


class MockPredictionStage(Stage):
    """Mock prediction stage for testing."""

    def __init__(self, name: str, prediction_type: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.prediction_type = prediction_type

    async def process(self, df, datastore, **kwargs):
        """Mock processing - just add random values."""
        import numpy as np

        df = df.copy()
        df[self.prediction_type] = np.random.uniform(50, 90, len(df))

        return df, StageResult(
            stage_name=self.name,
            input_count=len(df),
            output_count=len(df),
            computed_count=len(df),
        )


class TestDataPipeline(unittest.TestCase):
    """Test DataPipeline functionality."""

    def setUp(self):
        """Create temporary directory."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)

    def test_pipeline_creation(self):
        """Test creating a pipeline."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]

        pipeline = DataPipeline(sequences=sequences, output_dir=self.test_dir)

        self.assertIsNotNone(pipeline)
        self.assertEqual(pipeline.pipeline_type, "DataPipeline")

    def test_add_stage(self):
        """Test adding stages to pipeline."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]

        pipeline = DataPipeline(sequences=sequences, output_dir=self.test_dir)

        # Add mock stage
        stage = MockPredictionStage("test_stage", "test_pred")
        pipeline.add_stage(stage)

        self.assertEqual(len(pipeline.stages), 1)
        self.assertEqual(pipeline.stages[0].name, "test_stage")

    def test_add_filter(self):
        """Test adding filter to pipeline."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ", "MKTAY"]

        pipeline = DataPipeline(sequences=sequences, output_dir=self.test_dir)

        pipeline.add_filter(SequenceLengthFilter(min_length=10))

        self.assertEqual(len(pipeline.stages), 1)

    def test_stage_dependencies(self):
        """Test stage dependency resolution."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]

        pipeline = DataPipeline(sequences=sequences, output_dir=self.test_dir)

        stage1 = MockPredictionStage("stage1", "pred1")
        stage2 = MockPredictionStage("stage2", "pred2", depends_on=["stage1"])

        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)

        # Resolve dependencies
        levels = pipeline._resolve_dependencies()

        self.assertEqual(len(levels), 2)
        self.assertEqual(levels[0][0].name, "stage1")
        self.assertEqual(levels[1][0].name, "stage2")

    def test_parallel_stages(self):
        """Test parallel stage execution."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]

        pipeline = DataPipeline(sequences=sequences, output_dir=self.test_dir)

        # Add two stages with no dependencies (can run in parallel)
        stage1 = MockPredictionStage("stage1", "pred1")
        stage2 = MockPredictionStage("stage2", "pred2")

        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)

        levels = pipeline._resolve_dependencies()

        # Both should be in same level
        self.assertEqual(len(levels), 1)
        self.assertEqual(len(levels[0]), 2)

    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        sequences = ["MKTAYIAKQRQ"]

        pipeline = DataPipeline(sequences=sequences, output_dir=self.test_dir)

        # Create circular dependency manually
        stage1 = MockPredictionStage("stage1", "pred1", depends_on=["stage2"])
        stage2 = MockPredictionStage("stage2", "pred2", depends_on=["stage1"])

        pipeline.stages = [stage1, stage2]

        # Should raise error
        with self.assertRaises(ValueError):
            pipeline._resolve_dependencies()

    def test_add_predictions_parallel(self):
        """Test adding multiple predictions at once."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]

        pipeline = DataPipeline(
            sequences=sequences, output_dir=self.test_dir, verbose=False
        )

        # Add multiple predictions (they should be at same level)
        # Note: We'll use mock stages since real API calls won't work in tests
        stage1 = MockPredictionStage("model1", "pred1")
        stage2 = MockPredictionStage("model2", "pred2")
        stage3 = MockPredictionStage("model3", "pred3")

        pipeline.add_stage(stage1)
        pipeline.add_stage(stage2)
        pipeline.add_stage(stage3)

        # All should be in same level (parallel)
        levels = pipeline._resolve_dependencies()
        self.assertEqual(len(levels), 1)
        self.assertEqual(len(levels[0]), 3)

    def test_export_to_csv(self):
        """Test exporting results to CSV."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]

        pipeline = DataPipeline(
            sequences=sequences, output_dir=self.test_dir, verbose=False
        )

        # Add and run mock stage
        stage = MockPredictionStage("test", "tm")
        pipeline.add_stage(stage)

        # Run pipeline
        pipeline.run()

        # Export
        output_path = Path(self.test_dir) / "results.csv"
        pipeline.export_to_csv(output_path)

        # Verify file exists
        self.assertTrue(output_path.exists())

        # Read back and verify
        df = pd.read_csv(output_path)
        self.assertEqual(len(df), 2)
        self.assertIn("sequence", df.columns)
        self.assertIn("tm", df.columns)

    def test_pipeline_summary(self):
        """Test pipeline summary generation."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]

        pipeline = DataPipeline(
            sequences=sequences, output_dir=self.test_dir, verbose=False
        )

        stage = MockPredictionStage("test", "tm")
        pipeline.add_stage(stage)

        pipeline.run()

        # Get summary
        summary = pipeline.summary()

        self.assertIsInstance(summary, pd.DataFrame)
        self.assertGreater(len(summary), 0)
        self.assertIn("Stage", summary.columns)
        self.assertIn("Input", summary.columns)
        self.assertIn("Output", summary.columns)


class TestSingleStepPipeline(unittest.TestCase):
    """Test SingleStepPipeline."""

    def setUp(self):
        """Create temporary directory."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)

    def test_single_step_creation(self):
        """Test creating single-step pipeline."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]

        # Note: This will fail without real API, but tests the structure
        pipeline = SingleStepPipeline(
            model_name="test_model",
            sequences=sequences,
            output_dir=self.test_dir,
            verbose=False,
        )

        # Should have one stage automatically added
        self.assertEqual(len(pipeline.stages), 1)


class TestPipelineResumability(unittest.TestCase):
    """Test pipeline resumability."""

    def setUp(self):
        """Create temporary directory."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)

    def test_run_id_persistence(self):
        """Test that run_id persists."""
        sequences = ["MKTAYIAKQRQ"]

        pipeline = DataPipeline(
            sequences=sequences,
            output_dir=self.test_dir,
            run_id="test_run_123",
            verbose=False,
        )

        self.assertEqual(pipeline.run_id, "test_run_123")

    def test_stage_completion_marker(self):
        """Test stage completion markers."""
        sequences = ["MKTAYIAKQRQ"]

        pipeline = DataPipeline(
            sequences=sequences,
            output_dir=self.test_dir,
            run_id="test_run",
            verbose=False,
        )

        stage = MockPredictionStage("test_stage", "tm")
        pipeline.add_stage(stage)

        # Run pipeline
        pipeline.run()

        # Check stage completion
        stage_id = f"{pipeline.run_id}_{stage.name}"
        self.assertTrue(pipeline.datastore.is_stage_complete(stage_id))


class TestPipelineInputFormats(unittest.TestCase):
    """Test different input formats."""

    def setUp(self):
        """Create temporary directory."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)

    def test_list_input(self):
        """Test list input."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]

        pipeline = DataPipeline(
            sequences=sequences, output_dir=self.test_dir, verbose=False
        )

        # Get initial data
        df = asyncio.run(pipeline._get_initial_data())

        self.assertEqual(len(df), 2)
        self.assertIn("sequence", df.columns)
        self.assertIn("sequence_id", df.columns)

    def test_dataframe_input(self):
        """Test DataFrame input."""
        df = pd.DataFrame(
            {"sequence": ["MKTAYIAKQRQ", "MKLAVIDSAQ"], "name": ["seq1", "seq2"]}
        )

        pipeline = DataPipeline(sequences=df, output_dir=self.test_dir, verbose=False)

        df_init = asyncio.run(pipeline._get_initial_data())

        self.assertEqual(len(df_init), 2)
        self.assertIn("name", df_init.columns)  # Additional columns preserved

    def test_csv_input(self):
        """Test CSV input."""
        df = pd.DataFrame({"sequence": ["MKTAYIAKQRQ", "MKLAVIDSAQ"]})

        csv_path = Path(self.test_dir) / "sequences.csv"
        df.to_csv(csv_path, index=False)

        pipeline = DataPipeline(
            sequences=str(csv_path), output_dir=self.test_dir, verbose=False
        )

        df_init = asyncio.run(pipeline._get_initial_data())

        self.assertEqual(len(df_init), 2)

    def test_fasta_input(self):
        """Test FASTA input."""
        from biolmai.pipeline.utils import write_fasta

        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ"]
        fasta_path = Path(self.test_dir) / "sequences.fasta"

        write_fasta(sequences, fasta_path)

        pipeline = DataPipeline(
            sequences=str(fasta_path), output_dir=self.test_dir, verbose=False
        )

        df_init = asyncio.run(pipeline._get_initial_data())

        self.assertEqual(len(df_init), 2)


class TestPipelineDeduplication(unittest.TestCase):
    """Test sequence deduplication."""

    def setUp(self):
        """Create temporary directory."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)

    def test_automatic_deduplication(self):
        """Test automatic deduplication on load."""
        sequences = ["MKTAYIAKQRQ", "MKLAVIDSAQ", "MKTAYIAKQRQ", "MKTAYIAKQRQ"]

        pipeline = DataPipeline(
            sequences=sequences, output_dir=self.test_dir, verbose=False
        )

        df = asyncio.run(pipeline._get_initial_data())

        # Should be deduplicated to 2 unique
        self.assertEqual(len(df), 2)


if __name__ == "__main__":
    unittest.main()
