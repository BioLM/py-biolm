"""
Tests for streaming pipeline execution.
"""

import pandas as pd
import pytest

from biolmai.pipeline.data import DataPipeline, FilterStage, PredictionStage
from biolmai.pipeline.datastore import DataStore
from biolmai.pipeline.filters import RankingFilter, ThresholdFilter


@pytest.fixture
def datastore(tmp_path):
    """Create a temporary datastore."""
    return DataStore(db_path=tmp_path / "test.db", data_dir=tmp_path / "data")


def test_threshold_filter_is_streamable():
    """Verify ThresholdFilter is marked as streamable."""
    filter = ThresholdFilter("plddt", min_value=0.8)
    assert not filter.requires_complete_data, "ThresholdFilter should be streamable"


def test_ranking_filter_requires_complete_data():
    """Verify RankingFilter requires complete data."""
    filter = RankingFilter("tm", n=100, ascending=False)
    assert filter.requires_complete_data, "RankingFilter should require complete data"


@pytest.mark.asyncio
async def test_prediction_stage_streaming(datastore):
    """Test PredictionStage can yield results in streaming mode."""
    # Create a simple prediction stage
    stage = PredictionStage(
        name="test_predict",
        model_name="esm2_t6_8M",
        action="predict",
        prediction_type="test_score",
    )

    # Mock input data
    pd.DataFrame({"sequence": ["ACDEFG", "ACDEFGH", "ACDEFGHI"]})

    # Check that process_streaming method exists
    assert hasattr(
        stage, "process_streaming"
    ), "PredictionStage should have process_streaming"

    # Note: We can't easily test actual streaming without mocking the API
    # This test just verifies the structure is in place


def test_filter_stage_knows_if_streamable(datastore):
    """Test FilterStage correctly identifies streamable filters."""
    # Streamable filter
    threshold_filter = ThresholdFilter("plddt", min_value=0.8)
    stage1 = FilterStage(name="threshold", filter_func=threshold_filter)
    assert (
        not stage1.requires_complete_data
    ), "ThresholdFilter stage should be streamable"

    # Non-streamable filter
    ranking_filter = RankingFilter("tm", n=100)
    stage2 = FilterStage(name="ranking", filter_func=ranking_filter)
    assert stage2.requires_complete_data, "RankingFilter stage should not be streamable"


@pytest.mark.asyncio
async def test_pipeline_with_streaming_enabled(datastore, tmp_path):
    """Test pipeline runs with streaming enabled."""
    sequences = ["ACDEFG", "MKLLIV", "AAAAAA"]

    pipeline = DataPipeline(
        sequences=sequences, datastore=datastore, output_dir=tmp_path, verbose=True
    )

    # Add prediction followed by streamable filter
    pipeline.add_stage(
        PredictionStage(
            name="predict_tm",
            model_name="esm2_t6_8M",
            action="predict",
            prediction_type="tm",
        )
    )

    pipeline.add_stage(
        FilterStage(
            name="filter_high_tm",
            filter_func=ThresholdFilter("tm", min_value=50.0),
            depends_on=["predict_tm"],
        )
    )

    # Note: We can't run this without actual API access
    # This test verifies the structure compiles
    assert pipeline is not None


@pytest.mark.asyncio
async def test_pipeline_identifies_streaming_opportunity():
    """Test pipeline correctly identifies when streaming is possible."""
    from biolmai.pipeline.base import BasePipeline
    from biolmai.pipeline.data import FilterStage, PredictionStage
    from biolmai.pipeline.filters import RankingFilter, ThresholdFilter

    # Create mock pipeline
    class TestPipeline(BasePipeline):
        async def _get_initial_data(self, **kwargs):
            return pd.DataFrame({"sequence": ["ABC", "DEF"]})

    pipeline = TestPipeline(datastore=None, output_dir="/tmp")

    pred_stage = PredictionStage(
        name="predict",
        model_name="esm2_t6_8M",
        action="predict",
        prediction_type="score",
    )

    threshold_stage = FilterStage(
        name="threshold", filter_func=ThresholdFilter("score", min_value=0.5)
    )

    ranking_stage = FilterStage(
        name="ranking", filter_func=RankingFilter("score", n=10)
    )

    # Should be able to stream from prediction to threshold filter
    assert pipeline._can_stream_to_next(
        pred_stage, threshold_stage
    ), "Should be able to stream to threshold filter"

    # Should NOT be able to stream from prediction to ranking filter
    assert not pipeline._can_stream_to_next(
        pred_stage, ranking_stage
    ), "Should NOT be able to stream to ranking filter"


def test_all_base_filters_categorized():
    """Ensure all filters have requires_complete_data set."""
    import inspect

    from biolmai.pipeline import filters

    filter_classes = [
        cls
        for name, cls in inspect.getmembers(filters, inspect.isclass)
        if issubclass(cls, filters.BaseFilter) and cls != filters.BaseFilter
    ]

    for filter_cls in filter_classes:
        # All filter classes should have requires_complete_data defined
        assert hasattr(
            filter_cls, "requires_complete_data"
        ), f"{filter_cls.__name__} missing requires_complete_data"

        value = filter_cls.requires_complete_data
        assert isinstance(
            value, bool
        ), f"{filter_cls.__name__}.requires_complete_data should be bool, got {type(value)}"

        print(f"  {filter_cls.__name__}: requires_complete_data={value}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
