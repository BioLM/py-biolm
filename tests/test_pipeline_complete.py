"""
Complete end-to-end pipeline tests.

Tests cover:
- FilterStage actually filters in batch mode (regression for Bug 6)
- Resume: skip completed stage and reload output from DuckDB
- Deduplication across runs
- Diff mode counts
- Multi-stage pipeline (generation → prediction → filter)
- explore() / stats() / query() methods
- DirectGenerationConfig with mocked API
- Streaming trickles results
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd

from biolmai.pipeline.data import DataPipeline, PredictionStage
from biolmai.pipeline.datastore_duckdb import DuckDBDataStore
from biolmai.pipeline.filters import (
    RankingFilter,
    ThresholdFilter,
)
from biolmai.pipeline.generative import (
    DirectGenerationConfig,
    GenerationStage,
    GenerativePipeline,
)
from biolmai.pipeline.mlm_remasking import RemaskingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEQS = ["MKTAYIAKQRQ", "ACDEFGHIKLM", "NQRSTUVWXYZ", "AAAABBBBCCCC"]


def make_api_mock(values=None, embeddings=False, generate_seqs=None):
    """Return an AsyncMock whose predict/encode/generate returns canned results."""
    mock = AsyncMock()

    async def _predict(items, params=None):
        base = values or [65.0] * len(items)
        return [{"melting_temperature": base[i % len(base)]} for i in range(len(items))]

    async def _encode(items, params=None):
        return [{"embedding": list(np.ones(32))} for _ in items]

    async def _generate(items, params=None):
        seqs = generate_seqs or [
            f"GENERATED{i:03d}AAAA" for i in range(params.get("num_sequences", 3))
        ]
        return [{"sequence": s} for s in seqs]

    mock.predict = AsyncMock(side_effect=_predict)
    mock.encode = AsyncMock(side_effect=_encode)
    mock.generate = AsyncMock(side_effect=_generate)
    mock.shutdown = AsyncMock()
    return mock


def _make_pipeline(tmp_path, sequences=None, **kwargs):
    """Create a DataPipeline backed by a temp DuckDB."""
    db = tmp_path / "test.duckdb"
    data_dir = tmp_path / "data"
    ds = DuckDBDataStore(db_path=db, data_dir=data_dir)
    return DataPipeline(
        sequences=sequences or SEQS,
        datastore=ds,
        verbose=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test 1: FilterStage actually filters in batch mode (Bug 6 regression)
# ---------------------------------------------------------------------------


def test_filter_stage_actually_filters_in_batch_mode(tmp_path):
    """Filters must reduce the DataFrame in non-streaming batch mode."""
    # Values for SEQS[0..3]: 80, 40, 90, 30 — only 0 and 2 pass min_value=60
    api_values = [80.0, 40.0, 90.0, 30.0]

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=api_values)
        pipeline = _make_pipeline(tmp_path, sequences=SEQS[:4])
        pipeline.add_prediction("temberture", prediction_type="tm")
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=60.0),
            stage_name="filter_tm",
            depends_on=["temberture_predict"],
        )
        pipeline.run()

    df = pipeline.get_final_data()
    assert len(df) == 2, f"Expected 2 sequences after filter, got {len(df)}"
    assert all(df["tm"] >= 60.0), "All remaining sequences should have tm >= 60"


# ---------------------------------------------------------------------------
# Test 2: mark_stage_complete() does not crash (Bug 1 regression)
# ---------------------------------------------------------------------------


def test_mark_stage_complete_no_crash(tmp_path):
    """mark_stage_complete with status kwarg must not raise TypeError."""
    db = tmp_path / "test.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "data")
    # Should not raise
    ds.mark_stage_complete(
        stage_id="run1_stage1",
        run_id="run1",
        stage_name="stage1",
        input_count=10,
        output_count=8,
        status="completed",
    )
    assert ds.is_stage_complete("run1_stage1")


# ---------------------------------------------------------------------------
# Test 3: Deduplication — no duplicate sequences across runs
# ---------------------------------------------------------------------------


def test_deduplication_no_dupes_across_runs(tmp_path):
    """Re-inserting sequences should not create duplicates in the datastore."""
    db = tmp_path / "test.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "data")

    ids1 = ds.add_sequences_batch(["AAAA", "BBBB"])
    ids2 = ds.add_sequences_batch(["AAAA", "BBBB", "CCCC"])

    total = ds.conn.execute("SELECT COUNT(*) FROM sequences").fetchone()[0]
    assert total == 3, f"Expected 3 unique sequences, found {total}"
    # AAAA and BBBB IDs should be stable
    assert ids1[0] == ids2[0]
    assert ids1[1] == ids2[1]


# ---------------------------------------------------------------------------
# Test 4: Resume — completed stage reloads from DB without re-calling API
# ---------------------------------------------------------------------------


def test_resume_skips_completed_stage_and_reloads_output(tmp_path):
    """On a resume run the prediction stage should not call the API again."""
    run_id = "resume_test_run"
    api_values = [70.0, 60.0, 80.0, 50.0]

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        mock_inst = make_api_mock(values=api_values)
        MockCls.return_value = mock_inst

        # First run — populates DB
        pipeline1 = _make_pipeline(tmp_path, sequences=SEQS[:4], run_id=run_id)
        pipeline1.add_prediction("temberture", prediction_type="tm")
        pipeline1.run()

        call_count_first = mock_inst.predict.call_count

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls2:
        mock_inst2 = make_api_mock(values=api_values)
        MockCls2.return_value = mock_inst2

        # Second run — resume=True with same run_id
        pipeline2 = _make_pipeline(
            tmp_path, sequences=SEQS[:4], run_id=run_id, resume=True
        )
        pipeline2.add_prediction("temberture", prediction_type="tm")
        pipeline2.run()

        call_count_second = mock_inst2.predict.call_count

    assert call_count_first > 0, "First run should have called the API"
    assert call_count_second == 0, "Resume run should not call the API again"

    df = pipeline2.get_final_data()
    assert len(df) == 4
    assert "tm" in df.columns


# ---------------------------------------------------------------------------
# Test 5: Diff mode — count of new vs cached sequences
# ---------------------------------------------------------------------------


def test_diff_mode_skips_existing_sequences(tmp_path):
    """Diff mode: sequences already in DB should not get duplicate predictions."""
    api_values = [65.0, 75.0, 55.0, 85.0]

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=api_values)
        p1 = _make_pipeline(tmp_path, sequences=SEQS[:2], diff_mode=True)
        p1.add_prediction("temberture", prediction_type="tm")
        p1.run()

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        mock_inst = make_api_mock(values=api_values)
        MockCls.return_value = mock_inst
        # Add 2 new sequences — only they should need predictions
        p2 = _make_pipeline(tmp_path, sequences=SEQS, diff_mode=True)
        p2.add_prediction("temberture", prediction_type="tm")
        p2.run()
        call_count = mock_inst.predict.call_count

    # Only the 2 new sequences should have been predicted
    total_items_called = sum(
        len(call.kwargs.get("items", call.args[0] if call.args else []))
        for call in mock_inst.predict.call_args_list
    )
    assert (
        total_items_called == 2
    ), f"Expected 2 new predictions, got {total_items_called}"


# ---------------------------------------------------------------------------
# Test 6: explore() and stats() methods
# ---------------------------------------------------------------------------


def test_explore_and_stats_methods(tmp_path):
    """explore() and stats() return correct counts after a pipeline run."""
    api_values = [70.0, 80.0, 55.0, 90.0]

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=api_values)
        pipeline = _make_pipeline(tmp_path, sequences=SEQS[:4])
        pipeline.add_prediction("temberture", prediction_type="tm")
        pipeline.run()

    info = pipeline.explore()
    assert info["sequences"] == 4
    assert info["predictions"].get("tm", 0) == 4
    assert info["completed_stages"] >= 1

    stats_df = pipeline.stats()
    assert len(stats_df) >= 1
    assert "temberture_predict" in stats_df["stage_name"].tolist()


# ---------------------------------------------------------------------------
# Test 7: query() method exposes raw DuckDB SQL
# ---------------------------------------------------------------------------


def test_query_method(tmp_path):
    """pipeline.query() executes SQL against the datastore."""
    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=[70.0] * 4)
        pipeline = _make_pipeline(tmp_path, sequences=SEQS[:4])
        pipeline.add_prediction("temberture", prediction_type="tm")
        pipeline.run()

    df = pipeline.query("SELECT COUNT(*) AS n FROM sequences")
    assert df["n"].iloc[0] == 4


# ---------------------------------------------------------------------------
# Test 8: DirectGenerationConfig — structure-conditioned generation
# ---------------------------------------------------------------------------


def test_direct_generation_stage_with_structure_string(tmp_path):
    """GenerationStage with DirectGenerationConfig generates sequences."""
    generated = ["GENERATED001AAAA", "GENERATED002AAAA", "GENERATED003AAAA"]
    # Write a minimal fake PDB file
    pdb_content = (
        "ATOM      1  CA  ALA A   1       1.000   1.000   1.000  1.00  0.00\nEND\n"
    )
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(pdb_content)

    config = DirectGenerationConfig(
        model_name="proteinmpnn",
        structure_path=str(pdb_file),
        num_sequences=3,
        temperature=1.0,
    )

    db = tmp_path / "gen.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "gen_data")

    with patch("biolmai.pipeline.generative.BioLMApiClient") as MockCls:
        mock_api = make_api_mock(generate_seqs=generated)
        MockCls.return_value = mock_api

        stage = GenerationStage(name="generation", config=config)
        df_out, result = asyncio.run(stage.process(pd.DataFrame(), ds))

    assert len(df_out) == 3
    assert set(df_out["sequence"].tolist()) == set(generated)
    assert result.output_count == 3


# ---------------------------------------------------------------------------
# Test 9: RemaskingConfig dispatch via GenerationStage
# ---------------------------------------------------------------------------


def test_remasking_stage_generates_variants(tmp_path):
    """GenerationStage with RemaskingConfig calls MLMRemasker.generate_variants."""
    parent = "MKTAYIAKQRQ"
    fake_variants = [("MKTAYIAKQRA", {"num_mutations": 1, "mutation_rate": 0.09})]

    config = RemaskingConfig(
        model_name="esm2-8m",
        mask_fraction=0.15,
        num_iterations=1,
    )
    # Attach parent_sequence and num_variants dynamically for dispatch
    config.parent_sequence = parent
    config.num_variants = 1

    db = tmp_path / "remask.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "remask_data")

    # Patch MLMRemasker where it is used (generative.py imports it directly)
    with (
        patch("biolmai.pipeline.generative.MLMRemasker") as MockRemasker,
        patch("biolmai.pipeline.generative.BioLMApiClient") as MockCls,
    ):
        instance = MagicMock()
        instance.generate_variants = AsyncMock(return_value=fake_variants)
        MockRemasker.return_value = instance
        MockCls.return_value = make_api_mock()

        stage = GenerationStage(name="generation", config=config)
        df_out, result = asyncio.run(stage.process(pd.DataFrame(), ds))

    assert len(df_out) >= 1
    assert "MKTAYIAKQRA" in df_out["sequence"].tolist()


# ---------------------------------------------------------------------------
# Test 10: Multi-stage pipeline — generation → prediction → filter
# ---------------------------------------------------------------------------


def test_multi_stage_pipeline_generation_prediction_filter(tmp_path):
    """GenerativePipeline: generate → predict → filter works end-to-end."""
    gen_seqs = ["GENAAAAAAAA", "GENBBBBBBB", "GENCCCCCCC", "GENDDDDDDD"]
    pred_values = [80.0, 45.0, 75.0, 30.0]  # Only 0 and 2 pass >= 60

    db = tmp_path / "gen_pipe.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "gen_pipe_data")

    pdb_content = (
        "ATOM      1  CA  ALA A   1       1.000   1.000   1.000  1.00  0.00\nEND\n"
    )
    pdb_file = tmp_path / "struct.pdb"
    pdb_file.write_text(pdb_content)

    gen_config = DirectGenerationConfig(
        model_name="proteinmpnn",
        structure_path=str(pdb_file),
        num_sequences=4,
    )

    with (
        patch("biolmai.pipeline.generative.BioLMApiClient") as GenCls,
        patch("biolmai.pipeline.data.BioLMApiClient") as PredCls,
    ):
        gen_api = make_api_mock(generate_seqs=gen_seqs)
        GenCls.return_value = gen_api

        pred_api = make_api_mock(values=pred_values)
        PredCls.return_value = pred_api

        pipeline = GenerativePipeline(
            generation_configs=[gen_config],
            datastore=ds,
            verbose=False,
        )
        pipeline.add_prediction(
            "temberture",
            prediction_type="tm",
        )
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=60.0),
            stage_name="filter_tm",
            depends_on=["temberture_predict"],
        )
        pipeline.run()

    df = pipeline.get_final_data()
    assert len(df) == 2, f"Expected 2 sequences after filter, got {len(df)}"
    assert all(df["tm"] >= 60.0)


# ---------------------------------------------------------------------------
# Bug #10: Missing tests — streaming stage completion, parallel merge,
#           RankingFilter top/bottom, FilterStage copy safety
# ---------------------------------------------------------------------------


def test_streaming_marks_stages_complete(tmp_path):
    """Bug #1 fix: _execute_stage_streaming must persist stage_completions records."""
    api_values = [80.0, 40.0, 90.0, 30.0]

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=api_values)
        pipeline = _make_pipeline(tmp_path, sequences=SEQS[:4])
        pipeline.add_prediction("temberture", prediction_type="tm")
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=60.0),
            stage_name="filter_tm",
            depends_on=["temberture_predict"],
        )
        pipeline.run(enable_streaming=True)

    ds = pipeline.datastore
    run_id = pipeline.run_id

    # Both the prediction stage and the filter stage should be marked complete
    pred_stage_id = f"{run_id}_temberture_predict"
    filter_stage_id = f"{run_id}_filter_tm"

    assert ds.is_stage_complete(
        pred_stage_id
    ), "Prediction stage must be marked complete in streaming mode"
    assert ds.is_stage_complete(
        filter_stage_id
    ), "Filter stage must be marked complete in streaming mode"


def test_parallel_merge_uses_intersection_of_rows(tmp_path):
    """Bug #4 fix: parallel stages share the same input; merged output should
    contain columns from ALL parallel stages with the same row count."""
    # Use separate mocks per model_name to avoid asyncio call-order fragility
    tm_values = [80.0, 40.0, 90.0, 30.0]
    sol_values = [0.9, 0.3, 0.8, 0.2]

    def make_model_mock(values, key):
        m = MagicMock()

        async def _predict(items, params=None):
            return [{key: values[i % len(values)]} for i in range(len(items))]

        m.predict = AsyncMock(side_effect=_predict)
        m.encode = AsyncMock()
        m.shutdown = AsyncMock()
        return m

    tm_mock = make_model_mock(tm_values, "melting_temperature")
    sol_mock = make_model_mock(sol_values, "solubility_score")
    model_mocks = {"temberture": tm_mock, "pro4s": sol_mock}

    def make_client(model_name, **kwargs):
        return model_mocks.get(model_name, tm_mock)

    with patch("biolmai.pipeline.data.BioLMApiClient", side_effect=make_client):
        pipeline = _make_pipeline(tmp_path, sequences=SEQS[:4])
        # Two prediction stages with no explicit depends_on → same dep level → parallel
        pipeline.add_stage(
            PredictionStage(
                name="tm_stage",
                model_name="temberture",
                action="predict",
                prediction_type="tm",
                batch_size=32,
            )
        )
        pipeline.add_stage(
            PredictionStage(
                name="sol_stage",
                model_name="pro4s",
                action="predict",
                prediction_type="sol",
                batch_size=32,
            )
        )
        pipeline.run()

    df = pipeline.get_final_data()
    # Both columns added, same 4 rows
    assert len(df) == 4
    assert "tm" in df.columns, f"'tm' missing from columns: {list(df.columns)}"
    assert "sol" in df.columns, f"'sol' missing from columns: {list(df.columns)}"


def test_ranking_filter_top_n_ascending(tmp_path):
    """RankingFilter(ascending=True, n=2) keeps the 2 lowest values."""
    api_values = [80.0, 40.0, 90.0, 30.0]

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=api_values)
        pipeline = _make_pipeline(tmp_path, sequences=SEQS[:4])
        pipeline.add_prediction("temberture", prediction_type="tm")
        pipeline.add_filter(
            RankingFilter("tm", n=2, ascending=True),  # keep 2 lowest tm
            stage_name="rank_filter",
            depends_on=["temberture_predict"],
        )
        pipeline.run()

    df = pipeline.get_final_data()
    assert len(df) == 2, f"Expected 2 sequences after RankingFilter, got {len(df)}"
    # The 2 lowest values are 30.0 and 40.0
    assert set(df["tm"].tolist()) == {30.0, 40.0}


def test_ranking_filter_top_n_descending(tmp_path):
    """RankingFilter(ascending=False, n=2) keeps the 2 highest values."""
    api_values = [80.0, 40.0, 90.0, 30.0]

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=api_values)
        pipeline = _make_pipeline(tmp_path, sequences=SEQS[:4])
        pipeline.add_prediction("temberture", prediction_type="tm")
        pipeline.add_filter(
            RankingFilter("tm", n=2, ascending=False),  # keep 2 highest tm
            stage_name="rank_filter",
            depends_on=["temberture_predict"],
        )
        pipeline.run()

    df = pipeline.get_final_data()
    assert len(df) == 2, f"Expected 2 sequences after RankingFilter, got {len(df)}"
    # The 2 highest values are 80.0 and 90.0
    assert set(df["tm"].tolist()) == {80.0, 90.0}


def test_filter_stage_does_not_mutate_input_df(tmp_path):
    """Bug #8 fix: FilterStage must not mutate the input DataFrame in-place."""
    api_values = [80.0, 40.0, 90.0, 30.0]

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=api_values)
        pipeline = _make_pipeline(tmp_path, sequences=SEQS[:4])
        pipeline.add_prediction("temberture", prediction_type="tm")
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=60.0),
            stage_name="filter_tm",
            depends_on=["temberture_predict"],
        )
        pipeline.run()

    # The prediction stage's stored data should still have all 4 rows
    pred_df = pipeline._stage_data.get("temberture_predict")
    assert pred_df is not None
    assert (
        len(pred_df) == 4
    ), "Prediction stage data must not be mutated by downstream FilterStage"
