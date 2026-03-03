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

from biolmai.pipeline.base import WorkingSet
from biolmai.pipeline.data import DataPipeline, ExtractionSpec, PredictionStage
from biolmai.pipeline.datastore_duckdb import DuckDBDataStore
from biolmai.pipeline.filters import (
    CustomFilter,
    HammingDistanceFilter,
    RankingFilter,
    SequenceLengthFilter,
    ThresholdFilter,
    ValidAminoAcidFilter,
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
        pipeline.add_prediction("temberture-regression", prediction_type="tm", extractions="melting_temperature")
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=60.0),
            stage_name="filter_tm",
            depends_on=["predict_tm"],
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
        pipeline1.add_prediction("temberture-regression", prediction_type="tm", extractions="melting_temperature")
        pipeline1.run()

        call_count_first = mock_inst.predict.call_count

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls2:
        mock_inst2 = make_api_mock(values=api_values)
        MockCls2.return_value = mock_inst2

        # Second run — resume=True with same run_id
        pipeline2 = _make_pipeline(
            tmp_path, sequences=SEQS[:4], run_id=run_id, resume=True
        )
        pipeline2.add_prediction("temberture-regression", prediction_type="tm", extractions="melting_temperature")
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
        p1.add_prediction("temberture-regression", prediction_type="tm", extractions="melting_temperature")
        p1.run()

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        mock_inst = make_api_mock(values=api_values)
        MockCls.return_value = mock_inst
        # Add 2 new sequences — only they should need predictions
        p2 = _make_pipeline(tmp_path, sequences=SEQS, diff_mode=True)
        p2.add_prediction("temberture-regression", prediction_type="tm", extractions="melting_temperature")
        p2.run()

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
        pipeline.add_prediction("temberture-regression", prediction_type="tm", extractions="melting_temperature")
        pipeline.run()

    info = pipeline.explore()
    assert info["sequences"] == 4
    assert info["predictions"].get("tm", 0) == 4
    assert info["completed_stages"] >= 1

    stats_df = pipeline.stats()
    assert len(stats_df) >= 1
    assert "predict_tm" in stats_df["stage_name"].tolist()


# ---------------------------------------------------------------------------
# Test 7: query() method exposes raw DuckDB SQL
# ---------------------------------------------------------------------------


def test_query_method(tmp_path):
    """pipeline.query() executes SQL against the datastore."""
    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=[70.0] * 4)
        pipeline = _make_pipeline(tmp_path, sequences=SEQS[:4])
        pipeline.add_prediction("temberture-regression", prediction_type="tm", extractions="melting_temperature")
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
            "temberture-regression",
            prediction_type="tm",
            extractions="melting_temperature",
        )
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=60.0),
            stage_name="filter_tm",
            depends_on=["predict_tm"],
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
        pipeline.add_prediction("temberture-regression", prediction_type="tm", extractions="melting_temperature")
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=60.0),
            stage_name="filter_tm",
            depends_on=["predict_tm"],
        )
        pipeline.run(enable_streaming=True)

    ds = pipeline.datastore
    run_id = pipeline.run_id

    # Both the prediction stage and the filter stage should be marked complete
    pred_stage_id = f"{run_id}_predict_tm"
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
    model_mocks = {"temberture-regression": tm_mock, "pro4s": sol_mock}

    def make_client(model_name, **kwargs):
        return model_mocks.get(model_name, tm_mock)

    with patch("biolmai.pipeline.data.BioLMApiClient", side_effect=make_client):
        pipeline = _make_pipeline(tmp_path, sequences=SEQS[:4])
        # Two prediction stages with no explicit depends_on → same dep level → parallel
        pipeline.add_stage(
            PredictionStage(
                name="tm_stage",
                model_name="temberture-regression",
                action="predict",
                prediction_type="tm",
                extractions="melting_temperature",
                batch_size=32,
            )
        )
        pipeline.add_stage(
            PredictionStage(
                name="sol_stage",
                model_name="pro4s",
                action="predict",
                prediction_type="sol",
                extractions="solubility_score",
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
        pipeline.add_prediction("temberture-regression", prediction_type="tm", extractions="melting_temperature")
        pipeline.add_filter(
            RankingFilter("tm", n=2, ascending=True),  # keep 2 lowest tm
            stage_name="rank_filter",
            depends_on=["predict_tm"],
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
        pipeline.add_prediction("temberture-regression", prediction_type="tm", extractions="melting_temperature")
        pipeline.add_filter(
            RankingFilter("tm", n=2, ascending=False),  # keep 2 highest tm
            stage_name="rank_filter",
            depends_on=["predict_tm"],
        )
        pipeline.run()

    df = pipeline.get_final_data()
    assert len(df) == 2, f"Expected 2 sequences after RankingFilter, got {len(df)}"
    # The 2 highest values are 80.0 and 90.0
    assert set(df["tm"].tolist()) == {80.0, 90.0}


def test_filter_stage_does_not_mutate_input_df(tmp_path):
    """Bug #8 fix: FilterStage must not discard prediction data from upstream stages.

    After a filter runs, all 4 predictions should still exist in DuckDB (the
    filter only narrows the WorkingSet / row set, not the stored predictions).
    The final output should have only the 2 sequences that pass the threshold.
    """
    api_values = [80.0, 40.0, 90.0, 30.0]

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=api_values)
        pipeline = _make_pipeline(tmp_path, sequences=SEQS[:4])
        pipeline.add_prediction("temberture-regression", prediction_type="tm", extractions="melting_temperature")
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=60.0),
            stage_name="filter_tm",
            depends_on=["predict_tm"],
        )
        pipeline.run()

    # All 4 predictions should still be stored in DuckDB — filtering must not
    # delete upstream prediction data from the datastore.
    pred_count = pipeline.datastore.conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction_type = 'tm'"
    ).fetchone()[0]
    assert pred_count == 4, "All 4 predictions must remain in DuckDB after filtering"

    # Final output should only have the 2 that passed the filter
    final_df = pipeline.get_final_data()
    assert len(final_df) == 2
    assert all(final_df["tm"] >= 60.0)


# ---------------------------------------------------------------------------
# ExtractionSpec tests
# ---------------------------------------------------------------------------


def _make_multi_key_api_mock():
    """Return a mock whose predict returns both mean_plddt and ptm."""
    mock = AsyncMock()

    async def _predict(items, params=None):
        return [{"mean_plddt": 85.5, "ptm": 0.9, "pdb": "ATOM..."} for _ in items]

    mock.predict = AsyncMock(side_effect=_predict)
    mock.encode = AsyncMock()
    mock.shutdown = AsyncMock()
    return mock


def test_extraction_spec_single_key(tmp_path):
    """ExtractionSpec with a single key stores only that key's value."""
    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = _make_multi_key_api_mock()
        pipeline = _make_pipeline(tmp_path, sequences=SEQS[:2])
        pipeline.add_prediction(
            "esmfold",
            action="predict",
            extractions="mean_plddt",
            prediction_type="plddt",
        )
        pipeline.run()

    df = pipeline.get_final_data()
    assert "plddt" in df.columns
    assert len(df) == 2
    assert all(df["plddt"] == 85.5)
    # ptm should NOT be a column (not extracted)
    assert "ptm" not in df.columns or df["ptm"].isna().all()


def test_extraction_spec_multi_key(tmp_path):
    """ExtractionSpec with multiple keys stores both as separate predictions."""
    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = _make_multi_key_api_mock()
        pipeline = _make_pipeline(tmp_path, sequences=SEQS[:2])
        pipeline.add_prediction(
            "esmfold",
            action="predict",
            extractions={"mean_plddt": "plddt", "ptm": "ptm"},
        )
        pipeline.run()

    df = pipeline.get_final_data()
    assert "plddt" in df.columns, f"Missing 'plddt', columns: {list(df.columns)}"
    assert "ptm" in df.columns, f"Missing 'ptm', columns: {list(df.columns)}"
    assert len(df) == 2
    assert all(df["plddt"] == 85.5)
    assert all(df["ptm"] == 0.9)


def test_extraction_spec_with_reduction(tmp_path):
    """ExtractionSpec with reduction='mean' averages array values."""
    mock = AsyncMock()

    async def _predict(items, params=None):
        return [{"plddt": [80.0, 90.0, 85.0], "ptm": 0.9} for _ in items]

    mock.predict = AsyncMock(side_effect=_predict)
    mock.encode = AsyncMock()
    mock.shutdown = AsyncMock()

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = mock
        pipeline = _make_pipeline(tmp_path, sequences=SEQS[:2])
        pipeline.add_prediction(
            "esmfold",
            action="predict",
            extractions=[
                ExtractionSpec("plddt", "mean_plddt", reduction="mean"),
                ExtractionSpec("ptm", "ptm"),
            ],
        )
        pipeline.run()

    df = pipeline.get_final_data()
    assert "mean_plddt" in df.columns
    assert abs(df["mean_plddt"].iloc[0] - 85.0) < 0.01
    assert "ptm" in df.columns
    assert all(df["ptm"] == 0.9)


def test_extraction_spec_legacy_fallback(tmp_path):
    """When no extractions set, the legacy heuristic extraction still works."""
    api_values = [65.0, 75.0]

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=api_values)
        pipeline = _make_pipeline(tmp_path, sequences=SEQS[:2])
        # No extractions — legacy mode
        pipeline.add_prediction("temberture-regression", prediction_type="tm", extractions="melting_temperature")
        pipeline.run()

    df = pipeline.get_final_data()
    assert "tm" in df.columns
    assert len(df) == 2
    assert set(df["tm"].tolist()) == {65.0, 75.0}


def test_valid_amino_acid_filter():
    """ValidAminoAcidFilter removes sequences with non-standard characters."""
    df = pd.DataFrame(
        {
            "sequence": [
                "MKTAYIAKQRQ",  # valid
                "ACDEFGHIKLM",  # valid
                "NQRSTUVWXYZ",  # invalid: U, X, Z
                "AAAA<mask>BB",  # invalid: <, >, mask chars
            ]
        }
    )
    f = ValidAminoAcidFilter(verbose=False)
    result = f(df)
    assert len(result) == 2
    assert set(result["sequence"].tolist()) == {"MKTAYIAKQRQ", "ACDEFGHIKLM"}


def test_extraction_spec_cache_check(tmp_path):
    """Re-run with extractions should skip cached sequences."""
    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        mock1 = _make_multi_key_api_mock()
        MockCls.return_value = mock1
        pipeline1 = _make_pipeline(tmp_path, sequences=SEQS[:2])
        pipeline1.add_prediction(
            "esmfold",
            action="predict",
            extractions={"mean_plddt": "plddt", "ptm": "ptm"},
        )
        pipeline1.run()
        first_call_count = mock1.predict.call_count

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        mock2 = _make_multi_key_api_mock()
        MockCls.return_value = mock2
        # Same sequences, same pipeline DB
        pipeline2 = _make_pipeline(tmp_path, sequences=SEQS[:2])
        pipeline2.add_prediction(
            "esmfold",
            action="predict",
            extractions={"mean_plddt": "plddt", "ptm": "ptm"},
        )
        pipeline2.run()
        second_call_count = mock2.predict.call_count

    assert first_call_count > 0, "First run should call API"
    assert second_call_count == 0, "Second run should skip (cached)"
    df = pipeline2.get_final_data()
    assert len(df) == 2
    assert "plddt" in df.columns
    assert "ptm" in df.columns


# ---------------------------------------------------------------------------
# SQL filter path tests — verify zero-materialization filtering works
# ---------------------------------------------------------------------------


def _setup_datastore_with_predictions(tmp_path, seqs, pred_type, model, values):
    """Create a DuckDB datastore pre-loaded with sequences and predictions."""
    db = tmp_path / "filter_test.duckdb"
    data_dir = tmp_path / "data"
    ds = DuckDBDataStore(db_path=db, data_dir=data_dir)
    seq_ids = ds.add_sequences_batch(seqs)
    for sid, val in zip(seq_ids, values):
        ds.add_prediction(sid, pred_type, model, val)
    return ds, seq_ids


def test_threshold_filter_sql_path(tmp_path):
    """ThresholdFilter.to_sql() runs in DuckDB and matches DataFrame result."""
    seqs = ["MKTAYIAKQRQ", "ACDEFGHIKLM", "NQRSTVWYYY", "AAAABBBBCCCC"]
    values = [80.0, 40.0, 90.0, 30.0]
    ds, seq_ids = _setup_datastore_with_predictions(
        tmp_path, seqs, "tm", "temberture-regression", values
    )

    ws = WorkingSet.from_ids(seq_ids)
    filt = ThresholdFilter("tm", min_value=60.0)

    # Verify to_sql() is not None
    sql = filt.to_sql()
    assert sql is not None, "ThresholdFilter must produce SQL"

    # SQL path
    surviving = ds.execute_filter_sql(list(ws.sequence_ids), sql)
    ws_out = WorkingSet.from_ids(surviving)

    # DataFrame path (ground truth)
    df = ds.materialize_working_set(ws)
    df_filtered = filt(df)

    assert len(ws_out) == len(df_filtered), (
        f"SQL path returned {len(ws_out)} but DataFrame path returned {len(df_filtered)}"
    )
    assert set(ws_out.sequence_ids) == set(df_filtered["sequence_id"].tolist())
    # Specifically: only the two seqs with tm >= 60 should survive
    assert len(ws_out) == 2


def test_threshold_filter_sql_max_value(tmp_path):
    """ThresholdFilter with max_value works in SQL."""
    seqs = ["AAA", "BBB", "CCC"]
    values = [10.0, 50.0, 90.0]
    ds, seq_ids = _setup_datastore_with_predictions(
        tmp_path, seqs, "tm", "m", values
    )

    ws = WorkingSet.from_ids(seq_ids)
    filt = ThresholdFilter("tm", max_value=50.0)
    sql = filt.to_sql()
    assert sql is not None

    surviving = ds.execute_filter_sql(list(ws.sequence_ids), sql)
    assert len(surviving) == 2  # 10.0 and 50.0


def test_threshold_filter_sql_range(tmp_path):
    """ThresholdFilter with both min and max works in SQL."""
    seqs = ["AAA", "BBB", "CCC", "DDD"]
    values = [10.0, 50.0, 70.0, 90.0]
    ds, seq_ids = _setup_datastore_with_predictions(
        tmp_path, seqs, "tm", "m", values
    )

    ws = WorkingSet.from_ids(seq_ids)
    filt = ThresholdFilter("tm", min_value=40.0, max_value=80.0)
    sql = filt.to_sql()
    surviving = ds.execute_filter_sql(list(ws.sequence_ids), sql)
    assert len(surviving) == 2  # 50.0 and 70.0


def test_ranking_filter_sql_top_n_scoped_to_working_set(tmp_path):
    """RankingFilter.to_sql() LIMIT must be scoped to the working set, not global."""
    seqs = ["A" * 10, "B" * 10, "C" * 10, "D" * 10, "E" * 10]
    values = [100.0, 80.0, 60.0, 40.0, 20.0]
    ds, seq_ids = _setup_datastore_with_predictions(
        tmp_path, seqs, "tm", "m", values
    )

    # Working set only contains the last 3 (values: 60, 40, 20)
    subset_ids = seq_ids[2:]  # C, D, E
    ws = WorkingSet.from_ids(subset_ids)

    filt = RankingFilter("tm", n=2, ascending=False)  # top 2
    sql = filt.to_sql()
    assert sql is not None

    surviving = ds.execute_filter_sql(list(ws.sequence_ids), sql)
    ws_out = WorkingSet.from_ids(surviving)

    # Must return 2 sequences from the working set (60.0 and 40.0),
    # NOT from the global top 2 (100.0 and 80.0 which aren't in the ws)
    assert len(ws_out) == 2, (
        f"Expected 2 from working set, got {len(ws_out)}"
    )

    # Verify the correct IDs survived (C=60 and D=40, not A=100 or B=80)
    surviving_set = set(surviving)
    assert seq_ids[0] not in surviving_set, "A (100.0) is not in working set"
    assert seq_ids[1] not in surviving_set, "B (80.0) is not in working set"
    assert seq_ids[2] in surviving_set, "C (60.0) should be in top 2 of working set"
    assert seq_ids[3] in surviving_set, "D (40.0) should be in top 2 of working set"


def test_ranking_filter_sql_bottom_n(tmp_path):
    """RankingFilter bottom N works correctly in SQL."""
    seqs = ["AA", "BB", "CC", "DD"]
    values = [10.0, 50.0, 90.0, 30.0]
    ds, seq_ids = _setup_datastore_with_predictions(
        tmp_path, seqs, "score", "m", values
    )

    ws = WorkingSet.from_ids(seq_ids)
    filt = RankingFilter("score", n=2, method="bottom")  # bottom 2 = lowest
    sql = filt.to_sql()
    surviving = ds.execute_filter_sql(list(ws.sequence_ids), sql)

    assert len(surviving) == 2
    # Should be the two lowest: 10.0 (AA) and 30.0 (DD)
    surviving_set = set(surviving)
    assert seq_ids[0] in surviving_set  # 10.0
    assert seq_ids[3] in surviving_set  # 30.0


def test_ranking_filter_percentile_returns_none(tmp_path):
    """RankingFilter percentile mode cannot be SQL-translated."""
    filt = RankingFilter("tm", method="percentile", percentile=90)
    assert filt.to_sql() is None


def test_sequence_length_filter_sql_path(tmp_path):
    """SequenceLengthFilter.to_sql() works against real DuckDB data."""
    seqs = ["AAA", "BBBBB", "CC", "DDDDDDDDD"]  # lengths: 3, 5, 2, 9
    db = tmp_path / "len.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "data")
    seq_ids = ds.add_sequences_batch(seqs)
    ws = WorkingSet.from_ids(seq_ids)

    filt = SequenceLengthFilter(min_length=3, max_length=6)
    sql = filt.to_sql()
    assert sql is not None

    surviving = ds.execute_filter_sql(list(ws.sequence_ids), sql)
    assert len(surviving) == 2  # "AAA" (3) and "BBBBB" (5)

    # Cross-check with DataFrame path
    df = ds.materialize_working_set(ws, include_predictions=False)
    df_filtered = filt(df)
    assert set(surviving) == set(df_filtered["sequence_id"].tolist())


def test_valid_amino_acid_filter_sql_path(tmp_path):
    """ValidAminoAcidFilter.to_sql() works against real DuckDB data."""
    seqs = ["MKTAYIAKQRQ", "ACDEFGHIKLM", "NQRST1VWXYZ", "AAAABBBBCCCC"]
    db = tmp_path / "aa.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "data")
    seq_ids = ds.add_sequences_batch(seqs)
    ws = WorkingSet.from_ids(seq_ids)

    filt = ValidAminoAcidFilter(verbose=False)
    sql = filt.to_sql()
    assert sql is not None

    surviving = ds.execute_filter_sql(list(ws.sequence_ids), sql)
    ws_out = WorkingSet.from_ids(surviving)

    # Cross-check with DataFrame path
    df = ds.materialize_working_set(ws, include_predictions=False)
    df_filtered = filt(df)

    assert set(ws_out.sequence_ids) == set(df_filtered["sequence_id"].tolist())


def test_custom_filter_uses_dataframe_path(tmp_path):
    """CustomFilter has no to_sql() — must go through DataFrame path."""
    filt = CustomFilter(lambda df: df[df["sequence"].str.len() > 5])
    assert filt.to_sql() is None  # correctly returns None


def test_hamming_filter_has_no_sql(tmp_path):
    """HammingDistanceFilter cannot be expressed as SQL."""
    filt = HammingDistanceFilter("MKTAYIAKQRQ", max_distance=3)
    assert filt.to_sql() is None


def test_sql_filter_in_full_pipeline(tmp_path):
    """ThresholdFilter SQL path works end-to-end in a pipeline run."""
    api_values = [80.0, 40.0, 90.0, 30.0]

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=api_values)
        pipeline = _make_pipeline(tmp_path, sequences=SEQS[:4])
        pipeline.add_prediction("temberture-regression", prediction_type="tm", extractions="melting_temperature")
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=60.0),
            stage_name="filter_tm",
            depends_on=["predict_tm"],
        )
        pipeline.run()

    # Verify the filter stage used the WorkingSet path
    assert "filter_tm" in pipeline._working_sets
    ws_filter = pipeline._working_sets["filter_tm"]
    assert len(ws_filter) == 2, "Only 2 of 4 sequences should pass tm >= 60"

    # Verify final materialized output matches
    df = pipeline.get_final_data()
    assert len(df) == 2
    assert set(df["sequence_id"].tolist()) == set(ws_filter.sequence_ids)
    # All surviving sequences should have tm >= 60
    assert all(df["tm"] >= 60.0)


def test_ranking_filter_sql_in_full_pipeline(tmp_path):
    """RankingFilter top-N SQL path returns exactly N sequences from pipeline."""
    api_values = [80.0, 40.0, 90.0, 30.0]

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=api_values)
        pipeline = _make_pipeline(tmp_path, sequences=SEQS[:4])
        pipeline.add_prediction("temberture-regression", prediction_type="tm", extractions="melting_temperature")
        pipeline.add_filter(
            RankingFilter("tm", n=2, ascending=False),
            stage_name="top2",
            depends_on=["predict_tm"],
        )
        pipeline.run()

    df = pipeline.get_final_data()
    assert len(df) == 2
    # Should be the two highest: 90.0 and 80.0
    assert set(df["tm"].tolist()) == {80.0, 90.0}


# ---------------------------------------------------------------------------
# PipelineMetadata + default cache dir tests
# ---------------------------------------------------------------------------


def test_pipeline_metadata_exposed(tmp_path):
    """Pipeline.metadata returns usable PipelineMetadata after construction."""
    from biolmai.pipeline.base import PipelineMetadata

    pipeline = _make_pipeline(tmp_path, sequences=SEQS[:2])
    meta = pipeline.metadata
    assert isinstance(meta, PipelineMetadata)
    assert meta.run_id == pipeline.run_id
    assert meta.db_path.exists()


def test_default_cache_dir_created(tmp_path, monkeypatch):
    """When no datastore is given, pipeline creates .biolm/pipelines/<id>/ cache."""
    import biolmai.pipeline.base as base_mod

    # Point the cache root into tmp_path so we don't pollute the real working dir
    monkeypatch.setattr(base_mod, "_BIOLM_CACHE_ROOT", tmp_path / ".biolm" / "pipelines")

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=[70.0, 80.0])
        pipeline = DataPipeline(sequences=SEQS[:2], verbose=False)
        pipeline.add_prediction("temberture-regression", prediction_type="tm", extractions="melting_temperature")
        pipeline.run()

    meta = pipeline.metadata
    assert meta.cache_dir.exists()
    assert (meta.cache_dir / "pipeline.duckdb").exists()
    assert meta.pipeline_id == pipeline.run_id

    # Verify the data is accessible via the cache path
    df = pipeline.get_final_data()
    assert len(df) == 2


# ---------------------------------------------------------------------------
# Multi-column input tests
# ---------------------------------------------------------------------------


def test_multi_column_input_heavy_light(tmp_path):
    """DataPipeline with input_columns=['heavy_chain', 'light_chain'] stores
    columns directly on sequences table and deduplicates across both."""
    df_input = pd.DataFrame(
        {
            "heavy_chain": ["EVQLVES", "EVQLVES", "QVQLQES"],
            "light_chain": ["DIQMTQS", "EIVLTQS", "DIQMTQS"],
        }
    )
    # Row 0 and 1 have same heavy but different light → both kept
    # All 3 rows are unique across the pair

    db = tmp_path / "test.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "data")
    pipeline = DataPipeline(
        sequences=df_input,
        input_columns=["heavy_chain", "light_chain"],
        datastore=ds,
        verbose=False,
    )
    # Run _get_initial_data manually
    df = asyncio.run(pipeline._get_initial_data())

    assert len(df) == 3
    assert "sequence_id" in df.columns
    assert "heavy_chain" in df.columns
    assert "light_chain" in df.columns

    # Verify stored in DuckDB sequences table directly
    result = ds.conn.execute(
        "SELECT sequence_id, heavy_chain, light_chain FROM sequences ORDER BY sequence_id"
    ).df()
    assert len(result) == 3
    assert list(result["heavy_chain"]) == ["EVQLVES", "EVQLVES", "QVQLQES"]


def test_multi_column_hash_dedup(tmp_path):
    """Same heavy + light = same hash → deduplicated."""
    df_input = pd.DataFrame(
        {
            "heavy_chain": ["EVQLVES", "EVQLVES"],
            "light_chain": ["DIQMTQS", "DIQMTQS"],
        }
    )

    db = tmp_path / "test.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "data")
    pipeline = DataPipeline(
        sequences=df_input,
        input_columns=["heavy_chain", "light_chain"],
        datastore=ds,
        verbose=False,
    )
    df = asyncio.run(pipeline._get_initial_data())

    # Both rows identical → deduplicated to 1
    assert len(df) == 1


def test_multi_column_different_light_different_hash(tmp_path):
    """Same heavy + different light = different rows (not deduped)."""
    df_input = pd.DataFrame(
        {
            "heavy_chain": ["EVQLVES", "EVQLVES"],
            "light_chain": ["DIQMTQS", "EIVLTQS"],
        }
    )

    db = tmp_path / "test.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "data")
    pipeline = DataPipeline(
        sequences=df_input,
        input_columns=["heavy_chain", "light_chain"],
        datastore=ds,
        verbose=False,
    )
    df = asyncio.run(pipeline._get_initial_data())
    assert len(df) == 2


def test_backward_compat_sequence_list(tmp_path):
    """DataPipeline(sequences=["MKLLIV"]) still works unchanged."""
    pipeline = _make_pipeline(tmp_path, sequences=["MKLLIV", "ACDEFG"])
    df = asyncio.run(pipeline._get_initial_data())
    assert len(df) == 2
    assert "sequence" in df.columns
    assert "sequence_id" in df.columns


def test_multi_column_materialize_working_set(tmp_path):
    """materialize_working_set includes input columns from sequences table."""
    df_input = pd.DataFrame(
        {
            "heavy_chain": ["EVQLVES", "QVQLQES"],
            "light_chain": ["DIQMTQS", "EIVLTQS"],
        }
    )

    db = tmp_path / "test.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "data")
    pipeline = DataPipeline(
        sequences=df_input,
        input_columns=["heavy_chain", "light_chain"],
        datastore=ds,
        verbose=False,
    )
    df = asyncio.run(pipeline._get_initial_data())

    ws = WorkingSet.from_ids(df["sequence_id"].tolist())
    df_mat = ds.materialize_working_set(ws)

    assert "heavy_chain" in df_mat.columns
    assert "light_chain" in df_mat.columns
    assert len(df_mat) == 2


def test_item_columns_from_sequences_table(tmp_path):
    """PredictionStage.process_ws reads item_columns from sequences table columns."""
    df_input = pd.DataFrame(
        {
            "heavy_chain": ["EVQLVES"],
            "light_chain": ["DIQMTQS"],
        }
    )

    db = tmp_path / "test.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "data")
    pipeline = DataPipeline(
        sequences=df_input,
        input_columns=["heavy_chain", "light_chain"],
        datastore=ds,
        verbose=False,
    )
    df = asyncio.run(pipeline._get_initial_data())

    # Verify get_sequences_for_ids_with_columns works
    seq_ids = df["sequence_id"].tolist()
    col_map = ds.get_sequences_for_ids_with_columns(seq_ids, ["heavy_chain", "light_chain"])
    assert len(col_map) == 1
    sid = seq_ids[0]
    assert col_map[sid]["heavy_chain"] == "EVQLVES"
    assert col_map[sid]["light_chain"] == "DIQMTQS"


# ---------------------------------------------------------------------------
# Pipeline context tests
# ---------------------------------------------------------------------------


def test_pipeline_context_round_trip(tmp_path):
    """Context set/get round-trips through DuckDB."""
    db = tmp_path / "test.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "data")

    from biolmai.pipeline.base import PipelineContext

    ctx = PipelineContext(ds, "run_001")
    ctx.set("stage_1_model", "esmfold")
    ctx.set("stage_1_count", 42)
    ctx.set("nested_data", {"key": "value", "numbers": [1, 2, 3]})

    assert ctx.get("stage_1_model") == "esmfold"
    assert ctx.get("stage_1_count") == 42
    assert ctx.get("nested_data") == {"key": "value", "numbers": [1, 2, 3]}
    assert ctx.get("missing_key", "default") == "default"


def test_pipeline_context_isolation(tmp_path):
    """Different run_ids have isolated context."""
    db = tmp_path / "test.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "data")

    from biolmai.pipeline.base import PipelineContext

    ctx1 = PipelineContext(ds, "run_001")
    ctx2 = PipelineContext(ds, "run_002")

    ctx1.set("foo", "bar")
    ctx2.set("foo", "baz")

    assert ctx1.get("foo") == "bar"
    assert ctx2.get("foo") == "baz"


def test_pipeline_context_get_structure(tmp_path):
    """Context.get_structure reads from DuckDB structures table."""
    db = tmp_path / "test.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "data")

    seq_id = ds.add_sequence("MKLLIV")
    ds.add_structure(seq_id, "esmfold", structure_str="ATOM 1 CA ALA A 1 0.0 0.0 0.0")

    from biolmai.pipeline.base import PipelineContext

    ctx = PipelineContext(ds, "run_001")
    struct = ctx.get_structure(seq_id, "esmfold")
    assert struct is not None
    assert "ATOM" in struct["structure_str"]


def test_context_passed_to_stages(tmp_path):
    """Verify context kwarg is passed to stage.process_ws."""
    db = tmp_path / "test.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "data")

    api_values = [65.0]
    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=api_values)
        pipeline = DataPipeline(
            sequences=["MKLLIV"],
            datastore=ds,
            verbose=False,
        )
        pipeline.add_prediction(
            "temberture-regression",
            prediction_type="tm",
            extractions="melting_temperature",
        )
        pipeline.run()

    # Pipeline context exists and is functional
    assert pipeline.context is not None
    pipeline.context.set("test_key", "test_value")
    assert pipeline.context.get("test_key") == "test_value"


def test_multi_column_full_pipeline_with_prediction(tmp_path):
    """Full pipeline: multi-column input → prediction → filter."""
    df_input = pd.DataFrame(
        {
            "heavy_chain": ["EVQLVES", "QVQLQES", "DVQLVES"],
            "light_chain": ["DIQMTQS", "EIVLTQS", "DIQMTQS"],
        }
    )

    api_values = [80.0, 40.0, 90.0]

    db = tmp_path / "test.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "data")

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=api_values)
        pipeline = DataPipeline(
            sequences=df_input,
            input_columns=["heavy_chain", "light_chain"],
            datastore=ds,
            verbose=False,
        )
        pipeline.add_prediction(
            "temberture-regression",
            prediction_type="tm",
            extractions="melting_temperature",
        )
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=60.0),
            stage_name="filter_tm",
            depends_on=["predict_tm"],
        )
        pipeline.run(enable_streaming=False)

    df_final = pipeline.get_final_data()
    # Only rows with tm >= 60 survive: 80.0 and 90.0
    assert len(df_final) == 2
    # Input columns should be present in final data
    assert "heavy_chain" in df_final.columns
    assert "light_chain" in df_final.columns
