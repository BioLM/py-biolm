"""
Tests for pipeline definition metadata store (pipeline_def.py).

Covers:
- Stage roundtrip serialization (to_spec → stage_from_spec)
- Filter roundtrip serialization (to_spec → filter_from_spec)
- Pipeline reconstruction via from_db()
- Content-hash dedup (same stages → same definition_id)
- Cross-pipeline column collision detection
- Input schema mismatch error + warning
- CustomFilter raises on reconstruct
- Definition written to DB on run
- sequences=None with resume=True
"""
import pytest
pytest.importorskip("duckdb")

import asyncio
import json
import tempfile
import warnings
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from biolmai.pipeline.data import (
    ClusteringStage,
    DataPipeline,
    EmbeddingSpec,
    ExtractionSpec,
    FilterStage,
    PredictionStage,
    _ResolvedExtraction,
)
from biolmai.pipeline.datastore_duckdb import DuckDBDataStore
from biolmai.pipeline.filters import (
    ConservedResidueFilter,
    CustomFilter,
    DiversitySamplingFilter,
    HammingDistanceFilter,
    RankingFilter,
    SequenceLengthFilter,
    ThresholdFilter,
    ValidAminoAcidFilter,
)
from biolmai.pipeline.pipeline_def import (
    _pipeline_def_hash,
    filter_from_spec,
    pipeline_from_definition,
    stage_from_spec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path) -> DuckDBDataStore:
    return DuckDBDataStore(str(tmp_path / "pipeline.duckdb"), str(tmp_path / "data"))


def _make_prediction_stage(name="tm_pred", depends_on=None) -> PredictionStage:
    return PredictionStage(
        name=name,
        model_name="temberture-regression",
        action="predict",
        extractions="prediction",
        columns="tm",
        batch_size=16,
        max_concurrent=3,
        depends_on=depends_on or [],
    )


def _make_filter_stage(name="filter_tm", depends_on=None) -> FilterStage:
    return FilterStage(
        name=name,
        filter_func=ThresholdFilter("tm", min_value=50.0),
        depends_on=depends_on or ["tm_pred"],
    )


# ---------------------------------------------------------------------------
# 1. PredictionStage roundtrip
# ---------------------------------------------------------------------------

def test_prediction_stage_roundtrip():
    stage = PredictionStage(
        name="embed",
        model_name="esm2-8m",
        action="encode",
        extractions=None,
        embedding_extractor=EmbeddingSpec(key="embeddings", layer=33, reduction="mean"),
        batch_size=8,
        max_concurrent=2,
        item_columns=None,
        skip_on_error=True,
        depends_on=[],
    )
    spec = stage.to_spec()
    assert spec["type"] == "PredictionStage"
    assert spec["model_name"] == "esm2-8m"
    assert spec["action"] == "encode"
    assert spec["embedding_extractor"]["key"] == "embeddings"
    assert spec["embedding_extractor"]["layer"] == 33
    assert spec["embedding_extractor"]["reduction"] == "mean"
    assert spec["skip_on_error"] is True

    rebuilt = stage_from_spec(spec)
    assert rebuilt.name == "embed"
    assert rebuilt.model_name == "esm2-8m"
    assert rebuilt.action == "encode"
    assert isinstance(rebuilt._embedding_extractor, EmbeddingSpec)
    assert rebuilt._embedding_extractor.key == "embeddings"
    assert rebuilt.skip_on_error is True


def test_prediction_stage_roundtrip_predict():
    stage = _make_prediction_stage()
    spec = stage.to_spec()
    assert spec["type"] == "PredictionStage"
    assert len(spec["resolved"]) == 1
    assert spec["resolved"][0]["response_key"] == "prediction"
    assert spec["resolved"][0]["column"] == "tm"

    rebuilt = stage_from_spec(spec)
    assert rebuilt.name == "tm_pred"
    assert rebuilt._resolved[0].column == "tm"
    assert rebuilt._resolved[0].response_key == "prediction"


def test_prediction_stage_callable_extractor_raises():
    stage = PredictionStage(
        name="bad",
        model_name="esm2-8m",
        action="encode",
        embedding_extractor=lambda r: [(r.get("embeddings", []), None)],
    )
    with pytest.raises(NotImplementedError, match="callable embedding_extractor"):
        stage.to_spec()


# ---------------------------------------------------------------------------
# 2. All filter types roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filter_obj,check_attr", [
    (ThresholdFilter("tm", min_value=45.0, max_value=90.0, keep_na=True),
     lambda f: f.min_value == 45.0 and f.max_value == 90.0 and f.keep_na is True),
    (SequenceLengthFilter(min_length=10, max_length=500),
     lambda f: f.min_length == 10 and f.max_length == 500),
    (HammingDistanceFilter("MKLLIV", max_distance=3, normalize=True),
     lambda f: f.reference_sequence == "MKLLIV" and f.max_distance == 3 and f.normalize is True),
    (RankingFilter("tm", n=50, method="top"),
     lambda f: f.n == 50 and f.method == "top"),
    (RankingFilter("tm", method="percentile", percentile=90.0),
     lambda f: f.percentile == 90.0 and f.method == "percentile"),
    (ConservedResidueFilter({5: ["M", "L"], 10: ["K"]}, reference_length=200),
     lambda f: f.conserved_positions == {5: ["M", "L"], 10: ["K"]} and f.reference_length == 200),
    (DiversitySamplingFilter(100, method="random", random_seed=7),
     lambda f: f.n_samples == 100 and f.random_seed == 7),
    (ValidAminoAcidFilter(column="heavy_chain"),
     lambda f: f.column == "heavy_chain"),
])
def test_all_filter_types_roundtrip(filter_obj, check_attr):
    spec = filter_obj.to_spec()
    assert spec["type"] == type(filter_obj).__name__
    rebuilt = filter_from_spec(spec)
    assert type(rebuilt).__name__ == spec["type"]
    assert check_attr(rebuilt), f"Attribute check failed for {spec['type']}"


# ---------------------------------------------------------------------------
# 3. from_db roundtrip (mocked API)
# ---------------------------------------------------------------------------

def test_pipeline_from_db(tmp_path):
    db_path = tmp_path / "pipeline.duckdb"
    ds = _make_db(tmp_path)

    # Build and run a pipeline with mocked API
    mock_result = {"prediction": 55.0}
    with patch("biolmai.pipeline.data.BioLMApiClient") as MockClient:
        mock_client = MagicMock()
        mock_client.predict = AsyncMock(return_value=[mock_result, mock_result])
        mock_client.shutdown = AsyncMock()
        MockClient.return_value = mock_client

        pipeline = DataPipeline(
            sequences=["MKLLIV", "MKTAYIAKQ"],
            datastore=ds,
            verbose=False,
        )
        pipeline.add_stage(_make_prediction_stage())
        pipeline.add_stage(_make_filter_stage())
        pipeline.run(enable_streaming=False)

    # Reconstruct from DB
    ds2 = DuckDBDataStore(str(db_path), str(tmp_path / "data"))
    defn = ds2.load_pipeline_definition()
    assert defn is not None
    rebuilt = pipeline_from_definition(defn, datastore=ds2, verbose=False)

    assert len(rebuilt.stages) == 2
    stage0 = rebuilt.stages[0]
    stage1 = rebuilt.stages[1]
    assert isinstance(stage0, PredictionStage)
    assert stage0.model_name == "temberture-regression"
    assert isinstance(stage1, FilterStage)
    assert isinstance(stage1.filter_func, ThresholdFilter)
    assert stage1.filter_func.min_value == 50.0


# ---------------------------------------------------------------------------
# 4. Same stages → same definition_id (content hash dedup)
# ---------------------------------------------------------------------------

def test_same_stages_same_definition_id(tmp_path):
    specs_a = [_make_prediction_stage().to_spec()]
    specs_b = [_make_prediction_stage().to_spec()]
    id_a = _pipeline_def_hash("DataPipeline", None, specs_a)
    id_b = _pipeline_def_hash("DataPipeline", None, specs_b)
    assert id_a == id_b

    # Different stage → different hash
    stage_diff = PredictionStage(
        name="soluprot",
        model_name="soluprot",
        action="predict",
        extractions="soluble",
        columns="solubility",
    )
    specs_c = [stage_diff.to_spec()]
    id_c = _pipeline_def_hash("DataPipeline", None, specs_c)
    assert id_c != id_a


# ---------------------------------------------------------------------------
# 5. Cross-pipeline column error
# ---------------------------------------------------------------------------

def test_cross_pipeline_column_error(tmp_path):
    ds = _make_db(tmp_path)

    # First pipeline registers "tm" from temberture-regression
    pipeline1 = DataPipeline(
        sequences=["MKLLIV"],
        datastore=ds,
        verbose=False,
    )
    pipeline1.add_stage(_make_prediction_stage())
    # Persist the registry by manually calling save helper
    pipeline1._save_definition_and_register_columns()

    # Second pipeline tries to use column "tm" with a DIFFERENT model
    pipeline2 = DataPipeline(
        sequences=["MKLLIV"],
        datastore=ds,
        verbose=False,
    )
    conflicting_stage = PredictionStage(
        name="tm_from_other_model",
        model_name="other-tm-model",
        action="predict",
        extractions="tm_val",
        columns="tm",  # same column name!
    )
    with pytest.raises(ValueError, match="Column 'tm' was previously used by"):
        pipeline2.add_stage(conflicting_stage)


# ---------------------------------------------------------------------------
# 6. Input schema mismatch → error
# ---------------------------------------------------------------------------

def test_input_schema_mismatch_error(tmp_path):
    ds = _make_db(tmp_path)

    # First pipeline uses multi-column input (H/L antibody)
    df_hl = pd.DataFrame({
        "heavy_chain": ["EVQLVESGG"],
        "light_chain": ["DIQMTQ"],
    })
    pipeline1 = DataPipeline(
        sequences=df_hl,
        datastore=ds,
        input_columns=["heavy_chain", "light_chain"],
        verbose=False,
    )
    asyncio.run(pipeline1._get_initial_data())

    # Second pipeline tries single-sequence on same DB → error
    with pytest.raises(ValueError, match="multi-column"):
        pipeline2 = DataPipeline(
            sequences=["MKLLIV"],
            datastore=ds,
            verbose=False,
        )
        asyncio.run(pipeline2._get_initial_data())


# ---------------------------------------------------------------------------
# 7. Input schema mismatch → warning (H/L DB, no input_columns)
# ---------------------------------------------------------------------------

def test_input_schema_mismatch_warning(tmp_path):
    ds = _make_db(tmp_path)

    df_hl = pd.DataFrame({
        "heavy_chain": ["EVQLVESGG"],
        "light_chain": ["DIQMTQ"],
    })
    pipeline1 = DataPipeline(
        sequences=df_hl,
        datastore=ds,
        input_columns=["heavy_chain", "light_chain"],
        verbose=False,
    )
    asyncio.run(pipeline1._get_initial_data())

    # The error is raised (not a warning) because schema mismatch is unambiguous
    with pytest.raises(ValueError, match="multi-column"):
        pipeline2 = DataPipeline(
            sequences=["MKLLIV"],
            datastore=ds,
            verbose=False,
        )
        asyncio.run(pipeline2._get_initial_data())


# ---------------------------------------------------------------------------
# 8. CustomFilter raises on reconstruct
# ---------------------------------------------------------------------------

def test_custom_filter_raises_on_reconstruct():
    f = CustomFilter(func=lambda df: df, name="my_custom")
    # CustomFilter.to_spec() now raises NotImplementedError directly
    with pytest.raises(NotImplementedError, match="CustomFilter"):
        f.to_spec()


def test_filter_stage_with_custom_filter_raises_on_to_spec():
    stage = FilterStage(
        name="custom_filter_stage",
        filter_func=CustomFilter(func=lambda df: df, name="my_filter"),
    )
    # FilterStage.to_spec() propagates the NotImplementedError from CustomFilter.to_spec()
    with pytest.raises(NotImplementedError, match="CustomFilter"):
        stage.to_spec()


# ---------------------------------------------------------------------------
# 9. Definition written to DB on run
# ---------------------------------------------------------------------------

def test_definition_written_on_run(tmp_path):
    ds = _make_db(tmp_path)

    mock_result = {"prediction": 65.0}
    with patch("biolmai.pipeline.data.BioLMApiClient") as MockClient:
        mock_client = MagicMock()
        mock_client.predict = AsyncMock(return_value=[mock_result, mock_result, mock_result])
        mock_client.shutdown = AsyncMock()
        MockClient.return_value = mock_client

        pipeline = DataPipeline(
            sequences=["MKLLIV", "MKTAY", "ACDEF"],
            datastore=ds,
            verbose=False,
        )
        pipeline.add_stage(_make_prediction_stage())
        pipeline.run(enable_streaming=False)

    # pipeline_definitions must have exactly one row
    count = ds.conn.execute("SELECT COUNT(*) FROM pipeline_definitions").fetchone()[0]
    assert count == 1

    defn = ds.load_pipeline_definition()
    assert defn is not None
    assert defn["pipeline_type"] == "DataPipeline"
    stages = json.loads(defn["stages_json"])
    assert len(stages) == 1
    assert stages[0]["type"] == "PredictionStage"

    # prediction_column_registry must have the "tm" column registered
    entry = ds.get_column_registry_entry("tm")
    assert entry is not None
    assert entry["model_name"] == "temberture-regression"


# ---------------------------------------------------------------------------
# 10. sequences=None with resume=True
# ---------------------------------------------------------------------------

def test_sequences_none_with_resume(tmp_path):
    ds = _make_db(tmp_path)
    # Pre-populate sequences table
    ds.add_sequences_batch(["MKLLIV", "MKTAY"])

    pipeline = DataPipeline(
        sequences=None,
        datastore=ds,
        resume=True,
        verbose=False,
    )
    df = asyncio.run(pipeline._get_initial_data())
    assert len(df) == 2
    assert "sequence_id" in df.columns
    assert set(df["sequence"].tolist()) == {"MKLLIV", "MKTAY"}


def test_sequences_none_without_resume_raises():
    with pytest.raises(ValueError, match="sequences is required unless resume=True"):
        DataPipeline(sequences=None, datastore=":memory:", verbose=False)


# ---------------------------------------------------------------------------
# 11. ClusteringStage + GenerationStage roundtrip
# ---------------------------------------------------------------------------

def test_clustering_stage_roundtrip():
    stage = ClusteringStage(
        name="cluster",
        method="kmeans",
        n_clusters=5,
        similarity_metric="hamming",
        max_sample=1000,
    )
    spec = stage.to_spec()
    assert spec["type"] == "ClusteringStage"
    assert spec["n_clusters"] == 5
    assert spec["max_sample"] == 1000

    rebuilt = stage_from_spec(spec)
    assert rebuilt.name == "cluster"
    assert rebuilt.n_clusters == 5
    assert rebuilt.method == "kmeans"


def test_generation_stage_roundtrip():
    from biolmai.pipeline.generative import DirectGenerationConfig, GenerationStage

    config = DirectGenerationConfig(
        model_name="protein-mpnn",
        item_field="pdb",
        num_sequences=50,
        temperature=0.5,
        n_runs=2,
    )
    stage = GenerationStage(name="gen", config=config, deduplicate=True)
    spec = stage.to_spec()
    assert spec["type"] == "GenerationStage"
    assert len(spec["configs"]) == 1
    assert spec["configs"][0]["type"] == "DirectGenerationConfig"
    assert spec["configs"][0]["temperature"] == 0.5
    assert spec["configs"][0]["n_runs"] == 2

    rebuilt = stage_from_spec(spec)
    assert rebuilt.name == "gen"
    assert isinstance(rebuilt.configs[0], DirectGenerationConfig)
    assert rebuilt.configs[0].temperature == 0.5
