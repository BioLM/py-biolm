# BioLM Pipeline System

Multi-stage orchestration framework for protein sequence generation, prediction, and analysis.

## Features

- **Multi-column input**: Antibody H+L chains, multi-chain proteins — `sequence` column not required
- **Explicit extraction**: `ExtractionSpec` for predictions, `EmbeddingSpec` for embeddings — no heuristic guessing
- **DuckDB-native**: WorkingSet transport, SQL filters with zero materialization, columnar storage
- **Multi-model generation**: MPNN, ZymCTRL, DSM, ProGen2-OAS — all in one pipeline
- **Caching + resume**: DuckDB prediction cache, stage completion tracking, trickle new sequences
- **Pipeline context**: Inter-stage shared state backed by DuckDB

## Quick Start

```python
from biolmai.pipeline import DataPipeline
from biolmai.pipeline.filters import ThresholdFilter, RankingFilter

pipeline = DataPipeline(sequences=["MKLLIV...", "ACDEFG..."])

pipeline.add_prediction(
    "temberture-regression",
    prediction_type="tm",
    extractions="prediction",         # Required: which response key to read
)
pipeline.add_prediction(
    "soluprot",
    prediction_type="solubility",
    extractions="soluble",
)
pipeline.add_filter(RankingFilter("tm", n=10, ascending=False),
                    depends_on=["predict_tm", "predict_solubility"])

pipeline.run()
df = pipeline.get_final_data()
```

## Extraction

Every prediction stage requires `extractions=` (predict/score) or `embedding_extractor=` (encode).

### Predictions

```python
# String shorthand — response key = column name
extractions="prediction"

# Dict — response key → column name mapping
extractions={"mean_plddt": "plddt", "ptm": "ptm"}

# ExtractionSpec — with array reduction
from biolmai.pipeline.data import ExtractionSpec
extractions=[ExtractionSpec("plddt", "plddt", reduction="mean")]
```

### Embeddings

```python
from biolmai.pipeline.data import EmbeddingSpec

# Key lookup
embedding_extractor=EmbeddingSpec(key="seqcoding")

# Layer selection
embedding_extractor=EmbeddingSpec(key="embeddings", layer=33)

# Mean-pool per-token to single vector
embedding_extractor=EmbeddingSpec(key="embeddings", layer=33, reduction="mean")

# Custom callable: (dict) -> list[(np.ndarray, Optional[int])]
embedding_extractor=lambda r: [(np.array(r["my_key"]), None)]
```

## Multi-Column Input

```python
import pandas as pd

df = pd.DataFrame({
    "heavy_chain": ["EVQLVES...", "QVQLQES..."],
    "light_chain": ["DIQMTQS...", "SYELTQP..."],
})

pipeline = DataPipeline(
    sequences=df,
    input_columns=["heavy_chain", "light_chain"],
)

# Map columns to API fields
pipeline.add_prediction("temberture-regression",
    prediction_type="tm_heavy",
    extractions="prediction",
    item_columns={"sequence": "heavy_chain"},
)
```

- Columns stored directly on DuckDB `sequences` table
- Hash dedup across all input columns
- `get_final_data()` and `export_to_dataframe()` include all columns

## Generation Models

```python
from biolmai.pipeline import GenerativePipeline, DirectGenerationConfig

configs = [
    # Structure-conditioned (MPNN family)
    DirectGenerationConfig(model_name="protein-mpnn", item_field="pdb",
        structure_path="protein.pdb", params={"batch_size": 50, "temperature": 0.3}),

    # Sequence-conditioned (DSM)
    DirectGenerationConfig(model_name="dsm-650m-base", item_field="sequence",
        sequence="MKTAYIAK...", params={"num_sequences": 20, "temperature": 1.0}),

    # Enzyme (ZymCTRL) — conditioned on EC number
    DirectGenerationConfig(model_name="zymctrl", item_field="ec_number",
        sequence="3.1.1.101", params={"temperature": 1.0, "max_length": 300}),

    # Antibody (ProGen2-OAS) — seeded from VH
    DirectGenerationConfig(model_name="progen2-oas", item_field="context",
        sequence="EVQLVES", params={"temperature": 1.0, "max_length": 120}),
]

pipeline = GenerativePipeline(generation_configs=configs, deduplicate=True)
pipeline.add_prediction("temberture-regression", prediction_type="tm", extractions="prediction")
pipeline.run()
```

## Pipeline Context

```python
pipeline.context.set("experiment", "thermo_screen")
pipeline.context.get("experiment")          # → "thermo_screen"
pipeline.context.get_structure(seq_id)      # → structure from DuckDB
```

Backed by DuckDB, scoped per `run_id`, passed to stages via `kwargs["context"]`.

## Filters

| Filter | SQL-native | Description |
|---|---|---|
| `ThresholdFilter` | Yes | Min/max on prediction column |
| `RankingFilter` | Yes | Top-N / bottom-N by column |
| `SequenceLengthFilter` | Yes | Min/max sequence length |
| `ValidAminoAcidFilter` | Yes | Remove non-standard residues (any column) |
| `HammingDistanceFilter` | No | Distance to reference sequence |
| `ConservedResidueFilter` | No | Required residues at positions |
| `DiversitySamplingFilter` | No | Subsample for diversity |
| `CustomFilter` | No | Any callable `(df) -> df` |

SQL-native filters execute in DuckDB with zero DataFrame materialization.

## Architecture

```
sequences (list / CSV / FASTA / DataFrame)
       |
  DataPipeline._get_initial_data()
       | (adds sequence_ids, stores in DuckDB)
  BasePipeline.run_async()
       |-- _resolve_dependencies() -> topological sort
       |-- for each level:
             |-- _execute_stage_ws(stage, WorkingSet)
             |     stage.process_ws(ws, datastore, context=...)
             |     -> (WorkingSet, StageResult)
             |-- or asyncio.gather() for parallel stages
       |
  get_final_data() -> materialize_working_set() -> DataFrame
```

### DuckDB Schema

```
sequences           — sequence_id, sequence, length, hash, [input_columns...]
predictions         — prediction_id, sequence_id, model_name, prediction_type, value
embeddings          — embedding_id, sequence_id, model_name, layer, values (FLOAT[])
structures          — structure_id, sequence_id, model_name, format, structure_data (BLOB)
pipeline_runs       — run_id, pipeline_type, config, status
stage_completions   — stage_id, run_id, stage_name, status, input/output counts
pipeline_context    — run_id, key, value
generation_metadata — metadata_id, sequence_id, model_name, temperature, ...
filter_results      — filter_id, run_id, stage_name, sequence_id, passed
```

## Module Reference

| File | Purpose |
|---|---|
| `base.py` | `BasePipeline`, `Stage`, `StageResult`, `WorkingSet`, `InputSchema`, `PipelineContext` |
| `data.py` | `DataPipeline`, `PredictionStage`, `FilterStage`, `ClusteringStage`, `ExtractionSpec`, `EmbeddingSpec` |
| `generative.py` | `GenerativePipeline`, `GenerationStage`, `DirectGenerationConfig`, `FoldingEntity` |
| `datastore_duckdb.py` | `DuckDBDataStore` — all DuckDB operations |
| `filters.py` | All filter classes + `combine_filters()` |
| `mlm_remasking.py` | `MLMRemasker`, `RemaskingConfig` |
| `clustering.py` | `SequenceClusterer`, `DiversityAnalyzer` |
| `visualization.py` | `PipelinePlotter` |
