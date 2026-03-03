# BioLM Pipeline System - Quick Start Guide

**Last Updated**: 2026-03-02

---

## Installation

```bash
pip install biolmai[pipeline]
```

Set your API token:

```bash
export BIOLMAI_TOKEN="your-token-here"
# Get tokens at https://biolm.ai/ui/accounts/user-api-tokens/
```

---

## Your First Pipeline

### Example 1: Predict Melting Temperature

```python
from biolmai.pipeline import DataPipeline
from biolmai.pipeline.filters import ThresholdFilter

pipeline = DataPipeline(sequences=[
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSG",
    "MRVLKFGGTSVANAERFLRVADILESNARQGQVATVLSAPAKITNHLVAM",
    "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSY",
])

# Predict Tm — extractions tells the stage which response key to read
pipeline.add_prediction(
    "temberture-regression",
    prediction_type="tm",
    extractions="prediction",        # API returns {"prediction": 48.6}
)

# Filter: keep Tm > 48
pipeline.add_filter(
    ThresholdFilter("tm", min_value=48.0),
    depends_on=["predict_tm"],
)

results = pipeline.run()
df = pipeline.get_final_data()
print(df[["sequence", "tm"]])
```

### Example 2: Parallel Predictions (Tm + Solubility)

```python
from biolmai.pipeline import DataPipeline
from biolmai.pipeline.filters import RankingFilter

pipeline = DataPipeline(sequences=my_sequences)

# These run in parallel (no dependencies between them)
pipeline.add_prediction(
    "temberture-regression",
    prediction_type="tm",
    extractions="prediction",
    stage_name="predict_tm",
)
pipeline.add_prediction(
    "soluprot",
    prediction_type="solubility",
    extractions="soluble",           # API returns {"soluble": 0.37}
    stage_name="predict_sol",
)

# Rank top 10 by solubility (depends on both predictions)
pipeline.add_filter(
    RankingFilter("solubility", n=10, ascending=False),
    depends_on=["predict_tm", "predict_sol"],
)

pipeline.run()
df = pipeline.get_final_data()
print(df[["sequence", "tm", "solubility"]])
```

### Example 3: Generate + Score

```python
from biolmai.pipeline import GenerativePipeline, DirectGenerationConfig
from biolmai.pipeline.filters import ThresholdFilter, RankingFilter

config = DirectGenerationConfig(
    model_name="protein-mpnn",
    item_field="pdb",
    structure_path="my_protein.pdb",
    params={"batch_size": 100, "temperature": 0.3},
)

pipeline = GenerativePipeline(generation_configs=[config])
pipeline.add_prediction(
    "temberture-regression",
    prediction_type="tm",
    extractions="prediction",
)
pipeline.add_filter(
    RankingFilter("tm", n=20, ascending=False),
    depends_on=["predict_tm"],
)

pipeline.run()
```

---

## Extraction: How Stages Read API Responses

Every prediction and embedding stage requires explicit extraction — the stage needs to know which key in the API response holds the value.

### Prediction Extraction (`extractions=`)

```python
# Simple: single response key
pipeline.add_prediction("temberture-regression",
    prediction_type="tm",
    extractions="prediction",          # Response: {"prediction": 48.6}
)

# Multiple values from one response
pipeline.add_prediction("esmfold",
    prediction_type="plddt",
    extractions={"mean_plddt": "plddt", "ptm": "ptm"},
)

# With array reduction (per-residue → scalar)
from biolmai.pipeline.data import ExtractionSpec
pipeline.add_prediction("esmfold",
    prediction_type="plddt",
    extractions=[ExtractionSpec("plddt", "plddt", reduction="mean")],
)
```

### Embedding Extraction (`embedding_extractor=`)

```python
from biolmai.pipeline.data import EmbeddingSpec

# Simple: single key
pipeline.add_prediction("esm2-8m", action="encode",
    embedding_extractor=EmbeddingSpec(key="embeddings"),
)

# Paired antibody model (ablang2 returns "seqcoding")
pipeline.add_prediction("ablang2", action="encode",
    embedding_extractor=EmbeddingSpec(key="seqcoding"),
    item_columns={"heavy": "heavy_chain", "light": "light_chain"},
)

# Layer selection + mean-pooling per-token embeddings
pipeline.add_prediction("esm2-650m", action="encode",
    embedding_extractor=EmbeddingSpec(key="embeddings", layer=33, reduction="mean"),
)

# Custom callable for full control
pipeline.add_prediction("my-model", action="encode",
    embedding_extractor=lambda r: [(np.array(r["my_key"]), None)],
)
```

### Known API Response Keys

| Model | Action | Response Key | `extractions=` |
|---|---|---|---|
| `temberture-regression` | predict | `prediction` | `extractions="prediction"` |
| `soluprot` | predict | `soluble` | `extractions="soluble"` |
| `esmc-300m` | score | `log_prob` | `extractions="log_prob"` |
| `esm2-8m` | encode | `embeddings` | `EmbeddingSpec(key="embeddings")` |
| `ablang2` | encode | `seqcoding` | `EmbeddingSpec(key="seqcoding")` |
| `dnabert2` | encode | `embedding` | `EmbeddingSpec(key="embedding")` |

---

## Multi-Column Input (Antibodies, Multi-Chain)

For models that take multiple chains (antibody H+L, paired proteins):

```python
import pandas as pd
from biolmai.pipeline import DataPipeline

df = pd.DataFrame({
    "heavy_chain": ["EVQLVESGGGLVQ...", "QVQLQESGPGLVK..."],
    "light_chain": ["DIQMTQSPSSLSA...", "SYELTQPPSVSSGA..."],
})

pipeline = DataPipeline(
    sequences=df,
    input_columns=["heavy_chain", "light_chain"],
)

# Predict Tm on heavy chain only
pipeline.add_prediction("temberture-regression",
    prediction_type="tm_heavy",
    extractions="prediction",
    item_columns={"sequence": "heavy_chain"},   # Send heavy_chain as "sequence"
    stage_name="predict_tm_heavy",
)

# Predict Tm on light chain (runs in parallel)
pipeline.add_prediction("temberture-regression",
    prediction_type="tm_light",
    extractions="prediction",
    item_columns={"sequence": "light_chain"},
    stage_name="predict_tm_light",
)

# Paired embeddings with ablang2
pipeline.add_prediction("ablang2", action="encode",
    embedding_extractor=EmbeddingSpec(key="seqcoding"),
    item_columns={"heavy": "heavy_chain", "light": "light_chain"},
    stage_name="embed",
)

pipeline.run()
df = pipeline.get_final_data()
# Columns: heavy_chain, light_chain, tm_heavy, tm_light
```

Key behaviors:
- **Dedup**: hashes across all input columns (same H + different L = different rows)
- **Columns stored on DuckDB**: `heavy_chain` and `light_chain` are real columns, not EAV attributes
- **`item_columns` mapping**: `{"api_field": "df_column"}` controls what gets sent to the API

---

## Pipeline Context

Share data between stages (e.g., pass structures from prediction to generation):

```python
pipeline = DataPipeline(sequences=my_seqs)

# Set metadata
pipeline.context.set("experiment", "thermostability_screen")
pipeline.context.set("config", {"target_tm": 65})

# Read in downstream code
pipeline.context.get("experiment")   # → "thermostability_screen"
pipeline.context.get("missing", "default")  # → "default"

# Access structures predicted by upstream stages
struct = pipeline.context.get_structure(seq_id, "esmfold")
```

Context is backed by DuckDB (`pipeline_context` table), persists across stages, and is scoped per `run_id`.

---

## Generation Models

### DirectGenerationConfig (Recommended)

```python
from biolmai.pipeline import DirectGenerationConfig, GenerativePipeline

# Structure-conditioned (MPNN family)
config = DirectGenerationConfig(
    model_name="protein-mpnn",    # or hyper-mpnn, ligand-mpnn, soluble-mpnn
    item_field="pdb",
    structure_path="protein.pdb",
    params={"batch_size": 50, "temperature": 0.3},
)

# Sequence-conditioned (DSM)
config = DirectGenerationConfig(
    model_name="dsm-650m-base",
    item_field="sequence",
    sequence="MKTAYIAK...",
    params={"num_sequences": 20, "temperature": 1.0},
)

# Enzyme generation (ZymCTRL)
config = DirectGenerationConfig(
    model_name="zymctrl",
    item_field="ec_number",
    sequence="3.1.1.101",          # PETase EC number
    params={"temperature": 1.0, "max_length": 300},
)

# Antibody generation (ProGen2-OAS)
config = DirectGenerationConfig(
    model_name="progen2-oas",
    item_field="context",
    sequence="EVQLVES",            # VH seed
    params={"temperature": 1.0, "max_length": 120},
)

# Multi-model pipeline
pipeline = GenerativePipeline(
    generation_configs=[config1, config2, config3],
    deduplicate=True,
)
```

### Structure from Upstream Stage

```python
config = DirectGenerationConfig(
    model_name="protein-mpnn",
    structure_from_model="esmfold",   # Read from DuckDB structures table
    params={"batch_size": 50, "temperature": 0.3},
)
```

---

## Filters

```python
from biolmai.pipeline.filters import (
    ThresholdFilter,          # Numeric threshold (min/max)
    RankingFilter,            # Top-N or bottom-N by column
    SequenceLengthFilter,     # Min/max sequence length
    ValidAminoAcidFilter,     # Remove non-standard residues
    HammingDistanceFilter,    # Distance to reference sequence
    ConservedResidueFilter,   # Required residues at positions
    DiversitySamplingFilter,  # Subsample for diversity
    CustomFilter,             # Any callable
    combine_filters,          # Chain multiple filters
)

# ValidAminoAcidFilter works on any column
ValidAminoAcidFilter(column="heavy_chain")

# RankingFilter: top 20 by Tm
RankingFilter("tm", n=20, ascending=False)

# Combine
combined = combine_filters(
    ThresholdFilter("tm", min_value=50),
    SequenceLengthFilter(min_length=100),
)
```

SQL-translatable filters (`ThresholdFilter`, `RankingFilter`, `SequenceLengthFilter`, `ValidAminoAcidFilter`) execute directly in DuckDB with zero DataFrame materialization.

---

## Caching and Resume

```python
# Predictions are cached automatically in DuckDB
# Re-running with the same sequences skips cached predictions

# Resume a failed/interrupted run
pipeline = DataPipeline(
    sequences=my_seqs,
    run_id="my_analysis_v1",
    resume=True,             # Skip completed stages
)

# Trickle new sequences into existing cache
pipeline = DataPipeline(
    sequences=old_seqs + new_seqs,
    datastore="project.duckdb",   # Reuse existing DB
)
# Old sequences hit cache; only new ones call the API
```

---

## DuckDB DataStore

```python
from biolmai.pipeline import DuckDBDataStore

store = DuckDBDataStore("pipeline.duckdb", "data/")

# Add sequences (batch, deduplicated)
ids = store.add_sequences_batch(["MKLLIV", "ACDEFG"])

# Add predictions
store.add_prediction(seq_id, "tm", "temberture-regression", 48.6)

# Query with SQL
df = store.query("SELECT * FROM predictions WHERE value > 60")

# Export
df = store.export_to_dataframe(include_predictions=True)
store.export_to_csv("results.csv")
```

---

## Data Exploration

```python
pipeline.run()

# Summary stats
pipeline.explore()       # → {"sequences": 100, "predictions": {...}, ...}
pipeline.stats()         # → DataFrame of stage completions
pipeline.summary()       # → DataFrame of stage input/output counts

# SQL queries
pipeline.query("SELECT * FROM sequences WHERE length > 200")

# Visualization
pipeline.plot(kind="funnel")
pipeline.plot(kind="predictions")
pipeline.plot(kind="distributions")
```

---

## Full Example: Multi-Model PETase Design

```python
from biolmai.pipeline import GenerativePipeline, DirectGenerationConfig, DataPipeline
from biolmai.pipeline.data import EmbeddingSpec
from biolmai.pipeline.filters import (
    ValidAminoAcidFilter, SequenceLengthFilter,
    ThresholdFilter, RankingFilter,
)

LCC = "SNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGG..."

# Step 1: Generate from multiple sources
configs = [
    DirectGenerationConfig(model_name="protein-mpnn", item_field="pdb",
                           structure_path="lcc.pdb",
                           params={"batch_size": 50, "temperature": 0.3}),
    DirectGenerationConfig(model_name="zymctrl", item_field="ec_number",
                           sequence="3.1.1.101",
                           params={"temperature": 1.0, "max_length": 300}),
    DirectGenerationConfig(model_name="dsm-650m-base", item_field="sequence",
                           sequence=LCC,
                           params={"num_sequences": 20, "temperature": 1.0}),
]

gen_pipeline = GenerativePipeline(generation_configs=configs, deduplicate=True)
gen_pipeline.run()
generated = gen_pipeline.get_final_data()

# Step 2: Score all variants
pipeline = DataPipeline(sequences=[LCC] + generated["sequence"].tolist())
pipeline.add_filter(ValidAminoAcidFilter())
pipeline.add_filter(SequenceLengthFilter(min_length=100, max_length=500),
                    depends_on=["filter_0"])
pipeline.add_prediction("temberture-regression", prediction_type="tm",
                        extractions="prediction", depends_on=["filter_1"])
pipeline.add_prediction("soluprot", prediction_type="solubility",
                        extractions="soluble", depends_on=["filter_1"])
pipeline.add_prediction("esmc-300m", action="score", prediction_type="log_prob",
                        extractions="log_prob", depends_on=["filter_1"])
pipeline.add_filter(ThresholdFilter("tm", min_value=45),
                    depends_on=["predict_tm"])
pipeline.add_filter(RankingFilter("solubility", n=20, ascending=False),
                    depends_on=["filter_3", "predict_solubility", "predict_log_prob"])

pipeline.run()
df = pipeline.get_final_data()
print(df[["sequence", "tm", "solubility", "log_prob"]].to_string())
```
