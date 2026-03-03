# BioLM Pipeline System

Multi-stage orchestration framework for protein sequence generation, prediction, and analysis.

## Quick Start

```python
from biolmai.pipeline import DataPipeline
from biolmai.pipeline.filters import ThresholdFilter

pipeline = DataPipeline(sequences=["MKLLIV...", "ACDEFG..."])

pipeline.add_prediction(
    "temberture-regression",
    prediction_type="tm",
    extractions="prediction",
)

pipeline.add_filter(ThresholdFilter("tm", min_value=50), depends_on=["predict_tm"])
pipeline.run()
df = pipeline.get_final_data()
```

---

## Extraction: Reading API Responses

When you call a model, the API returns a JSON dict per sequence. You need to tell the pipeline two things:

1. **What key to read** from the response dict (`extractions=`)
2. **What column name** to store it under in DuckDB and the final DataFrame (`prediction_type=`)

### API response examples

```python
# temberture-regression → {"prediction": 48.6}
# soluprot              → {"soluble": 0.37, "is_soluble": false}
# esmfold               → {"pdb": "ATOM...", "mean_plddt": 90.3, "ptm": 0.94}
# esmc-300m (score)     → {"log_prob": -70.1}
```

### `add_prediction` parameters

```python
pipeline.add_prediction(
    "temberture-regression",       # model_name: which BioLM model to call
    prediction_type="tm",          # column name in DuckDB + final DataFrame
                                   #   also used as cache key and auto stage name
    extractions="prediction",      # response key: reads result["prediction"]
    stage_name="predict_tm",       # (optional) explicit stage name
)
```

**`prediction_type`** = the **column name** you see in `get_final_data()`. It's also the cache key — if a sequence already has a `"tm"` prediction cached, it won't re-call the API. If omitted, defaults to the first extraction key.

**`extractions`** = which **response key(s)** to read from each API result dict. Required for predict/score actions. Three forms:

```python
# Form 1: String — one key, column name = prediction_type (or the key itself)
pipeline.add_prediction("temberture-regression",
    prediction_type="tm",          # ← column name
    extractions="prediction",      # ← reads result["prediction"]
)
# Result: column "tm" with value from result["prediction"]

# Form 2: Dict — multiple keys from one response, each with its own column name
pipeline.add_prediction("esmfold",
    prediction_type="plddt",
    extractions={
        "mean_plddt": "plddt",     # result["mean_plddt"] → column "plddt"
        "ptm": "ptm",              # result["ptm"]        → column "ptm"
    },
)
# Result: columns "plddt" AND "ptm" from one API call

# Form 3: ExtractionSpec — full control, with array reduction
from biolmai.pipeline.data import ExtractionSpec
pipeline.add_prediction("esmfold",
    prediction_type="plddt",
    extractions=[
        ExtractionSpec("plddt", "plddt_mean", reduction="mean"),
        ExtractionSpec("plddt", "plddt_min", reduction="min"),
        ExtractionSpec("mean_plddt", "plddt_overall"),
    ],
)
# Result: columns "plddt_mean", "plddt_min", "plddt_overall"
```

### `ExtractionSpec` fields

| Field | Description |
|---|---|
| `response_key` | Key in the API response dict to read |
| `prediction_type` | Column name to store the value under |
| `reduction` | For array values: `"mean"`, `"max"`, `"min"`, `"sum"` — reduces `[90.1, 88.3, ...]` to a scalar |

### Auto-naming

When you don't set `stage_name`, it's auto-generated from `prediction_type`:

```python
pipeline.add_prediction("temberture-regression", prediction_type="tm", extractions="prediction")
# stage_name auto-set to "predict_tm"
# depends_on references: depends_on=["predict_tm"]
```

---

## How Embedding Extraction Works

Embedding models (`action="encode"`) return arrays, not scalars. The `embedding_extractor` parameter tells the stage how to read them.

### `EmbeddingSpec` (declarative)

```python
from biolmai.pipeline.data import EmbeddingSpec

# esm2-8m returns: {"embeddings": [{"embedding": [0.1, 0.2, ...], "layer": 33}]}
pipeline.add_prediction("esm2-8m", action="encode",
    embedding_extractor=EmbeddingSpec(key="embeddings"),        # Stores all layers
)

# ablang2 returns: {"seqcoding": [0.1, 0.2, ...]}
pipeline.add_prediction("ablang2", action="encode",
    embedding_extractor=EmbeddingSpec(key="seqcoding"),
)

# dnabert2 returns: {"embedding": [0.1, 0.2, ...]}
pipeline.add_prediction("dnabert2", action="encode",
    embedding_extractor=EmbeddingSpec(key="embedding"),
)
```

**`EmbeddingSpec` parameters:**

| Param | Type | Description |
|---|---|---|
| `key` | `str` | Response dict key containing embedding data |
| `layer` | `Optional[int]` | For multi-layer responses: `None` = all, `int` = specific layer |
| `reduction` | `Optional[str]` | Reduce 2-D per-token array to 1-D: `"mean"`, `"first"`, `"last"`, `"sum"` |

```python
# Store only layer 33 from a multi-layer model
EmbeddingSpec(key="embeddings", layer=33)

# Mean-pool per-token embeddings to a single vector
EmbeddingSpec(key="embedding", reduction="mean")

# Layer 33, mean-pooled
EmbeddingSpec(key="embeddings", layer=33, reduction="mean")
```

### Custom callable

For full control, pass a function that returns `list[tuple[np.ndarray, Optional[int]]]`:

```python
import numpy as np

def my_extractor(result: dict):
    """Extract and mean-pool the first layer only."""
    layers = result.get("embeddings", [])
    if layers and isinstance(layers[0], dict):
        arr = np.array(layers[0]["embedding"])
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        return [(arr, layers[0].get("layer"))]
    return []

pipeline.add_prediction("esm2-650m", action="encode",
    embedding_extractor=my_extractor,
)
```

---

## Multi-Column Input

For models that take multiple chains (antibody H+L, paired proteins), pass a DataFrame with `input_columns`:

```python
import pandas as pd

df = pd.DataFrame({
    "heavy_chain": ["EVQLVESGGGLVQ...", "QVQLQESGPGLVK..."],
    "light_chain": ["DIQMTQSPSSLSA...", "SYELTQPPSVSSGA..."],
})

pipeline = DataPipeline(
    sequences=df,
    input_columns=["heavy_chain", "light_chain"],
)
```

**What happens internally:**
- Both columns are added to the DuckDB `sequences` table as real columns (not EAV attributes)
- Dedup hashes across all input columns (same H + different L = different row)
- A synthetic `sequence` column is created (`"heavy:light"`) for backward compat
- `get_final_data()` and `export_to_dataframe()` include all input columns

### Sending columns to the API with `item_columns`

By default, `PredictionStage` sends `{"sequence": row["sequence"]}` to the API. Use `item_columns` to map different columns to the API's expected fields:

```python
# Send heavy_chain as "sequence" to temberture-regression
pipeline.add_prediction("temberture-regression",
    prediction_type="tm_heavy",
    extractions="prediction",
    item_columns={"sequence": "heavy_chain"},
    stage_name="predict_tm_heavy",
)

# Send both chains to ablang2
pipeline.add_prediction("ablang2", action="encode",
    embedding_extractor=EmbeddingSpec(key="seqcoding"),
    item_columns={"heavy": "heavy_chain", "light": "light_chain"},
)
```

`item_columns` format: `{"api_field_name": "dataframe_column_name"}`

### Parallel per-chain predictions

```python
# Predict Tm on heavy and light chains separately, in parallel
pipeline.add_prediction("temberture-regression",
    prediction_type="tm_heavy", extractions="prediction",
    item_columns={"sequence": "heavy_chain"},
    stage_name="predict_tm_heavy",
)
pipeline.add_prediction("temberture-regression",
    prediction_type="tm_light", extractions="prediction",
    item_columns={"sequence": "light_chain"},
    stage_name="predict_tm_light",
)
# Both run in parallel (same dependency level), results merged back per sequence
```

---

## Co-Folding (Boltz2, Chai-1)

For multi-molecule structure prediction, use `add_cofolding_prediction`. Each pipeline sequence becomes one chain, and `static_entities` adds ligands, cofactors, or other chains that are the same for every request.

```python
from biolmai.pipeline import FoldingEntity

pipeline = DataPipeline(sequences=protein_sequences)

pipeline.add_cofolding_prediction(
    model_name="boltz2",
    prediction_type="structure",
    sequence_chain_id="A",              # Chain ID for the pipeline sequence
    sequence_entity_type="protein",     # Entity type for the pipeline sequence
    static_entities=[
        # Ligand (SMILES)
        FoldingEntity(id="L", entity_type="ligand", smiles="c1ccccc1"),
        # Another protein chain
        FoldingEntity(id="B", entity_type="protein", sequence="MKTAYIAK..."),
        # DNA strand
        FoldingEntity(id="D", entity_type="dna", sequence="ATGCGATCG"),
        # Ligand by CCD code
        FoldingEntity(id="X", entity_type="ligand", ccd="ATP"),
    ],
    params={"recycling_steps": 3, "sampling_steps": 20},
    batch_size=1,                       # Co-folding is typically batch-size-1
    depends_on=["filter_top50"],
)
```

**`FoldingEntity` fields:**

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Chain/molecule identifier |
| `entity_type` | `str` | `"protein"`, `"dna"`, `"rna"`, or `"ligand"` |
| `sequence` | `Optional[str]` | AA/nucleotide sequence (for protein/DNA/RNA) |
| `smiles` | `Optional[str]` | SMILES string (for ligands) |
| `ccd` | `Optional[str]` | CCD code (for standard ligands like ATP, HEM) |

The stage builds a `{"molecules": [...]}` item per sequence and stores:
- The CIF structure in the `structures` table
- The confidence score as a prediction (column = `prediction_type`)

---

## Generation Models

### DirectGenerationConfig

The recommended way to configure generation. You specify the model, the item field name, and model-specific params.

```python
from biolmai.pipeline import GenerativePipeline, DirectGenerationConfig
```

**Structure-conditioned (MPNN family):**

```python
# protein-mpnn, hyper-mpnn, ligand-mpnn, soluble-mpnn, global-label-membrane-mpnn
DirectGenerationConfig(
    model_name="protein-mpnn",
    item_field="pdb",                   # API expects PDB string
    structure_path="protein.pdb",       # Read from file
    params={"batch_size": 50, "temperature": 0.3},
)

# Or read structure from DuckDB (predicted by upstream stage)
DirectGenerationConfig(
    model_name="protein-mpnn",
    item_field="pdb",
    structure_from_model="esmfold",     # Read from structures table
    params={"batch_size": 50, "temperature": 0.3},
)
```

**Sequence-conditioned (DSM):**

```python
DirectGenerationConfig(
    model_name="dsm-650m-base",         # or dsm-150m-base
    item_field="sequence",
    sequence="MKTAYIAK...",             # Parent sequence
    params={"num_sequences": 20, "temperature": 1.0},
)
```

**Enzyme (ZymCTRL):**

```python
DirectGenerationConfig(
    model_name="zymctrl",
    item_field="ec_number",             # Conditioned on EC number
    sequence="3.1.1.101",              # EC number as the "sequence" value
    params={"temperature": 1.0, "max_length": 300},
)
```

**Antibody (ProGen2-OAS):**

```python
DirectGenerationConfig(
    model_name="progen2-oas",
    item_field="context",               # Autoregressive seed
    sequence="EVQLVES",                 # VH N-terminal seed
    params={"temperature": 1.0, "max_length": 120},
)
```

**MLM Remasking (ESM2):**

Iterative masked-LM refinement: mask random positions in a parent sequence, predict replacements with ESM2, repeat. Generates variants with controlled mutation rates.

```python
from biolmai.pipeline.mlm_remasking import RemaskingConfig

config = RemaskingConfig(
    model_name="esm2-650m",         # ESM2 fill-mask model
    mask_fraction=0.15,             # Mask 15% of positions per iteration
    num_iterations=3,               # 3 rounds of mask-predict-replace
    temperature=1.0,                # Sampling temperature for logit decoding
    top_k=5,                        # (optional) Top-k sampling
    top_p=0.95,                     # (optional) Nucleus sampling
    conserved_positions=[107, 109], # (optional) Never mask these positions
)
# parent_sequence is set as an attribute (not a constructor arg)
config.parent_sequence = "SNPYARGPNPTAASLEASAG..."
config.num_variants = 10

pipeline = GenerativePipeline(generation_configs=[config])
pipeline.run()
```

How it works internally:
1. Selects positions to mask (random, respecting `conserved_positions`)
2. Inserts `<mask>` tokens at those positions in the sequence
3. Sends the masked sequence to the ESM2 predict endpoint
4. Receives `logits` (per-position, per-residue probabilities)
5. Samples from the logits at mask positions (with temperature/top-k/top-p)
6. Repeats for `num_iterations` rounds
7. Returns the variant sequence and mutation metadata

**Multi-model pipeline:**

```python
pipeline = GenerativePipeline(
    generation_configs=[mpnn_config, dsm_config, zymctrl_config, remasking_config],
    deduplicate=True,                   # Remove duplicates across all models
)
pipeline.add_prediction("temberture-regression", prediction_type="tm", extractions="prediction")
pipeline.add_filter(RankingFilter("tm", n=20, ascending=False), depends_on=["predict_tm"])
pipeline.run()
```

---

## Pipeline Context

Share data between stages. Backed by DuckDB (`pipeline_context` table), scoped per `run_id`.

```python
pipeline.context.set("experiment", "thermo_screen")
pipeline.context.set("config", {"target_tm": 65, "batch": 3})

pipeline.context.get("experiment")          # → "thermo_screen"
pipeline.context.get("config")["target_tm"] # → 65
pipeline.context.get("missing", "default")  # → "default"

# Fetch structures from the structures table
struct = pipeline.context.get_structure(seq_id, "esmfold")
structs_df = pipeline.context.get_structures_for_ws(working_set, "esmfold")
```

---

## Filters

| Filter | SQL-native | Description |
|---|---|---|
| `ThresholdFilter(col, min_value, max_value)` | Yes | Numeric threshold on a prediction column |
| `RankingFilter(col, n, ascending)` | Yes | Top-N or bottom-N by column value |
| `SequenceLengthFilter(min_length, max_length)` | Yes | Filter by sequence length |
| `ValidAminoAcidFilter(column)` | Yes | Remove sequences with non-standard residues. `column` defaults to `"sequence"` but works on any column (e.g. `"heavy_chain"`) |
| `HammingDistanceFilter(ref, max_distance)` | No | Hamming distance to a reference sequence |
| `ConservedResidueFilter(positions)` | No | Required residues at specific positions |
| `DiversitySamplingFilter(n_samples, method)` | No | Subsample for diversity |
| `CustomFilter(func)` | No | Any callable `(DataFrame) -> DataFrame` |
| `combine_filters(f1, f2, ...)` | No | Chain multiple filters sequentially |

SQL-native filters execute directly in DuckDB — no DataFrame materialization, no memory overhead.

---

## Caching and Resume

```python
# Predictions are cached in DuckDB by (sequence_id, prediction_type, model_name)
# Re-running skips cached predictions automatically

# Resume from a specific run
pipeline = DataPipeline(sequences=seqs, run_id="my_run", resume=True)
# Completed stages are skipped entirely

# Trickle new sequences into existing cache
pipeline = DataPipeline(
    sequences=old_seqs + new_seqs,
    datastore="project.duckdb",
)
# Old sequences hit cache; only new ones call the API
```

---

## Architecture

```
sequences (list / CSV / FASTA / DataFrame with input_columns)
       |
  _get_initial_data() → add to DuckDB, return WorkingSet
       |
  run_async() → resolve dependencies → topological sort into levels
       |
  for each level:
     | single stage → _execute_stage_ws(stage, WorkingSet)
     | parallel    → asyncio.gather(stage1, stage2, ...) → intersect WorkingSets
       |
  get_final_data() → materialize_working_set() → DataFrame with all columns
```

### DuckDB Schema

| Table | Columns | Purpose |
|---|---|---|
| `sequences` | `sequence_id`, `sequence`, `length`, `hash`, + input columns | Primary data |
| `predictions` | `prediction_id`, `sequence_id`, `model_name`, `prediction_type`, `value` | Cached predictions |
| `embeddings` | `embedding_id`, `sequence_id`, `model_name`, `layer`, `values` (FLOAT[]) | Embedding vectors |
| `structures` | `structure_id`, `sequence_id`, `model_name`, `format`, `structure_data` (BLOB) | PDB/CIF structures |
| `pipeline_runs` | `run_id`, `pipeline_type`, `config`, `status` | Run tracking |
| `stage_completions` | `stage_id`, `run_id`, `stage_name`, `status`, counts | Resume support |
| `pipeline_context` | `run_id`, `key`, `value` | Inter-stage shared state |
| `generation_metadata` | `metadata_id`, `sequence_id`, `model_name`, `temperature`, ... | Generation params |
| `filter_results` | `filter_id`, `run_id`, `stage_name`, `sequence_id`, `passed` | Filter resume support |

### Module Reference

| File | Key exports |
|---|---|
| `base.py` | `BasePipeline`, `Stage`, `StageResult`, `WorkingSet`, `InputSchema`, `PipelineContext` |
| `data.py` | `DataPipeline`, `PredictionStage`, `FilterStage`, `ClusteringStage`, `ExtractionSpec`, `EmbeddingSpec` |
| `generative.py` | `GenerativePipeline`, `GenerationStage`, `DirectGenerationConfig`, `FoldingEntity` |
| `datastore_duckdb.py` | `DuckDBDataStore` |
| `filters.py` | `ThresholdFilter`, `RankingFilter`, `SequenceLengthFilter`, `ValidAminoAcidFilter`, `HammingDistanceFilter`, `ConservedResidueFilter`, `DiversitySamplingFilter`, `CustomFilter`, `combine_filters` |
| `mlm_remasking.py` | `MLMRemasker`, `RemaskingConfig` |
| `clustering.py` | `SequenceClusterer`, `DiversityAnalyzer` |
| `visualization.py` | `PipelinePlotter` |
