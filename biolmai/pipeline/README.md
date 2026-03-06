# BioLM Pipeline System

Multi-stage orchestration framework for protein sequence generation, prediction, and analysis.

## Quick Start: Generate and Score

```python
from biolmai.pipeline import GenerativePipeline, DirectGenerationConfig
from biolmai.pipeline.filters import RankingFilter

# Generate 50 variants from a PDB structure
config = DirectGenerationConfig(
    model_name="protein-mpnn",
    item_field="pdb",
    structure_path="my_protein.pdb",
    params={"batch_size": 50, "temperature": 0.3},
)

pipeline = GenerativePipeline(generation_configs=[config], deduplicate=True)

# Score with Tm and solubility
pipeline.add_prediction("temberture-regression", extractions="prediction", columns="tm")
pipeline.add_prediction("soluprot", extractions="soluble", columns="solubility")
pipeline.add_filter(RankingFilter("tm", n=20, ascending=False),
                    depends_on=["predict_tm", "predict_solubility"])

pipeline.run()
df = pipeline.get_final_data()
# Columns: sequence, tm, solubility
```

---

## Generation

### Structure-Conditioned (MPNN Family)

Inverse folding — design sequences that fold into a target structure.

```python
from biolmai.pipeline import DirectGenerationConfig

# protein-mpnn, hyper-mpnn, ligand-mpnn, soluble-mpnn, global-label-membrane-mpnn
config = DirectGenerationConfig(
    model_name="protein-mpnn",
    item_field="pdb",
    structure_path="protein.pdb",       # Read PDB from file
    params={"batch_size": 50, "temperature": 0.3},
)
```

**Temperature scanning** — generate at multiple temperatures in one pipeline:

```python
configs = [
    DirectGenerationConfig(
        model_name="protein-mpnn",
        item_field="pdb",
        structure_path="protein.pdb",
        params={"batch_size": 20, "temperature": t},
    )
    for t in [0.1, 0.3, 0.5, 1.0]
]

pipeline = GenerativePipeline(generation_configs=configs, deduplicate=True)
# 4 configs × 20 seqs = up to 80 unique sequences
```

**Multiple MPNN models in parallel:**

```python
configs = []
for model in ["protein-mpnn", "hyper-mpnn", "ligand-mpnn", "soluble-mpnn"]:
    for temp in [0.1, 0.3]:
        configs.append(DirectGenerationConfig(
            model_name=model,
            item_field="pdb",
            structure_path="protein.pdb",
            params={"batch_size": 25, "temperature": temp},
        ))

pipeline = GenerativePipeline(generation_configs=configs, deduplicate=True)
# 4 models × 2 temps × 25 seqs = up to 200 unique sequences
```

**Parallel runs** (`n_runs`) — call the same generation config multiple times concurrently:

```python
# Each MPNN call returns batch_size sequences.
# n_runs=3 makes 3 parallel API calls = 3 × 50 = 150 sequences (before dedup).
DirectGenerationConfig(
    model_name="protein-mpnn",
    item_field="pdb",
    structure_path="protein.pdb",
    params={"batch_size": 50, "temperature": 0.3},
    n_runs=3,
)
```

This is useful when a model's per-call output is limited (e.g., ZymCTRL returns ~5 sequences per call — `n_runs=10` gives ~50).

**Structure from upstream stage** (e.g., ESMFold → MPNN):

```python
config = DirectGenerationConfig(
    model_name="protein-mpnn",
    item_field="pdb",
    structure_from_model="esmfold",     # Read from DuckDB structures table
    params={"batch_size": 50, "temperature": 0.3},
)
```

### Masked Infilling (DSM, ESM2)

Mask positions in a parent sequence, fill them with a language model. Works with any model that accepts `<mask>` tokens — ESM2 (returns logits, decoded client-side) and DSM (returns filled sequences directly).

```python
from biolmai.pipeline.mlm_remasking import RemaskingConfig

config = RemaskingConfig(
    model_name="esm2-650m",         # or "dsm-650m-base"
    action="predict",               # "predict" for ESM2 (logits), "generate" for DSM
    mask_fraction=0.15,             # Mask 15% of positions per iteration
    num_iterations=3,               # 3 rounds of mask-predict-replace
    temperature=1.0,
    mask_token="<mask>",            # Default; set "*" for ablang
    conserved_positions=[107, 109], # Never mask these positions (e.g., active site)
    top_k=5,                        # (optional) Top-k sampling
    top_p=0.95,                     # (optional) Nucleus sampling
)
config.parent_sequence = "SNPYARGPNPTAASLEASAG..."
config.num_variants = 20

pipeline = GenerativePipeline(generation_configs=[config])
pipeline.run()
```

**How it works:**
1. `select_mask_positions()` picks positions to mask (random/low_confidence/blocks strategy), skipping `conserved_positions`
2. `create_masked_sequence()` inserts `mask_token` at those positions
3. Sends masked sequence to the API
4. **ESM2** (`action="predict"`): receives logits per position, decodes with temperature/top-k/top-p sampling
5. **DSM** (`action="generate"`): receives filled sequence directly
6. Repeats for `num_iterations` rounds, each time re-masking different positions
7. Returns variant + mutation metadata (num_mutations, mutation_rate)

**DSM-specific example:**

```python
config = RemaskingConfig(
    model_name="dsm-650m-base",
    action="generate",              # DSM returns filled sequences, not logits
    mask_fraction=0.2,
    num_iterations=2,
    temperature=1.0,
    conserved_positions=[0, 1, 2],  # Keep first 3 residues
)
config.parent_sequence = "MKTAYIAKQRQISFVK..."
config.num_variants = 10
```

### De Novo Generation (DSM)

Full sequence generation (not infilling):

```python
DirectGenerationConfig(
    model_name="dsm-650m-base",
    item_field="sequence",
    sequence="MKTAYIAK...",             # Seed/parent sequence
    params={"num_sequences": 20, "temperature": 1.0},
)
```

### Enzyme Generation (ZymCTRL)

Generate enzyme sequences conditioned on EC number:

```python
DirectGenerationConfig(
    model_name="zymctrl",
    item_field="ec_number",
    sequence="3.1.1.101",               # PETase EC number
    params={"temperature": 1.0, "max_length": 300},
)
```

### Antibody Generation (ProGen2-OAS)

Autoregressive generation seeded from a VH fragment:

```python
DirectGenerationConfig(
    model_name="progen2-oas",
    item_field="context",
    sequence="EVQLVES",                 # VH N-terminal seed
    params={"temperature": 1.0, "max_length": 120},
)
```

### Multi-Model Pipelines

Combine any generation methods — all run in parallel, deduped:

```python
from biolmai.pipeline import GenerativePipeline, DirectGenerationConfig
from biolmai.pipeline.mlm_remasking import RemaskingConfig
from biolmai.pipeline.filters import ThresholdFilter, RankingFilter

# MPNN at multiple temperatures
mpnn_configs = [
    DirectGenerationConfig(model_name="protein-mpnn", item_field="pdb",
        structure_path="protein.pdb", params={"batch_size": 25, "temperature": t})
    for t in [0.1, 0.3, 0.5]
]

# DSM de novo
dsm_config = DirectGenerationConfig(model_name="dsm-650m-base",
    item_field="sequence", sequence=parent_seq,
    params={"num_sequences": 20, "temperature": 1.0})

# ZymCTRL enzyme
zymctrl_config = DirectGenerationConfig(model_name="zymctrl",
    item_field="ec_number", sequence="3.1.1.101",
    params={"temperature": 1.0, "max_length": 300})

# ESM2 remasking
esm2_config = RemaskingConfig(model_name="esm2-650m", action="predict",
    mask_fraction=0.15, num_iterations=3, temperature=1.0,
    conserved_positions=[107, 109])
esm2_config.parent_sequence = parent_seq
esm2_config.num_variants = 20

# DSM remasking (infilling)
dsm_remask = RemaskingConfig(model_name="dsm-650m-base", action="generate",
    mask_fraction=0.2, num_iterations=2, temperature=1.0)
dsm_remask.parent_sequence = parent_seq
dsm_remask.num_variants = 20

# All in one pipeline
all_configs = mpnn_configs + [dsm_config, zymctrl_config, esm2_config, dsm_remask]
pipeline = GenerativePipeline(generation_configs=all_configs, deduplicate=True)

# Score everything
pipeline.add_prediction("temberture-regression", extractions="prediction", columns="tm")
pipeline.add_prediction("soluprot", extractions="soluble", columns="solubility")
pipeline.add_prediction("esmc-300m", action="score", extractions="log_prob")
pipeline.add_filter(ThresholdFilter("tm", min_value=45), depends_on=["predict_tm"])
pipeline.add_filter(RankingFilter("solubility", n=20, ascending=False),
    depends_on=["filter_1", "predict_solubility", "predict_log_prob"])

pipeline.run()
df = pipeline.get_final_data()
```

---

## Co-Folding (Boltz2, Chai-1)

Multi-molecule structure prediction. Each pipeline sequence becomes one chain; `static_entities` adds ligands, cofactors, DNA/RNA, or extra protein chains.

```python
from biolmai.pipeline import DataPipeline, FoldingEntity

pipeline = DataPipeline(sequences=protein_sequences)

pipeline.add_cofolding_prediction(
    model_name="boltz2",
    prediction_type="structure",
    sequence_chain_id="A",
    sequence_entity_type="protein",
    static_entities=[
        FoldingEntity(id="L", entity_type="ligand", smiles="c1ccccc1"),
        FoldingEntity(id="B", entity_type="protein", sequence="MKTAYIAK..."),
        FoldingEntity(id="D", entity_type="dna", sequence="ATGCGATCG"),
        FoldingEntity(id="X", entity_type="ligand", ccd="ATP"),
    ],
    params={"recycling_steps": 3, "sampling_steps": 20},
    batch_size=1,
    depends_on=["filter_top50"],
)
```

| `FoldingEntity` field | Type | Description |
|---|---|---|
| `id` | `str` | Chain/molecule identifier |
| `entity_type` | `str` | `"protein"`, `"dna"`, `"rna"`, or `"ligand"` |
| `sequence` | `Optional[str]` | AA/nucleotide sequence |
| `smiles` | `Optional[str]` | SMILES string (ligands) |
| `ccd` | `Optional[str]` | CCD code (e.g. `"ATP"`, `"HEM"`) |

Stores CIF structures + confidence scores in DuckDB.

---

## Scoring and Prediction

### Extraction: Reading API Responses

Every model returns a JSON dict per sequence. You tell the pipeline **what key to read** (`extractions=`) and **what column name to use** (`columns=`).

```python
pipeline.add_prediction(
    "temberture-regression",       # model_name
    extractions="prediction",      # reads result["prediction"]
    columns="tm",                  # output column name in DataFrame
)
# Stage name auto-derived: "predict_tm"
# Cache key (DuckDB prediction_type): "temberture-regression::predict::prediction"
# Output column: "tm"
# Filter: ThresholdFilter("tm", min_value=45)
```

**`extractions`** = which response key(s) to read. Two forms:

```python
# String — one key
extractions="prediction"

# List — multiple keys from one API call
extractions=["mean_plddt", "ptm"]

# ExtractionSpec — with array reduction
from biolmai.pipeline.data import ExtractionSpec
extractions=[
    ExtractionSpec("plddt", reduction="mean"),   # [90.1, 88.3, ...] → 89.2
    ExtractionSpec("ptm"),
]
```

**`columns`** = output column name(s). Defaults to the response key name.

```python
# String — rename a single extraction
columns="tm"                                    # "prediction" → "tm"

# Dict — rename specific keys (unmapped = identity)
columns={"mean_plddt": "plddt"}                 # "mean_plddt" → "plddt", "ptm" stays "ptm"

# Omitted — column name = response key
extractions="log_prob"                          # column name = "log_prob"
```

**Cache key** is auto-derived from `(model_name, action, sorted_response_keys)`. Renaming columns doesn't invalidate cache.

| `ExtractionSpec` field | Description |
|---|---|
| `response_key` | Key in the API response dict |
| `reduction` | For arrays: `"mean"`, `"max"`, `"min"`, `"sum"` |

### Known API Response Keys

| Model | Action | Response Key | Usage |
|---|---|---|---|
| `temberture-regression` | predict | `prediction` | `extractions="prediction"` |
| `soluprot` | predict | `soluble` | `extractions="soluble"` |
| `esmc-300m` | score | `log_prob` | `extractions="log_prob"` |

### Embedding Extraction

Embedding models (`action="encode"`) return arrays. Use `EmbeddingSpec` or a custom callable:

```python
from biolmai.pipeline.data import EmbeddingSpec

# ESM2: {"embeddings": [{"embedding": [...], "layer": 33}]}
EmbeddingSpec(key="embeddings")                          # All layers
EmbeddingSpec(key="embeddings", layer=33)                # Specific layer
EmbeddingSpec(key="embeddings", layer=33, reduction="mean")  # Mean-pool per-token

# AbLang2: {"seqcoding": [...]}
EmbeddingSpec(key="seqcoding")

# DNABERT2: {"embedding": [...]}
EmbeddingSpec(key="embedding")

# Custom callable: (dict) -> list[(np.ndarray, Optional[int])]
embedding_extractor=lambda r: [(np.array(r["my_key"]), None)]
```

| `EmbeddingSpec` param | Description |
|---|---|
| `key` | Response dict key |
| `layer` | `None` = all layers, `int` = specific layer |
| `reduction` | `"mean"`, `"first"`, `"last"`, `"sum"` — reduces 2-D to 1-D |

---

## Data Pipelines (Existing Sequences)

Score, filter, and analyze sequences you already have:

```python
from biolmai.pipeline import DataPipeline
from biolmai.pipeline.filters import ThresholdFilter, RankingFilter

pipeline = DataPipeline(sequences=["MKLLIV...", "ACDEFG..."])
# Also: DataPipeline(sequences="sequences.csv")
# Also: DataPipeline(sequences="sequences.fasta")
# Also: DataPipeline(sequences=my_dataframe)

pipeline.add_prediction("temberture-regression", extractions="prediction", columns="tm")
pipeline.add_prediction("soluprot", extractions="soluble", columns="solubility")
pipeline.add_filter(RankingFilter("tm", n=10, ascending=False),
                    depends_on=["predict_tm", "predict_solubility"])

pipeline.run()
df = pipeline.get_final_data()
```

### Multi-Column Input (Antibodies, Multi-Chain)

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

- Columns stored directly on DuckDB `sequences` table
- Dedup hashes across all input columns (same H + different L = different row)
- `get_final_data()` includes all input columns
- The `sequence` column is empty when `input_columns` is set and doesn't include `"sequence"` — real data lives in the input columns directly
- If you need a `sequence` column alongside other columns, include it in `input_columns`
- Schema is validated on each run: you cannot mix single-column and multi-column inputs on the same datastore

**`item_columns`** — map DataFrame columns to API fields:

```python
# Send heavy_chain as "sequence" to a single-chain model
pipeline.add_prediction("temberture-regression",
    extractions="prediction", columns="tm_heavy",
    item_columns={"sequence": "heavy_chain"},
    stage_name="predict_tm_heavy",
)

# Send both chains to a paired model
pipeline.add_prediction("ablang2", action="encode",
    embedding_extractor=EmbeddingSpec(key="seqcoding"),
    item_columns={"heavy": "heavy_chain", "light": "light_chain"},
)
```

---

## Filters

| Filter | SQL-native | Description |
|---|---|---|
| `ThresholdFilter(col, min_value, max_value)` | Yes | Numeric threshold |
| `RankingFilter(col, n, ascending)` | Yes | Top-N / bottom-N |
| `SequenceLengthFilter(min_length, max_length)` | Yes | Sequence length |
| `ValidAminoAcidFilter(column)` | Yes | Remove non-standard residues (works on any column) |
| `HammingDistanceFilter(ref, max_distance)` | No | Distance to reference |
| `ConservedResidueFilter(positions)` | No | Required residues at positions |
| `DiversitySamplingFilter(n_samples, method)` | No | Subsample for diversity |
| `CustomFilter(func)` | No | Any callable `(DataFrame) -> DataFrame` |

SQL-native filters execute in DuckDB with zero DataFrame materialization.

---

## Pipeline Context

Share data between stages. Backed by DuckDB, scoped per `run_id`.

```python
pipeline.context.set("experiment", "thermo_screen")
pipeline.context.get("experiment")              # → "thermo_screen"
pipeline.context.get_structure(seq_id, "esmfold")
pipeline.context.get_structures_for_ws(ws, "esmfold")
```

---

## Caching and Resume

```python
# Predictions cached in DuckDB — re-running skips cached sequences
# Resume a failed run:
pipeline = DataPipeline(sequences=seqs, run_id="my_run", resume=True)

# Trickle new sequences into existing cache:
pipeline = DataPipeline(sequences=old + new, datastore="project.duckdb")
```

---

## Batching and Concurrency

Prediction stages batch API calls at two levels with **bounded concurrency**, controlled by three knobs:

| Parameter | Default | Controls |
|---|---|---|
| `batch_size` | 32 | Sequences per pipeline batch (each batch = one SDK call) |
| `max_concurrent` | 5 | Max pipeline batches in flight at once |
| `max_connections` | 10 | Max concurrent HTTP requests to the API |

**Pipeline level** (`batch_size` + `max_concurrent`):
- Chunks uncached sequences into groups of `batch_size`
- Keeps up to `max_concurrent` batches in flight at once
- As each batch completes, results are written to DuckDB immediately and a new batch is dispatched
- Memory bounded to at most `max_concurrent * batch_size` items plus their responses

**SDK level** (`max_connections`):
- Each model has a `maxItems` limit from its API schema (e.g. 10 for structure models, 100+ for scoring)
- If a pipeline batch exceeds `maxItems`, the SDK auto-splits into sub-batches
- Sub-batches from all in-flight pipeline batches share a single connection pool throttled by `max_connections`
- Rate limiting and retry are handled automatically

**How the two levels interact:**

```
1000 sequences, batch_size=32, max_concurrent=3, max_connections=10, maxItems=10

Pipeline dispatches 3 batches of 32 concurrently (batches 4-31 wait)
  Batch 1 (32 items) → SDK splits into 4 sub-requests of [10, 10, 10, 2]
  Batch 2 (32 items) → SDK splits into 4 sub-requests of [10, 10, 10, 2]
  Batch 3 (32 items) → SDK splits into 4 sub-requests of [10, 10, 10, 2]
  = 12 sub-requests total, but max_connections=10, so 10 run and 2 wait

Batch 1 finishes → results written to DuckDB → Batch 4 starts
```

**Why bounded concurrency?** Three approaches were considered:
- *Sequential*: simple but leaves the API idle between batches — wasted throughput
- *All-at-once*: maximum throughput but no backpressure — 10k sequences creates 300+ in-flight tasks with all payloads in memory
- *Bounded concurrency* (chosen): keeps the API saturated while bounding memory. DuckDB writes after each batch provide natural backpressure — a slow datastore slows down new batch dispatch

### Tuning for throughput

```python
pipeline.add_prediction("temberture-regression",
    extractions="prediction",
    columns="tm",
    batch_size=64,          # Sequences per pipeline batch
    max_concurrent=8,       # Pipeline batches in flight
    max_connections=16,     # HTTP connections to the API
)
```

**Start with the defaults** (`batch_size=32, max_concurrent=5, max_connections=10`) — they work well for most models. Then tune if needed:

**To increase throughput** (more sequences per second):
- Raise `max_concurrent` first (e.g. 8-10) — this keeps more batches in the pipeline
- Raise `max_connections` if you see batches completing fast but HTTP sub-requests queueing (e.g. 16-20)
- Raise `batch_size` if the model's `maxItems` is large (e.g. 64-128 for scoring models) — fewer SDK splits per batch

**To reduce memory** (large sequences or embeddings):
- Lower `max_concurrent` (e.g. 2-3) — fewer batches of results in memory at once
- Lower `batch_size` — smaller payloads per batch

**For slow/expensive models** (structure prediction, co-folding):
- Lower `batch_size` to 1-4 (these models often have `maxItems=1`)
- Keep `max_concurrent` at 3-5 — enough to saturate the API without overwhelming it
- `max_connections` doesn't matter much here since each batch is 1 request

**For fast models** (scoring, property prediction):
- Raise `batch_size` to 64-128
- Raise `max_concurrent` to 8-10
- Raise `max_connections` to 16-20

### Streaming mode

Streaming mode sends all pipeline-level batch tasks immediately and yields results as they complete to the next stage (filter):

```python
pipeline.run(enable_streaming=True)
# Prediction batches fire concurrently, results flow to next stage as they arrive
```

Streaming uses the same `batch_size` for chunking but fires all batches at once (no `max_concurrent` cap). This gives maximum throughput when the downstream stage is a lightweight filter, at the cost of higher peak memory.

**Generation `n_runs`** fires the same generation call `n_runs` times in parallel:

```python
DirectGenerationConfig(
    model_name="zymctrl",
    item_field="ec_number",
    sequence="3.1.1.101",
    params={"temperature": 1.0, "max_length": 300},
    n_runs=5,               # 5 parallel API calls → ~25 sequences
)
```

---

## Architecture

```
Generation / Input sequences
       |
  _get_initial_data() → DuckDB sequences table → WorkingSet
       |
  run_async() → topological sort → execute levels
       |
  each level: single stage or asyncio.gather(parallel stages)
       |     stage.process_ws(WorkingSet, datastore) → (WorkingSet, StageResult)
       |
  get_final_data() → materialize_working_set() → DataFrame
```

### DuckDB Tables

| Table | Purpose |
|---|---|
| `sequences` | Sequences + input columns + hash dedup |
| `predictions` | Cached prediction values (keyed by sequence_id, prediction_type, model_name) |
| `embeddings` | Embedding vectors (FLOAT[]) |
| `structures` | PDB/CIF structures (gzip BLOB) |
| `pipeline_context` | Inter-stage shared state |
| `generation_metadata` | Generation params per sequence |
| `stage_completions` | Resume support |
| `filter_results` | Filter resume support |

### Modules

| File | Key exports |
|---|---|
| `base.py` | `BasePipeline`, `Stage`, `WorkingSet`, `InputSchema`, `PipelineContext` |
| `data.py` | `DataPipeline`, `PredictionStage`, `FilterStage`, `ExtractionSpec`, `EmbeddingSpec` |
| `generative.py` | `GenerativePipeline`, `DirectGenerationConfig`, `FoldingEntity` |
| `mlm_remasking.py` | `MLMRemasker`, `RemaskingConfig` |
| `datastore_duckdb.py` | `DuckDBDataStore` |
| `filters.py` | All filter classes |
| `clustering.py` | `SequenceClusterer`, `DiversityAnalyzer` |
| `visualization.py` | `PipelinePlotter` |
