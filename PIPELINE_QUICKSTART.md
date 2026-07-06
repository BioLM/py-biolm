# Pipeline Quickstart

Quick examples for the `biolmai.pipeline` design-primitive configs.  
Install: `pip install 'biolmai[pipeline]'`

---

## SaturationMutagenesisConfig — single-mutant library + scoring

Generates every single amino-acid substitution at a list of positions and scores each variant with a BioLM scoring model.

```python
import asyncio
from biolmai.pipeline import GenerativePipeline, SaturationMutagenesisConfig

config = SaturationMutagenesisConfig(
    scoring_model="esm2-650m",          # any BioLM model with action="score"
    scoring_action="score",             # "score" or "predict" depending on model
    score_field="score",                # dotted path into API response, e.g. "ddg"
    positions=[10, 15, 42],             # 1-indexed residue positions to scan
    alphabet="ACDEFGHIKLMNPQRSTVWY",    # AA alphabet (default: 20 standard AAs)
    exclude_synonymous=True,            # skip wt→wt substitutions (default True)
    batch_size=32,
    label="sat_scan_round1",
)

pipeline = GenerativePipeline(configs=[config])

df = asyncio.run(pipeline.run_async(sequences=["MKTAYIAKQRQISFVKSHFSRQ"]))
# df has columns: sequence, sat_position, sat_wt_aa, sat_mut_aa, <score_field>
print(df.sort_values("score", ascending=False).head(10))
```

---

## IterativeMaskingDMSConfig — greedy MLM 2-point DMS

Round 1: mask each position independently, take the top-K substitutions per position from MLM logits.  
Round 2: combine all round-1 winners as 2-point variants and rescore.

```python
from biolmai.pipeline import GenerativePipeline, IterativeMaskingDMSConfig

config = IterativeMaskingDMSConfig(
    mlm_model="esm2-650m",             # any BioLM MLM with action="predict" (logits)
    positions=[10, 15, 42],
    top_k=3,                           # keep top-3 AAs per position from round-1
    rounds=2,                          # 1 = round-1 only; 2 = add 2-point combinations
    mask_token="<mask>",               # model-specific mask token
    alphabet="ACDEFGHIKLMNPQRSTVWY",
    exclude_synonymous=True,
    batch_size=16,
    label="itmasking_dms",
)

pipeline = GenerativePipeline(configs=[config])
df = asyncio.run(pipeline.run_async(sequences=["MKTAYIAKQRQISFVKSHFSRQ"]))
# df has columns: sequence, dms_round, dms_pos1, dms_aa1, dms_pos2, dms_aa2
print(df)
```

---

## DirectGenerationConfig — ProteinMPNN / DSM / AntiFold

Generates sequences from structure with inherently generative models.

```python
from biolmai.pipeline import GenerativePipeline, DirectGenerationConfig

with open("my_structure.pdb") as f:
    pdb_str = f.read()

config = DirectGenerationConfig(
    model="proteinmpnn-v48-002",
    action="generate",
    n_runs=10,
    params={"temperature": 0.1},
    label="mpnn_gen",
)

pipeline = GenerativePipeline(configs=[config])
df = asyncio.run(pipeline.run_async(sequences=[pdb_str]))
print(df[["sequence", "global_score"]].head())
```

---

## ProtocolRun.to_duckdb() — load protocol results into DuckDB

After a ProtocolClient run completes, load results into DuckDB for analysis:

```python
import duckdb
from biolmai.protocol_runs import ProtocolClient

client = ProtocolClient()
run = client.submit(protocol_slug="my-protocol", sequences=["MKTAY..."])
run.wait()

# In-memory DuckDB (returns the connection)
con = run.to_duckdb()
print(con.execute("SELECT * FROM my_protocol LIMIT 5").df())

# Persistent file
con = run.to_duckdb(con="results.duckdb", if_exists="replace")
```
