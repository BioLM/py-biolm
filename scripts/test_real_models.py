"""
Real-API tests for untested model types: embeddings, scoring, generation, DNA.

NO MOCKS. Requires BIOLMAI_TOKEN.
Avoids structure prediction (esmfold/alphafold2).

Models tested:
  - esm2-8m (encode) — protein embeddings
  - esmc-300m (score) — log-probability scoring
  - ablang2 (encode) — paired antibody embeddings
  - dnabert2 (encode) — DNA embeddings
  - dsm-150m-base (generate) — sequence generation
  - progen2-oas (generate) — antibody generative model

Run:
    python scripts/test_real_models.py
"""

import asyncio
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from biolmai.pipeline.base import WorkingSet
from biolmai.pipeline.data import DataPipeline, ExtractionSpec
from biolmai.pipeline.datastore_duckdb import DuckDBDataStore
from biolmai.pipeline.filters import RankingFilter, ThresholdFilter, ValidAminoAcidFilter
from biolmai.pipeline.generative import DirectGenerationConfig, GenerativePipeline

TOKEN = os.environ.get("BIOLMAI_TOKEN", "")
if not TOKEN:
    print("ERROR: BIOLMAI_TOKEN not set.")
    sys.exit(1)

# ---------------------------------------------------------------------------
SEQS = [
    "MKILILLGAEKGIGKSTIAKLLAQKFGKIVIETNEKDEDAKSIAEQLGKPFDSVSQLDTIAPQVLAQLLREELSSIMTQIPDVSIVVLDSQGAAITQSALENIRPDYIIVNKMDLKKKDQFAPAAILEQKIRDYFPELDAQSKELEDLFQKAGVEVISQAESFIAQYISQRKDLP",
    "MRVLKFGGTSVANAERFLRVADILESNARQGQVATVLSAPAKITNHLVAMIEKTISGQDALPNISDAERIFAELLTGLAAAQPGFPLAQLKTFVDQEFAQIKHVLHGISGGVGGVATITAPKKVTLLGRDSFEVAVALMK",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLK",
]

VH = [
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSRYTMSWVRQAPGKGLEWVSSISSSSSYIYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDDHYSLDYWGQGTLVTVSS",
    "QVQLQESGPGLVKPSETLSLTCTVSGGSISSYYWSWIRQPPGKGLEWIGYIYYSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARVTSANWFDPWGQGTLVTVSS",
]
VL = [
    "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK",
    "SYELTQPPSVSSGAPGQRVTISCTGSSSNIGAGYDVHWYQQLPGTAPKLLIYDNTNRPSGVPDRFSGSKSGTSASLAISGLRSEDEADYYCATWDDSLSGYVFGGGTKLTVL",
]

DNA_SEQS = [
    "ATGCGATCGATCGATCGATCGATCGATCGATCGATCG",
    "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG",
    "AATTCCGGAATTCCGGAATTCCGGAATTCCGGAATTCC",
]


def _check(df, name, min_rows=1, required_cols=None, no_all_null=True):
    print(f"\n  [{name}] shape={df.shape}")
    if len(df) < min_rows:
        raise AssertionError(f"FAIL: {name} has {len(df)} rows, expected >= {min_rows}")
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise AssertionError(f"FAIL: missing columns {missing}. Has: {list(df.columns)}")
    if no_all_null and len(df) > 0:
        for col in df.columns:
            if df[col].isna().all():
                raise AssertionError(f"FAIL: column '{col}' is ALL NULL ({len(df)} rows)")
    if "sequence_id" in df.columns and df["sequence_id"].nunique() != len(df):
        raise AssertionError(f"FAIL: duplicate sequence_ids")
    print(f"  Columns: {list(df.columns)}")
    for col in df.columns:
        if col in ("sequence_id", "sequence", "length", "hash", "created_at"):
            continue
        n = df[col].notna().sum()
        suffix = ""
        if df[col].dtype in ("float64", "int64", "float32"):
            v = df[col].dropna()
            if len(v):
                suffix = f"  [{v.min():.3f}, {v.max():.3f}]"
        print(f"    {col}: {n}/{len(df)} non-null{suffix}")
    print(f"  OK")


# ===========================================================================
# 1. ESM2-8M embeddings
# ===========================================================================
async def test_esm2_embeddings(tmp):
    """esm2-8m encode → embeddings stored in DuckDB → retrievable."""
    print("\n" + "=" * 70)
    print("TEST: esm2-8m embeddings (encode)")
    print("=" * 70)

    db = tmp / "esm2.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp / "esm2_data")

    pipeline = DataPipeline(sequences=SEQS, datastore=ds, verbose=True)
    pipeline.add_prediction(
        "esm2-8m", action="encode", prediction_type="embedding",
        stage_name="embed_esm2",
    )
    await pipeline.run_async(enable_streaming=False)

    # Verify embeddings stored in DuckDB
    emb_count = ds.conn.execute("SELECT COUNT(*) FROM embeddings WHERE model_name='esm2-8m'").fetchone()[0]
    print(f"\n  Embeddings in DB: {emb_count}")
    assert emb_count == len(SEQS), f"Expected {len(SEQS)} embeddings, got {emb_count}"

    # Verify we can retrieve them
    seq_ids = ds.conn.execute("SELECT sequence_id FROM sequences").df()["sequence_id"].tolist()
    emb_map = ds.get_embeddings_bulk(seq_ids, model_name="esm2-8m")
    assert len(emb_map) == len(SEQS), f"Bulk fetch returned {len(emb_map)}, expected {len(SEQS)}"

    for sid, emb in emb_map.items():
        assert isinstance(emb, np.ndarray), f"seq_id {sid}: not ndarray"
        assert emb.shape[0] > 0, f"seq_id {sid}: empty embedding"
        print(f"  seq_id {sid}: shape={emb.shape} dtype={emb.dtype}")

    ds.close()


# ===========================================================================
# 2. ESMC-300M scoring
# ===========================================================================
async def test_esmc_scoring(tmp):
    """esmc-300m score → log_prob stored as prediction → filter on it."""
    print("\n" + "=" * 70)
    print("TEST: esmc-300m scoring (log-probability)")
    print("=" * 70)

    db = tmp / "esmc.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp / "esmc_data")

    pipeline = DataPipeline(sequences=SEQS, datastore=ds, verbose=True)
    pipeline.add_prediction(
        "esmc-300m", action="score", prediction_type="log_prob",
        extractions="log_prob",
        stage_name="score_esmc",
    )
    # Keep top 2 by log_prob (least negative = best)
    pipeline.add_filter(
        RankingFilter("log_prob", n=2, ascending=False),
        stage_name="rank_lp",
        depends_on=["score_esmc"],
    )
    await pipeline.run_async(enable_streaming=False)

    df = pipeline.get_final_data()
    _check(df, "esmc scoring", min_rows=2, required_cols=["log_prob"])

    assert len(df) == 2, f"Expected 2 after top-2 filter, got {len(df)}"
    # log_prob should be negative
    assert (df["log_prob"] < 0).all(), f"log_prob should be negative: {df['log_prob'].tolist()}"
    print(f"\n  Top 2 by log-prob:")
    for _, row in df.iterrows():
        print(f"    {row['sequence'][:35]}...  LP={row['log_prob']:.1f}")

    ds.close()


# ===========================================================================
# 3. AbLang2 paired antibody embeddings
# ===========================================================================
async def test_ablang2_paired(tmp):
    """ablang2 encode with heavy+light → embeddings stored."""
    print("\n" + "=" * 70)
    print("TEST: ablang2 paired embeddings (heavy + light)")
    print("=" * 70)

    df_input = pd.DataFrame({"heavy_chain": VH, "light_chain": VL})
    db = tmp / "ablang2.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp / "ablang2_data")

    pipeline = DataPipeline(
        sequences=df_input,
        input_columns=["heavy_chain", "light_chain"],
        datastore=ds, verbose=True,
    )
    # ablang2 expects {heavy: ..., light: ...}
    pipeline.add_prediction(
        "ablang2", action="encode", prediction_type="embedding",
        stage_name="embed_ablang2",
        item_columns={"heavy": "heavy_chain", "light": "light_chain"},
    )
    await pipeline.run_async(enable_streaming=False)

    emb_count = ds.conn.execute("SELECT COUNT(*) FROM embeddings WHERE model_name='ablang2'").fetchone()[0]
    print(f"\n  Embeddings in DB: {emb_count}")
    assert emb_count == len(VH), f"Expected {len(VH)} embeddings, got {emb_count}"

    seq_ids = ds.conn.execute("SELECT sequence_id FROM sequences").df()["sequence_id"].tolist()
    emb_map = ds.get_embeddings_bulk(seq_ids, model_name="ablang2")
    for sid, emb in emb_map.items():
        print(f"  seq_id {sid}: shape={emb.shape}")
        assert emb.shape[0] > 0

    ds.close()


# ===========================================================================
# 4. DNABERT2 DNA embeddings
# ===========================================================================
async def test_dnabert2_embeddings(tmp):
    """dnabert2 encode DNA sequences → embeddings stored."""
    print("\n" + "=" * 70)
    print("TEST: dnabert2 DNA embeddings")
    print("=" * 70)

    db = tmp / "dnabert2.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp / "dnabert2_data")

    pipeline = DataPipeline(sequences=DNA_SEQS, datastore=ds, verbose=True)
    pipeline.add_prediction(
        "dnabert2", action="encode", prediction_type="embedding",
        stage_name="embed_dna",
    )
    await pipeline.run_async(enable_streaming=False)

    emb_count = ds.conn.execute("SELECT COUNT(*) FROM embeddings WHERE model_name='dnabert2'").fetchone()[0]
    print(f"\n  Embeddings in DB: {emb_count}")
    assert emb_count == len(DNA_SEQS), f"Expected {len(DNA_SEQS)}, got {emb_count}"

    seq_ids = ds.conn.execute("SELECT sequence_id FROM sequences").df()["sequence_id"].tolist()
    emb_map = ds.get_embeddings_bulk(seq_ids, model_name="dnabert2")
    for sid, emb in emb_map.items():
        print(f"  seq_id {sid}: shape={emb.shape}")
        assert emb.shape[0] > 0

    ds.close()


# ===========================================================================
# 5. DSM-150M generation → scoring → filter
# ===========================================================================
async def test_dsm_generation(tmp):
    """dsm-150m-base generate → esmc-300m score → filter by log-prob."""
    print("\n" + "=" * 70)
    print("TEST: DSM-150M generation → ESMC scoring → filter")
    print("=" * 70)

    db = tmp / "dsm.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp / "dsm_data")

    config = DirectGenerationConfig(
        model_name="dsm-150m-base",
        item_field="sequence",
        sequence=SEQS[0],
        params={"num_sequences": 5, "temperature": 1.0},
    )

    pipeline = GenerativePipeline(
        generation_configs=[config],
        datastore=ds,
        verbose=True,
    )
    pipeline.add_prediction(
        "esmc-300m", action="score", prediction_type="log_prob",
        extractions="log_prob",
        stage_name="score_lp",
    )
    pipeline.add_filter(
        RankingFilter("log_prob", n=3, ascending=False),
        stage_name="rank_top3",
        depends_on=["score_lp"],
    )
    await pipeline.run_async(enable_streaming=False)

    df = pipeline.get_final_data()
    _check(df, "DSM gen+score", min_rows=1, required_cols=["sequence", "log_prob"])

    # Generated sequences should be unique
    assert df["sequence"].nunique() == len(df), "Duplicate generated sequences"
    # log_prob should be negative
    assert (df["log_prob"] < 0).all()

    # Check generation metadata
    gen_count = ds.conn.execute("SELECT COUNT(*) FROM generation_metadata").fetchone()[0]
    print(f"\n  Generation metadata rows: {gen_count}")
    assert gen_count >= 1, "No generation metadata stored"

    print(f"\n  Top {len(df)} by log-prob:")
    for _, row in df.iterrows():
        print(f"    {row['sequence'][:40]}...  LP={row['log_prob']:.1f}")

    ds.close()


# ===========================================================================
# 6. ProGen2-OAS antibody generation
# ===========================================================================
async def test_progen2_generation(tmp):
    """progen2-oas generate antibody sequences from VH seed."""
    print("\n" + "=" * 70)
    print("TEST: progen2-oas antibody generation")
    print("=" * 70)

    db = tmp / "progen2.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp / "progen2_data")

    config = DirectGenerationConfig(
        model_name="progen2-oas",
        item_field="context",
        sequence=VH[0][:20],  # Short seed from VH
        params={"temperature": 1.0, "max_length": 80},
    )

    pipeline = GenerativePipeline(
        generation_configs=[config],
        datastore=ds,
        verbose=True,
    )
    await pipeline.run_async(enable_streaming=False)

    df = pipeline.get_final_data()
    _check(df, "progen2-oas gen", min_rows=1, required_cols=["sequence"])

    assert df["sequence"].nunique() == len(df), "Duplicate generated sequences"

    print(f"\n  Generated {len(df)} sequences:")
    for _, row in df.iterrows():
        print(f"    {row['sequence'][:60]}...  len={len(row['sequence'])}")

    ds.close()


# ===========================================================================
# 7. Full pipeline: embed → Tm+Sol → cluster-ready
# ===========================================================================
async def test_embed_plus_predict(tmp):
    """ESM2 embed + Tm + solubility in parallel → export with all columns."""
    print("\n" + "=" * 70)
    print("TEST: ESM2 embed + Tm + solubility (parallel)")
    print("=" * 70)

    db = tmp / "full.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp / "full_data")

    pipeline = DataPipeline(sequences=SEQS, datastore=ds, verbose=True)
    pipeline.add_prediction(
        "esm2-8m", action="encode", prediction_type="embedding",
        stage_name="embed",
    )
    pipeline.add_prediction(
        "temberture-regression", prediction_type="tm",
        extractions="prediction", stage_name="predict_tm",
    )
    pipeline.add_prediction(
        "soluprot", prediction_type="solubility",
        extractions="soluble", stage_name="predict_sol",
    )
    await pipeline.run_async(enable_streaming=False)

    df = pipeline.get_final_data()
    _check(df, "full pipeline", min_rows=3, required_cols=["tm", "solubility"])

    # Embeddings stored
    emb_count = ds.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert emb_count == len(SEQS), f"Expected {len(SEQS)} embeddings, got {emb_count}"

    # Both predictions non-null
    assert df["tm"].notna().all()
    assert df["solubility"].notna().all()

    print(f"\n  Results:")
    for _, row in df.iterrows():
        print(f"    {row['sequence'][:30]}...  Tm={row['tm']:.1f}  Sol={row['solubility']:.3f}")

    # Verify export
    df_exp = ds.export_to_dataframe(include_predictions=True)
    assert "tm" in df_exp.columns and "solubility" in df_exp.columns
    print(f"  Export has {len(df_exp.columns)} columns: {list(df_exp.columns)}")

    ds.close()


# ===========================================================================
async def main():
    results = {}
    tests = [
        ("ESM2-8M embeddings", test_esm2_embeddings),
        ("ESMC-300M scoring", test_esmc_scoring),
        ("AbLang2 paired", test_ablang2_paired),
        ("DNABERT2 DNA", test_dnabert2_embeddings),
        ("DSM gen → score", test_dsm_generation),
        ("ProGen2-OAS gen", test_progen2_generation),
        ("Embed + Tm + Sol", test_embed_plus_predict),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for name, func in tests:
            try:
                await func(tmp_path)
                results[name] = "PASS"
            except Exception as e:
                results[name] = f"FAIL: {e}"
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, status in results.items():
        icon = "+" if status == "PASS" else "X"
        print(f"  [{icon}] {name}: {status}")
        if status != "PASS":
            all_pass = False

    if all_pass:
        print(f"\nAll {len(results)} tests passed!")
    else:
        failed = sum(1 for s in results.values() if s != "PASS")
        print(f"\n{failed}/{len(results)} tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
