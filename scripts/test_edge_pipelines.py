"""
Edge-case pipeline integration tests — exercises multi-column input, pipeline
context, diverse filter combos, and various model response formats.

All API calls are mocked so this runs without BIOLMAI_TOKEN.

Run:
    python scripts/test_edge_pipelines.py
"""

import asyncio
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd

from biolmai.pipeline.base import PipelineContext, WorkingSet
from biolmai.pipeline.data import DataPipeline, PredictionStage
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

# ---------------------------------------------------------------------------
# Shared test sequences
# ---------------------------------------------------------------------------
HEAVY_CHAINS = [
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSRYTMSWVRQAPGKGLEWVSSISSSSSYIYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARYYDDHYSLDYWGQGTLVTVSS",
    "QVQLQESGPGLVKPSETLSLTCTVSGGSISSYYWSWIRQPPGKGLEWIGYIYYSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARVTSANWFDPWGQGTLVTVSS",
    "EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVSNIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARDGNYYGSGFAYWGQGTLVTVSS",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGIINPSGGSTSYAQKFQGRVTMTRDTSTSTVYMELSSLRSEDTAVYYCARSTYYGGDWFDPWGQGTLVTVSS",
    "EVQLVESGGGLVQPNNSLRLSCAASGFTLDDYAMGWYRQAPGKQRELVSTITGGGSITYYADSVKGRFTISRDNAKNTLYLQMNSLKPEDTAVYYCARRGSYYDSSGYNYWGQGTLVTVSS",
]

LIGHT_CHAINS = [
    "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK",
    "SYELTQPPSVSSGAPGQRVTISCTGSSSNIGAGYDVHWYQQLPGTAPKLLIYDNTNRPSGVPDRFSGSKSGTSASLAISGLRSEDEADYYCATWDDSLSGYVFGGGTKLTVL",
    "DIVMTQSPDSLAVSLGERATINCKSSQSVLYSSNNKNYLAWYQQKPGQPPKLLIYWASTRESGVPDRFSGSGSGTDFTLTISSLQAEDVAVYYCQQYYSTPLTFGGGTKVEIK",
    "SYELTQPPSVSVSPGQTARITCSGDALPKQYAYWYQQKSGQAPVLVIYKDSERPSGIPERFSGSSSGTTVTLTISGVQAEDEADYYCQSADSSGTYVFGGGTKLTVL",
    "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK",
]

SINGLE_CHAIN_SEQS = [
    "MKVLKAGISFTLGSGVILGAFIFFVLVKDNPKLTTGELTLQTAVEMAPQPIAGLSHEIQGVGYEITDDMIEPVTLGSGLVNPVGQILMGGIDGGISALPNLEKFNKTIEGLPDKLKPTFTTIEDAMKLYAAYGG",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLK",
    "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGAAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFH",
    "MRIILLGAPGAGKGTQAKFIEEKGYIPHISTGDMLRAAVKSGSELGKQAKDIMDAGKLVTDELVIALVKERIAQEDCRNGFLLDGFPRTIPQADAMKEAGINVDYVLEFDVPDELIVDRIVGRRVHAPSG",
    "MKILILLGAEKGIGKSTIAKLLAQKFGKIVIETNEKDEDAKSIAEQLGKPFDSVSQLDTIAPQVLAQLLREELSSIMTQIPDVSIVVLDSQGAAITQSALENIRPDYIIVNKMDLKKKDQFAPAAILEQK",
    "MRVLKFGGTSVANAERFLRVADILESNARQGQVATVLSAPAKITNHLVAMIEKTISGQDALPNISDAERIFAELLTGLAAAQPGFPLAQLKTFVDQEFAQIKHVLHGISGGVGGVATITAPKKVTLLGRDSFEV",
    "MKQLEDKVEELLSKNYHLENEVARLKKLVGERMKTAYIAKQRQGHQAMAEIKQKLREEKNQ",
    "MASMTGGQQMGRGSMDELEQKLISEEDLNSAVDHHHHHHMKTAYIAKQRQ",
]

# Sequences with non-standard residues (should be caught by ValidAminoAcidFilter)
BAD_SEQUENCES = [
    "MKTXAYIAKQRQ",  # X
    "MKLLIV12345",  # digits
    "EVQLVES*GGGLVQ",  # *
]


def make_api_mock(
    values=None,
    generate_seqs=None,
    response_format="scalar",
    model_name="mock",
):
    """Return an AsyncMock API client with configurable response formats."""
    mock = AsyncMock()

    async def _predict(items, params=None):
        base = values or [65.0] * len(items)
        if response_format == "scalar":
            return [
                {"melting_temperature": base[i % len(base)]} for i in range(len(items))
            ]
        elif response_format == "multi_field":
            return [
                {
                    "melting_temperature": base[i % len(base)],
                    "solubility_score": 0.3 + (i % 5) * 0.15,
                    "prediction": base[i % len(base)] * 0.9,
                }
                for i in range(len(items))
            ]
        elif response_format == "plddt_array":
            return [
                {
                    "plddt": [70.0 + j for j in range(10)],
                    "mean_plddt": base[i % len(base)],
                    "ptm": 0.85 + (i % 3) * 0.05,
                }
                for i in range(len(items))
            ]
        elif response_format == "paired_score":
            # Antibody model response: score per pair
            return [
                {
                    "score": base[i % len(base)],
                    "global_score": base[i % len(base)] * -0.1,
                }
                for i in range(len(items))
            ]
        else:
            return [{"value": base[i % len(base)]} for i in range(len(items))]

    async def _encode(items, params=None):
        dim = 32
        return [{"embedding": list(np.random.randn(dim).astype(float))} for _ in items]

    async def _generate(items, params=None):
        n = params.get("num_sequences", 5) if params else 5
        seqs = generate_seqs or [f"GENERATED{i:04d}AAAA" for i in range(n)]
        return [{"sequence": s} for s in seqs]

    async def _score(items, params=None):
        return [
            {"log_probability": -100.0 + i * 3.5} for i in range(len(items))
        ]

    mock.predict = AsyncMock(side_effect=_predict)
    mock.encode = AsyncMock(side_effect=_encode)
    mock.generate = AsyncMock(side_effect=_generate)
    mock.score = AsyncMock(side_effect=_score)
    mock.shutdown = AsyncMock()
    return mock


def _check_df(df, name, min_rows=1, required_cols=None, no_all_null_cols=True):
    """Validate a DataFrame and print diagnostics."""
    print(f"\n  [{name}] shape={df.shape}")
    if len(df) < min_rows:
        raise AssertionError(
            f"  FAIL: {name} has {len(df)} rows, expected >= {min_rows}"
        )

    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise AssertionError(
                f"  FAIL: {name} missing columns: {missing}. Has: {list(df.columns)}"
            )

    if no_all_null_cols:
        for col in df.columns:
            if df[col].isna().all() and len(df) > 0:
                raise AssertionError(
                    f"  FAIL: {name} column '{col}' is ALL NULL ({len(df)} rows)"
                )

    # Check for uniqueness of sequence_id
    if "sequence_id" in df.columns:
        n_unique = df["sequence_id"].nunique()
        n_total = len(df)
        if n_unique != n_total:
            raise AssertionError(
                f"  FAIL: {name} has {n_total} rows but only {n_unique} unique sequence_ids"
            )

    # Print sample
    print(f"  Columns: {list(df.columns)}")
    if "sequence" in df.columns:
        n_unique_seq = df["sequence"].nunique()
        print(f"  Unique sequences: {n_unique_seq}/{len(df)}")
    for col in df.columns:
        if col in ("sequence_id", "sequence", "length", "hash", "created_at"):
            continue
        n_notna = df[col].notna().sum()
        print(f"    {col}: {n_notna}/{len(df)} non-null", end="")
        if df[col].dtype in ("float64", "int64", "float32"):
            vals = df[col].dropna()
            if len(vals) > 0:
                print(
                    f"  min={vals.min():.3f} max={vals.max():.3f} mean={vals.mean():.3f}",
                    end="",
                )
        print()

    print(f"  OK ✓")


# ===========================================================================
# Pipeline 1: Multi-column antibody input → parallel predictions → filters
# ===========================================================================
async def pipeline_1_antibody_multi_column(tmp_path):
    """Antibody heavy+light input columns through prediction and filtering."""
    print("\n" + "=" * 70)
    print("PIPELINE 1: Antibody multi-column (heavy + light)")
    print("=" * 70)

    df_input = pd.DataFrame(
        {"heavy_chain": HEAVY_CHAINS, "light_chain": LIGHT_CHAINS}
    )
    print(f"Input: {len(df_input)} antibody H+L pairs")

    db = tmp_path / "p1.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "p1_data")

    # Values: indices 0..4 get [80, 55, 70, 40, 90] for Tm
    tm_values = [80.0, 55.0, 70.0, 40.0, 90.0]
    sol_values = [0.8, 0.6, 0.3, 0.9, 0.7]

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        call_count = [0]
        def _make_mock(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                return make_api_mock(values=tm_values)
            else:
                return make_api_mock(values=sol_values)
        MockCls.side_effect = _make_mock

        pipeline = DataPipeline(
            sequences=df_input,
            input_columns=["heavy_chain", "light_chain"],
            datastore=ds,
            verbose=False,
        )

        # Parallel predictions on antibody pairs
        pipeline.add_prediction(
            "temberture-regression",
            extractions="melting_temperature",
            columns="tm",
            stage_name="predict_tm",
        )
        pipeline.add_prediction(
            "soluprot",
            extractions="melting_temperature",  # mock returns this key
            columns="solubility",
            stage_name="predict_sol",
        )

        # Filter: Tm >= 60
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=60.0),
            stage_name="filter_tm",
            depends_on=["predict_tm"],
        )

        # Rank: top 2 by solubility among survivors
        pipeline.add_filter(
            RankingFilter("solubility", n=2, ascending=False),
            stage_name="rank_sol",
            depends_on=["filter_tm", "predict_sol"],
        )

        await pipeline.run_async(enable_streaming=False)

    df_final = pipeline.get_final_data()
    _check_df(
        df_final,
        "Pipeline 1 final",
        min_rows=1,
        required_cols=["heavy_chain", "light_chain", "tm"],
    )

    # Verify multi-column columns survived
    assert "heavy_chain" in df_final.columns, "heavy_chain missing from final data"
    assert "light_chain" in df_final.columns, "light_chain missing from final data"

    # Verify DuckDB has the columns
    db_cols = ds.conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='sequences'"
    ).fetchall()
    col_names = {r[0] for r in db_cols}
    assert "heavy_chain" in col_names, "heavy_chain not in DuckDB sequences table"
    assert "light_chain" in col_names, "light_chain not in DuckDB sequences table"

    ds.close()
    print("  Pipeline 1 passed!")


# ===========================================================================
# Pipeline 2: Multi-column dedup edge cases
# ===========================================================================
async def pipeline_2_dedup_edge_cases(tmp_path):
    """Test dedup: same H + different L = different rows; duplicate pairs removed."""
    print("\n" + "=" * 70)
    print("PIPELINE 2: Dedup edge cases (multi-column)")
    print("=" * 70)

    df_input = pd.DataFrame(
        {
            "heavy_chain": [
                "EVQLVES",
                "EVQLVES",  # same H, different L
                "EVQLVES",  # same H, same L as row 0 (dup)
                "QVQLQES",  # different H
                "QVQLQES",  # same H+L as row 3 (dup)
            ],
            "light_chain": [
                "DIQMTQS",
                "EIVLTQS",
                "DIQMTQS",  # dup of row 0
                "DIQMTQS",
                "DIQMTQS",  # dup of row 3
            ],
        }
    )
    print(f"Input: {len(df_input)} rows, expect 3 unique after dedup")

    db = tmp_path / "p2.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "p2_data")

    pipeline = DataPipeline(
        sequences=df_input,
        input_columns=["heavy_chain", "light_chain"],
        datastore=ds,
        verbose=False,
    )
    df = await pipeline._get_initial_data()

    assert len(df) == 3, f"Expected 3 unique rows, got {len(df)}"
    assert df["sequence_id"].nunique() == 3
    print(f"  Dedup: 5 → {len(df)} unique rows ✓")

    # Verify second insert of same data returns same IDs
    df2 = await pipeline._get_initial_data()
    assert list(df["sequence_id"]) == list(df2["sequence_id"]), "IDs changed on re-insert"
    total_in_db = ds.conn.execute("SELECT COUNT(*) FROM sequences").fetchone()[0]
    assert total_in_db == 3, f"Expected 3 rows in DB, got {total_in_db}"
    print(f"  Re-insert: same IDs, still 3 rows in DB ✓")

    ds.close()
    print("  Pipeline 2 passed!")


# ===========================================================================
# Pipeline 3: ValidAminoAcidFilter on custom column
# ===========================================================================
async def pipeline_3_valid_aa_custom_column(tmp_path):
    """ValidAminoAcidFilter with column='heavy_chain'."""
    print("\n" + "=" * 70)
    print("PIPELINE 3: ValidAminoAcidFilter on custom column")
    print("=" * 70)

    df_input = pd.DataFrame(
        {
            "heavy_chain": [
                "EVQLVES",  # valid
                "EVXLVES",  # X = invalid
                "QVQLQES",  # valid
                "EV1LVES",  # 1 = invalid
            ],
            "light_chain": [
                "DIQMTQS",
                "DIQMTQS",
                "EIVLTQS",
                "DIQMTQS",
            ],
        }
    )

    db = tmp_path / "p3.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "p3_data")

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=[65.0, 70.0])

        pipeline = DataPipeline(
            sequences=df_input,
            input_columns=["heavy_chain", "light_chain"],
            datastore=ds,
            verbose=False,
        )
        # Filter heavy chain for valid amino acids
        pipeline.add_filter(
            ValidAminoAcidFilter(column="heavy_chain", verbose=False),
            stage_name="filter_valid_hc",
        )
        await pipeline.run_async(enable_streaming=False)

    df_final = pipeline.get_final_data()
    _check_df(df_final, "Pipeline 3 final", min_rows=2)

    # Only rows 0 and 2 should survive (valid heavy chains)
    assert len(df_final) == 2, f"Expected 2 valid rows, got {len(df_final)}"
    hcs = set(df_final["heavy_chain"].tolist())
    assert "EVXLVES" not in hcs, "Invalid heavy chain was not filtered"
    assert "EV1LVES" not in hcs, "Invalid heavy chain was not filtered"
    print(f"  Filtered to {len(df_final)} valid heavy chains ✓")

    ds.close()
    print("  Pipeline 3 passed!")


# ===========================================================================
# Pipeline 4: Single-chain with bad seqs + chained filters
# ===========================================================================
async def pipeline_4_chained_filters(tmp_path):
    """Single-chain: ValidAA → Length → Tm threshold → top-N ranking."""
    print("\n" + "=" * 70)
    print("PIPELINE 4: Single-chain, chained filters (4 stages)")
    print("=" * 70)

    all_seqs = SINGLE_CHAIN_SEQS + BAD_SEQUENCES
    print(f"Input: {len(all_seqs)} sequences ({len(BAD_SEQUENCES)} with bad residues)")

    db = tmp_path / "p4.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "p4_data")

    # Tm values cycling: gives range of values across sequences
    tm_values = [82.0, 45.0, 71.0, 38.0, 90.0, 55.0, 63.0, 79.0]

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=tm_values)

        pipeline = DataPipeline(
            sequences=all_seqs,
            datastore=ds,
            verbose=False,
        )

        # 1. Filter invalid amino acids
        pipeline.add_filter(
            ValidAminoAcidFilter(verbose=False),
            stage_name="filter_valid",
        )
        # 2. Filter by length
        pipeline.add_filter(
            SequenceLengthFilter(min_length=50),
            stage_name="filter_length",
            depends_on=["filter_valid"],
        )
        # 3. Predict Tm
        pipeline.add_prediction(
            "temberture-regression",
            extractions="melting_temperature",
            columns="tm",
            stage_name="predict_tm",
            depends_on=["filter_length"],
        )
        # 4. Filter Tm >= 60
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=60.0),
            stage_name="filter_tm",
            depends_on=["predict_tm"],
        )
        # 5. Rank top 3
        pipeline.add_filter(
            RankingFilter("tm", n=3, ascending=False),
            stage_name="rank_top3",
            depends_on=["filter_tm"],
        )

        await pipeline.run_async(enable_streaming=False)

    df_final = pipeline.get_final_data()
    _check_df(
        df_final,
        "Pipeline 4 final",
        min_rows=1,
        required_cols=["sequence", "tm"],
    )

    # Verify no bad sequences survived
    for seq in BAD_SEQUENCES:
        assert seq not in df_final["sequence"].values, f"Bad sequence survived: {seq}"

    # Verify all Tm values >= 60
    assert (df_final["tm"] >= 60.0).all(), f"Some Tm values < 60: {df_final['tm'].tolist()}"
    assert len(df_final) <= 3, f"Top-3 filter let through {len(df_final)} rows"

    # Print stage funnel
    print("\n  Stage funnel:")
    for name, result in pipeline.stage_results.items():
        print(f"    {name}: {result.input_count} → {result.output_count}")

    ds.close()
    print("  Pipeline 4 passed!")


# ===========================================================================
# Pipeline 5: Generation → prediction → filter (GenerativePipeline)
# ===========================================================================
async def pipeline_5_generative_pipeline(tmp_path):
    """GenerativePipeline: generate seqs → predict Tm → filter → rank."""
    print("\n" + "=" * 70)
    print("PIPELINE 5: Generative pipeline (DSM-style generation)")
    print("=" * 70)

    # Generate 20 unique sequences
    gen_seqs = [
        f"MKTAYIAKQRQGHQAMAEIKQKLR{'AEKL' * (i + 3)}" for i in range(20)
    ]
    print(f"Will generate {len(gen_seqs)} sequences")

    db = tmp_path / "p5.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "p5_data")

    tm_values = [82.0, 45.0, 71.0, 38.0, 90.0, 55.0, 63.0, 79.0, 50.0, 68.0,
                 42.0, 88.0, 75.0, 33.0, 61.0, 57.0, 84.0, 46.0, 72.0, 39.0]

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls1, \
         patch("biolmai.pipeline.generative.BioLMApiClient") as MockCls2:
        MockCls1.return_value = make_api_mock(values=tm_values)
        MockCls2.return_value = make_api_mock(generate_seqs=gen_seqs)

        config = DirectGenerationConfig(
            model_name="dsm-150m-base",
            item_field="sequence",
            sequence="MKTAYIAKQRQ",
            params={"num_sequences": 20, "temperature": 1.0},
        )

        pipeline = GenerativePipeline(
            generation_configs=[config],
            datastore=ds,
            verbose=False,
        )
        pipeline.add_prediction(
            "temberture-regression",
            extractions="melting_temperature",
            columns="tm",
            stage_name="predict_tm",
        )
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=60.0),
            stage_name="filter_tm",
            depends_on=["predict_tm"],
        )
        pipeline.add_filter(
            RankingFilter("tm", n=5, ascending=False),
            stage_name="rank_top5",
            depends_on=["filter_tm"],
        )

        await pipeline.run_async(enable_streaming=False)

    df_final = pipeline.get_final_data()
    _check_df(
        df_final,
        "Pipeline 5 final",
        min_rows=1,
        required_cols=["sequence", "tm"],
    )

    # Verify generated sequences are unique
    assert df_final["sequence"].nunique() == len(df_final), "Duplicate sequences in output"
    assert (df_final["tm"] >= 60.0).all(), f"Some Tm < 60: {df_final['tm'].tolist()}"
    assert len(df_final) <= 5, f"Top-5 filter let through {len(df_final)}"

    # Verify generation metadata was stored
    gen_meta_count = ds.conn.execute(
        "SELECT COUNT(*) FROM generation_metadata"
    ).fetchone()[0]
    assert gen_meta_count >= len(gen_seqs), (
        f"Expected >= {len(gen_seqs)} gen_metadata rows, got {gen_meta_count}"
    )
    print(f"  Generation metadata: {gen_meta_count} rows ✓")

    ds.close()
    print("  Pipeline 5 passed!")


# ===========================================================================
# Pipeline 6: Pipeline context round-trip
# ===========================================================================
async def pipeline_6_context_usage(tmp_path):
    """Test pipeline context: set values, verify persistence across stages."""
    print("\n" + "=" * 70)
    print("PIPELINE 6: Pipeline context round-trip")
    print("=" * 70)

    db = tmp_path / "p6.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "p6_data")

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=[65.0, 70.0, 80.0])

        pipeline = DataPipeline(
            sequences=SINGLE_CHAIN_SEQS[:3],
            datastore=ds,
            verbose=False,
        )
        pipeline.add_prediction(
            "temberture-regression",
            extractions="melting_temperature",
            columns="tm",
            stage_name="predict_tm",
        )

        # Set context before running
        pipeline.context.set("experiment_name", "test_run")
        pipeline.context.set("params", {"batch_size": 32, "temperature": 0.5})
        pipeline.context.set("tags", ["thermostable", "engineered"])

        await pipeline.run_async(enable_streaming=False)

    # Verify context persists
    assert pipeline.context.get("experiment_name") == "test_run"
    assert pipeline.context.get("params") == {"batch_size": 32, "temperature": 0.5}
    assert pipeline.context.get("tags") == ["thermostable", "engineered"]
    assert pipeline.context.get("missing_key", "default") == "default"
    print("  Context round-trip: ✓")

    # Verify context isolation: new PipelineContext with different run_id sees nothing
    ctx2 = PipelineContext(ds, "other_run")
    assert ctx2.get("experiment_name") is None
    print("  Context isolation: ✓")

    # Verify structure storage & retrieval via context
    seq_id = ds.add_sequence("MKTAYIAKQRQ")
    ds.add_structure(seq_id, "esmfold", structure_str="ATOM 1 CA ALA A 1 0.0 0.0 0.0")
    struct = pipeline.context.get_structure(seq_id, "esmfold")
    assert struct is not None and "ATOM" in struct["structure_str"]
    print("  Structure via context: ✓")

    ds.close()
    print("  Pipeline 6 passed!")


# ===========================================================================
# Pipeline 7: HammingDistance + DiversitySampling (non-SQL filters)
# ===========================================================================
async def pipeline_7_non_sql_filters(tmp_path):
    """Filters that can't be expressed as SQL: Hamming distance + custom."""
    print("\n" + "=" * 70)
    print("PIPELINE 7: Non-SQL filters (Hamming, Custom)")
    print("=" * 70)

    reference = SINGLE_CHAIN_SEQS[0]
    print(f"Reference sequence: {reference[:30]}... ({len(reference)} aa)")

    db = tmp_path / "p7.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "p7_data")

    tm_values = [82.0, 65.0, 71.0, 48.0, 90.0, 55.0, 73.0, 79.0]

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=tm_values)

        pipeline = DataPipeline(
            sequences=SINGLE_CHAIN_SEQS,
            datastore=ds,
            verbose=False,
        )
        pipeline.add_prediction(
            "temberture-regression",
            extractions="melting_temperature",
            columns="tm",
            stage_name="predict_tm",
        )
        # Custom filter: sequences starting with 'M'
        pipeline.add_filter(
            CustomFilter(
                lambda df: df[df["sequence"].str.startswith("M")],
                name="starts_with_M",
            ),
            stage_name="filter_m",
            depends_on=["predict_tm"],
        )
        # Threshold on Tm
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=60.0),
            stage_name="filter_tm",
            depends_on=["filter_m"],
        )

        await pipeline.run_async(enable_streaming=False)

    df_final = pipeline.get_final_data()
    _check_df(df_final, "Pipeline 7 final", min_rows=1)

    # All surviving sequences should start with M and have Tm >= 60
    assert all(s.startswith("M") for s in df_final["sequence"]), "Non-M sequence survived"
    assert (df_final["tm"] >= 60.0).all(), "Some Tm < 60"

    ds.close()
    print("  Pipeline 7 passed!")


# ===========================================================================
# Pipeline 8: Resume / cache reuse
# ===========================================================================
async def pipeline_8_resume_and_cache(tmp_path):
    """Run pipeline twice with resume=True, verify no redundant API calls."""
    print("\n" + "=" * 70)
    print("PIPELINE 8: Resume + cache reuse")
    print("=" * 70)

    db = tmp_path / "p8.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "p8_data")
    run_id = "test_resume_001"

    tm_values = [82.0, 55.0, 71.0, 48.0, 90.0]
    call_counter = {"predict": 0}

    def _counting_mock():
        mock = make_api_mock(values=tm_values)
        original_predict = mock.predict.side_effect
        async def _counting_predict(items, params=None):
            call_counter["predict"] += len(items)
            return await original_predict(items, params)
        mock.predict = AsyncMock(side_effect=_counting_predict)
        return mock

    # Run 1
    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = _counting_mock()

        pipeline1 = DataPipeline(
            sequences=SINGLE_CHAIN_SEQS[:5],
            datastore=ds,
            run_id=run_id,
            verbose=False,
        )
        pipeline1.add_prediction(
            "temberture-regression",
            extractions="melting_temperature",
            columns="tm",
            stage_name="predict_tm",
        )
        pipeline1.add_filter(
            ThresholdFilter("tm", min_value=60.0),
            stage_name="filter_tm",
            depends_on=["predict_tm"],
        )
        await pipeline1.run_async(enable_streaming=False)

    df1 = pipeline1.get_final_data()
    calls_run1 = call_counter["predict"]
    print(f"  Run 1: {calls_run1} API calls, {len(df1)} final rows")

    # Run 2: resume — should skip prediction stage entirely
    call_counter["predict"] = 0

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = _counting_mock()

        pipeline2 = DataPipeline(
            sequences=SINGLE_CHAIN_SEQS[:5],
            datastore=ds,
            run_id=run_id,
            resume=True,
            verbose=False,
        )
        pipeline2.add_prediction(
            "temberture-regression",
            extractions="melting_temperature",
            columns="tm",
            stage_name="predict_tm",
        )
        pipeline2.add_filter(
            ThresholdFilter("tm", min_value=60.0),
            stage_name="filter_tm",
            depends_on=["predict_tm"],
        )
        await pipeline2.run_async(enable_streaming=False)

    df2 = pipeline2.get_final_data()
    calls_run2 = call_counter["predict"]
    print(f"  Run 2 (resume): {calls_run2} API calls, {len(df2)} final rows")

    # Resume should reuse stage completion → 0 API calls
    assert calls_run2 == 0, f"Resume made {calls_run2} API calls, expected 0"
    assert len(df2) == len(df1), f"Resume output differs: {len(df2)} vs {len(df1)}"

    ds.close()
    print("  Pipeline 8 passed!")


# ===========================================================================
# Pipeline 9: export_to_dataframe with extra columns
# ===========================================================================
async def pipeline_9_export_dataframe_extra_cols(tmp_path):
    """export_to_dataframe should include input_columns from sequences table."""
    print("\n" + "=" * 70)
    print("PIPELINE 9: export_to_dataframe with extra columns")
    print("=" * 70)

    df_input = pd.DataFrame(
        {
            "heavy_chain": HEAVY_CHAINS[:3],
            "light_chain": LIGHT_CHAINS[:3],
        }
    )

    db = tmp_path / "p9.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "p9_data")

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=[80.0, 55.0, 70.0])

        pipeline = DataPipeline(
            sequences=df_input,
            input_columns=["heavy_chain", "light_chain"],
            datastore=ds,
            verbose=False,
        )
        pipeline.add_prediction(
            "temberture-regression",
            extractions="melting_temperature",
            columns="tm",
            stage_name="predict_tm",
        )
        await pipeline.run_async(enable_streaming=False)

    # Test export_to_dataframe
    df_export = ds.export_to_dataframe(include_predictions=True)
    _check_df(
        df_export,
        "export_to_dataframe",
        min_rows=3,
        required_cols=["heavy_chain", "light_chain", "tm"],
    )

    # Test materialize_working_set
    ws = WorkingSet.from_ids(
        ds.conn.execute("SELECT sequence_id FROM sequences").df()["sequence_id"].tolist()
    )
    df_mat = ds.materialize_working_set(ws)
    assert "heavy_chain" in df_mat.columns, "heavy_chain missing from materialize"
    assert "light_chain" in df_mat.columns, "light_chain missing from materialize"
    print(f"  materialize_working_set: {len(df_mat)} rows with extra columns ✓")

    ds.close()
    print("  Pipeline 9 passed!")


# ===========================================================================
# Pipeline 10: Backward compat — sequence list still works
# ===========================================================================
async def pipeline_10_backward_compat(tmp_path):
    """Classic DataPipeline(sequences=[...]) still works after multi-column changes."""
    print("\n" + "=" * 70)
    print("PIPELINE 10: Backward compatibility (sequence list)")
    print("=" * 70)

    db = tmp_path / "p10.duckdb"
    ds = DuckDBDataStore(db_path=db, data_dir=tmp_path / "p10_data")

    tm_values = [82.0, 45.0, 71.0, 38.0, 90.0, 55.0, 63.0, 79.0]

    with patch("biolmai.pipeline.data.BioLMApiClient") as MockCls:
        MockCls.return_value = make_api_mock(values=tm_values)

        pipeline = DataPipeline(
            sequences=SINGLE_CHAIN_SEQS,
            datastore=ds,
            verbose=False,
        )
        pipeline.add_filter(
            SequenceLengthFilter(min_length=50),
            stage_name="filter_length",
        )
        pipeline.add_prediction(
            "temberture-regression",
            extractions="melting_temperature",
            columns="tm",
            stage_name="predict_tm",
            depends_on=["filter_length"],
        )
        pipeline.add_filter(
            ThresholdFilter("tm", min_value=60.0),
            stage_name="filter_tm",
            depends_on=["predict_tm"],
        )

        await pipeline.run_async(enable_streaming=False)

    df_final = pipeline.get_final_data()
    _check_df(
        df_final,
        "Pipeline 10 final",
        min_rows=1,
        required_cols=["sequence", "tm"],
    )

    assert pipeline.input_columns is None, "input_columns should be None"
    assert pipeline.input_schema is None, "input_schema should be None"

    ds.close()
    print("  Pipeline 10 passed!")


# ===========================================================================
# Main
# ===========================================================================
async def main():
    results = {}
    pipelines = [
        ("Pipeline 1: Antibody multi-column", pipeline_1_antibody_multi_column),
        ("Pipeline 2: Dedup edge cases", pipeline_2_dedup_edge_cases),
        ("Pipeline 3: ValidAA custom column", pipeline_3_valid_aa_custom_column),
        ("Pipeline 4: Chained filters", pipeline_4_chained_filters),
        ("Pipeline 5: Generative pipeline", pipeline_5_generative_pipeline),
        ("Pipeline 6: Context usage", pipeline_6_context_usage),
        ("Pipeline 7: Non-SQL filters", pipeline_7_non_sql_filters),
        ("Pipeline 8: Resume + cache", pipeline_8_resume_and_cache),
        ("Pipeline 9: Export extra cols", pipeline_9_export_dataframe_extra_cols),
        ("Pipeline 10: Backward compat", pipeline_10_backward_compat),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        for name, func in pipelines:
            try:
                await func(tmp_path)
                results[name] = "PASS"
            except Exception as e:
                results[name] = f"FAIL: {e}"
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, status in results.items():
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {name}: {status}")
        if status != "PASS":
            all_pass = False

    if all_pass:
        print(f"\nAll {len(results)} pipelines passed!")
    else:
        failed = sum(1 for s in results.values() if s != "PASS")
        print(f"\n{failed}/{len(results)} pipelines FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
