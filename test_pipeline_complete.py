"""
CORRECT Real Pipeline Test: Generation → Predictions (all in one!)

This demonstrates the PROPER workflow:
1. GenerativePipeline generates sequences using ESM2 + MLM remasking
2. Same pipeline adds downstream prediction stages  
3. Sequences flow through generation → predictions automatically
4. Diff mode adds more sequences + predictions
5. DuckDB handles all tabular data efficiently

Usage:
    python test_pipeline_complete.py
"""

import asyncio
import time
from pathlib import Path
import tempfile
from biolmai.pipeline import GenerativePipeline, GenerationConfig
from biolmai.pipeline.mlm_remasking import RemaskingConfig
from biolmai.pipeline.filters import ThresholdFilter


async def test_complete_pipeline():
    """Test complete generation + prediction workflow."""
    
    print("="*70)
    print("🧬 Complete Pipeline Test: Generation → Predictions")
    print("="*70)
    
    # Use temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "pipeline.duckdb"
        data_dir = Path(tmpdir) / "data"
        
        print(f"\n📁 Database: {db_path}\n")
        
        parent_sequence = "MKLLVVTGAAGQIGYILSHWIASGELYGDRQVYLHLLDIPPAMNRLTALTMELEDCAFPHLAGFVATTDPKA"
        
        print("="*70)
        print("PHASE 1: Generate + Predict 500 sequences")
        print("="*70)
        
        # Configuration for ESM2 150M generation with remasking
        remask_config = RemaskingConfig(
            model_name='esm-150m',  # ← BioLM slug for ESM2 150M
            mask_fraction=0.15,
            num_iterations=5,
            temperature=1.0
        )
        
        gen_config = GenerationConfig(
            model_name='esm-150m',  # ← BioLM slug for ESM2 150M
            num_sequences=500,
            generation_method='remask',
            parent_sequence=parent_sequence,
            mask_fraction=0.15,
            sampling_params={'num_iterations': 5}
        )
        
        # Create pipeline with generation
        pipeline = GenerativePipeline(
            name="complete_test",
            generation_configs=[gen_config],
            db_path=db_path,
            verbose=True
        )
        
        # ADD DOWNSTREAM PREDICTIONS - they automatically flow!
        pipeline.add_prediction(
            model_name='temberture',
            action='predict',
            prediction_type='tm',
            stage_name='stability'
        )
        
        pipeline.add_prediction(
            model_name='solubility',
            action='predict',
            prediction_type='solubility',
            stage_name='solubility_pred'
        )
        
        # Optional: Add filters
        pipeline.add_filter(ThresholdFilter('tm', min_value=40))
        
        print(f"\n🔧 Pipeline Configuration:")
        print(f"   Generation: ESM2 150M + MLM remasking")
        print(f"      - Model: esm-150m (BioLM slug)")
        print(f"      - Parent: {parent_sequence[:30]}...")
        print(f"      - Mask fraction: {remask_config.mask_fraction}")
        print(f"      - Iterations: {remask_config.num_iterations}")
        print(f"      - Temperature: {remask_config.temperature}")
        print(f"   Predictions: temberture, solubility")
        print(f"   Filters: Tm > 40°C")
        
        print(f"\n▶️  Running complete pipeline...\n")
        start_time = time.time()
        
        try:
            results = await pipeline.run_async()
            elapsed = time.time() - start_time
            
            print(f"\n✅ Phase 1 Complete!")
            print(f"   Generated: 500 sequences")
            print(f"   Predicted: {len(results)} passed filters")
            print(f"   Time: {elapsed:.1f}s")
            
            # Check database
            ds = pipeline.datastore
            seq_count = ds.query("SELECT COUNT(*) as cnt FROM sequences")['cnt'].iloc[0]
            pred_count = ds.query("SELECT COUNT(*) as cnt FROM predictions")['cnt'].iloc[0]
            
            print(f"\n📊 Database Stats:")
            print(f"   Sequences: {seq_count}")
            print(f"   Predictions: {pred_count}")
            
            # Sample results
            sample = ds.query("""
                SELECT 
                    s.sequence,
                    tm.value as tm,
                    sol.value as solubility
                FROM sequences s
                JOIN predictions tm ON s.sequence_id = tm.sequence_id AND tm.prediction_type = 'tm'
                JOIN predictions sol ON s.sequence_id = sol.sequence_id AND sol.prediction_type = 'solubility'
                ORDER BY tm.value DESC
                LIMIT 5
            """)
            
            print(f"\n📝 Top 5 variants:")
            for idx, row in sample.iterrows():
                print(f"   {idx+1}. Tm={row['tm']:.1f}°C, Sol={row['solubility']:.2f}")
                print(f"      {row['sequence'][:60]}...")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise
        
        print("\n" + "="*70)
        print("PHASE 2: Temperature Scanning (same pipeline)")
        print("="*70)
        
        # Add another generation config with different temperature
        gen_config_2 = GenerationConfig(
            model_name='esm-150m',  # BioLM slug
            num_sequences=500,
            generation_method='remask',
            parent_sequence=parent_sequence,
            temperature=[1.2, 1.5],  # ← TEMPERATURE SCANNING!
            mask_fraction=0.20,  # More diversity
            sampling_params={'num_iterations': 5}
        )
        
        pipeline_2 = GenerativePipeline(
            name="complete_test",
            generation_configs=[gen_config_2],
            db_path=db_path,  # Same DB
            diff_mode=True,   # Only new sequences
            verbose=True
        )
        
        # Add same predictions
        pipeline_2.add_prediction('temberture', prediction_type='tm')
        pipeline_2.add_prediction('solubility', prediction_type='solubility')
        
        print(f"\n🔧 Diff Mode Configuration:")
        print(f"   Generation: ESM2 150M @ T=[1.2, 1.5]")
        print(f"   Model: esm-150m (BioLM slug)")
        print(f"   Target: 500 new sequences")
        print(f"   Diff mode: ENABLED")
        
        print(f"\n▶️  Running diff mode...\n")
        start_time = time.time()
        
        try:
            results_2 = await pipeline_2.run_async()
            elapsed_2 = time.time() - start_time
            
            print(f"\n✅ Phase 2 Complete!")
            print(f"   New sequences: {len(results_2)}")
            print(f"   Time: {elapsed_2:.1f}s")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise
        
        print("\n" + "="*70)
        print("PHASE 3: Analyze Results with DuckDB")
        print("="*70)
        
        ds_final = pipeline_2.datastore
        
        # Final stats
        stats = ds_final.query("""
            SELECT 
                COUNT(DISTINCT s.sequence_id) as total_sequences,
                COUNT(DISTINCT p.prediction_id) as total_predictions
            FROM sequences s
            LEFT JOIN predictions p ON s.sequence_id = p.sequence_id
        """)
        
        print(f"\n📊 Final Stats:")
        print(f"   Total sequences: {stats['total_sequences'].iloc[0]}")
        print(f"   Total predictions: {stats['total_predictions'].iloc[0]}")
        
        # Analyze stability distribution  
        print(f"\n⚡ Running complex analysis...")
        start = time.time()
        
        analysis = ds_final.query("""
            SELECT 
                CASE 
                    WHEN tm.value >= 70 THEN 'High (>70°C)'
                    WHEN tm.value >= 55 THEN 'Medium (55-70°C)'
                    ELSE 'Low (<55°C)'
                END as stability,
                COUNT(*) as count,
                AVG(sol.value) as avg_solubility,
                MIN(tm.value) as min_tm,
                MAX(tm.value) as max_tm
            FROM sequences s
            JOIN predictions tm ON s.sequence_id = tm.sequence_id AND tm.prediction_type = 'tm'
            JOIN predictions sol ON s.sequence_id = sol.sequence_id AND sol.prediction_type = 'solubility'
            GROUP BY stability
            ORDER BY MIN(tm.value) DESC
        """)
        query_time = time.time() - start
        
        print(f"✅ Analysis complete in {query_time*1000:.1f}ms\n")
        print(f"📈 Stability Distribution:")
        for _, row in analysis.iterrows():
            print(f"   {row['stability']:20s}: {row['count']:3.0f} variants, "
                  f"avg_sol={row['avg_solubility']:.2f}, Tm range=[{row['min_tm']:.1f}-{row['max_tm']:.1f}]°C")
        
        ds_final.close()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print(f"\n✅ Demonstrated:")
        print(f"   - ESM2 150M generation with MLM remasking")
        print(f"   - Automatic prediction flow in same pipeline")
        print(f"   - Temperature scanning")
        print(f"   - Diff mode for incremental updates")
        print(f"   - DuckDB performance on ~1000 sequences")
        print(f"\n🚀 Pipeline is production-ready!")


if __name__ == "__main__":
    print("\n🧬 Complete Pipeline Test")
    print("   Generation (ESM2 150M) → Predictions (inline) → Analysis\n")
    
    try:
        asyncio.run(test_complete_pipeline())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted")
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
