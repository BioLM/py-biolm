# BioLM Pipeline System - Quick Start Guide

**Last Updated**: 2026-02-12  
**Status**: Ready for Testing  

---

## Installation

### 1. Install BioLM (if not already installed)

```bash
pip install biolmai
```

### 2. Install Pipeline Dependencies

```bash
cd biolmai/pipeline
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pandas numpy tqdm matplotlib seaborn scikit-learn umap-learn
```

---

## Your First Pipeline (5 minutes)

### Example 1: Simple Prediction

```python
from biolmai.pipeline import Predict

# Quick prediction - one line!
df = Predict('temberture', sequences=[
    'MKTAYIAKQRQGHQAMAEIKQ',
    'MKLAVIDSAQRQGHQAMAEIKQ'
])

print(df)
```

### Example 2: Multi-Stage Pipeline

```python
from biolmai.pipeline import DataPipeline
from biolmai.pipeline.filters import ThresholdFilter, SequenceLengthFilter

# Load sequences
sequences = [
    'MKTAYIAKQRQGHQAMAEIKQ',
    'MKLAVIDSAQRQGHQAMAEIKQ',
    'MKTAYIDSAQRQGHQAMAEIKQ',
]

# Create pipeline
pipeline = DataPipeline(sequences=sequences)

# Add stages
pipeline.add_filter(SequenceLengthFilter(min_length=20))
pipeline.add_prediction('temberture', prediction_type='tm')
pipeline.add_filter(ThresholdFilter('tm', min_value=50))

# Run
results = pipeline.run()

# Get results
df = pipeline.get_final_data()
print(df[['sequence', 'tm']])

# Summary
print(pipeline.summary())
```

### Example 3: Generative Pipeline

```python
from biolmai.pipeline import GenerativePipeline, GenerationConfig
from biolmai.pipeline.filters import ThresholdFilter

# Configure generation
config = GenerationConfig(
    model_name='proteinmpnn',
    num_sequences=100,
    temperature=[0.5, 1.0, 1.5],  # Temperature scanning!
    parent_sequence='MKTAYIAKQRQGHQAMAEIKQ'
)

# Create pipeline
pipeline = GenerativePipeline(generation_configs=[config])

# Add downstream predictions and filters
pipeline.add_prediction('temberture', prediction_type='tm')
pipeline.add_filter(ThresholdFilter('tm', min_value=60))

# Run
results = pipeline.run()

# Get top sequences
df = pipeline.get_final_data()
print(df.sort_values('tm', ascending=False).head(10))
```

---

## Key Concepts

### 1. Pipelines

Three types:
- **DataPipeline**: Process existing sequences
- **GenerativePipeline**: Generate sequences, then process
- **SingleStepPipeline**: Quick single predictions

### 2. Stages

Stages are processing steps:
- **Prediction stages**: Run models (esmfold, temberture, etc.)
- **Filter stages**: Filter sequences by criteria
- **Custom stages**: Your own processing

Stages are executed in order, respecting dependencies.

### 3. DataStore

Automatic caching system using SQLite:
- Stores sequences (deduplicated)
- Caches predictions
- Saves structures and embeddings
- Enables resumability

### 4. Caching

Automatic and transparent:
- Check cache before API calls
- Store results immediately
- Resume interrupted pipelines
- No manual cache management needed

---

## Performance Optimization

### Streaming Execution (New!)

Enable streaming for better performance on prediction-heavy pipelines:

```python
# Enable streaming - results flow immediately to next stage
results = pipeline.run(enable_streaming=True)
```

**When it helps:**
- Multiple prediction stages in sequence
- Filters that work on individual sequences (thresholds, length, etc.)
- Large datasets (> 1000 sequences)

**Performance gain:** ~20-30% faster for prediction → filter → prediction patterns.

### Async Execution

For maximum control, use async mode:

```python
import asyncio

async def main():
    results = await pipeline.run_async(enable_streaming=True)
    df = pipeline.get_final_data()
    return df

df = asyncio.run(main())
```

### Concurrency Control

Control concurrent API requests per stage:

```python
from biolmai.pipeline import PredictionStage

stage = PredictionStage(
    name='predict',
    model_name='esm2_t30_150M',
    action='predict',
    prediction_type='score',
    max_concurrent=20  # Increase for more parallelism
)
```

---

## Common Patterns

### Pattern 1: Filter → Predict → Filter

```python
pipeline = DataPipeline(sequences='sequences.csv')

# Pre-filter
pipeline.add_filter(SequenceLengthFilter(min_length=50, max_length=500))

# Expensive prediction
pipeline.add_prediction('esmfold', prediction_type='structure')

# Post-filter
pipeline.add_filter(ThresholdFilter('plddt', min_value=70))

results = pipeline.run()
```

### Pattern 2: Multi-Model Generation

```python
# Generate with multiple models
config1 = GenerationConfig(model_name='proteinmpnn', num_sequences=500)
config2 = GenerationConfig(model_name='ligandmpnn', num_sequences=500)

pipeline = GenerativePipeline(generation_configs=[config1, config2])
pipeline.add_prediction('temberture', prediction_type='tm')

results = pipeline.run()
```

### Pattern 3: Temperature Scanning

```python
config = GenerationConfig(
    model_name='proteinmpnn',
    temperature=[0.1, 0.5, 1.0, 1.5, 2.0],  # 5 temperatures
    num_sequences=200  # 200 per temperature = 1000 total
)

pipeline = GenerativePipeline(generation_configs=[config])
results = pipeline.run()

# Analyze by temperature
df = pipeline.get_final_data()
print(df.groupby('temperature')['tm'].mean())
```

### Pattern 4: Dependency Chain

```python
pipeline = DataPipeline(sequences='sequences.csv')

# Structure prediction (expensive)
pipeline.add_prediction('esmfold', stage_name='structure')

# Filter by structure quality
pipeline.add_filter(
    ThresholdFilter('plddt', min_value=70),
    depends_on=['structure']
)

# Structure-dependent prediction
pipeline.add_prediction(
    'pro4s',
    stage_name='solubility',
    depends_on=['structure']  # Only runs after structure stage
)

results = pipeline.run()
```

### Pattern 5: Diversity Sampling

```python
from biolmai.pipeline.filters import DiversitySamplingFilter

pipeline = GenerativePipeline(generation_configs=[config])
pipeline.add_prediction('temberture', prediction_type='tm')

# Sample top 100 by Tm
pipeline.add_filter(
    DiversitySamplingFilter(
        n_samples=100,
        method='top',
        score_column='tm'
    )
)

results = pipeline.run()
```

---

## File I/O

### Loading Sequences

```python
# From list
pipeline = DataPipeline(sequences=['MKTAYIAKQRQ', 'MKLAVID'])

# From DataFrame
import pandas as pd
df = pd.DataFrame({'sequence': ['MKTAYIAKQRQ', 'MKLAVID']})
pipeline = DataPipeline(sequences=df)

# From CSV (must have 'sequence' column)
pipeline = DataPipeline(sequences='sequences.csv')

# From FASTA
pipeline = DataPipeline(sequences='sequences.fasta')
```

### Exporting Results

```python
# To CSV
pipeline.export_to_csv('results.csv')

# Or get DataFrame
df = pipeline.get_final_data()
df.to_csv('results.csv', index=False)

# To FASTA
from biolmai.pipeline.utils import write_fasta
write_fasta(df, 'results.fasta', header_column='sequence_id')
```

---

## Visualization

```python
from biolmai.pipeline.visualization import PipelinePlotter

# Create plotter
plotter = PipelinePlotter(pipeline)

# Pipeline funnel (stages)
plotter.plot_funnel()

# Distribution plots
plotter.plot_distribution('tm', title='Melting Temperature')
plotter.plot_scatter('tm', 'plddt', color_col='temperature')

# Temperature scanning results
plotter.plot_temperature_scan('tm')

# Sequence diversity
plotter.plot_diversity(reference_sequence='MKTAYIAKQRQ')
```

---

## DataStore Direct Usage

```python
from biolmai.pipeline import DataStore

# Create/open datastore
store = DataStore('my_pipeline.db', 'my_pipeline_data')

# Add sequences
seq_id = store.add_sequence('MKTAYIAKQRQ')

# Add predictions
store.add_prediction(seq_id, 'stability', 'ddg_predictor', 2.5)
store.add_prediction(seq_id, 'tm', 'temberture', 65.3)

# Query predictions
preds = store.get_predictions_by_sequence('MKTAYIAKQRQ')
print(preds)

# Export all data
df = store.export_to_dataframe()
df.to_csv('all_results.csv', index=False)

# Stats
print(store.get_stats())
```

---

## Resumability

```python
# Start a pipeline with a specific run_id
pipeline = DataPipeline(
    sequences='sequences.csv',
    run_id='my_analysis_v1',
    output_dir='my_output'
)
pipeline.add_prediction('esmfold')
pipeline.add_prediction('temberture')

# Run (might take a while)
results = pipeline.run()

# Later, resume from the same run_id
# Completed stages will be skipped!
pipeline2 = DataPipeline(
    sequences='sequences.csv',
    run_id='my_analysis_v1',
    output_dir='my_output',
    resume=True  # Enable resuming
)
pipeline2.add_prediction('esmfold')  # Will be skipped (already complete)
pipeline2.add_prediction('temberture')  # Will be skipped (already complete)
pipeline2.add_prediction('pro4s')  # Will run (new stage)

results = pipeline2.run()
```

---

## Advanced Filtering

### Built-in Filters

```python
from biolmai.pipeline.filters import (
    ThresholdFilter,
    SequenceLengthFilter,
    HammingDistanceFilter,
    ConservedResidueFilter,
    DiversitySamplingFilter,
    CustomFilter,
    combine_filters
)

# Numeric thresholds
filter1 = ThresholdFilter('tm', min_value=60, max_value=100)

# Sequence length
filter2 = SequenceLengthFilter(min_length=50, max_length=500)

# Sequence similarity
filter3 = HammingDistanceFilter(
    reference_sequence='MKTAYIAKQRQ',
    max_distance=50
)

# Conserved residues (e.g., active site)
filter4 = ConservedResidueFilter({
    10: ['K', 'R'],  # Position 10 must be K or R
    25: ['C'],       # Position 25 must be C
    50: ['D', 'E']   # Position 50 must be D or E
})

# Diversity sampling
filter5 = DiversitySamplingFilter(
    n_samples=1000,
    method='top',
    score_column='tm'
)

# Custom filter
def my_filter(df):
    return df[df['sequence'].str.startswith('M')]

filter6 = CustomFilter(my_filter, name='starts_with_M')

# Combine filters
combined = combine_filters(filter1, filter2, filter3)
```

---

## Performance Tips

### 1. Batch Size

```python
pipeline.add_prediction(
    'esmfold',
    batch_size=16,  # Smaller for large structures
    max_concurrent=3  # Limit concurrent batches
)

pipeline.add_prediction(
    'temberture',
    batch_size=128,  # Larger for fast predictions
    max_concurrent=10
)
```

### 2. Early Filtering

```python
# Good: Filter early to reduce downstream work
pipeline.add_filter(SequenceLengthFilter(min_length=50))
pipeline.add_prediction('esmfold')  # Fewer sequences to predict

# Bad: Filter after expensive prediction
pipeline.add_prediction('esmfold')
pipeline.add_filter(SequenceLengthFilter(min_length=50))
```

### 3. Caching

```python
# Use persistent datastore for large projects
pipeline = DataPipeline(
    sequences='sequences.csv',
    datastore='project.db',  # Reuse across runs
)
```

### 4. Deduplication

```python
# Enable for generative pipelines
pipeline = GenerativePipeline(
    generation_configs=[config],
    deduplicate=True  # Remove duplicate sequences
)
```

---

## Troubleshooting

### Issue: "Module not found"

```python
# Make sure you're in the right directory
import sys
sys.path.insert(0, '/path/to/py-biolm')

from biolmai.pipeline import DataPipeline
```

### Issue: "Database is locked"

```python
# Close previous connections
store.close()

# Or use context manager
with DataStore('pipeline.db') as store:
    # ... use store
    pass
```

### Issue: "API rate limit exceeded"

```python
# Reduce concurrency
pipeline.add_prediction(
    'esmfold',
    max_concurrent=2,  # Lower from default (5)
    batch_size=8       # Smaller batches
)
```

### Issue: "Out of memory"

```python
# Process in smaller batches
sequences_all = load_sequences('all_sequences.fasta')

for i in range(0, len(sequences_all), 1000):
    batch = sequences_all[i:i+1000]
    pipeline = DataPipeline(
        sequences=batch,
        datastore='shared.db'  # Same datastore for all batches
    )
    pipeline.add_prediction('esmfold')
    pipeline.run()
```

---

## Next Steps

1. **Run the examples**:
   ```bash
   python examples/simple_pipeline_example.py
   ```

2. **Read the full documentation**:
   - `biolmai/pipeline/README.md` - Complete guide
   - `PIPELINE_DEVELOPMENT_PLAN.md` - Architecture and design
   - `PIPELINE_IMPLEMENTATION_SUMMARY.md` - Implementation details

3. **Try with your data**:
   - Start with a small dataset (10-100 sequences)
   - Test caching and resumability
   - Add visualizations

4. **Explore advanced features**:
   - Multi-model generation
   - Temperature scanning
   - Stage dependencies
   - Custom filters

---

## Getting Help

### Documentation Files

- `PIPELINE_QUICKSTART.md` - This file
- `biolmai/pipeline/README.md` - Complete user guide
- `PIPELINE_DEVELOPMENT_PLAN.md` - Technical design document
- `PIPELINE_IMPLEMENTATION_SUMMARY.md` - Implementation status

### Example Code

- `examples/simple_pipeline_example.py` - Working examples
- `pipeline_ex/` - Original reference implementation

### Code Structure

```
biolmai/pipeline/
├── __init__.py              - Main exports
├── datastore.py             - Storage system
├── base.py                  - Base classes
├── generative.py            - Generative pipelines
├── data.py                  - Data pipelines
├── filters.py               - Filter implementations
├── async_executor.py        - Async execution
├── visualization.py         - Plotting functions
└── utils.py                 - Utility functions
```

---

## FAQ

**Q: Do I need to manage caching manually?**  
A: No, caching is automatic. Results are cached when computed and checked before each API call.

**Q: Can I resume a failed pipeline?**  
A: Yes, use `resume=True` when creating the pipeline with the same `run_id`.

**Q: How do I run multiple models in parallel?**  
A: For generation, pass multiple `GenerationConfig` objects. For predictions, add multiple prediction stages without dependencies.

**Q: How do I filter sequences before prediction?**  
A: Add filter stages before prediction stages: `pipeline.add_filter(...)` then `pipeline.add_prediction(...)`.

**Q: Can I use custom models?**  
A: Yes, create a custom `Stage` subclass. See `PIPELINE_IMPLEMENTATION_SUMMARY.md` for examples.

**Q: How do I visualize results?**  
A: Use `PipelinePlotter`: `plotter = PipelinePlotter(pipeline); plotter.plot_funnel()`.

**Q: What if my sequences are in a different format?**  
A: Use utility functions in `biolmai.pipeline.utils` to load FASTA, CSV, or convert between formats.

---

## Success!

You're now ready to use the BioLM Pipeline System. Start with simple examples and gradually add more features as you become comfortable with the system.

Happy pipelining! 🚀

