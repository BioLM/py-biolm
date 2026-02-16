# BioLM Pipeline System

A comprehensive pipeline framework for biological sequence generation, prediction, and analysis.

## Features

- **Three Pipeline Types**:
  - `GenerativePipeline`: Generate sequences using language models
  - `DataPipeline`: Process existing sequences with predictions
  - `SingleStepPipeline`: Quick single-step predictions

- **Advanced Capabilities**:
  - Automatic caching and deduplication
  - Async execution with rate limiting
  - Temperature and parameter scanning
  - Multi-model generation in parallel
  - Stage dependencies and filtering
  - Resumable pipelines
  - Comprehensive visualizations

- **Storage**:
  - SQLite-based datastore
  - Efficient caching of sequences, predictions, structures, and embeddings
  - Lazy loading for large objects

## Quick Start

### Installation

```bash
pip install biolmai
```

### Single-Step Prediction

```python
from biolmai.pipeline import Predict

# Quick prediction
df = Predict('esmfold', sequences=['MKTAYIAKQRQ', 'MKLAVID'])
print(df)
```

### Data Pipeline

```python
from biolmai.pipeline import DataPipeline
from biolmai.pipeline.filters import ThresholdFilter

# Load sequences and run predictions
pipeline = DataPipeline(sequences='sequences.csv')

# Add prediction stages
pipeline.add_prediction('esmfold', prediction_type='structure')
pipeline.add_filter(ThresholdFilter('plddt', min_value=70))
pipeline.add_prediction('temberture', prediction_type='tm')

# Run pipeline
results = pipeline.run()

# Get results
df = pipeline.get_final_data()
pipeline.export_to_csv('results.csv')

# Visualize
from biolmai.pipeline.visualization import PipelinePlotter
plotter = PipelinePlotter(pipeline)
plotter.plot_funnel()
plotter.plot_distribution('tm')
```

### Generative Pipeline

```python
from biolmai.pipeline import GenerativePipeline, GenerationConfig

# Configure generation
config1 = GenerationConfig(
    model_name='proteinmpnn',
    num_sequences=1000,
    temperature=[0.5, 1.0, 1.5],  # Temperature scanning
    parent_sequence='MKTAYIAKQRQ'
)

config2 = GenerationConfig(
    model_name='esm2',
    num_sequences=500,
    generation_method='remask',  # Masked language model
    parent_sequence='MKTAYIAKQRQ',
    mask_fraction=0.15
)

# Create pipeline
pipeline = GenerativePipeline(
    generation_configs=[config1, config2],
    deduplicate=True
)

# Add downstream predictions
pipeline.add_prediction('esmfold', prediction_type='structure')
pipeline.add_filter(ThresholdFilter('plddt', min_value=70))
pipeline.add_prediction('temberture', prediction_type='tm')

# Run
results = pipeline.run()
df = pipeline.get_final_data()
```

## Architecture

### Pipeline Flow

```
[Initial Data] → [Stage 1] → [Stage 2] → ... → [Stage N] → [Final Results]
     ↓              ↓            ↓                   ↓
[DataStore: SQLite + Binary Files]
```

### Stage Types

1. **Generation Stages**: Generate sequences using language models
2. **Prediction Stages**: Run predictions (structure, stability, etc.)
3. **Filter Stages**: Filter sequences by criteria
4. **Custom Stages**: User-defined processing

### DataStore Schema

```
sequences          → Deduplicated sequences with IDs
predictions        → Prediction results (keyed by sequence_id)
structures         → Structure files (PDB/CIF)
embeddings         → Embedding arrays
generation_metadata → Source model and params for generated sequences
pipeline_runs      → Pipeline execution records
stage_completions  → Completion markers for resumability
```

## Advanced Features

### Temperature Scanning

```python
config = GenerationConfig(
    model_name='proteinmpnn',
    temperature=[0.1, 0.5, 1.0, 1.5, 2.0],  # Multiple temperatures
    num_sequences=200  # Per temperature
)
```

### Stage Dependencies

```python
# Structure prediction must complete before downstream analysis
pipeline.add_prediction('esmfold', stage_name='structure')
pipeline.add_prediction('pro4s', stage_name='solubility', depends_on=['structure'])
```

### Filtering

```python
from biolmai.pipeline.filters import (
    ThresholdFilter,
    SequenceLengthFilter,
    HammingDistanceFilter,
    ConservedResidueFilter,
    DiversitySamplingFilter
)

# Threshold filtering
pipeline.add_filter(ThresholdFilter('tm', min_value=60))

# Length filtering
pipeline.add_filter(SequenceLengthFilter(min_length=50, max_length=500))

# Sequence similarity
pipeline.add_filter(HammingDistanceFilter('MKTAYIAKQRQ', max_distance=50))

# Conserved residues (e.g., active site)
pipeline.add_filter(ConservedResidueFilter({
    107: ['H'],  # Position 107 must be H
    109: ['H'],
    126: ['H']
}))

# Diversity sampling
pipeline.add_filter(DiversitySamplingFilter(
    n_samples=1000,
    method='top',
    score_column='tm'
))
```

### Masked Language Model Remasking

```python
config = GenerationConfig(
    model_name='esm2',
    num_sequences=1000,
    generation_method='remask',
    parent_sequence='MKTAYIAKQRQGHQAMAEIKQ',
    mask_positions='auto',  # Auto-select positions
    mask_fraction=0.15  # Mask 15% of residues
)
```

### Resumability

```python
# Start pipeline
pipeline = DataPipeline(sequences='sequences.csv', run_id='my_run')
pipeline.add_prediction('esmfold')
results = pipeline.run()

# Later, resume from same run_id
pipeline2 = DataPipeline(sequences='sequences.csv', run_id='my_run', resume=True)
# Cached stages will be skipped
```

### Visualization

```python
from biolmai.pipeline.visualization import (
    plot_pipeline_funnel,
    plot_distribution,
    plot_scatter,
    plot_correlation_matrix,
    plot_temperature_scan,
    plot_embedding_pca,
    plot_embedding_umap,
    plot_sequence_diversity
)

# Pipeline funnel
plot_pipeline_funnel(pipeline.stage_results)

# Distribution plots
plot_distribution(df, 'tm', title='Melting Temperature Distribution')
plot_scatter(df, 'tm', 'plddt', color_col='temperature')

# Temperature scanning
plot_temperature_scan(df, 'tm', temperature_col='temperature')

# Embeddings
embeddings = ...  # Load from datastore
plot_embedding_pca(embeddings, labels=df['temperature'])
plot_embedding_umap(embeddings, labels=df['temperature'])

# Sequence diversity
plot_sequence_diversity(df, reference_sequence='MKTAYIAKQRQ')
```

## API Reference

### DataStore

```python
from biolmai.pipeline import DataStore

store = DataStore('pipeline.db', 'pipeline_data')

# Sequences
seq_id = store.add_sequence('MKTAYIAKQRQ')
seq = store.get_sequence(seq_id)

# Predictions
store.add_prediction(seq_id, 'stability', 'ddg_predictor', 2.5)
preds = store.get_predictions_by_sequence('MKTAYIAKQRQ')

# Structures
store.add_structure(seq_id, 'esmfold', pdb_string, plddt=85.2)
struct = store.get_structure(structure_id)

# Embeddings
store.add_embedding(seq_id, 'esm2', embedding_array)
embs = store.get_embeddings_by_sequence('MKTAYIAKQRQ')

# Export
df = store.export_to_dataframe()
store.export_to_csv('results.csv')
```

### Pipeline Classes

```python
# Data Pipeline
pipeline = DataPipeline(sequences=['MKTAYIAKQRQ'])
pipeline.add_prediction('esmfold')
pipeline.add_filter(filter_func)
results = pipeline.run()

# Generative Pipeline
pipeline = GenerativePipeline(generation_configs=[config])
pipeline.add_prediction('esmfold')
results = pipeline.run()

# Single-Step Pipeline
pipeline = SingleStepPipeline('esmfold', sequences=['MKTAYIAKQRQ'])
results = pipeline.run()

# Convenience functions
df = Predict('esmfold', sequences=['MKTAYIAKQRQ'])
df = Embed('esm2', sequences=['MKTAYIAKQRQ'])
df = Generate('proteinmpnn', num_sequences=100)
```

### Filters

```python
from biolmai.pipeline.filters import *

# Threshold
filter = ThresholdFilter('tm', min_value=60, max_value=100)

# Length
filter = SequenceLengthFilter(min_length=50, max_length=500)

# Hamming distance
filter = HammingDistanceFilter('MKTAYIAKQRQ', max_distance=50)

# Conserved residues
filter = ConservedResidueFilter({107: ['H'], 109: ['H']})

# Diversity sampling
filter = DiversitySamplingFilter(n_samples=1000, method='random')

# Custom filter
filter = CustomFilter(lambda df: df[df['tm'] > 60])

# Combine filters
filter = combine_filters(
    ThresholdFilter('tm', min_value=60),
    SequenceLengthFilter(min_length=100)
)
```

## Examples

See the `examples/` directory for complete examples:

- `examples/simple_prediction.py`: Basic prediction pipeline
- `examples/generative_pipeline.py`: Multi-model generation with temperature scanning
- `examples/variant_selection.py`: Reproduce the carbonic anhydrase pipeline
- `examples/embeddings_analysis.py`: Embedding generation and visualization

## Performance Tips

1. **Batch Size**: Adjust `batch_size` for API calls based on sequence length and model
2. **Concurrency**: Set `max_concurrent` based on API rate limits
3. **Caching**: Use persistent datastore for resumability
4. **Filtering**: Apply filters early to reduce downstream computations
5. **Deduplication**: Enable `deduplicate=True` for generative pipelines

## Migration from pipeline_ex

Old pattern (pipeline_ex):
```python
from pipeline_cache import StageCache

stage = StageCache('output.csv', cache_name='stage', cache_columns=['result'])
df_todo, df_cache = stage.get_todo(df_input)
# ... compute results ...
stage.update_cache(df_new, df_cache)
stage.save(df_merged)
```

New pattern (pipeline):
```python
from biolmai.pipeline import DataPipeline

pipeline = DataPipeline(sequences=df_input)
pipeline.add_prediction('model_name', prediction_type='result')
results = pipeline.run()
df = pipeline.get_final_data()
```

## Contributing

See CONTRIBUTING.md for development guidelines.

## License

MIT License - see LICENSE file for details.
