"""
Simple Pipeline Example

This example demonstrates basic usage of the pipeline system.
"""

import sys
sys.path.insert(0, '/home/c/py-biolm')

from biolmai.pipeline import (
    DataPipeline,
    GenerativePipeline,
    GenerationConfig,
    SingleStepPipeline,
    Predict,
    Embed
)
from biolmai.pipeline.filters import (
    ThresholdFilter,
    SequenceLengthFilter,
    DiversitySamplingFilter
)
from biolmai.pipeline.visualization import PipelinePlotter


def example_1_quick_prediction():
    """Example 1: Quick single-step prediction."""
    print("="*60)
    print("Example 1: Quick Prediction")
    print("="*60)
    
    sequences = [
        'MKTAYIAKQRQGHQAMAEIKQ',
        'MKLAVIDSAQRQGHQAMAEIKQ',
        'MKTAYIDSAQRQGHQAMAEIKQ'
    ]
    
    # Quick prediction (returns DataFrame)
    df = Predict('temberture', sequences=sequences, verbose=False)
    print(df)


def example_2_data_pipeline():
    """Example 2: Data pipeline with multiple stages."""
    print("\n" + "="*60)
    print("Example 2: Data Pipeline")
    print("="*60)
    
    sequences = [
        'MKTAYIAKQRQGHQAMAEIKQ',
        'MKLAVIDSAQRQGHQAMAEIKQ',
        'MKTAYIDSAQRQGHQAMAEIKQ',
        'MKTAYIAKQRQGHQAMAEI',  # Shorter
        'MKTAYIAKQRQGHQAMAEIKQGHQAMAEIKQ'  # Longer
    ]
    
    # Create pipeline
    pipeline = DataPipeline(sequences=sequences)
    
    # Add stages
    pipeline.add_filter(
        SequenceLengthFilter(min_length=20, max_length=30),
        stage_name='length_filter'
    )
    
    pipeline.add_prediction(
        'temberture',
        prediction_type='tm',
        stage_name='tm_prediction'
    )
    
    pipeline.add_filter(
        ThresholdFilter('tm', min_value=50),
        stage_name='tm_filter'
    )
    
    # Run pipeline
    print("\nRunning pipeline...")
    results = pipeline.run()
    
    # Get results
    df = pipeline.get_final_data()
    print("\nFinal results:")
    print(df[['sequence', 'tm']])
    
    # Summary
    print("\nPipeline summary:")
    print(pipeline.summary())


def example_3_generative_pipeline():
    """Example 3: Generative pipeline with temperature scanning."""
    print("\n" + "="*60)
    print("Example 3: Generative Pipeline")
    print("="*60)
    
    parent_sequence = 'MKTAYIAKQRQGHQAMAEIKQ'
    
    # Configure generation
    config = GenerationConfig(
        model_name='proteinmpnn',
        num_sequences=50,
        temperature=[0.5, 1.0],  # Temperature scanning
        parent_sequence=parent_sequence
    )
    
    # Create pipeline
    pipeline = GenerativePipeline(
        generation_configs=[config],
        deduplicate=True
    )
    
    # Add downstream predictions
    pipeline.add_prediction(
        'temberture',
        prediction_type='tm'
    )
    
    # Add filtering
    pipeline.add_filter(
        ThresholdFilter('tm', min_value=55)
    )
    
    # Add diversity sampling
    pipeline.add_filter(
        DiversitySamplingFilter(
            n_samples=10,
            method='top',
            score_column='tm'
        )
    )
    
    # Run pipeline
    print("\nRunning generative pipeline...")
    results = pipeline.run()
    
    # Get results
    df = pipeline.get_final_data()
    print("\nTop 10 sequences by Tm:")
    print(df[['sequence', 'tm', 'temperature']].head(10))
    
    # Summary
    print("\nPipeline summary:")
    print(pipeline.summary())


def example_4_visualization():
    """Example 4: Pipeline visualization."""
    print("\n" + "="*60)
    print("Example 4: Visualization")
    print("="*60)
    
    sequences = [
        'MKTAYIAKQRQGHQAMAEIKQ',
        'MKLAVIDSAQRQGHQAMAEIKQ',
        'MKTAYIDSAQRQGHQAMAEIKQ',
        'MKTAYIAKQRQGHQAMAEIKQGHQ'
    ]
    
    # Create pipeline
    pipeline = DataPipeline(sequences=sequences)
    pipeline.add_prediction('temberture', prediction_type='tm')
    
    # Run
    results = pipeline.run()
    
    # Visualize
    print("\nGenerating visualizations...")
    
    plotter = PipelinePlotter(pipeline)
    
    # Funnel plot
    plotter.plot_funnel(save_path='pipeline_funnel.png')
    
    # Distribution
    plotter.plot_distribution('tm', save_path='tm_distribution.png')
    
    print("Visualizations saved to pipeline_funnel.png and tm_distribution.png")


def example_5_datastore_usage():
    """Example 5: Direct DataStore usage."""
    print("\n" + "="*60)
    print("Example 5: DataStore")
    print("="*60)
    
    from biolmai.pipeline import DataStore
    
    # Create datastore
    store = DataStore('example.db', 'example_data')
    
    # Add sequences
    seq1 = 'MKTAYIAKQRQ'
    seq2 = 'MKLAVIDSAQRQ'
    
    id1 = store.add_sequence(seq1)
    id2 = store.add_sequence(seq2)
    
    print(f"Added sequences: {id1}, {id2}")
    
    # Add predictions
    store.add_prediction(id1, 'stability', 'ddg_predictor', 2.5)
    store.add_prediction(id2, 'stability', 'ddg_predictor', 3.1)
    
    # Query
    preds = store.get_predictions(prediction_type='stability')
    print("\nPredictions:")
    print(preds)
    
    # Export
    df = store.export_to_dataframe()
    print("\nExported DataFrame:")
    print(df)
    
    # Stats
    print("\nDataStore stats:")
    print(store.get_stats())


if __name__ == '__main__':
    print("BioLM Pipeline Examples\n")
    
    # Run examples
    try:
        example_1_quick_prediction()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_2_data_pipeline()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        example_5_datastore_usage()
    except Exception as e:
        print(f"Example 5 failed: {e}")
    
    # Note: Examples 3 and 4 require actual API access
    print("\n" + "="*60)
    print("Note: Examples 3 and 4 require BioLM API access")
    print("="*60)
    
    print("\nAll examples completed!")
