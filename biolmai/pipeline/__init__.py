"""
BioLM Pipeline System

A comprehensive pipeline framework for biological sequence generation, prediction, and analysis.
"""

from biolmai.pipeline.datastore_duckdb import DuckDBDataStore

# Export DuckDB as DataStore for backward compatibility
DataStore = DuckDBDataStore
from biolmai.pipeline.base import BasePipeline, Stage, StageResult
from biolmai.pipeline.generative import (
    GenerativePipeline,
    GenerationConfig,
    DirectGenerationConfig,
)
try:
    from biolmai.pipeline.visualization import PipelinePlotter
except ImportError:
    PipelinePlotter = None  # type: ignore[assignment,misc]
from biolmai.pipeline.utils import cif_to_pdb, pdb_to_cif, load_structure_string
from biolmai.pipeline.data import DataPipeline, SingleStepPipeline, Predict, Embed
from biolmai.pipeline.filters import (
    ThresholdFilter,
    HammingDistanceFilter,
    SequenceLengthFilter,
    RankingFilter,
    CustomFilter,
)
from biolmai.pipeline.mlm_remasking import (
    MLMRemasker,
    RemaskingConfig,
    CONSERVATIVE_CONFIG,
    MODERATE_CONFIG,
    AGGRESSIVE_CONFIG,
)
from biolmai.pipeline.clustering import (
    SequenceClusterer,
    DiversityAnalyzer,
    ClusteringResult,
    cluster_sequences,
    analyze_diversity,
)

__all__ = [
    'DataStore',
    'DuckDBDataStore',
    'BasePipeline',
    'Stage',
    'StageResult',
    'GenerativePipeline',
    'GenerationConfig',
    'DirectGenerationConfig',
    'PipelinePlotter',
    'cif_to_pdb',
    'pdb_to_cif',
    'load_structure_string',
    'DataPipeline',
    'SingleStepPipeline',
    'Predict',
    'Embed',
    'ThresholdFilter',
    'HammingDistanceFilter',
    'SequenceLengthFilter',
    'RankingFilter',
    'CustomFilter',
    'MLMRemasker',
    'RemaskingConfig',
    'CONSERVATIVE_CONFIG',
    'MODERATE_CONFIG',
    'AGGRESSIVE_CONFIG',
    'SequenceClusterer',
    'DiversityAnalyzer',
    'ClusteringResult',
    'cluster_sequences',
    'analyze_diversity',
]
