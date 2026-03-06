"""
BioLM Pipeline System

A comprehensive pipeline framework for biological sequence generation, prediction, and analysis.
"""

from biolmai.pipeline.base import (
    BasePipeline,
    InputSchema,
    PipelineContext,
    PipelineMetadata,
    Stage,
    StageResult,
    WorkingSet,
)
from biolmai.pipeline.clustering import (
    ClusteringResult,
    DiversityAnalyzer,
    SequenceClusterer,
    analyze_diversity,
    cluster_sequences,
)
from biolmai.pipeline.data import (
    CofoldingPredictionStage,
    DataPipeline,
    Embed,
    EmbeddingSpec,
    ExtractionSpec,
    Predict,
    SingleStepPipeline,
)
from biolmai.pipeline.datastore_duckdb import DuckDBDataStore
from biolmai.pipeline.filters import (
    CompositeFilter,
    ConservedResidueFilter,
    CustomFilter,
    DiversitySamplingFilter,
    HammingDistanceFilter,
    RankingFilter,
    SequenceLengthFilter,
    ThresholdFilter,
    ValidAminoAcidFilter,
    combine_filters,
)
from biolmai.pipeline.generative import (
    DirectGenerationConfig,
    FoldingEntity,
    GenerationConfig,
    GenerativePipeline,
    SequenceSourceConfig,
)
from biolmai.pipeline.mlm_remasking import (
    AGGRESSIVE_CONFIG,
    CONSERVATIVE_CONFIG,
    MODERATE_CONFIG,
    MLMRemasker,
    RemaskingConfig,
)
from biolmai.pipeline.utils import cif_to_pdb, load_structure_string, pdb_to_cif

try:
    from biolmai.pipeline.visualization import PipelinePlotter
except ImportError:
    PipelinePlotter = None  # type: ignore[assignment,misc]

# Export DuckDB as DataStore for backward compatibility
DataStore = DuckDBDataStore

__all__ = [
    "DataStore",
    "DuckDBDataStore",
    "BasePipeline",
    "Stage",
    "StageResult",
    "WorkingSet",
    "InputSchema",
    "PipelineContext",
    "PipelineMetadata",
    "GenerativePipeline",
    "GenerationConfig",
    "DirectGenerationConfig",
    "SequenceSourceConfig",
    "FoldingEntity",
    "CofoldingPredictionStage",
    "PipelinePlotter",
    "cif_to_pdb",
    "pdb_to_cif",
    "load_structure_string",
    "DataPipeline",
    "SingleStepPipeline",
    "Predict",
    "Embed",
    "ThresholdFilter",
    "HammingDistanceFilter",
    "SequenceLengthFilter",
    "RankingFilter",
    "CustomFilter",
    "CompositeFilter",
    "combine_filters",
    "ValidAminoAcidFilter",
    "ConservedResidueFilter",
    "DiversitySamplingFilter",
    "ExtractionSpec",
    "EmbeddingSpec",
    "MLMRemasker",
    "RemaskingConfig",
    "CONSERVATIVE_CONFIG",
    "MODERATE_CONFIG",
    "AGGRESSIVE_CONFIG",
    "SequenceClusterer",
    "DiversityAnalyzer",
    "ClusteringResult",
    "cluster_sequences",
    "analyze_diversity",
]
