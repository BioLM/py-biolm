"""
BioLM Pipeline System

A comprehensive pipeline framework for biological sequence generation, prediction, and analysis.

Requires optional dependencies — install with::

    pip install biolmai[pipeline]
"""

# Detect missing optional dependencies BEFORE importing anything that pulls
# them transitively, so users see a single actionable error instead of the
# traceback from a half-imported module.
_MISSING = []
for _name in ("duckdb", "pandas", "numpy", "pyarrow"):
    try:
        __import__(_name)
    except ImportError:
        _MISSING.append(_name)

if _MISSING:
    raise ImportError(
        "biolmai.pipeline requires optional dependencies that are not installed: "
        f"{', '.join(_MISSING)}.\n\n"
        "Install with:\n\n"
        "    pip install 'biolmai[pipeline]'\n\n"
        "If you only use the BioLM API client, you can ignore this — the "
        "pipeline package is opt-in."
    )
del _MISSING
try:
    del _name
except NameError:
    pass

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
    MatrixExtractionSpec,
    PipelineAPIAuthError,
    Predict,
    SingleStepPipeline,
    StructureSpec,
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

# Backward-compat alias.  Prefer the concrete name `DuckDBDataStore` in new code;
# this alias may be removed in a future major version once a generic DataStore
# Protocol is introduced.
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
    "PipelineAPIAuthError",
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
    "StructureSpec",
    "MatrixExtractionSpec",
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
