"""Top-level package for BioLM AI."""

__author__ = """Nikhil Haas"""
__email__ = "nikhil@biolm.ai"
__version__ = "0.3.0"

from biolmai.biolmai import BioLM
from biolmai.core.http import BioLMApi, BioLMApiClient
from biolmai.examples import get_example, list_models
from biolmai.io import (
    load_csv,
    load_fasta,
    load_json,
    load_pdb,
    to_csv,
    to_fasta,
    to_json,
    to_pdb,
)
from biolmai.models import Model, encode, generate, predict
from biolmai.protocols import Protocol
from biolmai.volumes import Volume
from biolmai.workspaces import Workspace

# Pipeline system (optional dependency)
try:
    from biolmai import pipeline
except ImportError:
    # Pipeline dependencies might not be installed
    pass

from typing import Any, List, Optional, Union

__all__ = [
    # Main interfaces (main backward compat)
    "BioLM",
    "biolm",
    "BioLMApi",
    "BioLMApiClient",
    # New interfaces from issue49
    "Model",
    "Protocol",
    "Workspace",
    "Volume",
    # Convenience functions
    "predict",
    "encode",
    "generate",
    # Example generation
    "get_example",
    "list_models",
    # IO utilities
    "load_fasta",
    "to_fasta",
    "load_csv",
    "to_csv",
    "load_pdb",
    "to_pdb",
    "load_json",
    "to_json",
    # Pipeline (optional)
    "pipeline",
]


def biolm(
    *,
    entity: str,
    action: str,
    type: Optional[str] = None,
    items: Union[Any, List[Any]],
    params: Optional[dict] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> Any:
    """Top-level convenience function that wraps the BioLM class and returns the result.

    Additional kwargs (e.g., compress_requests, compress_threshold) are passed through to BioLMApiClient.
    """
    return BioLM(
        entity=entity,
        action=action,
        type=type,
        items=items,
        params=params,
        api_key=api_key,
        **kwargs,
    )
