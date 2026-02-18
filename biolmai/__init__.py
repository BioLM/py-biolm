"""Top-level package for BioLM AI."""
__author__ = """Nikhil Haas"""
__email__ = "nikhil@biolm.ai"
__version__ = '0.2.18'

from biolmai.core.http import BioLMApi, BioLMApiClient
from biolmai.biolmai import BioLM
from biolmai.models import Model, predict, encode, generate
from biolmai.protocols import Protocol
from biolmai.workspaces import Workspace
from biolmai.volumes import Volume
from biolmai.examples import get_example, list_models
from biolmai.io import (
    load_fasta,
    to_fasta,
    load_csv,
    to_csv,
    load_pdb,
    to_pdb,
    load_json,
    to_json,
)
from typing import Optional, Union, List, Any

__all__ = [
    # Main interfaces (main backward compat)
    'BioLM',
    'biolm',
    'BioLMApi',
    'BioLMApiClient',
    # New interfaces from issue49
    'Model',
    'Protocol',
    'Workspace',
    'Volume',
    # Convenience functions
    'predict',
    'encode',
    'generate',
    # Example generation
    'get_example',
    'list_models',
    # IO utilities
    'load_fasta',
    'to_fasta',
    'load_csv',
    'to_csv',
    'load_pdb',
    'to_pdb',
    'load_json',
    'to_json',
]


def biolm(
    *,
    entity: str,
    action: str,
    type: Optional[str] = None,
    items: Union[Any, List[Any]],
    params: Optional[dict] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Any:
    """Top-level convenience function that wraps the BioLM class and returns the result.

    Additional kwargs (e.g., compress_requests, compress_threshold) are passed through to BioLMApiClient.
    """
    return BioLM(entity=entity, action=action, type=type, items=items, params=params, api_key=api_key, **kwargs)
