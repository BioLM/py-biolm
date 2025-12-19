"""Top-level package for BioLM AI."""
__author__ = """Nikhil Haas"""
__email__ = "nikhil@biolm.ai"
__version__ = '0.2.8'

from biolmai.core.http import BioLMApi, BioLMApiClient
from biolmai.models import Model, BioLM, predict, encode, generate
from biolmai.protocols import Protocol
from biolmai.workspaces import Workspace
from biolmai.volumes import Volume
from biolmai.examples import get_example, list_models
from typing import Optional, Union, List, Any

__all__ = [
    # Main interfaces
    'Model',
    'Protocol',
    'Workspace',
    'Volume',
    # Convenience functions
    'biolm',
    'predict',
    'encode',
    'generate',
    # Example generation
    'get_example',
    'list_models',
    # Advanced/legacy
    'BioLM',
    'BioLMApi',
    'BioLMApiClient',
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
    """Top-level convenience function that wraps the BioLM class and returns the result."""
    return BioLM(entity=entity, action=action, type=type, items=items, params=params, api_key=api_key, **kwargs)
