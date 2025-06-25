"""Top-level package for BioLM AI."""
__author__ = """Nikhil Haas"""
__email__ = "nikhil@biolm.ai"
__version__ = '0.2.4'

from biolmai.client import BioLMApi, BioLMApiClient
from biolmai.biolmai import BioLM
from typing import Optional, Union, List, Any

__all__ = ['biolm']


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
