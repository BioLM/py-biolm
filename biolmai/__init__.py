"""Top-level package for BioLM AI."""
__author__ = """Nikhil Haas"""
__email__ = "nikhil@biolm.ai"
__version__ = '0.4.1'

from biolmai.core.http import BioLMApi, BioLMApiClient
from biolmai.biolmai import BioLM
from biolmai.models import Model, predict, encode, generate
from biolmai.protocols import Protocol
from biolmai.finetune import Finetune
from biolmai.protocol_runs import ProtocolClient, ProtocolRun, ProtocolRunError, ProtocolNotFoundError
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

# Pipeline system (optional dependency).  Don't crash `import biolmai` when
# the [pipeline] extra is not installed — the pipeline namespace will simply
# be unavailable until the user runs `pip install biolmai[pipeline]`.
try:
    from biolmai import pipeline
    _HAS_PIPELINE = True
except ImportError:
    _HAS_PIPELINE = False

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
    'Finetune',
    'Workspace',
    'Volume',
    # Protocol Submission API
    'ProtocolClient',
    'ProtocolRun',
    'ProtocolRunError',
    'ProtocolNotFoundError',
    'run_protocol',
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
if _HAS_PIPELINE:
    __all__.append('pipeline')


def run_protocol(
    slug: str,
    inputs: dict,
    *,
    run_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 3600.0,
    show_progress: bool = True,
) -> dict:
    """Submit a BioLM protocol run and block until results are ready.

    One-liner convenience wrapper around :class:`ProtocolClient`.

    Args:
        slug: Protocol slug (e.g. ``"antibody-optimization"``).
        inputs: Dict of input field values. Use
            ``ProtocolClient().get(slug)["inputs_schema"]`` to discover required fields.
        run_name: Optional human-readable label.
        api_key: BioLM API token. Reads ``BIOLMAI_TOKEN`` env var if not provided.
        base_url: Override API base domain (default ``https://biolm.ai``).
        timeout: Max seconds to wait before raising :class:`TimeoutError` (default 3600).
        show_progress: Print progress updates to stdout (default True).

    Returns:
        The results payload (``return_json``) from the completed run.

    Raises:
        :class:`ProtocolRunError`: If the run fails or is cancelled.
        :class:`TimeoutError`: If ``timeout`` is exceeded.

    Example::

        import biolmai

        results = biolmai.run_protocol(
            "antibody-optimization",
            inputs={"sequence": "MKTAYIAKQRQ", "n_rounds": 3},
        )
        print(results["designed_sequences"])
    """
    client = ProtocolClient(api_key=api_key, base_url=base_url)
    return client.run_and_wait(
        slug,
        inputs,
        run_name=run_name,
        timeout=timeout,
        show_progress=show_progress,
    )


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
