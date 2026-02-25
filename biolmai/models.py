"""Model API operations for BioLM."""
import logging
from typing import Callable, Optional, Union, List, Any

from biolmai.core.http import BioLMApi, BioLMApiClient
from biolmai.core.utils import is_list_of_lists
from biolmai.examples import ExampleGeneratorSync
from biolmai.progress import rich_progress

log = logging.getLogger("biolm_util")


class Model:
    """
    User-friendly model interface for BioLM API.
    
    Args:
        name (str): The model name (e.g., 'esm2-8m', 'esmfold').
        api_key (Optional[str]): API key for authentication.
        ``**kwargs``: Additional arguments passed to BioLMApi.
    """
    def __init__(self, name: str, api_key: Optional[str] = None, **kwargs):
        self.name = name
        self.api_key = api_key
        self._client = BioLMApi(name, api_key=api_key, **kwargs)
    
    def predict(self, items: Union[Any, List[Any]], type: Optional[str] = None, params: Optional[dict] = None, progress: bool = True, progress_callback: Optional[Callable[[int, int], None]] = None, **kwargs):
        """Predict using the model.
        
        Args:
            items: Single item or list of items to predict.
            type: Type of item (e.g., 'sequence', 'pdb'). Required if items are not dicts.
            params: Optional parameters for the prediction.
            progress: If True (default), show Rich progress bar (multiple items only).
            progress_callback: Optional (completed, total) callback; overrides progress=True if provided.
            ``**kwargs``: Additional arguments (stop_on_error, output, file_path, etc.).

        Returns:
            Prediction results.
        """
        items_dicts = self._prepare_items(items, type)
        return self._run_with_progress("predict", items_dicts, params=params, progress=progress, progress_callback=progress_callback, **kwargs)
    
    def encode(self, items: Union[Any, List[Any]], type: Optional[str] = None, params: Optional[dict] = None, progress: bool = True, progress_callback: Optional[Callable[[int, int], None]] = None, **kwargs):
        """Encode using the model.
        
        Args:
            items: Single item or list of items to encode.
            type: Type of item (e.g., 'sequence'). Required if items are not dicts.
            params: Optional parameters for the encoding.
            progress: If True (default), show Rich progress bar (multiple items only).
            progress_callback: Optional (completed, total) callback; overrides progress=True if provided.
            ``**kwargs``: Additional arguments (stop_on_error, output, file_path, etc.).

        Returns:
            Encoding results.
        """
        items_dicts = self._prepare_items(items, type)
        return self._run_with_progress("encode", items_dicts, params=params, progress=progress, progress_callback=progress_callback, **kwargs)
    
    def generate(self, items: Union[Any, List[Any]], type: Optional[str] = None, params: Optional[dict] = None, progress: bool = True, progress_callback: Optional[Callable[[int, int], None]] = None, **kwargs):
        """Generate using the model.
        
        Args:
            items: Single item or list of items to generate from.
            type: Type of item (e.g., 'context', 'pdb'). Required if items are not dicts.
            params: Optional parameters for the generation.
            progress: If True (default), show Rich progress bar (multiple items only).
            progress_callback: Optional (completed, total) callback; overrides progress=True if provided.
            ``**kwargs``: Additional arguments (stop_on_error, output, file_path, etc.).

        Returns:
            Generation results.
        """
        items_dicts = self._prepare_items(items, type)
        return self._run_with_progress("generate", items_dicts, params=params, progress=progress, progress_callback=progress_callback, **kwargs)
    
    def lookup(self, query: Union[dict, List[dict]], **kwargs):
        """Lookup using the model.
        
        Args:
            query: Query dict or list of query dicts.
            ``**kwargs``: Additional arguments (raw, output, file_path).
            
        Returns:
            Lookup results.
        """
        return self._client.lookup(query=query, **kwargs)

    def _run_with_progress(
        self,
        action: str,
        items_dicts: Union[List[dict], List[List[dict]]],
        params: Optional[dict] = None,
        progress: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ):
        """Run predict/encode/generate with optional Rich progress or external callback."""
        method = getattr(self._client, action)
        if progress_callback is not None:
            return method(items=items_dicts, params=params, progress_callback=progress_callback, **kwargs)
        if progress and items_dicts:
            is_lol = isinstance(items_dicts, (list, tuple)) and items_dicts and isinstance(items_dicts[0], (list, tuple))
            total_items = sum(len(b) for b in items_dicts) if is_lol else len(items_dicts)
            if total_items > 1:
                with rich_progress(total_items, description=f"Processing {total_items} item(s) with {self.name}...") as callback:
                    return method(items=items_dicts, params=params, progress_callback=callback, **kwargs)
        return method(items=items_dicts, params=params, **kwargs)
    
    def get_example(self, action: Optional[str] = None, format: str = 'python', **kwargs) -> str:
        """Get SDK usage example for this model.
        
        Args:
            action: Action name (encode, predict, generate, lookup). If None, generates for first available action.
            format: Output format ('python', 'markdown', 'rst', 'json').
            ``**kwargs``: Additional arguments passed to ExampleGenerator (base_url).
            
        Returns:
            Formatted example string.
        """
        # Use stored api_key or allow override via kwargs
        api_key = kwargs.pop('api_key', self.api_key)
        
        # Get base_url from client if available
        base_url = kwargs.get('base_url')
        if base_url is None and hasattr(self._client, 'base_url'):
            # Extract domain from base_url (remove /api/v3 suffix)
            client_base = self._client.base_url.rstrip('/')
            if client_base.endswith('/api/v3'):
                base_url = client_base[:-7]  # Remove '/api/v3'
            elif client_base.endswith('/api/v2'):
                base_url = client_base[:-7]  # Remove '/api/v2'
        
        generator = ExampleGeneratorSync(api_key=api_key, base_url=base_url)
        try:
            return generator.generate_example(self.name, action, format)
        finally:
            generator.shutdown()
    
    def get_examples(self, format: str = 'python', **kwargs) -> str:
        """Get SDK usage examples for all supported actions of this model.
        
        Args:
            format: Output format ('python', 'markdown', 'rst', 'json').
            ``**kwargs``: Additional arguments passed to ExampleGenerator (base_url).
            
        Returns:
            Formatted examples string with all actions.
        """
        # Use stored api_key or allow override via kwargs
        api_key = kwargs.pop('api_key', self.api_key)
        
        # Get base_url from client if available
        base_url = kwargs.get('base_url')
        if base_url is None and hasattr(self._client, 'base_url'):
            # Extract domain from base_url (remove /api/v3 suffix)
            client_base = self._client.base_url.rstrip('/')
            if client_base.endswith('/api/v3'):
                base_url = client_base[:-7]  # Remove '/api/v3'
            elif client_base.endswith('/api/v2'):
                base_url = client_base[:-7]  # Remove '/api/v2'
        
        generator = ExampleGeneratorSync(api_key=api_key, base_url=base_url)
        try:
            # Pass None as action to generate for all actions
            return generator.generate_example(self.name, None, format)
        finally:
            generator.shutdown()
    
    def _prepare_items(self, items: Union[Any, List[Any]], type: Optional[str] = None) -> List[dict]:
        """Prepare items for API calls."""
        if isinstance(items, list):
            items_list = items
        else:
            items_list = [items]
        
        is_lol, first_n, rest_iter = is_list_of_lists(items_list, check_n=10)
        if is_lol:
            for batch in first_n:
                if not all(isinstance(x, dict) for x in batch):
                    raise ValueError("All items in each batch must be dicts when passing a list of lists.")
            if type is not None:
                raise ValueError("Do not specify `type` when passing a list of lists of dicts for `items`.")
            return list(first_n) + list(rest_iter)
        elif all(isinstance(v, dict) for v in items_list):
            return items_list
        else:
            if type is None:
                raise ValueError("If `items` are not dicts, `type` must be specified.")
            return [{type: v} for v in items_list]


# Backward compatibility alias
class BioLM:
    """
    Universal client for BioLM API (deprecated, use Model instead).
    
    This class is kept for backward compatibility. New code should use Model.
    """
    def __new__(
        cls,
        *,
        entity: str,
        action: str,
        type: Optional[str] = None,
        items: Union[Any, List[Any]],
        params: Optional[dict] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        self = super().__new__(cls)
        self.entity = entity
        self.action = action
        self.type = type
        self.items = items
        self.params = params
        self.api_key = api_key
        self._class_kwargs = kwargs
        return self.run()

    def run(self) -> Any:
        """
        Run the specified action on the entity with the given item(s).
        Returns the result(s), unpacked if a single item was provided.
        """
        # Always pass a list of items
        if isinstance(self.items, list):
            items = self.items
        else:
            items = [self.items]

        is_lol, first_n, rest_iter = is_list_of_lists(items, check_n=10)
        if is_lol:
            for batch in first_n:
                if not all(isinstance(x, dict) for x in batch):
                    raise ValueError("All items in each batch must be dicts when passing a list of lists.")
            if self.type is not None:
                raise ValueError("Do not specify `type` when passing a list of lists of dicts for `items`.")
            items_dicts = list(first_n) + list(rest_iter)
        elif all(isinstance(v, dict) for v in items):
            items_dicts = items
        else:
            if self.type is None:
                raise ValueError("If `items` are not dicts, `type` must be specified.")
            items_dicts = [{self.type: v} for v in items]

        unwrap_single = self._class_kwargs.pop('unwrap_single', True)

        # Instantiate BioLMApi with correct settings
        action_kwargs = {k: v for k, v in dict(
            stop_on_error=self._class_kwargs.pop('stop_on_error', None),
            output=self._class_kwargs.pop('output', None),
            file_path=self._class_kwargs.pop('file_path', None),
            overwrite=self._class_kwargs.pop('overwrite', None),
        ).items() if v is not None}

        model = BioLMApi(
            self.entity,
            api_key=self.api_key,
            unwrap_single=unwrap_single,
            **self._class_kwargs,
        )

        # Map action to method
        action_map = {
            'generate': model.generate,
            'predict': model.predict,
            'encode': model.encode,
            'search': getattr(model, 'search', None),
            'finetune': getattr(model, 'finetune', None),
            'lookup': model.lookup,
        }
        if self.action not in action_map or action_map[self.action] is None:
            raise ValueError(f"Action '{self.action}' is not amongst the available actions {', '.join(action_map.keys())}.")

        # Prepare kwargs for the method
        method = action_map[self.action]
        kwargs = {
            'items': items_dicts,
            'params': self.params,
        }
        kwargs.update(action_kwargs)
        # Remove None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Call the method
        result = method(**kwargs)

        return result


# Convenience functions
def predict(model_name: str, items: Union[Any, List[Any]], type: Optional[str] = None, params: Optional[dict] = None, **kwargs):
    """Quick prediction using a model."""
    model = Model(model_name)
    return model.predict(items=items, type=type, params=params, **kwargs)


def encode(model_name: str, items: Union[Any, List[Any]], type: Optional[str] = None, params: Optional[dict] = None, **kwargs):
    """Quick encoding using a model."""
    model = Model(model_name)
    return model.encode(items=items, type=type, params=params, **kwargs)


def generate(model_name: str, items: Union[Any, List[Any]], type: Optional[str] = None, params: Optional[dict] = None, **kwargs):
    """Quick generation using a model."""
    model = Model(model_name)
    return model.generate(items=items, type=type, params=params, **kwargs)

