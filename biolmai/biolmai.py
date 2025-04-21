"""Main module."""
import logging

log = logging.getLogger("biolm_util")

from typing import Optional, Union, List, Any
from biolmai.client import BioLMApi, is_list_of_lists


class BioLM:
    """
    Universal client for BioLM API.

    Args:
        entity (str): The entity name (model, database, calculation, etc).
        action (str): The action to perform (e.g., 'generate', 'encode', 'predict', 'search', 'finetune').
        type (str): The type of item (e.g., 'sequence', 'pdb', 'fasta_str').
        item (Union[Any, List[Any]]): The item(s) to process.
        params (Optional[dict]): Optional parameters for the action.
        raise_httpx (bool): Whether to raise HTTPX errors.
        stop_on_error (bool): Stop on first error if True.
        output (str): 'memory' or 'disk'.
        file_path (Optional[str]): Output file path if output='disk'.
        api_key (Optional[str]): API key for authentication.
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
        # if not self.items:
            # return self.items

        # Always pass a list of items to BioLMModel
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

        # Instantiate BioLMModel with correct settings
        # Need these for the `action` method on BioLMApi; other kwargs to BioLMApi init
        action_kwargs = {k: v for k, v in dict(
            stop_on_error=self._class_kwargs.pop('stop_on_error', None),
            output=self._class_kwargs.pop('output', None),
            file_path=self._class_kwargs.pop('file_path', None),
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

# Example usage:
# result = BioLM(entity="esmfold", action="predict", type="sequence", item="MKT...").run()
