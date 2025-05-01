======
BioLM
======

.. autoclass:: biolmai.biolmai.BioLM
   :members:
   :undoc-members:
   :show-inheritance:

**Description**

A universal, high-level client for the BioLM API. Instantiating this class immediately runs the specified action and returns the result.

**Parameters**

- entity (str): The model, database, or calculation to use (e.g., "esm2-8m", "esmfold", "progen2-oas").
- action (str): The action to perform ("generate", "encode", "predict", "search", "finetune", "lookup").
- type (str, optional): The type of item ("sequence", "pdb", "context", etc). Required if `items` are not dicts.
- items (Any or List[Any]): The item(s) to process. Can be a string, list of strings, dict, or list of dicts.
- params (dict, optional): Additional parameters for the action (e.g., temperature, top_p, num_samples).
- api_key (str, optional): API key for authentication.
- raise_httpx (bool, optional): Raise HTTPX errors on bad status codes (default: False).
- stop_on_error (bool, optional): Stop processing on first error (default: False).
- output (str, optional): "memory" (default) or "disk". If "disk", results are written to `file_path`.
- file_path (str, optional): Output file path if `output='disk'`.
- unwrap_single (bool, optional): If True, return a single result instead of a list when only one item is provided.

**Returns**

- The result(s) of the API call. If a single item is provided and `unwrap_single=True`, returns a dict; otherwise, returns a list of dicts.

**Usage Examples**

.. code-block:: python

    # ESM2-8M: encode a single sequence
    result = biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")

    # ESMFold: predict structure for a batch
    result = biolm(entity="esmfold", action="predict", type="sequence", items=["MDNELE", "MENDEL"])

    # ProGen2-OAS: generate new sequences
    result = biolm(
        entity="progen2-oas",
        action="generate",
        type="context",
        items="M",
        params={"temperature": 0.7, "top_p": 0.6, "num_samples": 2, "max_length": 17}
    )
