========================
Error Handling
========================

The BioLM Python client provides flexible error handling for both single and batch API calls. You can control error behavior using the `raise_httpx` and `stop_on_error` options.

------------------------
Key Options
------------------------

- **raise_httpx (default: False)**  
  - If `True`, HTTP errors (e.g., 400/422/500) raise an `httpx.HTTPStatusError` exception immediately.
  - If `False`, errors are returned as dicts in the results (with `"error"` and `"status_code"` keys).

- **stop_on_error (default: False)**  
  - If `True`, processing stops after the first error batch. Only results up to and including the error are returned (or written to disk).
  - If `False`, all items are processed; errors are included in the results for failed items.

- **retry_error_batches (BioLMApi/BioLMApiClient only, default: False)**
  - If `True`, failed batches are retried as single items, so you may recover partial results from a batch that would otherwise be all errors.

------------------------
How the Options Interact
------------------------

- `raise_httpx=True` always takes precedence:  
  - Any HTTP error will immediately raise an exception, regardless of `stop_on_error`.
  - If you want to handle errors as return values, set `raise_httpx=False`.

- `stop_on_error` only applies when `raise_httpx=False`:
  - If `stop_on_error=True`, processing halts after the first error batch (no further items are processed).
  - If `stop_on_error=False`, all items are processed, and errors are included in the results.

- `retry_error_batches` is only relevant when `raise_httpx=False`:
  - If a batch fails, each item in the batch is retried individually. This can help recover partial results.

------------------------
Behavior Matrix
------------------------

+-------------------+-------------------+-------------------+-------------------------------------------------------------+
| raise_httpx       | stop_on_error     | retry_error_batches| Behavior                                                    |
+===================+===================+===================+=============================================================+
| True              | (any)             | (any)             | Exception raised on first HTTP error (no results returned)   |
+-------------------+-------------------+-------------------+-------------------------------------------------------------+
| False             | True              | False              | Stop after first error batch; errors returned as dicts       |
+-------------------+-------------------+-------------------+-------------------------------------------------------------+
| False             | False             | False              | Continue on errors; errors returned as dicts in results      |
+-------------------+-------------------+-------------------+-------------------------------------------------------------+
| False             | True/False        | True               | Failed batches retried as single items; errors as dicts      |
+-------------------+-------------------+-------------------+-------------------------------------------------------------+

------------------------
Examples
------------------------

**1. Raising exceptions on error (default for BioLMApi/BioLMApiClient):**

.. code-block:: python

    from biolmai import biolm
    try:
        result = biolm(entity="esmfold", action="predict", items="BADSEQ", raise_httpx=True)
    except Exception as e:
        print("Caught exception:", e)

**2. Continue on errors, errors as dicts:**

.. code-block:: python

    result = biolm(entity="esmfold", action="predict", items=["GOODSEQ", "BADSEQ"], raise_httpx=False, stop_on_error=False)
    # result[0] is a normal result, result[1] is a dict with "error" and "status_code"

**3. Stop on first error:**

.. code-block:: python

    result = biolm(entity="esmfold", action="predict", items=["GOODSEQ", "BADSEQ", "ANOTHER"], raise_httpx=False, stop_on_error=True)
    # Only results up to and including the first error are returned

**4. Retrying failed batches as single items (BioLMApi only):**

.. code-block:: python

    from biolmai.client import BioLMApi
    model = BioLMApi("esm2-8m", raise_httpx=False, retry_error_batches=True)
    result = model.encode(items=[{"sequence": "GOOD"}, {"sequence": "BAD"}])
    # If a batch fails, each item is retried individually

------------------------
What do error results look like?
------------------------

If `raise_httpx=False`, errors are returned as dicts, e.g.:

.. code-block:: python

    {
        "error": "Validation error: ...",
        "status_code": 422
    }

For batch calls, the result is a list, with each item either a normal result or an error dict.

------------------------
Catching Exceptions
------------------------

If you set `raise_httpx=True`, you must catch exceptions:

.. code-block:: python

    from biolmai import biolm
    try:
        result = biolm(entity="esmfold", action="predict", items="BADSEQ", raise_httpx=True)
    except Exception as e:
        print("Caught exception:", e)

If you set `raise_httpx=False`, you can check for errors in the results:

.. code-block:: python

    result = biolm(entity="esmfold", action="predict", items=["GOODSEQ", "BADSEQ"], raise_httpx=False)
    for r in result:
        if "error" in r:
            print("Error:", r["error"])
        else:
            print("Success:", r)

------------------------
Best Practices
------------------------

- Use `raise_httpx=True` for strict error handling (fail fast, catch exceptions).
- Use `raise_httpx=False, stop_on_error=False` to process all items and collect all errors.
- Use `raise_httpx=False, stop_on_error=True` to halt on the first error batch.
- Use `retry_error_batches=True` (with `raise_httpx=False`) to maximize successful results in large batches.
- Always check for `"error"` in results if not raising exceptions.

------------------------
See Also
------------------------

- :doc:`batching`
- :doc:`disk_output`
- :doc:`faq`
