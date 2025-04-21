# File: docs/error_handling.rst
========================
Error Handling
========================

You can control error handling with `raise_httpx` and `stop_on_error`:

- `raise_httpx=True`: Raise HTTPX errors on bad status codes.
- `raise_httpx=False, stop_on_error=True`: Stop on first error, no exception raised.
- `raise_httpx=False, stop_on_error=False`: Continue on errors, errors are returned as dicts.

**Examples:**

.. code-block:: python

    # Raise HTTPX errors
    BioLM(entity="esmfold", action="predict", items="BAD", raise_httpx=True)

    # Continue on errors
    BioLM(entity="esmfold", action="predict", items=["GOOD", "BAD"], stop_on_error=False, raise_httpx=False)

    # Stop on first error
    BioLM(entity="esmfold", action="predict", items=["GOOD", "BAD"], stop_on_error=True, raise_httpx=False)
