# File: docs/rate_limiting.rst
========================
Rate Limiting and Throttling
========================

- By default, the client uses the API's throttle rate.
- You can disable throttling, provide your own `asyncio.Semaphore`, or set a custom rate limit (e.g., "1000/second").

**Examples:**

.. code-block:: python

    # Use API's throttle rate (default)
    model = BioLMApi("esmfold")

    # Custom rate limit
    model = BioLMApi("esmfold", rate_limit="1000/second")

    # Custom semaphore
    import asyncio
    sem = asyncio.Semaphore(5)
    model = BioLMApiClient("esmfold", semaphore=sem)
