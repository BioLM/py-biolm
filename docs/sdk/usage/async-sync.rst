========================
Async and Sync Usage
========================

**Synchronous usage:**

.. code-block:: python

    from biolmai import biolm
    result = biolm(entity="esmfold", action="predict", items="MDNELE")

**Asynchronous usage:**

.. code-block:: python

    from biolmai.core.http import BioLMApiClient
    import asyncio

    async def main():
        model = BioLMApiClient("esmfold")
        result = await model.predict(items=[{"sequence": "MDNELE"}])
        print(result)

    asyncio.run(main())


------------------------
High-Level Summary
------------------------

- **BioLM**: Synchronous, easy-to-use, ideal for quick scripts, Jupyter, and most users.
- **BioLMApi / BioLMApiClient**: Fully asynchronous, for advanced users, high-throughput, or integration in async applications.

------------------------
Synchronous Usage (BioLM)
------------------------

- **Convenient interface**: Just call ``biolm(...)`` and get your result.
- **Unpacks single-item results**: If you pass a single item, you get a single result (not a list).
- **Runs in the main thread**: No need for ``asyncio`` or event loops.
- **Great for Jupyter, scripts, and simple batch jobs**.

**Example:**

.. code-block:: python

    from biolmai import biolm

    # Single item: returns a dict
    result = biolm(entity="esmfold", action="predict", items="MDNELE")
    print(result["mean_plddt"])

    # Batch: returns a list of dicts
    result = biolm(entity="esmfold", action="predict", items=["MDNELE", "MENDEL"])
    print(result[0]["mean_plddt"], result[1]["mean_plddt"])

------------------------
Asynchronous Usage (BioLMApi/BioLMApiClient)
------------------------

- **True async**: Designed for async Python (e.g., FastAPI, web servers, or high-throughput pipelines).
- **Concurrent requests**: Can send many requests in parallel, maximizing API throughput.
- **Manual control**: You manage the event loop and can await results.
- **No GIL/threading issues**: All network I/O is non-blocking.

**Example:**

.. code-block:: python

    import asyncio
    from biolmai.core.http import BioLMApiClient

    async def main():
        model = BioLMApiClient("esmfold")
        # Batch: returns a list of dicts
        result = await model.predict(items=[{"sequence": "MDNELE"}, {"sequence": "MENDEL"}])
        print(result[0]["mean_plddt"], result[1]["mean_plddt"])

    asyncio.run(main())

------------------------
How It Works Internally
------------------------

- **BioLM** is a thin synchronous wrapper around the async client, using the ``synchronicity`` package to run async code in a blocking way.
- **BioLMApi** is a synchronous wrapper for ``BioLMApiClient`` (async), for users who want a sync interface but more control than ``BioLM``.
- **BioLMApiClient** is the core async client.

------------------------
Choosing Between Sync and Async
------------------------

- **Use BioLM** if:
    - You want the simplest interface.
    - You're in a Jupyter notebook or a script.
    - You don't need to manage concurrency yourself.

- **Use BioLMApiClient** if:
    - You want to process many requests in parallel (e.g., thousands of sequences).
    - You're building a web server, pipeline, or async application.
    - You want to control concurrency, rate limiting, or batching.

- **Use BioLMApi** if:
    - You want a sync interface but with more control/options than BioLM.

------------------------
Advanced Async Features
------------------------

- **Concurrent requests**: The async client can batch and send multiple requests at once, using semaphores and rate limiters.
- **Context manager support**: Use a context manager for clean shutdown:

  .. code-block:: python

      async with BioLMApiClient("esmfold") as model:
          result = await model.predict(items=[{"sequence": "MDNELE"}])
- **Disk output**: Async disk writing is supported for large jobs (see :ref:`disk-output` in :doc:`usage`).
- **Manual batching**: You can control batch size and composition for maximum throughput.

------------------------
Sync/Async Interoperability
------------------------

- You can use the async client in a thread pool from sync code:

.. code-block:: python

    import asyncio
    from biolmai.core.http import BioLMApiClient

    def run_sync():
        model = BioLMApiClient("esmfold")
        return asyncio.run(model.predict(items=[{"sequence": "MDNELE"}]))

    result = run_sync()

- Or, use the sync wrapper (``BioLMApi``) for a blocking interface.

------------------------
Unpacking Single-Item Results
------------------------

- **BioLM** and **BioLMApi**: If you pass a single item, you get a single result (dict), not a list.
- **BioLMApiClient**: Always returns a list, even for a single item (unless you set ``unwrap_single=True``).

------------------------
Jupyter Notebook Usage
------------------------

- In Jupyter, the library detects the notebook environment and applies ``nest_asyncio`` so that sync wrappers (e.g. ``biolm()``, ``BioLMApi``) work correctly inside the notebook kernel.
- You can use the sync interface as usual: ``result = biolm(...)`` or ``model = BioLMApi("esmfold"); result = model.predict(...)``.
- For **best performance** in Jupyter (e.g. large batches), use the async client with top-level ``await`` so the event loop runs natively:

.. code-block:: python

    from biolmai.core.http import BioLMApiClient
    model = BioLMApiClient("esmfold")
    result = await model.predict(items=[{"sequence": "MDNELE"}, {"sequence": "MENDEL"}])

- Sync wrappers also work in Jupyter; use top-level ``await`` with ``BioLMApiClient`` when you want maximum throughput.

------------------------
Best Practices
------------------------

- For quick jobs, use ``BioLM`` in sync mode.
- For high-throughput or async apps, use ``BioLMApiClient`` and ``await`` your calls.
- For batch jobs in scripts, ``BioLMApi`` gives you more control but stays synchronous.
- Always use the async client in async code (e.g., FastAPI, aiohttp, etc).

------------------------
See Also
------------------------

- :doc:`batching`
- :doc:`error-handling`
- :doc:`rate_limiting`
- :doc:`usage` (includes Disk output)
