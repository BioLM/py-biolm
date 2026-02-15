.. _getting-started-concepts:

=========
Concepts
=========

This page explains core concepts for using the BioLM Python client: the different client interfaces, sync vs async usage, batching, error handling, disk output, and rate limiting.

------------------------
BioLM
------------------------

**BioLM** is the simplest synchronous interface. Use it when you want quick one-liner calls or a straightforward scripting experience.

- **Convenient interface**: Call `biolm(...)` and get your result.
- **Unpacks single-item results**: If you pass a single item, you get a single result (dict), not a list.
- **Runs in the main thread**: No need for `asyncio` or event loops.
- **Great for Jupyter, scripts, and simple batch jobs.**

**Example:**

.. code-block:: python

    from biolmai import biolm

    # Single item: returns a dict
    result = biolm(entity="esmfold", action="predict", items="MDNELE")
    print(result["mean_plddt"])

    # Batch: returns a list of dicts
    result = biolm(entity="esmfold", action="predict", items=["MDNELE", "MENDEL"])

Internally, **BioLM** is a thin synchronous wrapper around the async client, using the `synchronicity` package to run async code in a blocking way.

------------------------
BioLMApi and BioLMApiClient
------------------------

For more control or high-throughput use cases, the SDK provides:

- **BioLMApi** – Synchronous wrapper for `BioLMApiClient`, for users who want a sync interface but more options than `BioLM` (e.g., rate limits, semaphores, retry behavior).
- **BioLMApiClient** – The core **async** client for maximum throughput and integration in async applications.

**When to use which:**

- **Use BioLM** if you want the simplest interface, are in a Jupyter notebook or script, and don't need to manage concurrency.
- **Use BioLMApiClient** if you want to process many requests in parallel, are building a web server or async application, or need to control concurrency, rate limiting, or batching.
- **Use BioLMApi** if you want a sync interface with more control than BioLM (e.g., custom rate limits, retry options).

**BioLMApiClient** always returns a list, even for a single item, unless you set `unwrap_single=True`. **BioLM** and **BioLMApi** return a single dict when you pass a single item.

------------------------
Async and Sync Usage
------------------------

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

**Summary:**

- **BioLM**: Synchronous, easy-to-use, ideal for quick scripts, Jupyter, and most users.
- **BioLMApi / BioLMApiClient**: Fully asynchronous (BioLMApiClient) or sync wrapper (BioLMApi), for advanced users, high-throughput, or async applications.

**Advanced async features:**

- **Concurrent requests**: The async client can batch and send multiple requests at once, using semaphores and rate limiters.
- **Context manager**: Use `async with BioLMApiClient(...) as model:` for clean shutdown.
- **Disk output**: Async disk writing is supported for large jobs.

**Using the async client from sync code:**

.. code-block:: python

    import asyncio
    from biolmai.core.http import BioLMApiClient

    def run_sync():
        model = BioLMApiClient("esmfold")
        return asyncio.run(model.predict(items=[{"sequence": "MDNELE"}]))

    result = run_sync()

Or use the sync wrapper **BioLMApi** for a blocking interface.

**Best practices:**

- For quick jobs, use **BioLM** in sync mode.
- For high-throughput or async apps, use **BioLMApiClient** and `await` your calls.
- For batch jobs in scripts with more control, **BioLMApi** stays synchronous.

------------------------
Batching and Input Flexibility
------------------------

The client supports many input formats and automatic batching.

**Supported input formats:**

1. **Single item (string or dict)** – One sequence or context.
2. **List of values** – Batch of simple items; you must specify `type` (e.g., `type="sequence"`).
3. **List of dicts** – Batch of structured items (e.g., `{"sequence": ...}`); type can be inferred.
4. **List of lists of dicts** – Manual batching: each inner list is one API request (auto-batching disabled).

**How auto-batching works:**

- The client queries the API schema for the model/action to get the maximum batch size (`maxItems`).
- It splits your input into batches of up to that size, sends each batch as a separate request, and returns results in the same order as your input.
- You do **not** need to manually split input; just pass a list of items.

**Manual batching (list of lists):**  
If you pass a list of lists of dicts, each inner list is treated as one batch. Use this for custom batch sizes, error isolation, or testing.

**Input validation:**

- List of dicts: type is inferred from dict keys.
- List of values (not dicts): you **must** specify `type` (e.g., `type="sequence"`).

**Batching and errors:**

- Use `stop_on_error=True` to halt after the first error batch.
- Use `stop_on_error=False` to process all batches and include errors in results.
- Use `retry_error_batches=True` (BioLMApi/BioLMApiClient only) to retry failed batches as single items.

**Summary table:**

+--------------------------+-----------------------------+-----------------------------+
| Input Format             | Auto-batching?              | Use Case                    |
+==========================+=============================+=============================+
| Single value/dict        | Yes                         | Single item                 |
+--------------------------+-----------------------------+-----------------------------+
| List of values           | Yes (needs `type`)          | Batch of simple items       |
+--------------------------+-----------------------------+-----------------------------+
| List of dicts            | Yes                         | Batch of structured items   |
+--------------------------+-----------------------------+-----------------------------+
| List of lists of dicts   | No (manual batching)        | Custom batch control        |
+--------------------------+-----------------------------+-----------------------------+

------------------------
Error Handling
------------------------

Control error behavior with `raise_httpx`, `stop_on_error`, and `retry_error_batches`.

**Key options:**

- **raise_httpx** (default: False for `biolm`/`BioLM`, True for `BioLMApi`/`BioLMApiClient`)
  - If `True`, HTTP errors raise an `httpx.HTTPStatusError` immediately.
  - If `False`, errors are returned as dicts in the results (with `"error"` and `"status_code"` keys).

- **stop_on_error** (default: False)
  - If `True`, processing stops after the first error batch.
  - If `False`, all items are processed; errors appear in the results for failed items.

- **retry_error_batches** (default: False) – **BioLMApi** / **BioLMApiClient** only
  - If `True`, failed batches are retried as single items so you can get partial results.

**Behavior:**

- `raise_httpx=True` always wins: any HTTP error raises immediately.
- `stop_on_error` only applies when `raise_httpx=False`.
- `retry_error_batches` only applies when `raise_httpx=False`.

**Error result shape (when `raise_httpx=False`):**

.. code-block:: python

    {
        "error": "Validation error: ...",
        "status_code": 422
    }

**Parameter availability:**

- **biolm()** / **BioLM**: `raise_httpx` and `stop_on_error` as keyword args; no `retry_error_batches`.
- **BioLMApi** / **BioLMApiClient**: all three; `raise_httpx` and `retry_error_batches` in constructor, `stop_on_error` as method parameter.

------------------------
Disk Output and Batch Error Handling
------------------------

**Disk output**

When you set `output='disk'` and provide a `file_path`, results are written as JSONL (one JSON object per line). Supported in both **BioLM** and **BioLMApi** / **BioLMApiClient**.

- One line per input item, in input order.
- If a batch fails, the error can be repeated for each item in that batch (or handled by retries; see below).
- `stop_on_error=True`: writing stops after the first error batch.
- `stop_on_error=False`: all items are processed; errors are written for failed items.
- `retry_error_batches=True` (BioLMApi/BioLMApiClient): failed batches are retried as single items, so you may get a mix of success and error lines for a batch.

**Batch error handling**

- By default, a failed batch returns/writes an error dict per item in the batch.
- With `retry_error_batches=True`, each item in the failed batch is retried individually for partial recovery.
- With `stop_on_error=True`, processing stops after the first error batch.

**Best practices:**

- Use `output='disk'` for large jobs to avoid memory issues.
- Use `retry_error_batches=True` to maximize successful results when some items may be invalid.
- Use `stop_on_error=True` to halt as soon as any error occurs.

------------------------
Rate Limiting and Throttling
------------------------

The client supports configurable rate limiting and throttling to stay within API quotas.

**Order of application:**

1. **Semaphore (concurrency limit)** – If you pass a semaphore (e.g., `asyncio.Semaphore(5)`), it is acquired first.
2. **Rate limiter** – After the semaphore, the rate limiter enforces a maximum number of requests per time window (sliding window).

Both can be used together: a request must acquire the semaphore and pass the rate limiter.

**Configuration:**

- **Default:** The client uses the API schema’s recommended throttle; no configuration needed.
- **Disable:** Pass `rate_limit=None` and `semaphore=None`.
- **Custom rate limit:** `rate_limit="N/second"` or `rate_limit="N/minute"` (e.g., `"1000/second"`, `"60/minute"`).
- **Custom concurrency:** `semaphore=asyncio.Semaphore(N)`.

**Rate limit format:** `"N/second"` or `"N/minute"`. The limiter uses a sliding window; if the limit is reached, the client waits until a slot is available.

**Examples:**

.. code-block:: python

    from biolmai.core.http import BioLMApi, BioLMApiClient
    import asyncio

    # Default API throttle
    model = BioLMApi("esmfold")

    # Custom rate limit
    model = BioLMApi("esmfold", rate_limit="1000/second")
    model = BioLMApi("esmfold", rate_limit="60/minute")

    # Custom concurrency
    sem = asyncio.Semaphore(5)
    model = BioLMApiClient("esmfold", semaphore=sem)

    # Both: 5 concurrent, 1000/second
    model = BioLMApiClient("esmfold", semaphore=sem, rate_limit="1000/second")

**Best practices:**

- For most users, the default API throttle is sufficient.
- Use a custom rate limit if you have a dedicated quota or want to avoid 429s.
- Use a semaphore to limit load on your own network or resources.
