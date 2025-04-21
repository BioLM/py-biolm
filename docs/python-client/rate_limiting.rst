========================
Rate Limiting and Throttling
========================

The BioLM Python client supports flexible rate limiting and throttling to help you stay within API quotas and avoid overloading the server. You can use the default API throttle, disable it, or set your own custom limits.

------------------------
Order of Throttling/Concurrency
------------------------

When making a request, the client applies throttling in this order:

1. **Semaphore (Concurrency Limit):**
   - If you provide a semaphore (e.g., `asyncio.Semaphore(5)`), it is acquired first.
   - This limits the number of concurrent requests.

2. **Rate Limiter (Requests per Time Window):**
   - After acquiring the semaphore, the rate limiter is applied.
   - This enforces a maximum number of requests per second or minute, using a sliding window.

**Both can be used together**:  
If both are set, a request must acquire the semaphore *and* pass the rate limiter before proceeding.

------------------------
How to Configure
------------------------

- **Default (API throttle):**
  - The client queries the API schema for the recommended throttle rate and uses it.
  - No need to set anything.

- **Disable Throttling:**
  - Pass `rate_limit=None` and `semaphore=None` (default for advanced users).

- **Custom Rate Limit:**
  - Pass `rate_limit="N/second"` or `rate_limit="N/minute"`.
  - Example: `rate_limit="1000/second"` or `rate_limit="60/minute"`.

- **Custom Semaphore:**
  - Pass `semaphore=asyncio.Semaphore(N)` to limit concurrency to N requests at a time.

------------------------
Rate Limit String Parsing and Windowing
------------------------

- The `rate_limit` string must be in the form `"N/second"` or `"N/minute"`.
- The limiter uses a **sliding window**:
    - For `"1000/second"`, at most 1000 requests can start in any rolling 1-second window.
    - For `"60/minute"`, at most 60 requests can start in any rolling 60-second window.
- If the limit is reached, the client waits until a slot is available.

**Examples:**

.. code-block:: python

    # Use API's default throttle rate (recommended)
    model = BioLMApi("esmfold")

    # Custom rate limit: 1000 requests per second
    model = BioLMApi("esmfold", rate_limit="1000/second")

    # Custom rate limit: 60 requests per minute
    model = BioLMApi("esmfold", rate_limit="60/minute")

    # Custom concurrency limit: at most 5 requests at once
    import asyncio
    sem = asyncio.Semaphore(5)
    model = BioLMApiClient("esmfold", semaphore=sem)

    # Both: at most 5 concurrent, and at most 1000 per second
    model = BioLMApiClient("esmfold", semaphore=sem, rate_limit="1000/second")

------------------------
Implementation Details
------------------------

- **Semaphore**: Limits the number of requests in flight at any moment.
- **Rate Limiter**: Tracks timestamps of recent requests and enforces the N-per-window rule.
- **Sliding Window**: The limiter removes timestamps older than the window (1s or 60s) and only allows a new request if the count is below N.
- **Acquisition Order**: Semaphore is acquired first, then the rate limiter.

------------------------
Best Practices
------------------------

- For most users, the default API throttle is sufficient and safest.
- Use a custom rate limit if you have a dedicated quota or want to avoid 429 errors.
- Use a semaphore to avoid overwhelming your own network or compute resources.
- For very high throughput, combine both.

------------------------
See Also
------------------------

- :doc:`error_handling`
- :doc:`batching`
- :doc:`faq`
