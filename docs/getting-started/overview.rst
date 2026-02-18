========================
Overview
========================

The BioLM Python client provides a high-level, user-friendly interface for interacting with the BioLM API. It supports both synchronous and asynchronous usage, automatic batching, flexible error handling, and efficient processing of biological data.

Main features:

- **High-level interface:** One-line API calls with `biolm()` or `BioLM`; instant results, no setup.
- **Sync and async:** Use `biolm()` for scripts/notebooks, or `BioLMApiClient` for async apps (FastAPI, high throughput).
- **Built-in performance:** By default, the client auto-batches your input, runs up to 16 requests concurrently (semaphore), and applies API-recommended rate limiting—no configuration needed for most use cases.
- **Flexible input:** Lists, tuples, or **generators**—pass a generator to process huge datasets without loading all items into memory (consumed batch-by-batch).
- **Error handling:** Raise on error, continue and collect errors, or stop on first error; optional retry of failed batches.
- **Disk output:** Write results as JSONL for large jobs.

See :doc:`getting-started/quickstart` for examples, :doc:`../sdk/usage/batching` for input flexibility, and :doc:`../sdk/usage/rate_limiting` for concurrency and throttling options.
