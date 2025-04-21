========
Features
========

- **High-level constructor**: Instantly run an API call with a single line.
- **Sync and async**: Use `BioLM` for sync, or `BioLMApiClient` for async.
- **Flexible rate limiting**: Use API throttle, disable, or set your own (e.g., '1000/second').
- **Schema-based batching**: Automatically queries API for max batch size.
- **Flexible input**: Accepts a single key and list, or list of dicts, or list of lists for advanced batching.
- **Low memory**: Uses generators for validation and batching.
- **Error handling**: Raise HTTPX errors, continue on error, or stop on first error.
- **Disk output**: Write results as JSONL to disk.
- **Universal HTTP client**: Efficient for both sync and async.

See :doc:`usage` for details and examples.
