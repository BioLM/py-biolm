===
FAQ
===

**Q: How do I process a large batch of sequences?**

A: Provide a list of dicts or a list of values; batching is automatic. For advanced control, use `BioLMApi` and access the schema to determine batch size.

**Q: How do I handle errors gracefully?**

A: Set `raise_httpx=False` and choose `stop_on_error=True` or `False`. With `BioLMApi`, you can also set `retry_error_batches=True` to retry failed batches as single items.

**Q: How do I write results to disk?**

A: Set `output='disk'` and provide `file_path` in either `BioLM` or `BioLMApi`.

**Q: How do I use the async client?**

A: Use `BioLMApiClient` and `await` the methods.

**Q: How do I set a custom rate limit?**

A: Use `rate_limit="1000/second"` or provide your own semaphore to `BioLMApi` or `BioLMApiClient`.

**Q: When should I use BioLMApi instead of BioLM?**

A: Use `BioLMApi` if you need:
    - To reuse a client for multiple calls (avoids re-auth)
    - To access the schema or batch size programmatically
    - To call lower-level methods like `.call()` or `.schema()`
    - To do advanced batching or error handling

**Q: What are `.schema()`, `.call()`, and `._batch_call_autoschema_or_manual()` for?**

A:
- `.schema(model, action)`: Fetches the API schema for a model/action, useful for inspecting input/output formats and max batch size.
- `.call(func, items, ...)`: Makes a direct API call for a given function (e.g., "encode"), bypassing batching logic. Useful for custom workflows or debugging.
- `._batch_call_autoschema_or_manual(func, items, ...)`: Internal batching logic that splits items into batches based on schema, handles errors, and can write to disk. Advanced users may use this for custom batching or error handling.
