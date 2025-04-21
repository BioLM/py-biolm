# File: docs/faq.rst
========================
FAQ
========================

**Q: How do I process a large batch of sequences?**

A: Provide a list of dicts or a list of values; batching is automatic.

**Q: How do I handle errors gracefully?**

A: Set `raise_httpx=False` and choose `stop_on_error=True` or `False`.

**Q: How do I write results to disk?**

A: Set `output='disk'` and provide `file_path`.

**Q: How do I use the async client?**

A: Use `BioLMApiClient` and `await` the methods.

**Q: How do I set a custom rate limit?**

A: Use `rate_limit="1000/second"` or provide your own semaphore.
