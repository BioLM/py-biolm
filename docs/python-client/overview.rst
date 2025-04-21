========================
Overview
========================

The BioLM Python client provides a high-level, user-friendly interface for interacting with the BioLM API. It supports both synchronous and asynchronous usage, automatic batching, flexible error handling, and efficient processing of biological data.

Main features:

- High-level BioLM constructor for quick requests
- Sync and async interfaces
- Automatic or custom rate limiting/throttling
- Schema-based batch size detection
- Flexible input formats (single key + list, or list of dicts)
- Low memory usage via generators
- Flexible error handling (raise, continue, or stop on error)
- Universal HTTP client for both sync and async

See :doc:`quickstart` for examples.