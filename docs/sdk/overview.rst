.. _sdk-overview:

Overview
========

The BioLM Python SDK lets you call BioLM models (ESM2, ESMFold, ProGen2-OAS, and others) from Python with minimal setup. You can encode sequences, predict structures, and generate sequences using a simple function call or a class-based interface.

**What you can do**

- **Encode** sequences (e.g. ``esm2-8m``) to get embeddings.
- **Predict** protein structures (e.g. ``esmfold``) from sequences.
- **Generate** new sequences (e.g. ``progen2-oas``) from context.

**Ways to use the SDK**

- **Quick one-off calls:** :code:`biolm(entity=..., action=..., items=...)` or :code:`Model("esmfold").predict(...)` — see :doc:`usage/usage`.
- **Sync or async:** Use :code:`biolm` / :code:`Model` for scripts and notebooks, or :code:`BioLMApi` / :code:`BioLMApiClient` for high throughput and async apps — see :doc:`usage/async-sync`.
- **Batching and control:** Automatic batching, rate limiting, disk output, and error handling — see :doc:`usage/batching`, :doc:`usage/rate_limiting`, :doc:`usage/error-handling`.

**Next steps**

- :doc:`models` — Available models and quick examples.
- :doc:`usage/usage` — Usage patterns, disk output, and when to use which interface.
- :doc:`faq` — Common questions.
- :doc:`api-reference/index` — Full API reference.
