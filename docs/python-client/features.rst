========================
Features
========================

- **High-level constructor**: Instantly run an API call with a single line.
- **Sync and async**: Use `biolm()` or `BioLM` for sync, or `BioLMApiClient` for async.
- **Auto-batching + concurrency**: Items are split into batches (schema-based max size) and sent in parallel. By default, up to 16 requests run concurrently, with API-recommended rate limiting. See :doc:`rate_limiting`.
- **Generators supported**: Pass a generator or iterator instead of a list; items are consumed batch-by-batch so you never load the full dataset into memory. Ideal for large FASTA/FASTQ files or streaming pipelines.
- **Flexible input**: Single value, list, tuple, generator, or list of lists (manual batching). Use `type="sequence"` when items are strings.
- **Configurable concurrency**: Custom semaphore (in-flight limit), rate limits (e.g. ``1000/second``), or disable throttling for advanced control.
- **Error handling**: Raise HTTPX errors, continue on error, or stop on first error. Optionally retry failed batches as single items.
- **Disk output**: Write results as JSONL to disk for very large jobs.
- **Direct access**: Use `BioLMApi` for `.schema()`, `.call()`, and advanced batching/error control.

**Example endpoints and actions:**

- `esm2-8m/encode`: Embedding for protein sequences.
- `esmfold/predict`: Structure prediction for protein sequences.
- `progen2-oas/generate`: Sequence generation from a context string.
- `dnabert2/predict`: Masked prediction for protein sequences.
- `ablang2/encode`: Embeddings for paired-chain antibodies.
