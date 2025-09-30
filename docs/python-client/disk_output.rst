========================
Disk Output and Batch Error Handling
========================

Disk Output
-----------

When you set `output='disk'` and provide a `file_path`, results are written as JSONL (one JSON object per line). This is supported in both `BioLM` and `BioLMApi`/`BioLMApiClient`.

**Key points:**

- Each input item produces one line in the output file, in the same order as the input.
- If a batch fails (e.g., due to a validation error), the error is repeated for each item in that batch.
- If you use `stop_on_error=True`, writing stops after the first error batch.
- If you use `stop_on_error=False`, all items are processed, and errors are written for failed items.
- If you use `retry_error_batches=True` (BioLMApi/BioLMApiClient only), failed batches are retried as single items, so you may get a mix of error and success lines for a batch.

**Examples:**

.. code-block:: python

    # Write all results to disk, continue on errors
    biolm(entity="esmfold", action="predict", type="sequence", items=["SEQ1", "BADSEQ"], output='disk', file_path="results.jsonl", stop_on_error=False)

    # Write to disk, stop on first error
    biolm(entity="esmfold", action="predict", type="sequence", items=["SEQ1", "BADSEQ"], output='disk', file_path="results.jsonl", stop_on_error=True)

    # Advanced: retry failed batches as single items (BioLMApi only)
    from biolmai.client import BioLMApi
    model = BioLMApi("esm2-8m", retry_error_batches=True)
    model.encode(items=[{"sequence": "SEQ1"}, {"sequence": "BADSEQ"}], output='disk', file_path="out.jsonl")

Batch Error Handling
--------------------

**How errors are handled in batch mode:**

- If a batch fails (e.g., due to a validation error), the default is to return/write an error dict for each item in the batch.
- With `retry_error_batches=True`, the client will retry each item in the failed batch individually. This allows you to recover partial results from a batch that would otherwise be all errors.
- If `stop_on_error=True`, processing stops after the first error batch (no further items are processed or written).
- If `stop_on_error=False`, all batches are processed, and errors are included for failed items.

**Subtleties:**

- The output file will always have one line per input item, unless you use `stop_on_error=True` and an error occurs before the end.
- If a batch is retried and some items succeed, you will get a mix of error and success lines for that batch.
- The order of results in the file matches the order of input items.

**Example:**

Suppose you have 10 items, and item 5 is invalid. With a batch size of 8:

- If `stop_on_error=False`, you get 8 lines (first batch: 8 items, with one or more errors), then 2 more lines (second batch).
- If `stop_on_error=True`, you get only the first 8 lines (processing stops at the first error batch).
- If `retry_error_batches=True`, the failed batch is retried item-by-item, so you get 8 lines: 7 successes, 1 error.

**Best practices:**

- Use `output='disk'` for large jobs to avoid memory issues.
- Use `retry_error_batches=True` if you want to maximize the number of successful results, even if some items in a batch are invalid.
- Use `stop_on_error=True` if you want to halt processing as soon as any error occurs.

**See also:**

- :doc:`error_handling`
- :doc:`batching`
