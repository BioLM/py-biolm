========================
Batching and Input Flexibility
========================

The BioLM Python client supports a wide variety of input formats and batching strategies to maximize flexibility and efficiency. This document explains all supported input types, how auto-batching works, and how to use advanced batching for custom workflows.

------------------------
Supported Input Formats
------------------------

You can provide input in several ways:

**1. Single item (string or dict):**
  - For a single sequence or context.
  - Example:

    .. code-block:: python

        biolm(entity="esm2-8m", action="encode", type="sequence", items="MSILVTRPSPAGEEL")

**2. List of values (strings, numbers, etc):**
  - For a batch of simple items (e.g., sequences).
  - You must specify `type` (e.g., `type="sequence"`).
  - Example:

    .. code-block:: python

        biolm(entity="esm2-8m", action="encode", type="sequence", items=["SEQ1", "SEQ2"])

**3. List of dicts:**
  - For a batch of structured items (e.g., `{"sequence": ...}`).
  - No need to specify `type` if your dicts have the correct keys.
  - Example:

    .. code-block:: python

        biolm(entity="esmfold", action="predict", items=[{"sequence": "SEQ1"}, {"sequence": "SEQ2"}])

**4. List of lists of dicts (advanced/manual batching):**
  - Each inner list is treated as a batch and sent as a single API request.
  - Useful for custom batching, controlling batch size, or mixing valid/invalid items.
  - Example:

    .. code-block:: python

        batches = [
            [{"sequence": "SEQ1"}, {"sequence": "SEQ2"}],  # batch 1
            [{"sequence": "SEQ3"}],                        # batch 2
        ]
        biolm(entity="esmfold", action="predict", items=batches)

------------------------
How Auto-Batching Works
------------------------

- By default, the client will automatically batch your input according to the model's maximum batch size (queried from the API schema).
- You do **not** need to manually split your input into batches; just provide a list of items.
- The client will:
    1. Query the schema for the model/action to determine `maxItems` (batch size).
    2. Split your input into batches of up to `maxItems`.
    3. Send each batch as a separate API request.
    4. Collect and return results in the same order as your input.

**Example:**

.. code-block:: python

    # If the model's max batch size is 8, this will be split into 2 requests:
    items = ["SEQ" + str(i) for i in range(12)]
    result = biolm(entity="esm2-8m", action="encode", type="sequence", items=items)
    # result is a list of 12 results, in order

------------------------
Advanced: Manual Batching with List of Lists
------------------------

- If you provide a list of lists of dicts, **each inner list is treated as a batch**.
- This disables auto-batching: you control the batch size and composition.
- Useful for:
    - Forcing certain items to be batched together (e.g., for error isolation).
    - Working around API limits or bugs.
    - Testing error handling with mixed valid/invalid batches.

**Example:**

.. code-block:: python

    # Two batches: first has 2 items, second has 1
    items = [
        [{"sequence": "SEQ1"}, {"sequence": "BADSEQ"}],  # batch 1
        [{"sequence": "SEQ3"}],                          # batch 2
    ]
    result = biolm(entity="esmfold", action="predict", items=items, stop_on_error=False)
    # result is a flat list: [result1, result2, result3]

------------------------
Input Validation and Type Inference
------------------------

- If you provide a list of dicts, the client infers the input type from the dict keys.
- If you provide a list of values (not dicts), you **must** specify `type` (e.g., `type="sequence"`).
- If you provide a list of lists, each inner list must be a list of dicts (not strings).

------------------------
Batch Size and Schema
------------------------

- The client queries the API schema for the model/action to determine the maximum batch size (`maxItems`).
- You can inspect this yourself:

.. code-block:: python

    from biolmai.client import BioLMApi
    model = BioLMApi("esm2-8m")
    schema = model.schema("esm2-8m", "encode")
    max_batch = model.extract_max_items(schema)
    print("Max batch size:", max_batch)

------------------------
Batching and Error Handling
------------------------

- If a batch contains invalid items, the entire batch may fail (depending on the API).
- Use `stop_on_error=True` to halt processing after the first error batch.
- Use `stop_on_error=False` to continue processing all batches, with errors included in the results.
- Use `retry_error_batches=True` (BioLMApi/BioLMApiClient only) to retry failed batches as single items.

------------------------
Summary Table
------------------------

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
Examples
------------------------

**Batching with list of dicts:**

.. code-block:: python

    items = [{"sequence": "SEQ1"}, {"sequence": "SEQ2"}]
    result = biolm(entity="esm2-8m", action="encode", items=items)

**Batching with list of values:**

.. code-block:: python

    items = ["SEQ1", "SEQ2"]
    result = biolm(entity="esm2-8m", action="encode", type="sequence", items=items)

**Manual batching with list of lists:**

.. code-block:: python

    batches = [
        [{"sequence": "SEQ1"}, {"sequence": "BADSEQ"}],  # batch 1
        [{"sequence": "SEQ3"}],                          # batch 2
    ]
    result = biolm(entity="esmfold", action="predict", items=batches, stop_on_error=False)

------------------------
Best Practices
------------------------

- For most use cases, provide a list of values or dicts and let the client auto-batch.
- Use manual batching (list of lists) only for advanced workflows.
- Always specify `type` if your items are not dicts.
- For large jobs, consider `output='disk'` to avoid memory issues.

------------------------
See Also
------------------------

- :doc:`error_handling`
- :doc:`disk_output`
- :doc:`faq`
