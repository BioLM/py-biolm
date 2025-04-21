# File: docs/disk_output.rst
========================
Disk Output
========================

Write results to disk as JSONL:

.. code-block:: python

    BioLM(entity="esmfold", action="predict", items=["SEQ1", "SEQ2"], output='disk', file_path="results.jsonl")

    # Each result is written as a line in the file.
